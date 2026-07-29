"""Side-effect-free model configuration and compute capability preflight."""

from __future__ import annotations

import importlib.util
from typing import Any

from al_medlit.core.config import settings
from al_medlit.training.models import ComputeProfile
from al_medlit.training.registry import model_types
from al_medlit.training.runtime_profiles import runtime_profile_for_model_type


def _normalized(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    raise TypeError("Model configuration validators must return a Pydantic model or mapping")


def validate_configuration(model_type: str, config: dict) -> dict:
    try:
        plugin = model_types.get(model_type)
        normalized = _normalized(plugin.validate_config(config))
    except Exception as exc:
        return {
            "model_type": model_type,
            "valid": False,
            "normalized_config": None,
            "errors": [str(exc)],
        }
    return {
        "model_type": model_type,
        "valid": True,
        "normalized_config": normalized,
        "errors": [],
    }


def capability_preflight(*, model_type: str, config: dict, compute_profile: ComputeProfile) -> dict:
    descriptor = model_types.get_descriptor(model_type)
    runtime_profile = runtime_profile_for_model_type(model_type)
    validation = validate_configuration(model_type, config)
    checks: list[dict[str, str]] = []
    if validation["valid"]:
        checks.append(
            {"key": "configuration", "status": "pass", "message": "Configuration is valid"}
        )
    else:
        checks.append(
            {
                "key": "configuration",
                "status": "fail",
                "message": "; ".join(validation["errors"]),
            }
        )

    backend = compute_profile.backend
    if backend in descriptor.capabilities.supported_compute_backends:
        checks.append({"key": "backend", "status": "pass", "message": f"{backend} is supported"})
    else:
        checks.append(
            {
                "key": "backend",
                "status": "fail",
                "message": f"{backend} is not supported by this model type",
            }
        )

    configured_runtime = str(compute_profile.config.get("runtime_profile", "auto"))
    if configured_runtime == "auto":
        checks.append(
            {
                "key": "runtime_profile",
                "status": "warning",
                "message": (
                    "Legacy auto runtime uses process-local dependency checks; "
                    f"configure {runtime_profile.key} for an isolated worker"
                ),
            }
        )
    elif configured_runtime == runtime_profile.key:
        checks.append(
            {
                "key": "runtime_profile",
                "status": "pass",
                "message": f"Runtime profile {configured_runtime} matches this model type",
            }
        )
    else:
        checks.append(
            {
                "key": "runtime_profile",
                "status": "fail",
                "message": (
                    f"Model type {model_type} requires runtime profile "
                    f"{runtime_profile.key}, not {configured_runtime}"
                ),
            }
        )

    if (
        backend == "local"
        and configured_runtime != "auto"
        and settings.celery_task_always_eager
    ):
        checks.append(
            {
                "key": "execution_mode",
                "status": "fail",
                "message": (
                    "Named local runtimes require Redis-backed worker execution; "
                    "the API is currently configured for eager in-process tasks"
                ),
            }
        )
    else:
        checks.append(
            {
                "key": "execution_mode",
                "status": "pass",
                "message": "Execution mode is compatible with the compute profile",
            }
        )

    missing = [
        dependency
        for dependency in descriptor.capabilities.required_worker_dependencies
        if importlib.util.find_spec(dependency) is None
    ]
    qlora_worker_attested = model_type == "evidence_qlora" and _profile_attests(
        compute_profile,
        "qlora_4bit",
    )
    dependencies_attested = _profile_attests_dependencies(
        compute_profile,
        descriptor.capabilities.required_worker_dependencies,
    )
    if missing and dependencies_attested:
        checks.append(
            {
                "key": "dependencies",
                "status": "pass",
                "message": "Dependencies are verified by the image-bound runtime report",
            }
        )
    elif backend == "local" and missing and qlora_worker_attested:
        checks.append(
            {
                "key": "dependencies",
                "status": "pass",
                "message": "Dependencies are verified by the image-bound QLoRA capability report",
            }
        )
    elif backend == "local" and missing:
        checks.append(
            {
                "key": "dependencies",
                "status": "fail",
                "message": "Missing local worker dependencies: " + ", ".join(missing),
            }
        )
    elif backend != "local" and missing:
        checks.append(
            {
                "key": "dependencies",
                "status": "warning",
                "message": (
                    "Dependencies must be verified inside the remote worker image: "
                    + ", ".join(missing)
                ),
            }
        )
    else:
        checks.append({"key": "dependencies", "status": "pass", "message": "Dependencies found"})

    readiness_report = compute_profile.config.get("readiness_report")
    if configured_runtime != "auto":
        storage_ready = bool(
            isinstance(readiness_report, dict)
            and readiness_report.get("storage_access_verified")
        )
        checks.append(
            {
                "key": "storage",
                "status": "pass" if storage_ready else "fail",
                "message": (
                    "Object storage access was verified inside the worker"
                    if storage_ready
                    else "Worker readiness report has not verified object storage access"
                ),
            }
        )
        scratch_ready = bool(
            isinstance(readiness_report, dict)
            and int(readiness_report.get("scratch_available_bytes") or 0)
            >= runtime_profile.minimum_scratch_bytes
        )
        checks.append(
            {
                "key": "scratch",
                "status": "pass" if scratch_ready else "fail",
                "message": (
                    "Worker scratch capacity passed preflight"
                    if scratch_ready
                    else "Worker scratch capacity is below the runtime minimum"
                ),
            }
        )

    normalized = validation.get("normalized_config") or config
    device = str(normalized.get("device", "cpu"))
    if device != "auto" and device not in descriptor.capabilities.supported_devices:
        checks.append(
            {
                "key": "device",
                "status": "fail",
                "message": f"Device {device} is not supported by this model type",
            }
        )
    elif backend != "local":
        site_profile = str(compute_profile.config.get("site_profile", "generic"))
        if device == "ascend" and site_profile != "osc_ascend":
            checks.append(
                {
                    "key": "device",
                    "status": "fail",
                    "message": "Ascend device requires an Ascend compute profile",
                }
            )
        else:
            checks.append(
                {
                    "key": "device",
                    "status": "warning",
                    "message": "Device availability must be confirmed by remote preflight",
                }
            )
    elif backend == "local" and device != "auto" and _profile_attests(
        compute_profile,
        device,
    ):
        checks.append(
            {
                "key": "device",
                "status": "pass",
                "message": f"Device {device} is verified by the worker capability report",
            }
        )
    else:
        checks.append(_local_device_check(device))

    if model_type == "evidence_qlora":
        qlora_ok = device == "cuda" and quantization_profile_ready(compute_profile)
        checks.append(
            {
                "key": "quantization",
                "status": "pass" if qlora_ok else "fail",
                "message": (
                    "CUDA quantization preflight passed"
                    if qlora_ok
                    else "QLoRA requires a CUDA worker with a usable bitsandbytes installation"
                ),
            }
        )

    return {
        "model_type": model_type,
        "model_family": descriptor.family.value,
        "compute_backend": backend,
        "launchable": all(item["status"] != "fail" for item in checks),
        "normalized_config": validation.get("normalized_config"),
        "checks": checks,
    }


def quantization_profile_ready(compute_profile: ComputeProfile) -> bool:
    """Require either a live local check or an image-bound manager attestation."""

    capabilities = set(compute_profile.config.get("verified_capabilities") or ())
    attested = bool(
        {"cuda", "qlora_4bit"}.issubset(capabilities)
        and compute_profile.config.get("worker_image_digest")
        and compute_profile.config.get("capability_report_sha256")
    )
    if attested:
        return True
    return bool(
        compute_profile.backend == "local"
        and importlib.util.find_spec("bitsandbytes") is not None
        and _cuda_available()
    )


def _profile_attests(compute_profile: ComputeProfile, capability: str) -> bool:
    capabilities = set(compute_profile.config.get("verified_capabilities") or ())
    return bool(
        capability in capabilities
        and compute_profile.config.get("worker_image_digest")
        and compute_profile.config.get("capability_report_sha256")
    )


def _profile_attests_dependencies(
    compute_profile: ComputeProfile,
    dependencies: tuple[str, ...],
) -> bool:
    verified = set(compute_profile.config.get("verified_dependencies") or ())
    report = compute_profile.config.get("readiness_report")
    return bool(
        set(dependencies).issubset(verified)
        and isinstance(report, dict)
        and report.get("ready") is True
        and report.get("storage_access_verified") is True
        and compute_profile.config.get("worker_image_digest")
        and compute_profile.config.get("capability_report_sha256")
    )


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _local_device_check(device: str) -> dict[str, str]:
    if device in {"auto", "cpu"}:
        return {"key": "device", "status": "pass", "message": f"Device {device} is usable"}
    try:
        import torch
    except ImportError:
        return {
            "key": "device",
            "status": "fail",
            "message": "PyTorch is required for accelerator preflight",
        }
    if device == "cuda":
        available = bool(torch.cuda.is_available())
    elif device == "mps":
        available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    elif device == "ascend":
        available = importlib.util.find_spec("torch_npu") is not None
    else:
        available = False
    return {
        "key": "device",
        "status": "pass" if available else "fail",
        "message": f"Device {device} is {'available' if available else 'unavailable'}",
    }
