"""Probe an optional worker runtime and emit an image-bound readiness report."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import secrets
import shutil
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from al_medlit.core.storage import get_object_storage
from al_medlit.training.runtime_profiles import (
    RUNTIME_PROFILES,
    RuntimeReadinessReport,
    runtime_profile_report_sha256,
)


def normalize_worker_image_digest(value: str | None) -> str | None:
    normalized = (value or "").strip()
    if normalized.startswith("sha256:"):
        normalized = normalized.removeprefix("sha256:")
    return normalized or None


def _system_memory_bytes() -> int | None:
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return page_size * page_count


def _probe_device(required_device: str) -> tuple[str, bool, int | None]:
    if required_device == "cpu":
        return "cpu", True, _system_memory_bytes()
    try:
        import torch
    except ImportError:
        return ("cuda" if required_device == "cuda" else "accelerator"), False, None

    if bool(torch.cuda.is_available()):
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        return "cuda", True, int(properties.total_memory)
    if required_device == "accelerator":
        npu = getattr(torch, "npu", None)
        if npu is not None and bool(npu.is_available()):
            properties = npu.get_device_properties(npu.current_device())
            return "ascend", True, int(getattr(properties, "total_memory", 0)) or None
    return ("cuda" if required_device == "cuda" else "accelerator"), False, None


def _probe_storage() -> bool:
    storage = get_object_storage()
    key = f"runtime-preflight/{secrets.token_hex(16)}"
    payload = secrets.token_bytes(32)
    try:
        storage.put_bytes(key, payload, content_type="application/octet-stream")
        return storage.get_bytes(key) == payload
    finally:
        storage.delete(key)


def collect_runtime_readiness(
    runtime_profile: str,
    *,
    scratch_root: str | Path,
    worker_image_digest: str | None,
    storage_probe: Callable[[], bool] = _probe_storage,
) -> RuntimeReadinessReport:
    descriptor = RUNTIME_PROFILES[runtime_profile]
    dependency_versions: dict[str, str] = {}
    missing_dependencies: list[str] = []
    for import_name in descriptor.required_imports:
        if importlib.util.find_spec(import_name) is None:
            missing_dependencies.append(import_name)
            continue
        distribution = descriptor.distributions[import_name]
        try:
            dependency_versions[import_name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing_dependencies.append(import_name)

    scratch_path = Path(scratch_root)
    try:
        scratch_path.mkdir(parents=True, exist_ok=True)
        scratch_available_bytes = shutil.disk_usage(scratch_path).free
    except OSError:
        scratch_available_bytes = 0

    device, device_available, device_memory_bytes = _probe_device(
        descriptor.required_device
    )
    try:
        storage_access_verified = bool(storage_probe())
    except Exception:
        storage_access_verified = False

    ready = bool(
        worker_image_digest
        and not missing_dependencies
        and device_available
        and device_memory_bytes
        and scratch_available_bytes >= descriptor.minimum_scratch_bytes
        and storage_access_verified
    )
    return RuntimeReadinessReport(
        runtime_profile=runtime_profile,
        generated_at=datetime.now(UTC),
        python_version=platform.python_version(),
        worker_image_digest=worker_image_digest,
        dependency_versions=dependency_versions,
        missing_dependencies=missing_dependencies,
        device=device,
        device_available=device_available,
        device_memory_bytes=device_memory_bytes,
        scratch_path=str(scratch_path.resolve()),
        scratch_available_bytes=scratch_available_bytes,
        storage_access_verified=storage_access_verified,
        ready=ready,
    )


def compute_profile_fragment(report: RuntimeReadinessReport) -> dict:
    capabilities: list[str] = []
    if report.device == "cuda" and report.device_available:
        capabilities.append("cuda")
    elif report.device == "ascend" and report.device_available:
        capabilities.append("ascend")
    if report.runtime_profile == "qlora-cuda" and report.ready:
        capabilities.append("qlora_4bit")
    return {
        "runtime_profile": report.runtime_profile,
        "verified_dependencies": sorted(report.dependency_versions),
        "verified_capabilities": capabilities,
        "worker_image_digest": report.worker_image_digest,
        "capability_report_sha256": runtime_profile_report_sha256(report),
        "readiness_report": report.model_dump(mode="json"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(RUNTIME_PROFILES))
    parser.add_argument("--scratch-root", default="/var/lib/al-medlit/attempts")
    parser.add_argument(
        "--image-digest",
        default=os.environ.get("AL_MEDLIT_WORKER_IMAGE_DIGEST"),
        help="Lowercase SHA-256 image digest, with or without a sha256: prefix",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    image_digest = normalize_worker_image_digest(args.image_digest)
    report = collect_runtime_readiness(
        args.profile,
        scratch_root=args.scratch_root,
        worker_image_digest=image_digest,
    )
    payload = compute_profile_fragment(report)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, args.output)
    else:
        sys.stdout.write(serialized)
    return 0 if report.ready or not args.require_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
