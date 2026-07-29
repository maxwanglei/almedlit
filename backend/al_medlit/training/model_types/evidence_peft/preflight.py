"""Compute preflight for adapter training, with fail-closed QLoRA checks."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.training.model_types.evidence_neural.device import preflight_neural_device
from al_medlit.training.model_types.evidence_peft.config import (
    EvidencePeftConfig,
    EvidenceQLoRAConfig,
)


class PeftPreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_kind: str
    requested_device: str
    available: bool
    quantization_ready: bool
    reason: str | None = None
    missing_dependencies: tuple[str, ...] = ()
    details: dict = Field(default_factory=dict)


ModuleLoader = Callable[[str], object]


def preflight_peft(
    config: EvidencePeftConfig,
    *,
    module_loader: ModuleLoader = importlib.import_module,
) -> PeftPreflightResult:
    dependencies = ["torch", "transformers", "peft", "safetensors"]
    if isinstance(config, EvidenceQLoRAConfig):
        dependencies.append("bitsandbytes")
    loaded: dict[str, object] = {}
    missing: list[str] = []
    for dependency in dependencies:
        try:
            loaded[dependency] = module_loader(dependency)
        except (ImportError, ModuleNotFoundError):
            missing.append(dependency)
    if missing:
        return PeftPreflightResult(
            model_kind=config.model_kind,
            requested_device=config.device,
            available=False,
            quantization_ready=False,
            reason="Required PEFT dependencies are missing",
            missing_dependencies=tuple(missing),
        )

    device = preflight_neural_device(config.device, module_loader=module_loader)
    if not device.available:
        return PeftPreflightResult(
            model_kind=config.model_kind,
            requested_device=config.device,
            available=False,
            quantization_ready=False,
            reason=device.reason,
            missing_dependencies=device.missing_dependencies,
            details=device.details,
        )

    if isinstance(config, EvidenceQLoRAConfig):
        torch = loaded["torch"]
        transformers = loaded["transformers"]
        bitsandbytes = loaded["bitsandbytes"]
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        linear_4bit = getattr(getattr(bitsandbytes, "nn", None), "Linear4bit", None)
        quantization_config = getattr(transformers, "BitsAndBytesConfig", None)
        bf16_check = getattr(torch.cuda, "is_bf16_supported", None)
        bf16_ready = bool(bf16_check is not None and bf16_check())
        if (
            not cuda_runtime
            or linear_4bit is None
            or quantization_config is None
            or (config.compute_dtype == "bfloat16" and not bf16_ready)
        ):
            return PeftPreflightResult(
                model_kind=config.model_kind,
                requested_device=config.device,
                available=False,
                quantization_ready=False,
                reason=(
                    "QLoRA requires a CUDA PyTorch build and working bitsandbytes "
                    "4-bit/Transformers integration"
                ),
                details={
                    **device.details,
                    "cuda_runtime": str(cuda_runtime or ""),
                    "bfloat16_supported": bf16_ready,
                    "bitsandbytes_version": str(getattr(bitsandbytes, "__version__", "unknown")),
                },
            )
        return PeftPreflightResult(
            model_kind=config.model_kind,
            requested_device=config.device,
            available=True,
            quantization_ready=True,
            details={
                **device.details,
                "cuda_runtime": str(cuda_runtime),
                "bfloat16_supported": bf16_ready,
                "bitsandbytes_version": str(getattr(bitsandbytes, "__version__", "unknown")),
            },
        )

    return PeftPreflightResult(
        model_kind=config.model_kind,
        requested_device=config.device,
        available=True,
        quantization_ready=False,
        details=device.details,
    )


def require_peft_preflight(config: EvidencePeftConfig) -> PeftPreflightResult:
    result = preflight_peft(config)
    if not result.available:
        dependencies = ", ".join(result.missing_dependencies)
        suffix = f" (missing: {dependencies})" if dependencies else ""
        raise RuntimeError(f"{result.reason}{suffix}")
    if isinstance(config, EvidenceQLoRAConfig) and not result.quantization_ready:
        raise RuntimeError("QLoRA quantization preflight did not pass")
    return result


def build_qlora_quantization_config(config: EvidenceQLoRAConfig):
    """Create the exact Transformers config after quantization preflight."""

    require_peft_preflight(config)
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if config.compute_dtype == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quantization_type,
        bnb_4bit_use_double_quant=config.double_quantization,
        bnb_4bit_compute_dtype=compute_dtype,
    )
