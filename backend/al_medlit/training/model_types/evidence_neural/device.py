"""Fail-closed device capability checks shared by neural plugins."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.training.model_types.evidence_neural.config import EvidenceDevice


class DevicePreflightResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_device: EvidenceDevice
    resolved_device: Literal["cpu", "mps", "cuda", "ascend"] | None
    torch_device: str | None
    available: bool
    reason: str | None = None
    missing_dependencies: tuple[str, ...] = ()
    details: dict = Field(default_factory=dict)


ModuleLoader = Callable[[str], object]


def preflight_neural_device(
    requested_device: EvidenceDevice,
    *,
    module_loader: ModuleLoader = importlib.import_module,
) -> DevicePreflightResult:
    try:
        torch = module_loader("torch")
    except (ImportError, ModuleNotFoundError):
        return DevicePreflightResult(
            requested_device=requested_device,
            resolved_device=None,
            torch_device=None,
            available=False,
            reason="PyTorch is not installed in this compute environment",
            missing_dependencies=("torch",),
        )

    def available(device: str) -> bool:
        if device == "cuda":
            return bool(torch.cuda.is_available())
        if device == "mps":
            backend = getattr(getattr(torch, "backends", None), "mps", None)
            return bool(backend is not None and backend.is_available())
        if device == "ascend":
            try:
                module_loader("torch_npu")
            except (ImportError, ModuleNotFoundError):
                return False
            npu = getattr(torch, "npu", None)
            return bool(npu is not None and npu.is_available())
        return True

    resolved = requested_device
    if requested_device == "auto":
        resolved = next(
            (candidate for candidate in ("cuda", "mps", "ascend") if available(candidate)),
            "cpu",
        )
    if not available(resolved):
        missing = ("torch_npu",) if resolved == "ascend" else ()
        return DevicePreflightResult(
            requested_device=requested_device,
            resolved_device=resolved,
            torch_device="npu" if resolved == "ascend" else resolved,
            available=False,
            reason=f"Requested {resolved} device is unavailable",
            missing_dependencies=missing,
        )
    torch_device = "npu" if resolved == "ascend" else resolved
    return DevicePreflightResult(
        requested_device=requested_device,
        resolved_device=resolved,
        torch_device=torch_device,
        available=True,
        details={
            "torch_version": str(getattr(torch, "__version__", "unknown")),
            "cuda_runtime": str(getattr(getattr(torch, "version", None), "cuda", "") or ""),
        },
    )


def require_neural_device(requested_device: EvidenceDevice) -> DevicePreflightResult:
    result = preflight_neural_device(requested_device)
    if not result.available:
        missing = ", ".join(result.missing_dependencies)
        suffix = f" (missing: {missing})" if missing else ""
        raise RuntimeError(f"{result.reason}{suffix}")
    return result
