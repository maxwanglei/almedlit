"""Quantization readiness checks for the inference runtime."""

from __future__ import annotations

import importlib.util

from al_medlit.training.models import ComputeProfile


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


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
