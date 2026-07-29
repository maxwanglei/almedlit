"""Compute-profile support retained for the inference runtime."""

from sqlalchemy.orm import Session

from al_medlit.core.config import settings
from al_medlit.core.exceptions import NotFoundError
from al_medlit.training.compute.base import ComputeBackend
from al_medlit.training.compute.registry import (
    compute_backends,
    register_builtin_compute_backends,
)
from al_medlit.training.models import ComputeProfile


def get_compute_profile(db: Session, profile_id: int) -> ComputeProfile:
    profile = db.get(ComputeProfile, profile_id)
    if profile is None:
        raise NotFoundError("Compute profile not found")
    return profile


def build_compute_backend(
    profile: ComputeProfile,
    *,
    durable_local: bool | None = None,
) -> ComputeBackend:
    """Build a backend without persisting infrastructure credentials."""

    register_builtin_compute_backends()
    config = dict(profile.config)
    if profile.backend == "local" and durable_local is not None:
        config["synchronous"] = not durable_local
        if durable_local:
            config["runtime_root"] = settings.local_attempt_root
    return compute_backends.get(profile.backend).build(config)
