"""Safe, lazy helpers for staging and loading executable model artifacts."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import ArtifactPackage


def validate_executable_package(package: ArtifactPackage, *, package_format: str) -> None:
    """Require an available package governed by the safe in-process loader policy."""

    retention = package.retention
    if package.readiness != "ready":
        raise ConflictError("Model checkpoint package is not ready")
    if package.loader_policy != "safe":
        raise ConflictError("Model checkpoint package does not allow safe local loading")
    if package.package_format != package_format:
        raise ValidationError(
            f"Model checkpoint package must use the {package_format!r} format"
        )
    manifest_digest = hashlib.sha256(
        artifact_service.canonical_json_bytes(package.manifest)
    ).hexdigest()
    if manifest_digest != package.manifest_digest:
        raise ConflictError("Model checkpoint package manifest digest does not match")
    task_contract_hash = hashlib.sha256(
        artifact_service.canonical_json_bytes(package.task_contract)
    ).hexdigest()
    if task_contract_hash != package.task_contract_hash:
        raise ConflictError("Model checkpoint task contract digest does not match")
    if retention is None:
        raise ConflictError("Model checkpoint package is missing its retention record")
    if retention.archived_at is not None:
        raise ConflictError("Model checkpoint package has been archived")
    if retention.purged_at is not None:
        raise ConflictError("Model checkpoint package payload has been purged")


def stage_artifact_package(
    db: Session,
    storage: ObjectStorage,
    *,
    package: ArtifactPackage,
    destination: Path,
) -> Path:
    """Download one package into a read-only directory with integrity checks."""

    destination.mkdir(parents=True, exist_ok=False)
    for package_file in package.files:
        relative = artifact_service.validate_relative_path(package_file.relative_path)
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, chunks = artifact_service.iter_package_file(
            db,
            storage,
            project_id=package.project_id,
            package_id=package.id,
            relative_path=relative,
        )
        digest = hashlib.sha256()
        size = 0
        with target.open("xb") as sink:
            for chunk in chunks:
                sink.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if digest.hexdigest() != descriptor.checksum_sha256 or size != descriptor.size_bytes:
            raise ConflictError(f"Model checkpoint file '{relative}' failed integrity verification")
        target.chmod(0o444)
    for directory, directories, _files in os.walk(destination, topdown=False):
        for name in directories:
            (Path(directory) / name).chmod(0o555)
    destination.chmod(0o555)
    return destination


def load_safe_skops_model(
    model_path: Path,
    *,
    model_loader: Callable[[Path], Any] | None = None,
) -> Any:
    """Load a skops model without importing optional ML packages at module import time."""

    if model_path.is_symlink() or not model_path.is_file():
        raise ValidationError("TF-IDF model package requires a regular model.skops file")
    if model_loader is not None:
        return model_loader(model_path)
    try:
        from skops.io import get_untrusted_types, load
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValidationError(
            "The classical-cpu worker requires skops to load TF-IDF models"
        ) from exc
    untrusted = get_untrusted_types(file=model_path)
    if untrusted:
        raise ValidationError(
            "The generated sklearn package contains untrusted model types: "
            + ", ".join(sorted(untrusted))
        )
    return load(model_path, trusted=[])
