"""Domain operations for the canonical learning workflow."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from al_medlit.model_artifacts.models import ArtifactPackage
from al_medlit.project.models import Project
from al_medlit.workflow import models
from al_medlit.workspace.models import WorkspaceMember

MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATASET_UPLOAD_ITEMS = 100_000
MAX_JSONL_LINE_BYTES = 1024 * 1024
_PINNED_HUGGING_FACE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_AUTOMATIC_SELECTION_STRATEGIES = {
    "all",
    "random",
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_FEEDBACK_SELECTION_STRATEGIES = {
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_SENSITIVE_TRAINING_CONFIG_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}


def _normalized_sha256(value: str) -> str:
    return value.removeprefix("sha256:").lower()


@dataclass(frozen=True, slots=True)
class _FeedbackEventLineage:
    cycle: models.LearningCycle | None
    annotation_round: models.AnnotationRound | None
    round_item: models.RoundItem | None
    feedback_candidate: models.FeedbackCandidate | None
    dataset_item: models.DatasetItem | None


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValidationError("Versioned platform content must be canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_sensitive_training_config(value: Any, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if any(fragment in key for fragment in _SENSITIVE_TRAINING_CONFIG_FRAGMENTS):
                raise ValidationError(f"Training {path} cannot contain credentials or secrets")
            _reject_sensitive_training_config(
                nested,
                path=f"{path}.{raw_key}",
            )
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_sensitive_training_config(
                nested,
                path=f"{path}[{index}]",
            )


def _validate_json_schema_contract(schema: dict, *, label: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValidationError(f"{label} is not valid JSON Schema: {exc.message}") from exc


def _validate_schema_instance(
    schema: dict,
    value: Any,
    *,
    label: str,
    contract_label: str,
) -> None:
    validator = Draft202012Validator(schema)
    error = next(iter(validator.iter_errors(value)), None)
    if error is None:
        return
    path = ".".join(str(part) for part in error.absolute_path)
    location = f" at {path}" if path else ""
    raise ValidationError(
        f"{label} violates the {contract_label} schema{location}: {error.message}"
    )


def _validate_task_input(
    task: models.TaskVersion,
    value: Any,
    *,
    label: str,
) -> None:
    _validate_schema_instance(
        task.input_schema,
        value,
        label=label,
        contract_label="task input",
    )


def _validate_task_output(
    task: models.TaskVersion,
    value: Any,
    *,
    label: str,
) -> None:
    _validate_schema_instance(
        task.output_schema,
        value,
        label=label,
        contract_label="task output",
    )


def _commit(db: Session, message: str) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(message) from exc


def _persist(db: Session, message: str, *, commit: bool) -> None:
    if commit:
        _commit(db, message)
        return
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(message) from exc


def _project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _require_actor_role(
    db: Session,
    *,
    project_id: int,
    actor: User,
    minimum_role: str,
) -> None:
    if _actor_has_role(
        db,
        project_id=project_id,
        actor=actor,
        minimum_role=minimum_role,
    ):
        return
    raise ForbiddenError(f"{minimum_role.title()} role is required")


def _actor_has_role(
    db: Session,
    *,
    project_id: int,
    actor: User,
    minimum_role: str,
) -> bool:
    if actor.is_superuser:
        return True
    project = _project(db, project_id)
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == project.workspace_id,
            WorkspaceMember.user_id == actor.id,
        )
        .one_or_none()
    )
    role_rank = {"annotator": 0, "trainer": 1, "manager": 2, "admin": 3}
    return member is not None and (role_rank.get(member.role, -1) >= role_rank[minimum_role])


def _scoped[ModelT](
    db: Session,
    model_type: type[ModelT],
    record_id: int,
    project_id: int,
    label: str,
) -> ModelT:
    record = db.get(model_type, record_id)
    if record is None or getattr(record, "project_id", None) != project_id:
        raise NotFoundError(f"{label} not found")
    return record


def _optional_package(db: Session, project_id: int, package_id: int | None) -> None:
    if package_id is None:
        return
    _scoped(db, ArtifactPackage, package_id, project_id, "Artifact package")


def _next_version(db: Session, model_type, scope_column, scope_id: int) -> int:
    latest = db.query(func.max(model_type.version_number)).filter(scope_column == scope_id).scalar()
    return int(latest or 0) + 1


def _next_sequence(db: Session, model_type, project_id: int, *, cycle_id: int | None = None) -> int:
    query = db.query(func.max(model_type.sequence)).filter(model_type.project_id == project_id)
    if hasattr(model_type, "cycle_id"):
        if cycle_id is None:
            query = query.filter(model_type.cycle_id.is_(None))
        else:
            query = query.filter(model_type.cycle_id == cycle_id)
    return int(query.scalar() or 0) + 1
