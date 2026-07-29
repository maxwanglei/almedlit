"""Domain operations for the canonical learning workflow."""

import re

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _project,
    _scoped,
    _validate_json_schema_contract,
)

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




def create_task_definition(
    db: Session, data: schemas.TaskDefinitionCreate, actor: User
) -> models.TaskDefinition:
    _project(db, data.project_id)
    task = models.TaskDefinition(
        **data.model_dump(),
        created_by_user_id=actor.id,
    )
    db.add(task)
    _commit(db, "A task with this key already exists in the project")
    db.refresh(task)
    return task


def list_task_definitions(db: Session, project_id: int) -> list[models.TaskDefinition]:
    _project(db, project_id)
    return (
        db.query(models.TaskDefinition)
        .filter(models.TaskDefinition.project_id == project_id)
        .order_by(models.TaskDefinition.name, models.TaskDefinition.id)
        .all()
    )


def create_task_version(
    db: Session, data: schemas.TaskVersionCreate, actor: User
) -> models.TaskVersion:
    task = _scoped(
        db,
        models.TaskDefinition,
        data.task_definition_id,
        data.project_id,
        "Task definition",
    )
    _validate_json_schema_contract(data.input_schema, label="Task input schema")
    _validate_json_schema_contract(data.output_schema, label="Task output schema")
    (
        db.query(models.TaskDefinition)
        .filter(models.TaskDefinition.id == task.id)
        .with_for_update()
        .one()
    )
    content = data.model_dump(exclude={"project_id", "task_definition_id"})
    task_version = models.TaskVersion(
        project_id=data.project_id,
        task_definition_id=task.id,
        version_number=_next_version(
            db,
            models.TaskVersion,
            models.TaskVersion.task_definition_id,
            task.id,
        ),
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
        **content,
    )
    db.add(task_version)
    _commit(db, "This task version already exists")
    db.refresh(task_version)
    return task_version


def list_task_versions(
    db: Session, project_id: int, task_definition_id: int | None = None
) -> list[models.TaskVersion]:
    _project(db, project_id)
    query = db.query(models.TaskVersion).filter(models.TaskVersion.project_id == project_id)
    if task_definition_id is not None:
        _scoped(
            db,
            models.TaskDefinition,
            task_definition_id,
            project_id,
            "Task definition",
        )
        query = query.filter(models.TaskVersion.task_definition_id == task_definition_id)
    return query.order_by(
        models.TaskVersion.task_definition_id,
        models.TaskVersion.version_number,
    ).all()
