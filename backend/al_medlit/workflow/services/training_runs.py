"""Domain operations for the canonical learning workflow."""

import re
from typing import Any

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _project,
    _reject_sensitive_training_config,
    _scoped,
)
from .training_setup import (
    _validated_environment_readiness,
    _validated_storage_policy,
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




def _validate_task_trainer_contract(
    task: models.TaskVersion,
    recipe_definition: models.TrainingRecipe,
    recipe_version: models.TrainingRecipeVersion,
) -> Any:
    compatible_trainers = {
        identifier.strip()
        for identifier in task.trainer_compatibility
        if isinstance(identifier, str) and identifier.strip()
    }
    selected_identifiers = {
        recipe_definition.key,
        recipe_version.trainer_plugin_key,
    }
    if not compatible_trainers.intersection(selected_identifiers):
        raise ValidationError(
            "Training recipe and trainer plugin are not allowed by the task "
            "trainer compatibility contract"
        )

    from al_medlit.training.recipe_registry import training_recipes

    try:
        trusted = training_recipes.get(recipe_definition.key)
    except NotFoundError as exc:
        raise ValidationError(
            "Custom training recipes require a registered trusted worker contract "
            "before they can be launched"
        ) from exc
    trusted_task_kinds = {task_kind.value for task_kind in trusted.supported_task_kinds}
    if task.task_kind not in trusted_task_kinds:
        raise ValidationError("Training recipe is not compatible with the selected task kind")
    if recipe_version.trainer_plugin_key != trusted.trainer_key:
        raise ValidationError("Training recipe version does not use the trusted trainer plugin")
    if recipe_version.environment_class != trusted.environment.runtime_class.value:
        raise ValidationError("Training recipe version does not use the trusted runtime class")
    if recipe_version.config_schema != trusted.config_schema:
        raise ValidationError(
            "Training recipe version does not match the trusted configuration schema"
        )
    return trusted


def create_training_run(
    db: Session, data: schemas.TrainingRunCreate, actor: User
) -> models.TrainingRun:
    if data.parent_model_version_id is not None:
        raise ValidationError(
            "Parent model version launches are not supported until parent "
            "checkpoint execution is available"
        )
    launch_hash = _canonical_hash(data.model_dump(exclude={"idempotency_key"}))
    existing = (
        db.query(models.TrainingRun)
        .filter(
            models.TrainingRun.project_id == data.project_id,
            models.TrainingRun.idempotency_key == data.idempotency_key,
        )
        .first()
    )
    if existing is not None:
        if existing.launch_hash != launch_hash:
            raise ConflictError("The training-run idempotency key was reused with new inputs")
        return existing

    project = _project(db, data.project_id)
    _scoped(
        db,
        models.RegisteredModel,
        data.registered_model_id,
        data.project_id,
        "Registered model",
    )
    task = _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    training_dataset = _scoped(
        db,
        models.TrainingDatasetVersion,
        data.training_dataset_version_id,
        data.project_id,
        "Training dataset version",
    )
    recipe = _scoped(
        db,
        models.TrainingRecipeVersion,
        data.recipe_version_id,
        data.project_id,
        "Training recipe version",
    )
    recipe_definition = _scoped(
        db,
        models.TrainingRecipe,
        recipe.training_recipe_id,
        data.project_id,
        "Training recipe",
    )
    environment = _scoped(
        db,
        models.ExecutionEnvironment,
        data.environment_id,
        data.project_id,
        "Execution environment",
    )
    storage_policy = _scoped(
        db,
        models.StoragePolicy,
        data.storage_policy_id,
        data.project_id,
        "Storage policy",
    )
    storage_encryption = _validated_storage_policy(
        storage_policy,
        project=project,
    )
    if training_dataset.task_version_id != task.id:
        raise ValidationError("Training dataset and launch task versions must match")
    if task.task_kind not in recipe.compatible_task_kinds:
        raise ValidationError("Training recipe is not compatible with the selected task kind")
    trusted_recipe = _validate_task_trainer_contract(
        task,
        recipe_definition,
        recipe,
    )
    _reject_sensitive_training_config(data.config)
    from al_medlit.training.recipe_registry import training_recipes

    config_validation = training_recipes.validate(
        trusted_recipe.key,
        {
            **recipe.default_config,
            **data.config,
        },
    )
    if not config_validation.valid:
        raise ValidationError(
            "Training recipe configuration is invalid: " + "; ".join(config_validation.errors)
        )
    if recipe.environment_class != environment.environment_class:
        raise ValidationError("Training recipe and execution environment classes must match")
    _validated_environment_readiness(environment)
    training_run = models.TrainingRun(
        **data.model_dump(exclude={"artifact_reservation_bytes"}),
        launch_hash=launch_hash,
        runtime_snapshot={
            "environment_class": environment.environment_class,
            "image_digest": environment.image_digest,
            "package_manifest": environment.package_manifest,
            "hardware_constraints": environment.hardware_constraints,
            "verification_report": environment.verification_report,
            "verified_at": (
                environment.verified_at.isoformat() if environment.verified_at else None
            ),
        },
        storage_snapshot={
            "backend": storage_policy.backend,
            "artifact_prefix": storage_policy.artifact_prefix,
            "retention_class": storage_policy.retention_class,
            "encryption": storage_encryption,
            "cache_policy": storage_policy.cache_policy,
        },
        created_by_user_id=actor.id,
    )
    db.add(training_run)
    db.flush()
    reservation = artifact_quota.reserve_artifact_bytes(
        db,
        project_id=data.project_id,
        owner_type="training_run",
        owner_id=training_run.id,
        idempotency_key=f"training:{data.project_id}:{data.idempotency_key}",
        requested_bytes=data.artifact_reservation_bytes,
        actor_user_id=actor.id,
    )
    training_run.artifact_reservation_id = reservation.id
    training_run.storage_snapshot = {
        **training_run.storage_snapshot,
        "artifact_reservation_id": reservation.id,
        "artifact_reservation_bytes": reservation.reserved_bytes,
        "artifact_reservation_expires_at": reservation.expires_at.isoformat(),
    }
    _commit(db, "Could not launch the training run")
    db.refresh(training_run)
    return training_run


def list_training_runs(db: Session, project_id: int) -> list[models.TrainingRun]:
    _project(db, project_id)
    return (
        db.query(models.TrainingRun)
        .filter(models.TrainingRun.project_id == project_id)
        .order_by(models.TrainingRun.created_at.desc(), models.TrainingRun.id.desc())
        .all()
    )


def get_training_run(db: Session, project_id: int, training_run_id: int) -> models.TrainingRun:
    return _scoped(db, models.TrainingRun, training_run_id, project_id, "Training run")


def list_training_run_evaluations(
    db: Session,
    project_id: int,
    training_run_id: int,
) -> list[models.ModelEvaluation]:
    get_training_run(db, project_id, training_run_id)
    return (
        db.query(models.ModelEvaluation)
        .filter(models.ModelEvaluation.training_run_id == training_run_id)
        .order_by(models.ModelEvaluation.split_name, models.ModelEvaluation.id)
        .all()
    )


def get_model_evaluation(
    db: Session,
    project_id: int,
    evaluation_id: int,
) -> models.ModelEvaluation:
    return _scoped(
        db,
        models.ModelEvaluation,
        evaluation_id,
        project_id,
        "Model evaluation",
    )
