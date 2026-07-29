"""Domain operations for the canonical learning workflow."""

import re
from collections.abc import Iterable

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ValidationError,
)
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _optional_package,
    _project,
    _scoped,
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




def create_registered_model(
    db: Session, data: schemas.RegisteredModelCreate, actor: User
) -> models.RegisteredModel:
    _project(db, data.project_id)
    registered = models.RegisteredModel(**data.model_dump(), created_by_user_id=actor.id)
    db.add(registered)
    _commit(db, "A model with this name already exists in the project")
    db.refresh(registered)
    return registered


def list_registered_models(db: Session, project_id: int) -> list[models.RegisteredModel]:
    _project(db, project_id)
    return (
        db.query(models.RegisteredModel)
        .filter(models.RegisteredModel.project_id == project_id)
        .order_by(models.RegisteredModel.name, models.RegisteredModel.id)
        .all()
    )


def create_model_version(
    db: Session, data: schemas.ModelVersionCreate, actor: User
) -> models.ModelVersion:
    registered = _scoped(
        db,
        models.RegisteredModel,
        data.registered_model_id,
        data.project_id,
        "Registered model",
    )
    db.query(models.RegisteredModel).filter(
        models.RegisteredModel.id == registered.id
    ).with_for_update().one()
    _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    if data.training_dataset_version_id is not None:
        training_dataset = _scoped(
            db,
            models.TrainingDatasetVersion,
            data.training_dataset_version_id,
            data.project_id,
            "Training dataset version",
        )
        if training_dataset.task_version_id != data.task_version_id:
            raise ValidationError("Model and training dataset must use the same task version")
    if data.parent_version_id is not None:
        parent = _scoped(
            db,
            models.ModelVersion,
            data.parent_version_id,
            data.project_id,
            "Parent model version",
        )
        if parent.registered_model_id != registered.id:
            raise ValidationError("Parent version must belong to the same registered model")
    _optional_package(db, data.project_id, data.checkpoint_package_id)
    content = data.model_dump(exclude={"project_id", "registered_model_id"})
    version = models.ModelVersion(
        **data.model_dump(),
        version_number=_next_version(
            db,
            models.ModelVersion,
            models.ModelVersion.registered_model_id,
            registered.id,
        ),
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
    )
    db.add(version)
    _commit(db, "This model version already exists")
    db.refresh(version)
    return version


def _model_version_lineage_payloads(
    db: Session,
    versions: Iterable[models.ModelVersion],
) -> tuple[dict[int, dict], dict[int, models.TrainingDatasetVersion]]:
    version_list = list(versions)
    if not version_list:
        return {}, {}

    project_ids = {version.project_id for version in version_list}
    versions_by_id = {version.id: version for version in version_list}
    version_ids = {version.id for version in version_list}
    training_dataset_ids = {
        version.training_dataset_version_id
        for version in version_list
        if version.training_dataset_version_id is not None
    }
    training_datasets = (
        {
            record.id: record
            for record in db.query(models.TrainingDatasetVersion)
            .filter(
                models.TrainingDatasetVersion.id.in_(training_dataset_ids),
                models.TrainingDatasetVersion.project_id.in_(project_ids),
            )
            .all()
        }
        if training_dataset_ids
        else {}
    )
    source_dataset_version_ids = {
        record.dataset_version_id for record in training_datasets.values()
    }
    source_dataset_versions = (
        {
            record.id: record
            for record in db.query(models.DatasetVersion)
            .filter(
                models.DatasetVersion.id.in_(source_dataset_version_ids),
                models.DatasetVersion.project_id.in_(project_ids),
            )
            .all()
        }
        if source_dataset_version_ids
        else {}
    )

    latest_run_by_version: dict[int, models.TrainingRun] = {}
    for run in (
        db.query(models.TrainingRun)
        .filter(
            models.TrainingRun.output_model_version_id.in_(version_ids),
            models.TrainingRun.project_id.in_(project_ids),
        )
        .order_by(
            models.TrainingRun.output_model_version_id,
            models.TrainingRun.id.desc(),
        )
        .all()
    ):
        version = versions_by_id.get(run.output_model_version_id)
        if version is not None and version.project_id == run.project_id:
            latest_run_by_version.setdefault(run.output_model_version_id, run)

    environment_ids = {run.environment_id for run in latest_run_by_version.values()}
    environments = (
        {
            record.id: record
            for record in db.query(models.ExecutionEnvironment)
            .filter(
                models.ExecutionEnvironment.id.in_(environment_ids),
                models.ExecutionEnvironment.project_id.in_(project_ids),
            )
            .all()
        }
        if environment_ids
        else {}
    )
    storage_policy_ids = {run.storage_policy_id for run in latest_run_by_version.values()}
    storage_policies = (
        {
            record.id: record
            for record in db.query(models.StoragePolicy)
            .filter(
                models.StoragePolicy.id.in_(storage_policy_ids),
                models.StoragePolicy.project_id.in_(project_ids),
            )
            .all()
        }
        if storage_policy_ids
        else {}
    )
    creator_ids = {
        version.created_by_user_id
        for version in version_list
        if version.created_by_user_id is not None
    }
    creators = (
        {record.id: record for record in db.query(User).filter(User.id.in_(creator_ids)).all()}
        if creator_ids
        else {}
    )

    payloads: dict[int, dict] = {}
    for version in version_list:
        training_dataset = (
            training_datasets.get(version.training_dataset_version_id)
            if version.training_dataset_version_id is not None
            else None
        )
        if training_dataset is not None and training_dataset.project_id != version.project_id:
            training_dataset = None
        source_dataset_version = (
            source_dataset_versions.get(training_dataset.dataset_version_id)
            if training_dataset
            else None
        )
        if (
            source_dataset_version is not None
            and source_dataset_version.project_id != version.project_id
        ):
            source_dataset_version = None
        run = latest_run_by_version.get(version.id)
        environment = environments.get(run.environment_id) if run else None
        if environment is not None and environment.project_id != version.project_id:
            environment = None
        storage_policy = storage_policies.get(run.storage_policy_id) if run else None
        if storage_policy is not None and storage_policy.project_id != version.project_id:
            storage_policy = None
        creator = creators.get(version.created_by_user_id)
        payload = schemas.ModelVersionRead.model_validate(version).model_dump()
        payload.update(
            {
                "training_run_id": run.id if run else None,
                "source_dataset_version_id": (
                    source_dataset_version.id if source_dataset_version else None
                ),
                "source_dataset_version_number": (
                    source_dataset_version.version_number if source_dataset_version else None
                ),
                "runtime_id": environment.id if environment else None,
                "runtime_name": environment.name if environment else None,
                "storage_policy_id": (storage_policy.id if storage_policy else None),
                "storage_policy_name": (storage_policy.name if storage_policy else None),
                "creator_username": creator.username if creator else None,
                "creator_display_name": (creator.display_name if creator else None),
            }
        )
        payloads[version.id] = payload
    return payloads, training_datasets


def list_model_versions(db: Session, project_id: int, registered_model_id: int) -> list[dict]:
    _scoped(
        db,
        models.RegisteredModel,
        registered_model_id,
        project_id,
        "Registered model",
    )
    versions = (
        db.query(models.ModelVersion)
        .filter(
            models.ModelVersion.registered_model_id == registered_model_id,
            models.ModelVersion.project_id == project_id,
        )
        .order_by(models.ModelVersion.version_number)
        .all()
    )
    payloads, _training_datasets = _model_version_lineage_payloads(db, versions)
    return [payloads[version.id] for version in versions]


def list_model_version_evaluations(
    db: Session,
    project_id: int,
    registered_model_id: int,
    model_version_id: int,
) -> list[models.ModelEvaluation]:
    _scoped(
        db,
        models.RegisteredModel,
        registered_model_id,
        project_id,
        "Registered model",
    )
    model_version = _scoped(
        db,
        models.ModelVersion,
        model_version_id,
        project_id,
        "Model version",
    )
    if model_version.registered_model_id != registered_model_id:
        raise ValidationError("Model version does not belong to the selected registered model")
    return (
        db.query(models.ModelEvaluation)
        .filter(models.ModelEvaluation.model_version_id == model_version_id)
        .order_by(models.ModelEvaluation.split_name, models.ModelEvaluation.id)
        .all()
    )
