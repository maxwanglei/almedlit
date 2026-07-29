"""Domain operations for the canonical learning workflow."""

import hashlib
import re

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.core.storage import ObjectStorage
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _project,
    _scoped,
    _validate_task_input,
    _validate_task_output,
)
from .labels import (
    _compose_label_sets,
    create_imported_label_set_from_field,
    create_split_map,
    create_training_dataset_version,
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




def _resource_name(name: str, suffix: str) -> str:
    return f"{name[: 255 - len(suffix)]}{suffix}"


def _deterministic_group_split(
    items: list[models.DatasetItem],
    *,
    seed: int,
    train_percent: float,
    validation_percent: float,
) -> tuple[dict[str, str], dict[str, int], int]:
    groups = sorted(
        {item.group_key or item.stable_key for item in items},
        key=lambda group: (
            hashlib.sha256(f"{seed}\0{group}".encode()).digest(),
            group,
        ),
    )
    if len(groups) < 3:
        raise ValidationError(
            "At least three independent groups are required for train, "
            "validation, and protected test splits"
        )
    requested_train = round(len(groups) * train_percent / 100)
    requested_validation = round(len(groups) * validation_percent / 100)
    train_count = min(max(requested_train, 1), len(groups) - 2)
    validation_count = min(
        max(requested_validation, 1),
        len(groups) - train_count - 1,
    )
    split_by_group = {
        group: (
            "train"
            if index < train_count
            else "validation"
            if index < train_count + validation_count
            else "test"
        )
        for index, group in enumerate(groups)
    }
    assignments = {
        item.stable_key: split_by_group[item.group_key or item.stable_key] for item in items
    }
    split_counts = {
        split: sum(1 for assigned in assignments.values() if assigned == split)
        for split in ("train", "validation", "test")
    }
    return assignments, split_counts, len(groups)


def compose_training_dataset_version(
    db: Session,
    data: schemas.TrainingDatasetComposeCreate,
    actor: User,
    *,
    storage: ObjectStorage,
) -> dict:
    dataset_version = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    task_version = _scoped(
        db,
        models.TaskVersion,
        data.task_version_id,
        data.project_id,
        "Task version",
    )
    items = (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == dataset_version.id)
        .order_by(models.DatasetItem.id)
        .all()
    )
    if not items:
        raise ValidationError("Training composition requires a materialized dataset")
    missing_inputs = [item.stable_key for item in items if data.input_field not in item.payload]
    if missing_inputs:
        raise ValidationError(
            f"Input field {data.input_field!r} is missing from dataset items: "
            + ", ".join(missing_inputs[:5])
        )
    for item in items:
        _validate_task_input(
            task_version,
            item.payload,
            label=f"Dataset item {item.stable_key!r}",
        )
    if data.label_field is not None:
        missing_labels = [item.stable_key for item in items if data.label_field not in item.payload]
        if missing_labels:
            raise ValidationError(
                f"Label field {data.label_field!r} is missing from dataset items: "
                + ", ".join(missing_labels[:5])
            )
        for item in items:
            _validate_task_output(
                task_version,
                item.payload[data.label_field],
                label=f"Label for dataset item {item.stable_key!r}",
            )
        label_set = None
    else:
        assert data.label_set_version_id is not None
        label_set = _scoped(
            db,
            models.LabelSetVersion,
            data.label_set_version_id,
            data.project_id,
            "Label set version",
        )
        if (
            label_set.dataset_version_id != dataset_version.id
            or label_set.task_version_id != task_version.id
        ):
            raise ValidationError(
                "The selected label set must use the selected dataset and task versions"
            )
        if not label_set.labels:
            raise ValidationError("The selected label set is empty")

    assignments, split_counts, group_count = _deterministic_group_split(
        items,
        seed=data.seed,
        train_percent=data.train_percent,
        validation_percent=data.validation_percent,
    )
    split_name = _resource_name(data.name, " split")
    if (
        db.query(models.SplitMap.id)
        .filter(
            models.SplitMap.dataset_version_id == dataset_version.id,
            models.SplitMap.name == split_name,
        )
        .first()
        is not None
    ):
        raise ConflictError("A split map with this training dataset name already exists")

    try:
        if label_set is None:
            assert data.label_field is not None
            label_set = create_imported_label_set_from_field(
                db,
                schemas.ImportedLabelSetFromFieldCreate(
                    project_id=data.project_id,
                    dataset_version_id=dataset_version.id,
                    task_version_id=task_version.id,
                    name=_resource_name(data.name, " imported labels"),
                    label_field=data.label_field,
                    composition_policy="replace",
                ),
                actor,
                storage=storage,
                commit=False,
            )
        split_map = create_split_map(
            db,
            schemas.SplitMapCreate(
                project_id=data.project_id,
                dataset_version_id=dataset_version.id,
                name=split_name,
                strategy="deterministic_group_hash_percentages_v1",
                seed=data.seed,
                group_key_field=(
                    "group_key" if any(item.group_key is not None for item in items) else None
                ),
                assignments=assignments,
                protected_splits=["test"],
            ),
            actor,
            commit=False,
        )
        training_dataset = create_training_dataset_version(
            db,
            schemas.TrainingDatasetVersionCreate(
                project_id=data.project_id,
                name=data.name,
                dataset_version_id=dataset_version.id,
                task_version_id=task_version.id,
                label_set_version_ids=[label_set.id],
                split_map_id=split_map.id,
                composition=[
                    {
                        "label_set_version_id": label_set.id,
                        "policy": "replace",
                    }
                ],
                preprocessing={
                    "input_field": data.input_field,
                    "target_field": data.label_field or "label",
                    **(
                        {}
                        if data.label_field is not None
                        else {"target_label_set_version_id": label_set.id}
                    ),
                },
            ),
            actor,
            commit=False,
        )
        _commit(db, "Could not compose the training dataset")
    except Exception:
        db.rollback()
        raise
    db.refresh(label_set)
    db.refresh(split_map)
    db.refresh(training_dataset)
    return {
        "training_dataset_version": training_dataset,
        "label_set_version_id": label_set.id,
        "split_map_id": split_map.id,
        "split_counts": split_counts,
        "group_count": group_count,
    }


def compose_training_dataset_labels(
    db: Session,
    project_id: int,
    training_dataset_version_id: int,
) -> schemas.ComposedTrainingLabelsRead:
    training_dataset = _scoped(
        db,
        models.TrainingDatasetVersion,
        training_dataset_version_id,
        project_id,
        "Training dataset version",
    )
    label_sets = [
        _scoped(
            db,
            models.LabelSetVersion,
            label_set_id,
            project_id,
            "Label set version",
        )
        for label_set_id in training_dataset.label_set_version_ids
    ]
    composition, labels = _compose_label_sets(label_sets, training_dataset.composition)
    return schemas.ComposedTrainingLabelsRead(
        training_dataset_version_id=training_dataset.id,
        labels=labels,
        label_count=len(labels),
        content_hash=_canonical_hash(labels),
        composition=composition,
    )


def list_training_dataset_versions(
    db: Session, project_id: int
) -> list[models.TrainingDatasetVersion]:
    _project(db, project_id)
    return (
        db.query(models.TrainingDatasetVersion)
        .filter(models.TrainingDatasetVersion.project_id == project_id)
        .order_by(
            models.TrainingDatasetVersion.created_at.desc(),
            models.TrainingDatasetVersion.id.desc(),
        )
        .all()
    )


def list_split_maps(
    db: Session, project_id: int, dataset_version_id: int | None = None
) -> list[models.SplitMap]:
    _project(db, project_id)
    query = db.query(models.SplitMap).filter(models.SplitMap.project_id == project_id)
    if dataset_version_id is not None:
        _scoped(
            db,
            models.DatasetVersion,
            dataset_version_id,
            project_id,
            "Dataset version",
        )
        query = query.filter(models.SplitMap.dataset_version_id == dataset_version_id)
    return query.order_by(models.SplitMap.name, models.SplitMap.id).all()
