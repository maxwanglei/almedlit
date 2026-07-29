"""Domain operations for the canonical learning workflow."""

import hashlib
import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.core.storage import ObjectStorage
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _optional_package,
    _persist,
    _scoped,
    _validate_task_input,
    _validate_task_output,
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




def create_label_set_version(
    db: Session,
    data: schemas.LabelSetVersionCreate,
    actor: User,
    *,
    storage: ObjectStorage | None = None,
    source_annotation_round_id: int | None = None,
    source_submission_ids: Iterable[int] = (),
    source_decision_ids: Iterable[int] = (),
    commit: bool = True,
) -> models.LabelSetVersion:
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
    submission_ids = sorted(set(source_submission_ids))
    decision_ids = sorted(set(source_decision_ids))
    if data.source_kind in {"human", "adjudicated"}:
        if source_annotation_round_id is None or not submission_ids or not decision_ids:
            raise ValidationError(
                "Human and adjudicated label sets must be derived from finalized round submissions"
            )
    elif source_annotation_round_id is not None or submission_ids or decision_ids:
        raise ValidationError(
            "Only human or adjudicated label sets may reference annotation decisions"
        )
    if data.source_kind == "derived" and data.composition_policy != "exclude":
        raise ValidationError(
            "Derived label sets may only exclude items; predictions require an "
            "explicit human or adjudication decision before becoming labels"
        )
    _optional_package(db, data.project_id, data.artifact_package_id)
    parent = None
    if data.parent_version_id is not None:
        parent = _scoped(
            db,
            models.LabelSetVersion,
            data.parent_version_id,
            data.project_id,
            "Parent label set version",
        )
        if (
            parent.dataset_version_id != data.dataset_version_id
            or parent.task_version_id != data.task_version_id
        ):
            raise ValidationError("Parent label set must use the same dataset and task versions")

    known_keys = {
        key
        for (key,) in db.query(models.DatasetItem.stable_key)
        .filter(models.DatasetItem.dataset_version_id == dataset_version.id)
        .all()
    }
    unknown_keys = sorted(set(data.labels) - known_keys)
    if unknown_keys and dataset_version.item_count:
        raise ValidationError(
            f"Labels reference unknown dataset item keys: {', '.join(unknown_keys[:5])}"
        )
    if data.composition_policy != "exclude":
        for stable_key, output in data.labels.items():
            _validate_task_output(
                task_version,
                output,
                label=f"Label for dataset item {stable_key!r}",
            )

    db.query(models.DatasetVersion).filter(
        models.DatasetVersion.id == dataset_version.id
    ).with_for_update().one()
    latest = (
        db.query(func.max(models.LabelSetVersion.version_number))
        .filter(
            models.LabelSetVersion.dataset_version_id == data.dataset_version_id,
            models.LabelSetVersion.task_version_id == data.task_version_id,
            models.LabelSetVersion.name == data.name,
        )
        .scalar()
    )
    content = data.model_dump(exclude={"project_id", "parent_version_id", "artifact_package_id"})
    content["source_annotation_round_id"] = source_annotation_round_id
    content["source_submission_ids"] = submission_ids
    content["source_decision_ids"] = decision_ids
    artifact_package_id = data.artifact_package_id
    if storage is not None and artifact_package_id is None:
        label_bytes = artifact_service.canonical_json_bytes(data.labels)
        label_digest = hashlib.sha256(label_bytes).hexdigest()
        package = artifact_service.publish_artifact_package(
            db,
            storage,
            project_id=data.project_id,
            data=ArtifactPackageCreate(
                package_kind="label_set",
                package_format="json",
                schema_version="label-set-v1",
                display_name=data.name,
                loader_policy="safe",
                task_contract={
                    "dataset_version_id": data.dataset_version_id,
                    "task_version_id": data.task_version_id,
                    "composition_policy": data.composition_policy,
                },
                sensitivity="project",
                license_info=dataset_version.license_info,
                metadata={
                    "source_kind": data.source_kind,
                    "label_count": len(data.labels),
                    "source_annotation_round_id": source_annotation_round_id,
                    "source_submission_ids": submission_ids,
                    "source_decision_ids": decision_ids,
                },
            ),
            files=[
                artifact_service.PackageFileUpload(
                    relative_path="labels.json",
                    source=label_bytes,
                    role="label_set",
                    content_type="application/json",
                    expected_checksum_sha256=label_digest,
                    expected_size_bytes=len(label_bytes),
                )
            ],
            actor_user_id=actor.id,
        )
        artifact_package_id = package.id
    label_set = models.LabelSetVersion(
        **data.model_dump(exclude={"artifact_package_id"}),
        artifact_package_id=artifact_package_id,
        source_annotation_round_id=source_annotation_round_id,
        source_submission_ids=submission_ids,
        source_decision_ids=decision_ids,
        version_number=int(latest or 0) + 1,
        label_count=len(data.labels),
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
    )
    db.add(label_set)
    _persist(
        db,
        "This label set version already exists",
        commit=commit,
    )
    db.refresh(label_set)
    return label_set


def create_imported_label_set_from_field(
    db: Session,
    data: schemas.ImportedLabelSetFromFieldCreate,
    actor: User,
    *,
    storage: ObjectStorage | None = None,
    commit: bool = True,
) -> models.LabelSetVersion:
    dataset_version = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    items = (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == dataset_version.id)
        .order_by(models.DatasetItem.id)
        .all()
    )
    if not items:
        raise ValidationError("Imported labels require a materialized dataset snapshot")
    missing = [item.stable_key for item in items if data.label_field not in item.payload]
    if missing:
        raise ValidationError(
            f"Label field {data.label_field!r} is missing from dataset items: "
            + ", ".join(missing[:5])
        )
    labels = {item.stable_key: item.payload[data.label_field] for item in items}
    return create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=data.project_id,
            dataset_version_id=dataset_version.id,
            task_version_id=data.task_version_id,
            parent_version_id=data.parent_version_id,
            name=data.name,
            source_kind="imported",
            composition_policy=data.composition_policy,
            labels=labels,
        ),
        actor,
        storage=storage,
        commit=commit,
    )


def create_round_label_set(
    db: Session,
    annotation_round_id: int,
    data: schemas.RoundLabelSetCreate,
    actor: User,
    *,
    storage: ObjectStorage | None = None,
) -> models.LabelSetVersion:
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        data.project_id,
        "Annotation round",
    )
    if annotation_round.status != "closed":
        raise ConflictError(
            "A round must be closed before its finalized decisions become a label set"
        )
    if len(data.submission_ids) != len(set(data.submission_ids)):
        raise ValidationError("submission_ids must be unique")
    submissions = [
        _scoped(
            db,
            models.RoundSubmission,
            submission_id,
            data.project_id,
            "Round submission",
        )
        for submission_id in data.submission_ids
    ]
    if any(submission.annotation_round_id != annotation_round.id for submission in submissions):
        raise ValidationError("All submissions must belong to the selected round")

    selected_decisions: list[models.RoundAnnotationDecision] = []
    accepted_kinds = (
        {"adjudication"} if data.source_kind == "adjudicated" else {"annotation", "correction"}
    )
    for submission in submissions:
        for decision_id in submission.decision_ids:
            decision = _scoped(
                db,
                models.RoundAnnotationDecision,
                decision_id,
                data.project_id,
                "Submitted annotation decision",
            )
            if (
                decision.annotator_user_id != submission.annotator_user_id
                or decision.decision_kind not in accepted_kinds
            ):
                continue
            selected_decisions.append(decision)
    if not selected_decisions:
        raise ValidationError(
            f"No submitted {data.source_kind} decisions are available for composition"
        )

    output_by_key: dict[str, list[tuple[str, Any]]] = {}
    for decision in selected_decisions:
        round_item = _scoped(
            db,
            models.RoundItem,
            decision.round_item_id,
            data.project_id,
            "Round item",
        )
        if round_item.annotation_round_id != annotation_round.id:
            raise ValidationError("Submitted decision does not belong to the round")
        dataset_item = _scoped(
            db,
            models.DatasetItem,
            round_item.dataset_item_id,
            data.project_id,
            "Dataset item",
        )
        output_by_key.setdefault(dataset_item.stable_key, []).append(
            (_canonical_hash(decision.output), decision.output)
        )

    labels: dict[str, Any] = {}
    disagreements: list[str] = []
    for stable_key, outputs in sorted(output_by_key.items()):
        unique = {output_hash: output for output_hash, output in outputs}
        if len(unique) != 1:
            disagreements.append(stable_key)
            continue
        labels[stable_key] = next(iter(unique.values()))
    if disagreements:
        raise ConflictError(
            "Submitted decisions disagree and require adjudication before training: "
            + ", ".join(disagreements[:5])
        )

    return create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=data.project_id,
            dataset_version_id=annotation_round.dataset_version_id,
            task_version_id=annotation_round.task_version_id,
            parent_version_id=data.parent_version_id,
            name=data.name,
            source_kind=data.source_kind,
            composition_policy=data.composition_policy,
            labels=labels,
        ),
        actor,
        storage=storage,
        source_annotation_round_id=annotation_round.id,
        source_submission_ids=[submission.id for submission in submissions],
        source_decision_ids=[decision.id for decision in selected_decisions],
    )


def list_label_set_versions(
    db: Session, project_id: int, dataset_version_id: int
) -> list[models.LabelSetVersion]:
    _scoped(
        db,
        models.DatasetVersion,
        dataset_version_id,
        project_id,
        "Dataset version",
    )
    return (
        db.query(models.LabelSetVersion)
        .filter(models.LabelSetVersion.dataset_version_id == dataset_version_id)
        .order_by(
            models.LabelSetVersion.name,
            models.LabelSetVersion.version_number,
        )
        .all()
    )


def create_split_map(
    db: Session,
    data: schemas.SplitMapCreate,
    actor: User,
    *,
    commit: bool = True,
) -> models.SplitMap:
    dataset_version = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    items = (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == dataset_version.id)
        .all()
    )
    item_by_key = {item.stable_key: item for item in items}
    unknown = sorted(set(data.assignments) - set(item_by_key))
    if unknown and dataset_version.item_count:
        raise ValidationError(
            f"Split assignments reference unknown dataset item keys: {', '.join(unknown[:5])}"
        )
    missing = sorted(set(item_by_key) - set(data.assignments))
    if missing:
        raise ValidationError(
            f"Split assignments must cover every dataset item: {', '.join(missing[:5])}"
        )
    protected = set(data.protected_splits)
    if not protected.issubset({"pool", "train", "validation", "test"}):
        raise ValidationError("protected_splits contains an unsupported split")

    group_splits: dict[str, str] = {}
    for stable_key, split in data.assignments.items():
        item = item_by_key.get(stable_key)
        if item is None or item.group_key is None:
            continue
        prior = group_splits.setdefault(item.group_key, split)
        if prior != split:
            raise ValidationError(f"Dataset group {item.group_key!r} cannot span multiple splits")

    split_map = models.SplitMap(
        **data.model_dump(),
        content_hash=_canonical_hash(data.model_dump(exclude={"project_id"})),
        created_by_user_id=actor.id,
    )
    db.add(split_map)
    _persist(
        db,
        "A split map with this name already exists for the dataset version",
        commit=commit,
    )
    db.refresh(split_map)
    return split_map


def _compose_label_sets(
    label_sets: list[models.LabelSetVersion],
    composition: list[dict],
) -> tuple[list[dict], dict]:
    by_id = {label_set.id: label_set for label_set in label_sets}
    if not composition:
        normalized = [
            {
                "label_set_version_id": label_set.id,
                "policy": label_set.composition_policy,
            }
            for label_set in label_sets
        ]
    else:
        normalized = []
        for index, raw_step in enumerate(composition):
            if not isinstance(raw_step, dict):
                raise ValidationError(f"Label composition step {index + 1} must be an object")
            unknown = set(raw_step) - {"label_set_version_id", "policy"}
            if unknown:
                raise ValidationError(
                    f"Label composition step {index + 1} has unsupported fields: "
                    + ", ".join(sorted(unknown))
                )
            label_set_id = raw_step.get("label_set_version_id")
            policy = raw_step.get("policy")
            if (
                isinstance(label_set_id, bool)
                or not isinstance(label_set_id, int)
                or label_set_id not in by_id
            ):
                raise ValidationError(
                    f"Label composition step {index + 1} references an undeclared label set"
                )
            if policy not in {"replace", "inherit", "exclude"}:
                raise ValidationError(
                    f"Label composition step {index + 1} has an unsupported policy"
                )
            normalized.append(
                {
                    "label_set_version_id": label_set_id,
                    "policy": policy,
                }
            )
    step_ids = [step["label_set_version_id"] for step in normalized]
    if len(step_ids) != len(set(step_ids)):
        raise ValidationError("Each label set may appear only once in label composition")
    if set(step_ids) != set(by_id):
        raise ValidationError(
            "Label composition must reference every declared label set exactly once"
        )

    composed: dict = {}
    for step in normalized:
        label_set = by_id[step["label_set_version_id"]]
        policy = step["policy"]
        if policy == "replace":
            composed.update(label_set.labels)
        elif policy == "inherit":
            for stable_key, label in label_set.labels.items():
                composed.setdefault(stable_key, label)
        else:
            for stable_key in label_set.labels:
                composed.pop(stable_key, None)
    return normalized, composed


def create_training_dataset_version(
    db: Session,
    data: schemas.TrainingDatasetVersionCreate,
    actor: User,
    *,
    commit: bool = True,
) -> models.TrainingDatasetVersion:
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
    for dataset_item in (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == dataset_version.id)
        .order_by(models.DatasetItem.id)
        .all()
    ):
        _validate_task_input(
            task_version,
            dataset_item.payload,
            label=f"Dataset item {dataset_item.stable_key!r}",
        )
    split_map = _scoped(db, models.SplitMap, data.split_map_id, data.project_id, "Split map")
    if split_map.dataset_version_id != data.dataset_version_id:
        raise ValidationError("Split map must use the training dataset version")
    protected_assignments = {
        split
        for split in split_map.assignments.values()
        if split in set(split_map.protected_splits)
    }
    if "test" not in set(split_map.protected_splits) or "test" not in protected_assignments:
        raise ValidationError("Training datasets require a non-empty protected test split")
    label_sets = [
        _scoped(db, models.LabelSetVersion, item_id, data.project_id, "Label set version")
        for item_id in data.label_set_version_ids
    ]
    if len(set(data.label_set_version_ids)) != len(data.label_set_version_ids):
        raise ValidationError("label_set_version_ids must be unique")
    for label_set in label_sets:
        if (
            label_set.dataset_version_id != data.dataset_version_id
            or label_set.task_version_id != data.task_version_id
        ):
            raise ValidationError("All label sets must use the selected dataset and task versions")
    _optional_package(db, data.project_id, data.artifact_package_id)
    normalized_composition, composed_labels = _compose_label_sets(
        label_sets,
        data.composition,
    )
    if not composed_labels:
        raise ValidationError("Training label composition cannot be empty")
    payload = data.model_dump(exclude={"artifact_package_id"})
    payload["composition"] = normalized_composition
    content = {
        **data.model_dump(exclude={"project_id", "artifact_package_id", "composition"}),
        "composition": normalized_composition,
        "composed_labels_hash": _canonical_hash(composed_labels),
    }
    training_dataset = models.TrainingDatasetVersion(
        **payload,
        artifact_package_id=data.artifact_package_id,
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
    )
    db.add(training_dataset)
    _persist(
        db,
        "This exact training dataset version already exists",
        commit=commit,
    )
    db.refresh(training_dataset)
    return training_dataset
