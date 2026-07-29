"""Worker-only execution for immutable training runs.

This module deliberately imports trainer implementations only when a worker
executes a run. API processes may import the queue entry point without loading
optional ML runtimes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import mimetypes
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from al_medlit.core.storage import ObjectStorage
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import ArtifactPackage, BaseModelAsset
from al_medlit.model_artifacts.schemas import (
    ArtifactPackageCreate,
    ArtifactPackageReferenceCreate,
)
from al_medlit.project.models import Project
from al_medlit.training.evaluators.contracts import (
    EvaluationInput,
    EvaluationOutput,
    EvaluatorPlugin,
    EvaluatorPluginRegistry,
)
from al_medlit.training.recipe_registry import TrainingRecipeRegistry, training_recipes
from al_medlit.training.runtime_profiles import (
    RuntimeReadinessReport,
    runtime_profile_report_sha256,
    validate_ready_runtime_report,
)
from al_medlit.training.trainers.contracts import (
    TrainerPlugin,
    TrainerPluginRegistry,
    TrainerPreflight,
    TrainingInput,
    TrainingOutput,
    TrainingPlan,
)
from al_medlit.workflow import models as workflow_models
from al_medlit.workflow import schemas as workflow_schemas
from al_medlit.workflow import service as workflow_service
from al_medlit.workspace.dependencies import ROLE_RANK
from al_medlit.workspace.models import WorkspaceMember

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_WORKER_HEARTBEAT_KEY = "worker_heartbeat_at"
_TRAINING_SPLITS = {"train", "validation"}
_NEVER_TRAIN_SPLITS = {"test", "pool"}
_KNOWN_RETENTION_CLASSES = {"indefinite", "resume_14d", "candidate"}
_SENSITIVE_KEY_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}


@dataclass(frozen=True, slots=True)
class _PinnedRun:
    run: workflow_models.TrainingRun
    workspace_id: int
    registered_model: workflow_models.RegisteredModel
    task: workflow_models.TaskVersion
    training_dataset: workflow_models.TrainingDatasetVersion
    dataset: workflow_models.DatasetVersion
    split_map: workflow_models.SplitMap
    label_sets: tuple[workflow_models.LabelSetVersion, ...]
    recipe: workflow_models.TrainingRecipe
    recipe_version: workflow_models.TrainingRecipeVersion
    environment: workflow_models.ExecutionEnvironment
    storage_policy: workflow_models.StoragePolicy


@dataclass(frozen=True, slots=True)
class _BaseModel:
    asset: BaseModelAsset
    path: Path


@dataclass(frozen=True, slots=True)
class _ProtectedEvaluationResult:
    evaluation_input: EvaluationInput
    evaluator: EvaluatorPlugin | None
    output: EvaluationOutput | None
    artifact_package: ArtifactPackage | None
    status: str
    status_reason: str | None


ArtifactPublisher = Callable[..., ArtifactPackage]


def _canonical_hash(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValidationError("Training lineage must be canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _claim_training_run(
    db: Session,
    training_run_id: int,
) -> tuple[workflow_models.TrainingRun, bool]:
    run = (
        db.query(workflow_models.TrainingRun)
        .filter(workflow_models.TrainingRun.id == training_run_id)
        .with_for_update()
        .one_or_none()
    )
    if run is None:
        raise NotFoundError("Training run not found")
    if run.status in _TERMINAL_STATUSES or run.status == "running":
        db.rollback()
        return db.get(workflow_models.TrainingRun, training_run_id), False
    if run.status != "queued":
        db.rollback()
        raise ConflictError(f"Cannot execute training run from status '{run.status}'")
    reservation = artifact_quota.renew_owner_artifact_reservation(
        db,
        owner_type="training_run",
        owner_id=run.id,
    )
    if reservation is None or reservation.status != "active":
        db.rollback()
        raise ConflictError("Training run has no active artifact reservation")
    claimed_at = datetime.now(UTC)
    run.status = "running"
    run.started_at = claimed_at
    run.completed_at = None
    run.failure_code = None
    run.failure_reason = None
    run.runtime_snapshot = {
        **run.runtime_snapshot,
        _WORKER_HEARTBEAT_KEY: claimed_at.isoformat(),
    }
    db.commit()
    return db.get(workflow_models.TrainingRun, training_run_id), True


def heartbeat_training_run(
    db: Session,
    training_run_id: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Renew one running worker lease and record its durable heartbeat."""

    run = (
        db.query(workflow_models.TrainingRun)
        .filter(workflow_models.TrainingRun.id == training_run_id)
        .with_for_update()
        .one_or_none()
    )
    if run is None:
        raise NotFoundError("Training run not found")
    if run.status != "running":
        db.rollback()
        return False
    reservation = artifact_quota.renew_owner_artifact_reservation(
        db,
        owner_type="training_run",
        owner_id=run.id,
    )
    if reservation is None or reservation.status != "active":
        db.rollback()
        return False
    heartbeat_at = now or datetime.now(UTC)
    run.runtime_snapshot = {
        **run.runtime_snapshot,
        _WORKER_HEARTBEAT_KEY: heartbeat_at.isoformat(),
    }
    db.commit()
    return True


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _worker_heartbeat_at(run: workflow_models.TrainingRun) -> datetime | None:
    value = run.runtime_snapshot.get(_WORKER_HEARTBEAT_KEY)
    if isinstance(value, str):
        try:
            return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            pass
    if run.started_at is not None:
        return _as_utc(run.started_at)
    return None


def reconcile_stale_training_runs(
    db: Session,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = 300,
    limit: int = 200,
) -> list[int]:
    """Fail abandoned worker leases so runs and quota cannot remain stuck."""

    if stale_after_seconds < 60:
        raise ValidationError("Training worker timeout must be at least 60 seconds")
    effective_now = now or datetime.now(UTC)
    cutoff = effective_now - timedelta(seconds=stale_after_seconds)
    candidates = (
        db.query(workflow_models.TrainingRun)
        .filter(workflow_models.TrainingRun.status == "running")
        .order_by(workflow_models.TrainingRun.id)
        .limit(min(max(limit, 1), 1000))
        .all()
    )
    recovered_ids: list[int] = []
    for candidate in candidates:
        heartbeat_at = _worker_heartbeat_at(candidate)
        if heartbeat_at is not None and heartbeat_at > cutoff:
            continue
        run = (
            db.query(workflow_models.TrainingRun)
            .filter(workflow_models.TrainingRun.id == candidate.id)
            .with_for_update()
            .one()
        )
        heartbeat_at = _worker_heartbeat_at(run)
        if run.status != "running" or (
            heartbeat_at is not None and heartbeat_at > cutoff
        ):
            db.rollback()
            continue
        snapshot = {
            **run.runtime_snapshot,
            "worker_recovery": {
                "detected_at": effective_now.isoformat(),
                "last_heartbeat_at": (
                    heartbeat_at.isoformat() if heartbeat_at is not None else None
                ),
                "stale_after_seconds": stale_after_seconds,
                "outcome": "failed_safe_to_relaunch",
            },
        }
        workflow_service.transition_training_run(
            db,
            run.project_id,
            run.id,
            workflow_schemas.TrainingRunTransition(
                status="failed",
                failure_code="worker_heartbeat_expired",
                failure_reason=(
                    "The training worker stopped reporting progress. "
                    "The run was closed safely and may be relaunched."
                ),
                runtime_snapshot=snapshot,
            ),
        )
        recovered_ids.append(run.id)
    return recovered_ids


def _scoped(
    db: Session,
    model_type,
    record_id: int,
    project_id: int,
    label: str,
):
    record = db.get(model_type, record_id)
    if record is None or getattr(record, "project_id", None) != project_id:
        raise NotFoundError(f"{label} not found")
    return record


def _load_pinned_run(db: Session, run: workflow_models.TrainingRun) -> _PinnedRun:
    project_id = run.project_id
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    registered = _scoped(
        db,
        workflow_models.RegisteredModel,
        run.registered_model_id,
        project_id,
        "Registered model",
    )
    task = _scoped(
        db,
        workflow_models.TaskVersion,
        run.task_version_id,
        project_id,
        "Task version",
    )
    training_dataset = _scoped(
        db,
        workflow_models.TrainingDatasetVersion,
        run.training_dataset_version_id,
        project_id,
        "Training dataset version",
    )
    dataset = _scoped(
        db,
        workflow_models.DatasetVersion,
        training_dataset.dataset_version_id,
        project_id,
        "Dataset version",
    )
    split_map = _scoped(
        db,
        workflow_models.SplitMap,
        training_dataset.split_map_id,
        project_id,
        "Split map",
    )
    label_sets = tuple(
        _scoped(
            db,
            workflow_models.LabelSetVersion,
            label_set_id,
            project_id,
            "Label set version",
        )
        for label_set_id in training_dataset.label_set_version_ids
    )
    recipe_version = _scoped(
        db,
        workflow_models.TrainingRecipeVersion,
        run.recipe_version_id,
        project_id,
        "Training recipe version",
    )
    recipe = _scoped(
        db,
        workflow_models.TrainingRecipe,
        recipe_version.training_recipe_id,
        project_id,
        "Training recipe",
    )
    environment = _scoped(
        db,
        workflow_models.ExecutionEnvironment,
        run.environment_id,
        project_id,
        "Execution environment",
    )
    storage_policy = _scoped(
        db,
        workflow_models.StoragePolicy,
        run.storage_policy_id,
        project_id,
        "Storage policy",
    )
    if (
        training_dataset.task_version_id != task.id
        or split_map.dataset_version_id != dataset.id
        or training_dataset.dataset_version_id != dataset.id
    ):
        raise ValidationError("Training run has inconsistent task, dataset, or split lineage")
    for label_set in label_sets:
        if (
            label_set.dataset_version_id != dataset.id
            or label_set.task_version_id != task.id
        ):
            raise ValidationError("Training run contains an incompatible label set")
    if recipe_version.environment_class != environment.environment_class:
        raise ValidationError("Training recipe and environment runtime classes do not match")
    if environment.status != "available":
        raise ConflictError("Execution environment is no longer available")
    if task.task_kind not in recipe_version.compatible_task_kinds:
        raise ValidationError("Training recipe is incompatible with the pinned task")
    return _PinnedRun(
        run=run,
        workspace_id=project.workspace_id,
        registered_model=registered,
        task=task,
        training_dataset=training_dataset,
        dataset=dataset,
        split_map=split_map,
        label_sets=label_sets,
        recipe=recipe,
        recipe_version=recipe_version,
        environment=environment,
        storage_policy=storage_policy,
    )


def _assert_no_sensitive_config(value: Any, *, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).lower()
            if any(fragment in key for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValidationError(f"Training {path} cannot contain credentials or secrets")
            _assert_no_sensitive_config(nested, path=f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _assert_no_sensitive_config(nested, path=f"{path}[{index}]")


def _resolve_recipe(
    pinned: _PinnedRun,
    recipe_catalog: TrainingRecipeRegistry,
) -> tuple[Any, dict]:
    try:
        descriptor = recipe_catalog.get(pinned.recipe.key)
    except NotFoundError as exc:
        raise ValidationError(
            f"Custom training recipe '{pinned.recipe.key}' cannot execute yet because "
            "its model-family and artifact-format contract is not registered"
        ) from exc
    if descriptor.trainer_key != pinned.recipe_version.trainer_plugin_key:
        raise ValidationError(
            "Pinned trainer plugin does not match the trusted recipe contract"
        )
    if descriptor.environment.runtime_class.value != pinned.environment.environment_class:
        raise ValidationError(
            "Pinned execution environment does not match the trusted recipe contract"
        )
    if pinned.task.task_kind not in {
        task_kind.value for task_kind in descriptor.supported_task_kinds
    }:
        raise ValidationError("Pinned task kind is not supported by the trusted recipe")
    config = {
        **pinned.recipe_version.default_config,
        **pinned.run.config,
    }
    _assert_no_sensitive_config(config)
    validation = recipe_catalog.validate(pinned.recipe.key, config)
    if not validation.valid or validation.normalized_config is None:
        raise ValidationError(
            "Training recipe configuration is invalid: " + "; ".join(validation.errors)
        )
    return descriptor, validation.normalized_config


def _target_field(config: Mapping) -> str:
    fields = config.get("fields")
    if isinstance(fields, Mapping) and isinstance(fields.get("target_field"), str):
        return fields["target_field"]
    for key in ("target_field", "completion_field"):
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValidationError("Training recipe does not declare a target field")


def _label_value(label: Any, target_field: str) -> Any:
    if isinstance(label, Mapping) and target_field in label:
        return label[target_field]
    return label


def _declared_label_vocabulary(
    task: workflow_models.TaskVersion,
    target_field: str,
    labels: Sequence[Any],
) -> tuple[str, ...]:
    candidates: Any = None
    for source in (task.label_rules, task.output_schema):
        if not isinstance(source, Mapping):
            continue
        for key in ("labels", "classes", target_field):
            value = source.get(key)
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                candidates = value
                break
        if candidates is not None:
            break
    if candidates is None:
        discovered: set[str] = set()
        for label in labels:
            value = _label_value(label, target_field)
            if isinstance(value, str):
                discovered.add(value)
            elif isinstance(value, list):
                discovered.update(item for item in value if isinstance(item, str))
        candidates = sorted(discovered)
    return tuple(dict.fromkeys(candidates))


def _training_input(
    db: Session,
    pinned: _PinnedRun,
    config: Mapping,
    *,
    base_model: _BaseModel | None,
) -> TrainingInput:
    composed = workflow_service.compose_training_dataset_labels(
        db,
        pinned.run.project_id,
        pinned.training_dataset.id,
    )
    item_by_key = {
        item.stable_key: item
        for item in db.query(workflow_models.DatasetItem)
        .filter(workflow_models.DatasetItem.dataset_version_id == pinned.dataset.id)
        .order_by(workflow_models.DatasetItem.stable_key)
        .all()
    }
    target_field = _target_field(config)
    rows_by_split: dict[str, list[dict]] = {"train": [], "validation": []}
    labels_used: list[Any] = []
    for stable_key in sorted(composed.labels):
        item = item_by_key.get(stable_key)
        if item is None:
            raise ValidationError(
                f"Composed label references missing dataset item '{stable_key}'"
            )
        split = pinned.split_map.assignments.get(stable_key)
        if split in _NEVER_TRAIN_SPLITS:
            continue
        if split not in _TRAINING_SPLITS:
            raise ValidationError(
                f"Labeled item '{stable_key}' has no governed train/validation split"
            )
        if split in set(pinned.split_map.protected_splits):
            raise ValidationError(
                f"Protected split '{split}' cannot be used by a training run"
            )
        row = dict(item.payload)
        label = composed.labels[stable_key]
        row[target_field] = _label_value(label, target_field)
        rows_by_split[split].append(row)
        labels_used.append(label)
    if not rows_by_split["train"]:
        raise ValidationError(
            "Training dataset has no labeled items assigned to the train split"
        )
    return TrainingInput(
        rows=tuple(rows_by_split["train"]),
        validation_rows=tuple(rows_by_split["validation"]),
        dataset_fingerprint=pinned.training_dataset.content_hash,
        split_fingerprint=pinned.split_map.content_hash,
        task_kind=pinned.task.task_kind,
        task_schema={
            "input_schema": pinned.task.input_schema,
            "output_schema": pinned.task.output_schema,
            "label_rules": pinned.task.label_rules,
        },
        label_vocabulary=_declared_label_vocabulary(
            pinned.task,
            target_field,
            labels_used,
        ),
        base_model_path=str(base_model.path) if base_model is not None else None,
        base_model_fingerprint=(
            base_model.asset.package.manifest_digest if base_model is not None else None
        ),
    )


def _evaluation_plan_splits(pinned: _PinnedRun) -> tuple[str, ...]:
    raw_splits = pinned.run.evaluation_plan.get("splits")
    if raw_splits is None:
        raw_splits = pinned.run.evaluation_plan.get("protected_splits", ())
    if raw_splits is None:
        return ()
    if not isinstance(raw_splits, (list, tuple)) or any(
        not isinstance(split, str) or not split.strip() for split in raw_splits
    ):
        raise ValidationError("Evaluation plan splits must be a list of split names")
    normalized = tuple(dict.fromkeys(split.strip() for split in raw_splits))
    unsupported = sorted(set(normalized) - {"validation", "test"})
    if unsupported:
        raise ValidationError(
            "Evaluation plan contains unsupported splits: "
            + ", ".join(unsupported)
        )
    return normalized


def _requested_evaluation_metrics(pinned: _PinnedRun) -> tuple[str, ...]:
    raw_metrics = pinned.run.evaluation_plan.get("metrics")
    if raw_metrics is None:
        raw_metrics = pinned.recipe_version.evaluation_defaults.get("metrics", ())
    if raw_metrics is None:
        return ()
    if not isinstance(raw_metrics, (list, tuple)) or any(
        not isinstance(metric, str) or not metric.strip() for metric in raw_metrics
    ):
        raise ValidationError("Evaluation plan metrics must be a list of metric names")
    return tuple(dict.fromkeys(metric.strip() for metric in raw_metrics))


def _protected_test_evaluation_input(
    db: Session,
    pinned: _PinnedRun,
    config: Mapping,
    training_input: TrainingInput,
    *,
    model_fingerprint: str,
) -> EvaluationInput:
    if "test" not in set(pinned.split_map.protected_splits):
        raise ValidationError("Test evaluation requires a protected test split")
    item_by_key = {
        item.stable_key: item
        for item in db.query(workflow_models.DatasetItem)
        .filter(workflow_models.DatasetItem.dataset_version_id == pinned.dataset.id)
        .order_by(workflow_models.DatasetItem.stable_key)
        .all()
    }
    test_keys = sorted(
        stable_key
        for stable_key, split in pinned.split_map.assignments.items()
        if split == "test"
    )
    if not test_keys:
        raise ValidationError("Protected test evaluation requires at least one test item")
    composed = workflow_service.compose_training_dataset_labels(
        db,
        pinned.run.project_id,
        pinned.training_dataset.id,
    )
    missing_labels = [stable_key for stable_key in test_keys if stable_key not in composed.labels]
    if missing_labels:
        raise ValidationError(
            "Protected test evaluation requires labels for every test item: "
            + ", ".join(missing_labels[:5])
        )
    target_field = _target_field(config)
    rows = []
    for stable_key in test_keys:
        item = item_by_key.get(stable_key)
        if item is None:
            raise ValidationError(
                f"Protected test split references missing dataset item '{stable_key}'"
            )
        row = dict(item.payload)
        row[target_field] = _label_value(
            composed.labels[stable_key],
            target_field,
        )
        rows.append(row)
    return EvaluationInput(
        rows=tuple(rows),
        split_name="test",
        protected_split=True,
        requested_metrics=_requested_evaluation_metrics(pinned),
        dataset_fingerprint=pinned.dataset.content_hash,
        training_dataset_fingerprint=pinned.training_dataset.content_hash,
        split_fingerprint=pinned.split_map.content_hash,
        model_fingerprint=model_fingerprint,
        task_kind=pinned.task.task_kind,
        task_schema={
            "input_schema": pinned.task.input_schema,
            "output_schema": pinned.task.output_schema,
            "label_rules": pinned.task.label_rules,
        },
        label_vocabulary=training_input.label_vocabulary,
    )


def _stage_base_model(
    db: Session,
    storage: ObjectStorage,
    pinned: _PinnedRun,
    config: Mapping,
    destination: Path,
) -> _BaseModel | None:
    asset_id = config.get("base_model_asset_id")
    if asset_id is None:
        return None
    if isinstance(asset_id, bool) or not isinstance(asset_id, int):
        raise ValidationError("base_model_asset_id must be an integer")
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None or asset.project_id != pinned.run.project_id:
        raise NotFoundError("Base model asset not found")
    retention = asset.package.retention
    if (
        asset.state is None
        or asset.state.readiness != "ready"
        or asset.state.archived_at is not None
        or asset.package.package_kind != "base_model"
        or asset.package.readiness != "ready"
        or not asset.package.deployable
        or asset.package.loader_policy != "safe"
        or retention is None
        or retention.archived_at is not None
        or retention.purged_at is not None
    ):
        raise ConflictError("Base model asset is not ready for safe local execution")
    creator = (
        db.get(User, pinned.run.created_by_user_id)
        if pinned.run.created_by_user_id is not None
        else None
    )
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == asset.workspace_id,
            WorkspaceMember.user_id == pinned.run.created_by_user_id,
        )
        .one_or_none()
        if pinned.run.created_by_user_id is not None
        else None
    )
    can_use_manager_assets = bool(
        creator
        and (
            creator.is_superuser
            or (
                member is not None
                and ROLE_RANK.get(member.role, -1) >= ROLE_RANK["manager"]
            )
        )
    )
    if asset.access_mode == "manager_only" and not can_use_manager_assets:
        raise ForbiddenError("This base model is restricted to workspace managers")

    destination.mkdir(parents=True, exist_ok=False)
    for package_file in asset.package.files:
        relative = artifact_service.validate_relative_path(package_file.relative_path)
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, chunks = artifact_service.iter_package_file(
            db,
            storage,
            project_id=pinned.run.project_id,
            package_id=asset.package_id,
            relative_path=relative,
        )
        digest = hashlib.sha256()
        size = 0
        with target.open("xb") as sink:
            for chunk in chunks:
                sink.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if (
            digest.hexdigest() != descriptor.checksum_sha256
            or size != descriptor.size_bytes
        ):
            raise ConflictError(f"Base model file '{relative}' failed integrity verification")
        target.chmod(0o444)
    for directory, directories, _files in os.walk(destination, topdown=False):
        for name in directories:
            (Path(directory) / name).chmod(0o555)
    destination.chmod(0o555)
    return _BaseModel(asset=asset, path=destination)


def _validate_preflight(
    report: TrainerPreflight,
    pinned: _PinnedRun,
) -> None:
    if report.runtime_class != pinned.environment.environment_class:
        raise ValidationError("Trainer preflight reported the wrong runtime class")
    failing_checks = [
        str(check.get("message", "worker check failed"))
        for check in report.checks
        if check.get("status") != "pass"
    ]
    if report.ready and report.checks and not failing_checks:
        return
    raise ConflictError(
        "Selected worker runtime is unavailable"
        + (": " + "; ".join(failing_checks) if failing_checks else "")
    )


def _validate_environment_and_storage_binding(
    pinned: _PinnedRun,
    storage: ObjectStorage,
) -> None:
    runtime_snapshot = pinned.run.runtime_snapshot
    if (
        runtime_snapshot.get("environment_class")
        != pinned.environment.environment_class
        or runtime_snapshot.get("image_digest") != pinned.environment.image_digest
        or runtime_snapshot.get("package_manifest")
        != pinned.environment.package_manifest
        or runtime_snapshot.get("hardware_constraints")
        != pinned.environment.hardware_constraints
    ):
        raise ConflictError("Execution environment no longer matches the launch snapshot")
    verification = runtime_snapshot.get("verification_report")
    if not isinstance(verification, dict):
        raise ConflictError("Training run has no pinned readiness attestation")
    raw_report = verification.get("readiness_report")
    if not isinstance(raw_report, dict):
        raise ConflictError("Execution environment has no verified readiness report")
    try:
        report = RuntimeReadinessReport.model_validate(raw_report)
        validate_ready_runtime_report(report)
    except Exception as exc:
        raise ConflictError("Execution environment readiness report is invalid") from exc
    if (
        report.runtime_profile != pinned.environment.environment_class
        or verification.get("report_sha256") != runtime_profile_report_sha256(report)
    ):
        raise ConflictError("Execution environment readiness attestation is not valid")
    expected_digest = pinned.environment.image_digest.removeprefix("sha256:").lower()
    if (
        report.worker_image_digest is None
        or report.worker_image_digest.removeprefix("sha256:").lower() != expected_digest
    ):
        raise ConflictError("Execution environment image attestation does not match")
    worker_digest = os.environ.get("AL_MEDLIT_WORKER_IMAGE_DIGEST", "")
    if worker_digest.removeprefix("sha256:").lower() != expected_digest:
        raise ConflictError("Training run reached a worker with the wrong image digest")
    backend_name = getattr(storage, "backend_name", None)
    storage_snapshot = pinned.run.storage_snapshot
    if (
        storage_snapshot.get("backend") != pinned.storage_policy.backend
        or storage_snapshot.get("artifact_prefix")
        != pinned.storage_policy.artifact_prefix
        or storage_snapshot.get("retention_class")
        != pinned.storage_policy.retention_class
        or storage_snapshot.get("encryption") != pinned.storage_policy.encryption
        or storage_snapshot.get("cache_policy") != pinned.storage_policy.cache_policy
    ):
        raise ConflictError("Storage policy no longer matches the launch snapshot")
    if backend_name != pinned.storage_policy.backend:
        raise ConflictError(
            "Selected storage policy does not match the worker object-storage backend"
        )
    expected_prefix = artifact_service.workspace_blob_prefix(pinned.workspace_id)
    actual_prefix = pinned.storage_policy.artifact_prefix.strip("/")
    if actual_prefix != expected_prefix:
        raise ConflictError("Selected storage policy does not match the blob namespace")
    encryption = pinned.storage_policy.encryption
    if not isinstance(encryption, dict) or encryption.get("mode") not in {
        "none",
        "sse-s3",
        "sse-kms",
    }:
        raise ConflictError("Selected storage policy encryption is invalid")
    expected_key_id = (
        encryption.get("key_id") if encryption.get("mode") == "sse-kms" else None
    )
    if (
        getattr(storage, "encryption_mode", None) != encryption["mode"]
        or getattr(storage, "encryption_key_id", None) != expected_key_id
    ):
        raise ConflictError(
            "Selected storage encryption does not match the worker object store"
        )


def _validate_plan(
    plan: TrainingPlan,
    pinned: _PinnedRun,
    descriptor: Any,
    training_input: TrainingInput,
    normalized_config: dict,
) -> None:
    manifest = plan.manifest
    if manifest.get("dataset_fingerprint") != training_input.dataset_fingerprint:
        raise ValidationError("Trainer plan changed the dataset fingerprint")
    if manifest.get("split_fingerprint") != training_input.split_fingerprint:
        raise ValidationError("Trainer plan changed the split fingerprint")
    runtime_class = manifest.get("runtime_class")
    if runtime_class != descriptor.environment.runtime_class.value:
        raise ValidationError("Trainer plan changed the execution runtime class")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, Mapping) or recipe.get("key") != pinned.recipe.key:
        raise ValidationError("Trainer plan changed the recipe identity")
    if plan.normalized_config != normalized_config:
        raise ValidationError("Trainer plan changed the pinned recipe configuration")


def _safe_output_files(
    output: TrainingOutput,
    destination: Path,
) -> tuple[artifact_service.PackageFileUpload, ...]:
    declared = tuple(
        artifact_service.validate_relative_path(path) for path in output.artifact_paths
    )
    if not declared or len(declared) != len(set(declared)):
        raise ValidationError("Trainer output must declare unique artifact paths")
    actual: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValidationError("Trainer output cannot contain symbolic links")
        if path.is_file():
            actual.add(path.relative_to(destination).as_posix())
    if set(declared) != actual:
        raise ValidationError("Trainer artifact inventory does not match output files")
    manifest_path = destination / "manifest.json"
    if "manifest.json" not in actual:
        raise ValidationError("Trainer output must contain manifest.json")
    try:
        on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError("Trainer output manifest is not valid UTF-8 JSON") from exc
    if on_disk_manifest != output.manifest:
        raise ValidationError("Trainer output manifest does not match the returned manifest")

    uploads = []
    for relative in sorted(declared):
        path = destination / Path(relative)
        content_type = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        role = "model_file"
        if relative == "manifest.json":
            role = "manifest"
        elif relative.endswith(".safetensors") or relative.endswith(".skops"):
            role = "weights"
        elif relative.endswith(".json"):
            role = "configuration"
        elif "tokenizer" in Path(relative).name or Path(relative).name in {
            "vocab.txt",
            "vocab.json",
            "spiece.model",
            "sentencepiece.bpe.model",
        }:
            role = "tokenizer"
        uploads.append(
            artifact_service.PackageFileUpload(
                relative_path=relative,
                source=path,
                role=role,
                content_type=content_type,
            )
        )
    return tuple(uploads)


def _package_lineage(
    db: Session,
    pinned: _PinnedRun,
    base_model: _BaseModel | None,
) -> tuple[list[ArtifactPackageReferenceCreate], list[int]]:
    references: list[ArtifactPackageReferenceCreate] = []
    packages: list[tuple[int, str]] = []
    if pinned.dataset.artifact_package_id is not None:
        packages.append((pinned.dataset.artifact_package_id, "derived_from_dataset"))
    if pinned.training_dataset.artifact_package_id is not None:
        packages.append(
            (pinned.training_dataset.artifact_package_id, "derived_from_training_dataset")
        )
    packages.extend(
        (label_set.artifact_package_id, "uses_labels")
        for label_set in pinned.label_sets
        if label_set.artifact_package_id is not None
    )
    if base_model is not None:
        packages.append((base_model.asset.package_id, "uses_base_model"))
    if pinned.run.parent_model_version_id is not None:
        parent = _scoped(
            db,
            workflow_models.ModelVersion,
            pinned.run.parent_model_version_id,
            pinned.run.project_id,
            "Parent model version",
        )
        if parent.checkpoint_package_id is not None:
            packages.append(
                (parent.checkpoint_package_id, "derived_from_parent_model")
            )

    upstream_ids: list[int] = []
    seen: set[tuple[int, str]] = set()
    for package_id, relationship in packages:
        key = (package_id, relationship)
        if key in seen:
            continue
        seen.add(key)
        package = _scoped(
            db,
            ArtifactPackage,
            package_id,
            pinned.run.project_id,
            "Lineage artifact package",
        )
        references.append(
            ArtifactPackageReferenceCreate(
                target_package_id=package.id,
                relationship_type=relationship,
            )
        )
        upstream_ids.append(package.lineage_artifact_id)
    return references, sorted(set(upstream_ids))


def _framework_for_plugin(plugin: TrainerPlugin) -> str:
    if plugin.key == "sklearn_tfidf":
        return "scikit-learn"
    if plugin.key.startswith("huggingface_"):
        return "transformers"
    return plugin.key


def _package_contract(descriptor: Any) -> tuple[str, str]:
    parameterization = descriptor.parameterization.value
    if parameterization in {"lora", "qlora"}:
        return "peft_adapter", "peft_safetensors"
    if descriptor.model_family.value == "conventional_ml":
        return "trained_model", "skops"
    return "trained_model", "huggingface_safetensors"


def _checkpoint_package_data(
    pinned: _PinnedRun,
    descriptor: Any,
    output: TrainingOutput,
    runtime_snapshot: dict,
    base_model: _BaseModel | None,
    *,
    package_kind: str,
    package_format: str,
    display_name: str,
    deployable: bool,
    references: list[ArtifactPackageReferenceCreate],
    retention_class: str,
    protected_test_gate: dict,
) -> ArtifactPackageCreate:
    return ArtifactPackageCreate(
        package_kind=package_kind,
        package_format=package_format,
        display_name=display_name,
        model_family=descriptor.model_family.value,
        model_type=descriptor.architecture_family,
        readiness="ready",
        deployable=deployable,
        loader_policy="safe",
        task_contract={
            "task_version_id": pinned.task.id,
            "task_kind": pinned.task.task_kind,
            "task_content_hash": pinned.task.content_hash,
            "input_schema": pinned.task.input_schema,
            "output_schema": pinned.task.output_schema,
            "label_rules": pinned.task.label_rules,
            "metrics": pinned.task.metrics,
        },
        sensitivity="project",
        license_info={
            "dataset": pinned.dataset.license_info,
            "base_model": (
                base_model.asset.package.license_info
                if base_model is not None
                else {}
            ),
        },
        runtime=runtime_snapshot,
        metadata={
            "training_run_id": pinned.run.id,
            "launch_hash": pinned.run.launch_hash,
            "training_dataset_version_id": pinned.training_dataset.id,
            "training_dataset_content_hash": pinned.training_dataset.content_hash,
            "split_map_id": pinned.split_map.id,
            "split_map_content_hash": pinned.split_map.content_hash,
            "label_set_version_ids": [
                label_set.id for label_set in pinned.label_sets
            ],
            "label_composition": pinned.training_dataset.composition,
            "preprocessing": pinned.training_dataset.preprocessing,
            "recipe_version_id": pinned.recipe_version.id,
            "recipe_content_hash": pinned.recipe_version.content_hash,
            "trainer_output_manifest_sha256": _canonical_hash(output.manifest),
            "evaluation_plan": pinned.run.evaluation_plan,
            "protected_test_gate": protected_test_gate,
            "parent_model_version_id": pinned.run.parent_model_version_id,
            "seed": pinned.run.seed,
            "storage_policy": {
                "id": pinned.storage_policy.id,
                "retention_class": pinned.storage_policy.retention_class,
                "encryption": pinned.storage_policy.encryption,
            },
        },
        references=references,
        retention_class=retention_class,
    )


def _plugin_code_digest(
    plugin: TrainerPlugin,
    pinned: _PinnedRun,
) -> str:
    source_digest = None
    try:
        source_file = inspect.getsourcefile(type(plugin))
        if source_file is not None:
            source_digest = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    except OSError:
        source_digest = None
    return _canonical_hash(
        {
            "plugin_key": plugin.key,
            "plugin_version": pinned.recipe_version.trainer_plugin_version,
            "implementation": f"{type(plugin).__module__}.{type(plugin).__qualname__}",
            "source_sha256": source_digest,
            "recipe_contract_hash": pinned.recipe_version.content_hash,
        }
    )


def _evaluator_code_digest(
    evaluator: EvaluatorPlugin | None,
    pinned: _PinnedRun,
) -> str:
    if evaluator is None:
        return _canonical_hash(
            {
                "evaluator": None,
                "recipe_key": pinned.recipe.key,
                "recipe_contract_hash": pinned.recipe_version.content_hash,
            }
        )
    source_digest = None
    try:
        source_file = inspect.getsourcefile(type(evaluator))
        if source_file is not None:
            source_digest = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    except OSError:
        source_digest = None
    return _canonical_hash(
        {
            "evaluator_key": evaluator.key,
            "evaluator_version": evaluator.evaluator_version,
            "implementation": (
                f"{type(evaluator).__module__}.{type(evaluator).__qualname__}"
            ),
            "source_sha256": source_digest,
            "recipe_contract_hash": pinned.recipe_version.content_hash,
        }
    )


def _evaluation_runtime_digest(
    pinned: _PinnedRun,
    evaluator: EvaluatorPlugin | None,
) -> str:
    return _canonical_hash(
        {
            "environment_id": pinned.environment.id,
            "environment_class": pinned.environment.environment_class,
            "image_digest": pinned.environment.image_digest,
            "verification_report": pinned.environment.verification_report,
            "evaluator_key": evaluator.key if evaluator is not None else None,
            "evaluator_version": (
                evaluator.evaluator_version if evaluator is not None else None
            ),
        }
    )


def _validate_aggregate_evaluation_output(
    output: EvaluationOutput,
    evaluation_input: EvaluationInput,
) -> None:
    if output.prediction_count != len(evaluation_input.rows):
        raise ValidationError(
            "Evaluator prediction count does not match the protected test row count"
        )
    if not output.metrics:
        raise ValidationError("A successful protected test evaluation requires metrics")
    for name, value in output.metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Evaluation metric names must be non-empty")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValidationError(
                f"Evaluation metric '{name}' must be finite or null"
            )

    forbidden_report_keys = {
        "dataset_items",
        "expected",
        "labels",
        "predictions",
        "rows",
        "targets",
    }

    def inspect_report(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key).lower() in forbidden_report_keys:
                    raise ValidationError(
                        "Evaluation reports cannot persist protected row-level data"
                    )
                inspect_report(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                inspect_report(nested)

    inspect_report(output.report)
    _canonical_hash(output.model_dump(mode="json"))


def _evaluation_package_lineage(
    db: Session,
    pinned: _PinnedRun,
    checkpoint_package: ArtifactPackage,
) -> tuple[list[ArtifactPackageReferenceCreate], list[int]]:
    packages: list[tuple[ArtifactPackage, str]] = [
        (checkpoint_package, "evaluates_model"),
    ]
    optional_packages = [
        (
            pinned.dataset.artifact_package_id,
            "uses_evaluation_dataset",
        ),
        (
            pinned.training_dataset.artifact_package_id,
            "uses_training_dataset",
        ),
        *[
            (label_set.artifact_package_id, "uses_labels")
            for label_set in pinned.label_sets
        ],
    ]
    for package_id, relationship in optional_packages:
        if package_id is None:
            continue
        package = _scoped(
            db,
            ArtifactPackage,
            package_id,
            pinned.run.project_id,
            "Evaluation lineage artifact package",
        )
        packages.append((package, relationship))
    references: list[ArtifactPackageReferenceCreate] = []
    upstream_ids: set[int] = set()
    seen: set[tuple[int, str]] = set()
    for package, relationship in packages:
        identity = (package.id, relationship)
        if identity in seen:
            continue
        seen.add(identity)
        references.append(
            ArtifactPackageReferenceCreate(
                target_package_id=package.id,
                relationship_type=relationship,
            )
        )
        upstream_ids.add(package.lineage_artifact_id)
    return references, sorted(upstream_ids)


def _create_model_evaluation(
    db: Session,
    pinned: _PinnedRun,
    model_version: workflow_models.ModelVersion,
    evaluation_input: EvaluationInput,
    *,
    status: str,
    evaluator: EvaluatorPlugin | None,
    metrics: dict[str, float | None],
    report: dict,
    artifact_package_id: int | None,
    status_reason: str | None,
) -> workflow_models.ModelEvaluation:
    payload = {
        "project_id": pinned.run.project_id,
        "training_run_id": pinned.run.id,
        "model_version_id": model_version.id,
        "task_version_id": pinned.task.id,
        "training_dataset_version_id": pinned.training_dataset.id,
        "dataset_version_id": pinned.dataset.id,
        "split_map_id": pinned.split_map.id,
        "artifact_package_id": artifact_package_id,
        "split_name": evaluation_input.split_name,
        "status": status,
        "evaluator_key": evaluator.key if evaluator is not None else None,
        "evaluator_version": (
            evaluator.evaluator_version if evaluator is not None else None
        ),
        "row_count": len(evaluation_input.rows),
        "requested_metrics": list(evaluation_input.requested_metrics),
        "metrics": metrics,
        "report": report,
        "evaluation_plan": pinned.run.evaluation_plan,
        "runtime_digest": _evaluation_runtime_digest(pinned, evaluator),
        "code_digest": _evaluator_code_digest(evaluator, pinned),
        "status_reason": status_reason,
    }
    evaluation = workflow_models.ModelEvaluation(
        **payload,
        content_hash=_canonical_hash(payload),
        created_by_user_id=pinned.run.created_by_user_id,
    )
    db.add(evaluation)
    db.flush()
    return evaluation


def _run_protected_test_evaluation(
    db: Session,
    storage: ObjectStorage,
    pinned: _PinnedRun,
    candidate_package: ArtifactPackage,
    model_directory: Path,
    normalized_config: dict,
    training_input: TrainingInput,
    descriptor: Any,
    evaluator: EvaluatorPlugin | None,
    artifact_publisher: ArtifactPublisher,
) -> _ProtectedEvaluationResult:
    evaluation_input = _protected_test_evaluation_input(
        db,
        pinned,
        normalized_config,
        training_input,
        model_fingerprint=candidate_package.manifest_digest,
    )
    if evaluator is None:
        reason = (
            f"No trusted protected-test evaluator is registered for recipe "
            f"'{pinned.recipe.key}'"
        )
        return _ProtectedEvaluationResult(
            evaluation_input=evaluation_input,
            evaluator=None,
            output=None,
            artifact_package=None,
            status="unsupported",
            status_reason=reason,
        )

    output = evaluator.evaluate(
        recipe_key=pinned.recipe.key,
        config=normalized_config,
        evaluation_input=evaluation_input,
        model_directory=model_directory,
    )
    _validate_aggregate_evaluation_output(output, evaluation_input)
    report_payload = {
        "schema_version": "al-medlit-protected-evaluation-v1",
        "status": "succeeded",
        "training_run_id": pinned.run.id,
        "task_version_id": pinned.task.id,
        "training_dataset_version_id": pinned.training_dataset.id,
        "dataset_version_id": pinned.dataset.id,
        "split_map_id": pinned.split_map.id,
        "split": {
            "name": evaluation_input.split_name,
            "protected": True,
            "row_count": len(evaluation_input.rows),
        },
        "evaluator": {
            "key": evaluator.key,
            "version": evaluator.evaluator_version,
            "code_digest": _evaluator_code_digest(evaluator, pinned),
        },
        "lineage": {
            "task_content_hash": pinned.task.content_hash,
            "dataset_content_hash": pinned.dataset.content_hash,
            "training_dataset_content_hash": pinned.training_dataset.content_hash,
            "split_map_content_hash": pinned.split_map.content_hash,
            "candidate_checkpoint_package_id": candidate_package.id,
            "candidate_checkpoint_manifest_digest": (
                candidate_package.manifest_digest
            ),
        },
        "evaluation_plan": pinned.run.evaluation_plan,
        "requested_metrics": list(evaluation_input.requested_metrics),
        "metrics": output.metrics,
        "report": output.report,
    }
    references, upstream_ids = _evaluation_package_lineage(
        db,
        pinned,
        candidate_package,
    )
    retention = pinned.storage_policy.retention_class
    if retention not in _KNOWN_RETENTION_CLASSES:
        retention = "indefinite"
    evaluation_package = artifact_publisher(
        db,
        storage,
        project_id=pinned.run.project_id,
        data=ArtifactPackageCreate(
            package_kind="evaluation_report",
            package_format="json",
            schema_version="evaluation-package-v1",
            display_name=(
                f"{pinned.registered_model.name} candidate test run {pinned.run.id}"
            ),
            model_family=descriptor.model_family.value,
            model_type=descriptor.architecture_family,
            readiness="ready",
            deployable=False,
            loader_policy="safe",
            task_contract={
                "task_version_id": pinned.task.id,
                "task_kind": pinned.task.task_kind,
                "task_content_hash": pinned.task.content_hash,
                "input_schema": pinned.task.input_schema,
                "output_schema": pinned.task.output_schema,
                "label_rules": pinned.task.label_rules,
                "metrics": pinned.task.metrics,
            },
            sensitivity="restricted",
            license_info={"dataset": pinned.dataset.license_info},
            runtime={
                "environment_id": pinned.environment.id,
                "environment_class": pinned.environment.environment_class,
                "image_digest": pinned.environment.image_digest,
                "evaluator_key": evaluator.key,
                "evaluator_version": evaluator.evaluator_version,
                "runtime_digest": _evaluation_runtime_digest(pinned, evaluator),
            },
            metadata={
                "training_run_id": pinned.run.id,
                "candidate_checkpoint_package_id": candidate_package.id,
                "training_dataset_version_id": pinned.training_dataset.id,
                "dataset_version_id": pinned.dataset.id,
                "split_map_id": pinned.split_map.id,
                "split_map_content_hash": pinned.split_map.content_hash,
                "protected_split": "test",
                "row_count": len(evaluation_input.rows),
                "aggregate_only": True,
            },
            references=references,
            retention_class=retention,
        ),
        files=(
            artifact_service.PackageFileUpload(
                relative_path="evaluation.json",
                source=json.dumps(
                    report_payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii"),
                role="evaluation_report",
                content_type="application/json",
            ),
        ),
        actor_user_id=pinned.run.created_by_user_id,
        upstream_lineage_artifact_ids=upstream_ids,
    )
    return _ProtectedEvaluationResult(
        evaluation_input=evaluation_input,
        evaluator=evaluator,
        output=output,
        artifact_package=evaluation_package,
        status="succeeded",
        status_reason=None,
    )


def _record_protected_evaluation_result(
    db: Session,
    pinned: _PinnedRun,
    model_version: workflow_models.ModelVersion,
    result: _ProtectedEvaluationResult,
) -> workflow_models.ModelEvaluation:
    if result.output is None:
        report = {
            "gate": "blocked",
            "reason": result.status_reason,
            "protected_split": True,
        }
        metrics: dict[str, float | None] = {}
    else:
        report = result.output.report
        metrics = result.output.metrics
    return _create_model_evaluation(
        db,
        pinned,
        model_version,
        result.evaluation_input,
        status=result.status,
        evaluator=result.evaluator,
        metrics=metrics,
        report=report,
        artifact_package_id=(
            result.artifact_package.id
            if result.artifact_package is not None
            else None
        ),
        status_reason=result.status_reason,
    )


def _model_version_payload(
    pinned: _PinnedRun,
    descriptor: Any,
    plugin: TrainerPlugin,
    package: ArtifactPackage,
    output: TrainingOutput,
    normalized_config: dict,
    runtime_snapshot: dict,
    base_model: _BaseModel | None,
) -> workflow_schemas.ModelVersionCreate:
    base_model_lineage: dict = {}
    if base_model is not None:
        base_model_lineage = {
            "asset_id": base_model.asset.id,
            "provider": base_model.asset.provider,
            "source_model_id": base_model.asset.source_model_id,
            "exact_revision": base_model.asset.exact_revision,
            "package_id": base_model.asset.package_id,
            "package_manifest_digest": base_model.asset.package.manifest_digest,
        }
    return workflow_schemas.ModelVersionCreate(
        project_id=pinned.run.project_id,
        registered_model_id=pinned.registered_model.id,
        task_version_id=pinned.task.id,
        training_dataset_version_id=pinned.training_dataset.id,
        parent_version_id=pinned.run.parent_model_version_id,
        family=descriptor.model_family.value,
        framework=_framework_for_plugin(plugin),
        base_model=base_model_lineage,
        training_method=descriptor.parameterization.value,
        recipe_key=pinned.recipe.key,
        recipe_version=str(pinned.recipe_version.version_number),
        parameters={
            **normalized_config,
            "_lineage": {
                "training_run_id": pinned.run.id,
                "launch_hash": pinned.run.launch_hash,
                "task_content_hash": pinned.task.content_hash,
                "dataset_content_hash": pinned.dataset.content_hash,
                "training_dataset_content_hash": pinned.training_dataset.content_hash,
                "split_map_content_hash": pinned.split_map.content_hash,
                "label_set_content_hashes": [
                    label_set.content_hash for label_set in pinned.label_sets
                ],
                "recipe_content_hash": pinned.recipe_version.content_hash,
                "checkpoint_manifest_digest": package.manifest_digest,
                "evaluation_plan": pinned.run.evaluation_plan,
            },
        },
        metrics=output.validation_metrics,
        runtime_digest=_canonical_hash(runtime_snapshot),
        code_digest=_plugin_code_digest(plugin, pinned),
        seed=pinned.run.seed,
        checkpoint_package_id=package.id,
    )


def _ensure_model_version(
    db: Session,
    pinned: _PinnedRun,
    payload: workflow_schemas.ModelVersionCreate,
) -> workflow_models.ModelVersion:
    db.query(workflow_models.RegisteredModel).filter(
        workflow_models.RegisteredModel.id == pinned.registered_model.id
    ).with_for_update().one()
    content = payload.model_dump(exclude={"project_id", "registered_model_id"})
    content_hash = _canonical_hash(content)
    model_version = (
        db.query(workflow_models.ModelVersion)
        .filter(
            workflow_models.ModelVersion.registered_model_id == pinned.registered_model.id,
            workflow_models.ModelVersion.content_hash == content_hash,
        )
        .one_or_none()
    )
    if model_version is None:
        latest = (
            db.query(func.max(workflow_models.ModelVersion.version_number))
            .filter(
                workflow_models.ModelVersion.registered_model_id
                == pinned.registered_model.id
            )
            .scalar()
        )
        model_version = workflow_models.ModelVersion(
            **payload.model_dump(),
            version_number=int(latest or 0) + 1,
            content_hash=content_hash,
            created_by_user_id=pinned.run.created_by_user_id,
        )
        db.add(model_version)
        db.flush()
    return model_version


def _finish_training_run(
    db: Session,
    pinned: _PinnedRun,
    model_version: workflow_models.ModelVersion,
    package: ArtifactPackage,
    runtime_snapshot: dict,
    evaluation: workflow_models.ModelEvaluation | None,
) -> workflow_models.TrainingRun:
    run = (
        db.query(workflow_models.TrainingRun)
        .filter(workflow_models.TrainingRun.id == pinned.run.id)
        .with_for_update()
        .one()
    )
    if run.status == "succeeded":
        db.rollback()
        return db.get(workflow_models.TrainingRun, run.id)
    if run.status != "running":
        raise ConflictError(f"Cannot finish training run from status '{run.status}'")
    run.status = "succeeded"
    run.output_model_version_id = model_version.id
    run.runtime_snapshot = {
        **runtime_snapshot,
        "protected_test_evaluation": (
            {
                "evaluation_id": evaluation.id,
                "status": evaluation.status,
                "artifact_package_id": evaluation.artifact_package_id,
                "content_hash": evaluation.content_hash,
            }
            if evaluation is not None
            else {
                "status": "not_requested",
                "artifact_package_id": None,
            }
        ),
    }
    run.storage_snapshot = {
        **run.storage_snapshot,
        "artifact_package_id": package.id,
        "manifest_digest": package.manifest_digest,
        "logical_size_bytes": package.logical_size_bytes,
        "file_count": package.file_count,
    }
    run.failure_code = None
    run.failure_reason = None
    run.completed_at = datetime.now(UTC)
    artifact_quota.complete_owner_artifact_reservation(
        db,
        owner_type="training_run",
        owner_id=run.id,
    )
    db.commit()
    return db.get(workflow_models.TrainingRun, run.id)


def _mark_failed(
    db: Session,
    training_run_id: int,
    exc: Exception,
) -> workflow_models.TrainingRun:
    db.rollback()
    run = db.get(workflow_models.TrainingRun, training_run_id)
    if run is None or run.status != "running":
        return run
    return workflow_service.transition_training_run(
        db,
        run.project_id,
        run.id,
        workflow_schemas.TrainingRunTransition(
            status="failed",
            failure_code=type(exc).__name__[:80],
            failure_reason=str(exc)[:4000] or "Training execution failed",
        ),
    )


def execute_training_run(
    db: Session,
    storage: ObjectStorage,
    *,
    training_run_id: int,
    trainer_registry: TrainerPluginRegistry | None = None,
    evaluator_registry: EvaluatorPluginRegistry | None = None,
    recipe_catalog: TrainingRecipeRegistry = training_recipes,
    artifact_publisher: ArtifactPublisher = artifact_service.publish_artifact_package,
) -> workflow_models.TrainingRun:
    """Execute one immutable training run; repeated deliveries are harmless no-ops."""

    run, claimed = _claim_training_run(db, training_run_id)
    if not claimed:
        return run
    try:
        pinned = _load_pinned_run(db, run)
        _validate_environment_and_storage_binding(pinned, storage)
        descriptor, normalized_config = _resolve_recipe(pinned, recipe_catalog)
        if trainer_registry is None:
            # Worker-only lazy import: this registers plugin classes without loading
            # sklearn, torch, transformers, PEFT, or CUDA libraries.
            from al_medlit.training.trainers import trainer_plugins

            trainer_registry = trainer_plugins
        plugin = trainer_registry.require_recipe(
            pinned.recipe_version.trainer_plugin_key,
            pinned.recipe.key,
        )
        preflight = plugin.preflight(pinned.recipe.key)
        _validate_preflight(preflight, pinned)

        with tempfile.TemporaryDirectory(prefix=f"al-medlit-run-{run.id}-") as temporary:
            root = Path(temporary)
            base_model = _stage_base_model(
                db,
                storage,
                pinned,
                normalized_config,
                root / "base-model",
            )
            training_input = _training_input(
                db,
                pinned,
                normalized_config,
                base_model=base_model,
            )
            plan = plugin.plan(
                recipe_key=pinned.recipe.key,
                config=normalized_config,
                training_input=training_input,
                seed=run.seed,
            )
            _validate_plan(
                plan,
                pinned,
                descriptor,
                training_input,
                normalized_config,
            )
            output_root = root / "output"
            output = plugin.train(
                recipe_key=pinned.recipe.key,
                config=plan.normalized_config,
                training_input=training_input,
                destination=output_root,
                seed=run.seed,
            )
            evaluation_requested = "test" in _evaluation_plan_splits(pinned)
            evaluator = None
            if evaluation_requested:
                if evaluator_registry is None:
                    from al_medlit.training.evaluators import evaluator_plugins

                    evaluator_registry = evaluator_plugins
                evaluator = evaluator_registry.find_for_recipe(pinned.recipe.key)
            files = _safe_output_files(output, output_root)
            references, upstream_ids = _package_lineage(db, pinned, base_model)
            package_kind, package_format = _package_contract(descriptor)
            runtime_snapshot = {
                **run.runtime_snapshot,
                "trainer": {
                    "key": plugin.key,
                    "version": pinned.recipe_version.trainer_plugin_version,
                },
                "preflight": preflight.model_dump(mode="json"),
                "plan_sha256": _canonical_hash(plan.manifest),
            }
            retention = pinned.storage_policy.retention_class
            if retention not in _KNOWN_RETENTION_CLASSES:
                retention = "indefinite"
            evaluation_result = None
            if evaluation_requested:
                candidate_package = artifact_publisher(
                    db,
                    storage,
                    project_id=run.project_id,
                    data=_checkpoint_package_data(
                        pinned,
                        descriptor,
                        output,
                        runtime_snapshot,
                        base_model,
                        package_kind="model_candidate",
                        package_format=package_format,
                        display_name=(
                            f"{pinned.registered_model.name} run {run.id} candidate"
                        ),
                        deployable=False,
                        references=references,
                        retention_class="candidate",
                        protected_test_gate={
                            "required": True,
                            "status": (
                                "pending" if evaluator is not None else "unsupported"
                            ),
                            "evaluator_key": (
                                evaluator.key if evaluator is not None else None
                            ),
                            "evaluator_version": (
                                evaluator.evaluator_version
                                if evaluator is not None
                                else None
                            ),
                        },
                    ),
                    files=files,
                    actor_user_id=run.created_by_user_id,
                    upstream_lineage_artifact_ids=upstream_ids,
                    reservation_id=run.artifact_reservation_id,
                )
                db.commit()
                evaluation_result = _run_protected_test_evaluation(
                    db,
                    storage,
                    pinned,
                    candidate_package,
                    output_root,
                    plan.normalized_config,
                    training_input,
                    descriptor,
                    evaluator,
                    artifact_publisher,
                )
                if evaluation_result.status == "succeeded":
                    evaluation_package = evaluation_result.artifact_package
                    if evaluation_package is None:
                        raise ValidationError(
                            "Successful evaluation is missing its artifact package"
                        )
                    promoted_references = [
                        *references,
                        ArtifactPackageReferenceCreate(
                            target_package_id=candidate_package.id,
                            relationship_type="promoted_from_candidate",
                        ),
                        ArtifactPackageReferenceCreate(
                            target_package_id=evaluation_package.id,
                            relationship_type="validated_by_evaluation",
                        ),
                    ]
                    package = artifact_publisher(
                        db,
                        storage,
                        project_id=run.project_id,
                        data=_checkpoint_package_data(
                            pinned,
                            descriptor,
                            output,
                            runtime_snapshot,
                            base_model,
                            package_kind=package_kind,
                            package_format=package_format,
                            display_name=(
                                f"{pinned.registered_model.name} run {run.id}"
                            ),
                            deployable=True,
                            references=promoted_references,
                            retention_class=retention,
                            protected_test_gate={
                                "required": True,
                                "status": "succeeded",
                                "evaluator_key": evaluator.key,
                                "evaluator_version": evaluator.evaluator_version,
                                "candidate_package_id": candidate_package.id,
                                "evaluation_package_id": evaluation_package.id,
                            },
                        ),
                        files=files,
                        actor_user_id=run.created_by_user_id,
                        upstream_lineage_artifact_ids=sorted(
                            {
                                *upstream_ids,
                                candidate_package.lineage_artifact_id,
                                evaluation_package.lineage_artifact_id,
                            }
                        ),
                    )
                else:
                    package = candidate_package
            else:
                package = artifact_publisher(
                    db,
                    storage,
                    project_id=run.project_id,
                    data=_checkpoint_package_data(
                        pinned,
                        descriptor,
                        output,
                        runtime_snapshot,
                        base_model,
                        package_kind=package_kind,
                        package_format=package_format,
                        display_name=f"{pinned.registered_model.name} run {run.id}",
                        deployable=True,
                        references=references,
                        retention_class=retention,
                        protected_test_gate={
                            "required": False,
                            "status": "not_requested",
                            "evaluator_key": None,
                            "evaluator_version": None,
                        },
                    ),
                    files=files,
                    actor_user_id=run.created_by_user_id,
                    upstream_lineage_artifact_ids=upstream_ids,
                    reservation_id=run.artifact_reservation_id,
                )
            payload = _model_version_payload(
                pinned,
                descriptor,
                plugin,
                package,
                output,
                plan.normalized_config,
                runtime_snapshot,
                base_model,
            )
            model_version = _ensure_model_version(
                db,
                pinned,
                payload,
            )
            evaluation = None
            if evaluation_result is not None:
                evaluation = _record_protected_evaluation_result(
                    db,
                    pinned,
                    model_version,
                    evaluation_result,
                )
            return _finish_training_run(
                db,
                pinned,
                model_version,
                package,
                runtime_snapshot,
                evaluation,
            )
    except Exception as exc:
        _mark_failed(db, training_run_id, exc)
        raise
