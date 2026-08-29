"""Durable server-side scoring for registered TF-IDF model feedback runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from al_medlit.core.config import settings
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.model_artifacts.models import ArtifactPackage
from al_medlit.training.artifact_loading import (
    load_safe_skops_model,
    stage_artifact_package,
    validate_executable_package,
)
from al_medlit.workflow import models

from .common import _canonical_hash, _scoped, _validate_task_output
from .feedback_sets import persist_feedback_set_version

SCORER_KEY = "sklearn_tfidf_predict_proba"
SCORER_VERSION = "1"
SCORING_BATCH_SIZE = 512
EAGER_SCORING_ITEM_LIMIT = 5000
SCORING_CONFIGURATION_KEY = "server_scoring"


@dataclass(frozen=True, slots=True)
class FeedbackScoringDispatch:
    run: models.FeedbackRun
    should_enqueue: bool


@dataclass(frozen=True, slots=True)
class _ScoringContext:
    run: models.FeedbackRun
    dataset: models.DatasetVersion
    task: models.TaskVersion
    model_version: models.ModelVersion
    package: ArtifactPackage
    input_field: str
    target_field: str


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _normalized_configuration(configuration: Mapping[str, Any] | None) -> dict:
    normalized = dict(configuration or {})
    existing = normalized.get(SCORING_CONFIGURATION_KEY)
    expected = {
        "scorer_key": SCORER_KEY,
        "scorer_version": SCORER_VERSION,
        "uncertainty_method": "least_confidence",
        "candidate_key": "primary",
        "batch_size": SCORING_BATCH_SIZE,
    }
    if existing is not None:
        if not isinstance(existing, Mapping) or any(
            existing.get(key) != value for key, value in expected.items()
        ):
            raise ValidationError(
                f"configuration.{SCORING_CONFIGURATION_KEY} is reserved for server scoring"
            )
        expected = {**dict(existing), **expected}
    normalized[SCORING_CONFIGURATION_KEY] = expected
    return normalized


def _field_mapping(model_version: models.ModelVersion) -> tuple[str, str]:
    fields = (model_version.parameters or {}).get("fields")
    if not isinstance(fields, Mapping):
        raise ValidationError("TF-IDF model parameters require a fields mapping")
    input_field = fields.get("input_field")
    target_field = fields.get("target_field")
    if not isinstance(input_field, str) or not input_field.strip():
        raise ValidationError("TF-IDF model parameters require a non-empty input_field")
    if not isinstance(target_field, str) or not target_field.strip():
        raise ValidationError("TF-IDF model parameters require a non-empty target_field")
    return input_field, target_field


def _validate_output_shape(task: models.TaskVersion, target_field: str) -> None:
    schema = task.output_schema or {}
    if schema.get("type") != "object":
        return
    properties = schema.get("properties")
    required = schema.get("required", [])
    if (
        not isinstance(properties, Mapping)
        or set(properties) != {target_field}
        or not isinstance(required, list)
        or required != [target_field]
    ):
        raise ValidationError(
            "Object classification outputs must expose only the configured target field "
            "as a required prediction value"
        )


def _validate_scoring_context(db: Session, run: models.FeedbackRun) -> _ScoringContext:
    if run.producer_type != "registered_model":
        raise ValidationError("Only registered_model feedback runs can be materialized")
    if run.model_version_id is None:
        raise ValidationError("Registered-model feedback requires model_version_id")
    dataset = _scoped(
        db,
        models.DatasetVersion,
        run.dataset_version_id,
        run.project_id,
        "Dataset version",
    )
    if dataset.item_count <= 0:
        raise ValidationError("Feedback scoring requires a non-empty dataset version")
    task = _scoped(
        db,
        models.TaskVersion,
        run.task_version_id,
        run.project_id,
        "Task version",
    )
    if task.task_kind != "classification":
        raise ValidationError(
            "Server-side TF-IDF scoring currently supports single-label classification only"
        )
    model_version = _scoped(
        db,
        models.ModelVersion,
        run.model_version_id,
        run.project_id,
        "Model version",
    )
    if model_version.task_version_id != task.id:
        raise ValidationError("Feedback model must use the feedback task version")
    if model_version.recipe_key != "tfidf_logistic_regression":
        raise ValidationError(
            "Server-side feedback scoring supports recipe tfidf_logistic_regression only"
        )
    if model_version.framework != "scikit-learn" or model_version.family != "conventional_ml":
        raise ValidationError("Feedback scoring requires a conventional scikit-learn model")
    if model_version.checkpoint_package_id is None:
        raise ValidationError("Feedback model version has no checkpoint package")
    package = _scoped(
        db,
        ArtifactPackage,
        model_version.checkpoint_package_id,
        run.project_id,
        "Model checkpoint package",
    )
    validate_executable_package(package, package_format="skops")
    if package.package_kind not in {"trained_model", "model_candidate"}:
        raise ValidationError("Feedback scoring requires a trained-model checkpoint package")
    if package.model_family != model_version.family:
        raise ValidationError("Model checkpoint family does not match its model version")
    task_contract = package.task_contract or {}
    if task_contract.get("task_version_id") != task.id:
        raise ValidationError("Model checkpoint task contract does not match the feedback task")
    if task_contract.get("task_content_hash") != task.content_hash:
        raise ValidationError("Model checkpoint task hash does not match the feedback task")
    lineage = (model_version.parameters or {}).get("_lineage", {})
    if isinstance(lineage, Mapping):
        digest = lineage.get("checkpoint_manifest_digest")
        if digest not in (None, package.manifest_digest):
            raise ValidationError("Model-version checkpoint lineage does not match its package")
        if package.metadata_.get("recipe_content_hash") != lineage.get(
            "recipe_content_hash"
        ):
            raise ValidationError(
                "Model checkpoint recipe lineage does not match its model version"
            )
    training_run_id = package.metadata_.get("training_run_id")
    if not isinstance(training_run_id, int):
        raise ValidationError("Feedback model version has no training-run lineage")
    training_run = _scoped(
        db,
        models.TrainingRun,
        training_run_id,
        run.project_id,
        "Training run",
    )
    recipe_version = _scoped(
        db,
        models.TrainingRecipeVersion,
        training_run.recipe_version_id,
        run.project_id,
        "Training recipe version",
    )
    recipe = _scoped(
        db,
        models.TrainingRecipe,
        recipe_version.training_recipe_id,
        run.project_id,
        "Training recipe",
    )
    if (
        recipe.key != model_version.recipe_key
        or training_run.output_model_version_id != model_version.id
        or package.metadata_.get("recipe_version_id") != recipe_version.id
    ):
        raise ValidationError("Model checkpoint recipe or training-run lineage does not match")
    input_field, target_field = _field_mapping(model_version)
    _validate_output_shape(task, target_field)
    if run.cycle_id is not None:
        cycle = _scoped(
            db,
            models.LearningCycle,
            run.cycle_id,
            run.project_id,
            "Learning cycle",
        )
        if (
            cycle.source_dataset_version_id != dataset.id
            or cycle.task_version_id != task.id
            or cycle.baseline_model_version_id != model_version.id
        ):
            raise ValidationError(
                "Cycle-linked scoring must use the cycle dataset, task, and baseline model"
            )
    return _ScoringContext(
        run=run,
        dataset=dataset,
        task=task,
        model_version=model_version,
        package=package,
        input_field=input_field,
        target_field=target_field,
    )


def get_feedback_run(
    db: Session,
    *,
    project_id: int,
    feedback_run_id: int,
) -> models.FeedbackRun:
    return _scoped(
        db,
        models.FeedbackRun,
        feedback_run_id,
        project_id,
        "Feedback run",
    )


def request_feedback_run_materialization(
    db: Session,
    *,
    project_id: int,
    feedback_run_id: int,
    actor: Any,
) -> FeedbackScoringDispatch:
    del actor  # Authorization is enforced by the route; creator provenance is already pinned.
    candidate = get_feedback_run(
        db,
        project_id=project_id,
        feedback_run_id=feedback_run_id,
    )
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == candidate.id)
        .populate_existing()
        .with_for_update()
        .one()
    )
    _validate_scoring_context(db, run)
    if run.status == "completed":
        if run.output_feedback_set_version_id is None:
            db.rollback()
            raise ConflictError("Completed feedback scoring run has no output feedback set")
        output = db.get(models.FeedbackSetVersion, run.output_feedback_set_version_id)
        if output is None or output.feedback_run_id != run.id:
            db.rollback()
            raise ConflictError("Feedback scoring output does not belong to its run")
        db.rollback()
        return FeedbackScoringDispatch(
            run=db.get(models.FeedbackRun, run.id),
            should_enqueue=False,
        )
    if run.status == "running":
        db.rollback()
        return FeedbackScoringDispatch(
            run=db.get(models.FeedbackRun, run.id),
            should_enqueue=False,
        )
    if run.status not in {"planned", "failed", "queued"}:
        db.rollback()
        raise ConflictError(f"Cannot materialize feedback run from status '{run.status}'")
    run.configuration = _normalized_configuration(run.configuration)
    if run.status in {"planned", "failed"}:
        run.output_feedback_set_version_id = None
        run.failure_code = None
        run.failure_reason = None
        run.started_at = None
        run.heartbeat_at = None
        run.completed_at = None
    run.status = "queued"
    run.updated_at = datetime.now(UTC)
    db.commit()
    return FeedbackScoringDispatch(
        run=db.get(models.FeedbackRun, run.id),
        should_enqueue=True,
    )


def _claim_feedback_scoring_run(
    db: Session,
    feedback_run_id: int,
) -> tuple[models.FeedbackRun, bool]:
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == feedback_run_id)
        .with_for_update()
        .one_or_none()
    )
    if run is None:
        raise NotFoundError("Feedback run not found")
    if run.status in {"completed", "failed", "running"}:
        db.rollback()
        return db.get(models.FeedbackRun, feedback_run_id), False
    if run.status != "queued":
        db.rollback()
        raise ConflictError(f"Cannot execute feedback run from status '{run.status}'")
    claimed_at = datetime.now(UTC)
    run.status = "running"
    run.started_at = claimed_at
    run.heartbeat_at = claimed_at
    run.completed_at = None
    run.failure_code = None
    run.failure_reason = None
    db.commit()
    return db.get(models.FeedbackRun, feedback_run_id), True


def heartbeat_feedback_scoring_run(
    db: Session,
    feedback_run_id: int,
    *,
    now: datetime | None = None,
) -> bool:
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == feedback_run_id)
        .with_for_update()
        .one_or_none()
    )
    if run is None:
        raise NotFoundError("Feedback run not found")
    if run.status != "running":
        db.rollback()
        return False
    run.heartbeat_at = now or datetime.now(UTC)
    db.commit()
    return True


def reconcile_feedback_scoring_runs(
    db: Session,
    *,
    now: datetime | None = None,
    running_stale_after_seconds: int = 900,
    queued_stale_after_seconds: int = 300,
    limit: int = 200,
) -> dict[str, list[int]]:
    if running_stale_after_seconds < 60 or queued_stale_after_seconds < 60:
        raise ValidationError(
            "Feedback scoring reconciliation timeouts must be at least 60 seconds"
        )
    effective_now = now or datetime.now(UTC)
    running_cutoff = effective_now - timedelta(seconds=running_stale_after_seconds)
    queued_cutoff = effective_now - timedelta(seconds=queued_stale_after_seconds)
    bounded_limit = min(max(limit, 1), 1000)
    failed_ids: list[int] = []
    redispatch_ids: list[int] = []
    candidates = (
        db.query(models.FeedbackRun.id)
        .filter(models.FeedbackRun.status.in_(("running", "queued")))
        .order_by(models.FeedbackRun.id)
        .limit(bounded_limit)
        .all()
    )
    for (run_id,) in candidates:
        run = (
            db.query(models.FeedbackRun)
            .filter(models.FeedbackRun.id == run_id)
            .populate_existing()
            .with_for_update()
            .one()
        )
        if run.status == "running":
            heartbeat = run.heartbeat_at or run.started_at
            if heartbeat is not None and _as_utc(heartbeat) > running_cutoff:
                db.rollback()
                continue
            run.status = "failed"
            run.failure_code = "scoring_worker_stalled"
            run.failure_reason = "Feedback scoring worker heartbeat expired"
            run.completed_at = effective_now
            db.commit()
            failed_ids.append(run.id)
        elif run.status == "queued" and _as_utc(run.updated_at) <= queued_cutoff:
            run.updated_at = effective_now
            db.commit()
            redispatch_ids.append(run.id)
        else:
            db.rollback()
    return {"failed_ids": failed_ids, "redispatch_ids": redispatch_ids}


def _python_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except ValueError:
            pass
    converter = getattr(value, "tolist", None)
    return converter() if callable(converter) else value


def _model_classes(model: Any) -> list[Any]:
    raw = _python_value(getattr(model, "classes_", None))
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        raise ValidationError("TF-IDF model must expose at least two ordered classes_")
    classes = [_python_value(value) for value in raw]
    if len({_canonical_hash(value) for value in classes}) != len(classes):
        raise ValidationError("TF-IDF model classes_ must be unique JSON values")
    return classes


def _prediction_output(
    task: models.TaskVersion,
    target_field: str,
    predicted_label: Any,
) -> Any:
    output: Any = predicted_label
    if (task.output_schema or {}).get("type") == "object":
        output = {target_field: predicted_label}
    _validate_task_output(task, output, label="Model feedback prediction")
    return output


def _probability_rows(model: Any, texts: Sequence[str], class_count: int) -> list[list[float]]:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        raise ValidationError("TF-IDF model does not expose predict_proba")
    raw = _python_value(predict_proba(list(texts)))
    if not isinstance(raw, (list, tuple)) or len(raw) != len(texts):
        raise ValidationError("TF-IDF model returned the wrong probability row count")
    rows: list[list[float]] = []
    for index, raw_row in enumerate(raw):
        row_value = _python_value(raw_row)
        if not isinstance(row_value, (list, tuple)) or len(row_value) != class_count:
            raise ValidationError(
                f"TF-IDF probability row {index} does not match the model class count"
            )
        row: list[float] = []
        for probability in row_value:
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValidationError("TF-IDF probabilities must be numeric")
            number = float(probability)
            if not math.isfinite(number) or number < 0.0 or number > 1.0:
                raise ValidationError("TF-IDF probabilities must be finite values in [0, 1]")
            row.append(number)
        if not math.isclose(sum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValidationError("TF-IDF class probabilities must sum to one")
        rows.append(row)
    return rows


def _candidate_payload(
    context: _ScoringContext,
    item: models.DatasetItem,
    classes: Sequence[Any],
    probabilities: Sequence[float],
) -> dict:
    best_index = max(range(len(probabilities)), key=lambda index: probabilities[index])
    predicted_label = classes[best_index]
    confidence = float(probabilities[best_index])
    explanation: dict[str, Any] = {
        "uncertainty": 1.0 - confidence,
        "signal_basis": "least_confidence",
        "uncertainty_definition": (
            "1 - max(class probabilities); rank-equivalent to binary margin "
            "uncertainty, but not numerically equal"
        ),
        "predicted_probability": confidence,
        "scorer": {"key": SCORER_KEY, "version": SCORER_VERSION},
        "model_version_id": context.model_version.id,
        "model_version_content_hash": context.model_version.content_hash,
        "checkpoint_package_id": context.package.id,
        "checkpoint_manifest_digest": context.package.manifest_digest,
        "input_field": context.input_field,
        "class_count": len(classes),
    }
    if len(classes) <= 20:
        explanation["class_probabilities"] = [
            {"label": label, "probability": float(probability)}
            for label, probability in zip(classes, probabilities, strict=True)
        ]
    payload = {
        "dataset_item_id": item.id,
        "candidate_key": "primary",
        "output": _prediction_output(
            context.task,
            context.target_field,
            predicted_label,
        ),
        "score": confidence,
        "explanation": explanation,
    }
    payload["content_hash"] = _canonical_hash(payload)
    return payload


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _feedback_set_hash(
    spool_path: Path,
    *,
    feedback_run_id: int,
    output_schema: dict,
) -> str:
    digest = hashlib.sha256()
    digest.update(b'{"candidates":[')
    first = True
    with spool_path.open("rb") as source:
        for raw_line in source:
            line = raw_line.rstrip(b"\n")
            if not line:
                continue
            if not first:
                digest.update(b",")
            digest.update(line)
            first = False
    digest.update(b'],"feedback_run_id":')
    digest.update(_canonical_json(feedback_run_id))
    digest.update(b',"output_schema":')
    digest.update(_canonical_json(output_schema))
    digest.update(b"}")
    return digest.hexdigest()


def _score_to_spool(
    db: Session,
    context: _ScoringContext,
    model: Any,
    spool_path: Path,
) -> int:
    classes = _model_classes(model)
    for class_label in classes:
        _prediction_output(context.task, context.target_field, class_label)
    last_stable_key: str | None = None
    last_id = 0
    candidate_count = 0
    with spool_path.open("xb") as sink:
        while True:
            query = db.query(models.DatasetItem).filter(
                models.DatasetItem.project_id == context.run.project_id,
                models.DatasetItem.dataset_version_id == context.dataset.id,
            )
            if last_stable_key is not None:
                query = query.filter(
                    or_(
                        models.DatasetItem.stable_key > last_stable_key,
                        and_(
                            models.DatasetItem.stable_key == last_stable_key,
                            models.DatasetItem.id > last_id,
                        ),
                    )
                )
            items = (
                query.order_by(models.DatasetItem.stable_key, models.DatasetItem.id)
                .limit(SCORING_BATCH_SIZE)
                .all()
            )
            if not items:
                break
            texts: list[str] = []
            for item in items:
                text = item.payload.get(context.input_field)
                if not isinstance(text, str) or not text.strip():
                    raise ValidationError(
                        f"Dataset item '{item.stable_key}' field "
                        f"'{context.input_field}' must be non-empty text"
                    )
                texts.append(text)
            rows = _probability_rows(model, texts, len(classes))
            for item, probabilities in zip(items, rows, strict=True):
                sink.write(
                    _canonical_json(
                        _candidate_payload(context, item, classes, probabilities)
                    )
                )
                sink.write(b"\n")
                candidate_count += 1
            last_stable_key = items[-1].stable_key
            last_id = items[-1].id
    if candidate_count != context.dataset.item_count:
        raise ConflictError(
            "Dataset item count changed or does not match the immutable dataset version"
        )
    return candidate_count


def _record_runtime_versions(db: Session, feedback_run_id: int) -> None:
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == feedback_run_id)
        .with_for_update()
        .one()
    )
    if run.status != "running":
        db.rollback()
        raise ConflictError("Feedback scoring run is no longer running")
    configuration = dict(run.configuration or {})
    scoring = dict(configuration.get(SCORING_CONFIGURATION_KEY, {}))

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "injected-test-runtime"

    scoring["runtime_versions"] = {
        "scikit-learn": package_version("scikit-learn"),
        "skops": package_version("skops"),
    }
    configuration[SCORING_CONFIGURATION_KEY] = scoring
    run.configuration = configuration
    db.commit()


def _finalize_feedback_scoring_run(
    db: Session,
    context: _ScoringContext,
    spool_path: Path,
    candidate_count: int,
) -> models.FeedbackRun:
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == context.run.id)
        .populate_existing()
        .with_for_update()
        .one()
    )
    if run.status == "completed" and run.output_feedback_set_version_id is not None:
        db.rollback()
        return db.get(models.FeedbackRun, run.id)
    if run.status != "running":
        db.rollback()
        raise ConflictError(f"Cannot finish feedback scoring from status '{run.status}'")
    if run.output_feedback_set_version_id is not None:
        db.rollback()
        raise ConflictError("Feedback scoring run already has an output feedback set")
    if (
        db.query(models.FeedbackSetVersion.id)
        .filter(models.FeedbackSetVersion.feedback_run_id == run.id)
        .first()
        is not None
    ):
        db.rollback()
        raise ConflictError("Automated feedback scoring run already has a feedback set")
    def candidate_payloads():
        with spool_path.open("r", encoding="ascii") as source:
            for line in source:
                if line.strip():
                    yield json.loads(line)

    feedback_set = persist_feedback_set_version(
        db,
        run=run,
        output_schema=context.task.output_schema,
        candidate_payloads=candidate_payloads(),
        candidate_count=candidate_count,
        content_hash=_feedback_set_hash(
            spool_path,
            feedback_run_id=run.id,
            output_schema=context.task.output_schema,
        ),
        artifact_package_id=None,
        created_by_user_id=run.created_by_user_id,
    )
    completed_at = datetime.now(UTC)
    run.status = "completed"
    run.output_feedback_set_version_id = feedback_set.id
    run.heartbeat_at = completed_at
    run.completed_at = completed_at
    run.failure_code = None
    run.failure_reason = None
    db.commit()
    return db.get(models.FeedbackRun, run.id)


def _mark_feedback_scoring_failed(
    db: Session,
    feedback_run_id: int,
    exc: Exception,
) -> models.FeedbackRun | None:
    db.rollback()
    run = (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.id == feedback_run_id)
        .with_for_update()
        .one_or_none()
    )
    if run is None or run.status != "running":
        db.rollback()
        return run
    run.status = "failed"
    run.failure_code = type(exc).__name__[:80]
    run.failure_reason = (str(exc) or "Feedback scoring failed")[:4000]
    run.completed_at = datetime.now(UTC)
    db.commit()
    return db.get(models.FeedbackRun, feedback_run_id)


def execute_feedback_scoring_run(
    db: Session,
    storage: ObjectStorage,
    *,
    feedback_run_id: int,
    model_loader: Callable[[Path], Any] | None = None,
) -> models.FeedbackRun:
    """Score an immutable dataset; repeated task deliveries are harmless no-ops."""

    run, claimed = _claim_feedback_scoring_run(db, feedback_run_id)
    if not claimed:
        return run
    try:
        context = _validate_scoring_context(db, run)
        if (
            settings.celery_task_always_eager
            and context.dataset.item_count > EAGER_SCORING_ITEM_LIMIT
        ):
            raise ValidationError(
                "Eager feedback scoring is limited to 5,000 items; use `make lab-up` "
                "and `make runtime-up RUNTIME=classical-cpu` for larger datasets"
            )
        with tempfile.TemporaryDirectory(prefix=f"feedback-scoring-{run.id}-") as temp:
            root = Path(temp)
            model_root = stage_artifact_package(
                db,
                storage,
                package=context.package,
                destination=root / "checkpoint",
            )
            model = load_safe_skops_model(
                model_root / "model.skops",
                model_loader=model_loader,
            )
            _record_runtime_versions(db, run.id)
            spool_path = root / "candidates.jsonl"
            candidate_count = _score_to_spool(
                db,
                context,
                model,
                spool_path,
            )
            return _finalize_feedback_scoring_run(
                db,
                context,
                spool_path,
                candidate_count,
            )
    except Exception as exc:
        _mark_feedback_scoring_failed(db, feedback_run_id, exc)
        raise
