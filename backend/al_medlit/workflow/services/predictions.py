"""Standalone model predictions and explicit, transactional review handoff."""

from datetime import UTC, datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workflow import models
from al_medlit.workflow import prediction_schemas as schemas

from .common import _canonical_hash, _commit, _next_sequence, _scoped
from .feedback_scoring import _validate_scoring_context
from .rounds import _lock_and_validate_active_round_annotators

ORIGIN = "standalone_prediction_v1"


def _lock_project(db: Session, project_id: int) -> None:
    # A write serializes request-key checks on SQLite as well as PostgreSQL.
    if (
        not db.query(Project)
        .filter(Project.id == project_id)
        .update({Project.id: Project.id}, synchronize_session=False)
    ):
        raise NotFoundError("Project not found")


def _metadata(run: models.FeedbackRun) -> dict:
    value = (run.configuration or {}).get(ORIGIN)
    required = ("name", "request_key", "request_hash")
    return (
        value
        if isinstance(value, dict) and all(isinstance(value.get(key), str) for key in required)
        else {}
    )


def get_prediction_run(db: Session, project_id: int, run_id: int) -> models.FeedbackRun:
    run = _scoped(db, models.FeedbackRun, run_id, project_id, "Prediction run")
    if not _metadata(run) or run.cycle_id is not None or run.producer_type != "registered_model":
        raise NotFoundError("Prediction run not found")
    return run


def list_prediction_runs(db: Session, project_id: int) -> list[models.FeedbackRun]:
    return [
        run
        for run in db.query(models.FeedbackRun)
        .filter(
            models.FeedbackRun.project_id == project_id,
            models.FeedbackRun.cycle_id.is_(None),
            models.FeedbackRun.producer_type == "registered_model",
        )
        .order_by(models.FeedbackRun.id.desc())
        .all()
        if _metadata(run)
    ]


def prediction_read(db: Session, run: models.FeedbackRun) -> dict:
    output = (
        db.get(models.FeedbackSetVersion, run.output_feedback_set_version_id)
        if run.output_feedback_set_version_id
        else None
    )
    return {
        "id": run.id,
        "project_id": run.project_id,
        "name": _metadata(run)["name"],
        "dataset_version_id": run.dataset_version_id,
        "task_version_id": run.task_version_id,
        "model_version_id": run.model_version_id,
        "status": run.status,
        "result_count": output.candidate_count if output else 0,
        "failure_reason": run.failure_reason,
        "completed_at": run.completed_at,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def create_prediction_run(
    db: Session,
    project_id: int,
    data: schemas.PredictionRunCreate,
    actor: User,
) -> models.FeedbackRun:
    _lock_project(db, project_id)
    digest = _canonical_hash(data.model_dump(exclude={"request_key"}))
    for run in list_prediction_runs(db, project_id):
        marker = _metadata(run)
        if marker.get("request_key") == data.request_key:
            if marker.get("request_hash") != digest or run.created_by_user_id != actor.id:
                raise ConflictError("This request key was already used; submit a new request key")
            return run
    model = _scoped(db, models.ModelVersion, data.model_version_id, project_id, "Model version")
    run = models.FeedbackRun(
        project_id=project_id,
        dataset_version_id=data.dataset_version_id,
        task_version_id=model.task_version_id,
        producer_type="registered_model",
        model_version_id=model.id,
        status="planned",
        created_by_user_id=actor.id,
        configuration={
            ORIGIN: {"name": data.name, "request_key": data.request_key, "request_hash": digest}
        },
        data_egress_policy={},
    )
    _validate_scoring_context(db, run)
    db.add(run)
    _commit(db, "Could not create the prediction run")
    db.refresh(run)
    return run


def _output(db: Session, run: models.FeedbackRun) -> models.FeedbackSetVersion:
    if run.status != "completed" or run.output_feedback_set_version_id is None:
        raise ConflictError("Predictions are not ready; wait for the run to complete")
    output = _scoped(
        db,
        models.FeedbackSetVersion,
        run.output_feedback_set_version_id,
        run.project_id,
        "Prediction output",
    )
    if (
        output.feedback_run_id != run.id
        or output.dataset_version_id != run.dataset_version_id
        or output.task_version_id != run.task_version_id
    ):
        raise ConflictError("Prediction output does not match the pinned run")
    return output


def _submitted_items(db: Session, run: models.FeedbackRun) -> set[int]:
    submissions = (
        db.query(models.RoundSubmission)
        .join(
            models.AnnotationRound,
            models.RoundSubmission.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(
            models.AnnotationRound.project_id == run.project_id,
            models.AnnotationRound.task_version_id == run.task_version_id,
        )
        .all()
    )
    decision_ids = {
        decision_id for submission in submissions for decision_id in submission.decision_ids
    }
    if not decision_ids:
        return set()
    return {
        item_id
        for item_id, decision_id in db.query(
            models.RoundItem.dataset_item_id,
            models.RoundAnnotationDecision.id,
        )
        .join(
            models.RoundAnnotationDecision,
            models.RoundAnnotationDecision.round_item_id == models.RoundItem.id,
        )
        .filter(
            models.RoundItem.project_id == run.project_id,
        )
        .all()
        if decision_id in decision_ids
    }


def _protected_items(
    db: Session, items: list[models.DatasetItem], task_version_id: int
) -> set[int]:
    from .preparation import (
        _identity_tokens,
        _prepared_provenance_items,
        _protected_source_context,
    )

    if not items:
        return set()
    project_id = items[0].project_id
    protected_ids, protected_tokens = _protected_source_context(db, project_id, task_version_id)
    versions = {
        version.id: version
        for version in db.query(models.DatasetVersion)
        .filter(
            models.DatasetVersion.project_id == project_id,
        )
        .all()
    }
    ancestry = {
        version_id: _prepared_provenance_items(db, versions[version_id])
        for version_id in {item.dataset_version_id for item in items}
    }
    origins: dict[int, set[int]] = {}
    for item in items:
        metadata = ancestry[item.dataset_version_id].get(item.stable_key)
        source_origins = metadata.get("origins", []) if isinstance(metadata, dict) else []
        source_origins = source_origins if isinstance(source_origins, list) else []
        origins[item.id] = {
            origin["dataset_item_id"]
            for origin in source_origins
            if isinstance(origin, dict) and isinstance(origin.get("dataset_item_id"), int)
        }
    origin_ids = set().union(*origins.values())
    source_items = (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.project_id == project_id, models.DatasetItem.id.in_(origin_ids))
        .all()
        if origin_ids
        else []
    )
    candidates = {item.id: item for item in [*items, *source_items]}
    version_ids = {item.dataset_version_id for item in candidates.values()}
    local_keys: dict[int, set[str]] = {version_id: set() for version_id in version_ids}
    split_owners = db.query(models.TrainingDatasetVersion.id).filter(
        models.TrainingDatasetVersion.split_map_id == models.SplitMap.id
    )
    task_owners = split_owners.filter(
        models.TrainingDatasetVersion.task_version_id == task_version_id
    )
    for split in (
        db.query(models.SplitMap)
        .filter(
            models.SplitMap.project_id == project_id,
            models.SplitMap.dataset_version_id.in_(version_ids),
            or_(~split_owners.exists(), task_owners.exists()),
        )
        .all()
    ):
        local_keys[split.dataset_version_id].update(
            key for key, value in split.assignments.items() if value in set(split.protected_splits)
        )
    local_groups: dict[int, set[str]] = {version_id: set() for version_id in version_ids}
    for version_id, keys in local_keys.items():
        if keys:
            local_groups[version_id].update(
                group
                for (group,) in db.query(models.DatasetItem.group_key)
                .filter(
                    models.DatasetItem.dataset_version_id == version_id,
                    models.DatasetItem.stable_key.in_(keys),
                    models.DatasetItem.group_key.is_not(None),
                )
                .all()
            )
    protected = {
        item.id
        for item in candidates.values()
        if (
            item.id in protected_ids
            or _identity_tokens(versions[item.dataset_version_id].dataset_id, item)
            & protected_tokens
            or item.stable_key in local_keys[item.dataset_version_id]
            or item.group_key in local_groups[item.dataset_version_id]
        )
    }
    return {item.id for item in items if item.id in protected or origins[item.id] & protected}


def result_rows(
    db: Session, run: models.FeedbackRun, *, offset: int = 0, limit: int | None = None
) -> list[dict]:
    output = _output(db, run)
    query = (
        db.query(models.FeedbackCandidate, models.DatasetItem)
        .join(
            models.DatasetItem,
            models.FeedbackCandidate.dataset_item_id == models.DatasetItem.id,
        )
        .filter(
            models.FeedbackCandidate.project_id == run.project_id,
            models.FeedbackCandidate.feedback_set_version_id == output.id,
            models.FeedbackCandidate.candidate_key == "primary",
            models.DatasetItem.project_id == run.project_id,
            models.DatasetItem.dataset_version_id == run.dataset_version_id,
        )
        .order_by(models.DatasetItem.stable_key, models.DatasetItem.id)
        .offset(offset)
    )
    if limit is not None:
        query = query.limit(limit)
    records = query.all()
    submitted = _submitted_items(db, run)
    protected = _protected_items(db, [item for _, item in records], run.task_version_id)
    model = db.get(models.ModelVersion, run.model_version_id)
    input_field = ((model.parameters or {}).get("fields") or {}).get("input_field", "text")
    return [
        {
            "dataset_item_id": item.id,
            "stable_key": item.stable_key,
            "title": str(item.payload.get("title") or item.stable_key),
            "text": str(item.payload.get(input_field) or ""),
            "prediction": candidate.output,
            "confidence": candidate.score,
            "uncertainty": (candidate.explanation or {}).get("uncertainty"),
            "already_submitted": item.id in submitted,
            "protected": item.id in protected,
        }
        for candidate, item in records
    ]


def create_review_round(
    db: Session, project_id: int, run_id: int, data: schemas.PredictionReviewCreate, actor: User
) -> dict:
    _lock_and_validate_active_round_annotators(db, [actor.id])
    _lock_project(db, project_id)
    run = get_prediction_run(db, project_id, run_id)
    db.refresh(run)
    output = _output(db, run)
    marker = dict(_metadata(run))
    previous = dict(marker.get("review_requests", {}))
    request_hash = _canonical_hash(
        {
            **data.model_dump(exclude={"request_key"}),
            "dataset_item_ids": sorted(data.dataset_item_ids),
            "actor_id": actor.id,
        }
    )
    if data.request_key in previous:
        stored = previous[data.request_key]
        if stored["request_hash"] != request_hash:
            raise ConflictError("Review request key was already used for another selection")
        return stored["result"]
    items = (
        db.query(models.DatasetItem)
        .filter(
            models.DatasetItem.id.in_(data.dataset_item_ids),
            models.DatasetItem.project_id == project_id,
            models.DatasetItem.dataset_version_id == run.dataset_version_id,
        )
        .all()
    )
    if len(items) != len(data.dataset_item_ids):
        raise ValidationError("Every selected item must belong to this prediction dataset version")
    candidate_ids = {
        item_id
        for (item_id,) in db.query(models.FeedbackCandidate.dataset_item_id)
        .filter(
            models.FeedbackCandidate.feedback_set_version_id == output.id,
            models.FeedbackCandidate.project_id == project_id,
            models.FeedbackCandidate.candidate_key == "primary",
            models.FeedbackCandidate.dataset_item_id.in_(data.dataset_item_ids),
        )
        .all()
    }
    if candidate_ids != set(data.dataset_item_ids):
        raise ValidationError("Each selected item must have a prediction in the pinned output")
    protected = _protected_items(db, items, run.task_version_id)
    submitted = _submitted_items(db, run) if not data.include_submitted else set()
    eligible = [
        item_id
        for item_id in data.dataset_item_ids
        if item_id not in protected and item_id not in submitted
    ]
    if not eligible:
        raise ValidationError(
            "No eligible items remain; protected items and submitted annotations are excluded"
        )
    annotation_round = models.AnnotationRound(
        project_id=project_id,
        name=data.name,
        dataset_version_id=run.dataset_version_id,
        task_version_id=run.task_version_id,
        feedback_set_version_id=output.id,
        assistance_policy="immediate_suggestions",
        reannotation_mode="targeted_subset",
        annotator_user_ids=[actor.id],
        open_to_all_annotators=False,
        sequence=_next_sequence(db, models.AnnotationRound, project_id, cycle_id=None),
        reason=f"Manual review of prediction run {run.id}",
        status="open",
        opened_at=datetime.now(UTC),
        created_by_user_id=actor.id,
    )
    db.add(annotation_round)
    db.flush()
    for rank, item_id in enumerate(eligible, start=1):
        db.add(
            models.RoundItem(
                project_id=project_id,
                annotation_round_id=annotation_round.id,
                dataset_item_id=item_id,
                selection_rank=rank,
                selection_reason={
                    "strategy": "manual_prediction_review",
                    "prediction_run_id": run.id,
                },
                metadata_={"prediction_run_id": run.id, "feedback_set_version_id": output.id},
            )
        )
    result = {
        "round_id": annotation_round.id,
        "item_count": len(eligible),
        "excluded_protected": len(protected),
        "excluded_submitted": len(set(data.dataset_item_ids) & submitted - protected),
    }
    previous[data.request_key] = {"request_hash": request_hash, "result": result}
    marker["review_requests"] = previous
    run.configuration = {**run.configuration, ORIGIN: marker}
    _commit(db, "Could not create the prediction review round")
    return result
