"""Standalone model predictions and explicit, transactional review handoff."""

from datetime import UTC, datetime

from sqlalchemy import Integer, cast, select
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workflow import models
from al_medlit.workflow import prediction_schemas as schemas
from al_medlit.workspace.models import WorkspaceMember

from .common import _canonical_hash, _commit, _next_sequence, _scoped
from .feedback_scoring import _validate_scoring_context
from .holdout_queries import _json_entries
from .holdouts import protected_dataset_item_ids as _protected_items
from .rounds import _lock_and_validate_active_round_annotators
from .selection import _unavailable_selection_item_ids

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
    for run in db.query(models.FeedbackRun).filter(
        models.FeedbackRun.project_id == project_id,
        models.FeedbackRun.configuration[ORIGIN]["request_key"].as_string() == data.request_key,
    ):
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


def _submitted_items(
    db: Session, run: models.FeedbackRun, item_ids: list[int] | None = None,
) -> set[int]:
    entries = _json_entries(db, models.RoundSubmission.decision_ids, array=True)
    submitted_ids = select(cast(entries.c.value, Integer)).select_from(entries).correlate(
        models.RoundSubmission
    )
    query = db.query(models.RoundItem.dataset_item_id).join(
        models.RoundAnnotationDecision,
        models.RoundAnnotationDecision.round_item_id == models.RoundItem.id,
    ).join(
        models.RoundSubmission,
        (models.RoundSubmission.annotation_round_id == models.RoundItem.annotation_round_id)
        & (
            models.RoundSubmission.annotator_user_id
            == models.RoundAnnotationDecision.annotator_user_id
        ),
    ).join(
        models.AnnotationRound,
        models.AnnotationRound.id == models.RoundItem.annotation_round_id,
    ).filter(
        models.AnnotationRound.project_id == run.project_id,
        models.AnnotationRound.task_version_id == run.task_version_id,
        models.RoundItem.project_id == run.project_id,
        models.RoundAnnotationDecision.id.in_(submitted_ids),
    )
    if item_ids is not None:
        query = query.filter(models.RoundItem.dataset_item_id.in_(item_ids))
    return {item_id for (item_id,) in query.distinct()}


def result_rows(
    db: Session, run: models.FeedbackRun, *, offset: int = 0, limit: int | None = None,
    include_protected: bool = True,
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
    submitted = _submitted_items(db, run, [item.id for _, item in records])
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
        if include_protected or item.id not in protected
    ]


def create_review_round(
    db: Session, project_id: int, run_id: int, data: schemas.PredictionReviewCreate, actor: User
) -> dict:
    _lock_and_validate_active_round_annotators(db, [actor.id])
    _lock_project(db, project_id)
    run = get_prediction_run(db, project_id, run_id)
    db.refresh(run)
    output = _output(db, run)
    project = db.get(Project, project_id)
    if db.query(WorkspaceMember.id).filter(
        WorkspaceMember.workspace_id == project.workspace_id,
        WorkspaceMember.user_id == actor.id,
    ).with_for_update().first() is None:
        raise ValidationError("Review annotator must be a project workspace member")
    marker = dict(_metadata(run))
    previous = dict(marker.get("review_requests", {}))
    request_hash = _canonical_hash(
        {
            **data.model_dump(exclude={"request_key"}),
            "dataset_item_ids": sorted(data.dataset_item_ids),
            "actor_id": actor.id,
        }
    )
    review_reason = f"prediction_review:{run.id}:{data.request_key}"
    previous_round = db.query(models.AnnotationRound).filter(
        models.AnnotationRound.project_id == project_id,
        models.AnnotationRound.reason == review_reason,
    ).first()
    if previous_round is not None:
        item = db.query(models.RoundItem).filter(
            models.RoundItem.annotation_round_id == previous_round.id,
        ).order_by(models.RoundItem.id).first()
        details = item.selection_reason if item is not None else {}
        if details.get("request_hash") != request_hash:
            raise ConflictError("Review request key was already used for another selection")
        return details["review_result"]
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
    reserved = _unavailable_selection_item_ids(db, run, include_submitted=False)
    submitted = (
        _unavailable_selection_item_ids(db, run) - reserved
        if not data.include_submitted else set()
    )
    eligible = [
        item_id
        for item_id in data.dataset_item_ids
        if item_id not in protected and item_id not in submitted and item_id not in reserved
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
        reason=review_reason,
        status="open",
        opened_at=datetime.now(UTC),
        created_by_user_id=actor.id,
    )
    db.add(annotation_round)
    db.flush()
    result = {
        "round_id": annotation_round.id,
        "item_count": len(eligible),
        "excluded_protected": len(protected),
        "excluded_submitted": len(set(data.dataset_item_ids) & submitted - protected),
    }
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
                    "request_hash": request_hash,
                    "review_result": result,
                },
                metadata_={"prediction_run_id": run.id, "feedback_set_version_id": output.id},
            )
        )
    _commit(db, "Could not create the prediction review round")
    return result
