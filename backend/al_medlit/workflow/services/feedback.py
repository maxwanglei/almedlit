"""Domain operations for the canonical learning workflow."""

import re
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from al_medlit.workflow import models, schemas
from al_medlit.workspace.models import WorkspaceMember

from .common import (
    _commit,
    _FeedbackEventLineage,
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




def create_feedback_exposure(
    db: Session, data: schemas.FeedbackExposureCreate, actor: User
) -> models.FeedbackExposure:
    item = _scoped(db, models.RoundItem, data.round_item_id, data.project_id, "Round item")
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        item.annotation_round_id,
        data.project_id,
        "Annotation round",
    )
    candidate = _scoped(
        db,
        models.FeedbackCandidate,
        data.feedback_candidate_id,
        data.project_id,
        "Feedback candidate",
    )
    if annotation_round.status != "open":
        raise ConflictError("Feedback may only be exposed in an open annotation round")
    if (
        not annotation_round.open_to_all_annotators
        and actor.id not in annotation_round.annotator_user_ids
    ):
        raise ConflictError("Current user is not assigned to this annotation round")
    if annotation_round.assistance_policy == "blind":
        raise ConflictError("Feedback cannot be exposed in a blind annotation round")
    if data.exposure_mode != annotation_round.assistance_policy:
        raise ValidationError("Exposure mode must match the annotation round assistance policy")
    if candidate.feedback_set_version_id != annotation_round.feedback_set_version_id:
        raise ValidationError("Feedback candidate is not pinned to this annotation round")
    if candidate.dataset_item_id != item.dataset_item_id:
        raise ValidationError("Feedback candidate and round item must reference the same item")
    if annotation_round.assistance_policy == "reveal_after_first_pass":
        initial = (
            db.query(models.RoundAnnotationDecision.id)
            .filter(
                models.RoundAnnotationDecision.round_item_id == item.id,
                models.RoundAnnotationDecision.annotator_user_id == actor.id,
                models.RoundAnnotationDecision.is_initial_checkpoint.is_(True),
            )
            .first()
        )
        if initial is None:
            raise ConflictError(
                "An independent initial annotation is required before feedback is revealed"
            )
    exposure = models.FeedbackExposure(
        **data.model_dump(),
        user_id=actor.id,
    )
    db.add(exposure)
    _commit(db, "Could not record the feedback exposure")
    db.refresh(exposure)
    return exposure


def reveal_round_feedback(
    db: Session,
    *,
    project_id: int,
    annotation_round_id: int,
    round_item_id: int,
    candidate_key: str,
    context: dict,
    actor: User,
) -> tuple[models.FeedbackExposure, models.FeedbackCandidate]:
    """Reveal one pinned candidate and append its exposure in one transaction."""

    item = (
        db.query(models.RoundItem)
        .filter(
            models.RoundItem.id == round_item_id,
            models.RoundItem.project_id == project_id,
            models.RoundItem.annotation_round_id == annotation_round_id,
        )
        .with_for_update()
        .one_or_none()
    )
    if item is None:
        raise NotFoundError("Round item not found")
    annotation_round = (
        db.query(models.AnnotationRound)
        .filter(
            models.AnnotationRound.id == annotation_round_id,
            models.AnnotationRound.project_id == project_id,
        )
        .with_for_update()
        .one_or_none()
    )
    if annotation_round is None:
        raise NotFoundError("Annotation round not found")
    if annotation_round.status != "open":
        raise ConflictError("Feedback may only be revealed in an open annotation round")
    if (
        not annotation_round.open_to_all_annotators
        and actor.id not in annotation_round.annotator_user_ids
    ):
        raise ConflictError("Current user is not assigned to this annotation round")
    if annotation_round.assistance_policy == "blind":
        raise ConflictError("Feedback cannot be revealed in a blind annotation round")
    if annotation_round.feedback_set_version_id is None:
        raise ConflictError("This annotation round has no pinned feedback set")
    candidate = (
        db.query(models.FeedbackCandidate)
        .filter(
            models.FeedbackCandidate.project_id == project_id,
            models.FeedbackCandidate.feedback_set_version_id
            == annotation_round.feedback_set_version_id,
            models.FeedbackCandidate.dataset_item_id == item.dataset_item_id,
            models.FeedbackCandidate.candidate_key == candidate_key,
        )
        .one_or_none()
    )
    if candidate is None:
        raise NotFoundError("Pinned feedback candidate not found for this round item")
    if annotation_round.assistance_policy == "reveal_after_first_pass":
        initial = (
            db.query(models.RoundAnnotationDecision.id)
            .filter(
                models.RoundAnnotationDecision.round_item_id == item.id,
                models.RoundAnnotationDecision.annotator_user_id == actor.id,
                models.RoundAnnotationDecision.is_initial_checkpoint.is_(True),
            )
            .first()
        )
        if initial is None:
            raise ConflictError(
                "An independent initial annotation is required before feedback is revealed"
            )
    exposure = models.FeedbackExposure(
        project_id=project_id,
        round_item_id=item.id,
        feedback_candidate_id=candidate.id,
        user_id=actor.id,
        exposure_mode=annotation_round.assistance_policy,
        context=context,
    )
    db.add(exposure)
    _commit(db, "Could not reveal and record the feedback exposure")
    db.refresh(exposure)
    db.refresh(candidate)
    return exposure, candidate


def create_feedback_decision(
    db: Session, data: schemas.FeedbackDecisionCreate, actor: User
) -> models.FeedbackDecision:
    exposure = _scoped(
        db,
        models.FeedbackExposure,
        data.feedback_exposure_id,
        data.project_id,
        "Feedback exposure",
    )
    if exposure.user_id != actor.id:
        raise ValidationError("Feedback decisions must belong to the exposed user")
    if data.round_annotation_decision_id is not None:
        annotation_decision = _scoped(
            db,
            models.RoundAnnotationDecision,
            data.round_annotation_decision_id,
            data.project_id,
            "Annotation decision",
        )
        if (
            annotation_decision.annotator_user_id != actor.id
            or annotation_decision.round_item_id != exposure.round_item_id
        ):
            raise ValidationError(
                "Feedback and annotation decisions must belong to the same user and round item"
            )
    decision = models.FeedbackDecision(**data.model_dump(), user_id=actor.id)
    db.add(decision)
    _commit(db, "Could not record the feedback decision")
    db.refresh(decision)
    return decision


def _resolve_feedback_event_lineage(
    db: Session,
    *,
    project_id: int,
    cycle_id: int | None,
    annotation_round_id: int | None,
    round_item_id: int | None,
    feedback_candidate_id: int | None,
) -> _FeedbackEventLineage:
    cycle = (
        _scoped(db, models.LearningCycle, cycle_id, project_id, "Learning cycle")
        if cycle_id is not None
        else None
    )
    annotation_round = (
        _scoped(
            db,
            models.AnnotationRound,
            annotation_round_id,
            project_id,
            "Annotation round",
        )
        if annotation_round_id is not None
        else None
    )
    round_item = (
        _scoped(db, models.RoundItem, round_item_id, project_id, "Round item")
        if round_item_id is not None
        else None
    )
    feedback_candidate = (
        _scoped(
            db,
            models.FeedbackCandidate,
            feedback_candidate_id,
            project_id,
            "Feedback candidate",
        )
        if feedback_candidate_id is not None
        else None
    )

    dataset_version_id: int | None = None
    task_version_id: int | None = None

    def bind_context(
        candidate_dataset_version_id: int,
        candidate_task_version_id: int,
    ) -> None:
        nonlocal dataset_version_id, task_version_id
        if dataset_version_id is None:
            dataset_version_id = candidate_dataset_version_id
            task_version_id = candidate_task_version_id
            return
        if (
            dataset_version_id != candidate_dataset_version_id
            or task_version_id != candidate_task_version_id
        ):
            raise ValidationError(
                "Feedback event references must use the same task and dataset versions"
            )

    if cycle is not None:
        bind_context(cycle.source_dataset_version_id, cycle.task_version_id)

    dataset_item = None
    if round_item is not None:
        item_round = _scoped(
            db,
            models.AnnotationRound,
            round_item.annotation_round_id,
            project_id,
            "Round item annotation round",
        )
        if annotation_round is not None and annotation_round.id != item_round.id:
            raise ValidationError(
                "Feedback event round item does not belong to the referenced annotation round"
            )
        annotation_round = item_round
        dataset_item = _scoped(
            db,
            models.DatasetItem,
            round_item.dataset_item_id,
            project_id,
            "Round dataset item",
        )
        if dataset_item.dataset_version_id != annotation_round.dataset_version_id:
            raise ValidationError(
                "Feedback event round item does not belong to the round dataset version"
            )

    if annotation_round is not None:
        bind_context(
            annotation_round.dataset_version_id,
            annotation_round.task_version_id,
        )
        if annotation_round.cycle_id is not None:
            round_cycle = _scoped(
                db,
                models.LearningCycle,
                annotation_round.cycle_id,
                project_id,
                "Annotation round learning cycle",
            )
            if cycle is not None and cycle.id != round_cycle.id:
                raise ValidationError(
                    "Feedback event cycle does not match the referenced annotation round"
                )
            cycle = round_cycle
        elif cycle is not None:
            raise ValidationError(
                "Feedback event cycle does not match the independent annotation round"
            )

    if feedback_candidate is not None:
        feedback_set = _scoped(
            db,
            models.FeedbackSetVersion,
            feedback_candidate.feedback_set_version_id,
            project_id,
            "Feedback candidate set",
        )
        feedback_run = _scoped(
            db,
            models.FeedbackRun,
            feedback_set.feedback_run_id,
            project_id,
            "Feedback candidate run",
        )
        candidate_item = _scoped(
            db,
            models.DatasetItem,
            feedback_candidate.dataset_item_id,
            project_id,
            "Feedback candidate dataset item",
        )
        if candidate_item.dataset_version_id != feedback_set.dataset_version_id:
            raise ValidationError(
                "Feedback candidate does not belong to its feedback-set dataset version"
            )
        bind_context(feedback_set.dataset_version_id, feedback_set.task_version_id)
        if dataset_item is not None and dataset_item.id != candidate_item.id:
            raise ValidationError(
                "Feedback event candidate and round item must reference the same dataset item"
            )
        dataset_item = candidate_item
        if annotation_round is not None and (
            annotation_round.feedback_set_version_id != feedback_set.id
        ):
            raise ValidationError(
                "Feedback event candidate is not pinned to the referenced annotation round"
            )
        if feedback_run.cycle_id is not None:
            feedback_cycle = _scoped(
                db,
                models.LearningCycle,
                feedback_run.cycle_id,
                project_id,
                "Feedback run learning cycle",
            )
            if cycle is not None and cycle.id != feedback_cycle.id:
                raise ValidationError(
                    "Feedback event cycle does not match the feedback candidate run"
                )
            cycle = feedback_cycle

    if cycle is not None:
        bind_context(cycle.source_dataset_version_id, cycle.task_version_id)
    if annotation_round is not None and (
        annotation_round.cycle_id != (cycle.id if cycle is not None else None)
    ):
        raise ValidationError(
            "Feedback event annotation round and candidate have inconsistent cycles"
        )

    return _FeedbackEventLineage(
        cycle=cycle,
        annotation_round=annotation_round,
        round_item=round_item,
        feedback_candidate=feedback_candidate,
        dataset_item=dataset_item,
    )


def _dataset_item_is_in_protected_group(
    db: Session,
    dataset_item: models.DatasetItem,
) -> bool:
    stable_keys = {dataset_item.stable_key}
    if dataset_item.group_key is not None:
        stable_keys.update(
            stable_key
            for (stable_key,) in db.query(models.DatasetItem.stable_key)
            .filter(
                models.DatasetItem.dataset_version_id == dataset_item.dataset_version_id,
                models.DatasetItem.group_key == dataset_item.group_key,
            )
            .all()
        )
    split_maps = (
        db.query(models.SplitMap)
        .filter(
            models.SplitMap.project_id == dataset_item.project_id,
            models.SplitMap.dataset_version_id == dataset_item.dataset_version_id,
        )
        .all()
    )
    return any(
        any(
            split_map.assignments.get(stable_key) in set(split_map.protected_splits)
            for stable_key in stable_keys
        )
        for split_map in split_maps
    )


def create_feedback_event(
    db: Session, data: schemas.FeedbackEventCreate, actor: User
) -> models.FeedbackEvent:
    _project(db, data.project_id)
    lineage = _resolve_feedback_event_lineage(
        db,
        project_id=data.project_id,
        cycle_id=data.cycle_id,
        annotation_round_id=data.annotation_round_id,
        round_item_id=data.round_item_id,
        feedback_candidate_id=data.feedback_candidate_id,
    )
    payload = data.model_dump()
    payload["cycle_id"] = lineage.cycle.id if lineage.cycle is not None else None
    payload["annotation_round_id"] = (
        lineage.annotation_round.id if lineage.annotation_round is not None else None
    )
    event_record = models.FeedbackEvent(
        **payload,
        created_by_user_id=actor.id,
    )
    db.add(event_record)
    _commit(db, "Could not record the feedback event")
    db.refresh(event_record)
    return event_record


def list_feedback_events(
    db: Session,
    project_id: int,
    *,
    annotation_round_id: int | None = None,
) -> list[models.FeedbackEvent]:
    _project(db, project_id)
    query = db.query(models.FeedbackEvent).filter(models.FeedbackEvent.project_id == project_id)
    if annotation_round_id is not None:
        _scoped(
            db,
            models.AnnotationRound,
            annotation_round_id,
            project_id,
            "Annotation round",
        )
        query = query.filter(models.FeedbackEvent.annotation_round_id == annotation_round_id)
    return query.order_by(
        models.FeedbackEvent.occurred_at.desc(),
        models.FeedbackEvent.id.desc(),
    ).all()


def create_review_case(db: Session, data: schemas.ReviewCaseCreate) -> models.ReviewCase:
    _scoped(
        db,
        models.FeedbackEvent,
        data.feedback_event_id,
        data.project_id,
        "Feedback event",
    )
    if data.assigned_to_user_id is not None:
        project = _project(db, data.project_id)
        member = (
            db.query(WorkspaceMember.id)
            .filter(
                WorkspaceMember.workspace_id == project.workspace_id,
                WorkspaceMember.user_id == data.assigned_to_user_id,
            )
            .first()
        )
        if member is None:
            raise ValidationError("Review-case assignee must be a workspace member")
    review_case = models.ReviewCase(**data.model_dump())
    db.add(review_case)
    _commit(db, "Could not create the review case")
    db.refresh(review_case)
    return review_case


def list_review_cases(db: Session, project_id: int) -> list[models.ReviewCase]:
    _project(db, project_id)
    return (
        db.query(models.ReviewCase)
        .filter(models.ReviewCase.project_id == project_id)
        .order_by(models.ReviewCase.status, models.ReviewCase.created_at.desc())
        .all()
    )


def resolve_review_case(
    db: Session,
    project_id: int,
    review_case_id: int,
    data: schemas.ReviewCaseResolve,
) -> models.ReviewCase:
    review_case = _scoped(db, models.ReviewCase, review_case_id, project_id, "Review case")
    if review_case.status == "resolved":
        raise ConflictError("Review case is already resolved")
    review_case.status = "resolved"
    review_case.resolution = data.resolution
    review_case.resolved_at = datetime.now(UTC)
    _commit(db, "Could not resolve the review case")
    db.refresh(review_case)
    return review_case
