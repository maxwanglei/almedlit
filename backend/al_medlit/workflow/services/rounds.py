"""Domain operations for the canonical learning workflow."""

import re
from datetime import UTC, datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from al_medlit.workflow import models, schemas
from al_medlit.workspace.models import WorkspaceMember

from .common import (
    _actor_has_role,
    _canonical_hash,
    _commit,
    _next_sequence,
    _project,
    _require_actor_role,
    _scoped,
    _validate_task_output,
)
from .workspace_training import (
    _workspace_projects_with_modules,
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


def _lock_and_validate_active_round_annotators(
    db: Session,
    annotator_user_ids: list[int],
) -> None:
    if not annotator_user_ids:
        return
    annotators = (
        db.query(User)
        .filter(User.id.in_(annotator_user_ids))
        .order_by(User.id)
        .populate_existing()
        .with_for_update()
        .all()
    )
    active_annotator_ids = {user.id for user in annotators if user.is_active}
    inactive_or_missing = sorted(set(annotator_user_ids) - active_annotator_ids)
    if inactive_or_missing:
        raise ValidationError("All round annotators must be active users")




def create_annotation_round(
    db: Session, data: schemas.AnnotationRoundCreate, actor: User
) -> models.AnnotationRound:
    dataset = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    task = _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    if len(data.annotator_user_ids) != len(set(data.annotator_user_ids)):
        raise ValidationError("annotator_user_ids must be unique")
    if data.annotator_user_ids and data.open_to_all_annotators:
        raise ValidationError("Choose explicit annotators or an open round, not both")
    if not data.annotator_user_ids and not data.open_to_all_annotators:
        raise ValidationError(
            "Annotation rounds require explicit annotators or open_to_all_annotators"
        )
    project = _project(db, data.project_id)
    if data.annotator_user_ids:
        # Account deactivation locks the User row before withdrawing mutable
        # work. Use the same boundary here so a new explicit round assignment
        # cannot race in after offboarding has completed.
        _lock_and_validate_active_round_annotators(db, data.annotator_user_ids)
        member_user_ids = {
            user_id
            for (user_id,) in db.query(WorkspaceMember.user_id)
            .filter(
                WorkspaceMember.workspace_id == project.workspace_id,
                WorkspaceMember.user_id.in_(data.annotator_user_ids),
            )
            .all()
        }
        missing = sorted(set(data.annotator_user_ids) - member_user_ids)
        if missing:
            raise ValidationError("All round annotators must be project workspace members")
    if data.parent_round_id is not None:
        parent = _scoped(
            db,
            models.AnnotationRound,
            data.parent_round_id,
            data.project_id,
            "Parent annotation round",
        )
        if parent.task_version_id != task.id:
            raise ValidationError("Parent annotation round must use the same task version")
    cycle = None
    if data.cycle_id is not None:
        cycle = _scoped(db, models.LearningCycle, data.cycle_id, data.project_id, "Learning cycle")
        if cycle.source_dataset_version_id != dataset.id or cycle.task_version_id != task.id:
            raise ValidationError("Annotation round must match its learning cycle")
    if data.guideline_revision_id is not None:
        guideline = _scoped(
            db,
            models.GuidelineRevision,
            data.guideline_revision_id,
            data.project_id,
            "Guideline revision",
        )
        if guideline.task_version_id != task.id:
            raise ValidationError("Guideline revision must use the annotation task version")
    selection_set = None
    if data.selection_set_version_id is not None:
        selection_set = _scoped(
            db,
            models.SelectionSetVersion,
            data.selection_set_version_id,
            data.project_id,
            "Selection set version",
        )
        selection_run = _scoped(
            db,
            models.SelectionRun,
            selection_set.selection_run_id,
            data.project_id,
            "Selection run",
        )
        if (
            selection_run.dataset_version_id != dataset.id
            or selection_run.task_version_id != task.id
        ):
            raise ValidationError("Selection set must use the annotation task and dataset")
        if selection_run.cycle_id != data.cycle_id:
            raise ValidationError("Selection set must belong to the annotation round's exact cycle")
    if data.reannotation_mode == "targeted_subset" and selection_set is None:
        raise ValidationError("targeted_subset rounds require a selection_set_version_id")
    if data.feedback_set_version_id is not None:
        feedback_set = _scoped(
            db,
            models.FeedbackSetVersion,
            data.feedback_set_version_id,
            data.project_id,
            "Feedback set version",
        )
        if feedback_set.dataset_version_id != dataset.id or feedback_set.task_version_id != task.id:
            raise ValidationError("Feedback set must use the annotation task and dataset")
        feedback_run = _scoped(
            db,
            models.FeedbackRun,
            feedback_set.feedback_run_id,
            data.project_id,
            "Feedback run",
        )
        if feedback_run.cycle_id != data.cycle_id:
            raise ValidationError("Feedback set must belong to the annotation round's exact cycle")
    elif data.assistance_policy != "blind":
        raise ValidationError("Assisted annotation rounds require a pinned feedback set version")

    sequence = _next_sequence(
        db,
        models.AnnotationRound,
        data.project_id,
        cycle_id=data.cycle_id,
    )
    annotation_round = models.AnnotationRound(
        **data.model_dump(),
        sequence=sequence,
        created_by_user_id=actor.id,
    )
    db.add(annotation_round)
    db.flush()

    if selection_set is not None:
        item_payloads = selection_set.items
        for selected in item_payloads:
            db.add(
                models.RoundItem(
                    project_id=data.project_id,
                    annotation_round_id=annotation_round.id,
                    dataset_item_id=selected["dataset_item_id"],
                    selection_rank=selected.get("rank"),
                    selection_score=selected.get("score"),
                    selection_probability=selected.get("probability"),
                    selection_reason=selected.get("reason") or {},
                    metadata_={},
                )
            )
    elif data.reannotation_mode == "full_dataset":
        for (item_id,) in (
            db.query(models.DatasetItem.id)
            .filter(models.DatasetItem.dataset_version_id == dataset.id)
            .order_by(models.DatasetItem.id)
            .all()
        ):
            db.add(
                models.RoundItem(
                    project_id=data.project_id,
                    annotation_round_id=annotation_round.id,
                    dataset_item_id=item_id,
                    selection_reason={"strategy": "all"},
                    metadata_={},
                )
            )
    _commit(db, "Could not create the annotation round")
    db.refresh(annotation_round)
    return annotation_round


def list_annotation_rounds(
    db: Session, project_id: int, cycle_id: int | None = None
) -> list[models.AnnotationRound]:
    _project(db, project_id)
    query = db.query(models.AnnotationRound).filter(models.AnnotationRound.project_id == project_id)
    if cycle_id is not None:
        _scoped(db, models.LearningCycle, cycle_id, project_id, "Learning cycle")
        query = query.filter(models.AnnotationRound.cycle_id == cycle_id)
    return query.order_by(models.AnnotationRound.created_at, models.AnnotationRound.id).all()


def get_annotation_round(
    db: Session,
    annotation_round_id: int,
) -> models.AnnotationRound:
    annotation_round = db.get(models.AnnotationRound, annotation_round_id)
    if annotation_round is None:
        raise NotFoundError("Annotation round not found")
    return annotation_round


def require_annotation_round_actor_access(
    db: Session,
    annotation_round: models.AnnotationRound,
    actor: User,
) -> models.AnnotationRound:
    _resolve_round_history_annotator_id(
        db,
        annotation_round=annotation_round,
        project_id=annotation_round.project_id,
        actor=actor,
        annotator_user_id=None,
    )
    return annotation_round


def get_round_work_context(
    db: Session,
    annotation_round: models.AnnotationRound,
    actor: User,
) -> dict:
    require_annotation_round_actor_access(db, annotation_round, actor)
    return _round_work_context_payload(db, annotation_round)


def _round_work_context_payload(
    db: Session,
    annotation_round: models.AnnotationRound,
) -> dict:
    project = _project(db, annotation_round.project_id)
    task_version = _scoped(
        db,
        models.TaskVersion,
        annotation_round.task_version_id,
        annotation_round.project_id,
        "Task version",
    )
    task = _scoped(
        db,
        models.TaskDefinition,
        task_version.task_definition_id,
        annotation_round.project_id,
        "Task definition",
    )

    cycle_payload = None
    if annotation_round.cycle_id is not None:
        cycle = _scoped(
            db,
            models.LearningCycle,
            annotation_round.cycle_id,
            annotation_round.project_id,
            "Learning cycle",
        )
        if cycle.task_version_id != task_version.id:
            raise ValidationError("Annotation round cycle does not match its pinned task version")
        cycle_payload = {
            "id": cycle.id,
            "name": cycle.name,
            "sequence": cycle.sequence,
        }

    guideline_payload = None
    if annotation_round.guideline_revision_id is not None:
        revision = _scoped(
            db,
            models.GuidelineRevision,
            annotation_round.guideline_revision_id,
            annotation_round.project_id,
            "Guideline revision",
        )
        guideline = _scoped(
            db,
            models.Guideline,
            revision.guideline_id,
            annotation_round.project_id,
            "Guideline",
        )
        if revision.task_version_id != task_version.id or guideline.task_definition_id != task.id:
            raise ValidationError(
                "Annotation round guideline does not match its pinned task version"
            )
        guideline_payload = {
            "guideline_id": guideline.id,
            "guideline_revision_id": revision.id,
            "name": guideline.name,
            "version_number": revision.version_number,
            "status": revision.status,
        }

    return {
        "project": {
            "id": project.id,
            "name": project.name,
        },
        "round": {
            "id": annotation_round.id,
            "created_at": annotation_round.created_at,
            "updated_at": annotation_round.updated_at,
            "project_id": annotation_round.project_id,
            "name": annotation_round.name,
            "sequence": annotation_round.sequence,
            "dataset_version_id": annotation_round.dataset_version_id,
            "task_version_id": annotation_round.task_version_id,
            "assistance_policy": annotation_round.assistance_policy,
            "feedback_available": (
                annotation_round.feedback_set_version_id is not None
                and annotation_round.assistance_policy != "blind"
            ),
            "status": annotation_round.status,
            "opened_at": annotation_round.opened_at,
            "closed_at": annotation_round.closed_at,
        },
        "task": {
            "id": task.id,
            "key": task.key,
            "name": task.name,
        },
        "task_version": task_version,
        "cycle": cycle_payload,
        "guideline": guideline_payload,
    }


def list_workspace_round_work_contexts(
    db: Session,
    *,
    workspace_id: int,
    actor: User,
) -> list[dict]:
    eligible_projects = _workspace_projects_with_modules(
        db,
        workspace_id=workspace_id,
        required_modules={"annotate"},
    )
    project_ids = [project.id for project, _modules in eligible_projects]
    if not project_ids:
        return []

    rounds = (
        db.query(models.AnnotationRound)
        .filter(
            models.AnnotationRound.project_id.in_(project_ids),
            models.AnnotationRound.status == "open",
        )
        .order_by(
            models.AnnotationRound.opened_at.desc(),
            models.AnnotationRound.id.desc(),
        )
        .all()
    )
    visible_rounds = [
        annotation_round
        for annotation_round in rounds
        if annotation_round.open_to_all_annotators
        or actor.id in annotation_round.annotator_user_ids
    ]
    return [
        _round_work_context_payload(db, annotation_round) for annotation_round in visible_rounds
    ]


def list_round_items(
    db: Session, project_id: int, annotation_round_id: int
) -> list[models.RoundItem]:
    _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        project_id,
        "Annotation round",
    )
    return (
        db.query(models.RoundItem)
        .filter(models.RoundItem.annotation_round_id == annotation_round_id)
        .order_by(models.RoundItem.selection_rank, models.RoundItem.id)
        .all()
    )


def list_round_work_items(
    db: Session,
    project_id: int,
    annotation_round_id: int,
) -> list[tuple[models.RoundItem, models.DatasetItem]]:
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        project_id,
        "Annotation round",
    )
    round_item_count = (
        db.query(models.RoundItem.id)
        .filter(
            models.RoundItem.project_id == project_id,
            models.RoundItem.annotation_round_id == annotation_round.id,
        )
        .count()
    )
    work_items = (
        db.query(models.RoundItem, models.DatasetItem)
        .join(
            models.DatasetItem,
            models.DatasetItem.id == models.RoundItem.dataset_item_id,
        )
        .filter(
            models.RoundItem.project_id == project_id,
            models.RoundItem.annotation_round_id == annotation_round.id,
            models.DatasetItem.project_id == project_id,
            models.DatasetItem.dataset_version_id == annotation_round.dataset_version_id,
        )
        .order_by(models.RoundItem.selection_rank, models.RoundItem.id)
        .all()
    )
    if len(work_items) != round_item_count:
        raise ValidationError(
            "Annotation round item lineage does not match its pinned dataset version"
        )
    return work_items


def _resolve_round_history_annotator_id(
    db: Session,
    *,
    annotation_round: models.AnnotationRound,
    project_id: int,
    actor: User,
    annotator_user_id: int | None,
) -> int | None:
    if _actor_has_role(
        db,
        project_id=project_id,
        actor=actor,
        minimum_role="manager",
    ):
        return annotator_user_id
    if annotator_user_id is not None and annotator_user_id != actor.id:
        raise ForbiddenError("Only managers may view another annotator's round history")
    if (
        not annotation_round.open_to_all_annotators
        and actor.id not in annotation_round.annotator_user_ids
    ):
        raise ForbiddenError("Current user is not assigned to this annotation round")
    return actor.id


def list_round_annotation_decisions(
    db: Session,
    project_id: int,
    annotation_round_id: int,
    actor: User,
    annotator_user_id: int | None = None,
) -> list[models.RoundAnnotationDecision]:
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        project_id,
        "Annotation round",
    )
    scoped_annotator_id = _resolve_round_history_annotator_id(
        db,
        annotation_round=annotation_round,
        project_id=project_id,
        actor=actor,
        annotator_user_id=annotator_user_id,
    )
    query = (
        db.query(models.RoundAnnotationDecision)
        .join(
            models.RoundItem,
            models.RoundItem.id == models.RoundAnnotationDecision.round_item_id,
        )
        .filter(
            models.RoundItem.annotation_round_id == annotation_round.id,
            models.RoundAnnotationDecision.project_id == project_id,
        )
    )
    if scoped_annotator_id is not None:
        query = query.filter(
            models.RoundAnnotationDecision.annotator_user_id == scoped_annotator_id
        )
    return query.order_by(
        models.RoundAnnotationDecision.created_at,
        models.RoundAnnotationDecision.id,
    ).all()


def list_round_submissions(
    db: Session,
    project_id: int,
    annotation_round_id: int,
    actor: User,
    annotator_user_id: int | None = None,
) -> list[models.RoundSubmission]:
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        project_id,
        "Annotation round",
    )
    scoped_annotator_id = _resolve_round_history_annotator_id(
        db,
        annotation_round=annotation_round,
        project_id=project_id,
        actor=actor,
        annotator_user_id=annotator_user_id,
    )
    query = db.query(models.RoundSubmission).filter(
        models.RoundSubmission.annotation_round_id == annotation_round.id,
        models.RoundSubmission.project_id == project_id,
    )
    if scoped_annotator_id is not None:
        query = query.filter(models.RoundSubmission.annotator_user_id == scoped_annotator_id)
    return query.order_by(
        models.RoundSubmission.submitted_at,
        models.RoundSubmission.id,
    ).all()


def transition_annotation_round(
    db: Session, project_id: int, annotation_round_id: int, status: str
) -> models.AnnotationRound:
    candidate = _scoped(
        db,
        models.AnnotationRound,
        annotation_round_id,
        project_id,
        "Annotation round",
    )
    candidate_annotator_ids = list(candidate.annotator_user_ids or [])
    if status == "open":
        # Deactivation and round creation also lock users before mutable work.
        # Acquire the explicit assignees in canonical order, then refresh and
        # lock the round so an offboarding transaction cannot race this open.
        _lock_and_validate_active_round_annotators(db, candidate_annotator_ids)
    annotation_round = (
        db.query(models.AnnotationRound)
        .filter(
            models.AnnotationRound.id == annotation_round_id,
            models.AnnotationRound.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if annotation_round is None:
        raise NotFoundError("Annotation round not found")
    if status == "open":
        if list(annotation_round.annotator_user_ids or []) != candidate_annotator_ids:
            raise ConflictError("Annotation round assignments changed; retry the transition")
        if not annotation_round.open_to_all_annotators and not candidate_annotator_ids:
            raise ValidationError(
                "Annotation rounds require explicit active annotators or "
                "open_to_all_annotators"
            )
    allowed = {
        "draft": {"open", "cancelled"},
        "open": {"closed", "cancelled"},
        "closed": set(),
        "cancelled": set(),
    }
    if status not in allowed.get(annotation_round.status, set()):
        raise ConflictError(
            f"Cannot transition annotation round from {annotation_round.status} to {status}"
        )
    if status == "open":
        has_items = (
            db.query(models.RoundItem.id)
            .filter(models.RoundItem.annotation_round_id == annotation_round.id)
            .first()
        )
        if has_items is None:
            raise ValidationError("An annotation round must contain items before it opens")
        annotation_round.opened_at = datetime.now(UTC)
    elif status == "closed":
        round_item_ids = {
            item_id
            for (item_id,) in db.query(models.RoundItem.id)
            .filter(models.RoundItem.annotation_round_id == annotation_round.id)
            .all()
        }
        submissions = (
            db.query(models.RoundSubmission)
            .filter(models.RoundSubmission.annotation_round_id == annotation_round.id)
            .all()
        )
        submitted_item_ids = {
            decision.round_item_id
            for submission in submissions
            for decision in (
                db.query(models.RoundAnnotationDecision)
                .filter(models.RoundAnnotationDecision.id.in_(submission.decision_ids))
                .all()
            )
        }
        if not submissions or submitted_item_ids != round_item_ids:
            raise ValidationError(
                "A round can close only after finalized submissions cover every item"
            )
        annotation_round.closed_at = datetime.now(UTC)
    elif status == "cancelled":
        annotation_round.closed_at = datetime.now(UTC)
    annotation_round.status = status
    _commit(db, "Could not transition the annotation round")
    db.refresh(annotation_round)
    return annotation_round


def create_annotation_decision(
    db: Session, data: schemas.AnnotationDecisionCreate, actor: User
) -> models.RoundAnnotationDecision:
    item = _scoped(db, models.RoundItem, data.round_item_id, data.project_id, "Round item")
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        item.annotation_round_id,
        data.project_id,
        "Annotation round",
    )
    if annotation_round.status != "open":
        raise ConflictError("Annotation decisions may only be added to an open round")
    task_version = _scoped(
        db,
        models.TaskVersion,
        annotation_round.task_version_id,
        data.project_id,
        "Task version",
    )
    _validate_task_output(
        task_version,
        data.output,
        label="Annotation decision",
    )
    if (
        not annotation_round.open_to_all_annotators
        and actor.id not in annotation_round.annotator_user_ids
    ):
        raise ConflictError("Current user is not assigned to this annotation round")
    if data.decision_kind == "adjudication":
        _require_actor_role(
            db,
            project_id=data.project_id,
            actor=actor,
            minimum_role="manager",
        )
    if data.supersedes_decision_id is not None:
        prior = _scoped(
            db,
            models.RoundAnnotationDecision,
            data.supersedes_decision_id,
            data.project_id,
            "Superseded annotation decision",
        )
        if prior.round_item_id != item.id or prior.annotator_user_id != actor.id:
            raise ValidationError(
                "A revision may only supersede the actor's decision for the same round item"
            )
        already_superseded = (
            db.query(models.RoundAnnotationDecision.id)
            .filter(models.RoundAnnotationDecision.supersedes_decision_id == prior.id)
            .first()
        )
        if already_superseded is not None:
            raise ConflictError("The selected annotation decision was already superseded")
    content = data.model_dump(exclude={"project_id"})
    decision = models.RoundAnnotationDecision(
        **data.model_dump(),
        annotator_user_id=actor.id,
        content_hash=_canonical_hash(
            {
                **content,
                "annotator_user_id": actor.id,
            }
        ),
    )
    db.add(decision)
    _commit(db, "Could not add the annotation decision")
    db.refresh(decision)
    return decision


def create_round_submission(
    db: Session, data: schemas.RoundSubmissionCreate, actor: User
) -> models.RoundSubmission:
    annotation_round = _scoped(
        db,
        models.AnnotationRound,
        data.annotation_round_id,
        data.project_id,
        "Annotation round",
    )
    if annotation_round.status != "open":
        raise ConflictError("Only open annotation rounds accept submissions")
    if (
        not annotation_round.open_to_all_annotators
        and actor.id not in annotation_round.annotator_user_ids
    ):
        raise ConflictError("Current user is not assigned to this annotation round")
    if len(data.decision_ids) != len(set(data.decision_ids)):
        raise ValidationError("decision_ids must be unique")
    decisions = [
        _scoped(
            db,
            models.RoundAnnotationDecision,
            decision_id,
            data.project_id,
            "Annotation decision",
        )
        for decision_id in data.decision_ids
    ]
    for decision in decisions:
        item = _scoped(db, models.RoundItem, decision.round_item_id, data.project_id, "Round item")
        if (
            item.annotation_round_id != annotation_round.id
            or decision.annotator_user_id != actor.id
        ):
            raise ValidationError(
                "Submissions may contain only the actor's decisions from this round"
            )
        superseded = (
            db.query(models.RoundAnnotationDecision.id)
            .filter(models.RoundAnnotationDecision.supersedes_decision_id == decision.id)
            .first()
        )
        if superseded is not None:
            raise ValidationError("Submissions cannot include a superseded annotation decision")
    latest = (
        db.query(func.max(models.RoundSubmission.sequence))
        .filter(
            models.RoundSubmission.annotation_round_id == annotation_round.id,
            models.RoundSubmission.annotator_user_id == actor.id,
        )
        .scalar()
    )
    submission = models.RoundSubmission(
        **data.model_dump(),
        annotator_user_id=actor.id,
        sequence=int(latest or 0) + 1,
        content_hash=_canonical_hash(
            {
                "annotation_round_id": annotation_round.id,
                "annotator_user_id": actor.id,
                "decision_ids": sorted(data.decision_ids),
            }
        ),
    )
    db.add(submission)
    _commit(db, "Could not finalize the round submission")
    db.refresh(submission)
    return submission
