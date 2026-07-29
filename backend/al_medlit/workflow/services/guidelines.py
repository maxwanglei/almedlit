"""Domain operations for the canonical learning workflow."""

import re
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _project,
    _scoped,
)
from .feedback import (
    _dataset_item_is_in_protected_group,
    _resolve_feedback_event_lineage,
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




def create_guideline(db: Session, data: schemas.GuidelineCreate, actor: User) -> models.Guideline:
    _scoped(
        db,
        models.TaskDefinition,
        data.task_definition_id,
        data.project_id,
        "Task definition",
    )
    guideline = models.Guideline(**data.model_dump(), created_by_user_id=actor.id)
    db.add(guideline)
    _commit(db, "A guideline with this name already exists for the task")
    db.refresh(guideline)
    return guideline


def list_guidelines(db: Session, project_id: int) -> list[models.Guideline]:
    _project(db, project_id)
    return (
        db.query(models.Guideline)
        .filter(models.Guideline.project_id == project_id)
        .order_by(models.Guideline.name, models.Guideline.id)
        .all()
    )


def create_guideline_revision(
    db: Session, data: schemas.GuidelineRevisionCreate, actor: User
) -> models.GuidelineRevision:
    guideline = _scoped(db, models.Guideline, data.guideline_id, data.project_id, "Guideline")
    db.query(models.Guideline).filter(models.Guideline.id == guideline.id).with_for_update().one()
    task_version = _scoped(
        db, models.TaskVersion, data.task_version_id, data.project_id, "Task version"
    )
    if task_version.task_definition_id != guideline.task_definition_id:
        raise ValidationError("Guideline revision task version must belong to its task")
    if data.parent_revision_id is not None:
        parent = _scoped(
            db,
            models.GuidelineRevision,
            data.parent_revision_id,
            data.project_id,
            "Parent guideline revision",
        )
        if parent.guideline_id != guideline.id:
            raise ValidationError("Parent revision must belong to the same guideline")
    proposals = [
        _scoped(
            db,
            models.GuidelineChangeProposal,
            proposal_id,
            data.project_id,
            "Guideline proposal",
        )
        for proposal_id in data.source_proposal_ids
    ]
    if any(
        proposal.guideline_id != guideline.id or proposal.status != "accepted"
        for proposal in proposals
    ):
        raise ValidationError("Source proposals must be accepted proposals for this guideline")
    content = data.model_dump(exclude={"project_id", "guideline_id"})
    revision = models.GuidelineRevision(
        **data.model_dump(),
        version_number=_next_version(
            db,
            models.GuidelineRevision,
            models.GuidelineRevision.guideline_id,
            guideline.id,
        ),
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
    )
    db.add(revision)
    db.flush()
    for proposal in proposals:
        proposal.resulting_revision_id = revision.id
    _commit(db, "This exact guideline revision already exists")
    db.refresh(revision)
    return revision


def list_guideline_revisions(
    db: Session, project_id: int, guideline_id: int
) -> list[models.GuidelineRevision]:
    _scoped(db, models.Guideline, guideline_id, project_id, "Guideline")
    return (
        db.query(models.GuidelineRevision)
        .filter(models.GuidelineRevision.guideline_id == guideline_id)
        .order_by(models.GuidelineRevision.version_number)
        .all()
    )


def transition_guideline_revision(
    db: Session,
    project_id: int,
    revision_id: int,
    data: schemas.GuidelineRevisionTransition,
    actor: User,
) -> models.GuidelineRevision:
    revision = _scoped(
        db,
        models.GuidelineRevision,
        revision_id,
        project_id,
        "Guideline revision",
    )
    allowed = {
        "draft": {"pilot", "retired"},
        "pilot": {"active", "retired"},
        "active": {"retired"},
        "retired": set(),
    }
    if data.status not in allowed.get(revision.status, set()):
        raise ConflictError(
            f"Cannot transition guideline revision from {revision.status} to {data.status}"
        )
    if data.status == "active":
        passed_impact = (
            db.query(models.GuidelineImpactEvaluation.id)
            .filter(
                models.GuidelineImpactEvaluation.guideline_revision_id == revision.id,
                models.GuidelineImpactEvaluation.status == "completed",
                models.GuidelineImpactEvaluation.passed.is_(True),
            )
            .first()
        )
        if passed_impact is None:
            raise ConflictError(
                "Guideline activation requires a completed, passing pilot impact evaluation"
            )
        active_revisions = (
            db.query(models.GuidelineRevision)
            .filter(
                models.GuidelineRevision.guideline_id == revision.guideline_id,
                models.GuidelineRevision.status == "active",
                models.GuidelineRevision.id != revision.id,
            )
            .all()
        )
        for active in active_revisions:
            active.status = "retired"
        revision.approved_by_user_id = actor.id
        revision.approved_at = datetime.now(UTC)
    revision.status = data.status
    _commit(db, "Could not transition the guideline revision")
    db.refresh(revision)
    return revision


def create_guideline_proposal(
    db: Session, data: schemas.GuidelineProposalCreate, actor: User
) -> models.GuidelineChangeProposal:
    guideline = _scoped(db, models.Guideline, data.guideline_id, data.project_id, "Guideline")
    if data.base_revision_id is not None:
        base = _scoped(
            db,
            models.GuidelineRevision,
            data.base_revision_id,
            data.project_id,
            "Base guideline revision",
        )
        if base.guideline_id != guideline.id:
            raise ValidationError("Base revision must belong to the proposal guideline")
    for event_id in data.feedback_event_ids:
        feedback_event = _scoped(
            db,
            models.FeedbackEvent,
            event_id,
            data.project_id,
            "Feedback event",
        )
        lineage = _resolve_feedback_event_lineage(
            db,
            project_id=data.project_id,
            cycle_id=feedback_event.cycle_id,
            annotation_round_id=feedback_event.annotation_round_id,
            round_item_id=feedback_event.round_item_id,
            feedback_candidate_id=feedback_event.feedback_candidate_id,
        )
        if lineage.dataset_item is not None and _dataset_item_is_in_protected_group(
            db, lineage.dataset_item
        ):
            raise ValidationError(
                "Protected evaluation items or their groups cannot support guideline proposals"
            )
    proposal = models.GuidelineChangeProposal(**data.model_dump(), created_by_user_id=actor.id)
    db.add(proposal)
    _commit(db, "Could not create the guideline proposal")
    db.refresh(proposal)
    return proposal


def list_guideline_proposals(
    db: Session,
    project_id: int,
    *,
    guideline_id: int | None = None,
) -> list[models.GuidelineChangeProposal]:
    _project(db, project_id)
    query = db.query(models.GuidelineChangeProposal).filter(
        models.GuidelineChangeProposal.project_id == project_id
    )
    if guideline_id is not None:
        _scoped(db, models.Guideline, guideline_id, project_id, "Guideline")
        query = query.filter(models.GuidelineChangeProposal.guideline_id == guideline_id)
    return query.order_by(
        models.GuidelineChangeProposal.created_at.desc(),
        models.GuidelineChangeProposal.id.desc(),
    ).all()


def review_guideline_proposal(
    db: Session,
    project_id: int,
    proposal_id: int,
    data: schemas.GuidelineProposalReview,
    actor: User,
) -> models.GuidelineChangeProposal:
    proposal = _scoped(
        db,
        models.GuidelineChangeProposal,
        proposal_id,
        project_id,
        "Guideline proposal",
    )
    if proposal.status != "pending":
        raise ConflictError("Guideline proposal was already reviewed")
    proposal.status = data.decision
    proposal.reviewed_by_user_id = actor.id
    proposal.reviewed_at = datetime.now(UTC)
    _commit(db, "Could not review the guideline proposal")
    db.refresh(proposal)
    return proposal


def create_guideline_impact(
    db: Session, data: schemas.GuidelineImpactCreate, actor: User
) -> models.GuidelineImpactEvaluation:
    revision = _scoped(
        db,
        models.GuidelineRevision,
        data.guideline_revision_id,
        data.project_id,
        "Guideline revision",
    )
    if revision.status != "pilot":
        raise ConflictError("Impact evaluations require a guideline revision in pilot status")
    pilot_round = _scoped(
        db,
        models.AnnotationRound,
        data.pilot_round_id,
        data.project_id,
        "Pilot annotation round",
    )
    if pilot_round.guideline_revision_id != revision.id:
        raise ValidationError("Pilot round must pin the evaluated guideline revision")
    if data.protected_split_map_id is not None:
        _scoped(
            db,
            models.SplitMap,
            data.protected_split_map_id,
            data.project_id,
            "Protected split map",
        )
    evaluation = models.GuidelineImpactEvaluation(**data.model_dump(), created_by_user_id=actor.id)
    db.add(evaluation)
    _commit(db, "Could not create the guideline impact evaluation")
    db.refresh(evaluation)
    return evaluation


def list_guideline_impacts(
    db: Session,
    project_id: int,
    *,
    guideline_revision_id: int | None = None,
) -> list[models.GuidelineImpactEvaluation]:
    _project(db, project_id)
    query = db.query(models.GuidelineImpactEvaluation).filter(
        models.GuidelineImpactEvaluation.project_id == project_id
    )
    if guideline_revision_id is not None:
        _scoped(
            db,
            models.GuidelineRevision,
            guideline_revision_id,
            project_id,
            "Guideline revision",
        )
        query = query.filter(
            models.GuidelineImpactEvaluation.guideline_revision_id == guideline_revision_id
        )
    return query.order_by(
        models.GuidelineImpactEvaluation.created_at.desc(),
        models.GuidelineImpactEvaluation.id.desc(),
    ).all()


def complete_guideline_impact(
    db: Session,
    project_id: int,
    evaluation_id: int,
    data: schemas.GuidelineImpactComplete,
    actor: User,
) -> models.GuidelineImpactEvaluation:
    evaluation = _scoped(
        db,
        models.GuidelineImpactEvaluation,
        evaluation_id,
        project_id,
        "Guideline impact evaluation",
    )
    if evaluation.status == "completed":
        raise ConflictError("Guideline impact evaluation is already completed")
    pilot_round = _scoped(
        db,
        models.AnnotationRound,
        evaluation.pilot_round_id,
        project_id,
        "Pilot annotation round",
    )
    if pilot_round.status != "closed":
        raise ConflictError(
            "Guideline impact evaluation can complete only after its pilot round closes"
        )
    submissions = (
        db.query(models.RoundSubmission)
        .filter(models.RoundSubmission.annotation_round_id == pilot_round.id)
        .order_by(
            models.RoundSubmission.annotator_user_id,
            models.RoundSubmission.sequence,
        )
        .all()
    )
    latest_by_annotator = {submission.annotator_user_id: submission for submission in submissions}
    outputs_by_item: dict[int, list[str]] = {}
    for submission in latest_by_annotator.values():
        decisions = (
            db.query(models.RoundAnnotationDecision)
            .filter(models.RoundAnnotationDecision.id.in_(submission.decision_ids))
            .all()
        )
        for decision in decisions:
            superseded = (
                db.query(models.RoundAnnotationDecision.id)
                .filter(models.RoundAnnotationDecision.supersedes_decision_id == decision.id)
                .first()
            )
            if superseded is None:
                outputs_by_item.setdefault(decision.round_item_id, []).append(
                    _canonical_hash(decision.output)
                )
    round_item_ids = {
        item_id
        for (item_id,) in db.query(models.RoundItem.id)
        .filter(models.RoundItem.annotation_round_id == pilot_round.id)
        .all()
    }
    annotated_item_count = len(round_item_ids.intersection(outputs_by_item))
    compared_outputs = [
        outputs_by_item[item_id]
        for item_id in round_item_ids
        if len(outputs_by_item.get(item_id, [])) >= 2
    ]
    unanimous_count = sum(1 for outputs in compared_outputs if len(set(outputs)) == 1)
    item_count = len(round_item_ids)
    coverage = annotated_item_count / item_count if item_count else 0.0
    agreement = unanimous_count / len(compared_outputs) if compared_outputs else 0.0
    passed = (
        item_count > 0
        and annotated_item_count == item_count
        and len(compared_outputs) == item_count
        and agreement >= data.minimum_agreement
    )
    evaluation.status = "completed"
    evaluation.metrics = {
        "item_count": item_count,
        "annotated_item_count": annotated_item_count,
        "compared_item_count": len(compared_outputs),
        "annotator_count": len(latest_by_annotator),
        "coverage": coverage,
        "exact_agreement": agreement,
        "minimum_agreement": data.minimum_agreement,
    }
    evaluation.passed = passed
    evaluation.reviewed_by_user_id = actor.id
    evaluation.completed_at = datetime.now(UTC)
    _commit(db, "Could not complete the guideline impact evaluation")
    db.refresh(evaluation)
    return evaluation
