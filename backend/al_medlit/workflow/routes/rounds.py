"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import NotFoundError
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas, service

from .shared import (
    _read,
    _workspace_annotation_read,
    _write,
)

router = APIRouter(tags=["workflow"])




@router.post(
    "/rounds",
    response_model=schemas.AnnotationRoundRead,
    status_code=status.HTTP_201_CREATED,
)
def create_round(
    payload: schemas.AnnotationRoundCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        module="annotate",
        related_user_ids=payload.annotator_user_ids,
    )
    return service.create_annotation_round(db, payload, current_user)


@router.get("/rounds", response_model=list[schemas.AnnotationRoundRead])
def list_rounds(
    project_id: int = Query(...),
    cycle_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("annotate", "activity"),
    )
    return service.list_annotation_rounds(db, project_id, cycle_id)


@router.get("/rounds/{round_id}", response_model=schemas.AnnotationRoundRead)
def get_round(
    round_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    annotation_round = service.get_annotation_round(db, round_id)
    _read(
        db,
        current_user,
        annotation_round.project_id,
        min_role="manager",
        module="annotate",
    )
    return service.require_annotation_round_actor_access(
        db,
        annotation_round,
        current_user,
    )


@router.get(
    "/rounds/{round_id}/work-context",
    response_model=schemas.RoundWorkContextRead,
)
def get_round_work_context(
    round_id: int,
    workspace_id: int | None = Query(default=None, ge=1),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    annotation_round = service.get_annotation_round(db, round_id)
    if workspace_id is not None:
        project = db.get(Project, annotation_round.project_id)
        if project is None or project.workspace_id != workspace_id:
            raise NotFoundError("Annotation round not found")
    _read(
        db,
        current_user,
        annotation_round.project_id,
        module="annotate",
    )
    return service.get_round_work_context(
        db,
        annotation_round,
        current_user,
    )


@router.get(
    "/workspaces/{workspace_id}/my-work/rounds",
    response_model=list[schemas.RoundWorkContextRead],
)
def list_workspace_round_work_contexts(
    workspace_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _workspace_annotation_read(db, current_user, workspace_id)
    return service.list_workspace_round_work_contexts(
        db,
        workspace_id=workspace_id,
        actor=current_user,
    )


@router.get("/rounds/{round_id}/items", response_model=list[schemas.RoundItemRead])
def list_round_items(
    round_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module="annotate",
    )
    annotation_round = service.get_annotation_round(db, round_id)
    if annotation_round.project_id != project_id:
        raise NotFoundError("Annotation round not found")
    service.require_annotation_round_actor_access(
        db,
        annotation_round,
        current_user,
    )
    return service.list_round_items(db, project_id, round_id)


@router.get(
    "/rounds/{round_id}/work-items",
    response_model=list[schemas.RoundWorkItemRead],
)
def list_round_work_items(
    round_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(db, current_user, project_id, module="annotate")
    annotation_round = service.get_annotation_round(db, round_id)
    if annotation_round.project_id != project_id:
        raise NotFoundError("Annotation round not found")
    service.require_annotation_round_actor_access(
        db,
        annotation_round,
        current_user,
    )
    return [
        {"round_item": round_item, "dataset_item": dataset_item}
        for round_item, dataset_item in service.list_round_work_items(
            db,
            project_id,
            round_id,
        )
    ]


@router.get(
    "/rounds/{round_id}/decisions",
    response_model=list[schemas.AnnotationDecisionRead],
)
def list_round_decisions(
    round_id: int,
    project_id: int = Query(...),
    annotator_user_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(db, current_user, project_id, module="annotate")
    return service.list_round_annotation_decisions(
        db,
        project_id,
        round_id,
        current_user,
        annotator_user_id,
    )


@router.get(
    "/rounds/{round_id}/submissions",
    response_model=list[schemas.RoundSubmissionRead],
)
def list_round_submissions(
    round_id: int,
    project_id: int = Query(...),
    annotator_user_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(db, current_user, project_id, module="annotate")
    return service.list_round_submissions(
        db,
        project_id,
        round_id,
        current_user,
        annotator_user_id,
    )


@router.post("/rounds/{round_id}/transition", response_model=schemas.AnnotationRoundRead)
def transition_round(
    round_id: int,
    payload: schemas.AnnotationRoundTransition,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    related_user_ids: list[int] = []
    if payload.status == "open":
        candidate = (
            db.query(models.AnnotationRound.annotator_user_ids)
            .filter(
                models.AnnotationRound.id == round_id,
                models.AnnotationRound.project_id == project_id,
            )
            .first()
        )
        if candidate is not None:
            related_user_ids = list(candidate[0] or [])
    _write(
        db,
        current_user,
        project_id,
        module="annotate",
        related_user_ids=related_user_ids,
    )
    return service.transition_annotation_round(db, project_id, round_id, payload.status)


@router.post(
    "/rounds/decisions",
    response_model=schemas.AnnotationDecisionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_annotation_decision(
    payload: schemas.AnnotationDecisionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=None,
        module="annotate",
    )
    return service.create_annotation_decision(db, payload, current_user)


@router.post(
    "/rounds/submissions",
    response_model=schemas.RoundSubmissionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_round_submission(
    payload: schemas.RoundSubmissionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=None,
        module="annotate",
    )
    return service.create_round_submission(db, payload, current_user)


@router.post(
    "/feedback-runs/exposures",
    response_model=schemas.FeedbackExposureRead,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback_exposure(
    payload: schemas.FeedbackExposureCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="learning")
    return service.create_feedback_exposure(db, payload, current_user)


@router.post(
    "/rounds/{round_id}/items/{round_item_id}/feedback-reveal",
    response_model=schemas.FeedbackRevealRead,
    status_code=status.HTTP_201_CREATED,
)
def reveal_round_feedback(
    round_id: int,
    round_item_id: int,
    payload: schemas.FeedbackRevealRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=None,
        module=("learning", "annotate"),
    )
    exposure, candidate = service.reveal_round_feedback(
        db,
        project_id=payload.project_id,
        annotation_round_id=round_id,
        round_item_id=round_item_id,
        candidate_key=payload.candidate_key,
        context=payload.context,
        actor=current_user,
    )
    return {"exposure": exposure, "candidate": candidate}


@router.post(
    "/feedback-runs/decisions",
    response_model=schemas.FeedbackDecisionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback_decision(
    payload: schemas.FeedbackDecisionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=None,
        module=("learning", "annotate"),
    )
    return service.create_feedback_decision(db, payload, current_user)


@router.post(
    "/feedback-runs/events",
    response_model=schemas.FeedbackEventRead,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback_event(
    payload: schemas.FeedbackEventCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=None,
        module="learning",
    )
    return service.create_feedback_event(db, payload, current_user)


@router.get(
    "/feedback-runs/events",
    response_model=list[schemas.FeedbackEventRead],
)
def list_feedback_events(
    project_id: int = Query(...),
    annotation_round_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("learning", "activity"),
    )
    return service.list_feedback_events(
        db,
        project_id,
        annotation_round_id=annotation_round_id,
    )


@router.post(
    "/feedback-runs/review-cases",
    response_model=schemas.ReviewCaseRead,
    status_code=status.HTTP_201_CREATED,
)
def create_review_case(
    payload: schemas.ReviewCaseCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="learning")
    return service.create_review_case(db, payload)


@router.get(
    "/feedback-runs/review-cases",
    response_model=list[schemas.ReviewCaseRead],
)
def list_review_cases(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("learning", "activity"),
    )
    return service.list_review_cases(db, project_id)


@router.post(
    "/feedback-runs/review-cases/{review_case_id}/resolve",
    response_model=schemas.ReviewCaseRead,
)
def resolve_review_case(
    review_case_id: int,
    payload: schemas.ReviewCaseResolve,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, project_id, module="learning")
    return service.resolve_review_case(db, project_id, review_case_id, payload)
