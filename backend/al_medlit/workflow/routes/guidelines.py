"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.workflow import schemas, service

from .shared import (
    _read,
    _write,
)

router = APIRouter(tags=["workflow"])




@router.post(
    "/workflow-guidelines",
    response_model=schemas.GuidelineRead,
    status_code=status.HTTP_201_CREATED,
)
def create_guideline(
    payload: schemas.GuidelineCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="guidelines")
    return service.create_guideline(db, payload, current_user)


@router.get("/workflow-guidelines", response_model=list[schemas.GuidelineRead])
def list_guidelines(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("guidelines", "activity"),
    )
    return service.list_guidelines(db, project_id)


@router.post(
    "/workflow-guidelines/revisions",
    response_model=schemas.GuidelineRevisionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_guideline_revision(
    payload: schemas.GuidelineRevisionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="guidelines")
    return service.create_guideline_revision(db, payload, current_user)


@router.get(
    "/workflow-guidelines/revisions",
    response_model=list[schemas.GuidelineRevisionRead],
)
def list_guideline_revisions(
    project_id: int = Query(...),
    guideline_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("guidelines", "activity"),
    )
    return service.list_guideline_revisions(db, project_id, guideline_id)


@router.post(
    "/workflow-guidelines/revisions/{revision_id}/transition",
    response_model=schemas.GuidelineRevisionRead,
)
def transition_guideline_revision(
    revision_id: int,
    payload: schemas.GuidelineRevisionTransition,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, project_id, module="guidelines")
    return service.transition_guideline_revision(db, project_id, revision_id, payload, current_user)


@router.post(
    "/workflow-guidelines/proposals",
    response_model=schemas.GuidelineProposalRead,
    status_code=status.HTTP_201_CREATED,
)
def create_guideline_proposal(
    payload: schemas.GuidelineProposalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="guidelines")
    return service.create_guideline_proposal(db, payload, current_user)


@router.get(
    "/workflow-guidelines/proposals",
    response_model=list[schemas.GuidelineProposalRead],
)
def list_guideline_proposals(
    project_id: int = Query(...),
    guideline_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("guidelines", "activity"),
    )
    return service.list_guideline_proposals(
        db,
        project_id,
        guideline_id=guideline_id,
    )


@router.post(
    "/workflow-guidelines/proposals/{proposal_id}/review",
    response_model=schemas.GuidelineProposalRead,
)
def review_guideline_proposal(
    proposal_id: int,
    payload: schemas.GuidelineProposalReview,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, project_id, module="guidelines")
    return service.review_guideline_proposal(db, project_id, proposal_id, payload, current_user)


@router.post(
    "/workflow-guidelines/impact-evaluations",
    response_model=schemas.GuidelineImpactRead,
    status_code=status.HTTP_201_CREATED,
)
def create_guideline_impact(
    payload: schemas.GuidelineImpactCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="guidelines")
    return service.create_guideline_impact(db, payload, current_user)


@router.get(
    "/workflow-guidelines/impact-evaluations",
    response_model=list[schemas.GuidelineImpactRead],
)
def list_guideline_impacts(
    project_id: int = Query(...),
    guideline_revision_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("guidelines", "activity"),
    )
    return service.list_guideline_impacts(
        db,
        project_id,
        guideline_revision_id=guideline_revision_id,
    )


@router.post(
    "/workflow-guidelines/impact-evaluations/{evaluation_id}/complete",
    response_model=schemas.GuidelineImpactRead,
)
def complete_guideline_impact(
    evaluation_id: int,
    payload: schemas.GuidelineImpactComplete,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, project_id, module="guidelines")
    return service.complete_guideline_impact(db, project_id, evaluation_id, payload, current_user)
