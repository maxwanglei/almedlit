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
    "/cycles",
    response_model=schemas.LearningCycleRead,
    status_code=status.HTTP_201_CREATED,
)
def create_cycle(
    payload: schemas.LearningCycleCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="learning",
    )
    return service.create_learning_cycle(db, payload, current_user)


@router.get("/cycles", response_model=list[schemas.LearningCycleRead])
def list_cycles(
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
    return service.list_learning_cycles(db, project_id)


@router.post("/cycles/{cycle_id}/transition", response_model=schemas.LearningCycleRead)
def transition_cycle(
    cycle_id: int,
    payload: schemas.CycleTransition,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="learning",
    )
    return service.transition_learning_cycle(db, project_id, cycle_id, payload)


@router.post(
    "/selection-runs",
    response_model=schemas.SelectionRunRead,
    status_code=status.HTTP_201_CREATED,
)
def create_selection_run(
    payload: schemas.SelectionRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="learning",
    )
    return service.create_selection_run(db, payload, current_user)


@router.get("/selection-runs", response_model=list[schemas.SelectionRunRead])
def list_selection_runs(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module="learning",
    )
    return service.list_selection_runs(db, project_id)


@router.post(
    "/projects/{project_id}/selection-runs/{selection_run_id}/materialize",
    response_model=schemas.SelectionSetRead,
    status_code=status.HTTP_201_CREATED,
)
def materialize_selection_run(
    project_id: int,
    selection_run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="learning",
    )
    return service.materialize_selection_run(
        db,
        project_id=project_id,
        selection_run_id=selection_run_id,
        actor=current_user,
    )


@router.post(
    "/selection-runs/sets",
    response_model=schemas.SelectionSetRead,
    status_code=status.HTTP_201_CREATED,
)
def create_selection_set(
    payload: schemas.SelectionSetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="learning",
    )
    return service.create_selection_set(db, payload, current_user)


@router.post(
    "/feedback-runs",
    response_model=schemas.FeedbackRunRead,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback_run(
    payload: schemas.FeedbackRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    minimum_role = "manager" if payload.producer_type == "external_llm" else "trainer"
    _write(
        db,
        current_user,
        payload.project_id,
        min_role=minimum_role,
        module="learning",
    )
    return service.create_feedback_run(db, payload, current_user)


@router.get("/feedback-runs", response_model=list[schemas.FeedbackRunRead])
def list_feedback_runs(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("learning", "annotate"),
    )
    return service.list_feedback_runs(db, project_id)


@router.post(
    "/feedback-runs/sets",
    response_model=schemas.FeedbackSetRead,
    status_code=status.HTTP_201_CREATED,
)
def create_feedback_set(
    payload: schemas.FeedbackSetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="learning",
    )
    return service.create_feedback_set(db, payload, current_user)


@router.get("/feedback-runs/sets", response_model=list[schemas.FeedbackSetRead])
def list_feedback_sets(
    project_id: int = Query(...),
    feedback_run_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module=("learning", "annotate"),
    )
    return service.list_feedback_sets(db, project_id, feedback_run_id)


@router.get(
    "/feedback-runs/candidates",
    response_model=list[schemas.FeedbackCandidateRead],
)
def list_feedback_candidates(
    project_id: int = Query(...),
    feedback_set_version_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="manager",
        module="learning",
    )
    return service.list_feedback_candidates(db, project_id, feedback_set_version_id)
