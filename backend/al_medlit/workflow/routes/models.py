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
    "/models",
    response_model=schemas.RegisteredModelRead,
    status_code=status.HTTP_201_CREATED,
)
def create_registered_model(
    payload: schemas.RegisteredModelCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module=("models", "train"),
    )
    return service.create_registered_model(db, payload, current_user)


@router.get("/models", response_model=list[schemas.RegisteredModelRead])
def list_registered_models(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("models", "train", "learning"),
    )
    return service.list_registered_models(db, project_id)


@router.get(
    "/models/versions",
    response_model=list[schemas.WorkspaceModelVersionRead],
)
def list_model_versions(
    project_id: int = Query(...),
    registered_model_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("models", "train", "learning"),
    )
    return service.list_model_versions(db, project_id, registered_model_id)


@router.get(
    "/models/{registered_model_id}/versions/{model_version_id}/evaluations",
    response_model=list[schemas.ModelEvaluationRead],
)
def list_model_version_evaluations(
    registered_model_id: int,
    model_version_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("models", "train", "learning"),
    )
    return service.list_model_version_evaluations(
        db,
        project_id,
        registered_model_id,
        model_version_id,
    )
