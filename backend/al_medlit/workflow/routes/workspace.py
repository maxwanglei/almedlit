"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.workflow import schemas, service

from .shared import (
    _workspace_training_read,
)

router = APIRouter(tags=["workflow"])




@router.get(
    "/workspaces/{workspace_id}/training-contexts",
    response_model=schemas.WorkspaceTrainingContextPageRead,
)
def list_workspace_training_contexts(
    workspace_id: int,
    project_id: int | None = Query(default=None, ge=1),
    cursor: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _workspace_training_read(db, current_user, workspace_id)
    items, next_cursor = service.list_workspace_training_contexts(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        cursor=cursor,
        limit=limit,
    )
    return {"items": items, "next_cursor": next_cursor}


@router.get(
    "/workspaces/{workspace_id}/training-runs",
    response_model=schemas.WorkspaceTrainingRunPageRead,
)
def list_workspace_training_runs(
    workspace_id: int,
    project_id: int | None = Query(default=None, ge=1),
    cursor: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    run_status: str | None = Query(default=None, alias="status"),
    creator_user_id: int | None = Query(default=None, ge=1),
    task_version_id: int | None = Query(default=None, ge=1),
    family: str | None = Query(default=None, min_length=1, max_length=80),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _workspace_training_read(db, current_user, workspace_id)
    items, next_cursor = service.list_workspace_training_runs(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        cursor=cursor,
        limit=limit,
        status=run_status,
        creator_user_id=creator_user_id,
        task_version_id=task_version_id,
        family=family,
    )
    return {"items": items, "next_cursor": next_cursor}


@router.get(
    "/workspaces/{workspace_id}/models",
    response_model=schemas.WorkspaceRegisteredModelPageRead,
)
def list_workspace_registered_models(
    workspace_id: int,
    project_id: int | None = Query(default=None, ge=1),
    cursor: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    lifecycle_status: str | None = Query(default=None, alias="status"),
    creator_user_id: int | None = Query(default=None, ge=1),
    task_version_id: int | None = Query(default=None, ge=1),
    family: str | None = Query(default=None, min_length=1, max_length=80),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _workspace_training_read(db, current_user, workspace_id)
    items, next_cursor = service.list_workspace_registered_models(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        cursor=cursor,
        limit=limit,
        status=lifecycle_status,
        creator_user_id=creator_user_id,
        task_version_id=task_version_id,
        family=family,
    )
    return {"items": items, "next_cursor": next_cursor}
