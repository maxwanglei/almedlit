"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_project_member, lock_project_member_for_mutation
from al_medlit.core.database import get_db
from al_medlit.workflow import access, schemas

router = APIRouter(tags=["workflow"])



@router.get(
    "/projects/{project_id}/modules",
    response_model=schemas.ProjectModulesRead,
)
def get_project_modules(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(
        db,
        current_user,
        project_id,
        min_role="trainer",
    )
    return access.module_configuration(db, project_id)


@router.patch(
    "/projects/{project_id}/modules",
    response_model=schemas.ProjectModulesRead,
)
def update_project_modules(
    project_id: int,
    payload: schemas.ProjectModulesUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    return access.update_project_modules(db, project_id, payload.selected)
