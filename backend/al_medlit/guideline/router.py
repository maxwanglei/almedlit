from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_project_member_if_exists,
    require_guideline_access,
)
from al_medlit.core.database import get_db
from al_medlit.guideline import service
from al_medlit.guideline.schemas import GuidelineVersionCreate, GuidelineVersionRead
from al_medlit.project import service as project_service

router = APIRouter(prefix="/guidelines", tags=["guidelines"])


@router.post("", response_model=GuidelineVersionRead)
def create_guideline_version(
    payload: GuidelineVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member_if_exists(
        db,
        current_user,
        payload.project_id,
        min_role="manager",
    )
    payload.author_id = current_user.username
    guideline = service.create_guideline_version(db, payload)
    if guideline.status == "active":
        project_service.ensure_personal_project_self_assignments(
            db,
            payload.project_id,
            current_user,
            assigned_by_user=current_user,
        )
    return guideline


@router.get("", response_model=list[GuidelineVersionRead])
def list_guideline_versions(
    project_id: int = Query(...),
    _member=Depends(require_guideline_access()),
    db: Session = Depends(get_db),
):
    return service.list_guideline_versions(db, project_id)
