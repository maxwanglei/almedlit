"""Read-only annotation progress for the dataset registry."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.workflow.services.source_progress import annotation_progress

from .shared import _read

router = APIRouter(tags=["datasets"])


class SourceAnnotationProgress(BaseModel):
    dataset_id: int
    dataset_version_id: int
    total: int
    submitted: int


@router.get(
    "/projects/{project_id}/datasets/annotation-progress",
    response_model=list[SourceAnnotationProgress],
)
def get_annotation_progress(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(db, current_user, project_id, min_role="trainer", module="data")
    return annotation_progress(db, project_id)
