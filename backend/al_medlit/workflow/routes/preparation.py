"""Project-scoped source-to-training preparation, independent of annotation rounds."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.workflow import preparation_schemas as schemas
from al_medlit.workflow.services import preparation as service

from .shared import _read, _write

router = APIRouter(prefix="/projects/{project_id}/training-datasets", tags=["training data"])


@router.get("", response_model=list[schemas.TrainingDatasetSeriesRead])
def list_training_datasets(
    project_id: int, db: Session = Depends(get_db), actor: User = Depends(get_current_user)
):
    _read(db, actor, project_id, min_role="trainer", module="data")
    return service.list_training_datasets(db, project_id)


@router.post("/preview", response_model=schemas.TrainingPreparationPreview)
def preview_training_dataset(
    project_id: int,
    payload: schemas.TrainingPreparationRequest,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_user),
):
    _read(db, actor, project_id, min_role="trainer", module="data")
    _read(db, actor, project_id, min_role="trainer", module="train")
    if any(source.annotation_round_id is not None for source in payload.sources):
        _read(db, actor, project_id, min_role="manager", module="annotate")
    return service.preview_training_preparation(db, project_id, payload)


@router.post(
    "/prepare",
    response_model=schemas.TrainingPreparationResult,
    status_code=status.HTTP_201_CREATED,
)
def prepare_training_dataset(
    project_id: int,
    payload: schemas.TrainingPreparationRequest,
    db: Session = Depends(get_db),
    actor: User = Depends(get_current_user),
):
    _write(db, actor, project_id, min_role="trainer", module="data")
    _read(db, actor, project_id, min_role="trainer", module="train")
    if any(source.annotation_round_id is not None for source in payload.sources):
        _read(db, actor, project_id, min_role="manager", module="annotate")
    return service.prepare_training_dataset(db, project_id, payload, actor)
