from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from al_medlit.annotation_workbench import service
from al_medlit.annotation_workbench.schemas import AnnotationWorkbenchRead
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_document_member
from al_medlit.core.database import get_db

router = APIRouter(prefix="/annotation-workbench", tags=["annotation-workbench"])


@router.get("/documents/{document_id}", response_model=AnnotationWorkbenchRead)
def get_annotation_workbench(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_document_member(db, current_user, document_id)
    return service.get_annotation_workbench(
        db,
        document_id,
        current_user=current_user,
        member=member,
    )
