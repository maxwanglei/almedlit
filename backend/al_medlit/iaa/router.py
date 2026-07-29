from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from al_medlit.annotation.types import AnnotationType
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_document_member, require_project_access
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ValidationError
from al_medlit.corpus.models import Document
from al_medlit.iaa import service
from al_medlit.iaa.schemas import IaaReport
from al_medlit.workspace.capability_dependencies import enforce_capability
from al_medlit.workspace.models import WorkspaceMember

router = APIRouter(prefix="/projects", tags=["iaa"])


@router.get("/{project_id}/iaa", response_model=IaaReport)
def get_project_iaa(
    project_id: int,
    annotation_type: AnnotationType = Query(...),
    document_id: int | None = Query(None),
    target_version_id: int | None = Query(None),
    structure_version_id: int | None = Query(None),
    legacy_structure: bool = Query(
        False,
        description="Select only historical annotations with no structure version",
    ),
    guideline_version_id: int | None = Query(None),
    legacy_guideline: bool = Query(
        False,
        description="Select only historical annotations with no guideline version",
    ),
    _member: WorkspaceMember = Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    enforce_capability(db, project_id=project_id, key="multi_annotator_iaa")
    if legacy_guideline and guideline_version_id is not None:
        raise ValidationError(
            "legacy_guideline and guideline_version_id are mutually exclusive"
        )
    if legacy_structure and structure_version_id is not None:
        raise ValidationError(
            "legacy_structure and structure_version_id are mutually exclusive"
        )
    if document_id is not None:
        assert_document_member(db, current_user, document_id)
        document = db.get(Document, document_id)
        if document is None or document.project_id != project_id:
            raise ValidationError(f"Document {document_id} does not belong to project {project_id}")
    return service.compute_iaa(
        db,
        project_id=project_id,
        annotation_type=annotation_type,
        document_id=document_id,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        filter_structure_version=legacy_structure,
        guideline_version_id=guideline_version_id,
        filter_guideline_version=legacy_guideline,
    )
