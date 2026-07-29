from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_document_member,
    assert_global_manager_scope,
    assert_project_member,
    assert_project_member_if_exists,
    require_assigned_document_access,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError
from al_medlit.corpus import service
from al_medlit.corpus.schemas import (
    DocumentCreate,
    DocumentRead,
    DocumentStructureRead,
    StructureRebuildRequest,
)
from al_medlit.project import service as project_service
from al_medlit.workspace.dependencies import ROLE_RANK

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentRead)
def create_document(
    payload: DocumentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member_if_exists(db, current_user, payload.project_id)
    document = service.create_document(db, payload)
    project_service.ensure_personal_project_self_assignments(
        db,
        payload.project_id,
        current_user,
        document_ids=[document.id],
        assigned_by_user=current_user,
    )
    return document


@router.get("", response_model=list[DocumentRead])
def list_documents(
    project_id: int | None = Query(None),
    scope: str | None = Query(None, pattern="^(mine|all)$"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assignee_user_id = None
    if project_id is not None:
        member = assert_project_member(db, current_user, project_id)
        if scope == "mine" or (
            scope is None and ROLE_RANK.get(member.role, -1) < ROLE_RANK["manager"]
        ):
            assignee_user_id = current_user.id
        elif scope == "all" and ROLE_RANK.get(member.role, -1) < ROLE_RANK["manager"]:
            raise ForbiddenError("Insufficient workspace role to view all documents")
    elif scope == "mine":
        assignee_user_id = current_user.id
    elif scope == "all":
        assert_global_manager_scope(db, current_user)
    return service.list_documents(
        db,
        project_id,
        user=current_user,
        assignee_user_id=assignee_user_id,
    )


@router.get("/{document_id}", response_model=DocumentRead)
def get_document(
    document_id: int,
    _member=Depends(require_assigned_document_access()),
    db: Session = Depends(get_db),
):
    doc = service.get_document(db, document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/structure", response_model=DocumentStructureRead)
def get_document_structure(
    document_id: int,
    version_id: int | None = Query(None),
    sentence_start: int = Query(0, ge=0),
    sentence_limit: int = Query(500, ge=1, le=2000),
    _member=Depends(require_assigned_document_access()),
    db: Session = Depends(get_db),
):
    return service.get_document_structure(
        db,
        document_id,
        structure_version_id=version_id,
        sentence_start=sentence_start,
        sentence_limit=sentence_limit,
    )


@router.post(
    "/{document_id}/structure/rebuild",
    response_model=DocumentStructureRead,
)
def rebuild_document_structure(
    document_id: int,
    payload: StructureRebuildRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_document_member(db, current_user, document_id, min_role="manager")
    structure_version = service.rebuild_document_structure(
        db,
        document_id,
        activate=payload.activate if payload is not None else True,
    )
    if payload is None or payload.activate:
        document = service.get_document(db, document_id)
        project_service.ensure_personal_project_self_assignments(
            db,
            document.project_id,
            current_user,
            document_ids=[document_id],
            assigned_by_user=current_user,
        )
    return service.get_document_structure(
        db,
        document_id,
        structure_version_id=structure_version.id,
    )


@router.post(
    "/{document_id}/structure/{structure_version_id}/activate",
    response_model=DocumentStructureRead,
)
def activate_document_structure(
    document_id: int,
    structure_version_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_document_member(db, current_user, document_id, min_role="manager")
    structure_version = service.activate_document_structure(
        db,
        document_id,
        structure_version_id,
    )
    document = service.get_document(db, document_id)
    project_service.ensure_personal_project_self_assignments(
        db,
        document.project_id,
        current_user,
        document_ids=[document_id],
        assigned_by_user=current_user,
    )
    return service.get_document_structure(
        db,
        document_id,
        structure_version_id=structure_version.id,
    )
