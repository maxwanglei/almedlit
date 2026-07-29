from collections.abc import Iterator

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import require_project_access
from al_medlit.core.database import get_db
from al_medlit.importers import service
from al_medlit.importers.pubmed import ImporterFetchError
from al_medlit.importers.schemas import (
    ImportPreviewRequest,
    ImportPreviewResponse,
    ImportRequest,
    ImportResponse,
)
from al_medlit.project import service as project_service

router = APIRouter(prefix="/projects/{project_id}/import", tags=["import"])


def get_ncbi_client() -> Iterator[httpx.Client]:
    """Provide an httpx client for NCBI calls; overridden in tests."""
    with httpx.Client() as client:
        yield client


@router.post("/pubmed/preview", response_model=ImportPreviewResponse)
def preview_pubmed_import(
    project_id: int,
    payload: ImportPreviewRequest,
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
    client: httpx.Client = Depends(get_ncbi_client),
) -> ImportPreviewResponse:
    service.ensure_project(db, project_id)
    try:
        items = service.preview_import(client, payload.pmids)
    except ImporterFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ImportPreviewResponse(items=items)


@router.post("/pubmed", response_model=ImportResponse)
def run_pubmed_import(
    project_id: int,
    payload: ImportRequest,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client: httpx.Client = Depends(get_ncbi_client),
) -> ImportResponse:
    service.ensure_project(db, project_id)
    try:
        response = service.run_import(
            db,
            client,
            project_id,
            payload.pmids,
            include_abstract_only=payload.include_abstract_only,
        )
    except ImporterFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    project_service.ensure_personal_project_self_assignments(
        db,
        project_id,
        current_user,
        document_ids=[
            item.document_id for item in response.created if item.document_id is not None
        ],
        assigned_by_user=current_user,
    )
    return response
