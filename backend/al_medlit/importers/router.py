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
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    client: httpx.Client = Depends(get_ncbi_client),
) -> ImportPreviewResponse:
    actor = service.ImportActor(current_user.id, current_user.session_version)
    normalized = service.normalize_pmids(payload.pmids)
    existing = service.project_documents_by_pmid(db, project_id, normalized)
    db.rollback()
    try:
        items = service.preview_import(client, normalized, existing=existing)
    except ImporterFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    service.revalidate_import_actor(db, project_id, actor)
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
    actor = service.ImportActor(current_user.id, current_user.session_version)
    normalized = service.normalize_pmids(payload.pmids)
    existing = service.project_documents_by_pmid(db, project_id, normalized)
    db.rollback()
    try:
        fetched = service.fetch_import(client, normalized, existing=existing)
    except ImporterFetchError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    with db.begin():
        return service.run_import(
            db,
            project_id,
            fetched,
            include_abstract_only=payload.include_abstract_only,
            actor=actor,
        )
