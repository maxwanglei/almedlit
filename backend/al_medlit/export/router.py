from unicodedata import category
from urllib.parse import quote

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_project_member, lock_project_member_for_mutation
from al_medlit.core.database import get_db
from al_medlit.core.storage import ObjectStorage, get_object_storage
from al_medlit.export import service
from al_medlit.export.registry import export_formats
from al_medlit.export.schemas import ExportCreate, ExportFormatRead
from al_medlit.lineage.models import CorpusSnapshotDocument
from al_medlit.lineage.schemas import ExportArtifactRead
from al_medlit.workspace.capability_dependencies import enforce_capability

router = APIRouter(tags=["exports"])


def _download_content_disposition(file_name: str) -> str:
    leaf = file_name.replace("\\", "/").rsplit("/", 1)[-1]
    leaf = "".join("_" if category(char) in {"Cc", "Cs"} else char for char in leaf)
    if not leaf.strip() or leaf in {".", ".."}:
        leaf = "export"
    fallback = "".join(
        char if " " <= char <= "~" and char not in {'"', "%"} else "_" for char in leaf
    )
    return (
        f'attachment; filename="{fallback}"; '
        f"filename*=UTF-8''{quote(leaf, safe='')}"
    )


@router.get("/export-formats", response_model=list[ExportFormatRead])
def list_export_formats():
    service.register_builtin_export_formats()
    return [
        {
            "key": plugin.key,
            "content_type": plugin.content_type,
            "extension": plugin.extension,
        }
        for plugin in export_formats.list()
    ]


@router.post("/projects/{project_id}/exports", response_model=ExportArtifactRead)
def create_export(
    project_id: int,
    payload: ExportCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    lock_project_member_for_mutation(
        db, current_user, project_id, min_role="manager", lock_workspace=True
    )
    enforce_capability(db, project_id=project_id, key="export")
    return service.create_export(
        db,
        storage,
        project_id=project_id,
        data=payload,
        actor_user_id=current_user.id,
    )


@router.get("/projects/{project_id}/exports", response_model=list[ExportArtifactRead])
def list_exports(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="export")
    return service.list_exports(db, project_id)


@router.get("/exports/{export_id}", response_model=ExportArtifactRead)
def get_export(
    export_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    export = service.get_export(db, export_id)
    assert_project_member(db, current_user, export.project_id, min_role="trainer")
    enforce_capability(db, project_id=export.project_id, key="export")
    return export


@router.get("/exports/{export_id}/download")
def download_export(
    export_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    export = service.get_export(db, export_id)
    assert_project_member(db, current_user, export.project_id, min_role="trainer")
    enforce_capability(db, project_id=export.project_id, key="export")
    if db.query(CorpusSnapshotDocument.id).filter(
        CorpusSnapshotDocument.snapshot_id == export.corpus_snapshot_id,
        CorpusSnapshotDocument.split == "test",
    ).first() is not None:
        assert_project_member(db, current_user, export.project_id, min_role="manager")
    service.verify_export(storage, export)
    return StreamingResponse(
        storage.iter_bytes(export.artifact.storage_key),
        media_type=export.artifact.content_type,
        headers={
            "Content-Disposition": _download_content_disposition(export.file_name),
            "X-Checksum-SHA256": export.artifact.content_hash,
        },
    )
