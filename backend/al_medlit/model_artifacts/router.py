from __future__ import annotations

import logging
from collections.abc import Iterator
from json import JSONDecodeError
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_project_member, lock_project_member_for_mutation
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ConflictError, ForbiddenError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage, get_object_storage
from al_medlit.model_artifacts import service
from al_medlit.model_artifacts.models import (
    ArtifactPackage,
    ArtifactPackageFile,
    BaseModelAsset,
)
from al_medlit.model_artifacts.schemas import (
    ArtifactPackageArchiveCreate,
    ArtifactPackagePurgeCreate,
    ArtifactPackageRead,
    ArtifactPackageRetentionActionRead,
    ArtifactPackageRetentionUpdate,
    ArtifactStorageUsageRead,
    BaseModelAssetEventRead,
    BaseModelAssetRead,
    BaseModelImportCreate,
    BaseModelReadinessUpdate,
    BaseModelUploadCreate,
    LegacyCheckpointMigrationRead,
)
from al_medlit.training.tasks import enqueue_artifact_garbage_collection
from al_medlit.workspace.capability_dependencies import enforce_capability

router = APIRouter(tags=["model-artifacts"])
logger = logging.getLogger(__name__)


def _manager_access(current_user: User, member) -> bool:
    return current_user.is_superuser or member.role in {"manager", "admin"}


def _package_or_404(db: Session, package_id: int) -> ArtifactPackage:
    package = db.get(ArtifactPackage, package_id)
    if package is None:
        raise NotFoundError("Artifact package not found")
    return package


@router.get(
    "/projects/{project_id}/artifact-packages",
    response_model=list[ArtifactPackageRead],
)
def list_artifact_packages(
    project_id: int,
    readiness: str | None = None,
    model_family: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="training")
    query = db.query(ArtifactPackage).filter(ArtifactPackage.project_id == project_id)
    if readiness:
        query = query.filter(ArtifactPackage.readiness == readiness)
    if model_family:
        query = query.filter(ArtifactPackage.model_family == model_family)
    packages = query.order_by(ArtifactPackage.created_at.desc()).limit(limit).all()
    packages = [
        package
        for package in packages
        if service.artifact_package_visible_to_role(
            db,
            package,
            is_manager=_manager_access(current_user, member),
        )
    ]
    return [service.public_package_descriptor(package) for package in packages]


@router.get("/artifact-packages/{package_id}", response_model=ArtifactPackageRead)
def get_artifact_package(
    package_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    package = _package_or_404(db, package_id)
    member = assert_project_member(
        db, current_user, package.project_id, min_role="trainer"
    )
    enforce_capability(db, project_id=package.project_id, key="training")
    if not service.artifact_package_visible_to_role(
        db,
        package,
        is_manager=_manager_access(current_user, member),
    ):
        raise NotFoundError("Artifact package not found")
    return service.public_package_descriptor(package)


def _parse_range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return None
    if not value.startswith("bytes=") or "," in value:
        raise ValidationError("Only one bytes range is supported")
    start_text, separator, end_text = value[6:].partition("-")
    if not separator:
        raise ValidationError("Invalid Range header")
    try:
        if not start_text:
            suffix = int(end_text)
            if suffix <= 0:
                raise ValueError
            start, end = max(0, size - suffix), size - 1
        else:
            start = int(start_text)
            end = int(end_text) if end_text else size - 1
    except ValueError as exc:
        raise ValidationError("Invalid Range header") from exc
    if start < 0 or end < start or start >= size:
        raise ValidationError("Requested range is outside the package file")
    return start, min(end, size - 1)


def _slice_stream(
    chunks: Iterator[bytes], *, start: int, end: int
) -> Iterator[bytes]:
    offset = 0
    for chunk in chunks:
        chunk_end = offset + len(chunk) - 1
        if chunk_end < start:
            offset += len(chunk)
            continue
        left = max(0, start - offset)
        right = min(len(chunk), end - offset + 1)
        if left < right:
            yield chunk[left:right]
        offset += len(chunk)
        if offset > end:
            break


@router.get("/artifact-packages/{package_id}/files/{file_id}/download")
def download_artifact_package_file(
    package_id: int,
    file_id: int,
    range_header: str | None = Header(default=None, alias="Range"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    package = _package_or_404(db, package_id)
    member = assert_project_member(
        db, current_user, package.project_id, min_role="trainer"
    )
    enforce_capability(db, project_id=package.project_id, key="training")
    is_manager = _manager_access(current_user, member)
    if not service.artifact_package_visible_to_role(
        db,
        package,
        is_manager=is_manager,
    ):
        raise NotFoundError("Artifact package not found")
    if package.retention is None or package.retention.purged_at is not None:
        raise ConflictError("Artifact package payload has been purged")
    access_mode = service.base_model_package_access_mode(db, package)
    if access_mode == "execution_only":
        raise ForbiddenError("This model license permits execution but not file download")
    if access_mode == "manager_only" and not is_manager:
        raise ForbiddenError("This model package is restricted to project managers")
    package_file = db.get(ArtifactPackageFile, file_id)
    if package_file is None or package_file.package_id != package.id:
        raise NotFoundError("Package file not found")
    byte_range = _parse_range(range_header, package_file.size_bytes)
    stream = storage.iter_bytes(package_file.blob.storage_key)
    response_status = status.HTTP_200_OK
    content_length = package_file.size_bytes
    headers = {
        "Accept-Ranges": "bytes",
        "ETag": f'"sha256:{package_file.checksum_sha256}"',
        "X-Checksum-SHA256": package_file.checksum_sha256,
        "Content-Disposition": (
            "attachment; filename*=UTF-8''"
            + quote(package_file.relative_path.rsplit("/", 1)[-1])
        ),
    }
    if byte_range is not None:
        start, end = byte_range
        response_status = status.HTTP_206_PARTIAL_CONTENT
        content_length = end - start + 1
        headers["Content-Range"] = f"bytes {start}-{end}/{package_file.size_bytes}"
        stream = _slice_stream(stream, start=start, end=end)
    headers["Content-Length"] = str(content_length)
    return StreamingResponse(
        stream,
        status_code=response_status,
        media_type=package_file.content_type,
        headers=headers,
    )


@router.get(
    "/projects/{project_id}/artifact-storage-usage",
    response_model=ArtifactStorageUsageRead,
)
def get_artifact_storage_usage(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="training")
    return service.artifact_storage_usage(
        db,
        project_id=project_id,
        include_workspace_physical=(
            current_user.is_superuser or member.role in {"manager", "admin"}
        ),
    )


@router.post(
    "/artifact-packages/{package_id}/archive",
    response_model=ArtifactPackageRetentionActionRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_artifact_package_archive(
    package_id: int,
    payload: ArtifactPackageArchiveCreate | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    package = _package_or_404(db, package_id)
    lock_project_member_for_mutation(
        db, current_user, package.project_id, min_role="manager"
    )
    enforce_capability(db, project_id=package.project_id, key="training")
    retention = service.request_artifact_package_archive(
        db,
        package=package,
        actor_user_id=current_user.id,
        reason=payload.reason if payload else None,
    )
    db.commit()
    return _retention_action_read(package.id, retention)


@router.patch(
    "/artifact-packages/{package_id}/retention",
    response_model=ArtifactPackageRead,
)
def update_artifact_package_retention(
    package_id: int,
    payload: ArtifactPackageRetentionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    package = _package_or_404(db, package_id)
    lock_project_member_for_mutation(
        db, current_user, package.project_id, min_role="manager"
    )
    enforce_capability(db, project_id=package.project_id, key="training")
    service.update_artifact_package_retention(
        db,
        package=package,
        actor_user_id=current_user.id,
        pinned=payload.pinned,
        legal_hold=payload.legal_hold,
    )
    db.commit()
    return service.public_package_descriptor(package)


@router.post(
    "/artifact-packages/{package_id}/purge",
    response_model=ArtifactPackageRetentionActionRead,
    status_code=status.HTTP_202_ACCEPTED,
)
def approve_artifact_package_purge(
    package_id: int,
    payload: ArtifactPackagePurgeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    package = _package_or_404(db, package_id)
    lock_project_member_for_mutation(
        db, current_user, package.project_id, min_role="manager"
    )
    enforce_capability(db, project_id=package.project_id, key="training")
    retention = service.approve_artifact_package_purge(
        db,
        package=package,
        actor_user_id=current_user.id,
        reason=payload.reason,
    )
    db.commit()
    try:
        enqueue_artifact_garbage_collection(package.workspace_id)
    except Exception:  # pragma: no cover - broker outage is operationally retried
        logger.exception(
            "Artifact package %s was tombstoned but GC enqueue failed",
            package.id,
        )
    return _retention_action_read(package.id, retention)


def _retention_action_read(package_id: int, retention) -> dict:
    return {
        "package_id": package_id,
        "archived_at": retention.archived_at,
        "archived_by_user_id": retention.archived_by_user_id,
        "archive_reason": retention.archive_reason,
        "purge_after": retention.purge_after,
        "purged_at": retention.purged_at,
        "purged_by_user_id": retention.purged_by_user_id,
        "purge_reason": retention.purge_reason,
    }


@router.get(
    "/projects/{project_id}/base-models",
    response_model=list[BaseModelAssetRead],
)
def list_base_models(
    project_id: int,
    readiness: str | None = None,
    include_archived: bool = False,
    limit: int = Query(default=200, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_project_member(db, current_user, project_id, min_role="trainer")
    enforce_capability(db, project_id=project_id, key="training")
    is_manager = _manager_access(current_user, member)
    effective_readiness = readiness if is_manager else "ready"
    assets = service.list_base_model_assets(
        db,
        project_id=project_id,
        include_archived=include_archived and is_manager,
        include_manager_only=is_manager,
        readiness=effective_readiness,
        limit=limit,
    )
    return [service.public_base_model_descriptor(asset) for asset in assets]


@router.get("/base-models/{asset_id}", response_model=BaseModelAssetRead)
def get_base_model(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None:
        raise NotFoundError("Base model asset not found")
    member = assert_project_member(
        db,
        current_user,
        asset.project_id,
        min_role="trainer",
    )
    enforce_capability(db, project_id=asset.project_id, key="training")
    if not _manager_access(current_user, member) and (
        asset.access_mode == "manager_only"
        or asset.state is None
        or asset.state.readiness != "ready"
        or asset.state.archived_at is not None
    ):
        raise NotFoundError("Base model asset not found")
    return service.public_base_model_descriptor(asset)


@router.get(
    "/base-models/{asset_id}/events",
    response_model=list[BaseModelAssetEventRead],
)
def get_base_model_events(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None:
        raise NotFoundError("Base model asset not found")
    assert_project_member(db, current_user, asset.project_id, min_role="manager")
    enforce_capability(db, project_id=asset.project_id, key="training")
    return service.list_base_model_events(asset)


@router.post(
    "/projects/{project_id}/base-models/import",
    response_model=BaseModelAssetRead,
    status_code=status.HTTP_201_CREATED,
)
def import_base_model(
    project_id: int,
    data: BaseModelImportCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    enforce_capability(db, project_id=project_id, key="training")
    asset = service.import_base_model_from_package(
        db,
        storage,
        project_id=project_id,
        data=data,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(asset)
    return service.public_base_model_descriptor(asset)


@router.post(
    "/projects/{project_id}/base-models/upload",
    response_model=BaseModelAssetRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_base_model(
    project_id: int,
    metadata: str = Form(...),
    files: list[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    enforce_capability(db, project_id=project_id, key="training")
    try:
        data = BaseModelUploadCreate.model_validate_json(metadata)
    except (PydanticValidationError, JSONDecodeError, ValueError) as exc:
        raise ValidationError("Invalid base-model upload metadata") from exc
    try:
        asset = service.publish_uploaded_base_model(
            db,
            storage,
            project_id=project_id,
            data=data,
            sources=[item.file for item in files],
            actor_user_id=current_user.id,
        )
        db.commit()
        db.refresh(asset)
        return service.public_base_model_descriptor(asset)
    finally:
        for item in files:
            await item.close()


@router.patch("/base-models/{asset_id}/readiness", response_model=BaseModelAssetRead)
def set_base_model_readiness(
    asset_id: int,
    data: BaseModelReadinessUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None:
        raise NotFoundError("Base model asset not found")
    lock_project_member_for_mutation(
        db,
        current_user,
        asset.project_id,
        min_role="manager",
    )
    enforce_capability(db, project_id=asset.project_id, key="training")
    service.update_base_model_readiness(
        db,
        asset=asset,
        data=data,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(asset.state)
    return service.public_base_model_descriptor(asset)


@router.post("/base-models/{asset_id}/archive", response_model=BaseModelAssetRead)
def archive_base_model(
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None:
        raise NotFoundError("Base model asset not found")
    lock_project_member_for_mutation(
        db,
        current_user,
        asset.project_id,
        min_role="manager",
    )
    enforce_capability(db, project_id=asset.project_id, key="training")
    service.archive_base_model_asset(
        db,
        asset=asset,
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(asset.state)
    return service.public_base_model_descriptor(asset)


@router.post(
    "/projects/{project_id}/legacy-checkpoints/migrate",
    response_model=LegacyCheckpointMigrationRead,
)
def migrate_legacy_checkpoint_packages(
    project_id: int,
    checkpoint_id: int | None = Query(default=None, gt=0),
    limit: int = Query(default=100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    enforce_capability(db, project_id=project_id, key="training")
    result = service.migrate_legacy_checkpoints(
        db,
        storage,
        project_id=project_id,
        checkpoint_id=checkpoint_id,
        limit=limit,
        actor_user_id=current_user.id,
    )
    db.commit()
    return result
