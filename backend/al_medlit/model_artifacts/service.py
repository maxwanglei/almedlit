from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sqlalchemy import or_, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectNotFoundError, ObjectStorage, StoredObject
from al_medlit.lineage.models import LineageArtifact
from al_medlit.lineage.service import add_lineage_edge, register_stored_artifact
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactPackageFile,
    ArtifactPackageReference,
    ArtifactPackageRetention,
    BaseModelAsset,
    BaseModelAssetEvent,
    BaseModelAssetState,
)
from al_medlit.model_artifacts.schemas import (
    ArtifactPackageCreate,
    ArtifactPackageFileRead,
    ArtifactPackageRead,
    ArtifactPackageReferenceCreate,
    ArtifactPackageReferenceRead,
    ArtifactPackageRetentionRead,
    ArtifactStorageUsageRead,
    BaseModelAssetEventRead,
    BaseModelAssetRead,
    BaseModelCatalogFields,
    BaseModelImportCreate,
    BaseModelReadinessUpdate,
    BaseModelUploadCreate,
    LegacyCheckpointMigrationItemRead,
    LegacyCheckpointMigrationRead,
)
from al_medlit.project.models import Project

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_PACKAGE_FILES = 10_000
MAX_PACKAGE_FILE_BYTES = 16 * 1024 * 1024 * 1024
MAX_PACKAGE_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_RELATIVE_PATH_LENGTH = 1024
MAX_JSON_INSPECTION_BYTES = 5 * 1024 * 1024
MAX_MODEL_METADATA_JSON_BYTES = 128 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 5 * 1024 * 1024
DEFAULT_COPY_CHUNK_SIZE = 1024 * 1024
_ACCESS_MODE_RANK = {
    "downloadable": 0,
    "execution_only": 1,
    "manager_only": 2,
}

_FORBIDDEN_SUFFIXES = {
    ".7z",
    ".bat",
    ".bin",
    ".bz2",
    ".cmd",
    ".dll",
    ".dylib",
    ".exe",
    ".gz",
    ".jar",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ps1",
    ".py",
    ".pyc",
    ".rar",
    ".sh",
    ".so",
    ".tar",
    ".tgz",
    ".whl",
    ".xz",
}
_SENSITIVE_KEY_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}
_TOKENIZER_PAYLOAD_NAMES = {
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
}


@dataclass(frozen=True, slots=True)
class PackageFileUpload:
    relative_path: str
    source: bytes | bytearray | BinaryIO | str | Path
    role: str = "model_file"
    content_type: str = "application/octet-stream"
    expected_checksum_sha256: str | None = None
    expected_size_bytes: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ArtifactGarbageCollectionResult:
    scanned_blob_count: int
    deleted_blob_count: int
    reclaimed_bytes: int
    last_scanned_blob_id: int | None


@dataclass(frozen=True, slots=True)
class PurgedPackagePayloadCollectionResult:
    processed_package_count: int
    last_scanned_retention_id: int | None


@dataclass(slots=True)
class _PreparedFile:
    relative_path: str
    role: str
    content_type: str
    checksum_sha256: str
    size_bytes: int
    metadata: dict
    spool: BinaryIO
    json_value: Any | None = None

    def close(self) -> None:
        self.spool.close()


@dataclass(frozen=True, slots=True)
class _SafetensorsFile:
    tensor_names: frozenset[str]
    data_size_bytes: int


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError("Package metadata must be canonical JSON") from exc


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_RELATIVE_PATH_LENGTH:
        raise ValidationError("Package file path must contain 1 to 1024 characters")
    if "\\" in value or "\x00" in value or value.startswith("/") or value.endswith("/"):
        raise ValidationError(f"Invalid package file path: {value!r}")
    path = PurePosixPath(value)
    if (
        not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != value
    ):
        raise ValidationError(f"Package file path is not canonical: {value!r}")
    if any(len(part) > 255 for part in path.parts):
        raise ValidationError(f"Package file path segment is too long: {value!r}")
    return value


def workspace_blob_prefix(workspace_id: int) -> str:
    if workspace_id <= 0:
        raise ValidationError("A valid workspace is required")
    return f"workspaces/{workspace_id}/artifact-blobs"


def blob_storage_key(workspace_id: int, checksum_sha256: str) -> str:
    if not SHA256_RE.fullmatch(checksum_sha256):
        raise ValidationError("A valid workspace and lowercase SHA-256 digest are required")
    return (
        f"{workspace_blob_prefix(workspace_id)}/sha256/"
        f"{checksum_sha256[:2]}/{checksum_sha256}"
    )


def publish_artifact_package(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: ArtifactPackageCreate,
    files: Sequence[PackageFileUpload],
    actor_user_id: int | None,
    upstream_lineage_artifact_ids: Iterable[int] = (),
    verify_reused_blobs: bool = True,
    prepared_validator: Callable[[Sequence[_PreparedFile]], None] | None = None,
    reservation_id: int | None = None,
) -> ArtifactPackage:
    """Publish all files and their manifest, leaving commit control to the caller.

    Object writes necessarily precede the database transaction. If publication fails,
    database references are rolled back and any newly written objects are safe orphan
    candidates because their keys are content addressed.
    """

    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    prepared = _prepare_files(files)
    try:
        upstream_ids = sorted(set(upstream_lineage_artifact_ids))
        upstream_artifacts = _validate_upstream_artifacts(db, project_id, upstream_ids)
        references = _resolve_references(db, project, data)
        _validate_package_policy(data, prepared, references)
        if prepared_validator is not None:
            prepared_validator(prepared)
        _ensure_no_sensitive_keys(
            {
                "task_contract": data.task_contract,
                "license": data.license_info,
                "runtime": data.runtime,
                "metadata": data.metadata,
                "references": [reference.metadata for reference in data.references],
                "files": [item.metadata for item in prepared],
            }
        )
        task_contract_hash = hashlib.sha256(
            canonical_json_bytes(data.task_contract)
        ).hexdigest()
        manifest = _build_manifest(
            project=project,
            data=data,
            prepared=prepared,
            references=references,
            upstream_ids=upstream_ids,
            task_contract_hash=task_contract_hash,
        )
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        existing_package = (
            db.query(ArtifactPackage)
            .filter(
                ArtifactPackage.project_id == project_id,
                ArtifactPackage.manifest_digest == manifest_digest,
            )
            .first()
        )
        if existing_package is not None:
            if (
                existing_package.retention is None
                or existing_package.retention.archived_at is not None
                or existing_package.retention.purged_at is not None
            ):
                raise ConflictError(
                    "An archived or purged package identity cannot be republished in place"
                )
            if verify_reused_blobs:
                verify_artifact_package(
                    db,
                    storage,
                    project_id=project_id,
                    package_id=existing_package.id,
                )
            if reservation_id is not None:
                with db.begin_nested():
                    artifact_quota.admit_artifact_publication(
                        db,
                        workspace_id=project.workspace_id,
                        project_id=project.id,
                        candidates=[
                            *(
                                (item.checksum_sha256, item.size_bytes)
                                for item in prepared
                            ),
                            (manifest_digest, len(manifest_bytes)),
                        ],
                        reservation_id=reservation_id,
                    )
                    artifact_quota.link_reservation_artifact_package(
                        db,
                        reservation_id=reservation_id,
                        artifact_package_id=existing_package.id,
                    )
            return existing_package

        with db.begin_nested():
            artifact_quota.admit_artifact_publication(
                db,
                workspace_id=project.workspace_id,
                project_id=project.id,
                candidates=[
                    *((item.checksum_sha256, item.size_bytes) for item in prepared),
                    (manifest_digest, len(manifest_bytes)),
                ],
                reservation_id=reservation_id,
            )
            blob_cache: dict[str, ArtifactBlob] = {}
            file_rows: list[tuple[_PreparedFile, ArtifactBlob]] = []
            for item in prepared:
                blob = blob_cache.get(item.checksum_sha256)
                if blob is None:
                    item.spool.seek(0)
                    blob = _ensure_blob(
                        db,
                        storage,
                        workspace_id=project.workspace_id,
                        checksum_sha256=item.checksum_sha256,
                        size_bytes=item.size_bytes,
                        content_type=item.content_type,
                        source=item.spool,
                        actor_user_id=actor_user_id,
                        verify_reused=verify_reused_blobs,
                    )
                    blob_cache[item.checksum_sha256] = blob
                file_rows.append((item, blob))

            manifest_blob = _ensure_blob(
                db,
                storage,
                workspace_id=project.workspace_id,
                checksum_sha256=manifest_digest,
                size_bytes=len(manifest_bytes),
                content_type="application/json",
                source=_BytesReader(manifest_bytes),
                actor_user_id=actor_user_id,
                verify_reused=verify_reused_blobs,
            )
            lineage = register_stored_artifact(
                db,
                project_id=project_id,
                artifact_type="model_package",
                stored=_stored_object(manifest_blob, content_type="application/json"),
                manifest=manifest,
                created_by_user_id=actor_user_id,
                schema_version=data.schema_version,
            )
            package = ArtifactPackage(
                project_id=project_id,
                workspace_id=project.workspace_id,
                lineage_artifact_id=lineage.id,
                manifest_blob_id=manifest_blob.id,
                package_kind=data.package_kind,
                package_format=data.package_format,
                schema_version=data.schema_version,
                display_name=data.display_name,
                model_family=data.model_family,
                model_type=data.model_type,
                readiness=data.readiness,
                deployable=data.deployable,
                loader_policy=data.loader_policy,
                manifest_digest=manifest_digest,
                task_contract_hash=task_contract_hash,
                logical_size_bytes=sum(item.size_bytes for item in prepared),
                file_count=len(prepared),
                sensitivity=data.sensitivity,
                task_contract=data.task_contract,
                license_info=data.license_info,
                runtime=data.runtime,
                manifest=manifest,
                metadata_=data.metadata,
                created_by_user_id=actor_user_id,
            )
            db.add(package)
            db.flush()
            for item, blob in file_rows:
                db.add(
                    ArtifactPackageFile(
                        package_id=package.id,
                        blob_id=blob.id,
                        relative_path=item.relative_path,
                        role=item.role,
                        content_type=item.content_type,
                        checksum_sha256=item.checksum_sha256,
                        size_bytes=item.size_bytes,
                        metadata_=item.metadata,
                    )
                )
            for reference_data, target in references:
                db.add(
                    ArtifactPackageReference(
                        source_package_id=package.id,
                        target_package_id=target.id,
                        relationship_type=reference_data.relationship_type,
                        metadata_=reference_data.metadata,
                    )
                )
                if target.project_id == project_id:
                    add_lineage_edge(
                        db,
                        upstream_artifact_id=target.lineage_artifact_id,
                        downstream_artifact_id=lineage.id,
                        relationship_type=reference_data.relationship_type,
                    )
            retention_class = data.retention_class
            expires_at = None
            if retention_class == "resume_14d":
                expires_at = datetime.now(UTC) + timedelta(days=14)
            db.add(
                ArtifactPackageRetention(
                    package_id=package.id,
                    retention_class=retention_class,
                    pinned=data.pinned or retention_class == "candidate",
                    expires_at=expires_at,
                    updated_by_user_id=actor_user_id,
                )
            )
            if reservation_id is not None:
                artifact_quota.link_reservation_artifact_package(
                    db,
                    reservation_id=reservation_id,
                    artifact_package_id=package.id,
                )
            for upstream in upstream_artifacts:
                add_lineage_edge(
                    db,
                    upstream_artifact_id=upstream.id,
                    downstream_artifact_id=lineage.id,
                    relationship_type="packaged_from",
                )
            db.flush()
        db.refresh(package)
        return package
    finally:
        for item in prepared:
            item.close()


def get_artifact_package(
    db: Session,
    *,
    project_id: int,
    package_id: int,
) -> ArtifactPackage:
    package = db.get(ArtifactPackage, package_id)
    if package is None or package.project_id != project_id:
        raise NotFoundError("Artifact package not found in project")
    return package


def public_package_descriptor(package: ArtifactPackage) -> ArtifactPackageRead:
    """Return a descriptor that never exposes physical blob IDs or storage keys."""

    references = [
        ArtifactPackageReferenceRead(
            relationship_type=reference.relationship_type,
            target_package_id=(
                reference.target_package_id
                if reference.target_package.project_id == package.project_id
                else None
            ),
            target_manifest_digest=reference.target_package.manifest_digest,
            metadata=_sanitize_public_value(reference.metadata_),
        )
        for reference in sorted(
            package.outgoing_references,
            key=lambda item: (item.relationship_type, item.target_package_id),
        )
    ]
    files = [
        ArtifactPackageFileRead(
            id=item.id,
            relative_path=item.relative_path,
            role=item.role,
            content_type=item.content_type,
            checksum_sha256=item.checksum_sha256,
            size_bytes=item.size_bytes,
            metadata=_sanitize_public_value(item.metadata_),
        )
        for item in package.files
    ]
    retention = package.retention
    if retention is None:
        raise ConflictError("Artifact package is missing its retention record")
    return ArtifactPackageRead(
        id=package.id,
        project_id=package.project_id,
        lineage_artifact_id=package.lineage_artifact_id,
        package_kind=package.package_kind,
        package_format=package.package_format,
        schema_version=package.schema_version,
        display_name=package.display_name,
        model_family=package.model_family,
        model_type=package.model_type,
        readiness=package.readiness,
        deployable=package.deployable,
        loader_policy=package.loader_policy,
        manifest_digest=package.manifest_digest,
        task_contract_hash=package.task_contract_hash,
        logical_size_bytes=package.logical_size_bytes,
        file_count=package.file_count,
        sensitivity=package.sensitivity,
        task_contract=_sanitize_public_value(package.task_contract),
        license_info=_sanitize_public_value(package.license_info),
        runtime=_sanitize_public_value(package.runtime),
        metadata=_sanitize_public_value(package.metadata_),
        manifest=_sanitize_public_value(package.manifest),
        files=files,
        references=references,
        retention=ArtifactPackageRetentionRead(
            retention_class=retention.retention_class,
            pinned=retention.pinned,
            expires_at=retention.expires_at,
            archived_at=retention.archived_at,
            archived_by_user_id=retention.archived_by_user_id,
            archive_reason=retention.archive_reason,
            purge_after=retention.purge_after,
            purged_at=retention.purged_at,
            purged_by_user_id=retention.purged_by_user_id,
            purge_reason=retention.purge_reason,
            payload_gc_processed_at=retention.payload_gc_processed_at,
            legal_hold=retention.legal_hold,
        ),
        created_at=package.created_at,
    )


def artifact_storage_usage(
    db: Session,
    *,
    project_id: int,
    include_workspace_physical: bool = False,
) -> ArtifactStorageUsageRead:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    packages = (
        db.query(ArtifactPackage)
        .join(ArtifactPackageRetention)
        .filter(
            ArtifactPackage.project_id == project_id,
            ArtifactPackageRetention.purged_at.is_(None),
        )
        .all()
    )
    package_ids = [package.id for package in packages]
    files = (
        db.query(ArtifactPackageFile)
        .filter(ArtifactPackageFile.package_id.in_(package_ids))
        .all()
        if package_ids
        else []
    )
    project_blob_ids = {item.blob_id for item in files}
    project_blob_ids.update(package.manifest_blob_id for package in packages)
    project_blobs = (
        db.query(ArtifactBlob).filter(ArtifactBlob.id.in_(project_blob_ids)).all()
        if project_blob_ids
        else []
    )

    workspace_physical = None
    workspace_reclaimable = None
    workspace_deduplicated = None
    if include_workspace_physical:
        workspace_blobs = (
            db.query(ArtifactBlob)
            .filter(
                ArtifactBlob.workspace_id == project.workspace_id,
                ArtifactBlob.status == "ready",
            )
            .all()
        )
        workspace_packages = (
            db.query(ArtifactPackage)
            .join(ArtifactPackageRetention)
            .filter(
                ArtifactPackage.workspace_id == project.workspace_id,
                ArtifactPackageRetention.purged_at.is_(None),
            )
            .all()
        )
        workspace_package_ids = [package.id for package in workspace_packages]
        workspace_files = (
            db.query(ArtifactPackageFile)
            .filter(ArtifactPackageFile.package_id.in_(workspace_package_ids))
            .all()
            if workspace_package_ids
            else []
        )
        referenced_blob_ids = {item.blob_id for item in workspace_files}
        referenced_blob_ids.update(package.manifest_blob_id for package in workspace_packages)
        workspace_physical = sum(blob.size_bytes for blob in workspace_blobs)
        workspace_reclaimable = sum(
            blob.size_bytes for blob in workspace_blobs if blob.id not in referenced_blob_ids
        )
        logical_file_bytes = sum(item.size_bytes for item in workspace_files)
        unique_file_blob_ids = {item.blob_id for item in workspace_files}
        unique_file_bytes = sum(
            blob.size_bytes for blob in workspace_blobs if blob.id in unique_file_blob_ids
        )
        workspace_deduplicated = max(0, logical_file_bytes - unique_file_bytes)

    quota_snapshot = artifact_quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=project.workspace_id,
    )
    return ArtifactStorageUsageRead(
        project_id=project_id,
        workspace_id=project.workspace_id,
        package_count=len(packages),
        package_file_count=len(files),
        logical_bytes=sum(package.logical_size_bytes for package in packages),
        unique_project_blob_bytes=sum(blob.size_bytes for blob in project_blobs),
        workspace_physical_bytes=workspace_physical,
        workspace_reclaimable_bytes=workspace_reclaimable,
        workspace_deduplicated_bytes=workspace_deduplicated,
        workspace_quota_limit_bytes=quota_snapshot.limit_bytes,
        workspace_quota_used_bytes=quota_snapshot.used_bytes,
        workspace_reserved_bytes=quota_snapshot.reserved_bytes,
        workspace_available_bytes=quota_snapshot.available_bytes,
        workspace_accounting_consistent=quota_snapshot.accounting_consistent,
    )


def iter_package_file(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    package_id: int,
    relative_path: str,
    chunk_size: int = DEFAULT_COPY_CHUNK_SIZE,
) -> tuple[ArtifactPackageFileRead, Iterator[bytes]]:
    """Resolve an authorized logical path without returning its physical key."""

    package = get_artifact_package(db, project_id=project_id, package_id=package_id)
    _require_package_payload_available(package)
    normalized_path = validate_relative_path(relative_path)
    package_file = next(
        (item for item in package.files if item.relative_path == normalized_path),
        None,
    )
    if package_file is None:
        raise NotFoundError("Package file not found")
    descriptor = ArtifactPackageFileRead(
        id=package_file.id,
        relative_path=package_file.relative_path,
        role=package_file.role,
        content_type=package_file.content_type,
        checksum_sha256=package_file.checksum_sha256,
        size_bytes=package_file.size_bytes,
        metadata=_sanitize_public_value(package_file.metadata_),
    )
    return descriptor, storage.iter_bytes(
        package_file.blob.storage_key,
        chunk_size=chunk_size,
    )


def verify_artifact_package(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    package_id: int,
) -> ArtifactPackage:
    package = get_artifact_package(db, project_id=project_id, package_id=package_id)
    _require_package_payload_available(package)
    _verify_storage_blob(
        storage,
        package.manifest_blob.storage_key,
        checksum_sha256=package.manifest_digest,
        size_bytes=len(canonical_json_bytes(package.manifest)),
    )
    for item in package.files:
        _verify_storage_blob(
            storage,
            item.blob.storage_key,
            checksum_sha256=item.checksum_sha256,
            size_bytes=item.size_bytes,
        )
    return package


def publish_uploaded_base_model(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: BaseModelUploadCreate,
    sources: Sequence[BinaryIO],
    actor_user_id: int | None,
) -> BaseModelAsset:
    """Publish uploaded base-model files and register their exact revision."""

    if len(sources) != len(data.files):
        raise ValidationError("Each base-model file descriptor requires one upload")
    package = publish_artifact_package(
        db,
        storage,
        project_id=project_id,
        data=ArtifactPackageCreate(
            package_kind="base_model",
            package_format=data.package_format,
            display_name=data.display_name,
            model_family=data.model_family,
            model_type=data.model_type,
            readiness="ready",
            deployable=True,
            loader_policy="safe",
            task_contract=data.task_contract,
            sensitivity="restricted" if data.access_mode == "manager_only" else "project",
            license_info=_base_model_license_info(data),
            runtime=data.runtime,
            metadata=_base_model_package_metadata(data),
        ),
        files=[
            PackageFileUpload(
                relative_path=descriptor.relative_path,
                source=source,
                role=descriptor.role,
                content_type=descriptor.content_type,
                expected_checksum_sha256=descriptor.checksum_sha256,
                expected_size_bytes=descriptor.size_bytes,
                metadata=descriptor.metadata,
            )
            for descriptor, source in zip(data.files, sources, strict=True)
        ],
        actor_user_id=actor_user_id,
        prepared_validator=lambda prepared: _validate_uploaded_base_model(
            data,
            prepared,
        ),
    )
    verify_artifact_package(
        db,
        storage,
        project_id=project_id,
        package_id=package.id,
    )
    return _register_base_model_asset(
        db,
        project_id=project_id,
        package=package,
        data=data,
        actor_user_id=actor_user_id,
        action="uploaded",
    )


def import_base_model_from_package(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: BaseModelImportCreate,
    actor_user_id: int | None,
) -> BaseModelAsset:
    """Catalog an existing ready package under new immutable base-model metadata.

    A new logical base-model package is published, while its physical file blobs
    are deduplicated.  This prevents a later license/catalog edit from silently
    changing an existing package's immutable manifest.
    """

    source_package = get_artifact_package(
        db,
        project_id=project_id,
        package_id=data.source_package_id,
    )
    if (
        source_package.readiness != "ready"
        or not source_package.deployable
        or source_package.retention.archived_at is not None
    ):
        raise ConflictError("Only ready, deployable, non-archived packages may be imported")
    source_access_mode = source_package.license_info.get("access_mode", "downloadable")
    if source_access_mode not in _ACCESS_MODE_RANK:
        raise ConflictError("The source package has an unsupported license access mode")
    if _ACCESS_MODE_RANK[data.access_mode] < _ACCESS_MODE_RANK[source_access_mode]:
        raise ConflictError("A base-model import cannot broaden source package access")
    verify_artifact_package(
        db,
        storage,
        project_id=project_id,
        package_id=source_package.id,
    )

    package = publish_artifact_package(
        db,
        storage,
        project_id=project_id,
        data=ArtifactPackageCreate(
            package_kind="base_model",
            package_format=source_package.package_format,
            display_name=data.display_name,
            model_family=data.model_family,
            model_type=data.model_type,
            readiness="ready",
            deployable=True,
            loader_policy=source_package.loader_policy,
            task_contract=source_package.task_contract,
            sensitivity="restricted" if data.access_mode == "manager_only" else "project",
            license_info=_base_model_license_info(data),
            runtime=source_package.runtime,
            metadata=_base_model_package_metadata(data),
            references=[
                ArtifactPackageReferenceCreate(
                    target_package_id=source_package.id,
                    relationship_type="imported_from",
                )
            ],
        ),
        files=[
            PackageFileUpload(
                relative_path=item.relative_path,
                source=_IteratorBinaryReader(storage.iter_bytes(item.blob.storage_key)),
                role=item.role,
                content_type=item.content_type,
                expected_checksum_sha256=item.checksum_sha256,
                expected_size_bytes=item.size_bytes,
                metadata=item.metadata_,
            )
            for item in source_package.files
        ],
        actor_user_id=actor_user_id,
        upstream_lineage_artifact_ids=(source_package.lineage_artifact_id,),
    )
    verify_artifact_package(
        db,
        storage,
        project_id=project_id,
        package_id=package.id,
    )
    return _register_base_model_asset(
        db,
        project_id=project_id,
        package=package,
        data=data,
        actor_user_id=actor_user_id,
        action="imported",
    )


def get_base_model_asset(
    db: Session,
    *,
    project_id: int,
    asset_id: int,
) -> BaseModelAsset:
    asset = db.get(BaseModelAsset, asset_id)
    if asset is None or asset.project_id != project_id:
        raise NotFoundError("Base model asset not found in project")
    return asset


def list_base_model_assets(
    db: Session,
    *,
    project_id: int,
    include_archived: bool = False,
    include_manager_only: bool = False,
    readiness: str | None = None,
    limit: int = 200,
) -> list[BaseModelAsset]:
    query = (
        db.query(BaseModelAsset)
        .join(BaseModelAssetState)
        .filter(BaseModelAsset.project_id == project_id)
    )
    if not include_archived:
        query = query.filter(BaseModelAssetState.archived_at.is_(None))
    if not include_manager_only:
        query = query.filter(BaseModelAsset.access_mode != "manager_only")
    if readiness:
        query = query.filter(BaseModelAssetState.readiness == readiness)
    return (
        query.order_by(BaseModelAsset.created_at.desc(), BaseModelAsset.id.desc())
        .limit(limit)
        .all()
    )


def public_base_model_descriptor(asset: BaseModelAsset) -> BaseModelAssetRead:
    if asset.state is None:
        raise ConflictError("Base model asset is missing its mutable state record")
    return BaseModelAssetRead(
        id=asset.id,
        project_id=asset.project_id,
        package_id=asset.package_id,
        provider=asset.provider,
        source_model_id=asset.source_model_id,
        exact_revision=asset.exact_revision,
        display_name=asset.display_name,
        model_family=asset.model_family,
        model_type=asset.model_type,
        license_name=asset.license_name,
        license_url=asset.license_url,
        license_terms_sha256=asset.license_terms_sha256,
        access_mode=asset.access_mode,
        readiness=asset.state.readiness,
        archived_at=asset.state.archived_at,
        metadata=_sanitize_public_value(asset.metadata_),
        package=public_package_descriptor(asset.package),
        created_at=asset.created_at,
    )


def list_base_model_events(asset: BaseModelAsset) -> list[BaseModelAssetEventRead]:
    return [
        BaseModelAssetEventRead(
            id=event.id,
            action=event.action,
            prior_readiness=event.prior_readiness,
            resulting_readiness=event.resulting_readiness,
            details=_sanitize_public_value(event.details),
            actor_user_id=event.actor_user_id,
            created_at=event.created_at,
        )
        for event in asset.events
    ]


def update_base_model_readiness(
    db: Session,
    *,
    asset: BaseModelAsset,
    data: BaseModelReadinessUpdate,
    actor_user_id: int | None,
) -> BaseModelAsset:
    state = asset.state
    if state is None:
        raise ConflictError("Base model asset is missing its mutable state record")
    if state.archived_at is not None:
        raise ConflictError("Archived base models cannot change readiness")
    prior = state.readiness
    if prior == data.readiness:
        return asset
    state.readiness = data.readiness
    state.updated_by_user_id = actor_user_id
    db.add(
        BaseModelAssetEvent(
            base_model_asset_id=asset.id,
            action="readiness_changed",
            prior_readiness=prior,
            resulting_readiness=data.readiness,
            details={"reason": data.reason},
            actor_user_id=actor_user_id,
        )
    )
    db.flush()
    return asset


def archive_base_model_asset(
    db: Session,
    *,
    asset: BaseModelAsset,
    actor_user_id: int | None,
) -> BaseModelAsset:
    state = asset.state
    if state is None:
        raise ConflictError("Base model asset is missing its mutable state record")
    if state.archived_at is not None:
        return asset
    prior = state.readiness
    state.readiness = "archived"
    state.archived_at = datetime.now(UTC)
    state.updated_by_user_id = actor_user_id
    db.add(
        BaseModelAssetEvent(
            base_model_asset_id=asset.id,
            action="archived",
            prior_readiness=prior,
            resulting_readiness="archived",
            details={},
            actor_user_id=actor_user_id,
        )
    )
    db.flush()
    return asset


def base_model_package_access_mode(db: Session, package: ArtifactPackage) -> str:
    """Resolve the strictest access mode without exposing catalog internals."""

    asset = (
        db.query(BaseModelAsset)
        .filter(BaseModelAsset.package_id == package.id)
        .first()
    )
    if asset is not None:
        return asset.access_mode
    access_mode = package.license_info.get("access_mode", "downloadable")
    return access_mode if access_mode in _ACCESS_MODE_RANK else "manager_only"


def artifact_package_visible_to_role(
    db: Session,
    package: ArtifactPackage,
    *,
    is_manager: bool,
) -> bool:
    retention = package.retention
    if retention is None:
        return False
    if retention.purged_at is not None or retention.archived_at is not None:
        return is_manager
    asset = (
        db.query(BaseModelAsset)
        .filter(BaseModelAsset.package_id == package.id)
        .first()
    )
    if asset is not None:
        if not is_manager and (
            asset.access_mode == "manager_only"
            or asset.state is None
            or asset.state.readiness != "ready"
            or asset.state.archived_at is not None
        ):
            return False
    return is_manager or base_model_package_access_mode(db, package) != "manager_only"


def _require_package_payload_available(package: ArtifactPackage) -> None:
    retention = package.retention
    if retention is None:
        raise ConflictError("Artifact package is missing its retention record")
    if retention.purged_at is not None:
        raise ConflictError("Artifact package payload has been purged")


def request_artifact_package_archive(
    db: Session,
    *,
    package: ArtifactPackage,
    actor_user_id: int | None,
    reason: str | None = None,
    now: datetime | None = None,
) -> ArtifactPackageRetention:
    """Archive a package and start the non-bypassable seven-day purge grace."""

    normalized_reason = reason.strip() if reason is not None else None
    if normalized_reason is not None and not 3 <= len(normalized_reason) <= 500:
        raise ValidationError("An archive reason must contain 3 to 500 characters")
    retention = _locked_package_retention(db, package.id)
    if retention.purged_at is not None:
        raise ConflictError("Purged packages cannot be archived")
    if retention.archived_at is not None:
        return retention
    _raise_if_package_retention_blocked(db, package, action="archived")
    effective_now = now or datetime.now(UTC)
    retention.archived_at = effective_now
    retention.archived_by_user_id = actor_user_id
    retention.archive_reason = normalized_reason
    retention.purge_after = effective_now + timedelta(days=7)
    retention.updated_by_user_id = actor_user_id
    db.flush()
    return retention


def update_artifact_package_retention(
    db: Session,
    *,
    package: ArtifactPackage,
    actor_user_id: int | None,
    pinned: bool | None = None,
    legal_hold: bool | None = None,
) -> ArtifactPackageRetention:
    """Update manager-controlled retention protections under a row lock."""

    if pinned is None and legal_hold is None:
        raise ValidationError("At least one retention field is required")
    retention = _locked_package_retention(db, package.id)
    if retention.purged_at is not None:
        raise ConflictError("Purged package retention cannot be changed")
    if pinned is not None:
        retention.pinned = pinned
    if legal_hold is not None:
        retention.legal_hold = legal_hold
    retention.updated_by_user_id = actor_user_id
    db.flush()
    return retention


def approve_artifact_package_purge(
    db: Session,
    *,
    package: ArtifactPackage,
    actor_user_id: int | None,
    reason: str,
    now: datetime | None = None,
) -> ArtifactPackageRetention:
    """Create a durable tombstone; physical bytes are reclaimed separately by GC."""

    normalized_reason = reason.strip()
    if len(normalized_reason) < 3 or len(normalized_reason) > 500:
        raise ValidationError("A purge reason must contain 3 to 500 characters")
    retention = _locked_package_retention(db, package.id)
    if retention.purged_at is not None:
        return retention
    if retention.archived_at is None or retention.purge_after is None:
        raise ConflictError("A package must be archived before it can be purged")
    effective_now = now or datetime.now(UTC)
    if _as_utc(retention.purge_after) > _as_utc(effective_now):
        raise ConflictError("The seven-day package purge grace period has not elapsed")
    _raise_if_package_retention_blocked(db, package, action="purged")
    retention.purged_at = effective_now
    retention.purged_by_user_id = actor_user_id
    retention.purge_reason = normalized_reason
    retention.updated_by_user_id = actor_user_id
    db.flush()
    return retention


def garbage_collect_artifact_blobs(
    db: Session,
    storage: ObjectStorage,
    *,
    workspace_id: int | None = None,
    limit: int = 1000,
    after_blob_id: int = 0,
) -> ArtifactGarbageCollectionResult:
    """Delete only blobs with no live package reference, preserving DB tombstones.

    Blob rows are locked before their references are rechecked. Publication takes
    the same lock when reusing a checksum, preventing a package from committing a
    reference to bytes that GC deleted concurrently.
    """

    bounded_limit = min(max(limit, 1), 10_000)
    query = db.query(ArtifactBlob).filter(
        ArtifactBlob.status == "ready",
        ArtifactBlob.id > max(after_blob_id, 0),
    )
    if workspace_id is not None:
        query = query.filter(ArtifactBlob.workspace_id == workspace_id)
        artifact_quota.lock_workspace_artifact_quota(
            db,
            workspace_id=workspace_id,
        )
        candidate_ids = None
    else:
        # Discover the bounded batch before taking locks, then acquire every
        # workspace quota in a stable order. Publication uses the same
        # quota-before-blob ordering.
        candidate_rows = (
            query.with_entities(ArtifactBlob.id, ArtifactBlob.workspace_id)
            .order_by(ArtifactBlob.id)
            .limit(bounded_limit)
            .all()
        )
        candidate_ids = [blob_id for blob_id, _workspace_id in candidate_rows]
        for candidate_workspace_id in sorted(
            {row_workspace_id for _blob_id, row_workspace_id in candidate_rows}
        ):
            artifact_quota.lock_workspace_artifact_quota(
                db,
                workspace_id=candidate_workspace_id,
            )
        query = db.query(ArtifactBlob).filter(
            ArtifactBlob.id.in_(candidate_ids),
            ArtifactBlob.status == "ready",
        )
    candidates = (
        query.order_by(ArtifactBlob.id)
        .limit(bounded_limit)
        .with_for_update(skip_locked=True)
        .all()
    )
    deleted = 0
    reclaimed = 0
    reclaimed_by_workspace: dict[int, int] = {}
    for blob in candidates:
        if _blob_has_live_package_reference(db, blob.id):
            continue
        storage.delete(blob.storage_key)
        if _blob_has_any_package_reference(db, blob.id):
            blob.status = "purged"
        else:
            # Publication crash leftovers have no lineage tombstone to retain.
            db.delete(blob)
        deleted += 1
        reclaimed += blob.size_bytes
        reclaimed_by_workspace[blob.workspace_id] = (
            reclaimed_by_workspace.get(blob.workspace_id, 0) + blob.size_bytes
        )
    for reclaimed_workspace_id, reclaimed_bytes in reclaimed_by_workspace.items():
        artifact_quota.record_reclaimed_artifact_bytes(
            db,
            workspace_id=reclaimed_workspace_id,
            reclaimed_bytes=reclaimed_bytes,
        )
    db.flush()
    return ArtifactGarbageCollectionResult(
        scanned_blob_count=len(candidates),
        deleted_blob_count=deleted,
        reclaimed_bytes=reclaimed,
        last_scanned_blob_id=candidates[-1].id if candidates else None,
    )


def garbage_collect_purged_package_payloads(
    db: Session,
    storage: ObjectStorage,
    *,
    workspace_id: int | None = None,
    limit: int = 1000,
    after_retention_id: int = 0,
) -> PurgedPackagePayloadCollectionResult:
    """Remove legacy checkpoint objects while retaining their lineage tombstones."""

    from al_medlit.training.models import ModelCheckpoint

    bounded_limit = min(max(limit, 1), 10_000)
    query = (
        db.query(ArtifactPackageRetention)
        .join(ArtifactPackage)
        .filter(
            ArtifactPackageRetention.id > max(after_retention_id, 0),
            ArtifactPackageRetention.purged_at.is_not(None),
            ArtifactPackageRetention.payload_gc_processed_at.is_(None),
        )
    )
    if workspace_id is not None:
        query = query.filter(ArtifactPackage.workspace_id == workspace_id)
    retentions = (
        query.order_by(ArtifactPackageRetention.id)
        .limit(bounded_limit)
        .with_for_update(skip_locked=True)
        .all()
    )
    for retention in retentions:
        checkpoints = (
            db.query(ModelCheckpoint)
            .filter(ModelCheckpoint.package_id == retention.package_id)
            .all()
        )
        for checkpoint in checkpoints:
            storage_key = checkpoint.artifact.storage_key
            content_addressed = (
                db.query(ArtifactBlob.id)
                .filter(ArtifactBlob.storage_key == storage_key)
                .first()
            )
            if content_addressed is None:
                storage.delete(storage_key)
        retention.payload_gc_processed_at = datetime.now(UTC)
    db.flush()
    return PurgedPackagePayloadCollectionResult(
        processed_package_count=len(retentions),
        last_scanned_retention_id=retentions[-1].id if retentions else None,
    )


def _locked_package_retention(
    db: Session,
    package_id: int,
) -> ArtifactPackageRetention:
    retention = (
        db.query(ArtifactPackageRetention)
        .filter(ArtifactPackageRetention.package_id == package_id)
        .with_for_update()
        .one_or_none()
    )
    if retention is None:
        raise ConflictError("Artifact package is missing its retention record")
    return retention


def _raise_if_package_retention_blocked(
    db: Session,
    package: ArtifactPackage,
    *,
    action: str,
) -> None:
    blockers = _package_retention_blockers(db, package)
    if blockers:
        raise ConflictError(
            f"Package cannot be {action} while protected by: {', '.join(blockers)}"
        )


def _package_retention_blockers(
    db: Session,
    package: ArtifactPackage,
) -> list[str]:
    """Return fail-closed protection categories without leaking foreign project IDs."""

    # Lazy imports keep the immutable artifact model usable without forcing the
    # training and inference modules to import during standalone storage tools.
    from al_medlit.inference.models import InferenceRun
    from al_medlit.training.models import (
        ModelCheckpoint,
        ModelCheckpointPin,
        ModelReleaseAlias,
    )

    retention = _locked_package_retention(db, package.id)
    blockers: list[str] = []
    if retention.legal_hold:
        blockers.append("legal hold")
    if retention.pinned:
        blockers.append("package pin")
    active_candidate = (
        db.query(ModelCheckpointPin.id)
        .join(ModelCheckpoint, ModelCheckpoint.id == ModelCheckpointPin.checkpoint_id)
        .filter(
            ModelCheckpoint.package_id == package.id,
            ModelCheckpointPin.active.is_(True),
        )
        .first()
    )
    if active_candidate:
        blockers.append("trainer candidate pin")
    champion = (
        db.query(ModelReleaseAlias.id)
        .join(ModelCheckpoint, ModelCheckpoint.id == ModelReleaseAlias.checkpoint_id)
        .filter(
            ModelCheckpoint.package_id == package.id,
            ModelReleaseAlias.state == "champion",
        )
        .first()
    )
    if champion:
        blockers.append("champion alias")
    inference_record = (
        db.query(InferenceRun.id)
        .join(ModelCheckpoint, ModelCheckpoint.id == InferenceRun.checkpoint_id)
        .filter(ModelCheckpoint.package_id == package.id)
        .first()
    )
    if inference_record:
        blockers.append("inference record")
    source_retention = ArtifactPackageRetention
    incoming_reference = (
        db.query(ArtifactPackageReference.id)
        .join(
            source_retention,
            source_retention.package_id == ArtifactPackageReference.source_package_id,
            isouter=True,
        )
        .filter(
            ArtifactPackageReference.target_package_id == package.id,
            or_(source_retention.id.is_(None), source_retention.purged_at.is_(None)),
        )
        .first()
    )
    if incoming_reference:
        blockers.append("package dependency")
    active_base_model = (
        db.query(BaseModelAsset.id)
        .join(BaseModelAssetState)
        .filter(
            BaseModelAsset.package_id == package.id,
            BaseModelAssetState.archived_at.is_(None),
        )
        .first()
    )
    if active_base_model:
        blockers.append("active base-model catalog entry")
    return blockers


def _blob_has_live_package_reference(db: Session, blob_id: int) -> bool:
    file_reference = (
        db.query(ArtifactPackageFile.id)
        .join(ArtifactPackage, ArtifactPackage.id == ArtifactPackageFile.package_id)
        .join(
            ArtifactPackageRetention,
            ArtifactPackageRetention.package_id == ArtifactPackage.id,
            isouter=True,
        )
        .filter(
            ArtifactPackageFile.blob_id == blob_id,
            or_(
                ArtifactPackageRetention.id.is_(None),
                ArtifactPackageRetention.purged_at.is_(None),
            ),
        )
        .first()
    )
    if file_reference:
        return True
    manifest_reference = (
        db.query(ArtifactPackage.id)
        .join(
            ArtifactPackageRetention,
            ArtifactPackageRetention.package_id == ArtifactPackage.id,
            isouter=True,
        )
        .filter(
            ArtifactPackage.manifest_blob_id == blob_id,
            or_(
                ArtifactPackageRetention.id.is_(None),
                ArtifactPackageRetention.purged_at.is_(None),
            ),
        )
        .first()
    )
    return manifest_reference is not None


def _blob_has_any_package_reference(db: Session, blob_id: int) -> bool:
    file_reference = (
        db.query(ArtifactPackageFile.id)
        .filter(ArtifactPackageFile.blob_id == blob_id)
        .first()
    )
    if file_reference:
        return True
    return (
        db.query(ArtifactPackage.id)
        .filter(ArtifactPackage.manifest_blob_id == blob_id)
        .first()
        is not None
    )


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def migrate_legacy_checkpoints(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    actor_user_id: int | None,
    checkpoint_id: int | None = None,
    limit: int = 100,
) -> LegacyCheckpointMigrationRead:
    """Wrap legacy checkpoint ZIP bytes without extracting or rewriting them."""

    # Imported lazily to avoid coupling artifact model registration to training.
    from al_medlit.training.models import ModelCheckpoint

    query = db.query(ModelCheckpoint).filter(ModelCheckpoint.project_id == project_id)
    if checkpoint_id is not None:
        query = query.filter(ModelCheckpoint.id == checkpoint_id)
    else:
        query = query.filter(ModelCheckpoint.readiness == "legacy_unverified")
    checkpoints = query.order_by(ModelCheckpoint.id).limit(limit).all()
    if checkpoint_id is not None and not checkpoints:
        raise NotFoundError("Model checkpoint not found in project")

    items: list[LegacyCheckpointMigrationItemRead] = []
    for checkpoint in checkpoints:
        if checkpoint.package_id is not None:
            package = db.get(ArtifactPackage, checkpoint.package_id)
            if package is not None and package.package_format == "legacy_zip":
                package_file = package.files[0]
                items.append(
                    LegacyCheckpointMigrationItemRead(
                        checkpoint_id=checkpoint.id,
                        package_id=package.id,
                        checksum_sha256=package_file.checksum_sha256,
                        size_bytes=package_file.size_bytes,
                        migrated=False,
                    )
                )
            elif checkpoint_id is not None:
                raise ConflictError("Checkpoint already references a non-legacy package")
            continue
        if checkpoint.readiness != "legacy_unverified":
            if checkpoint_id is not None:
                raise ConflictError("Checkpoint is not marked as legacy unverified")
            continue

        artifact = checkpoint.artifact
        package = publish_artifact_package(
            db,
            storage,
            project_id=project_id,
            data=ArtifactPackageCreate(
                package_kind="trained_model",
                package_format="legacy_zip",
                display_name=f"Legacy checkpoint {checkpoint.id}",
                model_family=_legacy_model_family(checkpoint.model_type),
                model_type=checkpoint.model_type,
                readiness="legacy_unverified",
                deployable=False,
                loader_policy="isolated_internal",
                task_contract={
                    "key": checkpoint.task_contract_key,
                    "version": checkpoint.task_contract_version,
                },
                sensitivity="project",
                metadata={
                    "legacy_checkpoint_id": checkpoint.id,
                    "legacy_lineage_artifact_id": artifact.id,
                },
            ),
            files=[
                PackageFileUpload(
                    relative_path="checkpoint.zip",
                    source=_IteratorBinaryReader(storage.iter_bytes(artifact.storage_key)),
                    role="legacy_checkpoint_archive",
                    content_type="application/zip",
                    expected_checksum_sha256=artifact.content_hash,
                    expected_size_bytes=artifact.size_bytes,
                )
            ],
            actor_user_id=actor_user_id,
            upstream_lineage_artifact_ids=(artifact.id,),
        )
        verify_artifact_package(
            db,
            storage,
            project_id=project_id,
            package_id=package.id,
        )
        result = db.execute(
            update(ModelCheckpoint)
            .where(
                ModelCheckpoint.id == checkpoint.id,
                ModelCheckpoint.package_id.is_(None),
            )
            .values(package_id=package.id, readiness="legacy_unverified")
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise ConflictError("Legacy checkpoint was concurrently migrated")
        db.expire(checkpoint)
        package_file = package.files[0]
        items.append(
            LegacyCheckpointMigrationItemRead(
                checkpoint_id=checkpoint.id,
                package_id=package.id,
                checksum_sha256=package_file.checksum_sha256,
                size_bytes=package_file.size_bytes,
                migrated=True,
            )
        )
    return LegacyCheckpointMigrationRead(
        project_id=project_id,
        migrated_count=sum(item.migrated for item in items),
        existing_count=sum(not item.migrated for item in items),
        items=items,
    )


def _register_base_model_asset(
    db: Session,
    *,
    project_id: int,
    package: ArtifactPackage,
    data: BaseModelCatalogFields,
    actor_user_id: int | None,
    action: str,
) -> BaseModelAsset:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    retention = _locked_package_retention(db, package.id)
    if (
        package.project_id != project_id
        or package.workspace_id != project.workspace_id
        or package.package_kind != "base_model"
        or package.readiness != "ready"
        or not package.deployable
        or retention.archived_at is not None
        or retention.purged_at is not None
    ):
        raise ConflictError("Base model assets require a ready project base-model package")
    existing = (
        db.query(BaseModelAsset)
        .filter(
            BaseModelAsset.project_id == project_id,
            BaseModelAsset.provider == data.provider,
            BaseModelAsset.source_model_id == data.source_model_id,
            BaseModelAsset.exact_revision == data.exact_revision,
        )
        .first()
    )
    if existing is not None:
        if existing.package_id != package.id:
            raise ConflictError("This exact base-model revision has different package bytes")
        return existing
    asset = BaseModelAsset(
        project_id=project_id,
        workspace_id=project.workspace_id,
        package_id=package.id,
        provider=data.provider,
        source_model_id=data.source_model_id,
        exact_revision=data.exact_revision,
        display_name=data.display_name,
        model_family=data.model_family,
        model_type=data.model_type,
        license_name=data.license_name,
        license_url=data.license_url,
        license_terms_sha256=data.license_terms_sha256,
        access_mode=data.access_mode,
        metadata_=data.metadata,
        created_by_user_id=actor_user_id,
    )
    db.add(asset)
    db.flush()
    db.add_all(
        [
            BaseModelAssetState(
                base_model_asset_id=asset.id,
                readiness="ready",
                updated_by_user_id=actor_user_id,
            ),
            BaseModelAssetEvent(
                base_model_asset_id=asset.id,
                action=action,
                prior_readiness=None,
                resulting_readiness="ready",
                details={},
                actor_user_id=actor_user_id,
            ),
        ]
    )
    db.flush()
    db.refresh(asset)
    return asset


def _base_model_license_info(data: BaseModelCatalogFields) -> dict:
    return {
        "name": data.license_name,
        "url": data.license_url,
        "terms_sha256": data.license_terms_sha256,
        "access_mode": data.access_mode,
    }


def _base_model_package_metadata(data: BaseModelCatalogFields) -> dict:
    return {
        "base_model_source": {
            "provider": data.provider,
            "model_id": data.source_model_id,
            "exact_revision": data.exact_revision,
        },
        "catalog_metadata": data.metadata,
    }


def _legacy_model_family(model_type: str) -> str | None:
    normalized = model_type.lower()
    if any(name in normalized for name in ("crf", "svm", "forest", "rf")):
        return "conventional_ml"
    if any(name in normalized for name in ("lora", "qlora", "llm")):
        return "llm_finetune"
    return "deep_learning"


def _prepare_files(files: Sequence[PackageFileUpload]) -> list[_PreparedFile]:
    if not files or len(files) > MAX_PACKAGE_FILES:
        raise ValidationError(f"A package must contain 1 to {MAX_PACKAGE_FILES} files")
    paths: set[str] = set()
    prepared: list[_PreparedFile] = []
    total_size = 0
    try:
        for upload in files:
            relative_path = validate_relative_path(upload.relative_path)
            if relative_path in paths:
                raise ValidationError(f"Duplicate package file path: {relative_path!r}")
            paths.add(relative_path)
            if not IDENTIFIER_RE.fullmatch(upload.role) or len(upload.role) > 80:
                raise ValidationError(f"Invalid package file role: {upload.role!r}")
            if (
                not upload.content_type
                or len(upload.content_type) > 120
                or "\r" in upload.content_type
                or "\n" in upload.content_type
            ):
                raise ValidationError("Invalid package file content type")
            if upload.expected_checksum_sha256 is not None and not SHA256_RE.fullmatch(
                upload.expected_checksum_sha256
            ):
                raise ValidationError("Expected checksum must be a lowercase SHA-256 digest")
            if upload.expected_size_bytes is not None and not (
                0 <= upload.expected_size_bytes <= MAX_PACKAGE_FILE_BYTES
            ):
                raise ValidationError(
                    f"Expected size must be between 0 and {MAX_PACKAGE_FILE_BYTES} bytes"
                )
            _ensure_no_sensitive_keys(upload.metadata)
            spool = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024, mode="w+b")
            digest = hashlib.sha256()
            size = 0
            try:
                source, should_close = _open_source(upload.source)
                try:
                    while chunk := source.read(DEFAULT_COPY_CHUNK_SIZE):
                        if not isinstance(chunk, bytes):
                            raise ValidationError("Package file sources must yield bytes")
                        spool.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                        if size > MAX_PACKAGE_FILE_BYTES:
                            raise ValidationError(
                                f"Package file {relative_path!r} exceeds the byte limit"
                            )
                finally:
                    if should_close:
                        source.close()
            except Exception:
                spool.close()
                raise
            checksum = digest.hexdigest()
            if upload.expected_checksum_sha256 not in {None, checksum}:
                spool.close()
                raise ConflictError(f"Checksum mismatch for package file {relative_path!r}")
            if upload.expected_size_bytes not in {None, size}:
                spool.close()
                raise ConflictError(f"Size mismatch for package file {relative_path!r}")
            total_size += size
            if total_size > MAX_PACKAGE_TOTAL_BYTES:
                spool.close()
                raise ValidationError("Package exceeds the total byte limit")
            spool.seek(0)
            json_value = None
            if relative_path.endswith(".json") and size <= MAX_JSON_INSPECTION_BYTES:
                try:
                    json_value = json.load(spool)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    spool.close()
                    raise ValidationError(
                        f"JSON package file {relative_path!r} is invalid"
                    ) from exc
                finally:
                    if not spool.closed:
                        spool.seek(0)
            prepared.append(
                _PreparedFile(
                    relative_path=relative_path,
                    role=upload.role,
                    content_type=upload.content_type,
                    checksum_sha256=checksum,
                    size_bytes=size,
                    metadata=upload.metadata,
                    spool=spool,
                    json_value=json_value,
                )
            )
    except Exception:
        for item in prepared:
            item.close()
        raise
    return sorted(prepared, key=lambda item: item.relative_path)


def _open_source(source: bytes | bytearray | BinaryIO | str | Path) -> tuple[BinaryIO, bool]:
    if isinstance(source, (bytes, bytearray)):
        return _BytesReader(bytes(source)), False
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValidationError("Package path sources must be regular, non-symlink files")
        return path.open("rb"), True
    if not hasattr(source, "read"):
        raise ValidationError("Unsupported package file source")
    return source, False


def _validate_package_policy(
    data: ArtifactPackageCreate,
    prepared: Sequence[_PreparedFile],
    references: Sequence[tuple[Any, ArtifactPackage]],
) -> None:
    paths = {item.relative_path for item in prepared}
    for item in prepared:
        suffix = PurePosixPath(item.relative_path).suffix.lower()
        if suffix in _FORBIDDEN_SUFFIXES:
            raise ValidationError(
                f"Executable or unsafe package file is forbidden: {item.relative_path}"
            )
        if suffix == ".joblib" and not (
            data.package_format == "joblib-internal"
            and data.loader_policy == "isolated_internal"
        ):
            raise ValidationError("Joblib files require the isolated internal loader policy")
        if suffix == ".zip" and data.package_format != "legacy_zip":
            raise ValidationError("ZIP files are accepted only as legacy unverified packages")
    if data.package_format == "legacy_zip" and (
        len(prepared) != 1 or not prepared[0].relative_path.endswith(".zip")
    ):
        raise ValidationError("A legacy ZIP package must contain exactly one ZIP file")
    if data.loader_policy == "safe" and data.package_format == "joblib-internal":
        raise ValidationError("Internal joblib packages cannot use the safe loader policy")
    if data.deployable and data.package_format == "onnx" and not any(
        path.endswith(".onnx") for path in paths
    ):
        raise ValidationError("Deployable ONNX packages require an .onnx model file")
    if data.deployable and data.model_family == "conventional_ml":
        required_suffixes = {
            "crfsuite": ".crfsuite",
            "joblib-internal": ".joblib",
            "onnx": ".onnx",
            "skops": ".skops",
        }
        required_suffix = required_suffixes.get(data.package_format)
        if required_suffix is None:
            raise ValidationError("Deployable conventional models require a safe known format")
        if not any(path.endswith(required_suffix) for path in paths):
            raise ValidationError(
                f"Package format {data.package_format!r} requires a {required_suffix} file"
            )
    if data.deployable and data.model_family in {"deep_learning", "llm_finetune"}:
        synthetic_runtime_package = (
            data.package_format == "synthetic"
            and data.package_kind == "deployable_checkpoint"
            and data.metadata.get("synthetic_mode") is True
            and all(item.relative_path.endswith(".json") for item in prepared)
        )
        if not synthetic_runtime_package and data.package_format != "onnx" and not any(
            path.endswith(".safetensors") for path in paths
        ):
            raise ValidationError("Deployable neural packages require safetensors or ONNX")
    if data.package_kind == "peft_adapter":
        if "adapter_config.json" not in paths or "adapter_model.safetensors" not in paths:
            raise ValidationError(
                "PEFT adapters require adapter_config.json and adapter_model.safetensors"
            )
        base_references = [
            target
            for reference, target in references
            if reference.relationship_type == "uses_base_model"
        ]
        if len(base_references) != 1:
            raise ValidationError(
                "PEFT adapters require exactly one immutable base-model reference"
            )
    _validate_shard_indexes(prepared, paths)


def _validate_shard_indexes(prepared: Sequence[_PreparedFile], paths: set[str]) -> None:
    for item in prepared:
        if not item.relative_path.endswith(".safetensors.index.json"):
            continue
        index = _load_strict_json(
            item,
            max_bytes=MAX_MODEL_METADATA_JSON_BYTES,
            label="safetensors index",
        )
        if not isinstance(index, dict):
            raise ValidationError(f"Invalid safetensors index: {item.relative_path}")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValidationError(f"Safetensors index has no weight map: {item.relative_path}")
        shard_paths = set()
        index_parent = PurePosixPath(item.relative_path).parent
        for tensor_name, shard in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard, str):
                raise ValidationError("Safetensors index contains an invalid shard path")
            relative_shard = PurePosixPath(shard)
            resolved_shard = (
                relative_shard if str(index_parent) == "." else index_parent / relative_shard
            )
            normalized_shard = validate_relative_path(str(resolved_shard))
            if not normalized_shard.endswith(".safetensors"):
                raise ValidationError(
                    "Safetensors index may reference only .safetensors shard files"
                )
            shard_paths.add(normalized_shard)
        missing = shard_paths - paths
        if missing:
            raise ValidationError(
                "Safetensors index references missing shards: " + ", ".join(sorted(missing))
            )


def _validate_uploaded_base_model(
    data: BaseModelUploadCreate,
    prepared: Sequence[_PreparedFile],
) -> None:
    """Fail closed before uploaded neural bytes are published as ready."""

    safetensors_files = {
        item.relative_path: item
        for item in prepared
        if item.relative_path.endswith(".safetensors")
    }
    if not safetensors_files:
        if data.model_family in {"deep_learning", "llm_finetune"}:
            raise ValidationError(
                "Ready neural base-model uploads require validated safetensors weights"
            )
        return

    descriptors = {
        path: _validate_safetensors_file(item)
        for path, item in safetensors_files.items()
    }
    _validate_safetensors_index_contents(prepared, descriptors)
    if data.model_family in {"deep_learning", "llm_finetune"}:
        _validate_huggingface_model_metadata(prepared)


def _validate_safetensors_file(item: _PreparedFile) -> _SafetensorsFile:
    """Validate a safetensors header and its exact relationship to the data buffer."""

    stream = item.spool
    try:
        stream.seek(0)
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} is missing its header length"
            )
        header_size = int.from_bytes(prefix, byteorder="little", signed=False)
        if (
            header_size == 0
            or header_size % 8 != 0
            or header_size > MAX_SAFETENSORS_HEADER_BYTES
            or header_size > item.size_bytes - 8
        ):
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has an invalid header length"
            )
        header_bytes = stream.read(header_size)
        if len(header_bytes) != header_size or not header_bytes.startswith(b"{"):
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has an invalid JSON header"
            )
        try:
            header = json.loads(
                header_bytes.decode("utf-8"),
                object_pairs_hook=_unique_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has an invalid JSON header"
            ) from exc
        if not isinstance(header, dict):
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has a non-object header"
            )

        metadata = header.pop("__metadata__", None)
        if metadata is not None and (
            not isinstance(metadata, dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            )
        ):
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has invalid metadata"
            )
        if not header:
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} contains no tensors"
            )

        data_size = item.size_bytes - 8 - header_size
        intervals: list[tuple[int, int]] = []
        for tensor_name, tensor in header.items():
            if not isinstance(tensor_name, str) or not tensor_name:
                raise ValidationError("Safetensors tensor names must be non-empty strings")
            if not isinstance(tensor, dict) or set(tensor) != {
                "dtype",
                "shape",
                "data_offsets",
            }:
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} has an invalid descriptor"
                )
            dtype = tensor["dtype"]
            dtype_bytes = (
                _SAFETENSORS_DTYPE_BYTES.get(dtype) if isinstance(dtype, str) else None
            )
            if dtype_bytes is None:
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} uses an unsupported dtype"
                )
            shape = tensor["shape"]
            if not isinstance(shape, list) or any(
                type(dimension) is not int or dimension < 0 for dimension in shape
            ):
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} has an invalid shape"
                )
            element_count = 1
            for dimension in shape:
                element_count *= dimension
                if element_count * dtype_bytes > data_size:
                    raise ValidationError(
                        f"Safetensors tensor {tensor_name!r} exceeds the data buffer"
                    )
            offsets = tensor["data_offsets"]
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
            ):
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} has invalid data offsets"
                )
            start, end = offsets
            if start < 0 or end < start or end > data_size:
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} points outside the data buffer"
                )
            if end - start != element_count * dtype_bytes:
                raise ValidationError(
                    f"Safetensors tensor {tensor_name!r} has a size/shape mismatch"
                )
            intervals.append((start, end))

        cursor = 0
        for start, end in sorted(intervals):
            if start != cursor:
                raise ValidationError(
                    f"Safetensors file {item.relative_path!r} has overlapping or sparse data"
                )
            cursor = end
        if cursor != data_size or data_size == 0:
            raise ValidationError(
                f"Safetensors file {item.relative_path!r} has unindexed or empty data"
            )
        return _SafetensorsFile(
            tensor_names=frozenset(header),
            data_size_bytes=data_size,
        )
    finally:
        stream.seek(0)


def _validate_safetensors_index_contents(
    prepared: Sequence[_PreparedFile],
    descriptors: dict[str, _SafetensorsFile],
) -> None:
    indexes = [
        item for item in prepared if item.relative_path.endswith(".safetensors.index.json")
    ]
    sharded_names = [
        path
        for path in descriptors
        if re.search(r"-\d{5}-of-\d{5}\.safetensors$", path)
    ]
    if sharded_names and not indexes:
        raise ValidationError("Sharded safetensors uploads require a shard index")

    indexed_shards: set[str] = set()
    for item in indexes:
        index = _load_strict_json(
            item,
            max_bytes=MAX_MODEL_METADATA_JSON_BYTES,
            label="safetensors index",
        )
        if not isinstance(index, dict):
            raise ValidationError(f"Invalid safetensors index: {item.relative_path}")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValidationError(f"Safetensors index has no weight map: {item.relative_path}")
        index_parent = PurePosixPath(item.relative_path).parent
        mapped_by_shard: dict[str, set[str]] = {}
        for tensor_name, shard in weight_map.items():
            if not isinstance(tensor_name, str) or not tensor_name or not isinstance(shard, str):
                raise ValidationError("Safetensors index has an invalid weight map")
            relative_shard = PurePosixPath(shard)
            resolved_shard = (
                relative_shard if str(index_parent) == "." else index_parent / relative_shard
            )
            shard_path = validate_relative_path(str(resolved_shard))
            descriptor = descriptors.get(shard_path)
            if descriptor is None or tensor_name not in descriptor.tensor_names:
                raise ValidationError(
                    f"Safetensors index maps {tensor_name!r} to the wrong or missing shard"
                )
            mapped_by_shard.setdefault(shard_path, set()).add(tensor_name)
            indexed_shards.add(shard_path)

        for shard_path, mapped_names in mapped_by_shard.items():
            missing_names = descriptors[shard_path].tensor_names - mapped_names
            if missing_names:
                raise ValidationError(
                    "Safetensors index omits tensors from a referenced shard: "
                    + ", ".join(sorted(missing_names))
                )
        metadata = index.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValidationError("Safetensors index metadata must be an object")
        if isinstance(metadata, dict) and "total_size" in metadata:
            total_size = metadata["total_size"]
            referenced_size = sum(
                descriptors[path].data_size_bytes for path in mapped_by_shard
            )
            if type(total_size) is not int or total_size != referenced_size:
                raise ValidationError("Safetensors index total_size does not match its shards")
    missing_sharded_files = set(sharded_names) - indexed_shards
    if missing_sharded_files:
        raise ValidationError(
            "Safetensors shard index does not reference uploaded shards: "
            + ", ".join(sorted(missing_sharded_files))
        )


def _validate_huggingface_model_metadata(prepared: Sequence[_PreparedFile]) -> None:
    files = {item.relative_path: item for item in prepared}
    config = _required_json_object(files, "config.json", "Hugging Face model config")
    model_type = config.get("model_type")
    if not isinstance(model_type, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", model_type):
        raise ValidationError("Hugging Face config.json requires a valid model_type")
    architectures = config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list)
        or not architectures
        or any(not isinstance(name, str) or not name for name in architectures)
    ):
        raise ValidationError("Hugging Face config architectures must be model class names")

    tokenizer_config = _required_json_object(
        files,
        "tokenizer_config.json",
        "Hugging Face tokenizer config",
    )
    tokenizer_metadata_keys = {
        "tokenizer_class",
        "model_max_length",
        "unk_token",
        "bos_token",
        "eos_token",
        "pad_token",
        "mask_token",
        "added_tokens_decoder",
    }
    if not tokenizer_metadata_keys.intersection(tokenizer_config):
        raise ValidationError(
            "tokenizer_config.json lacks tokenizer class, limits, or special-token metadata"
        )
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if tokenizer_class is not None and (
        not isinstance(tokenizer_class, str) or not tokenizer_class
    ):
        raise ValidationError("tokenizer_config.json has an invalid tokenizer_class")
    model_max_length = tokenizer_config.get("model_max_length")
    if model_max_length is not None and (
        type(model_max_length) is not int or model_max_length <= 0
    ):
        raise ValidationError("tokenizer_config.json has an invalid model_max_length")

    payload_names = _TOKENIZER_PAYLOAD_NAMES.intersection(files)
    if not payload_names:
        raise ValidationError(
            "A ready Hugging Face base model requires tokenizer vocabulary/model files"
        )
    if "tokenizer.json" in files:
        tokenizer = _load_strict_json(
            files["tokenizer.json"],
            max_bytes=MAX_MODEL_METADATA_JSON_BYTES,
            label="tokenizer.json",
        )
        tokenizer_model = tokenizer.get("model") if isinstance(tokenizer, dict) else None
        vocabulary = tokenizer_model.get("vocab") if isinstance(tokenizer_model, dict) else None
        if (
            not isinstance(tokenizer_model, dict)
            or not isinstance(tokenizer_model.get("type"), str)
            or not tokenizer_model["type"]
            or not isinstance(vocabulary, (dict, list))
            or not vocabulary
        ):
            raise ValidationError("tokenizer.json does not contain a usable tokenizer model")
    if "vocab.json" in files:
        vocabulary = _load_strict_json(
            files["vocab.json"],
            max_bytes=MAX_MODEL_METADATA_JSON_BYTES,
            label="vocab.json",
        )
        if not isinstance(vocabulary, dict) or not vocabulary or any(
            not isinstance(token, str) or type(index) is not int or index < 0
            for token, index in vocabulary.items()
        ):
            raise ValidationError("vocab.json is not a token-to-index mapping")
    if "vocab.txt" in files:
        _validate_text_vocabulary(files["vocab.txt"])
    for name in {"tokenizer.model", "spiece.model", "sentencepiece.bpe.model"} & files.keys():
        if files[name].size_bytes < 8:
            raise ValidationError(f"Tokenizer model file {name!r} is empty or truncated")


def _required_json_object(
    files: dict[str, _PreparedFile],
    relative_path: str,
    label: str,
) -> dict:
    item = files.get(relative_path)
    if item is None:
        raise ValidationError(f"A ready safetensors base model requires {relative_path}")
    value = _load_strict_json(
        item,
        max_bytes=MAX_MODEL_METADATA_JSON_BYTES,
        label=label,
    )
    if not isinstance(value, dict) or not value:
        raise ValidationError(f"{label} must be a non-empty JSON object")
    return value


def _validate_text_vocabulary(item: _PreparedFile) -> None:
    if item.size_bytes == 0 or item.size_bytes > MAX_MODEL_METADATA_JSON_BYTES:
        raise ValidationError("vocab.txt is empty or exceeds the validation limit")
    try:
        item.spool.seek(0)
        text = item.spool.read().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("vocab.txt must be UTF-8 text") from exc
    finally:
        item.spool.seek(0)
    if not any(line.strip() for line in text.splitlines()):
        raise ValidationError("vocab.txt contains no tokens")


def _load_strict_json(
    item: _PreparedFile,
    *,
    max_bytes: int,
    label: str,
) -> Any:
    if item.size_bytes > max_bytes:
        raise ValidationError(f"{label} exceeds the safe validation byte limit")
    try:
        item.spool.seek(0)
        return json.load(item.spool, object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"{label} is not valid JSON") from exc
    finally:
        item.spool.seek(0)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValidationError(f"JSON object contains duplicate key {key!r}")
        value[key] = item
    return value


def _resolve_references(
    db: Session,
    project: Project,
    data: ArtifactPackageCreate,
) -> list[tuple[Any, ArtifactPackage]]:
    resolved = []
    for reference in data.references:
        target = db.get(ArtifactPackage, reference.target_package_id)
        if target is None:
            raise NotFoundError(f"Referenced package {reference.target_package_id} not found")
        if target.workspace_id != project.workspace_id:
            raise ValidationError("Package references cannot cross workspaces")
        retention = _locked_package_retention(db, target.id)
        if (
            retention.archived_at is not None
            or retention.purged_at is not None
        ):
            raise ConflictError("Package references require an active target package")
        resolved.append((reference, target))
    return resolved


def _validate_upstream_artifacts(
    db: Session,
    project_id: int,
    artifact_ids: Sequence[int],
) -> list[LineageArtifact]:
    artifacts: list[LineageArtifact] = []
    for artifact_id in artifact_ids:
        artifact = db.get(LineageArtifact, artifact_id)
        if artifact is None or artifact.project_id != project_id:
            raise ValidationError(f"Upstream lineage artifact {artifact_id} is not in project")
        artifacts.append(artifact)
    return artifacts


def _build_manifest(
    *,
    project: Project,
    data: ArtifactPackageCreate,
    prepared: Sequence[_PreparedFile],
    references: Sequence[tuple[Any, ArtifactPackage]],
    upstream_ids: Sequence[int],
    task_contract_hash: str,
) -> dict:
    return {
        "schema_version": data.schema_version,
        "project_id": project.id,
        "workspace_id": project.workspace_id,
        "package_kind": data.package_kind,
        "package_format": data.package_format,
        "display_name": data.display_name,
        "model_family": data.model_family,
        "model_type": data.model_type,
        "readiness": data.readiness,
        "deployable": data.deployable,
        "loader_policy": data.loader_policy,
        "task_contract": data.task_contract,
        "task_contract_hash": task_contract_hash,
        "sensitivity": data.sensitivity,
        "license": data.license_info,
        "runtime": data.runtime,
        "metadata": data.metadata,
        "files": [
            {
                "relative_path": item.relative_path,
                "role": item.role,
                "content_type": item.content_type,
                "checksum_sha256": item.checksum_sha256,
                "size_bytes": item.size_bytes,
                "metadata": item.metadata,
            }
            for item in prepared
        ],
        "references": [
            {
                "relationship_type": reference.relationship_type,
                "target_manifest_digest": target.manifest_digest,
                "metadata": reference.metadata,
            }
            for reference, target in sorted(
                references,
                key=lambda item: (item[0].relationship_type, item[1].manifest_digest),
            )
        ],
        "upstream_lineage_artifact_ids": list(upstream_ids),
    }


def _ensure_blob(
    db: Session,
    storage: ObjectStorage,
    *,
    workspace_id: int,
    checksum_sha256: str,
    size_bytes: int,
    content_type: str,
    source: BinaryIO,
    actor_user_id: int | None,
    verify_reused: bool,
) -> ArtifactBlob:
    existing = (
        db.query(ArtifactBlob)
        .filter(
            ArtifactBlob.workspace_id == workspace_id,
            ArtifactBlob.checksum_sha256 == checksum_sha256,
        )
        .with_for_update()
        .first()
    )
    if existing is not None:
        _validate_existing_blob(existing, checksum_sha256, size_bytes)
        if existing.status == "purged":
            if _blob_has_live_package_reference(db, existing.id):
                raise ConflictError("A live package references a purged artifact blob")
            _rehydrate_blob(storage, existing, source=source, content_type=content_type)
            existing.status = "ready"
            db.flush()
        elif verify_reused:
            try:
                _verify_storage_blob(
                    storage,
                    existing.storage_key,
                    checksum_sha256=checksum_sha256,
                    size_bytes=size_bytes,
                )
            except ConflictError:
                if _blob_has_live_package_reference(db, existing.id):
                    raise
                _rehydrate_blob(storage, existing, source=source, content_type=content_type)
        return existing

    key = blob_storage_key(workspace_id, checksum_sha256)
    stored = storage.put_stream(
        key,
        source,
        length=size_bytes,
        content_type=content_type,
    )
    if stored.checksum_sha256 != checksum_sha256 or stored.size_bytes != size_bytes:
        raise ConflictError("Object storage returned bytes that do not match their digest")
    candidate = ArtifactBlob(
        workspace_id=workspace_id,
        checksum_sha256=checksum_sha256,
        size_bytes=size_bytes,
        content_type=content_type,
        storage_key=key,
        status="ready",
        created_by_user_id=actor_user_id,
    )
    try:
        with db.begin_nested():
            db.add(candidate)
            db.flush()
        return candidate
    except IntegrityError as exc:
        concurrent = (
            db.query(ArtifactBlob)
            .filter(
                ArtifactBlob.workspace_id == workspace_id,
                ArtifactBlob.checksum_sha256 == checksum_sha256,
            )
            .first()
        )
        if concurrent is None:
            raise ConflictError("Content-addressed blob publication conflicted") from exc
        _validate_existing_blob(concurrent, checksum_sha256, size_bytes)
        if verify_reused:
            _verify_storage_blob(
                storage,
                concurrent.storage_key,
                checksum_sha256=checksum_sha256,
                size_bytes=size_bytes,
            )
        return concurrent


def _validate_existing_blob(blob: ArtifactBlob, checksum_sha256: str, size_bytes: int) -> None:
    if (
        blob.status not in {"ready", "purged"}
        or blob.checksum_sha256 != checksum_sha256
        or blob.size_bytes != size_bytes
    ):
        raise ConflictError("Registered content-addressed blob metadata is inconsistent")


def _rehydrate_blob(
    storage: ObjectStorage,
    blob: ArtifactBlob,
    *,
    source: BinaryIO,
    content_type: str,
) -> None:
    stored = storage.put_stream(
        blob.storage_key,
        source,
        length=blob.size_bytes,
        content_type=content_type,
    )
    if (
        stored.checksum_sha256 != blob.checksum_sha256
        or stored.size_bytes != blob.size_bytes
    ):
        raise ConflictError("Rehydrated artifact bytes do not match their blob tombstone")


def _verify_storage_blob(
    storage: ObjectStorage,
    key: str,
    *,
    checksum_sha256: str,
    size_bytes: int,
) -> None:
    digest = hashlib.sha256()
    actual_size = 0
    try:
        for chunk in storage.iter_bytes(key):
            digest.update(chunk)
            actual_size += len(chunk)
    except ObjectNotFoundError as exc:
        raise ConflictError("Registered artifact blob is missing from object storage") from exc
    if actual_size != size_bytes or digest.hexdigest() != checksum_sha256:
        raise ConflictError("Registered artifact blob failed checksum verification")


def _stored_object(blob: ArtifactBlob, *, content_type: str | None = None) -> StoredObject:
    return StoredObject(
        key=blob.storage_key,
        size_bytes=blob.size_bytes,
        checksum_sha256=blob.checksum_sha256,
        content_type=content_type or blob.content_type,
    )


def _ensure_no_sensitive_keys(value: Any, *, path: str = "metadata") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if any(fragment in normalized for fragment in _SENSITIVE_KEY_FRAGMENTS):
                raise ValidationError(
                    f"Sensitive field {path}.{key} cannot enter a package manifest"
                )
            _ensure_no_sensitive_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_no_sensitive_keys(item, path=f"{path}[{index}]")


def _sanitize_public_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _sanitize_public_value(item)
            for key, item in value.items()
            if not any(
                fragment in str(key).strip().lower()
                for fragment in _SENSITIVE_KEY_FRAGMENTS
            )
        }
    if isinstance(value, list):
        return [_sanitize_public_value(item) for item in value]
    return value


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._data):
            return b""
        if size < 0:
            size = len(self._data) - self._position
        end = min(len(self._data), self._position + size)
        chunk = bytes(self._data[self._position : end])
        self._position = end
        return chunk


class _IteratorBinaryReader:
    """Adapt an object-store chunk iterator to the binary ``read`` contract."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = iter(chunks)
        self._buffer = bytearray()
        self._exhausted = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            for chunk in self._chunks:
                self._buffer.extend(chunk)
            self._exhausted = True
            result = bytes(self._buffer)
            self._buffer.clear()
            return result
        while len(self._buffer) < size and not self._exhausted:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                self._exhausted = True
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result
