from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType
from al_medlit.lineage.models import ImmutableRecordError


class ArtifactBlob(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A physical object deduplicated only inside one trusted workspace."""

    __tablename__ = "artifact_blobs"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "checksum_sha256",
            name="uq_artifact_blobs_workspace_checksum",
        ),
        UniqueConstraint("storage_key", name="uq_artifact_blobs_storage_key"),
        CheckConstraint("size_bytes >= 0", name="ck_artifact_blobs_nonnegative_size"),
    )

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    content_type: Mapped[str] = mapped_column(String(120))
    storage_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(30), default="ready", index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )


class WorkspaceArtifactQuota(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Locked workspace counter used for atomic storage admission."""

    __tablename__ = "workspace_artifact_quotas"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            name="uq_workspace_artifact_quotas_workspace",
        ),
        CheckConstraint(
            "limit_bytes IS NULL OR limit_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_limit",
        ),
        CheckConstraint(
            "used_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_used",
        ),
        CheckConstraint(
            "reserved_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_reserved",
        ),
        CheckConstraint(
            "reservation_ttl_seconds > 0",
            name="ck_workspace_artifact_quotas_positive_ttl",
        ),
    )

    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        index=True,
    )
    limit_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    used_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    reserved_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    reservation_ttl_seconds: Mapped[int] = mapped_column(Integer, default=604_800)
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )


class ArtifactStorageReservation(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Durable quota claim owned by one immutable training run."""

    __tablename__ = "artifact_storage_reservations"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "idempotency_key",
            name="uq_artifact_storage_reservations_workspace_key",
        ),
        UniqueConstraint(
            "owner_type",
            "owner_id",
            name="uq_artifact_storage_reservations_owner",
        ),
        CheckConstraint(
            "reserved_bytes > 0",
            name="ck_artifact_storage_reservations_positive_reserved",
        ),
        CheckConstraint(
            "committed_bytes >= 0",
            name="ck_artifact_storage_reservations_nonnegative_committed",
        ),
    )

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    owner_type: Mapped[str] = mapped_column(String(40), index=True)
    owner_id: Mapped[int] = mapped_column(Integer, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(200))
    reserved_bytes: Mapped[int] = mapped_column(BigInteger)
    committed_bytes: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    committed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    released_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    release_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"),
        nullable=True,
        index=True,
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )


class ArtifactPackage(Base, IntPrimaryKeyMixin, TimestampMixin):
    """An immutable, project-scoped logical package over content-addressed blobs."""

    __tablename__ = "artifact_packages"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "manifest_digest",
            name="uq_artifact_packages_project_manifest",
        ),
        CheckConstraint(
            "logical_size_bytes >= 0",
            name="ck_artifact_packages_nonnegative_size",
        ),
        CheckConstraint("file_count > 0", name="ck_artifact_packages_positive_file_count"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    lineage_artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        unique=True,
        index=True,
    )
    manifest_blob_id: Mapped[int] = mapped_column(
        ForeignKey("artifact_blobs.id"),
        unique=True,
        index=True,
    )
    package_kind: Mapped[str] = mapped_column(String(80), index=True)
    package_format: Mapped[str] = mapped_column(String(80), index=True)
    schema_version: Mapped[str] = mapped_column(String(40), default="model-package-v1")
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_family: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    model_type: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    readiness: Mapped[str] = mapped_column(String(30), default="ready", index=True)
    deployable: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    loader_policy: Mapped[str] = mapped_column(String(40), default="safe")
    manifest_digest: Mapped[str] = mapped_column(String(64), index=True)
    task_contract_hash: Mapped[str] = mapped_column(String(64), index=True)
    logical_size_bytes: Mapped[int] = mapped_column(BigInteger)
    file_count: Mapped[int] = mapped_column(Integer)
    sensitivity: Mapped[str] = mapped_column(String(40), default="project")
    task_contract: Mapped[dict] = mapped_column(JSONType, default=dict)
    license_info: Mapped[dict] = mapped_column(JSONType, default=dict)
    runtime: Mapped[dict] = mapped_column(JSONType, default=dict)
    manifest: Mapped[dict] = mapped_column(JSONType, default=dict)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    lineage_artifact = relationship("LineageArtifact")
    manifest_blob = relationship("ArtifactBlob", foreign_keys=[manifest_blob_id])
    files = relationship(
        "ArtifactPackageFile",
        back_populates="package",
        cascade="all, delete-orphan",
        order_by="ArtifactPackageFile.relative_path",
    )
    outgoing_references = relationship(
        "ArtifactPackageReference",
        back_populates="source_package",
        foreign_keys="ArtifactPackageReference.source_package_id",
        cascade="all, delete-orphan",
    )
    incoming_references = relationship(
        "ArtifactPackageReference",
        back_populates="target_package",
        foreign_keys="ArtifactPackageReference.target_package_id",
    )
    retention = relationship(
        "ArtifactPackageRetention",
        back_populates="package",
        uselist=False,
        cascade="all, delete-orphan",
    )


class ArtifactPackageFile(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "artifact_package_files"
    __table_args__ = (
        UniqueConstraint(
            "package_id",
            "relative_path",
            name="uq_artifact_package_files_package_path",
        ),
        CheckConstraint("size_bytes >= 0", name="ck_artifact_package_files_size"),
    )

    package_id: Mapped[int] = mapped_column(ForeignKey("artifact_packages.id"), index=True)
    blob_id: Mapped[int] = mapped_column(ForeignKey("artifact_blobs.id"), index=True)
    relative_path: Mapped[str] = mapped_column(String(1024))
    role: Mapped[str] = mapped_column(String(80), index=True)
    content_type: Mapped[str] = mapped_column(String(120))
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    package = relationship("ArtifactPackage", back_populates="files")
    blob = relationship("ArtifactBlob")


class ArtifactPackageReference(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "artifact_package_references"
    __table_args__ = (
        UniqueConstraint(
            "source_package_id",
            "target_package_id",
            "relationship_type",
            name="uq_artifact_package_reference",
        ),
        CheckConstraint(
            "source_package_id <> target_package_id",
            name="ck_artifact_package_reference_not_self",
        ),
    )

    source_package_id: Mapped[int] = mapped_column(
        ForeignKey("artifact_packages.id"),
        index=True,
    )
    target_package_id: Mapped[int] = mapped_column(
        ForeignKey("artifact_packages.id"),
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(80), index=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    source_package = relationship(
        "ArtifactPackage",
        back_populates="outgoing_references",
        foreign_keys=[source_package_id],
    )
    target_package = relationship(
        "ArtifactPackage",
        back_populates="incoming_references",
        foreign_keys=[target_package_id],
    )


class ArtifactPackageRetention(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Mutable lifecycle policy kept separate from immutable package identity."""

    __tablename__ = "artifact_package_retention"

    package_id: Mapped[int] = mapped_column(
        ForeignKey("artifact_packages.id"),
        unique=True,
        index=True,
    )
    retention_class: Mapped[str] = mapped_column(String(40), default="indefinite")
    pinned: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    archive_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    purge_after: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    purged_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    purged_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    purge_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    payload_gc_processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    legal_hold: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    package = relationship("ArtifactPackage", back_populates="retention")


class BaseModelAsset(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Immutable catalog identity for an exact base-model revision.

    The package owns the immutable bytes and manifest.  Mutable readiness and
    archive state deliberately live in :class:`BaseModelAssetState` so a
    catalog operation never rewrites provenance or licensing metadata.
    """

    __tablename__ = "base_model_assets"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "provider",
            "source_model_id",
            "exact_revision",
            name="uq_base_model_assets_project_source_revision",
        ),
        UniqueConstraint("package_id", name="uq_base_model_assets_package"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    package_id: Mapped[int] = mapped_column(
        ForeignKey("artifact_packages.id"),
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(100), index=True)
    source_model_id: Mapped[str] = mapped_column(String(255), index=True)
    exact_revision: Mapped[str] = mapped_column(String(255), index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    model_family: Mapped[str] = mapped_column(String(40), index=True)
    model_type: Mapped[str] = mapped_column(String(100), index=True)
    license_name: Mapped[str] = mapped_column(String(255))
    license_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    license_terms_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    access_mode: Mapped[str] = mapped_column(String(40), index=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    package = relationship("ArtifactPackage")
    state = relationship(
        "BaseModelAssetState",
        back_populates="asset",
        uselist=False,
        cascade="all, delete-orphan",
    )
    events = relationship(
        "BaseModelAssetEvent",
        back_populates="asset",
        cascade="all, delete-orphan",
        order_by="BaseModelAssetEvent.created_at",
    )


class BaseModelAssetState(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Mutable operational state for an immutable base-model catalog entry."""

    __tablename__ = "base_model_asset_states"

    base_model_asset_id: Mapped[int] = mapped_column(
        ForeignKey("base_model_assets.id"),
        unique=True,
        index=True,
    )
    readiness: Mapped[str] = mapped_column(String(30), default="ready", index=True)
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
    )
    updated_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    asset = relationship("BaseModelAsset", back_populates="state")


class BaseModelAssetEvent(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Append-only audit trail for catalog state changes."""

    __tablename__ = "base_model_asset_events"

    base_model_asset_id: Mapped[int] = mapped_column(
        ForeignKey("base_model_assets.id"),
        index=True,
    )
    action: Mapped[str] = mapped_column(String(40), index=True)
    prior_readiness: Mapped[str | None] = mapped_column(String(30), nullable=True)
    resulting_readiness: Mapped[str] = mapped_column(String(30), index=True)
    details: Mapped[dict] = mapped_column(JSONType, default=dict)
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    asset = relationship("BaseModelAsset", back_populates="events")


_FULLY_IMMUTABLE_TYPES = (
    ArtifactPackage,
    ArtifactPackageFile,
    ArtifactPackageReference,
    BaseModelAsset,
    BaseModelAssetEvent,
)


def _reject_artifact_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(
        f"{type(target).__name__} {getattr(target, 'id', None)} is immutable"
    )


for _immutable_type in _FULLY_IMMUTABLE_TYPES:
    event.listen(_immutable_type, "before_update", _reject_artifact_mutation)
    event.listen(_immutable_type, "before_delete", _reject_artifact_mutation)


def _reject_blob_metadata_mutation(_mapper, _connection, target: ArtifactBlob) -> None:
    """Keep content identity immutable while allowing GC state transitions.

    Blob rows remain as checksum/size tombstones after physical deletion.  Only
    the operational ``status`` may move between ``ready`` and ``purged`` so a
    later publication can safely rehydrate the same content-addressed key.
    """

    state = inspect(target)
    changed = {
        attribute.key
        for attribute in state.attrs
        if attribute.history.has_changes()
    }
    metadata_changes = changed - {"status", "updated_at"}
    if metadata_changes:
        _reject_artifact_mutation(_mapper, _connection, target)
    if "status" in changed and target.status not in {"ready", "purged"}:
        raise ImmutableRecordError(
            f"ArtifactBlob {target.id} has an invalid lifecycle state"
        )


event.listen(ArtifactBlob, "before_update", _reject_blob_metadata_mutation)


_RESERVATION_MUTABLE_FIELDS = {
    "status",
    "committed_bytes",
    "expires_at",
    "committed_at",
    "released_at",
    "release_reason",
    "artifact_package_id",
    "updated_at",
}


def _protect_reservation_identity(
    _mapper,
    _connection,
    target: ArtifactStorageReservation,
) -> None:
    state = inspect(target)
    changed = {
        attribute.key
        for attribute in state.mapper.column_attrs
        if state.attrs[attribute.key].history.has_changes()
    }
    immutable_changes = changed - _RESERVATION_MUTABLE_FIELDS
    if immutable_changes:
        fields = ", ".join(sorted(immutable_changes))
        raise ImmutableRecordError(
            f"ArtifactStorageReservation {target.id} has immutable changes: {fields}"
        )


event.listen(
    ArtifactStorageReservation,
    "before_update",
    _protect_reservation_identity,
)
event.listen(
    ArtifactStorageReservation,
    "before_delete",
    _reject_artifact_mutation,
)
