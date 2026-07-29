from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ModelFamily = Literal["conventional_ml", "deep_learning", "llm_finetune"]
PackageReadiness = Literal["ready", "quarantined", "legacy_unverified"]
LoaderPolicy = Literal["safe", "isolated_internal", "quarantined"]
Sensitivity = Literal["public", "workspace", "project", "restricted"]
BaseModelAccessMode = Literal["downloadable", "execution_only", "manager_only"]
BaseModelReadiness = Literal["ready", "quarantined", "failed", "archived"]


class ArtifactPackageReferenceCreate(BaseModel):
    target_package_id: int = Field(gt=0)
    relationship_type: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    metadata: dict = Field(default_factory=dict)


class ArtifactPackageCreate(BaseModel):
    package_kind: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    package_format: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_version: str = Field(default="model-package-v1", min_length=1, max_length=40)
    display_name: str | None = Field(default=None, min_length=1, max_length=255)
    model_family: ModelFamily | None = None
    model_type: str | None = Field(default=None, min_length=1, max_length=100)
    readiness: PackageReadiness = "ready"
    deployable: bool = False
    loader_policy: LoaderPolicy = "safe"
    task_contract: dict = Field(default_factory=dict)
    sensitivity: Sensitivity = "project"
    license_info: dict = Field(default_factory=dict)
    runtime: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    references: list[ArtifactPackageReferenceCreate] = Field(default_factory=list)
    retention_class: Literal["indefinite", "resume_14d", "candidate"] = "indefinite"
    pinned: bool = False

    @model_validator(mode="after")
    def validate_policy(self):
        if self.deployable and self.readiness != "ready":
            raise ValueError("Only ready packages may be deployable")
        if self.readiness == "quarantined" and self.loader_policy != "quarantined":
            raise ValueError("Quarantined packages require the quarantined loader policy")
        if self.package_kind == "resume_state" and self.deployable:
            raise ValueError("Resume-state packages cannot be deployable")
        if self.package_kind == "resume_state" and self.retention_class == "indefinite":
            self.retention_class = "resume_14d"
        if self.package_format == "legacy_zip" and (
            self.readiness != "legacy_unverified" or self.deployable
        ):
            raise ValueError("Legacy ZIP packages must be unverified and non-deployable")
        reference_keys = [
            (reference.target_package_id, reference.relationship_type)
            for reference in self.references
        ]
        if len(reference_keys) != len(set(reference_keys)):
            raise ValueError("Package references must be unique")
        return self


class ArtifactPackageFileRead(BaseModel):
    id: int
    relative_path: str
    role: str
    content_type: str
    checksum_sha256: str
    size_bytes: int
    metadata: dict = Field(default_factory=dict)


class ArtifactPackageReferenceRead(BaseModel):
    relationship_type: str
    target_package_id: int | None
    target_manifest_digest: str
    metadata: dict = Field(default_factory=dict)


class ArtifactPackageRetentionRead(BaseModel):
    retention_class: str
    pinned: bool
    expires_at: datetime | None
    archived_at: datetime | None
    archived_by_user_id: int | None
    archive_reason: str | None
    purge_after: datetime | None
    purged_at: datetime | None
    purged_by_user_id: int | None
    purge_reason: str | None
    payload_gc_processed_at: datetime | None
    legal_hold: bool


class ArtifactPackageArchiveCreate(BaseModel):
    reason: str | None = Field(default=None, min_length=3, max_length=500)


class ArtifactPackagePurgeCreate(BaseModel):
    reason: str = Field(min_length=3, max_length=500)


class ArtifactPackageRetentionUpdate(BaseModel):
    pinned: bool | None = None
    legal_hold: bool | None = None

    @model_validator(mode="after")
    def require_change(self):
        if self.pinned is None and self.legal_hold is None:
            raise ValueError("At least one retention field is required")
        return self


class ArtifactPackageRetentionActionRead(BaseModel):
    package_id: int
    archived_at: datetime
    archived_by_user_id: int | None
    archive_reason: str | None
    purge_after: datetime
    purged_at: datetime | None
    purged_by_user_id: int | None
    purge_reason: str | None


class ArtifactPackageRead(BaseModel):
    id: int
    project_id: int
    lineage_artifact_id: int
    package_kind: str
    package_format: str
    schema_version: str
    display_name: str | None
    model_family: str | None
    model_type: str | None
    readiness: str
    deployable: bool
    loader_policy: str
    manifest_digest: str
    task_contract_hash: str
    logical_size_bytes: int
    file_count: int
    sensitivity: str
    task_contract: dict
    license_info: dict
    runtime: dict
    metadata: dict
    manifest: dict
    files: list[ArtifactPackageFileRead]
    references: list[ArtifactPackageReferenceRead]
    retention: ArtifactPackageRetentionRead
    created_at: datetime


class ArtifactStorageUsageRead(BaseModel):
    project_id: int
    workspace_id: int
    package_count: int
    package_file_count: int
    logical_bytes: int
    unique_project_blob_bytes: int
    workspace_physical_bytes: int | None = None
    workspace_reclaimable_bytes: int | None = None
    workspace_deduplicated_bytes: int | None = None
    workspace_quota_limit_bytes: int | None = None
    workspace_quota_used_bytes: int | None = None
    workspace_reserved_bytes: int | None = None
    workspace_available_bytes: int | None = None
    workspace_accounting_consistent: bool | None = None


class WorkspaceArtifactQuotaUpdate(BaseModel):
    limit_bytes: int | None = Field(default=None, ge=0)
    reservation_ttl_seconds: int | None = Field(default=None, ge=60, le=2_592_000)

    @model_validator(mode="after")
    def require_change(self):
        if not self.model_fields_set:
            raise ValueError("At least one artifact quota field is required")
        return self


class WorkspaceArtifactQuotaRead(BaseModel):
    workspace_id: int
    limit_bytes: int | None
    used_bytes: int
    reserved_bytes: int
    available_bytes: int | None
    reservation_ttl_seconds: int
    active_reservation_count: int
    next_expiration_at: datetime | None
    physical_bytes: int
    accounting_consistent: bool
    updated_at: datetime | None


class ArtifactStorageReservationRead(BaseModel):
    id: int
    workspace_id: int
    project_id: int
    owner_type: str
    owner_id: int
    idempotency_key: str
    reserved_bytes: int
    committed_bytes: int
    status: str
    expires_at: datetime
    committed_at: datetime | None
    released_at: datetime | None
    release_reason: str | None
    artifact_package_id: int | None
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ArtifactStorageReservationRelease(BaseModel):
    reason: str = Field(min_length=1, max_length=500)


class BaseModelCatalogFields(BaseModel):
    provider: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9_.-]*$")
    source_model_id: str = Field(min_length=1, max_length=255)
    exact_revision: str = Field(min_length=1, max_length=255)
    display_name: str = Field(min_length=1, max_length=255)
    model_family: ModelFamily
    model_type: str = Field(min_length=1, max_length=100)
    license_name: str = Field(min_length=1, max_length=255)
    license_url: str | None = Field(default=None, max_length=1024)
    license_terms_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_mode: BaseModelAccessMode = "execution_only"
    metadata: dict = Field(default_factory=dict)

    @field_validator("exact_revision")
    @classmethod
    def reject_floating_revision(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.lower() in {"head", "latest", "main", "master", "stable"}:
            raise ValueError("A base model requires an exact immutable revision")
        return normalized


class BaseModelImportCreate(BaseModelCatalogFields):
    source_package_id: int = Field(gt=0)


class BaseModelUploadFile(BaseModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    role: str = Field(default="model_file", min_length=1, max_length=80)
    content_type: str = Field(
        default="application/octet-stream",
        min_length=1,
        max_length=120,
    )
    checksum_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict = Field(default_factory=dict)


class BaseModelUploadCreate(BaseModelCatalogFields):
    package_format: str = Field(
        default="safetensors",
        min_length=1,
        max_length=80,
        pattern=r"^[a-z][a-z0-9_.-]*$",
    )
    task_contract: dict = Field(default_factory=dict)
    runtime: dict = Field(default_factory=dict)
    files: list[BaseModelUploadFile] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def unique_paths(self):
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Base-model upload paths must be unique")
        return self


class BaseModelReadinessUpdate(BaseModel):
    readiness: Literal["ready", "quarantined", "failed"]
    reason: str = Field(min_length=1, max_length=1000)


class BaseModelAssetEventRead(BaseModel):
    id: int
    action: str
    prior_readiness: str | None
    resulting_readiness: str
    details: dict
    actor_user_id: int | None
    created_at: datetime


class BaseModelAssetRead(BaseModel):
    id: int
    project_id: int
    package_id: int
    provider: str
    source_model_id: str
    exact_revision: str
    display_name: str
    model_family: str
    model_type: str
    license_name: str
    license_url: str | None
    license_terms_sha256: str | None
    access_mode: str
    readiness: str
    archived_at: datetime | None
    metadata: dict
    package: ArtifactPackageRead
    created_at: datetime


class LegacyCheckpointMigrationItemRead(BaseModel):
    checkpoint_id: int
    package_id: int
    checksum_sha256: str
    size_bytes: int
    migrated: bool


class LegacyCheckpointMigrationRead(BaseModel):
    project_id: int
    migrated_count: int
    existing_count: int
    items: list[LegacyCheckpointMigrationItemRead]
