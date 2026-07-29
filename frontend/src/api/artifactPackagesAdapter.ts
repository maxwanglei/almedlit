import type {
  ArtifactPackage,
  ArtifactPackageFile,
  ModelFamily,
} from "@/types/api";

interface WireArtifactPackage {
  id: number;
  project_id: number;
  package_kind: string;
  package_format: string;
  schema_version: string;
  model_family: ModelFamily | null;
  model_type: string | null;
  readiness: string;
  deployable: boolean;
  manifest_digest: string;
  logical_size_bytes: number;
  file_count: number;
  metadata?: Record<string, unknown>;
  files: Array<{
    id?: number;
    relative_path: string;
    role: string;
    content_type: string;
    size_bytes: number;
    checksum_sha256: string;
    downloadable?: boolean;
  }>;
  references: Array<{
    relationship_type: string;
    target_package_id: number | null;
    target_manifest_digest: string;
  }>;
  retention: {
    pinned: boolean;
    expires_at: string | null;
    archived_at?: string | null;
    purge_after?: string | null;
    purged_at?: string | null;
    legal_hold?: boolean;
  };
  created_at: string;
}

export function normalizeArtifactPackage(value: unknown): ArtifactPackage {
  const item = value as WireArtifactPackage & Partial<ArtifactPackage>;
  if (!("package_kind" in item)) return value as ArtifactPackage;
  const checkpointValue = item.metadata?.checkpoint_id;
  const trainingJobValue = item.metadata?.training_job_id;
  return {
    id: item.id,
    project_id: item.project_id,
    kind: item.package_kind,
    format: item.package_format,
    schema_version: item.schema_version,
    model_family: item.model_family,
    model_type: item.model_type,
    readiness: item.readiness,
    deployable: item.deployable,
    manifest_digest: item.manifest_digest,
    logical_size_bytes: item.logical_size_bytes,
    file_count: item.file_count,
    checkpoint_id: typeof checkpointValue === "number" ? checkpointValue : null,
    training_job_id: typeof trainingJobValue === "number" ? trainingJobValue : null,
    pinned: item.retention.pinned,
    champion: false,
    retention_expires_at: item.retention.expires_at,
    archived_at: item.retention.archived_at ?? null,
    purge_after: item.retention.purge_after ?? null,
    purged_at: item.retention.purged_at ?? null,
    legal_hold: item.retention.legal_hold ?? false,
    created_at: item.created_at,
    files: item.files.map((file, index): ArtifactPackageFile => ({
      id: file.id ?? index + 1,
      relative_path: file.relative_path,
      role: file.role,
      content_type: file.content_type,
      size_bytes: file.size_bytes,
      checksum_sha256: file.checksum_sha256,
      downloadable: file.downloadable ?? item.readiness !== "quarantined",
    })),
    references: item.references.map((reference) => ({
      relationship_type: reference.relationship_type,
      target_package_id: reference.target_package_id,
      manifest_digest: reference.target_manifest_digest,
    })),
  };
}

export function normalizeArtifactPackages(value: unknown): ArtifactPackage[] {
  const items = Array.isArray(value) ? value : (value as { items: unknown[] }).items;
  return (items ?? []).map(normalizeArtifactPackage);
}
