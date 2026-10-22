import { formatStatus } from "./components";
import type { ModelVersion } from "./types";

export function baseModelName(baseModel: Record<string, unknown>): string {
  for (const key of ["display_name", "source_model_id", "model_id", "name"]) {
    const value = baseModel[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return Object.keys(baseModel).length ? "Recorded in manifest" : "From scratch";
}

export function baseModelLineage(baseModel: Record<string, unknown>): {
  asset: string;
  source: string;
  revision: string;
} {
  const first = (...keys: string[]): string | null => {
    for (const key of keys) {
      const value = baseModel[key];
      if (
        (typeof value === "string" || typeof value === "number") &&
        String(value).trim()
      ) {
        return String(value);
      }
    }
    return null;
  };
  const empty = Object.keys(baseModel).length === 0;
  return {
    asset:
      first("base_model_asset_id", "asset_id", "id") ??
      (empty ? "Not required" : "Not recorded"),
    source:
      first("source_model_id", "source", "model_id", "name") ??
      (empty ? "From scratch" : "Not recorded"),
    revision:
      first("exact_revision", "revision", "commit") ??
      (empty ? "Not applicable" : "Not recorded"),
  };
}

export function metricSummary(metrics: Record<string, number | null>): string {
  const entries = Object.entries(metrics)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .slice(0, 3)
    .map(([name, value]) => `${formatStatus(name)} ${value.toFixed(3)}`);
  return entries.join(" · ") || "No aggregate metrics";
}

export function creatorName(
  run: Pick<
    ModelVersion,
    "creator_display_name" | "creator_username" | "created_by_user_id"
  >,
): string {
  return (
    run.creator_display_name?.trim() ||
    run.creator_username?.trim() ||
    (run.created_by_user_id ? `User ${run.created_by_user_id}` : "System")
  );
}

