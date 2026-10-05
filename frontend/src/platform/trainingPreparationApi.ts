import { request } from "@/api/client";
import type { TrainingDatasetVersion } from "./types";

export interface TrainingSource {
  dataset_version_id: number;
  input_mapping: Record<string, string>;
  label_set_version_id?: number;
  label_field?: string;
  annotation_round_id?: number;
  submission_ids?: number[];
  split_map_id?: number;
}
export interface TrainingPreparationDraft {
  name: string;
  task_version_id: number;
  sources: TrainingSource[];
  train_percent: number;
  validation_percent: number;
  seed: number;
  training_dataset_id?: number;
  parent_version_id?: number;
}
export interface TrainingPreview {
  ready: boolean;
  issues: Array<{ code: string; message: string; source_index?: number; item_key?: string }>;
  source_counts: Array<{ dataset_version_id: number; total_count: number; labeled_count: number; excluded_unlabeled_count: number }>;
  input_count: number; labeled_count: number; excluded_unlabeled_count: number;
  duplicate_count: number; item_count: number; group_count: number;
  split_counts: Record<string, number>;
  manifest_hash: string;
  resolved_sources?: TrainingSource[];
}
export interface TrainingPreparationResult {
  training_dataset: { id: number; project_id: number; name: string; task_version_id: number };
  training_dataset_version: TrainingDatasetVersion;
  preview: TrainingPreview;
}
export function previewTrainingDataset(projectId: number, draft: TrainingPreparationDraft): Promise<TrainingPreview> {
  return request(`/projects/${projectId}/training-datasets/preview`, { method: "POST", body: JSON.stringify(draft) });
}
export function prepareTrainingDataset(projectId: number, draft: TrainingPreparationDraft, preview: TrainingPreview, idempotencyKey: string): Promise<TrainingPreparationResult> {
  return request(`/projects/${projectId}/training-datasets/prepare`, { method: "POST", body: JSON.stringify({ ...draft,
    sources: preview.resolved_sources ?? draft.sources, preview_manifest_hash: preview.manifest_hash, idempotency_key: idempotencyKey,
  }) });
}
