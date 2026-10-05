import { request, requestBlob } from "@/api/client";

export interface PredictionRun {
  id: number;
  project_id: number;
  name: string;
  dataset_version_id: number;
  task_version_id: number;
  model_version_id: number;
  status: "planned" | "queued" | "running" | "completed" | "failed";
  result_count: number;
  failure_reason: string | null;
  completed_at: string | null;
  created_at?: string | null;
}

export interface PredictionResult {
  dataset_item_id: number;
  stable_key: string;
  title: string;
  text: string;
  prediction: unknown;
  confidence: number | null;
  uncertainty: number | null;
  already_submitted: boolean;
  protected: boolean;
}

export interface PredictionResultPage {
  items: PredictionResult[];
  total: number;
  offset: number;
  limit: number;
}

export interface PredictionDraft {
  name: string;
  dataset_version_id: number;
  model_version_id: number;
  request_key: string;
}

export interface PredictionReviewDraft {
  name: string;
  dataset_item_ids: number[];
  include_submitted: boolean;
  request_key: string;
}

const base = (projectId: number) => `/projects/${projectId}/prediction-runs`;

export const listPredictionRuns = (projectId: number): Promise<PredictionRun[]> => request(base(projectId));
export const getPredictionRun = (projectId: number, runId: number): Promise<PredictionRun> => request(`${base(projectId)}/${runId}`);
export const createPredictionRun = (projectId: number, draft: PredictionDraft): Promise<PredictionRun> => request(base(projectId), { method: "POST", body: JSON.stringify(draft) });
export const retryPredictionRun = (projectId: number, runId: number): Promise<PredictionRun> => request(`${base(projectId)}/${runId}/retry`, { method: "POST" });
export const getPredictionResults = (projectId: number, runId: number, offset = 0): Promise<PredictionResultPage> => request(`${base(projectId)}/${runId}/results?offset=${offset}&limit=50`);
export const predictionDownloadUrl = (projectId: number, runId: number, format: "csv" | "jsonl"): string => `/api${base(projectId)}/${runId}/download?format=${format}`;
export const downloadPredictions = (projectId: number, runId: number, format: "csv" | "jsonl"): Promise<Blob> => requestBlob(`${base(projectId)}/${runId}/download?format=${format}`);
export const createPredictionReview = (projectId: number, runId: number, draft: PredictionReviewDraft): Promise<{round_id: number; item_count: number; excluded_submitted: number; excluded_protected: number}> => request(`${base(projectId)}/${runId}/review-round`, { method: "POST", body: JSON.stringify(draft) });
