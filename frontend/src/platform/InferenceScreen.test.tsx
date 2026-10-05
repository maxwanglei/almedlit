// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import InferenceScreen from "./InferenceScreen";
import { EMPTY_PLATFORM_PROJECT_DATA, type PlatformProjectData } from "./types";
import type { PredictionRun } from "./predictionApi";

const mocks = vi.hoisted(() => ({ list: vi.fn(), create: vi.fn(), results: vi.fn(), review: vi.fn(), retry: vi.fn(), download: vi.fn() }));
vi.mock("./predictionApi", () => ({
  listPredictionRuns: mocks.list, createPredictionRun: mocks.create,
  getPredictionResults: mocks.results, createPredictionReview: mocks.review,
  retryPredictionRun: mocks.retry,
  downloadPredictions: mocks.download,
  predictionDownloadUrl: (project: number, run: number, format: string) => `/api/projects/${project}/prediction-runs/${run}/download?format=${format}`,
}));

const run: PredictionRun = { id: 91, project_id: 7, name: "Tucatinib predictions", dataset_version_id: 22, task_version_id: 12, model_version_id: 81, status: "completed", result_count: 3, failure_reason: null, completed_at: null };
const data: PlatformProjectData = {
  ...EMPTY_PLATFORM_PROJECT_DATA,
  datasets: [{ id: 2, project_id: 7, name: "Tucatinib papers", description: null, source_type: "project_corpus" }],
  datasetVersions: [{ id: 22, project_id: 7, dataset_id: 2, version_number: 1, item_count: 3, source_uri: null, source_revision: "abc", source_format: "jsonl", data_schema: {}, provenance: {}, license_info: {}, content_hash: "abc", artifact_package_id: null }],
  taskVersions: [{ id: 12, project_id: 7, task_definition_id: 1, version_number: 1, task_kind: "classification", input_schema: {}, output_schema: {}, label_rules: {}, annotation_ui: {}, metrics: [], trainer_compatibility: [], content_hash: "task" }],
  models: [{ id: 8, project_id: 7, name: "Study classifier", description: null, lifecycle_status: "active" }],
  modelVersions: [{ id: 81, project_id: 7, registered_model_id: 8, version_number: 1, parent_version_id: null, task_version_id: 12, training_dataset_version_id: null, family: "conventional_ml", framework: "scikit-learn", base_model: {}, training_method: "supervised", recipe_key: "tfidf_logistic_regression", recipe_version: "1", parameters: {}, metrics: {}, runtime_digest: "runtime", content_hash: "model", checkpoint_package_id: 1 }],
};
const rows = [
  { dataset_item_id: 101, stable_key: "PMID-1", title: "First paper", text: "Evidence", prediction: "positive", confidence: 0.9, uncertainty: 0.1, already_submitted: false, protected: false },
  { dataset_item_id: 102, stable_key: "PMID-2", title: "Submitted paper", text: "Evidence", prediction: "negative", confidence: 0.8, uncertainty: 0.2, already_submitted: true, protected: false },
  { dataset_item_id: 103, stable_key: "PMID-3", title: "Holdout paper", text: "Evidence", prediction: "positive", confidence: 0.7, uncertainty: 0.3, already_submitted: false, protected: true },
];

beforeEach(() => {
  vi.clearAllMocks();
  sessionStorage.clear();
  window.history.replaceState({}, "", "/projects/7/inference");
  mocks.list.mockResolvedValue([]);
  mocks.results.mockResolvedValue({ items: rows, total: 3, offset: 0, limit: 50 });
});
afterEach(() => { cleanup(); vi.useRealTimers(); });

const mount = (overrides: Partial<React.ComponentProps<typeof InferenceScreen>> = {}) => render(<InferenceScreen projectId={7} data={data} currentUserId={1} canReview onOpenRound={vi.fn()} onRefresh={vi.fn().mockResolvedValue(undefined)} {...overrides} />);

describe("Standalone inference", () => {
  it("announces export failures instead of downloading an error response", async () => {
    window.history.replaceState({}, "", "/projects/7/inference?predictionRunId=91");
    mocks.list.mockResolvedValue([run]);
    mocks.download.mockRejectedValue(new Error("Prediction export is unavailable. Retry later."));
    mount();
    fireEvent.click(await screen.findByRole("link", { name: "Download CSV" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Retry later");
    expect(mocks.download).toHaveBeenCalledWith(7, 91, "csv");
  });

  it("preserves request identity after a failed create and preselects the source link", async () => {
    window.history.replaceState({}, "", "/projects/7/inference?datasetVersionId=22");
    mocks.create.mockRejectedValueOnce(new Error("Connection interrupted")).mockResolvedValueOnce({ ...run, status: "queued" });
    mount();
    await screen.findByText("No prediction runs yet");
    expect((screen.getByLabelText("Source dataset") as HTMLSelectElement).value).toBe("22");
    fireEvent.click(screen.getByRole("button", { name: "Continue to model" }));
    fireEvent.change(screen.getByLabelText("Prediction name"), { target: { value: "Tucatinib predictions" } });
    fireEvent.change(screen.getByLabelText("Trained model"), { target: { value: "81" } });
    fireEvent.click(screen.getByRole("button", { name: "Review predictions" }));
    fireEvent.click(screen.getByRole("button", { name: "Run predictions" }));
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Run predictions" }));
    await waitFor(() => expect(mocks.create).toHaveBeenCalledTimes(2));
    expect(mocks.create.mock.calls[0][1].request_key).toBe(mocks.create.mock.calls[1][1].request_key);
    expect(mocks.create.mock.calls[1][1].dataset_version_id).toBe(22);
  });

  it("excludes protected and submitted results, then opens a pinned original-item review", async () => {
    mocks.list.mockResolvedValue([run]);
    mocks.review.mockResolvedValue({ round_id: 55, item_count: 1, excluded_submitted: 0, excluded_protected: 0 });
    const onOpenRound = vi.fn();
    mount({ onOpenRound });
    fireEvent.click(await screen.findByRole("button", { name: "View results" }));
    await screen.findByText("First paper");
    expect((screen.getByLabelText("Review PMID-2") as HTMLInputElement).disabled).toBe(true);
    expect((screen.getByLabelText("Review PMID-3") as HTMLInputElement).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Select eligible results on this page" }));
    fireEvent.click(screen.getByRole("button", { name: "Create review round" }));
    await waitFor(() => expect(onOpenRound).toHaveBeenCalledWith(55));
    expect(mocks.review.mock.calls[0][2]).toMatchObject({ dataset_item_ids: [101], include_submitted: false });
    expect(screen.getByRole("link", { name: "Download CSV" }).getAttribute("href")).toContain("/91/download?format=csv");
  });

  it("restores a completed run from its URL and allows explicit submitted re-review", async () => {
    window.history.replaceState({}, "", "/projects/7/inference?predictionRunId=91");
    mocks.list.mockResolvedValue([run]);
    mount();
    await screen.findByText("First paper");
    fireEvent.click(screen.getByLabelText("Include previously submitted annotations for another review"));
    expect((screen.getByLabelText("Review PMID-2") as HTMLInputElement).disabled).toBe(false);
    expect((screen.getByLabelText("Review PMID-3") as HTMLInputElement).disabled).toBe(true);
  });

  it("stops polling after unmount", async () => {
    vi.useFakeTimers();
    mocks.list.mockResolvedValue([{ ...run, status: "running" }]);
    const rendered = mount();
    await act(async () => { await Promise.resolve(); });
    await act(async () => { await vi.advanceTimersByTimeAsync(3_000); });
    expect(mocks.list).toHaveBeenCalledTimes(2);
    rendered.unmount();
    await act(async () => { await vi.advanceTimersByTimeAsync(6_000); });
    expect(mocks.list).toHaveBeenCalledTimes(2);
  });

  it("restores a retryable prediction draft and request key after remount", async () => {
    sessionStorage.setItem("prediction-draft:7", JSON.stringify({ name: "Tucatinib predictions", datasetVersionId: 22, modelVersionId: 81, step: 2, request: { fingerprint: JSON.stringify({ name: "Tucatinib predictions", dataset_version_id: 22, model_version_id: 81 }), key: "persisted-request-key" } }));
    mocks.create.mockResolvedValue({ ...run, status: "queued" });
    mount();
    await screen.findByText("No prediction runs yet");
    fireEvent.click(screen.getByRole("button", { name: "Run predictions" }));
    await waitFor(() => expect(mocks.create).toHaveBeenCalled());
    expect(mocks.create.mock.calls[0][1].request_key).toBe("persisted-request-key");
  });

  it("restores run selection on browser history navigation", async () => {
    mocks.list.mockResolvedValue([run]);
    mount();
    await screen.findByRole("button", { name: "View results" });
    await act(async () => {
      window.history.pushState({}, "", "/projects/7/inference?predictionRunId=91");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    await screen.findByText("First paper");
    await act(async () => {
      window.history.pushState({}, "", "/projects/7/inference");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.queryByText("First paper")).toBeNull();
  });
});
