// @vitest-environment jsdom
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TrainingDatasetBuilder, { inheritedTrainingSource } from "./TrainingDatasetBuilder";
import { EMPTY_PLATFORM_PROJECT_DATA, type Dataset, type DatasetVersion, type TaskVersion } from "./types";
import type { TrainingPreview } from "./trainingPreparationApi";

const mocks = vi.hoisted(() => ({ preview: vi.fn(), prepare: vi.fn() }));
vi.mock("./trainingPreparationApi", () => ({ previewTrainingDataset: mocks.preview, prepareTrainingDataset: mocks.prepare }));
const data = { ...EMPTY_PLATFORM_PROJECT_DATA,
  datasets: [{ id: 1, name: "External labels", source_type: "upload" }] as Dataset[],
  datasetVersions: [{ id: 11, dataset_id: 1, version_number: 1, item_count: 12 }] as DatasetVersion[],
  taskVersions: [{ id: 2, task_kind: "instruction_tuning", input_schema: { properties: { prompt: { type: "string" } } } }] as unknown as TaskVersion[],
};
const preview: TrainingPreview = { ready: true, issues: [], source_counts: [{ dataset_version_id: 11, total_count: 12, labeled_count: 12, excluded_unlabeled_count: 0 }], input_count: 12, labeled_count: 12, excluded_unlabeled_count: 0, duplicate_count: 0, item_count: 12, group_count: 12, split_counts: { train: 10, validation: 1, test: 1 }, manifest_hash: "f".repeat(64), resolved_sources: [{ dataset_version_id: 11, input_mapping: { prompt: "prompt" }, label_field: "label" }] };
beforeEach(() => { vi.clearAllMocks(); sessionStorage.clear(); mocks.preview.mockResolvedValue(preview); });
afterEach(cleanup);
describe("TrainingDatasetBuilder", () => {
  it("maps the selected task's fields and preserves the reviewed preview on a retry", async () => {
    const user = userEvent.setup(); mocks.prepare.mockRejectedValueOnce(new Error("Try again")).mockResolvedValue({ training_dataset_version: { id: 33, name: "Prepared", version_number: 1 } });
    render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
    await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
    expect((screen.getByLabelText("Source field for prompt") as HTMLInputElement).value).toBe("prompt");
    await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
    await waitFor(() => expect(mocks.preview).toHaveBeenCalledWith(7, expect.objectContaining({ sources: [{ dataset_version_id: 11, input_mapping: { prompt: "prompt" }, label_field: "label" }] })));
    await user.click(await screen.findByRole("button", { name: "Create training dataset" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Try again");
    await user.click(screen.getByRole("button", { name: "Create training dataset" }));
    await waitFor(() => expect(mocks.prepare).toHaveBeenCalledTimes(2));
    expect(mocks.prepare.mock.calls[0]?.[3]).toBe(mocks.prepare.mock.calls[1]?.[3]);
    expect(mocks.prepare.mock.calls[1]?.[2]).toEqual(preview);
  });
  it("removes historical metadata and chooses one immutable label source for a new version", () => {
    expect(inheritedTrainingSource({ dataset_version_id: 11, input_mapping: { text: "abstract" }, label_set_version_id: 8, label_field: "label", annotation_round_id: 9, submission_ids: [10] })).toEqual({ dataset_version_id: 11, input_mapping: { text: "abstract" }, label_set_version_id: 8 });
  });
  it("restores the exact reviewed submissions and request key after a lost save response and reload", async () => {
    const user = userEvent.setup();
    const annotatedData = { ...data, rounds: [{ id: 9, task_version_id: 2, dataset_version_id: 11, name: "Review" }] as typeof data.rounds };
    const pinned = { ...preview, resolved_sources: [{ dataset_version_id: 11, input_mapping: { prompt: "prompt" }, annotation_round_id: 9, submission_ids: [101] }] };
    mocks.preview.mockResolvedValueOnce(pinned).mockResolvedValue({ ...pinned, manifest_hash: "a".repeat(64), resolved_sources: [{ ...pinned.resolved_sources[0], submission_ids: [101, 102] }] });
    mocks.prepare.mockRejectedValueOnce(new Error("Response lost")).mockResolvedValue({ training_dataset_version: { id: 33, name: "Prepared", version_number: 1 } });
    const props = { projectId: 7, data: annotatedData, onImport: vi.fn(), onCreated: async () => undefined, onCancel: vi.fn() };
    const mounted = render(<TrainingDatasetBuilder {...props} />);
    await user.click(screen.getByLabelText("Training dataset name"));
    await user.paste("Prepared");
    await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
    await user.click(await screen.findByRole("button", { name: "Create training dataset" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Response lost");
    const originalRequest = mocks.prepare.mock.calls[0];
    mounted.unmount();
    render(<TrainingDatasetBuilder {...props} />);
    await user.click(await screen.findByRole("button", { name: "Create training dataset" }));
    await waitFor(() => expect(mocks.prepare).toHaveBeenCalledTimes(2));
    expect(mocks.preview).toHaveBeenCalledTimes(1);
    expect(mocks.prepare.mock.calls[1]).toEqual(originalRequest);
    expect(mocks.prepare.mock.calls[1]?.[2].resolved_sources[0].submission_ids).toEqual([101]);
  });
  it("explains missing project annotations instead of silently selecting external labels", async () => {
    const user = userEvent.setup();
    render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
    await user.click(screen.getByRole("radio", { name: "Use project annotations" }));
    expect(screen.getByText(/No submitted project annotations/).getAttribute("role")).toBe("status");
    expect(screen.queryByLabelText("Label field")).toBeNull();
    expect((screen.getByRole("button", { name: "Preview training dataset" }) as HTMLButtonElement).disabled).toBe(true);
    expect(mocks.preview).not.toHaveBeenCalled();
  });
});
