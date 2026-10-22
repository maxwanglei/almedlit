// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import TrainingDatasetBuilder, { inheritedTrainingSource } from "./TrainingDatasetBuilder";
import { EMPTY_PLATFORM_PROJECT_DATA, type PlatformProjectData, type AnnotationRound } from "./types";
import type { TrainingPreview } from "./trainingPreparationApi";
import { publishSessionChange } from "@/auth/session";

const mocks = vi.hoisted(() => ({ preview: vi.fn(), prepare: vi.fn() }));
vi.mock("./trainingPreparationApi", () => ({ previewTrainingDataset: mocks.preview, prepareTrainingDataset: mocks.prepare }));
const data: PlatformProjectData = { ...EMPTY_PLATFORM_PROJECT_DATA,
  datasets: [{ id: 1, project_id: 7, name: "External labels", source_type: "upload", description: null }],
  datasetVersions: [{ id: 11, project_id: 7, dataset_id: 1, version_number: 1, item_count: 12,
    data_schema: { properties: { label: { type: "string" } } }, source_uri: null, source_revision: "1",
    source_format: "json", provenance: {}, license_info: {}, content_hash: "source", artifact_package_id: null }],
  taskVersions: [{ id: 2, project_id: 7, task_definition_id: 1, version_number: 1, task_kind: "instruction_tuning",
    input_schema: { properties: { prompt: { type: "string" } } }, output_schema: {}, label_rules: {}, annotation_ui: {},
    metrics: [], trainer_compatibility: [], content_hash: "task" }],
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
  it("shows split adjustment warnings and allows saving a ready preview", async () => {
    const user = userEvent.setup();
    const adjustedPreview: TrainingPreview = { ...preview, warnings: [{ code: "split_percentage_adjusted", message: "Protected evaluation groups require train: 9, validation: 1, test: 2 instead of the requested split.", source_index: null, item_key: null }] };
    mocks.preview.mockResolvedValueOnce(adjustedPreview);
    mocks.prepare.mockResolvedValueOnce({ training_dataset_version: { id: 33, name: "Prepared", version_number: 1 } });
    render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
    await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
    await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
    const warnings = await screen.findByRole("status", { name: "Training dataset warnings" });
    expect(warnings.textContent).toContain(adjustedPreview.warnings?.[0].message);
    expect(warnings.textContent).not.toContain("Source 1:");
    const saveButton = screen.getByRole("button", { name: "Create training dataset" });
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);
    await user.click(saveButton);
    await waitFor(() => expect(mocks.prepare).toHaveBeenCalledWith(7, expect.anything(), adjustedPreview, expect.any(String)));
    expect(await screen.findByText(/Prepared · v1 is ready/)).toBeTruthy();
  });
  it("removes historical metadata and chooses one immutable label source for a new version", () => {
    expect(inheritedTrainingSource({ dataset_version_id: 11, input_mapping: { text: "abstract" }, label_set_version_id: 8, label_field: "label", annotation_round_id: 9, submission_ids: [10] })).toEqual({ dataset_version_id: 11, input_mapping: { text: "abstract" }, label_set_version_id: 8 });
  });
  it("restores the exact reviewed submissions and request key after a lost save response and reload", async () => {
    const user = userEvent.setup();
    const annotatedData = { ...data, rounds: [completedRound(9, 11)] };
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

function completedRound(id: number, datasetVersionId: number, status = "closed"): AnnotationRound {
  return { id, project_id: 7, cycle_id: null, name: `Round ${id}`, sequence: id,
    dataset_version_id: datasetVersionId, task_version_id: 2, guideline_revision_id: null,
    selection_set_version_id: null, feedback_set_version_id: null, assistance_policy: "blind",
    reannotation_mode: "full_dataset", annotator_user_ids: [], open_to_all_annotators: true,
    reason: null, status, opened_at: null, closed_at: null };
}
it("defaults to the latest task revision and requires an actual imported label field", async () => {
  const user = userEvent.setup();
  render(<TrainingDatasetBuilder projectId={7} data={{ ...data, datasetVersions: data.datasetVersions.map((version) => ({ ...version, data_schema: {} })), taskVersions: [
    { ...data.taskVersions[0], id: 2, version_number: 1 },
    { ...data.taskVersions[0], id: 3, version_number: 2 },
  ] }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  expect(screen.getByLabelText("Task")).toHaveProperty("value", "3");
  expect(screen.getByLabelText("Label field")).toHaveProperty("value", "");
  await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
  await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
  expect(mocks.preview).not.toHaveBeenCalled();
});
it("keeps annotation defaults on the launched source and chooses the latest completed round", async () => {
  const user = userEvent.setup();
  render(<TrainingDatasetBuilder projectId={7} initialDatasetId={1} data={{ ...data, rounds: [
    completedRound(1, 99), completedRound(2, 11), completedRound(3, 11), completedRound(4, 11, "draft"),
  ] }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  await user.click(screen.getByRole("radio", { name: "Import external labeled data" }));
  await user.click(screen.getByRole("radio", { name: "Use project annotations" }));
  expect(screen.getByLabelText("Source dataset version")).toHaveProperty("value", "11");
  expect(screen.getByLabelText("Labels from")).toHaveProperty("value", "round:3");
});

it("chooses the newest task across independent task revision series", () => {
  render(<TrainingDatasetBuilder projectId={7} data={{ ...data, taskVersions: [
    { ...data.taskVersions[0], id: 20, task_definition_id: 1, version_number: 9 },
    { ...data.taskVersions[0], id: 21, task_definition_id: 2, version_number: 1 },
  ] }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  expect(screen.getByLabelText("Task")).toHaveProperty("value", "21");
});

function labelSet(id: number, name: string, versionNumber: number, sourceKind: "human" | "imported" = "human"): PlatformProjectData["labelSets"][number] {
  return { id, project_id: 7, dataset_version_id: 11, task_version_id: 2, parent_version_id: null,
    name, version_number: versionNumber, source_kind: sourceKind, composition_policy: "replace",
    labels: {}, label_count: 12, content_hash: `labels-${id}` };
}

it("chooses the newest label set across independently named revision series", () => {
  render(<TrainingDatasetBuilder projectId={7} initialDatasetVersionId={11} data={{ ...data, labelSets: [
    labelSet(20, "Initial labels", 9), labelSet(21, "Adjudicated labels", 1),
  ] }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  expect(screen.getByLabelText("Labels from")).toHaveProperty("value", "layer:21");
});

it("chooses the newest closed round across independent cycle sequences", () => {
  render(<TrainingDatasetBuilder projectId={7} initialDatasetVersionId={11} data={{ ...data, rounds: [
    { ...completedRound(20, 11), cycle_id: 1, sequence: 9 },
    { ...completedRound(21, 11), cycle_id: 2, sequence: 1 },
  ] }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  expect(screen.getByLabelText("Labels from")).toHaveProperty("value", "round:21");
});

it("keeps imported label sets out of project annotation mode", async () => {
  const user = userEvent.setup();
  render(<TrainingDatasetBuilder projectId={7} initialDatasetVersionId={11} data={{ ...data,
    labelSets: [labelSet(30, "Imported labels", 9, "imported")], rounds: [completedRound(21, 11)],
  }} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  await user.click(screen.getByRole("radio", { name: "Import external labeled data" }));
  await user.click(screen.getByRole("radio", { name: "Use project annotations" }));
  expect(screen.getByLabelText("Labels from")).toHaveProperty("value", "round:21");
});

it("shows a failed parent continuation and retries without creating another version", async () => {
  const user = userEvent.setup();
  mocks.prepare.mockResolvedValue({ training_dataset_version: { id: 33, name: "Prepared", version_number: 1 } });
  const onCreated = vi.fn().mockRejectedValueOnce(new Error("Summary unavailable")).mockResolvedValue(undefined);
  render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={onCreated} onCancel={vi.fn()} />);
  await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
  await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
  await user.click(await screen.findByRole("button", { name: "Create training dataset" }));
  expect((await screen.findByRole("alert")).textContent).toContain("Summary unavailable");
  await user.click(screen.getByRole("button", { name: "Retry opening saved dataset" }));
  await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(2));
  expect(mocks.prepare).toHaveBeenCalledTimes(1);
  await waitFor(() => expect(sessionStorage.getItem("al-medlit:training-preparation:7:new:all")).toBeNull());
});
it.each([false, true])("preserves a replacement draft after a saved continuation resolves, session changed=%s", async (changeSession) => {
  const user = userEvent.setup();
  let resolveContinuation!: () => void;
  const continuation = new Promise<void>((resolve) => { resolveContinuation = resolve; });
  const onCreated = vi.fn(() => continuation);
  const key = "al-medlit:training-preparation:7:new:all";
  sessionStorage.setItem(key, JSON.stringify({ saved: { id: 33, name: "Prepared", version_number: 1 } }));
  const mounted = render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={onCreated} onCancel={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Open saved dataset" }));
  expect(onCreated).toHaveBeenCalledTimes(1);
  if (changeSession) publishSessionChange(false);
  mounted.unmount();
  const replacement = JSON.stringify({ draft: { name: "Another user's dataset" } });
  sessionStorage.setItem(key, replacement);
  await act(async () => { resolveContinuation(); await continuation; });
  expect(sessionStorage.getItem(key)).toBe(replacement);
});
it("clears its saved draft after a successful same-session navigation unmounts the builder", async () => {
  const user = userEvent.setup();
  const key = "al-medlit:training-preparation:7:new:all";
  sessionStorage.setItem(key, JSON.stringify({ saved: { id: 33, name: "Prepared", version_number: 1 } }));
  const mounted = render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => { mounted.unmount(); }} onCancel={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Open saved dataset" }));
  expect(sessionStorage.getItem(key)).toBeNull();
});
it("does not restore a cleared session draft when an in-flight save completes before unmount", async () => {
  const user = userEvent.setup();
  let resolveSave!: (value: { training_dataset_version: { id: number; name: string; version_number: number } }) => void;
  const save = new Promise<{ training_dataset_version: { id: number; name: string; version_number: number } }>((resolve) => { resolveSave = resolve; });
  mocks.prepare.mockReturnValueOnce(save);
  const onCreated = vi.fn(async () => undefined);
  render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={onCreated} onCancel={vi.fn()} />);
  await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
  await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
  await user.click(await screen.findByRole("button", { name: "Create training dataset" }));
  expect(mocks.prepare).toHaveBeenCalledTimes(1);
  publishSessionChange(false);
  await act(async () => { resolveSave({ training_dataset_version: { id: 33, name: "Prepared", version_number: 1 } }); await save; });
  expect(onCreated).not.toHaveBeenCalled();
  expect(sessionStorage.getItem("al-medlit:training-preparation:7:new:all")).toBeNull();
});
it("prevents duplicate source selections before preview", async () => {
  const user = userEvent.setup();
  render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Add another source" }));
  const sourceSelectors = screen.getAllByLabelText("Source dataset version");
  expect(sourceSelectors[1].querySelector('option[value="11"]')).toHaveProperty("disabled", true);
});
it("limits the source list to thirty before preview", async () => {
  render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  const add = screen.getByRole("button", { name: "Add another source" });
  for (let index = 1; index < 30; index += 1) fireEvent.click(add);
  expect(screen.getByRole("button", { name: "Add another source" })).toHaveProperty("disabled", true);
});
it("shows a global preview issue without a source-one prefix", async () => {
  const user = userEvent.setup();
  mocks.preview.mockResolvedValue({ ...preview, issues: [{ code: "global", message: "Global failure", source_index: null }] });
  render(<TrainingDatasetBuilder projectId={7} data={data} onImport={vi.fn()} onCreated={async () => undefined} onCancel={vi.fn()} />);
  await user.type(screen.getByLabelText("Training dataset name"), "Prepared");
  await user.click(screen.getByRole("button", { name: "Preview training dataset" }));
  expect((await screen.findByText("Global failure")).textContent).toBe("Global failure");
});
