// @vitest-environment jsdom
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { publishSessionChange } from "@/auth/session";
import SourceDatasetSetup from "./SourceDatasetSetup";
import { createPaperEntityTaskVersionPayload } from "./paperTaskContracts";
import type * as PlatformApi from "./api";
import { EMPTY_PLATFORM_PROJECT_DATA, type DatasetVersion, type TaskVersion } from "./types";

const mocks = vi.hoisted(() => ({ request: vi.fn(), task: vi.fn(), round: vi.fn() }));
vi.mock("@/api/client", () => ({ request: mocks.request }));
vi.mock("./api", () => ({ createTaskWithVersion: mocks.task, createRound: mocks.round }));
vi.mock("@/components/PubmedImportPanel", () => ({ default: ({ onImported }: { onImported: (value: unknown) => void }) => <button onClick={() => onImported({ created: [{ document_id: 42 }], skipped: [{ document_id: 43 }, { document_id: null }] })}>Import selected papers</button> }));

beforeEach(() => { vi.clearAllMocks(); sessionStorage.clear(); });
afterEach(cleanup);
function classificationTask(id = 20): TaskVersion {
  return { id, project_id: 7, task_definition_id: 21, version_number: 1, task_kind: "classification", input_schema: { properties: { text: { type: "string" } } }, output_schema: { type: "string" }, label_rules: { values: ["relevant", "irrelevant"] }, annotation_ui: {}, metrics: [], trainer_compatibility: [], content_hash: "classification" };
}
function entityTask(id = 32): TaskVersion {
  return { ...createPaperEntityTaskVersionPayload(7, 22, ["Drug", "Gene"]), id, version_number: 1, content_hash: "entities" };
}
function sourceData(tasks: TaskVersion[] = []) {
  return { ...EMPTY_PLATFORM_PROJECT_DATA,
    datasets: [{ id: 12, name: "Tucatinib papers", purposes: ["annotation"] }] as typeof EMPTY_PLATFORM_PROJECT_DATA.datasets,
    datasetVersions: [{ id: 13, dataset_id: 12, version_number: 1, item_count: 2, provenance: {} }] as DatasetVersion[],
    taskDefinitions: [{ id: 21, name: "Relevance" }, { id: 22, name: "Entities" }] as typeof EMPTY_PLATFORM_PROJECT_DATA.taskDefinitions,
    taskVersions: tasks,
  };
}
describe("SourceDatasetSetup", () => {
  it("continues an older source version's open round after adding papers", async () => {
    const user = userEvent.setup(); const navigate = vi.fn();
    const data = sourceData([classificationTask(31)]);
    const prior = data.datasetVersions[0];
    if (!prior) throw new Error("Fixture requires source version");
    data.datasetVersions.push({ ...prior, id: 14, version_number: 2, item_count: 3 });
    data.rounds = [{ id: 900, project_id: 7, name: "Earlier annotation", sequence: 1, status: "open", dataset_version_id: 13, task_version_id: 31, annotator_user_ids: [9], open_to_all_annotators: false, cycle_id: null, guideline_revision_id: null, selection_set_version_id: null, feedback_set_version_id: null, assistance_policy: "blind", reannotation_mode: "full_dataset", reason: null, opened_at: null, closed_at: null }];
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={data} datasetId={12} datasetVersionId={14} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Continue annotation" }));
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/900?view=annotate");
    expect(mocks.round).not.toHaveBeenCalled();
  });

  it("starts only unassigned source records after earlier rounds finish", async () => {
    const user = userEvent.setup(); const navigate = vi.fn();
    mocks.round.mockResolvedValue({ id: 901 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={sourceData([classificationTask(31)])} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ reannotationMode: "unassigned", taskVersionId: 31, datasetVersionId: 13 }), expect.any(Function));
  });

  it("resumes a partially opened round only for its pinned source and task and opens the editor", async () => {
    const user = userEvent.setup(); const navigate = vi.fn(); const task = classificationTask(31);
    const data = sourceData([task]);
    sessionStorage.setItem("al-medlit:source-setup:7:12", JSON.stringify({ datasetId: 12, version: data.datasetVersions[0], taskVersion: task, selectedTaskId: 31, roundId: 902, roundTaskVersionId: 31, roundDatasetVersionId: 13 }));
    mocks.request.mockImplementation(async (path: string) => ({ id: 902, status: path.includes("/transition") ? "open" : "draft" }));
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={data} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    expect(mocks.request).toHaveBeenCalledWith("/rounds/902/transition?project_id=7", expect.objectContaining({ body: JSON.stringify({ status: "open" }) }));
    expect(mocks.round).not.toHaveBeenCalled();
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/902?view=annotate");
  });
  it("preserves an explicitly selected task while importing and saving a new source", async () => {
    const user = userEvent.setup(); const navigate = vi.fn();
    mocks.request.mockImplementation(async (path: string) => path === "/datasets" ? { id: 12 } : { id: 13, dataset_id: 12, version_number: 1, item_count: 2 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={{ ...EMPTY_PLATFORM_PROJECT_DATA, taskVersions: [classificationTask(31), entityTask(32)] }} initialTaskVersionId={32} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Import selected papers" }));
    await user.click(screen.getByRole("button", { name: "Create source dataset" }));
    expect(navigate).toHaveBeenLastCalledWith("/projects/7/data?tab=source&flow=import&datasetId=12&datasetVersionId=13&taskVersionId=32", "replace");
    expect(JSON.parse(sessionStorage.getItem("al-medlit:source-setup:7:12")!).selectedTaskId).toBe(32);
    expect((screen.getByLabelText("Annotation task") as HTMLSelectElement).value).toBe("32");
  });
  it("creates an explicit document NER version from a selected token task while preserving the source and original version", async () => {
    const user = userEvent.setup();
    const token = { ...entityTask(), task_kind: "token_labeling", input_schema: { required: ["tokens"] }, output_schema: { type: "array" }, annotation_ui: { preset: "token_labeling" }, label_rules: { values: ["O", "B-Drug", "I-Drug", "B-Gene"] } } as TaskVersion;
    const converted = { ...entityTask(33), version_number: 2 };
    const original = structuredClone(token);
    mocks.request.mockImplementation(async (path: string) => path.includes("/tasks/versions?") ? [token] : converted);
    mocks.round.mockResolvedValue({ id: 901 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={sourceData([token])} datasetId={12} datasetVersionId={13} initialTaskVersionId={32} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
    expect((screen.getByRole("button", { name: "Start annotating" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByText(/requires pretokenized inputs/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Enable NER for these papers" }));
    expect((await screen.findByText(/Task type: Named entities in papers/)).textContent).toContain("Drug, Gene");
    const versionRequest = mocks.request.mock.calls.find(([path]) => path === "/tasks/versions");
    expect(JSON.parse(versionRequest![1].body)).toEqual(expect.objectContaining({ task_definition_id: 22, task_kind: "span_extraction", trainer_compatibility: [] }));
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ datasetVersionId: 13, taskVersionId: 33, name: "Tucatinib papers · Entities" }), expect.any(Function));
    expect(mocks.request.mock.calls.every(([path]) => String(path).startsWith("/tasks/versions"))).toBe(true);
    expect(token).toEqual(original);
  });
  it("reuses an already-created compatible NER version after a lost response", async () => {
    const user = userEvent.setup();
    const token = { ...entityTask(), task_kind: "token_labeling", annotation_ui: { preset: "token_labeling" } } as TaskVersion;
    mocks.request.mockResolvedValue([token, { ...entityTask(33), version_number: 2 }]);
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={sourceData([token])} datasetId={12} datasetVersionId={13} initialTaskVersionId={32} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Enable NER for these papers" }));
    await screen.findByText(/Task type: Named entities in papers/);
    expect(mocks.request).toHaveBeenCalledTimes(1);
    expect((screen.getByLabelText("Annotation task") as HTMLSelectElement).value).toBe("33");
  });
  it("switches tasks without reusing a classification round and continues only an exact matching open round", async () => {
    const user = userEvent.setup(); const navigate = vi.fn();
    const classification = classificationTask(31); const entity = entityTask();
    const data = { ...sourceData([classification, entity]), rounds: [{ id: 900, name: "Relevance annotation", status: "open", dataset_version_id: 13, task_version_id: 31, annotator_user_ids: [9], open_to_all_annotators: false }] as typeof EMPTY_PLATFORM_PROJECT_DATA.rounds };
    sessionStorage.setItem("al-medlit:source-setup:7:12", JSON.stringify({ datasetId: 12, version: data.datasetVersions[0], taskVersion: classification, roundId: 900 }));
    mocks.round.mockResolvedValue({ id: 901 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={data} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Continue annotation" }));
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/900?view=annotate");
    expect(mocks.round).not.toHaveBeenCalled();
    await user.selectOptions(screen.getByLabelText("Annotation task"), "32");
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ datasetVersionId: 13, taskVersionId: 32 }), expect.any(Function));
    expect(mocks.request).not.toHaveBeenCalled();
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/901?view=annotate");
  });
  it("creates a named entity task through the same setup flow as classification", async () => {
    const user = userEvent.setup(); mocks.task.mockResolvedValue(entityTask()); mocks.round.mockResolvedValue({ id: 901 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={sourceData()} datasetId={12} datasetVersionId={13} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
    await user.selectOptions(screen.getByLabelText("Task type"), "entities");
    await user.type(screen.getByLabelText("Labels"), "Drug, Gene");
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    expect(mocks.task).toHaveBeenCalledWith(7, expect.objectContaining({ taskKind: "span_extraction", annotationMode: "paper_entities", labelValues: ["Drug", "Gene"] }), expect.any(Function), null);
    expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ datasetVersionId: 13, taskVersionId: 32 }), expect.any(Function));
  });
  it("starts a new round for v2 instead of resuming the saved v1 round when adding papers", async () => {
    const user = userEvent.setup();
    const firstVersion = { id: 13, dataset_id: 12, version_number: 1, item_count: 1, provenance: { source_document_ids: [41] } } as unknown as DatasetVersion;
    const task = classificationTask(31);
    const data = { ...EMPTY_PLATFORM_PROJECT_DATA,
      datasets: [{ id: 12, name: "Tucatinib papers", purposes: ["annotation"] }] as typeof EMPTY_PLATFORM_PROJECT_DATA.datasets,
      datasetVersions: [firstVersion], taskVersions: [task],
    };
    sessionStorage.setItem("al-medlit:source-setup:7:12", JSON.stringify({ name: "Tucatinib papers", documentIds: [41], datasetId: 12, version: firstVersion, taskVersion: task, taskDefinitionId: 21, roundId: 900, message: "Old version saved" }));
    const props = { projectId: 7, projectName: "Tucatinib", data, datasetId: 12, currentUserId: 9, canAnnotate: true, canInfer: false, onRefresh: async () => undefined, onNavigate: vi.fn() };
    const rendered = render(<SourceDatasetSetup {...props} datasetVersionId={13} />);
    rendered.rerender(<SourceDatasetSetup {...props} />);
    expect(screen.queryByRole("heading", { name: "Source dataset saved" })).toBeNull();
    expect(screen.queryByText("Old version saved")).toBeNull();
    mocks.request.mockResolvedValue({ id: 14, dataset_id: 12, version_number: 2, item_count: 3 });
    mocks.round.mockResolvedValue({ id: 901 });
    await user.click(screen.getByRole("button", { name: "Import selected papers" }));
    await user.click(screen.getByRole("button", { name: "Save new source version" }));
    await user.click(await screen.findByRole("button", { name: "Start annotating" }));
    expect(mocks.request).toHaveBeenCalledWith("/projects/7/datasets/12/versions/project-corpus", expect.objectContaining({ body: JSON.stringify({ document_ids: [41, 42, 43] }) }));
    expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ datasetVersionId: 14, taskVersionId: 31 }), expect.any(Function));
    expect(mocks.request.mock.calls.some(([path]) => String(path).includes("/rounds/900"))).toBe(false);
    expect(props.onNavigate).toHaveBeenLastCalledWith("/my-work/rounds/901?view=annotate");
  });
  it("retains imported papers and the saved collection outcome when parent refreshes unmount the setup", async () => {
    const user = userEvent.setup();
    mocks.request.mockImplementation(async (path: string) => path === "/datasets" ? { id: 12 } : { id: 13, dataset_id: 12, version_number: 1, item_count: 2 });
    function RefreshingProject() {
      const [visible, setVisible] = useState(true);
      const [ids, setIds] = useState<{ datasetId?: number; datasetVersionId?: number }>({});
      return <>
        <button onClick={() => setVisible(true)}>Finish project refresh</button>
        {visible ? <SourceDatasetSetup projectId={7} projectName="Tucatinib" data={EMPTY_PLATFORM_PROJECT_DATA}
          {...ids} currentUserId={9} canAnnotate canInfer={false}
          onImported={async () => { setVisible(false); }}
          onRefresh={async () => { setVisible(false); }}
          onNavigate={(path) => {
            const params = new URL(path, "http://localhost").searchParams;
            setIds({ datasetId: Number(params.get("datasetId")) || undefined, datasetVersionId: Number(params.get("datasetVersionId")) || undefined });
          }} /> : <p>Reloading project</p>}
      </>;
    }
    render(<RefreshingProject />);
    await user.click(screen.getByRole("button", { name: "Import selected papers" }));
    await screen.findByText("Reloading project");
    await user.click(screen.getByRole("button", { name: "Finish project refresh" }));
    expect(screen.getByRole("status", { name: "Source dataset status" }).textContent).toContain("1 paper imported; 1 existing paper reused.");
    expect(screen.getByText(/2 papers are ready/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Create source dataset" }));
    await screen.findByText("Reloading project");
    await user.click(screen.getByRole("button", { name: "Finish project refresh" }));
    expect(screen.getByRole("heading", { name: "Source dataset saved" })).toBeTruthy();
    expect(screen.getByRole("status", { name: "Source dataset status" }).textContent).toContain("Source dataset saved: Tucatinib papers · v1 · 2 papers");
    expect(screen.getByRole("button", { name: "Start annotating" })).toBeTruthy();
    expect(mocks.request).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole("button", { name: "Import selected papers" })).toBeNull();
  });
  it("snapshots only selected imported and existing documents and starts the current user's blind round", async () => {
    const user = userEvent.setup(); const navigate = vi.fn();
    mocks.request.mockImplementation(async (path: string) => path === "/datasets" ? { id: 12 } : { id: 13, dataset_id: 12, version_number: 1, item_count: 2 });
    mocks.task.mockResolvedValue(classificationTask()); mocks.round.mockResolvedValue({ id: 30 });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib test" data={EMPTY_PLATFORM_PROJECT_DATA} currentUserId={9} canAnnotate canInfer onRefresh={async () => undefined} onNavigate={navigate} />);
    await user.click(screen.getByRole("button", { name: "Import selected papers" }));
    await user.click(screen.getByRole("button", { name: "Create source dataset" }));
    await waitFor(() => expect(mocks.request).toHaveBeenCalledWith("/projects/7/datasets/12/versions/project-corpus", expect.objectContaining({ body: JSON.stringify({ document_ids: [42, 43] }) })));
    await user.type(screen.getByLabelText("Labels"), "relevant, not relevant");
    await user.click(screen.getByRole("button", { name: "Start annotating" }));
    await waitFor(() => expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ datasetVersionId: 13, taskVersionId: 20, annotatorUserIds: [9], openToAllAnnotators: false, assistancePolicy: "blind" }), expect.any(Function)));
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/30?view=annotate");
  });
  it("retries snapshot creation using the parent already saved instead of creating another collection", async () => {
    const user = userEvent.setup(); let snapshots = 0;
    mocks.request.mockImplementation(async (path: string) => { if (path === "/datasets") return { id: 12 }; if (++snapshots === 1) throw new Error("Temporary snapshot failure"); return { id: 13, dataset_id: 12, version_number: 1, item_count: 2 }; });
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={EMPTY_PLATFORM_PROJECT_DATA} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Import selected papers" }));
    await user.click(screen.getByRole("button", { name: "Create source dataset" }));
    expect((await screen.findByRole("alert")).textContent).toContain("Temporary snapshot failure");
    await user.click(screen.getByRole("button", { name: "Create source dataset" }));
    await screen.findByRole("button", { name: "Start annotating" });
    expect(mocks.request.mock.calls.filter(([path]) => path === "/datasets")).toHaveLength(1);
  });
  it("only offers existing classification tasks compatible with imported paper text", () => {
    const data = { ...EMPTY_PLATFORM_PROJECT_DATA,
      datasetVersions: [{ id: 13, dataset_id: 12, version_number: 1, item_count: 2, provenance: {} }] as DatasetVersion[],
      taskDefinitions: [
        { id: 20, name: "Prompt classifier" }, { id: 21, name: "Paper classifier" },
      ] as typeof EMPTY_PLATFORM_PROJECT_DATA.taskDefinitions,
      taskVersions: [
        { id: 30, task_definition_id: 20, version_number: 1, task_kind: "classification", input_schema: { properties: { prompt: { type: "string" } } }, output_schema: { type: "string" } },
        classificationTask(31),
      ] as unknown as TaskVersion[],
    };
    render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={data} datasetId={12} datasetVersionId={13} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
    expect(screen.queryByRole("option", { name: "Prompt classifier · v1" })).toBeNull();
    expect(screen.getByRole("option", { name: "Paper classifier · v1" })).toBeTruthy();
    expect((screen.getByLabelText("Annotation task") as HTMLSelectElement).value).toBe("31");
  });
});

it("recovers a saved parent after remount when the source snapshot failed", async () => {
  const user = userEvent.setup();
  mocks.request.mockImplementation(async (path: string) => {
    if (path === "/datasets") return { id: 12 };
    throw new Error("Snapshot offline");
  });
  const props = { projectId: 7, projectName: "Tucatinib", data: EMPTY_PLATFORM_PROJECT_DATA, currentUserId: 9, canAnnotate: true, canInfer: false, onRefresh: async () => undefined, onNavigate: vi.fn() };
  const mounted = render(<SourceDatasetSetup {...props} />);
  await user.click(screen.getByRole("button", { name: "Import selected papers" }));
  await user.click(screen.getByRole("button", { name: "Create source dataset" }));
  await screen.findByRole("alert");
  mounted.unmount();
  mocks.request.mockImplementation(async (path: string) => path === "/datasets" ? { id: 99 } : { id: 13, dataset_id: 12, version_number: 1, item_count: 2 });
  render(<SourceDatasetSetup {...props} />);
  await user.click(screen.getByRole("button", { name: "Create source dataset" }));
  await screen.findByRole("button", { name: "Start annotating" });
  expect(mocks.request.mock.calls.filter(([path]) => path === "/datasets")).toHaveLength(1);
});
it("reconciles an already opened recovered round without repeating its transition", async () => {
  const user = userEvent.setup(); const navigate = vi.fn(); const task = classificationTask(31); const data = sourceData([task]);
  sessionStorage.setItem("al-medlit:source-setup:7:12", JSON.stringify({ datasetId: 12, version: data.datasetVersions[0], taskVersion: task, selectedTaskId: 31, roundId: 902, roundTaskVersionId: 31, roundDatasetVersionId: 13 }));
  mocks.request.mockResolvedValue({ id: 902, project_id: 7, task_version_id: 31, dataset_version_id: 13, status: "open" });
  render(<SourceDatasetSetup projectId={7} projectName="Tucatinib" data={data} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={navigate} />);
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  await waitFor(() => expect(navigate).toHaveBeenCalledWith("/my-work/rounds/902?view=annotate"));
  expect(mocks.request).toHaveBeenCalledWith("/rounds/902");
  expect(mocks.request.mock.calls.some(([path]) => String(path).includes("/transition"))).toBe(false);
});
it("bounds an automatic round name to the backend limit", async () => {
  const user = userEvent.setup(); const data = sourceData([classificationTask(31)]);
  data.datasets[0].name = "P".repeat(255);
  mocks.round.mockResolvedValue({ id: 901 });
  render(<SourceDatasetSetup projectId={7} projectName="Papers" data={data} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={async () => undefined} onNavigate={vi.fn()} />);
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  expect(mocks.round.mock.calls[0][1].name.length).toBeLessThanOrEqual(255);
});

it("persists a task version that finishes after its setup unmounts", async () => {
  const user = userEvent.setup();
  let finish: (task: TaskVersion) => void = () => { throw new Error("Task request was not started"); };
  mocks.task.mockImplementation(() => new Promise<TaskVersion>((resolve) => { finish = resolve; }));
  mocks.round.mockResolvedValue({ id: 901 });
  const props = { projectId: 7, projectName: "Papers", data: sourceData(), datasetId: 12, datasetVersionId: 13, currentUserId: 9, canAnnotate: true, canInfer: false, onRefresh: async () => undefined, onNavigate: vi.fn() };
  const mounted = render(<SourceDatasetSetup {...props} />);
  await user.type(screen.getByLabelText("Labels"), "relevant, irrelevant");
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  mounted.unmount();
  await act(async () => { finish(classificationTask(31)); });
  render(<SourceDatasetSetup {...props} />);
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  expect(mocks.task).toHaveBeenCalledTimes(1);
  expect(mocks.round).toHaveBeenCalledWith(7, expect.objectContaining({ taskVersionId: 31 }), expect.any(Function));
});

it("does not restore a source draft or create its snapshot when the parent finishes after logout", async () => {
  const user = userEvent.setup();
  let finish: (value: { id: number }) => void = () => { throw new Error("Dataset request was not started"); };
  mocks.request.mockImplementation(() => new Promise<{ id: number }>((resolve) => { finish = resolve; }));
  const onRefresh = vi.fn();
  const onNavigate = vi.fn();
  const mounted = render(<SourceDatasetSetup projectId={7} projectName="Papers" data={EMPTY_PLATFORM_PROJECT_DATA} currentUserId={9} canAnnotate canInfer={false} onRefresh={onRefresh} onNavigate={onNavigate} />);
  await user.click(screen.getByRole("button", { name: "Import selected papers" }));
  await user.click(screen.getByRole("button", { name: "Create source dataset" }));
  publishSessionChange(false);
  mounted.unmount();
  await act(async () => { finish({ id: 12 }); });
  expect(sessionStorage.getItem("al-medlit:source-setup:7:new")).toBeNull();
  expect(sessionStorage.getItem("al-medlit:source-setup:7:12")).toBeNull();
  expect(mocks.request).toHaveBeenCalledTimes(1);
  expect(onRefresh).not.toHaveBeenCalled();
  expect(onNavigate).not.toHaveBeenCalled();
});

it("does not restore a source draft when its snapshot finishes after logout", async () => {
  const user = userEvent.setup();
  let finish: (value: DatasetVersion) => void = () => { throw new Error("Snapshot request was not started"); };
  mocks.request.mockImplementation((path: string) => path === "/datasets" ? Promise.resolve({ id: 12 }) : new Promise<DatasetVersion>((resolve) => { finish = resolve; }));
  const onRefresh = vi.fn();
  const onNavigate = vi.fn();
  const mounted = render(<SourceDatasetSetup projectId={7} projectName="Papers" data={sourceData()} currentUserId={9} canAnnotate canInfer={false} onRefresh={onRefresh} onNavigate={onNavigate} />);
  await user.click(screen.getByRole("button", { name: "Import selected papers" }));
  await user.click(screen.getByRole("button", { name: "Create source dataset" }));
  await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(2));
  publishSessionChange(false);
  mounted.unmount();
  const source = sourceData().datasetVersions[0];
  if (!source) throw new Error("Fixture requires source version");
  await act(async () => { finish(source); });
  expect(sessionStorage.getItem("al-medlit:source-setup:7:new")).toBeNull();
  expect(sessionStorage.getItem("al-medlit:source-setup:7:12")).toBeNull();
  expect(onRefresh).not.toHaveBeenCalled();
  expect(onNavigate).not.toHaveBeenCalled();
});

it("stops task creation after a definition finishes in an obsolete session", async () => {
  const user = userEvent.setup();
  const actualApi = await vi.importActual<typeof PlatformApi>("./api");
  mocks.task.mockImplementation(actualApi.createTaskWithVersion);
  let finish: (value: { id: number }) => void = () => { throw new Error("Task request was not started"); };
  mocks.request.mockImplementation(() => new Promise<{ id: number }>((resolve) => { finish = resolve; }));
  const mounted = render(<SourceDatasetSetup projectId={7} projectName="Papers" data={sourceData()} datasetId={12} datasetVersionId={13} currentUserId={9} canAnnotate canInfer={false} onRefresh={vi.fn()} onNavigate={vi.fn()} />);
  await user.type(screen.getByLabelText("Labels"), "relevant, irrelevant");
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  publishSessionChange(false);
  mounted.unmount();
  await act(async () => { finish({ id: 21 }); });
  expect(sessionStorage.getItem("al-medlit:source-setup:7:12")).toBeNull();
  expect(mocks.request).toHaveBeenCalledTimes(1);
  expect(mocks.round).not.toHaveBeenCalled();
});

it("does not overwrite the new session's draft when an old task version finishes", async () => {
  const user = userEvent.setup();
  const actualApi = await vi.importActual<typeof PlatformApi>("./api");
  mocks.task.mockImplementation(actualApi.createTaskWithVersion);
  let finish: (value: TaskVersion) => void = () => { throw new Error("Task version request was not started"); };
  mocks.request.mockImplementation((path: string) => path === "/tasks" ? Promise.resolve({ id: 21 }) : new Promise<TaskVersion>((resolve) => { finish = resolve; }));
  const onNavigate = vi.fn();
  const mounted = render(<SourceDatasetSetup projectId={7} projectName="Papers" data={sourceData()} datasetId={12} datasetVersionId={13} currentUserId={9} canAnnotate canInfer={false} onRefresh={vi.fn()} onNavigate={onNavigate} />);
  await user.type(screen.getByLabelText("Labels"), "relevant, irrelevant");
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  await waitFor(() => expect(mocks.request).toHaveBeenCalledTimes(2));
  publishSessionChange(false);
  mounted.unmount();
  sessionStorage.setItem("al-medlit:source-setup:7:12", "new session's source draft");
  await act(async () => { finish(classificationTask(31)); });
  expect(sessionStorage.getItem("al-medlit:source-setup:7:12")).toBe("new session's source draft");
  expect(mocks.round).not.toHaveBeenCalled();
  expect(onNavigate).not.toHaveBeenCalled();
});

it("stops a newly created round before transition when its session has ended", async () => {
  const user = userEvent.setup();
  const actualApi = await vi.importActual<typeof PlatformApi>("./api");
  mocks.round.mockImplementation(actualApi.createRound);
  let finish: (value: { id: number }) => void = () => { throw new Error("Round request was not started"); };
  mocks.request.mockImplementation(() => new Promise<{ id: number }>((resolve) => { finish = resolve; }));
  const onNavigate = vi.fn();
  const mounted = render(<SourceDatasetSetup projectId={7} projectName="Papers" data={sourceData([classificationTask(31)])} datasetId={12} datasetVersionId={13} initialTaskVersionId={31} currentUserId={9} canAnnotate canInfer={false} onRefresh={vi.fn()} onNavigate={onNavigate} />);
  await user.click(screen.getByRole("button", { name: "Start annotating" }));
  publishSessionChange(false);
  mounted.unmount();
  await act(async () => { finish({ id: 901 }); });
  expect(sessionStorage.getItem("al-medlit:source-setup:7:12")).toBeNull();
  expect(mocks.request).toHaveBeenCalledTimes(1);
  expect(onNavigate).not.toHaveBeenCalled();
});

it("preserves a new session's draft when navigation changes the session after a save", async () => {
  const user = userEvent.setup();
  mocks.request.mockImplementation(async (path: string) => path === "/datasets" ? { id: 12 } : { id: 13, dataset_id: 12, version_number: 1, item_count: 2 });
  const onRefresh = vi.fn();
  const onNavigate = vi.fn(() => {
    publishSessionChange(false);
    sessionStorage.setItem("al-medlit:source-setup:7:new", "new session's source draft");
  });
  render(<SourceDatasetSetup projectId={7} projectName="Papers" data={EMPTY_PLATFORM_PROJECT_DATA} currentUserId={9} canAnnotate canInfer={false} onRefresh={onRefresh} onNavigate={onNavigate} />);
  await user.click(screen.getByRole("button", { name: "Import selected papers" }));
  await user.click(screen.getByRole("button", { name: "Create source dataset" }));
  await waitFor(() => expect(onNavigate).toHaveBeenCalled());
  expect(sessionStorage.getItem("al-medlit:source-setup:7:new")).toBe("new session's source draft");
  expect(onRefresh).not.toHaveBeenCalled();
});
