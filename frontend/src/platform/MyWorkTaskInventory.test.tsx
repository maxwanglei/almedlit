// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { request } from "@/api/client";
import MyWorkTaskInventory from "./MyWorkTaskInventory";
import type { Dataset, DatasetVersion, RoundWorkContext, TaskDefinition, TaskVersion } from "./types";

vi.mock("@/api/client", () => ({ request: vi.fn() }));

const definitions: TaskDefinition[] = [
  { id: 1, project_id: 1, key: "paper_classification_1", name: "Paper relevance", description: null },
  { id: 2, project_id: 1, key: "drug_ner", name: "drug name annoation", description: null },
];
const versions: TaskVersion[] = [
  { id: 1, project_id: 1, task_definition_id: 1, version_number: 1, task_kind: "classification", input_schema: { type: "object", required: ["text"], properties: { text: { type: "string" } } }, output_schema: { type: "string", enum: ["in vitro", "clinical", "not relevance"] }, annotation_ui: { preset: "classification" }, label_rules: {}, metrics: [], trainer_compatibility: [], content_hash: "class-v1" },
  { id: 2, project_id: 1, task_definition_id: 2, version_number: 1, task_kind: "token_labeling", input_schema: { type: "object", required: ["tokens"], properties: { tokens: { type: "array" } } }, output_schema: { type: "array", items: { type: "string" } }, annotation_ui: { preset: "token_labeling" }, label_rules: { values: ["drug", "disease"] }, metrics: [], trainer_compatibility: [], content_hash: "ner-v1" },
];
const context: RoundWorkContext = {
  project: { id: 1, name: "Tucatinib screening project" }, task: definitions[0], task_version: versions[0],
  round: { id: 1, project_id: 1, name: "Tucatinib screening project papers annotation", sequence: 1, dataset_version_id: 1, task_version_id: 1, assistance_policy: "blind", feedback_available: false, status: "open", opened_at: null, closed_at: null },
  cycle: null, guideline: null,
};
const project = { id: 1, tasks: [], workflow_task_count: 2 };
const source: DatasetVersion = { id: 1, project_id: 1, dataset_id: 1, version_number: 1, item_count: 281,
  source_uri: null, source_revision: "tucatinib-import", source_format: "other", data_schema: {},
  provenance: { ingestion: "project_corpus" }, license_info: {}, content_hash: "source-v1", artifact_package_id: null };
const dataset: Dataset = { id: 1, project_id: 1, name: "Tucatinib screening project papers", description: null, source_type: "project_corpus" };
function resources(path: string): TaskDefinition[] | TaskVersion[] | Dataset[] | DatasetVersion[] {
  const url = new URL(path, "http://localhost");
  if (url.searchParams.get("project_id") !== "1") throw new Error("Unexpected project scope");
  switch (url.pathname) {
    case "/tasks": return definitions;
    case "/tasks/versions": return versions;
    case "/datasets": return [dataset];
    case "/datasets/versions":
      // Match routes/data.py: this endpoint requires both project_id and dataset_id.
      if (url.searchParams.get("dataset_id") !== "1") throw new Error("422: dataset_id is required and must identify this source collection");
      return [source];
    default: throw new Error(`Unexpected API route: ${url.pathname}`);
  }
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(request).mockImplementation(async (path) => resources(path));
});
afterEach(cleanup);

describe("My Work saved task inventory", () => {
  it("shows the actual two saved tasks even when only classification has an assigned round", async () => {
    const navigate = vi.fn();
    render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks canManage onNavigate={navigate} />);
    await screen.findByText("drug name annoation");
    expect(screen.getByRole("heading", { name: "Annotation tasks 2" })).toBeTruthy();
    const ner = screen.getByText("drug name annoation").closest("article")!;
    expect(within(ner).getByText("Needs setup")).toBeTruthy();
    expect(within(ner).getByText("Set up this task for paper annotation.")).toBeTruthy();
    fireEvent.click(screen.getByRole("link", { name: "Continue Paper relevance" }));
    expect(navigate).toHaveBeenLastCalledWith("/my-work/rounds/1?view=annotate");
    fireEvent.click(screen.getByRole("link", { name: "Set up drug name annoation" }));
    expect(navigate).toHaveBeenLastCalledWith("/projects/1/data?tab=source&flow=import&datasetId=1&datasetVersionId=1&taskVersionId=2");
    expect(request).toHaveBeenCalledTimes(4);
    expect(request).toHaveBeenCalledWith("/datasets?project_id=1");
    expect(request).toHaveBeenCalledWith("/datasets/versions?project_id=1&dataset_id=1");
    expect(request).not.toHaveBeenCalledWith("/datasets/versions?project_id=1");
  });

  it("shows both ready tasks without suggesting either still needs setup", async () => {
    const entityVersion: TaskVersion = {
      ...versions[1], id: 3, version_number: 2, task_kind: "span_extraction",
      input_schema: versions[0].input_schema,
      output_schema: { type: "object", required: ["entities"], properties: { entities: { type: "array" } } },
      annotation_ui: { preset: "document_entities", offset_unit: "utf16", end_exclusive: true },
    };
    vi.mocked(request).mockImplementation(async (path) => path.startsWith("/tasks/versions") ? [...versions, entityVersion] : resources(path));
    const entityContext = { ...context, task: definitions[1], task_version: entityVersion, round: { ...context.round, id: 2, task_version_id: 3 } };
    render(<MyWorkTaskInventory project={project} contexts={[context, entityContext]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    await waitFor(() => expect(screen.queryByRole("status")).toBeNull());
    const inventory = screen.getByRole("region", { name: "Annotation tasks 2" });
    expect(within(inventory).getAllByText("Ready for annotation")).toHaveLength(2);
    expect(within(inventory).getByRole("link", { name: "Continue Paper relevance" })).toBeTruthy();
    expect(within(inventory).getByRole("link", { name: "Continue drug name annoation" })).toBeTruthy();
    expect(within(inventory).queryByText("Needs setup")).toBeNull();
    expect(within(inventory).queryByText(/Saved tasks without a round/)).toBeNull();
    expect(within(inventory).queryByRole("link", { name: /^Set up/ })).toBeNull();
  });

  it("rejects a collection from a different project before requesting any of its versions", async () => {
    vi.mocked(request).mockImplementation(async (path) => path === "/datasets?project_id=1" ? [{ ...dataset, project_id: 99 }] : resources(path));
    render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    expect((await screen.findByRole("alert")).textContent).toContain("different project");
    expect(vi.mocked(request).mock.calls.some(([path]) => path.startsWith("/datasets/versions"))).toBe(false);
  });

  it("rejects a source version whose parent differs from the requested collection", async () => {
    vi.mocked(request).mockImplementation(async (path) => path.startsWith("/datasets/versions") ? [{ ...source, dataset_id: 99 }] : resources(path));
    render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    expect((await screen.findByRole("alert")).textContent).toContain("different collection");
    expect(screen.queryByRole("link", { name: "Set up drug name annoation" })).toBeNull();
  });

  it("asks users to choose a collection when assigned rounds pin several source versions", async () => {
    const navigate = vi.fn();
    const secondSourceContext = { ...context, round: { ...context.round, id: 4, dataset_version_id: 3 } };
    render(<MyWorkTaskInventory project={project} contexts={[context, secondSourceContext]} canReadAllTasks canManage onNavigate={navigate} />);
    await screen.findByText("drug name annoation");
    fireEvent.click(screen.getByRole("link", { name: "Set up drug name annoation" }));
    expect(navigate).toHaveBeenLastCalledWith("/projects/1/data?tab=source&taskVersionId=2");
  });

  it("keeps ordinary annotators scoped to assigned tasks without requesting the manager task inventory", () => {
    render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks={false} canManage={false} onNavigate={vi.fn()} />);
    expect(screen.getByRole("heading", { name: "Assigned annotation tasks 1" })).toBeTruthy();
    expect(screen.getByText("Paper relevance")).toBeTruthy();
    expect(screen.queryByText("drug name annoation")).toBeNull();
    expect(request).not.toHaveBeenCalled();
  });

  it("never presents closed rounds as ready and retains tasks whose initial version was not saved", async () => {
    vi.mocked(request).mockImplementation(async (path) => path.startsWith("/tasks/versions") ? [] : resources(path));
    render(<MyWorkTaskInventory project={project} contexts={[{ ...context, round: { ...context.round, status: "closed" } }]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    await screen.findByText("drug name annoation");
    expect(screen.getAllByText("Needs setup")).toHaveLength(2);
    expect(screen.queryByRole("link", { name: "Continue Paper relevance" })).toBeNull();
    expect(screen.getAllByText("Finish task setup to begin.")).toHaveLength(2);
  });

  it("does not leak a late project response after switching projects", async () => {
    let finish!: (value: unknown) => void;
    vi.mocked(request).mockImplementation((path) => {
      if (path === "/tasks?project_id=1") return new Promise((resolve) => { finish = resolve; });
      if (path.includes("project_id=2")) return Promise.resolve([]);
      return Promise.resolve(resources(path));
    });
    const result = render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    result.rerender(<MyWorkTaskInventory project={{ id: 2, tasks: [], workflow_task_count: 0 }} contexts={[context]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    await screen.findByText("No annotation tasks have been saved for this project.");
    await act(async () => { finish(definitions); });
    expect(screen.queryByText("Paper relevance")).toBeNull();
    expect(screen.queryByText("drug name annoation")).toBeNull();
  });

  it("keeps assigned work available and offers retry when the saved task list fails", async () => {
    vi.mocked(request).mockRejectedValue(new Error("Task list unavailable"));
    render(<MyWorkTaskInventory project={project} contexts={[context]} canReadAllTasks canManage onNavigate={vi.fn()} />);
    expect((await screen.findByRole("alert")).textContent).toContain("Task list unavailable");
    expect(screen.getByRole("link", { name: "Continue Paper relevance" })).toBeTruthy();
    vi.mocked(request).mockImplementation(async (path) => resources(path));
    fireEvent.click(screen.getByRole("button", { name: "Retry loading tasks" }));
    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    await screen.findByText("drug name annoation");
  });
});
