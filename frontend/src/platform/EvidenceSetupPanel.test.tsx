// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Document, DocumentStructureRead, GuidelineVersion, ProjectTask } from "@/types/api";

import EvidenceSetupPanel from "./EvidenceSetupPanel";

const mocks = vi.hoisted(() => ({
  listDocumentSummaries: vi.fn(),
  listEvidenceTargets: vi.fn(),
  listGuidelineVersions: vi.fn(),
  listProjectTasks: vi.fn(),
  listTaskAssignments: vi.fn(),
  listWorkspaceMembers: vi.fn(),
  rebuildDocumentStructure: vi.fn(),
  targetProps: vi.fn(),
}));
vi.mock("@/api/client", () => mocks);
vi.mock("./DocumentAssignmentManagement", () => ({
  DocumentTaskManagement: ({ projectId, tasks }: { projectId: number; tasks: ProjectTask[] }) => (
    <section aria-label="Document task management" data-project-id={projectId}>{tasks.map((task) => task.display_name).join(", ")}</section>
  ),
  default: ({ projectId, documents }: { projectId: number; documents: Document[] }) => (
    <section aria-label="Document assignment management" data-project-id={projectId}>
      {documents.map((document) => <p key={document.id}>{document.title} · structure {document.active_structure_version_id ?? "unprepared"}</p>)}
    </section>
  ),
}));
vi.mock("./EvidenceGuidelineManagement", () => ({
  default: ({ onChanged }: { onChanged: () => Promise<void> }) => (
    <button type="button" onClick={() => void onChanged().catch(() => undefined)}>Refresh guidelines</button>
  ),
}));
vi.mock("./EvidenceTargetManagement", () => ({
  default: (props: { activationBlockedReason?: string }) => {
    mocks.targetProps(props);
    return <section aria-label="Target management">{props.activationBlockedReason ?? "Target activation allowed"}</section>;
  },
}));

function task(projectId: number): ProjectTask {
  return {
    id: projectId * 10, project_id: projectId, annotation_type: "evidence_block", display_name: `Evidence task ${projectId}`,
    description: null, enabled: true, sort_order: 0, labels: [], settings: {},
  };
}
function document(projectId: number, structure: number | null = 1): Document {
  return {
    id: projectId * 100, project_id: projectId, external_id: `PMID-${projectId}`, title: `Paper ${projectId}`,
    text: "Clinical evidence", source: "test", metadata_: {}, sentences: [[0, 17]], active_structure_version_id: structure,
  };
}
function guideline(status = "active"): GuidelineVersion {
  return { id: 31, project_id: 7, version_label: "Evidence v1", markdown: "Review treatment evidence.", author_id: null, status };
}
const defaults = { projectId: 7, workspaceId: 5, onChanged: async () => undefined, onNavigate: vi.fn() };
const preparedStructure = {
  document_id: 700,
  active_structure_version_id: 77,
  structure_version: {
    id: 77, document_id: 700, version: 1, segmenter_name: "test", segmenter_version: "1",
    source_hash: "test-source-hash", text_length: 17, status: "ready", created_at: "2026-10-15T12:00:00Z",
  },
  range: { start_ordinal: 0, end_ordinal: 1, total_sentences: 1, has_more: false },
  sections: [{ id: 1, ordinal: 0, title: null, path: [], kind: "body", start_offset: 0, end_offset: 17, locator: null }],
  paragraphs: [{ id: 1, section_id: 1, ordinal: 0, section_ordinal: 0, start_offset: 0, end_offset: 17, locator: null }],
  sentences: [{ id: 1, section_id: 1, paragraph_id: 1, ordinal: 0, paragraph_ordinal: 0,
    start_offset: 0, end_offset: 17, text: "Clinical evidence" }],
} satisfies DocumentStructureRead;

beforeEach(() => {
  vi.resetAllMocks();
  mocks.listProjectTasks.mockImplementation(async (projectId: number) => [task(projectId)]);
  mocks.listDocumentSummaries.mockImplementation(async (projectId: number) => [document(projectId)]);
  mocks.listEvidenceTargets.mockResolvedValue([]);
  mocks.listGuidelineVersions.mockResolvedValue([]);
  mocks.listTaskAssignments.mockResolvedValue([]);
  mocks.listWorkspaceMembers.mockResolvedValue([]);
});
afterEach(cleanup);

describe("EvidenceSetupPanel", () => {
  it("retries a failed setup load before making management controls available", async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    mocks.listEvidenceTargets.mockRejectedValueOnce(new Error("Evidence targets could not load"));
    render(<EvidenceSetupPanel {...defaults} onChanged={onChanged} />);

    expect((await screen.findByRole("alert")).textContent).toContain("Evidence targets could not load");
    expect(screen.queryByLabelText("Document task management")).toBeNull();
    expect(screen.queryByLabelText("Target management")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retry evidence setup" }));

    expect(await screen.findByLabelText("Document task management")).toBeTruthy();
    expect(screen.getByText("Paper 7 · structure 1")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(mocks.listDocumentSummaries).toHaveBeenLastCalledWith(7, { scope: "all" });
    expect(mocks.listTaskAssignments).toHaveBeenLastCalledWith(7, { scope: "all" });
    expect(mocks.listWorkspaceMembers).toHaveBeenLastCalledWith(5);
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("ignores a load from the previous project and workspace", async () => {
    let finishOldLoad!: (tasks: ProjectTask[]) => void;
    const oldLoad = new Promise<ProjectTask[]>((resolve) => { finishOldLoad = resolve; });
    mocks.listProjectTasks.mockImplementation((projectId: number) => projectId === 7 ? oldLoad : Promise.resolve([task(projectId)]));
    const { rerender } = render(<EvidenceSetupPanel {...defaults} />);
    await waitFor(() => expect(mocks.listProjectTasks).toHaveBeenCalledWith(7));

    rerender(<EvidenceSetupPanel {...defaults} projectId={8} workspaceId={6} />);
    expect(await screen.findByText("Evidence task 8")).toBeTruthy();
    await act(async () => {
      finishOldLoad([task(7)]);
      await oldLoad;
    });

    expect(screen.getByText("Evidence task 8")).toBeTruthy();
    expect(screen.getByText("Paper 8 · structure 1")).toBeTruthy();
    expect(screen.queryByText("Evidence task 7")).toBeNull();
    expect(screen.queryByText("Paper 7 · structure 1")).toBeNull();
    expect(screen.getByLabelText("Document assignment management").getAttribute("data-project-id")).toBe("8");
    expect(mocks.listWorkspaceMembers).toHaveBeenLastCalledWith(6);
  });

  it("retains a prepared document after read failure and retries the read without rebuilding it", async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    mocks.listDocumentSummaries.mockResolvedValueOnce([document(7, null)])
      .mockRejectedValueOnce(new Error("Document refresh failed"))
      .mockResolvedValueOnce([document(7, 77)]);
    mocks.rebuildDocumentStructure.mockResolvedValue(preparedStructure);
    render(<EvidenceSetupPanel {...defaults} onChanged={onChanged} />);
    fireEvent.click(await screen.findByRole("button", { name: "Prepare document 700" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Document prepared, but setup could not be refreshed.");
    expect(screen.getByText("Paper 7 · structure 77")).toBeTruthy();
    expect(screen.getByText("Paper 7 is ready for evidence annotation.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Prepare document 700" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Retry evidence setup" }));

    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(mocks.rebuildDocumentStructure).toHaveBeenCalledExactlyOnceWith(700, true);
    expect(mocks.listDocumentSummaries).toHaveBeenCalledTimes(3);
    expect(onChanged).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Paper 7 · structure 77")).toBeTruthy();
  });

  it("allows target activation only after an active legacy guideline is loaded", async () => {
    mocks.listGuidelineVersions.mockResolvedValueOnce([guideline("superseded")]).mockResolvedValue([guideline()]);
    render(<EvidenceSetupPanel {...defaults} />);
    expect(await screen.findByLabelText("Target management")).toHaveProperty("textContent",
      "Create an active evidence guideline before activating targets so assignments can be adjudicated.");
    fireEvent.click(screen.getByRole("button", { name: "Refresh guidelines" }));

    expect(await screen.findByText("Target activation allowed")).toBeTruthy();
    expect(mocks.listGuidelineVersions).toHaveBeenCalledTimes(2);
    expect(mocks.targetProps).toHaveBeenLastCalledWith(expect.objectContaining({ activationBlockedReason: undefined }));
  });

  it("keeps a document unprepared when rebuilding does not activate a structure", async () => {
    mocks.listDocumentSummaries.mockResolvedValue([document(7, null)]);
    mocks.rebuildDocumentStructure.mockResolvedValue({
      ...preparedStructure,
      active_structure_version_id: null,
      structure_version: { ...preparedStructure.structure_version, status: "failed" },
    } satisfies DocumentStructureRead);
    render(<EvidenceSetupPanel {...defaults} />);
    fireEvent.click(await screen.findByRole("button", { name: "Prepare document 700" }));

    expect((await screen.findByRole("alert")).textContent).toMatch(/structure|ready/);
    expect(screen.queryByText("Paper 7 is ready for evidence annotation.")).toBeNull();
    expect(screen.getByText("Paper 7 · structure unprepared")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Prepare document 700" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
