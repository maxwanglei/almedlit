// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Component, Profiler, type ReactNode, useState } from "react";
import { BrowserRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AnnotatorWorkspace, { type AnnotationMutationAdapter } from "@/pages/AnnotatorWorkspace";
import type {
  Annotation,
  AnnotationWorkbench,
  AnnotationWorkbenchTask,
  Document,
  Project,
  TaskAssignment,
} from "@/types/api";

const mocks = vi.hoisted(() => ({
  createAnnotation: vi.fn(),
  createSubmission: vi.fn(),
  deleteAnnotation: vi.fn(),
  reopenPersonalTaskAssignment: vi.fn(),
  updateAnnotation: vi.fn(),
  setDocumentLabel: vi.fn(),
  renderError: vi.fn(),
}));

vi.mock("@/api/client", () => mocks);
vi.mock("@/features/evidence-block/EvidenceBlockCanvas", () => ({
  default: () => <div>Evidence canvas</div>,
}));

const entityTask: AnnotationWorkbenchTask = {
  id: 1,
  project_id: 1,
  annotation_type: "entity",
  display_name: "Entity labels",
  description: null,
  enabled: true,
  sort_order: 0,
  labels: [{ name: "Disease", color: "#4d6e5b", description: null }],
  settings: {},
  annotation_type_spec: {
    name: "entity",
    requires_span: true,
    requires_head_tail: false,
    description: "Entity spans",
    selection_mode: "character_span",
    renderer_key: "legacy",
    relation_endpoint_allowed: true,
    handler_key: "generic",
  },
};
const relationTask: AnnotationWorkbenchTask = {
  ...entityTask,
  id: 2,
  annotation_type: "relation",
  display_name: "Relations",
  labels: [{ name: "Associated", color: "#4d6e5b", description: null }],
  annotation_type_spec: {
    ...entityTask.annotation_type_spec,
    name: "relation",
    requires_span: false,
    requires_head_tail: true,
    selection_mode: "head_tail",
  },
};
const tasks = [entityTask, relationTask];
const project: Project = {
  id: 1,
  name: "Relation navigation",
  description: null,
  annotation_schema: { labels: { entity: entityTask.labels, relation: relationTask.labels } },
  annotation_validation_mode: "strict",
  tasks,
  settings: {},
  workspace_id: 1,
};
const documents: Document[] = ["Alpha Beta", "Gamma Delta"].map((text, index) => ({
  id: index + 1,
  project_id: 1,
  external_id: `PMID-${index + 1}`,
  title: `Paper ${index + 1}`,
  text,
  source: "test",
  metadata_: {},
  sentences: [[0, text.length]],
  active_structure_version_id: 1,
}));

function taskAssignment(id: number, documentId: number, taskId: number, guidelineId = 1): TaskAssignment {
  return {
    id,
    project_id: 1,
    task_id: taskId,
    document_id: documentId,
    assignee_user_id: 1,
    annotator_id: "max",
    status: "assigned",
    assigned_by_user_id: 1,
    assigned_by: "max",
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: 1,
    guideline_version_id: guidelineId,
    assignment_scope_key: `relations-${id}`,
  };
}
const assignments = [
  taskAssignment(11, 1, entityTask.id),
  taskAssignment(12, 1, relationTask.id),
  taskAssignment(13, 1, relationTask.id, 2),
  taskAssignment(14, 1, relationTask.id),
  taskAssignment(21, 2, entityTask.id),
  taskAssignment(22, 2, relationTask.id),
];

function span(id: number, documentId: number, start: number, end: number): Annotation {
  return {
    id,
    project_id: 1,
    document_id: documentId,
    annotation_type: "entity",
    label: "Disease",
    start_offset: start,
    end_offset: end,
    text_span: documents[documentId - 1].text.slice(start, end),
    source: "human",
    status: "draft",
    confidence: null,
    annotator_user_id: 1,
    annotator_id: "max",
    model_checkpoint_id: null,
    guideline_version_id: 1,
    structure_version_id: 1,
    head_annotation_id: null,
    tail_annotation_id: null,
    evidence: {},
    attributes: {},
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };
}
const initialAnnotations = [span(101, 1, 0, 5), span(102, 1, 6, 10), span(201, 2, 0, 5), span(202, 2, 6, 11)];
const annotationApi: AnnotationMutationAdapter = mocks;

class WorkspaceBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error): { error: Error } {
    return { error };
  }

  componentDidCatch(error: Error): void {
    mocks.renderError(error.message);
  }

  render(): ReactNode {
    return this.state.error ? <div role="alert">{this.state.error.message}</div> : this.props.children;
  }
}

interface ScopeCommit {
  assignmentId: string;
  pickerVisible: boolean;
}

function Harness({ initialDocumentId = 1, onCommit = () => undefined, busy = false, constrained = false, allowReverse = false, overlapping = false }: {
  overlapping?: boolean;
  allowReverse?: boolean;
  busy?: boolean;
  constrained?: boolean;
  initialDocumentId?: number;
  onCommit?: (commit: ScopeCommit) => void;
}): React.ReactElement {
  const [documentId, setDocumentId] = useState<number | null>(initialDocumentId);
  const [annotations, setAnnotations] = useState<Annotation[]>(constrained ? [
    ...initialAnnotations.map((item) => item.id === 102 ? { ...item, label: "Drug" } : item),
    { ...initialAnnotations[0], id: 301, annotation_type: "relation", label: "Associated", start_offset: null, end_offset: null, text_span: null, head_annotation_id: 101, tail_annotation_id: 102 },
  ] : overlapping ? [
    ...initialAnnotations, span(103, 1, 0, 10),
    { ...initialAnnotations[0], id: 301, annotation_type: "relation", label: "Associated", start_offset: null, end_offset: null, text_span: null, head_annotation_id: 103, tail_annotation_id: 102 },
  ] : initialAnnotations);
  const workspaceTasks = constrained ? [entityTask, { ...relationTask, settings: { relation_constraints: { Associated: { head: allowReverse ? ["Disease", "Drug"] : ["Disease"], tail: allowReverse ? ["Disease", "Drug"] : ["Drug"] } } } }] : tasks;
  const documentItem = documents.find((item) => item.id === documentId)!;
  const workbench: AnnotationWorkbench = {
    project,
    document: documentItem,
    active_guideline: null,
    guideline_versions_by_id: {},
    tasks: workspaceTasks,
    annotation_type_specs: workspaceTasks.map((task) => task.annotation_type_spec),
    annotations: annotations.filter((item) => item.document_id === documentId),
    assignments: assignments.filter((item) => item.document_id === documentId),
    correction_locked_annotation_ids: [],
  };
  return <BrowserRouter>
    <button type="button" onClick={() => setAnnotations((current) => current.filter((item) => item.id !== 101))}>Refresh without source</button>
    <Profiler id="workspace" onRender={() => {
      const selection = document.querySelector<HTMLSelectElement>('select[aria-label="Assignment to submit"]');
      onCommit({ assignmentId: selection?.value ?? "", pickerVisible: Boolean(document.querySelector(".aw-popover.relation")) });
    }}><WorkspaceBoundary>
      <AnnotatorWorkspace
        projects={[project]} selectedProject={project} selectedProjectId={1}
        setSelectedProjectId={vi.fn()} documents={documents} assignments={assignments}
        annotatorId="max" projectProgress={null} selectedDocumentId={documentId} setSelectedDocumentId={setDocumentId}
        workbench={workbench} setWorkbench={() => undefined} busy={busy} setBusy={vi.fn()} setError={vi.fn()}
        refreshProjectData={async () => undefined} refreshWorkbench={async () => undefined}
        annotationApi={annotationApi} defaultView="annotate"
      />
    </WorkspaceBoundary></Profiler>
  </BrowserRouter>;
}

function workspaceUrl(documentId: number, assignmentId: number): string {
  return `/annotator/workbench?project=1&document=${documentId}&assignment=${assignmentId}&view=annotate&tool=legacy`;
}

function startRelation(text = "Alpha"): void {
  fireEvent.contextMenu(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: `Disease: ${text}` }));
  fireEvent.click(screen.getByRole("menuitem", { name: "Start relation from here" }));
  expect(screen.getByText(/Linking from/)).toBeTruthy();
}

beforeEach(() => {
  window.history.replaceState(null, "", workspaceUrl(1, 12));
  vi.clearAllMocks();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    value: class {
      observe(): void {}
      disconnect(): void {}
    },
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("relation state across workspace scopes", () => {
  it("cancels linking safely when browser Back changes the document", async () => {
    window.history.pushState(null, "", workspaceUrl(2, 22));
    render(<Harness initialDocumentId={2} />);
    startRelation("Gamma");

    act(() => window.history.back());

    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("1"));
    await waitFor(() => expect(screen.queryByRole("heading", { name: "Paper 1" }) ?? screen.queryByRole("alert")).toBeTruthy());
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Linking from/)).toBeNull();
    expect(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: "Disease: Alpha" })).toBeTruthy();
  });

  it("cancels linking safely when the assignment changes its guideline scope", () => {
    render(<Harness />);
    startRelation();

    fireEvent.change(screen.getByRole("combobox", { name: "Assignment to submit" }), { target: { value: "13" } });

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Linking from/)).toBeNull();
    expect(screen.queryByRole("button", { name: "Disease: Alpha" })).toBeNull();
  });

  it("cancels linking safely when a workbench refresh removes the source span", () => {
    render(<Harness />);
    startRelation();

    fireEvent.click(screen.getByRole("button", { name: "Refresh without source" }));

    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.queryByText(/Linking from/)).toBeNull();
    expect(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: "Disease: Beta" })).toBeTruthy();
    expect(mocks.createAnnotation).not.toHaveBeenCalled();
    startRelation("Beta");
  });

  it("does not commit an old relation picker under a new assignment even when both endpoints remain visible", () => {
    const onCommit = vi.fn<(commit: ScopeCommit) => void>();
    render(<Harness onCommit={onCommit} />);
    startRelation();
    fireEvent.click(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: "Disease: Beta" }));
    expect(screen.getByRole("menu", { name: "Relation labels" })).toBeTruthy();

    fireEvent.change(screen.getByRole("combobox", { name: "Assignment to submit" }), { target: { value: "14" } });

    expect(onCommit).toHaveBeenCalledWith({ assignmentId: "14", pickerVisible: false });
    expect(onCommit).not.toHaveBeenCalledWith({ assignmentId: "14", pickerVisible: true });
    expect(screen.queryByRole("button", { name: "Close relation picker" })).toBeNull();
    expect(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: "Disease: Alpha" })).toBeTruthy();
    expect(within(screen.getByRole("tabpanel", { name: "Document" })).getByRole("button", { name: "Disease: Beta" })).toBeTruthy();
    expect(mocks.createAnnotation).not.toHaveBeenCalled();
  });
});

it("disables the primary resume action while document work is busy", () => {
  window.history.replaceState(null, "", "/annotator/workbench?project=1&document=1&view=progress");
  render(<Harness busy />);
  expect(screen.getByRole<HTMLButtonElement>("button", { name: "Start annotating" }).disabled).toBe(true);
});

it("keeps content-relative relation arcs stable without rendering on scroll", () => {
  const onCommit = vi.fn();
  const { container } = render(<Harness onCommit={onCommit} constrained />);
  const reader = container.querySelector<HTMLDivElement>(".aw-reading")!;
  const path = container.querySelector(".aw-arc-hit")!.getAttribute("d");
  onCommit.mockClear();
  fireEvent.scroll(reader, { target: { scrollTop: 200 } });
  expect(onCommit).not.toHaveBeenCalled();
  expect(container.querySelector(".aw-arc-hit")!.getAttribute("d")).toBe(path);
});

it("selects nested outer spans and keeps their relation arcs after another segment changes", () => {
  const { container } = render(<Harness overlapping />);
  expect(container.querySelectorAll(".aw-arc-hit")).toHaveLength(1);
  const outer = screen.getAllByRole("button", { name: "Select Disease annotation 103" })[0];
  if (!outer) throw new Error("The outer annotation must be selectable");
  fireEvent.click(outer);
  fireEvent.click(screen.getByRole("menuitem", { name: "Start relation from here" }));
  expect(screen.getByText(/Linking from/)).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Select Disease annotation 102" }));
  expect(screen.getByRole("menu", { name: "Relation labels" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Refresh without source" }));
  expect(container.querySelectorAll(".aw-arc-hit")).toHaveLength(1);
});

it("does not offer a direction swap that violates endpoint constraints", async () => {
  render(<Harness constrained />);
  fireEvent.click(await screen.findByRole("button", { name: "Relation Associated" }));
  const swap = screen.getByRole<HTMLButtonElement>("menuitem", { name: "Swap direction" });
  expect(swap.disabled).toBe(true);
  fireEvent.click(swap);
  expect(mocks.updateAnnotation).not.toHaveBeenCalled();
});

it("allows a direction swap when both endpoint labels satisfy the reversed rule", async () => {
  mocks.updateAnnotation.mockResolvedValue({ ...initialAnnotations[0], id: 301, annotation_type: "relation", label: "Associated", head_annotation_id: 102, tail_annotation_id: 101 });
  render(<Harness constrained allowReverse />);
  fireEvent.click(await screen.findByRole("button", { name: "Relation Associated" }));
  const swap = screen.getByRole<HTMLButtonElement>("menuitem", { name: "Swap direction" });
  expect(swap.disabled).toBe(false);
  fireEvent.click(swap);
  await waitFor(() => expect(mocks.updateAnnotation).toHaveBeenCalledWith(301, { head_annotation_id: 102, tail_annotation_id: 101 }));
});
