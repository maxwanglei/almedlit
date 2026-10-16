// @vitest-environment jsdom

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useEffect, useState } from "react";
import { BrowserRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AnnotatorWorkspace, { type AnnotationMutationAdapter } from "@/pages/AnnotatorWorkspace";
import type {
  AnnotationWorkbench,
  AnnotationWorkbenchTask,
  Document,
  Project,
  TaskAssignment,
  TaskAssignmentStatus,
} from "@/types/api";

const mocks = vi.hoisted(() => ({
  createAnnotation: vi.fn(),
  createSubmission: vi.fn(),
  deleteAnnotation: vi.fn(),
  reopenPersonalTaskAssignment: vi.fn(),
  updateAnnotation: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  createAnnotation: mocks.createAnnotation,
  createSubmission: mocks.createSubmission,
  deleteAnnotation: mocks.deleteAnnotation,
  reopenPersonalTaskAssignment: mocks.reopenPersonalTaskAssignment,
  updateAnnotation: mocks.updateAnnotation,
}));

vi.mock("@/features/evidence-block/EvidenceBlockCanvas", () => ({
  default: function EvidenceCanvas({
    assignments,
    selectedAssignmentId,
    onActiveAssignmentChange,
  }: {
    assignments: TaskAssignment[];
    selectedAssignmentId?: number | null;
    onActiveAssignmentChange: (assignment: TaskAssignment | null) => void;
  }): React.ReactElement {
    const [selectedId, setSelectedId] = useState(assignments[0]?.id ?? null);
    const activeId = selectedAssignmentId === undefined ? selectedId : selectedAssignmentId;
    const selected = assignments.find((item) => item.id === activeId) ?? null;
    useEffect(() => {
      if (selectedAssignmentId !== undefined) return;
      onActiveAssignmentChange(selected);
    }, [onActiveAssignmentChange, selected, selectedAssignmentId]);
    return (
      <label>
        Evidence canvas assignment
        <select
          value={activeId ?? ""}
          onChange={(event) => {
            const assignmentId = Number(event.target.value);
            setSelectedId(assignmentId);
            onActiveAssignmentChange(assignments.find((item) => item.id === assignmentId) ?? null);
          }}
        >
          {assignments.map((item) => (
            <option key={item.id} value={item.id}>Target {item.target_version_id}</option>
          ))}
        </select>
      </label>
    );
  },
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

const documentItem: Document = {
  id: 1,
  project_id: 1,
  external_id: "PMID-1",
  title: "Clinical trial abstract",
  text: "Disease text",
  source: "test",
  metadata_: {},
  sentences: [[0, 12]],
  active_structure_version_id: 1,
};

const project: Project = {
  id: 1,
  name: "Personal pilot",
  description: null,
  annotation_schema: { labels: { entity: entityTask.labels } },
  annotation_validation_mode: "strict",
  tasks: [entityTask],
  settings: {},
  workspace_id: 1,
};

const workbench: AnnotationWorkbench = {
  project,
  document: documentItem,
  active_guideline: null,
  guideline_versions_by_id: {},
  tasks: [entityTask],
  annotation_type_specs: [entityTask.annotation_type_spec],
  annotations: [],
  assignments: [],
  correction_locked_annotation_ids: [],
};

function assignment(id: number, status: TaskAssignmentStatus): TaskAssignment {
  return {
    id,
    project_id: 1,
    task_id: 1,
    document_id: 1,
    assignee_user_id: 1,
    annotator_id: "max",
    status,
    assigned_by_user_id: 1,
    assigned_by: "max",
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: 1,
    guideline_version_id: null,
    assignment_scope_key: `test-${id}`,
  };
}

interface HarnessProps {
  personal: boolean;
  initialAssignments?: TaskAssignment[];
  finishFirstOnRefresh?: boolean;
  initialWorkbench?: AnnotationWorkbench;
}

function Harness({
  personal,
  initialAssignments = [assignment(10, "assigned")],
  finishFirstOnRefresh = false,
  initialWorkbench = workbench,
}: HarnessProps): React.ReactElement {
  const [assignments, setAssignments] = useState(initialAssignments);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(1);
  const [currentWorkbench, setWorkbench] = useState<AnnotationWorkbench | null>(
    initialWorkbench,
  );
  const [busy, setBusy] = useState(false);

  return (
    <BrowserRouter>
      <AnnotatorWorkspace
        projects={[initialWorkbench.project]}
        selectedProject={initialWorkbench.project}
        selectedProjectId={1}
        setSelectedProjectId={vi.fn()}
        documents={[documentItem]}
        assignments={assignments}
        annotatorId="max"
        projectProgress={null}
        selectedDocumentId={selectedDocumentId}
        setSelectedDocumentId={setSelectedDocumentId}
        workbench={currentWorkbench}
        setWorkbench={setWorkbench}
        busy={busy}
        setBusy={setBusy}
        setError={vi.fn()}
        refreshProjectData={async () => {
          if (mocks.reopenPersonalTaskAssignment.mock.calls.length > 0) {
            setAssignments((current) =>
              current.map((item, index) =>
                index === 0 ? { ...item, status: "in_progress" } : item,
              ),
            );
          } else if (finishFirstOnRefresh) {
            setAssignments((current) =>
              current.map((item, index) =>
                index === 0 ? { ...item, status: "submitted" } : item,
              ),
            );
          }
        }}
        refreshWorkbench={async () => undefined}
        allowAssignmentlessSubmit={personal}
        onLogout={vi.fn()}
      />
    </BrowserRouter>
  );
}

const classificationTask: AnnotationWorkbenchTask = {
  ...entityTask,
  id: 2,
  annotation_type: "doc_label",
  display_name: "Paper relevance",
  labels: [{ name: "Clinical", color: "#4d6e5b", description: null }],
  annotation_type_spec: { ...entityTask.annotation_type_spec, name: "doc_label", requires_span: false, selection_mode: "document" },
};
const navigationDocuments = [documentItem, ...[2, 3].map((id) => ({
  ...documentItem, id, title: `Paper ${id}`, external_id: `PMID-${id}`, text: `Text for paper ${id}`,
}))];
const navigationTasks = [classificationTask, entityTask];
const navigationProject = { ...project, tasks: navigationTasks };
const navigationAssignments = navigationDocuments.flatMap((document) => navigationTasks.map((task, index) => ({
  ...assignment(document.id * 10 + index, document.id === 1 ? "submitted" : "assigned"),
  task_id: task.id,
  document_id: document.id,
})));
const navigationApi: AnnotationMutationAdapter = {
  ...mocks,
  setDocumentLabel: vi.fn(),
};

function NavigationHarness({ canonical = true, defaultView = "annotate" }: { canonical?: boolean; defaultView?: "progress" | "annotate" }): React.ReactElement {
  const [documentId, setDocumentId] = useState<number | null>(1);
  const document = navigationDocuments.find((candidate) => candidate.id === documentId)!;
  return <BrowserRouter><AnnotatorWorkspace
    projects={[navigationProject]} selectedProject={navigationProject} selectedProjectId={1}
    setSelectedProjectId={vi.fn()} documents={navigationDocuments} assignments={navigationAssignments}
    annotatorId="max" projectProgress={null} selectedDocumentId={documentId} setSelectedDocumentId={setDocumentId}
    workbench={{ ...workbench, project: navigationProject, document, tasks: navigationTasks,
      annotation_type_specs: navigationTasks.map((task) => task.annotation_type_spec),
      assignments: navigationAssignments.filter((item) => item.document_id === documentId) }}
    setWorkbench={() => undefined} busy={false} setBusy={vi.fn()} setError={vi.fn()}
    refreshProjectData={async () => undefined} refreshWorkbench={async () => undefined}
    allowAssignmentlessSubmit annotationApi={canonical ? navigationApi : undefined} defaultView={defaultView}
  /></BrowserRouter>;
}

beforeEach(() => {
  window.history.replaceState(null, "", "/annotator/workbench");
  vi.clearAllMocks();
  mocks.createSubmission.mockResolvedValue({});
  mocks.reopenPersonalTaskAssignment.mockResolvedValue({
    ...assignment(10, "in_progress"),
  });
  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    value: class {
      observe(): void {}
      disconnect(): void {}
    },
  });
});

afterEach(cleanup);

describe("AnnotatorWorkspace workspace language", () => {
  it("uses personal task language and removes literal pseudo-icons", () => {
    render(<Harness personal />);
    fireEvent.click(
      screen.getByRole("button", { name: /Clinical trial abstract.*Start/ }),
    );

    expect(new URLSearchParams(window.location.search).get("view")).toBe(
      "annotate",
    );
    expect(new URLSearchParams(window.location.search).get("document")).toBe(
      "1",
    );
    expect(
      (screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement)
        .value,
    ).toBe("10");
    expect(screen.getByRole("option", { name: "Entity labels · To do" })).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Submit task for this paper" }),
    ).toBeTruthy();
    expect(screen.queryByText("Edit")).toBeNull();
    expect(screen.queryByText("Chart")).toBeNull();
    expect(screen.queryByRole("button", { name: "Submit assignment" })).toBeNull();
  });

  it("preserves team assignment language", () => {
    render(<Harness personal={false} />);
    fireEvent.click(screen.getByRole("button", { name: /Annotate/ }));

    expect(
      (
        screen.getByRole("combobox", {
          name: "Assignment to submit",
        }) as HTMLSelectElement
      ).value,
    ).toBe("10");
    expect(
      screen.getByRole("option", { name: "Entity labels · assigned" }),
    ).toBeTruthy();
    expect(screen.getByRole("button", { name: "Submit assignment" })).toBeTruthy();
  });

  it("shows only observed team metrics and an honest empty activity state", () => {
    render(<Harness personal={false} />);

    const annotations = screen
      .getByText("Annotations on document")
      .closest("article");
    expect(annotations).not.toBeNull();
    expect(within(annotations as HTMLElement).getByText("0")).toBeTruthy();
    expect(
      within(annotations as HTMLElement).getByText("PMID-1"),
    ).toBeTruthy();

    const agreement = screen.getByText("Agreement").closest("article");
    expect(agreement).not.toBeNull();
    expect(
      within(agreement as HTMLElement).getByText("Not available"),
    ).toBeTruthy();
    expect(
      within(agreement as HTMLElement).getByText(
        "No agreement result loaded",
      ),
    ).toBeTruthy();
    expect(
      screen.getByText("No recent annotation activity is available."),
    ).toBeTruthy();
    expect(screen.queryByText("148")).toBeNull();
    expect(screen.queryByText("0.84")).toBeNull();
    expect(screen.queryByText(/prototype activity/i)).toBeNull();
  });

  it("shows the exact paper/task scope and does not submit on the first click", () => {
    render(<Harness personal />);
    fireEvent.click(
      screen.getByRole("button", { name: /Clinical trial abstract.*Start/ }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Submit task for this paper" }),
    );

    const dialog = screen.getByRole("dialog", {
      name: "Submit this paper task?",
    });
    expect(dialog).toBeTruthy();
    expect(within(dialog).getByText("Clinical trial abstract")).toBeTruthy();
    expect(within(dialog).getByText("Entity labels")).toBeTruthy();
    expect(
      screen.getByText(
        "This submits only the selected task for this paper. It does not complete that task for your other papers.",
      ),
    ).toBeTruthy();
    expect(screen.getByText("This paper task has no annotations.")).toBeTruthy();
    expect(
      (
        screen.getByRole("button", {
          name: "Submit with no annotations",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(mocks.createSubmission).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Keep editing" }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(mocks.createSubmission).not.toHaveBeenCalled();
  });

  it("holds a finished personal task in review until the user continues", async () => {
    render(
      <Harness
        personal
        initialAssignments={[assignment(10, "assigned"), assignment(11, "assigned")]}
        finishFirstOnRefresh
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /Clinical trial abstract.*Start/ }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Submit task for this paper" }),
    );
    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "I confirm that this paper has no annotations for this task.",
      }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Submit with no annotations" }),
    );

    await waitFor(() =>
      expect(mocks.createSubmission).toHaveBeenCalledWith(1, 1, {
        annotator_id: "max",
        assignment_id: 10,
      }),
    );
    expect(await screen.findByText("Paper task submitted")).toBeTruthy();
    expect(
      (screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement)
        .value,
    ).toBe("10");
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe(
      "10",
    );
    expect(screen.getByRole("button", { name: "Back to My Work" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Next task" }));

    await waitFor(() =>
      expect(
        (screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement)
          .value,
      ).toBe("11"),
    );
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe(
      "11",
    );
    expect(
      screen.getByRole("button", { name: "Submit task for this paper" }),
    ).toBeTruthy();
    expect(screen.queryByText("Paper task submitted")).toBeNull();
  });

  it("lets a personal user reopen a submitted paper task and edit again", async () => {
    render(
      <Harness
        personal
        initialAssignments={[assignment(10, "submitted")]}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /Clinical trial abstract.*Review/ }),
    );

    fireEvent.click(
      screen.getByRole("button", { name: "Edit this paper task" }),
    );
    expect(
      screen.getByRole("dialog", {
        name: "Reopen this paper task for editing?",
      }),
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Your earlier submission will remain saved. You can edit the annotations and submit a corrected version when ready.",
      ),
    ).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reopen and edit" }));

    await waitFor(() =>
      expect(mocks.reopenPersonalTaskAssignment).toHaveBeenCalledWith(1, 10),
    );
    expect(
      await screen.findByRole("button", {
        name: "Submit task for this paper",
      }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("button", { name: "Edit this paper task" }),
    ).toBeNull();
  });
});

describe("explicit assignment selection", () => {
  const evidenceTask: AnnotationWorkbenchTask = {
    ...entityTask,
    annotation_type: "evidence_block",
    display_name: "Evidence blocks",
    annotation_type_spec: {
      ...entityTask.annotation_type_spec,
      name: "evidence_block",
      selection_mode: "sentence_range",
      renderer_key: "evidence_block",
    },
  };
  const evidenceWorkbench: AnnotationWorkbench = {
    ...workbench,
    project: { ...project, tasks: [evidenceTask] },
    tasks: [evidenceTask],
    annotation_type_specs: [evidenceTask.annotation_type_spec],
  };

  it.each(["URL", "task dropdown"])(
    "opens and reopens a submitted task chosen through the %s while another task is open",
    async (source) => {
      window.history.replaceState(
        null,
        "",
        `/my-work/rounds/1?project=1&document=1&assignment=${source === "URL" ? 10 : 11}&view=annotate`,
      );
      render(
        <Harness
          personal
          initialAssignments={[
            assignment(10, "submitted"),
            { ...assignment(11, "assigned"), task_id: classificationTask.id },
          ]}
          initialWorkbench={{
            ...workbench,
            project: navigationProject,
            tasks: navigationTasks,
            annotation_type_specs: navigationTasks.map((task) => task.annotation_type_spec),
          }}
        />,
      );
      if (source === "task dropdown") {
        fireEvent.change(screen.getByRole("combobox", { name: "Task" }), {
          target: { value: "10" },
        });
      }

      expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("10");
      expect(new URLSearchParams(window.location.search).get("assignment")).toBe("10");
      expect(screen.queryByRole("button", { name: "Submit task for this paper" })).toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "Edit this paper task" }));
      fireEvent.click(screen.getByRole("button", { name: "Reopen and edit" }));

      await waitFor(() => expect(mocks.reopenPersonalTaskAssignment).toHaveBeenCalledWith(1, 10));
      expect(await screen.findByRole("button", { name: "Submit task for this paper" })).toBeTruthy();
      expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("10");
      expect(new URLSearchParams(window.location.search).get("assignment")).toBe("10");
    },
  );

  it("submits the evidence canvas assignment chosen after submitting another target", async () => {
    window.history.replaceState(
      null,
      "",
      "/my-work/rounds/1?project=1&document=1&assignment=10&view=annotate&tool=evidence",
    );
    render(
      <Harness
        personal
        finishFirstOnRefresh
        initialAssignments={[
          { ...assignment(10, "assigned"), target_version_id: 100 },
          { ...assignment(11, "assigned"), target_version_id: 101 },
        ]}
        initialWorkbench={evidenceWorkbench}
      />,
    );

    const submitBlankTask = (): void => {
      fireEvent.click(screen.getByRole("button", { name: "Submit task for this paper" }));
      fireEvent.click(screen.getByRole("checkbox", {
        name: "I confirm that this paper has no annotations for this task.",
      }));
      fireEvent.click(screen.getByRole("button", { name: "Submit with no annotations" }));
    };
    submitBlankTask();
    await waitFor(() => expect(mocks.createSubmission).toHaveBeenNthCalledWith(1, 1, 1, {
      annotator_id: "max",
      assignment_id: 10,
    }));
    await screen.findByRole("button", { name: "Edit this paper task" });

    fireEvent.change(screen.getByRole("combobox", { name: "Evidence canvas assignment" }), {
      target: { value: "11" },
    });

    expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("11");
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("11");
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe("11");
    submitBlankTask();
    await waitFor(() => expect(mocks.createSubmission).toHaveBeenNthCalledWith(2, 1, 1, {
      annotator_id: "max",
      assignment_id: 11,
    }));
  });

  it("keeps the team submit control in sync with the evidence canvas assignment", async () => {
    window.history.replaceState(
      null,
      "",
      "/annotator/workbench?project=1&document=1&assignment=10&view=annotate&tool=evidence",
    );
    render(
      <Harness
        personal={false}
        initialAssignments={[
          { ...assignment(10, "submitted"), target_version_id: 100 },
          { ...assignment(11, "assigned"), target_version_id: 101 },
        ]}
        initialWorkbench={evidenceWorkbench}
      />,
    );

    const submissionChoice = screen.getByRole("combobox", { name: "Assignment to submit" });
    const canvasChoice = screen.getByRole("combobox", { name: "Evidence canvas assignment" });
    expect((submissionChoice as HTMLSelectElement).value).toBe("10");
    expect((canvasChoice as HTMLSelectElement).value).toBe("10");
    expect((screen.getByRole("button", { name: "Submit assignment" }) as HTMLButtonElement).disabled).toBe(true);

    fireEvent.change(submissionChoice, { target: { value: "11" } });
    expect((canvasChoice as HTMLSelectElement).value).toBe("11");
    fireEvent.click(screen.getByRole("button", { name: "Submit assignment" }));
    fireEvent.click(screen.getByRole("checkbox", {
      name: "I confirm that this paper has no annotations for this task.",
    }));
    fireEvent.click(screen.getByRole("button", { name: "Submit with no annotations" }));

    await waitFor(() => expect(mocks.createSubmission).toHaveBeenCalledWith(1, 1, {
      annotator_id: "max",
      assignment_id: 11,
    }));
  });
});

describe("annotation tool selection", () => {
  const evidenceTask: AnnotationWorkbenchTask = {
    ...entityTask,
    id: 3,
    annotation_type: "evidence_block",
    display_name: "Evidence blocks",
    labels: [],
    annotation_type_spec: {
      ...entityTask.annotation_type_spec,
      name: "evidence_block",
      requires_span: false,
      selection_mode: "sentence_range",
      renderer_key: "evidence_block",
    },
  };
  const relationTask: AnnotationWorkbenchTask = {
    ...entityTask,
    id: 2,
    annotation_type: "relation",
    display_name: "Relations",
    labels: [{ name: "Associated with", color: "#4d6e5b", description: null }],
    annotation_type_spec: {
      ...entityTask.annotation_type_spec,
      name: "relation",
      requires_span: false,
      requires_head_tail: true,
      selection_mode: "head_tail",
    },
  };
  const mixedTasks = [evidenceTask, entityTask, relationTask];
  const mixedAssignments = [
    { ...assignment(30, "assigned"), task_id: evidenceTask.id, target_version_id: 100 },
    assignment(10, "assigned"),
    { ...assignment(20, "assigned"), task_id: relationTask.id },
  ];
  const mixedDocuments = navigationDocuments.slice(0, 2);
  const documentAssignments = [
    assignment(10, "assigned"),
    { ...assignment(30, "assigned"), document_id: 2, task_id: evidenceTask.id, target_version_id: 100 },
  ];
  const mixedProject = { ...project, tasks: mixedTasks };

  function DocumentToolHarness({ initialDocumentId }: { initialDocumentId: number }): React.ReactElement {
    const [documentId, setDocumentId] = useState<number | null>(initialDocumentId);
    const document = mixedDocuments.find((item) => item.id === documentId)!;
    return <BrowserRouter><AnnotatorWorkspace
      projects={[mixedProject]} selectedProject={mixedProject} selectedProjectId={1}
      setSelectedProjectId={vi.fn()} documents={mixedDocuments} assignments={documentAssignments}
      annotatorId="max" projectProgress={null} selectedDocumentId={documentId} setSelectedDocumentId={setDocumentId}
      workbench={{ ...workbench, project: mixedProject, document, tasks: mixedTasks,
        annotation_type_specs: mixedTasks.map((task) => task.annotation_type_spec) }}
      setWorkbench={() => undefined} busy={false} setBusy={vi.fn()} setError={vi.fn()}
      refreshProjectData={async () => undefined} refreshWorkbench={async () => undefined}
      onLogout={vi.fn()}
    /></BrowserRouter>;
  }

  function renderMixedWorkspace(
    query: string,
    personal = false,
    initialAssignments: TaskAssignment[] = mixedAssignments,
  ): void {
    window.history.replaceState(null, "", `/annotator/workbench?project=1&document=1&view=annotate&${query}`);
    render(<Harness personal={personal} initialAssignments={initialAssignments} initialWorkbench={{
      ...workbench,
      project: { ...project, tasks: mixedTasks },
      tasks: mixedTasks,
      annotation_type_specs: mixedTasks.map((task) => task.annotation_type_spec),
    }} />);
  }

  afterEach(() => vi.restoreAllMocks());

  it("keeps an explicit legacy tool URL when an evidence assignment is selected", () => {
    renderMixedWorkspace("assignment=30&tool=legacy");

    expect(new URLSearchParams(window.location.search).get("tool")).toBe("legacy");
    expect(screen.getByRole("button", { name: /^Spans & relations/ }).getAttribute("aria-pressed")).toBe("true");
    expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe("30");
    expect((screen.getByRole("button", { name: "Submit assignment" }) as HTMLButtonElement).disabled).toBe(true);
  });

  it.each([
    { assignmentId: 30, tool: "evidence" },
    { assignmentId: 10, tool: "legacy" },
  ])("infers $tool for requested assignment $assignmentId when the URL has no explicit tool", ({ assignmentId, tool }) => {
    renderMixedWorkspace(`assignment=${assignmentId}`);

    expect(new URLSearchParams(window.location.search).get("tool")).toBe(tool);
    expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe(String(assignmentId));
    if (tool === "evidence") {
      expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
    } else {
      expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    }
  });

  it("selects the matching tool when the assignment dropdown changes between evidence, entities, and relations", () => {
    renderMixedWorkspace("assignment=30&tool=evidence");

    const assignments = screen.getByRole("combobox", { name: "Assignment to submit" });
    for (const assignmentId of [10, 20]) {
      fireEvent.change(assignments, { target: { value: String(assignmentId) } });
      expect(new URLSearchParams(window.location.search).get("tool")).toBe("legacy");
      expect((assignments as HTMLSelectElement).value).toBe(String(assignmentId));
      expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    }
    expect((screen.getByRole("button", { name: /^Associated with/ }) as HTMLButtonElement).disabled).toBe(false);

    fireEvent.change(assignments, { target: { value: "30" } });
    expect(new URLSearchParams(window.location.search).get("tool")).toBe("evidence");
    expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
  });

  it.each([
    { finishedId: 30, nextId: 10, currentTool: "evidence", nextTool: "legacy" },
    { finishedId: 10, nextId: 30, currentTool: "legacy", nextTool: "evidence" },
  ])("opens the matching $nextTool tool for Next task after assignment $finishedId", ({ finishedId, nextId, currentTool, nextTool }) => {
    renderMixedWorkspace(
      `assignment=${finishedId}&tool=${currentTool}`,
      true,
      mixedAssignments.map((item) => item.id === finishedId ? { ...item, status: "submitted" } : item),
    );

    fireEvent.click(screen.getByRole("button", { name: "Next task" }));

    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe(String(nextId));
    expect(new URLSearchParams(window.location.search).get("tool")).toBe(nextTool);
    expect((screen.getByRole("button", { name: "Submit task for this paper" }) as HTMLButtonElement).disabled).toBe(false);
    if (nextTool === "evidence") {
      expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
    } else {
      expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    }
  });

  it.each([
    { documentId: 1, assignmentId: 10, tool: "legacy", direction: "Next", nextDocumentId: 2, nextAssignmentId: 30, nextTool: "evidence" },
    { documentId: 2, assignmentId: 30, tool: "evidence", direction: "Prev", nextDocumentId: 1, nextAssignmentId: 10, nextTool: "legacy" },
  ])("opens the matching $nextTool tool after moving to document $nextDocumentId", ({ documentId, assignmentId, tool, direction, nextDocumentId, nextAssignmentId, nextTool }) => {
    window.history.replaceState(
      null,
      "",
      `/annotator/workbench?project=1&document=${documentId}&assignment=${assignmentId}&view=annotate&tool=${tool}`,
    );
    render(<DocumentToolHarness initialDocumentId={documentId} />);

    fireEvent.click(screen.getByRole("button", { name: direction }));

    expect(new URLSearchParams(window.location.search).get("document")).toBe(String(nextDocumentId));
    expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe(String(nextAssignmentId));
    expect(new URLSearchParams(window.location.search).get("tool")).toBe(nextTool);
    expect((screen.getByRole("button", { name: "Submit assignment" }) as HTMLButtonElement).disabled).toBe(false);
    if (nextTool === "evidence") {
      expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
    } else {
      expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    }
  });

  it("switches from evidence to spans and relations and back when evidence is the first assignment", () => {
    renderMixedWorkspace("assignment=30&tool=evidence");

    fireEvent.click(screen.getByRole("button", { name: /^Spans & relations/ }));
    expect(new URLSearchParams(window.location.search).get("tool")).toBe("legacy");
    expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe("10");
    expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
    expect(screen.getByRole("button", { name: /^Disease/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /^Associated with/ })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^Evidence blocks/ }));
    expect(new URLSearchParams(window.location.search).get("tool")).toBe("evidence");
    expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe("30");
    expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
  });

  it("preserves the evidence assignment and tool while switching mobile panes", () => {
    vi.spyOn(window, "matchMedia").mockImplementation((query) => ({
      matches: true,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: () => false,
    }));
    renderMixedWorkspace("assignment=30&tool=evidence&pane=document");

    expect(screen.getByRole("combobox", { name: "Evidence canvas assignment" })).toBeTruthy();
    for (const pane of ["Queue", "Tools", "Review", "Document"]) {
      fireEvent.click(screen.getByRole("tab", { name: pane }));
      expect(new URLSearchParams(window.location.search).get("pane")).toBe(pane.toLowerCase());
      expect(new URLSearchParams(window.location.search).get("assignment")).toBe("30");
      expect(new URLSearchParams(window.location.search).get("tool")).toBe("evidence");
      expect((screen.getByRole("combobox", { name: "Assignment to submit" }) as HTMLSelectElement).value).toBe("30");
      if (pane !== "Document") {
        expect(screen.queryByRole("combobox", { name: "Evidence canvas assignment" })).toBeNull();
      }
    }
    expect((screen.getByRole("combobox", { name: "Evidence canvas assignment" }) as HTMLSelectElement).value).toBe("30");
  });
});

describe("paper navigation", () => {
  it("initializes a bare round URL before advancing, reloading, and going Back", async () => {
    window.history.replaceState(null, "", "/my-work/rounds/1");
    const mounted = render(<NavigationHarness />);
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("1"));
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe("10");
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("10");
    fireEvent.click(screen.getByRole("button", { name: "Next document" }));
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("2"));
    mounted.unmount();
    render(<NavigationHarness />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 2 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("20");
    act(() => window.history.back());
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("1"));
    await waitFor(() => expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("10"));
  });

  it("initializes an empty-query My Work page and continues to the next unfinished paper", async () => {
    window.history.replaceState(null, "", "/my-work");
    render(<NavigationHarness defaultView="progress" />);
    expect(screen.getByRole("heading", { name: "My Work" })).toBeTruthy();
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("1"));
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe("10");
    fireEvent.click(screen.getByRole("button", { name: "Continue annotating" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 2 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect(new URLSearchParams(window.location.search).get("document")).toBe("2");
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe("20");
  });

  it.each([true, false])("moves from a submitted paper through Next document and queue, restoring browser history (canonical=%s)", async (canonical) => {
    window.history.replaceState(null, "", "/my-work/rounds/1?project=1&document=1&assignment=10&view=annotate");
    render(<NavigationHarness canonical={canonical} />);
    fireEvent.click(screen.getByRole("button", { name: "Next document" }));
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("2"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 2 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("20");
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe("20");

    fireEvent.click(screen.getByRole("button", { name: "Paper 3 · To do" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 3 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect(new URLSearchParams(window.location.search).get("document")).toBe("3");
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe("30");

    act(() => window.history.back());
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("document")).toBe("2"));
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 2 · To do" }).getAttribute("aria-current")).toBe("true"));
    act(() => window.history.forward());
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 3 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect(mocks.createSubmission).not.toHaveBeenCalled();
  });

  it.each([
    { requested: 10, resolved: 20 },
    { requested: 21, resolved: 21 },
  ])("opens a deep-linked paper with assignment $requested resolved to $resolved", async ({ requested, resolved }) => {
    window.history.replaceState(null, "", `/my-work/rounds/1?project=1&document=2&assignment=${requested}&view=annotate`);
    render(<NavigationHarness />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Paper 2 · To do" }).getAttribute("aria-current")).toBe("true"));
    expect(new URLSearchParams(window.location.search).get("document")).toBe("2");
    expect((screen.getByRole("combobox", { name: "Task" }) as HTMLSelectElement).value).toBe(String(resolved));
    expect(new URLSearchParams(window.location.search).get("assignment")).toBe(String(resolved));
  });
});
