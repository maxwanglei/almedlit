// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { useState } from "react";
import { BrowserRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AnnotatorWorkspace from "@/pages/AnnotatorWorkspace";
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
}

function Harness({
  personal,
  initialAssignments = [assignment(10, "assigned")],
  finishFirstOnRefresh = false,
}: HarnessProps): React.ReactElement {
  const [assignments, setAssignments] = useState(initialAssignments);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(1);
  const [currentWorkbench, setWorkbench] = useState<AnnotationWorkbench | null>(
    workbench,
  );
  const [busy, setBusy] = useState(false);

  return (
    <BrowserRouter>
      <AnnotatorWorkspace
        projects={[project]}
        selectedProject={project}
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
