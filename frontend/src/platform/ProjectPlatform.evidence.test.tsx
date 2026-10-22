// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createAccessSnapshot } from "@/navigation/AccessContext";
import type { Project, ProjectTask, TaskAssignment } from "@/types/api";

import ProjectPlatform from "./ProjectPlatform";
import { EMPTY_PLATFORM_PROJECT_DATA, type PlatformProjectData } from "./types";

const mocks = vi.hoisted(() => ({
  listProjectTasks: vi.fn(),
  listEvidenceTargets: vi.fn(),
  listDocumentSummaries: vi.fn(),
  listTaskAssignments: vi.fn(),
  listGuidelineVersions: vi.fn(),
  listWorkspaceMembers: vi.fn(),
}));
vi.mock("@/api/client", async (importOriginal) => ({
  ...await importOriginal<Record<string, unknown>>(),
  ...mocks,
}));

const project: Project = {
  id: 7, workspace_id: 5, name: "Evidence study", description: null,
  annotation_schema: { labels: {} }, annotation_validation_mode: "strict", tasks: [],
  settings: { modules: ["data", "annotate", "activity"] },
};
const data: PlatformProjectData = { ...EMPTY_PLATFORM_PROJECT_DATA, projectModules: {
  project_id: 7, selected: ["data", "annotate", "activity"],
  effective: ["data", "annotate", "activity"], workspace_capabilities: ["annotation"],
} };
const props = {
  pathname: "/projects/7/tasks", project, projects: [project], documents: [], assignments: [], progress: null, data,
  loading: false, busy: false, error: null,
  access: createAccessSnapshot({ workspaceId: 5, workspaceKind: "team", membershipRole: "manager", effectiveCapabilities: ["annotation"] }),
  currentUserId: 9, dialog: null,
  onDialogChange: vi.fn(), onProjectSelect: vi.fn(), onNavigate: vi.fn(), onOpenRound: vi.fn(),
  onUpdateProject: vi.fn(), onUpdateModules: vi.fn(), onRefresh: vi.fn().mockResolvedValue(undefined),
};

beforeEach(() => {
  vi.resetAllMocks();
  Object.values(mocks).forEach((mock) => mock.mockResolvedValue([]));
  props.onRefresh.mockResolvedValue(undefined);
});
afterEach(cleanup);

describe("live evidence setup", () => {
  it.each(["/projects/7/tasks", "/projects/7/overview"])("opens the selected legacy document task from %s", async (pathname) => {
    const task = { id: 41, project_id: 7, annotation_type: "entity", display_name: "Drug entities", description: null, enabled: true, sort_order: 0, labels: [], settings: {} } satisfies ProjectTask;
    render(<ProjectPlatform {...props} pathname={pathname} project={{ ...project, tasks: [task] }} documents={[{ id: 11, project_id: 7, external_id: null, title: "Paper", source: null, metadata_: {}, active_structure_version_id: null }]} />);
    const open = await screen.findAllByText("Open Drug entities");
    fireEvent.click(open[0]);
    expect(props.onNavigate).toHaveBeenCalledWith("/my-work?project=7&view=annotate&task=41&tool=legacy");
  });

  it("exposes evidence setup on the manager's Tasks route", async () => {
    render(<ProjectPlatform {...props} />);

    expect(await screen.findByRole("heading", { name: "Evidence setup" })).toBeTruthy();
    await waitFor(() => expect(mocks.listEvidenceTargets).toHaveBeenCalledWith(7));
    expect(mocks.listProjectTasks).toHaveBeenCalledWith(7);
    expect(mocks.listGuidelineVersions).toHaveBeenCalledWith(7);
    expect(mocks.listTaskAssignments).toHaveBeenCalledWith(7, { scope: "all" });
    expect(mocks.listWorkspaceMembers).toHaveBeenCalledWith(5);
  });

  it("offers Retry when setup resources fail to load", async () => {
    mocks.listEvidenceTargets.mockRejectedValueOnce(new Error("Targets are temporarily unavailable"));
    render(<ProjectPlatform {...props} />);

    expect(await screen.findByText("Targets are temporarily unavailable")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Retry evidence setup" }));

    await waitFor(() => expect(mocks.listEvidenceTargets).toHaveBeenCalledTimes(2));
    expect(await screen.findByRole("heading", { name: "Evidence targets" })).toBeTruthy();
    expect(screen.queryByText("Targets are temporarily unavailable")).toBeNull();
  });

  it.each(["trainer", "annotator"] as const)("does not load manager setup resources for a %s", (role) => {
    render(<ProjectPlatform {...props} access={createAccessSnapshot({
      workspaceId: 5, workspaceKind: "team", membershipRole: role, effectiveCapabilities: ["annotation"],
    })} />);

    expect(screen.queryByRole("heading", { name: "Evidence setup" })).toBeNull();
    expect(mocks.listEvidenceTargets).not.toHaveBeenCalled();
    expect(props.onNavigate).toHaveBeenCalledWith("/projects/7/overview", "replace");
  });

  it("links an empty evidence review back to the Tasks setup", async () => {
    render(<ProjectPlatform {...props} pathname="/projects/7/quality" />);

    fireEvent.click(await screen.findByRole("button", { name: "Set up evidence annotation" }));

    expect(props.onNavigate).toHaveBeenCalledWith("/projects/7/tasks#evidence-setup");
  });

  it("makes newly pinned evidence assignments available for adjudication", async () => {
    const assignment: TaskAssignment = {
      id: 41, project_id: 7, task_id: 11, document_id: 21, assignee_user_id: 9,
      annotator_id: "reviewer", status: "submitted", assigned_by_user_id: 9,
      assigned_by: "manager", notes: null, metadata_: {}, target_version_id: 101,
      structure_version_id: 201, guideline_version_id: 301, assignment_scope_key: "101:201:301",
    };
    const { rerender } = render(<ProjectPlatform {...props} pathname="/projects/7/quality" />);
    expect(screen.getByRole("heading", { name: "No pinned evidence scope" })).toBeTruthy();

    rerender(<ProjectPlatform {...props} pathname="/projects/7/quality" assignments={[assignment]} />);

    const scope = await screen.findByRole<HTMLSelectElement>("combobox", { name: "Review scope" });
    await waitFor(() => expect(scope.value).toBe("21:101:201:301"));
    expect(scope.options[0].textContent).toContain("Document 21");
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Load comparison" }).disabled).toBe(false);
    expect(screen.queryByRole("heading", { name: "No pinned evidence scope" })).toBeNull();
  });
});
