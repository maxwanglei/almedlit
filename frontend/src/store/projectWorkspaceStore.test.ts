import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  getProject,
  getProjectProgress,
  listDocuments,
  listMyWorkProjects,
  listProjects,
  listTaskAssignments,
} from "@/api/client";
import type { Document, Project, ProjectProgress, TaskAssignment } from "@/types/api";

import { useProjectWorkspaceStore } from "./projectWorkspaceStore";

vi.mock("@/api/client", () => ({
  getProject: vi.fn(),
  getProjectProgress: vi.fn(),
  listDocuments: vi.fn(),
  listMyWorkProjects: vi.fn(),
  listProjects: vi.fn(),
  listTaskAssignments: vi.fn(),
}));

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (error: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

function documentItem(id: number, projectId: number): Document {
  return {
    id,
    project_id: projectId,
    external_id: null,
    title: null,
    text: `Document ${id}`,
    source: null,
    metadata_: {},
    sentences: [],
    active_structure_version_id: null,
  };
}

function project(id: number, name: string): Project {
  return {
    id,
    name,
    description: null,
    annotation_schema: { labels: {} },
    annotation_validation_mode: "strict",
    tasks: [],
    settings: {},
    workspace_id: id,
  };
}

function assignment(id: number, projectId: number, documentId: number): TaskAssignment {
  return {
    id,
    project_id: projectId,
    task_id: id,
    document_id: documentId,
    assignee_user_id: id,
    annotator_id: "alice",
    status: "assigned",
    assigned_by_user_id: null,
    assigned_by: null,
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: null,
    guideline_version_id: null,
    assignment_scope_key: "document",
  };
}

function progress(projectId: number): ProjectProgress {
  return {
    project_id: projectId,
    total: 1,
    by_status: {},
    by_task: [],
    by_document: [],
    by_annotator: [],
  };
}

describe("project workspace store", () => {
  beforeEach(() => {
    vi.mocked(getProject).mockReset();
    vi.mocked(getProjectProgress).mockReset();
    vi.mocked(listDocuments).mockReset();
    vi.mocked(listMyWorkProjects).mockReset();
    vi.mocked(listProjects).mockReset();
    vi.mocked(listTaskAssignments).mockReset();
    useProjectWorkspaceStore.getState().reset();
  });

  it("ignores project data that resolves after the selected project changes", async () => {
    const projectADocuments = deferred<Document[]>();
    const projectAAssignments = deferred<TaskAssignment[]>();
    const projectAProgress = deferred<ProjectProgress>();
    const projectBDocuments = deferred<Document[]>();
    const projectBAssignments = deferred<TaskAssignment[]>();
    const projectBProgress = deferred<ProjectProgress>();

    vi.mocked(listDocuments).mockImplementation((projectId) =>
      projectId === 1 ? projectADocuments.promise : projectBDocuments.promise,
    );
    vi.mocked(listTaskAssignments).mockImplementation((projectId) =>
      projectId === 1 ? projectAAssignments.promise : projectBAssignments.promise,
    );
    vi.mocked(getProjectProgress).mockImplementation((projectId) =>
      projectId === 1 ? projectAProgress.promise : projectBProgress.promise,
    );

    useProjectWorkspaceStore.getState().setSelectedProjectId(1);
    const loadProjectA = useProjectWorkspaceStore.getState().loadProjectData(1, true);
    useProjectWorkspaceStore.getState().setSelectedProjectId(2);
    const loadProjectB = useProjectWorkspaceStore.getState().loadProjectData(2, true);

    projectBDocuments.resolve([documentItem(20, 2)]);
    projectBAssignments.resolve([assignment(200, 2, 20)]);
    projectBProgress.resolve(progress(2));

    await expect(loadProjectB).resolves.toBe(20);
    expect(useProjectWorkspaceStore.getState().documents).toEqual([documentItem(20, 2)]);
    expect(useProjectWorkspaceStore.getState().dataLoadedProjectId).toBe(2);

    projectADocuments.resolve([documentItem(10, 1)]);
    projectAAssignments.resolve([assignment(100, 1, 10)]);
    projectAProgress.resolve(progress(1));

    await expect(loadProjectA).resolves.toBe(20);
    expect(useProjectWorkspaceStore.getState().documents).toEqual([documentItem(20, 2)]);
    expect(useProjectWorkspaceStore.getState().assignments).toEqual([assignment(200, 2, 20)]);
    expect(useProjectWorkspaceStore.getState().projectProgress).toEqual(progress(2));
    expect(useProjectWorkspaceStore.getState().selectedDocumentId).toBe(20);
    expect(useProjectWorkspaceStore.getState().dataLoadedProjectId).toBe(2);
  });

  it("fully resets cached workspace data", () => {
    useProjectWorkspaceStore.setState({
      projects: [project(1, "Old project")],
      documents: [documentItem(10, 1)],
      assignments: [assignment(100, 1, 10)],
      projectProgress: progress(1),
      selectedProjectId: 1,
      selectedDocumentId: 10,
      projectsLoaded: true,
      dataLoadedProjectId: 1,
      dataLoadedScope: "mine",
      loading: true,
      busy: true,
      error: "old error",
    });

    useProjectWorkspaceStore.getState().reset();

    expect(useProjectWorkspaceStore.getState()).toMatchObject({
      projects: [],
      documents: [],
      assignments: [],
      projectProgress: null,
      selectedProjectId: null,
      selectedDocumentId: null,
      projectsLoaded: false,
      dataLoadedProjectId: null,
      dataLoadedScope: null,
      loading: false,
      busy: false,
      error: null,
    });
  });

  it("clears project-scoped records as soon as the project changes", () => {
    useProjectWorkspaceStore.setState({
      documents: [documentItem(10, 1)],
      assignments: [assignment(100, 1, 10)],
      projectProgress: progress(1),
      selectedProjectId: 1,
      selectedDocumentId: 10,
      dataLoadedProjectId: 1,
      dataLoadedScope: "mine",
      busy: true,
    });

    useProjectWorkspaceStore.getState().setSelectedProjectId(2);

    expect(useProjectWorkspaceStore.getState()).toMatchObject({
      documents: [],
      assignments: [],
      projectProgress: null,
      selectedProjectId: 2,
      selectedDocumentId: null,
      dataLoadedProjectId: null,
      dataLoadedScope: null,
      busy: false,
    });
  });

  it("does not restore an old project list after reset and a new session load", async () => {
    const oldProjects = deferred<Project[]>();
    const newProjects = deferred<Project[]>();
    vi.mocked(listProjects)
      .mockImplementationOnce(() => oldProjects.promise)
      .mockImplementationOnce(() => newProjects.promise);

    const oldLoad = useProjectWorkspaceStore.getState().loadProjects(undefined, true, 1);
    useProjectWorkspaceStore.getState().reset();
    const newLoad = useProjectWorkspaceStore.getState().loadProjects(undefined, true, 2);

    newProjects.resolve([project(2, "New user project")]);
    await expect(newLoad).resolves.toBe(2);

    oldProjects.resolve([project(1, "Old user project")]);
    await expect(oldLoad).resolves.toBe(2);
    expect(useProjectWorkspaceStore.getState().projects).toEqual([
      project(2, "New user project"),
    ]);
    expect(useProjectWorkspaceStore.getState().selectedProjectId).toBe(2);
  });

  it("loads assignment-scoped project summaries for an Annotator", async () => {
    vi.mocked(listMyWorkProjects).mockResolvedValue([
      project(3, "Assigned project"),
    ]);

    await expect(
      useProjectWorkspaceStore
        .getState()
        .loadProjects(undefined, true, 17, true),
    ).resolves.toBe(3);

    expect(listMyWorkProjects).toHaveBeenCalledWith(17);
    expect(listProjects).not.toHaveBeenCalled();
    expect(useProjectWorkspaceStore.getState().projects).toEqual([
      project(3, "Assigned project"),
    ]);
  });

  it("suppresses an invalidated project-list failure after reset", async () => {
    const oldProjects = deferred<Project[]>();
    vi.mocked(listProjects).mockImplementationOnce(() => oldProjects.promise);

    const oldLoad = useProjectWorkspaceStore.getState().loadProjects(undefined, true, 1);
    useProjectWorkspaceStore.getState().reset();
    oldProjects.reject(new Error("old user request failed"));

    await expect(oldLoad).resolves.toBeNull();
    expect(useProjectWorkspaceStore.getState()).toMatchObject({
      projects: [],
      selectedProjectId: null,
      loading: false,
      error: null,
    });
  });

  it("does not restore old scoped project data after reset", async () => {
    const oldDocuments = deferred<Document[]>();
    const oldAssignments = deferred<TaskAssignment[]>();
    const oldProgress = deferred<ProjectProgress>();
    const newDocuments = deferred<Document[]>();
    const newAssignments = deferred<TaskAssignment[]>();
    const newProgress = deferred<ProjectProgress>();
    vi.mocked(listDocuments)
      .mockImplementationOnce(() => oldDocuments.promise)
      .mockImplementationOnce(() => newDocuments.promise);
    vi.mocked(listTaskAssignments)
      .mockImplementationOnce(() => oldAssignments.promise)
      .mockImplementationOnce(() => newAssignments.promise);
    vi.mocked(getProjectProgress)
      .mockImplementationOnce(() => oldProgress.promise)
      .mockImplementationOnce(() => newProgress.promise);

    useProjectWorkspaceStore.getState().setSelectedProjectId(1);
    const oldLoad = useProjectWorkspaceStore.getState().loadProjectData(1, true, "mine");
    useProjectWorkspaceStore.getState().reset();
    useProjectWorkspaceStore.getState().setSelectedProjectId(1);
    const newLoad = useProjectWorkspaceStore.getState().loadProjectData(1, true, "mine");

    newDocuments.resolve([documentItem(20, 1)]);
    newAssignments.resolve([assignment(200, 1, 20)]);
    newProgress.resolve(progress(1));
    await expect(newLoad).resolves.toBe(20);

    oldDocuments.resolve([documentItem(10, 1)]);
    oldAssignments.resolve([assignment(100, 1, 10)]);
    oldProgress.resolve(progress(1));
    await expect(oldLoad).resolves.toBe(20);
    expect(useProjectWorkspaceStore.getState().documents).toEqual([documentItem(20, 1)]);
    expect(useProjectWorkspaceStore.getState().assignments).toEqual([
      assignment(200, 1, 20),
    ]);
    expect(useProjectWorkspaceStore.getState().selectedDocumentId).toBe(20);
  });
});
