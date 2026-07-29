import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

import {
  type AssignmentScope,
  getProject,
  getProjectProgress,
  listDocuments,
  listMyWorkProjects,
  listProjects,
  listTaskAssignments,
} from "@/api/client";
import type { Document, Project, ProjectProgress, TaskAssignment } from "@/types/api";

interface ProjectWorkspaceState {
  projects: Project[];
  documents: Document[];
  assignments: TaskAssignment[];
  projectProgress: ProjectProgress | null;
  selectedProjectId: number | null;
  selectedDocumentId: number | null;
  projectsLoaded: boolean;
  dataLoadedProjectId: number | null;
  dataLoadedScope: AssignmentScope | null;
  loading: boolean;
  busy: boolean;
  error: string | null;
  setSelectedProjectId: (projectId: number | null) => void;
  setSelectedDocumentId: (documentId: number | null) => void;
  setBusy: (busy: boolean) => void;
  setError: (message: string | null) => void;
  reset: () => void;
  clearProjectData: () => void;
  replaceProject: (project: Project) => void;
  loadProjects: (
    preferredProjectId?: number | null,
    force?: boolean,
    workspaceId?: number | null,
    assignmentScoped?: boolean,
  ) => Promise<number | null>;
  loadProject: (projectId: number) => Promise<Project>;
  loadProjectData: (
    projectId: number,
    force?: boolean,
    scope?: AssignmentScope,
  ) => Promise<number | null>;
}

function chooseProjectId(
  projects: Project[],
  preferredProjectId?: number | null,
  currentProjectId?: number | null,
): number | null {
  if (
    preferredProjectId !== undefined &&
    preferredProjectId !== null &&
    projects.some((project) => project.id === preferredProjectId)
  ) {
    return preferredProjectId;
  }
  if (
    currentProjectId !== undefined &&
    currentProjectId !== null &&
    projects.some((project) => project.id === currentProjectId)
  ) {
    return currentProjectId;
  }
  return projects[0]?.id ?? null;
}

function initialProjectWorkspaceState() {
  return {
    projects: [] as Project[],
    documents: [] as Document[],
    assignments: [] as TaskAssignment[],
    projectProgress: null as ProjectProgress | null,
    selectedProjectId: null as number | null,
    selectedDocumentId: null as number | null,
    projectsLoaded: false,
    dataLoadedProjectId: null as number | null,
    dataLoadedScope: null as AssignmentScope | null,
    loading: false,
    busy: false,
    error: null as string | null,
  };
}

let projectListRequestId = 0;
let projectRequestId = 0;
let projectDataRequestId = 0;

export const useProjectWorkspaceStore = create<ProjectWorkspaceState>()(
  immer((set, get) => ({
    ...initialProjectWorkspaceState(),
    setSelectedProjectId: (projectId) => {
      if (get().selectedProjectId !== projectId) {
        projectDataRequestId += 1;
      }
      set((state) => {
        if (state.selectedProjectId !== projectId) {
          state.documents = [];
          state.assignments = [];
          state.projectProgress = null;
          state.selectedDocumentId = null;
          state.dataLoadedProjectId = null;
          state.dataLoadedScope = null;
          state.busy = false;
        }
        state.selectedProjectId = projectId;
      });
    },
    setSelectedDocumentId: (documentId) =>
      set((state) => {
        state.selectedDocumentId = documentId;
      }),
    setBusy: (busy) =>
      set((state) => {
        state.busy = busy;
      }),
    setError: (message) =>
      set((state) => {
        state.error = message;
      }),
    reset: () => {
      projectListRequestId += 1;
      projectRequestId += 1;
      projectDataRequestId += 1;
      set((state) => {
        Object.assign(state, initialProjectWorkspaceState());
      });
    },
    clearProjectData: () => {
      projectDataRequestId += 1;
      set((state) => {
        state.documents = [];
        state.assignments = [];
        state.projectProgress = null;
        state.selectedDocumentId = null;
        state.dataLoadedProjectId = null;
        state.dataLoadedScope = null;
        state.busy = false;
      });
    },
    replaceProject: (project) =>
      set((state) => {
        const index = state.projects.findIndex((candidate) => candidate.id === project.id);
        if (index >= 0) {
          state.projects[index] = project;
        } else {
          state.projects.unshift(project);
        }
      }),
    loadProjects: async (
      preferredProjectId,
      force = false,
      workspaceId = null,
      assignmentScoped = false,
    ) => {
      const current = get();
      if (current.projectsLoaded && !force) {
        const nextProjectId = chooseProjectId(
          current.projects,
          preferredProjectId,
          current.selectedProjectId,
        );
        set((state) => {
          state.selectedProjectId = nextProjectId;
        });
        return nextProjectId;
      }

      const requestId = ++projectListRequestId;
      set((state) => {
        state.loading = true;
        state.error = null;
      });
      try {
        const projects =
          assignmentScoped && workspaceId !== null
            ? await listMyWorkProjects(workspaceId)
            : await listProjects(workspaceId);
        if (requestId !== projectListRequestId) {
          return get().selectedProjectId;
        }
        const nextProjectId = chooseProjectId(
          projects,
          preferredProjectId,
          get().selectedProjectId,
        );
        set((state) => {
          state.projects = projects;
          state.selectedProjectId = nextProjectId;
          state.projectsLoaded = true;
        });
        return nextProjectId;
      } catch (caught) {
        if (requestId !== projectListRequestId) {
          return get().selectedProjectId;
        }
        set((state) => {
          state.error = caught instanceof Error ? caught.message : "Unable to load projects";
        });
        throw caught;
      } finally {
        if (requestId === projectListRequestId) {
          set((state) => {
            state.loading = false;
          });
        }
      }
    },
    loadProject: async (projectId) => {
      const requestId = ++projectRequestId;
      set((state) => {
        state.error = null;
      });
      try {
        const project = await getProject(projectId);
        if (requestId !== projectRequestId) {
          return project;
        }
        set((state) => {
          const index = state.projects.findIndex((candidate) => candidate.id === project.id);
          if (index >= 0) {
            state.projects[index] = project;
          } else {
            state.projects.unshift(project);
          }
          state.selectedProjectId = project.id;
        });
        return project;
      } catch (caught) {
        if (requestId === projectRequestId) {
          set((state) => {
            state.error = caught instanceof Error ? caught.message : "Unable to load project";
          });
        }
        throw caught;
      }
    },
    loadProjectData: async (projectId, force = false, scope = "all") => {
      const current = get();
      if (current.dataLoadedProjectId === projectId && current.dataLoadedScope === scope && !force) {
        return current.selectedDocumentId;
      }

      const requestId = ++projectDataRequestId;
      set((state) => {
        state.busy = true;
        state.error = null;
      });
      try {
        const [documents, assignments, progress] = await Promise.all([
          listDocuments(projectId, { scope }),
          listTaskAssignments(projectId, { scope }),
          getProjectProgress(projectId, scope),
        ]);

        const latest = get();
        if (requestId !== projectDataRequestId || latest.selectedProjectId !== projectId) {
          return latest.selectedDocumentId;
        }

        if (
          progress.project_id !== projectId ||
          documents.some((documentItem) => documentItem.project_id !== projectId) ||
          assignments.some((assignment) => assignment.project_id !== projectId)
        ) {
          throw new Error("Project data response did not match the requested project");
        }

        const currentDocumentId = latest.selectedDocumentId;
        const nextDocumentId =
          currentDocumentId !== null &&
          documents.some((documentItem) => documentItem.id === currentDocumentId)
            ? currentDocumentId
            : documents[0]?.id ?? null;

        set((state) => {
          state.documents = documents;
          state.assignments = assignments;
          state.projectProgress = progress;
          state.selectedDocumentId = nextDocumentId;
          state.dataLoadedProjectId = projectId;
          state.dataLoadedScope = scope;
        });
        return nextDocumentId;
      } catch (caught) {
        if (requestId !== projectDataRequestId || get().selectedProjectId !== projectId) {
          return get().selectedDocumentId;
        }
        set((state) => {
          state.error = caught instanceof Error ? caught.message : "Unable to load project data";
        });
        throw caught;
      } finally {
        if (requestId === projectDataRequestId) {
          set((state) => {
            state.busy = false;
          });
        }
      }
    },
  })),
);
