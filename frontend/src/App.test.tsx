// @vitest-environment jsdom

import type { ReactNode } from "react";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const mocks = vi.hoisted(() => {
  const projectState = {
    projects: [],
    documents: [],
    assignments: [],
    projectProgress: null,
    selectedProjectId: null,
    selectedDocumentId: null,
    loading: false,
    busy: false,
    error: null,
    setSelectedProjectId: vi.fn(),
    setSelectedDocumentId: vi.fn(),
    setBusy: vi.fn(),
    setError: vi.fn(),
    clearProjectData: vi.fn(),
    replaceProject: vi.fn(),
    loadProjects: vi.fn().mockResolvedValue(null),
    loadProjectData: vi.fn().mockResolvedValue(null),
    reset: vi.fn(),
  };
  const projectStore = Object.assign(vi.fn(() => projectState), {
    getState: vi.fn(() => projectState),
  });
  const evidenceState = { reset: vi.fn() };
  const evidenceStore = Object.assign(vi.fn(), {
    getState: vi.fn(() => evidenceState),
  });
  return {
    evidenceState,
    evidenceStore,
    getAnnotationWorkbench: vi.fn(),
    getRoundWorkContext: vi.fn(),
    listWorkspaceRoundWorkContexts: vi.fn(),
    getMe: vi.fn(),
    getWorkspaceCapabilities: vi.fn(),
    logout: vi.fn(),
    updateProject: vi.fn(),
    usePlatformProject: vi.fn(),
    projectState,
    projectStore,
  };
});

vi.mock("@/api/client", () => ({
  getAnnotationWorkbench: mocks.getAnnotationWorkbench,
  getMe: mocks.getMe,
  getWorkspaceCapabilities: mocks.getWorkspaceCapabilities,
  logout: mocks.logout,
  updateProject: mocks.updateProject,
}));

vi.mock("@/platform/api", async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    getRoundWorkContext: mocks.getRoundWorkContext,
    listWorkspaceRoundWorkContexts: mocks.listWorkspaceRoundWorkContexts,
  };
});

vi.mock("@/store/projectWorkspaceStore", () => ({
  useProjectWorkspaceStore: mocks.projectStore,
}));

vi.mock("@/features/evidence-block/evidenceBlockStore", () => ({
  useEvidenceBlockStore: mocks.evidenceStore,
}));

vi.mock("@/pages/AnnotatorWorkspace", () => ({
  default: ({
    moduleSwitcher,
    workspaceSwitcher,
    roundContexts,
  }: {
    moduleSwitcher?: ReactNode;
    workspaceSwitcher?: ReactNode;
    roundContexts?: Array<{ round: { id: number } }>;
  }) => (
    <section
      aria-label="Mock annotator workspace"
      data-round-contexts={roundContexts?.map((item) => item.round.id).join(",")}
    >
      {moduleSwitcher}
      {workspaceSwitcher}
    </section>
  ),
}));
vi.mock("@/pages/ManagerConsole", () => ({
  default: ({
    moduleSwitcher,
    workspaceSwitcher,
  }: {
    moduleSwitcher?: ReactNode;
    workspaceSwitcher?: ReactNode;
  }) => (
    <section aria-label="Mock manager workspace">
      {moduleSwitcher}
      {workspaceSwitcher}
    </section>
  ),
}));
vi.mock("@/pages/LoginPage", () => ({
  default: ({
    onAuthed,
  }: {
    onAuthed: (justRegistered: boolean) => void;
  }) => (
    <button type="button" onClick={() => onAuthed(true)}>
      Register mock user
    </button>
  ),
}));
vi.mock("@/pages/AcceptInvitePage", () => ({
  default: ({ token, signedIn }: { token: string; signedIn: boolean }) => (
    <main
      aria-label="Mock invite acceptance"
      data-token={token}
      data-signed-in={String(signedIn)}
    />
  ),
}));
vi.mock("@/pages/AccountActionPage", () => ({
  default: ({ token }: { token: string }) => (
    <main aria-label="Mock account action" data-token={token} />
  ),
}));
vi.mock("@/pages/OnboardingPresetPicker", () => ({
  default: ({ onDone }: { onDone: () => Promise<void> }) => (
    <button type="button" onClick={() => void onDone()}>
      Complete onboarding
    </button>
  ),
}));
vi.mock("@/platform/ProjectPlatform", () => ({
  default: ({
    project,
    moduleSwitcher,
    onUpdateProject,
  }: {
    project: { id: number; name: string };
    moduleSwitcher?: ReactNode;
    onUpdateProject: (payload: { name: string }) => Promise<void>;
  }) => (
    <section
      aria-label="Mock project platform"
      data-project-id={project.id}
    >
      {project.name}
      {moduleSwitcher}
      <button
        type="button"
        onClick={() => void onUpdateProject({ name: `${project.name} updated` })}
      >
        Update mock project
      </button>
    </section>
  ),
}));
vi.mock("@/platform/ProjectsWorkspace", () => ({
  default: () => (
    <section aria-label="Mock projects directory">
      <h1>Projects</h1>
    </section>
  ),
}));
vi.mock("@/platform/RoundWorkbench", () => ({
  default: ({
    round,
    task,
  }: {
    round: { id: number; name: string };
    task: { id: number; name?: string };
  }) => (
    <main
      id="main-content"
      aria-label="Mock round workbench"
      data-round-id={round.id}
      data-task-version-id={task.id}
    >
      <h1>{round.name}</h1>
    </main>
  ),
}));
vi.mock("@/platform/TrainingWorkspace", () => ({
  default: ({
    view,
    runId,
  }: {
    view: string;
    runId: number | null;
  }) => (
    <main
      id="main-content"
      aria-label="Mock training workspace"
      data-run-id={runId ?? undefined}
    >
      <h1>Training</h1>
      <span>{view}</span>
    </main>
  ),
}));
vi.mock("@/platform/ModelsWorkspace", () => ({
  default: ({
    view,
    modelId,
    versionId,
  }: {
    view: string;
    modelId: number | null;
    versionId: number | null;
  }) => (
    <main
      id="main-content"
      aria-label="Mock models workspace"
      data-model-id={modelId ?? undefined}
      data-version-id={versionId ?? undefined}
    >
      <h1>Models</h1>
      <span>{view}</span>
    </main>
  ),
}));
vi.mock("@/platform/WorkspaceSettings", () => ({
  default: () => (
    <main id="main-content" aria-label="Mock workspace settings">
      <h1>Workspace settings</h1>
    </main>
  ),
}));
vi.mock("@/platform/SystemAdministration", () => ({
  default: () => (
    <main id="main-content" aria-label="Mock system administration">
      <h1>System administration</h1>
    </main>
  ),
}));
vi.mock("@/platform/usePlatformProject", () => ({
  usePlatformProject: (
    projectId: number | null,
    ...rest: unknown[]
  ) => {
    mocks.usePlatformProject(projectId, ...rest);
    return {
    data: {
      projectModules: {
        project_id: projectId ?? 0,
        selected: [],
        effective: [],
        workspace_capabilities: [],
      },
      taskDefinitions: [],
      taskVersions: [],
      datasets: [],
      datasetVersions: [],
      labelSets: [],
      splitMaps: [],
      trainingDatasets: [],
      cycles: [],
      rounds: [],
      models: [],
      modelVersions: [],
      guidelines: [],
      guidelineRevisions: [],
      guidelineProposals: [],
      guidelineImpacts: [],
      feedbackEvents: [],
      reviewCases: [],
      trainingRuns: [],
      recipes: [],
      baseModels: [],
      projectRecipes: [],
      recipeVersions: [],
      environments: [],
      storagePolicies: [],
    },
    loading: false,
    busy: false,
    error: null,
    reload: vi.fn(),
    addDataset: vi.fn(),
    addCycle: vi.fn(),
    addRound: vi.fn(),
    addGuideline: vi.fn(),
    addTask: vi.fn(),
    prepareTrainingData: vi.fn(),
    launch: vi.fn(),
    enablePlatform: vi.fn(),
    setModules: vi.fn(),
    };
  },
}));

const me = {
  user: {
    id: 7,
    username: "alice",
    display_name: "Alice",
    is_active: true,
    is_superuser: false,
  },
  memberships: [
    {
      workspace_id: 10,
      workspace_name: "Personal",
      workspace_kind: "individual",
      role: "admin",
    },
    {
      workspace_id: 20,
      workspace_name: "Research team",
      workspace_kind: "team",
      role: "annotator",
    },
  ],
};

function installStorage(): void {
  const values = new Map<string, string>();
  const localStorage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
    key: (index: number) => Array.from(values.keys())[index] ?? null,
    get length() {
      return values.size;
    },
  } as Storage;
  Object.defineProperty(window, "localStorage", {
    configurable: true,
    value: localStorage,
  });
}

beforeEach(() => {
  installStorage();
  window.history.replaceState(null, "", "/");
  vi.clearAllMocks();
  mocks.getMe.mockResolvedValue(me);
  mocks.logout.mockResolvedValue(undefined);
  mocks.getWorkspaceCapabilities.mockImplementation(async (workspaceId: number) => ({
    workspace_id: workspaceId,
    preset: "annotation_only",
    overrides: [],
    effective: ["annotation"],
    blocked: {},
  }));
  mocks.getRoundWorkContext.mockResolvedValue({
    project: {
      id: 81,
      name: "Assigned project",
    },
    round: {
      id: 71,
      project_id: 81,
      name: "Assigned screening",
      sequence: 1,
      dataset_version_id: 22,
      task_version_id: 12,
      assistance_policy: "blind",
      feedback_available: false,
      status: "open",
      opened_at: null,
      closed_at: null,
      created_at: "2026-07-28T12:00:00Z",
      updated_at: "2026-07-28T12:00:00Z",
    },
    task: {
      id: 11,
      key: "screening",
      name: "Screening",
    },
    task_version: {
      id: 12,
      project_id: 81,
      task_definition_id: 11,
      version_number: 1,
      task_kind: "classification",
      input_schema: {},
      output_schema: { enum: ["Include", "Exclude"] },
      label_rules: {},
      annotation_ui: {},
      metrics: ["accuracy"],
      trainer_compatibility: [],
      content_hash: "task-hash",
    },
    cycle: null,
    guideline: null,
  });
  mocks.listWorkspaceRoundWorkContexts.mockResolvedValue([]);
  mocks.projectState.loadProjects.mockResolvedValue(null);
  mocks.projectState.loadProjectData.mockResolvedValue(null);
  mocks.updateProject.mockImplementation(
    async (projectId: number, payload: Record<string, unknown>) => ({
      id: projectId,
      name: "Updated project",
      description: null,
      annotation_schema: [],
      annotation_validation_mode: "relaxed",
      tasks: [],
      settings: {},
      workspace_id: 10,
      ...payload,
    }),
  );
  Object.assign(mocks.projectState, {
    projects: [],
    selectedProjectId: null,
  });
});

afterEach(() => {
  cleanup();
});

describe("App workspace selection", () => {
  it("finishes onboarding with the refreshed capability snapshot", async () => {
    mocks.getMe
      .mockRejectedValueOnce(new Error("No active session"))
      .mockResolvedValue({
      ...me,
      memberships: [
        {
          workspace_id: 20,
          workspace_name: "New training team",
          workspace_kind: "team",
          role: "trainer",
        },
      ],
      });
    mocks.getWorkspaceCapabilities
      .mockResolvedValueOnce({
        workspace_id: 20,
        preset: "annotate",
        overrides: [],
        effective: ["annotation"],
        blocked: {},
      })
      .mockResolvedValueOnce({
        workspace_id: 20,
        preset: "train",
        overrides: [],
        effective: ["training", "lineage"],
        blocked: {},
      });

    render(<App />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Register mock user" }),
    );
    fireEvent.click(
      await screen.findByRole("button", { name: "Complete onboarding" }),
    );

    await waitFor(() => expect(window.location.pathname).toBe("/projects"));
    expect(
      await screen.findByLabelText("Mock projects directory"),
    ).toBeTruthy();
    expect(screen.queryByRole("link", { name: "My Work" })).toBeNull();
    expect(screen.getByRole("link", { name: "Training" })).toBeTruthy();
    expect(mocks.getWorkspaceCapabilities).toHaveBeenCalledTimes(2);
  });

  it("switches memberships, persists the choice, and invalidates scoped stores", async () => {
    render(<App />);

    const selector = await screen.findByLabelText("Workspace");
    expect((selector as HTMLSelectElement).value).toBe("10");

    fireEvent.change(selector, { target: { value: "20" } });

    await waitFor(() =>
      expect((screen.getByLabelText("Workspace") as HTMLSelectElement).value).toBe("20"),
    );
    expect(window.localStorage.getItem("al_medlit_active_workspace:7")).toBe("20");
    expect(mocks.projectState.reset).toHaveBeenCalledTimes(1);
    expect(mocks.evidenceState.reset).toHaveBeenCalledTimes(1);
    expect(mocks.getWorkspaceCapabilities).toHaveBeenCalledWith(10);
    expect(mocks.getWorkspaceCapabilities).toHaveBeenCalledWith(20);
    expect(mocks.getMe).toHaveBeenCalledTimes(2);
  });

  it("does not expose a transient not-found route while workspace capabilities load", async () => {
    let resolveTeamCapabilities: (
      value: {
        workspace_id: number;
        preset: string;
        overrides: string[];
        effective: string[];
        blocked: Record<string, string>;
      },
    ) => void = () => undefined;
    const teamCapabilities = new Promise<{
      workspace_id: number;
      preset: string;
      overrides: string[];
      effective: string[];
      blocked: Record<string, string>;
    }>((resolve) => {
      resolveTeamCapabilities = resolve;
    });
    mocks.getWorkspaceCapabilities.mockImplementation(async (workspaceId: number) => {
      if (workspaceId === 20) {
        return teamCapabilities;
      }
      return {
        workspace_id: workspaceId,
        preset: "annotation_only",
        overrides: [],
        effective: ["annotation"],
        blocked: {},
      };
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("link", { name: "Projects" }));
    await waitFor(() => expect(window.location.pathname).toBe("/projects"));
    fireEvent.change(screen.getByLabelText("Workspace"), {
      target: { value: "20" },
    });

    expect(await screen.findByText("Loading workspace…")).toBeTruthy();
    expect(window.location.pathname).toBe("/projects");
    expect(window.location.pathname).not.toBe("/not-found");

    resolveTeamCapabilities({
      workspace_id: 20,
      preset: "annotation_only",
      overrides: [],
      effective: ["annotation"],
      blocked: {},
    });
    await waitFor(() => expect(window.location.pathname).toBe("/my-work"));
  });

  it("shows personal modules and opens the stable Projects directory", async () => {
    render(<App />);

    const myWorkLink = await screen.findByRole("link", { name: "My Work" });
    const projectsLink = screen.getByRole("link", {
      name: "Projects",
    });
    expect(myWorkLink.getAttribute("aria-current")).toBe("page");
    expect(projectsLink.getAttribute("href")).toBe("/projects");

    fireEvent.click(projectsLink);

    await waitFor(() => {
      expect(window.location.pathname).toBe("/projects");
      expect(screen.getByLabelText("Mock projects directory")).toBeTruthy();
    }, { timeout: 10_000 });
    expect(
      screen.getByRole("link", { name: "Projects" }).getAttribute(
        "aria-current",
      ),
    ).toBe("page");
  });

  it("redirects an unauthorized team route to the role default", async () => {
    window.localStorage.setItem("al_medlit_active_workspace:7", "20");
    window.history.replaceState(null, "", "/admin/users");

    render(<App />);

    await waitFor(() => {
      expect(window.location.pathname).toBe("/my-work");
      expect(mocks.projectState.loadProjects).toHaveBeenCalledWith(
        undefined,
        true,
        20,
        true,
      );
    }, { timeout: 5_000 });
    expect(screen.getAllByRole("link", { name: "My Work" })).toHaveLength(1);
  });

  it("offers only available modules from the 404 page", async () => {
    window.history.replaceState(null, "", "/missing-page");

    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Page not found" }),
    ).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open My Work" })).toBeTruthy();
    expect(
      screen.getByRole("link", { name: "Open Projects" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("link", { name: "Open Administration" }),
    ).toBeNull();
  });

  it("redirects the legacy project Models tab to the global registry", async () => {
    window.history.replaceState(
      null,
      "",
      "/projects/81/models?status=ready#versions",
    );
    Object.assign(mocks.projectState, {
      selectedProjectId: 81,
      projects: [{
        id: 81,
        name: "Model study",
        description: null,
        annotation_schema: [],
        annotation_validation_mode: "relaxed",
        tasks: [],
        settings: {},
        workspace_id: 30,
      }],
    });
    mocks.getMe.mockResolvedValue({
      ...me,
      memberships: [
        {
          workspace_id: 30,
          workspace_name: "ML team",
          workspace_kind: "team",
          role: "manager",
        },
      ],
    });
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 30,
      preset: "full",
      overrides: [],
      effective: ["annotation", "training"],
      blocked: {},
    });

    render(<App />);

    expect(
      await screen.findByRole("heading", { name: "Models" }),
    ).toBeTruthy();
    expect(window.location.pathname).toBe("/models");
    expect(window.location.search).toBe("?status=ready&projectId=81");
    expect(window.location.hash).toBe("#versions");
    expect(
      screen.getByRole("link", { name: "Models" }).getAttribute(
        "aria-current",
      ),
    ).toBe("page");
  });

  it("falls back from Models when the workspace lacks Training", async () => {
    window.history.replaceState(null, "", "/projects/82/models");
    Object.assign(mocks.projectState, {
      selectedProjectId: 82,
      projects: [{
        id: 82,
        name: "Review study",
        description: null,
        annotation_schema: [],
        annotation_validation_mode: "relaxed",
        tasks: [],
        settings: {},
        workspace_id: 31,
      }],
    });
    mocks.getMe.mockResolvedValue({
      ...me,
      memberships: [
        {
          workspace_id: 31,
          workspace_name: "Review team",
          workspace_kind: "team",
          role: "manager",
        },
      ],
    });

    render(<App />);

    await screen.findByLabelText("Mock annotator workspace");
    expect(window.location.pathname).toBe("/my-work");
    expect(screen.queryByRole("link", { name: "Models" })).toBeNull();
  });

  it("gives a team trainer all inherited functional workspaces", async () => {
    window.history.replaceState(null, "", "/training/new");
    mocks.getMe.mockResolvedValue({
      ...me,
      memberships: [
        {
          workspace_id: 40,
          workspace_name: "Training team",
          workspace_kind: "team",
          role: "trainer",
        },
      ],
    });
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 40,
      preset: "annotate_train",
      overrides: [],
      effective: ["annotation", "training"],
      blocked: {},
    });

    render(<App />);

    await screen.findByLabelText("Mock training workspace");
    expect(mocks.projectState.loadProjects).toHaveBeenCalledWith(
      undefined,
      true,
      40,
      false,
    );
    expect(
      ["My Work", "Projects", "Training", "Models"].map(
        (name) => screen.getByRole("link", { name }).getAttribute("href"),
      ),
    ).toEqual(["/my-work", "/projects", "/training", "/models"]);
    expect(screen.queryByRole("link", { name: "Workspace Settings" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Administration" })).toBeNull();
  });

  it("passes the canonical training run parameter through the nested route", async () => {
    window.history.replaceState(
      null,
      "",
      "/training/runs/73?projectId=81#metrics",
    );
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 10,
      preset: "annotate_train",
      overrides: [],
      effective: ["annotation", "training"],
      blocked: {},
    });

    render(<App />);

    const workspace = await screen.findByLabelText("Mock training workspace");
    expect(workspace.getAttribute("data-run-id")).toBe("73");
    expect(screen.getByText("run-detail")).toBeTruthy();
    expect(window.location.search).toBe("?projectId=81");
    expect(window.location.hash).toBe("#metrics");
  });

  it("updates route feedback and focus for query and hash history", async () => {
    window.history.replaceState(
      null,
      "",
      "/training?status=queued#runs",
    );
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 10,
      preset: "train",
      overrides: [],
      effective: ["training", "lineage"],
      blocked: {},
    });

    render(<App />);

    const heading = await screen.findByRole("heading", { name: "Training" });
    await waitFor(() => {
      expect(document.title).toBe("Training | AL-MedLit");
      expect(
        screen.getByText(
          "Training loaded. Status queued. Section runs.",
        ),
      ).toBeTruthy();
      expect(document.activeElement).toBe(heading);
    });

    screen.getByRole("link", { name: "Training" }).focus();
    document.title = "Stale title";
    act(() => {
      window.history.pushState(
        null,
        "",
        "/training?status=running#metrics",
      );
      window.dispatchEvent(new PopStateEvent("popstate"));
    });

    await waitFor(() => {
      expect(window.location.search).toBe("?status=running");
      expect(document.title).toBe("Training | AL-MedLit");
      expect(
        screen.getByText(
          "Training loaded. Status running. Section metrics.",
        ),
      ).toBeTruthy();
      expect(document.activeElement).toBe(heading);
    });

    screen.getByRole("link", { name: "Training" }).focus();
    document.title = "Stale title";
    act(() => window.history.back());

    await waitFor(() => {
      expect(window.location.search).toBe("?status=queued");
      expect(window.location.hash).toBe("#runs");
      expect(document.title).toBe("Training | AL-MedLit");
      expect(
        screen.getByText(
          "Training loaded. Status queued. Section runs.",
        ),
      ).toBeTruthy();
      expect(document.activeElement).toBe(heading);
    });
  });

  it("passes model and version parameters through the nested registry route", async () => {
    window.history.replaceState(null, "", "/models/31/versions/47");
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 10,
      preset: "annotate_train",
      overrides: [],
      effective: ["annotation", "training"],
      blocked: {},
    });

    render(<App />);

    const workspace = await screen.findByLabelText("Mock models workspace");
    expect(workspace.getAttribute("data-model-id")).toBe("31");
    expect(workspace.getAttribute("data-version-id")).toBe("47");
    expect(screen.getByText("version")).toBeTruthy();
  });

  it("does not fetch manager-only legacy project data for a Trainer", async () => {
    window.history.replaceState(null, "", "/projects/81/overview");
    Object.assign(mocks.projectState, {
      selectedProjectId: 81,
      projects: [{
        id: 81,
        name: "Trainer-readable project",
        description: null,
        annotation_schema: [],
        annotation_validation_mode: "relaxed",
        tasks: [],
        settings: {
          modules: ["data", "train", "models", "activity"],
        },
        workspace_id: 40,
      }],
    });
    mocks.getMe.mockResolvedValue({
      ...me,
      memberships: [
        {
          workspace_id: 40,
          workspace_name: "Training team",
          workspace_kind: "team",
          role: "trainer",
        },
      ],
    });
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 40,
      preset: "train",
      overrides: [],
      effective: ["training", "lineage"],
      blocked: {},
    });

    render(<App />);

    await screen.findByLabelText("Mock project platform");
    expect(mocks.projectState.loadProjectData).not.toHaveBeenCalled();
  });

  it("uses the canonical route project for reads and writes across history navigation", async () => {
    window.history.replaceState(null, "", "/projects/81/data");
    Object.assign(mocks.projectState, {
      selectedProjectId: 82,
      projects: [
        {
          id: 81,
          name: "Route-owned project",
          description: null,
          annotation_schema: [],
          annotation_validation_mode: "relaxed",
          tasks: [],
          settings: { modules: ["data", "activity"] },
          workspace_id: 10,
        },
        {
          id: 82,
          name: "Previously selected project",
          description: null,
          annotation_schema: [],
          annotation_validation_mode: "relaxed",
          tasks: [],
          settings: { modules: ["data", "activity"] },
          workspace_id: 10,
        },
      ],
    });

    render(<App />);

    let project = await screen.findByLabelText("Mock project platform");
    expect(project.getAttribute("data-project-id")).toBe("81");
    await waitFor(() =>
      expect(mocks.projectState.loadProjectData).toHaveBeenCalledWith(
        81,
        false,
        "all",
      ),
    );
    expect(mocks.projectState.loadProjectData).not.toHaveBeenCalledWith(
      82,
      expect.anything(),
      expect.anything(),
    );

    fireEvent.click(screen.getByRole("button", { name: "Update mock project" }));
    await waitFor(() =>
      expect(mocks.updateProject).toHaveBeenLastCalledWith(
        81,
        { name: "Route-owned project updated" },
      ),
    );

    act(() => {
      window.history.pushState(null, "", "/projects/82/data");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    project = await screen.findByLabelText("Mock project platform");
    await waitFor(() =>
      expect(project.getAttribute("data-project-id")).toBe("82"),
    );
    fireEvent.click(screen.getByRole("button", { name: "Update mock project" }));
    await waitFor(() =>
      expect(mocks.updateProject).toHaveBeenLastCalledWith(
        82,
        { name: "Previously selected project updated" },
      ),
    );

    act(() => window.history.back());
    await waitFor(() => expect(window.location.pathname).toBe("/projects/81/data"));
    expect(
      (await screen.findByLabelText("Mock project platform")).getAttribute(
        "data-project-id",
      ),
    ).toBe("81");

    act(() => window.history.forward());
    await waitFor(() => expect(window.location.pathname).toBe("/projects/82/data"));
    expect(
      (await screen.findByLabelText("Mock project platform")).getAttribute(
        "data-project-id",
      ),
    ).toBe("82");
  });

  it("shows system Administration only to a deployment superuser", async () => {
    window.history.replaceState(null, "", "/admin/health");
    mocks.getMe.mockResolvedValue({
      ...me,
      user: { ...me.user, is_superuser: true },
      memberships: [
        {
          workspace_id: 50,
          workspace_name: "Operations",
          workspace_kind: "team",
          role: "admin",
        },
      ],
    });
    mocks.getWorkspaceCapabilities.mockResolvedValue({
      workspace_id: 50,
      preset: "full",
      overrides: [],
      effective: ["annotation", "training"],
      blocked: {},
    });

    render(<App />);

    await screen.findByLabelText("Mock system administration");
    expect(
      screen.getByRole("link", { name: "Administration" }).getAttribute(
        "aria-current",
      ),
    ).toBe("page");
    expect(screen.getByRole("link", { name: "Workspace Settings" })).toBeTruthy();
  });

  it("resolves a direct assigned-round URL to its owning project", async () => {
    window.history.replaceState(null, "", "/my-work/rounds/71");

    render(<App />);

    await waitFor(() =>
      expect(mocks.getRoundWorkContext).toHaveBeenCalledWith(71, 10),
    );
    expect(mocks.projectState.setSelectedProjectId).toHaveBeenCalledWith(81);
    expect(
      await screen.findByLabelText("Mock round workbench"),
    ).toBeTruthy();
    expect(
      mocks.usePlatformProject.mock.calls.some(
        ([projectId, , enabled]) => projectId !== null || enabled === true,
      ),
    ).toBe(false);
    expect(window.location.pathname).toBe("/my-work/rounds/71");
  });

  it("loads My Work rounds from the assignment-scoped workspace endpoint", async () => {
    window.history.replaceState(null, "", "/my-work");
    const context = await mocks.getRoundWorkContext(71);
    mocks.getRoundWorkContext.mockClear();
    mocks.listWorkspaceRoundWorkContexts.mockResolvedValue([context]);

    render(<App />);

    await waitFor(() =>
      expect(mocks.listWorkspaceRoundWorkContexts).toHaveBeenCalledWith(10),
    );
    expect(
      screen
        .getByLabelText("Mock annotator workspace")
        .getAttribute("data-round-contexts"),
    ).toBe("71");
    expect(mocks.getRoundWorkContext).not.toHaveBeenCalled();
    expect(
      mocks.usePlatformProject.mock.calls.some(
        ([projectId, , enabled]) => projectId !== null || enabled === true,
      ),
    ).toBe(false);
  });

  it("keeps the real system administration URL and location state for superusers", async () => {
    window.history.replaceState(
      null,
      "",
      "/admin/users?tab=members#invite",
    );
    mocks.getMe.mockResolvedValue({
      ...me,
      user: { ...me.user, is_superuser: true },
      memberships: [
        {
          workspace_id: 60,
          workspace_name: "Research administration",
          workspace_kind: "team",
          role: "admin",
        },
      ],
    });

    render(<App />);

    await screen.findByLabelText("Mock system administration");
    expect(window.location.pathname).toBe("/admin/users");
    expect(window.location.search).toBe("?tab=members");
    expect(window.location.hash).toBe("#invite");
  });

  it("defaults exact admin URLs for a superuser without a workspace membership", async () => {
    window.history.replaceState(null, "", "/admin");
    mocks.getMe.mockResolvedValue({
      ...me,
      user: { ...me.user, is_superuser: true },
      memberships: [],
    });

    render(<App />);

    await screen.findByLabelText("Mock system administration");
    await waitFor(() => expect(window.location.pathname).toBe("/admin/users"));
  });

  it("preserves an invite URL when a stored session token is stale", async () => {
    window.history.replaceState(null, "", "/invites/invite-token?source=test#accept");
    mocks.getMe.mockRejectedValueOnce(new Error("Invalid bearer token"));

    render(<App />);

    const page = await screen.findByLabelText("Mock invite acceptance");
    expect(page.getAttribute("data-token")).toBe("invite-token");
    expect(page.getAttribute("data-signed-in")).toBe("false");
    expect(window.location.pathname).toBe("/invites/invite-token");
    expect(window.location.search).toBe("?source=test");
    expect(window.location.hash).toBe("#accept");
  });

  it("preserves an account-action URL when a stored session token is stale", async () => {
    window.history.replaceState(null, "", "/account-actions/reset-token");
    mocks.getMe.mockRejectedValueOnce(new Error("Invalid bearer token"));

    render(<App />);

    const page = await screen.findByLabelText("Mock account action");
    expect(page.getAttribute("data-token")).toBe("reset-token");
    expect(window.location.pathname).toBe("/account-actions/reset-token");
  });
});
