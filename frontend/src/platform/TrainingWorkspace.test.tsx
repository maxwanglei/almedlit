// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Project } from "@/types/api";

import TrainingWorkspace, { trainingDraftKey } from "./TrainingWorkspace";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
  type WorkspaceTrainingContext,
  type WorkspaceTrainingRun,
} from "./types";

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

const mocks = vi.hoisted(() => ({
  cancelTrainingRun: vi.fn(),
  createDatasetWithVersion: vi.fn(),
  createExecutionEnvironment: vi.fn(),
  createStoragePolicy: vi.fn(),
  createTaskWithVersion: vi.fn(),
  createTrainingDataset: vi.fn(),
  createTrainingOnlyProject: vi.fn(),
  findWorkspaceTrainingRun: vi.fn(),
  launchTrainingRun: vi.fn(),
  listWorkspaceTrainingContexts: vi.fn(),
  listWorkspaceTrainingRuns: vi.fn(),
  loadPlatformProject: vi.fn(),
  verifyExecutionEnvironment: vi.fn(),
}));

vi.mock("./api", () => mocks);

const project: Project = {
  id: 7,
  name: "Evidence classification",
  description: null,
  annotation_schema: { labels: {} },
  annotation_validation_mode: "strict",
  tasks: [],
  settings: {},
  workspace_id: 5,
};

const otherProject: Project = {
  ...project,
  id: 8,
  name: "Generation benchmark",
  workspace_id: 6,
};

function trainingContext(
  sourceProject: Project,
): WorkspaceTrainingContext {
  return {
    project_id: sourceProject.id,
    project_name: sourceProject.name,
    project_description: sourceProject.description,
    training_only: true,
    effective_modules: ["data", "train", "models", "activity"],
    task_version_count: 1,
    training_dataset_version_count: 1,
    environment_count: 0,
    available_environment_count: 0,
    storage_policy_count: 0,
  };
}

function trainingProjectData(
  projectId: number,
  datasetName: string,
): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    projectModules: {
      ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
      project_id: projectId,
      selected: ["data", "train", "models"],
      effective: ["data", "train", "models"],
    },
    taskVersions: [
      {
        id: projectId * 10 + 1,
        project_id: projectId,
        task_definition_id: projectId * 10 + 2,
        version_number: 1,
        task_kind: "classification",
        input_schema: {},
        output_schema: {},
        label_rules: {},
        annotation_ui: {},
        metrics: ["accuracy"],
        trainer_compatibility: [],
        content_hash: `task-${projectId}`,
      },
    ],
    trainingDatasets: [
      {
        id: projectId * 10 + 3,
        project_id: projectId,
        name: datasetName,
        dataset_version_id: projectId * 10 + 4,
        task_version_id: projectId * 10 + 1,
        label_set_version_ids: [],
        split_map_id: projectId * 10 + 5,
        composition: [],
        preprocessing: {},
        content_hash: `dataset-${projectId}`,
      },
    ],
  };
}

function workspaceRun(
  id: number,
  projectId: number,
  modelName: string,
): WorkspaceTrainingRun {
  return {
    id,
    project_id: projectId,
    registered_model_id: id + 1,
    task_version_id: id + 2,
    training_dataset_version_id: id + 3,
    recipe_version_id: id + 4,
    environment_id: id + 5,
    storage_policy_id: id + 6,
    idempotency_key: `run-${id}`,
    status: "succeeded",
    seed: 42,
    config: {},
    evaluation_plan: {},
    output_model_version_id: id + 7,
    created_by_user_id: 12,
    project_name: `Project ${projectId}`,
    project_description: null,
    model_name: modelName,
    family: "conventional_ml",
    framework: "scikit-learn",
    base_model: {},
    training_method: "full",
    task_name: "Classification",
    task_kind: "classification",
    training_dataset_name: `Dataset ${projectId}`,
    dataset_version_number: 1,
    recipe_name: "Linear classifier",
    runtime_name: "Classical CPU",
    storage_policy_name: "Workspace storage",
    evaluation_status: "succeeded",
    evaluation_split: "test",
    evaluation_metrics: { accuracy: 0.9 },
    creator_username: "trainer",
    creator_display_name: "Tess Trainer",
  };
}

const baseProps = {
  workspaceId: 5,
  projects: [project],
  currentUserId: 12,
  currentUserName: "Tess Trainer",
  canCreateTrainingProject: false,
  canCreateTask: false,
  canProvisionRuntimes: false,
  onNavigate: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
  mocks.listWorkspaceTrainingContexts.mockResolvedValue({
    items: [],
    next_cursor: null,
  });
  mocks.listWorkspaceTrainingRuns.mockResolvedValue({
    items: [],
    next_cursor: null,
  });
  mocks.loadPlatformProject.mockResolvedValue(EMPTY_PLATFORM_PROJECT_DATA);
  mocks.createExecutionEnvironment.mockResolvedValue({
    id: 131,
    project_id: 7,
    name: "Classical CPU",
    environment_class: "classical-cpu",
    image_digest: `sha256:${"a".repeat(64)}`,
    package_manifest: {},
    hardware_constraints: {},
    verification_report: {},
    status: "unavailable",
    verified_at: null,
  });
  mocks.verifyExecutionEnvironment.mockResolvedValue({
    id: 131,
    project_id: 7,
    name: "Classical CPU",
    environment_class: "classical-cpu",
    image_digest: `sha256:${"a".repeat(64)}`,
    package_manifest: {},
    hardware_constraints: {},
    verification_report: { device: "cpu" },
    status: "available",
    verified_at: "2026-07-28T12:00:00Z",
  });
  mocks.createStoragePolicy.mockResolvedValue({
    id: 141,
    project_id: 7,
    name: "Research artifacts",
    backend: "minio",
    artifact_prefix: "workspaces/5/artifact-blobs",
    retention_class: "indefinite",
    encryption: {},
    cache_policy: {},
    is_default: true,
  });
});

afterEach(() => cleanup());

describe("TrainingWorkspace", () => {
  it.each(["new", "data", "runtimes"] as const)(
    "does not load an unconfirmed project from the %s route",
    async (view) => {
      render(
        <TrainingWorkspace
          {...baseProps}
          view={view}
          initialProjectId={999}
        />,
      );

      await waitFor(() =>
        expect(mocks.listWorkspaceTrainingContexts).toHaveBeenCalledWith(
          5,
          [project],
          { limit: 100 },
        ),
      );
      if (view === "new") {
        expect(
          await screen.findByRole("heading", { name: "Training context" }),
        ).toBeTruthy();
      } else if (view === "data") {
        expect(await screen.findByText("No training project")).toBeTruthy();
      } else {
        expect(
          await screen.findByRole("heading", {
            name: "Execution environments",
          }),
        ).toBeTruthy();
      }
      expect(mocks.loadPlatformProject).not.toHaveBeenCalled();
    },
  );

  it("keeps a late context response from replacing the active workspace", async () => {
    const oldContexts = deferred<{
      items: WorkspaceTrainingContext[];
      next_cursor: number | null;
    }>();
    mocks.listWorkspaceTrainingContexts.mockImplementation(
      async (workspaceId: number) =>
        workspaceId === 5
          ? oldContexts.promise
          : {
              items: [trainingContext(otherProject)],
              next_cursor: null,
            },
    );
    mocks.loadPlatformProject.mockImplementation(
      async (projectId: number) =>
        trainingProjectData(projectId, "Fresh workspace dataset"),
    );

    const { rerender } = render(
      <TrainingWorkspace
        {...baseProps}
        view="new"
        initialProjectId={project.id}
      />,
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceTrainingContexts).toHaveBeenCalledWith(
        5,
        [project],
        { limit: 100 },
      ),
    );

    rerender(
      <TrainingWorkspace
        {...baseProps}
        workspaceId={6}
        projects={[otherProject]}
        view="new"
        initialProjectId={otherProject.id}
      />,
    );

    expect(
      (await screen.findByText(/Project context:/i)).textContent,
    ).toContain(otherProject.name);
    expect(
      await screen.findByRole("option", {
        name: "Fresh workspace dataset",
      }),
    ).toBeTruthy();

    await act(async () => {
      oldContexts.resolve({
        items: [trainingContext(project)],
        next_cursor: null,
      });
      await oldContexts.promise;
    });

    expect(screen.getByText(/Project context:/i).textContent).toContain(
      otherProject.name,
    );
    expect(mocks.loadPlatformProject).not.toHaveBeenCalledWith(
      project.id,
      expect.anything(),
      5,
    );
  });

  it("keeps late project data from replacing the newly selected context", async () => {
    const oldTrainData = deferred<PlatformProjectData>();
    const oldSourceData = deferred<PlatformProjectData>();
    mocks.listWorkspaceTrainingContexts.mockResolvedValue({
      items: [trainingContext(project), trainingContext(otherProject)],
      next_cursor: null,
    });
    mocks.loadPlatformProject.mockImplementation(
      async (projectId: number, scope: string) => {
        if (projectId === project.id) {
          return scope === "train"
            ? oldTrainData.promise
            : oldSourceData.promise;
        }
        return trainingProjectData(otherProject.id, "Current dataset");
      },
    );

    const { rerender } = render(
      <TrainingWorkspace
        {...baseProps}
        projects={[project, otherProject]}
        view="new"
        initialProjectId={project.id}
      />,
    );
    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(
        project.id,
        "train",
        5,
      ),
    );

    rerender(
      <TrainingWorkspace
        {...baseProps}
        projects={[project, otherProject]}
        view="new"
        initialProjectId={otherProject.id}
      />,
    );
    expect(
      await screen.findByRole("option", { name: "Current dataset" }),
    ).toBeTruthy();

    await act(async () => {
      oldTrainData.resolve(trainingProjectData(project.id, "Stale dataset"));
      oldSourceData.resolve(trainingProjectData(project.id, "Stale dataset"));
      await Promise.all([oldTrainData.promise, oldSourceData.promise]);
    });

    expect(
      screen.getByRole("option", { name: "Current dataset" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("option", { name: "Stale dataset" }),
    ).toBeNull();
  });

  it("keeps late Training Data results from replacing the active project", async () => {
    const secondProject = { ...otherProject, workspace_id: 5 };
    const oldTrainData = deferred<PlatformProjectData>();
    const oldSourceData = deferred<PlatformProjectData>();
    mocks.listWorkspaceTrainingContexts.mockResolvedValue({
      items: [trainingContext(project), trainingContext(secondProject)],
      next_cursor: null,
    });
    mocks.loadPlatformProject.mockImplementation(
      async (projectId: number, scope: string) => {
        if (projectId === project.id) {
          return scope === "train"
            ? oldTrainData.promise
            : oldSourceData.promise;
        }
        return trainingProjectData(
          secondProject.id,
          "Current training dataset",
        );
      },
    );

    const { rerender } = render(
      <TrainingWorkspace
        {...baseProps}
        projects={[project, secondProject]}
        view="data"
        initialProjectId={project.id}
      />,
    );
    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(
        project.id,
        "data",
        5,
      ),
    );

    rerender(
      <TrainingWorkspace
        {...baseProps}
        projects={[project, secondProject]}
        view="data"
        initialProjectId={secondProject.id}
      />,
    );
    expect(
      await screen.findByText("Current training dataset"),
    ).toBeTruthy();

    await act(async () => {
      oldSourceData.resolve(
        trainingProjectData(project.id, "Stale training dataset"),
      );
      oldTrainData.resolve(
        trainingProjectData(project.id, "Stale training dataset"),
      );
      await Promise.all([oldSourceData.promise, oldTrainData.promise]);
    });

    expect(screen.getByText("Current training dataset")).toBeTruthy();
    expect(screen.queryByText("Stale training dataset")).toBeNull();
  });

  it("ignores an older workspace run page and its loading completion", async () => {
    const oldPage = deferred<{
      items: WorkspaceTrainingRun[];
      next_cursor: number | null;
    }>();
    const currentPage = deferred<{
      items: WorkspaceTrainingRun[];
      next_cursor: number | null;
    }>();
    mocks.listWorkspaceTrainingRuns.mockImplementation(
      async (workspaceId: number) =>
        workspaceId === 5 ? oldPage.promise : currentPage.promise,
    );

    const { rerender } = render(
      <TrainingWorkspace {...baseProps} view="runs" />,
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceTrainingRuns).toHaveBeenCalledWith(
        5,
        expect.anything(),
        [project],
      ),
    );

    rerender(
      <TrainingWorkspace
        {...baseProps}
        workspaceId={6}
        projects={[otherProject]}
        view="runs"
      />,
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceTrainingRuns).toHaveBeenCalledWith(
        6,
        expect.anything(),
        [otherProject],
      ),
    );

    await act(async () => {
      oldPage.resolve({
        items: [workspaceRun(71, project.id, "Stale model")],
        next_cursor: null,
      });
      await oldPage.promise;
    });
    expect(screen.getByText("Loading training runs…")).toBeTruthy();
    expect(screen.queryByText("Stale model")).toBeNull();

    await act(async () => {
      currentPage.resolve({
        items: [workspaceRun(72, otherProject.id, "Current model")],
        next_cursor: null,
      });
      await currentPage.promise;
    });

    expect(await screen.findByText("Current model")).toBeTruthy();
    expect(screen.queryByText("Stale model")).toBeNull();
  });

  it("ignores an older run detail response and its loading completion", async () => {
    const oldRun = deferred<WorkspaceTrainingRun>();
    const currentRun = deferred<WorkspaceTrainingRun>();
    mocks.findWorkspaceTrainingRun.mockImplementation(
      async (_workspaceId: number, runId: number) =>
        runId === 71 ? oldRun.promise : currentRun.promise,
    );

    const { rerender } = render(
      <TrainingWorkspace
        {...baseProps}
        view="run-detail"
        runId={71}
      />,
    );
    await waitFor(() =>
      expect(mocks.findWorkspaceTrainingRun).toHaveBeenCalledWith(
        5,
        71,
        [project],
      ),
    );

    rerender(
      <TrainingWorkspace
        {...baseProps}
        view="run-detail"
        runId={72}
      />,
    );
    await waitFor(() =>
      expect(mocks.findWorkspaceTrainingRun).toHaveBeenCalledWith(
        5,
        72,
        [project],
      ),
    );

    await act(async () => {
      oldRun.resolve(workspaceRun(71, project.id, "Stale detail model"));
      await oldRun.promise;
    });
    expect(
      screen.getByRole("heading", { name: "Training run" }),
    ).toBeTruthy();
    expect(screen.queryByText("Stale detail model")).toBeNull();

    await act(async () => {
      currentRun.resolve(
        workspaceRun(72, project.id, "Current detail model"),
      );
      await currentRun.promise;
    });
    expect(
      await screen.findByRole("heading", { name: "Training run 72" }),
    ).toBeTruthy();
    expect(
      screen.getAllByText(/Current detail model/).length,
    ).toBeGreaterThan(0);
    expect(screen.queryByText("Stale detail model")).toBeNull();
  });

  it("lets a team Trainer select an existing context but not create a project", async () => {
    render(<TrainingWorkspace {...baseProps} view="new" />);

    await userEvent.click(
      screen.getByRole("radio", { name: /training-only project/i }),
    );

    expect(
      screen.getByText(/ask a workspace manager to create/i),
    ).toBeTruthy();
    expect(
      screen.queryByRole("heading", { name: "Create training-only project" }),
    ).toBeNull();
  });

  it("creates a training-only project with manager authorization", async () => {
    const created = { ...project, id: 19, name: "Public benchmark" };
    mocks.createTrainingOnlyProject.mockResolvedValue(created);
    const onProjectCreated = vi.fn();
    const onNavigate = vi.fn();

    render(
      <TrainingWorkspace
        {...baseProps}
        view="new"
        canCreateTrainingProject
        onProjectCreated={onProjectCreated}
        onNavigate={onNavigate}
      />,
    );

    await userEvent.click(
      screen.getByRole("radio", { name: /training-only project/i }),
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Project name" }),
      "Public benchmark",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Description" }),
      "Pinned public data",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Create project" }),
    );

    await waitFor(() => {
      expect(mocks.createTrainingOnlyProject).toHaveBeenCalledWith(
        5,
        "Public benchmark",
        "Pinned public data",
      );
      expect(onProjectCreated).toHaveBeenCalledWith(created);
      expect(onNavigate).not.toHaveBeenCalled();
    });
    expect(
      await screen.findByRole("button", {
        name: "Step 1 of 5: Task & Data",
      }),
    ).toBeTruthy();
    await waitFor(() => {
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(19, "train", 5);
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(19, "data", 5);
    });
    expect(screen.getByText(/Project context:/i).textContent).toContain(
      "Public benchmark",
    );
  });

  it("keys drafts by workspace, project, and optional dataset", () => {
    expect(trainingDraftKey(5, 7)).toBe("5:7:none");
    expect(trainingDraftKey(5, 7, 22)).toBe("5:7:22");
    expect(trainingDraftKey(6, 7, 22)).not.toBe(
      trainingDraftKey(5, 7, 22),
    );
  });

  it("shows complete workspace run context and only offers creator cancellation", async () => {
    mocks.listWorkspaceTrainingRuns.mockResolvedValue({
      items: [
        {
          id: 151,
          project_id: 7,
          registered_model_id: 81,
          task_version_id: 12,
          training_dataset_version_id: 51,
          recipe_version_id: 121,
          environment_id: 131,
          storage_policy_id: 141,
          idempotency_key: "run-151",
          status: "running",
          seed: 42,
          config: {},
          evaluation_plan: {},
          output_model_version_id: null,
          created_by_user_id: 12,
          project_name: "Evidence classification",
          project_description: null,
          model_name: "Relevance baseline",
          family: "conventional_ml",
          framework: "scikit-learn",
          base_model: {
            base_model_asset_id: 161,
            source_model_id: "medical/abstract-encoder",
            exact_revision: "encoder-commit-42",
          },
          training_method: "full",
          task_name: "Abstract relevance",
          task_kind: "classification",
          training_dataset_name: "Relevance set",
          dataset_version_number: 3,
          recipe_name: "TF-IDF logistic regression",
          runtime_name: "Classical CPU",
          storage_policy_name: "Workspace MinIO",
          evaluation_status: "succeeded",
          evaluation_split: "test",
          evaluation_metrics: { macro_f1: 0.91 },
          creator_username: "trainer",
          creator_display_name: "Tess Trainer",
        },
      ],
      next_cursor: null,
    });
    mocks.cancelTrainingRun.mockResolvedValue({
      id: 151,
      status: "cancelled",
    });

    render(<TrainingWorkspace {...baseProps} view="runs" />);

    expect(await screen.findByText("Relevance baseline")).toBeTruthy();
    expect(screen.getByText(/scikit-learn/i)).toBeTruthy();
    expect(
      screen.getByText(
        /Asset: 161 · Source: medical\/abstract-encoder · Revision: encoder-commit-42/i,
      ),
    ).toBeTruthy();
    expect(screen.getByText(/Relevance set v3/i)).toBeTruthy();
    expect(screen.getByText("Classical CPU")).toBeTruthy();
    expect(screen.getByText("Workspace MinIO")).toBeTruthy();
    expect(screen.getByText("Tess Trainer")).toBeTruthy();

    await userEvent.type(
      screen.getByRole("combobox", { name: "Model family" }),
      "custom_adapter_family",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Apply filters" }),
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceTrainingRuns).toHaveBeenLastCalledWith(
        5,
        expect.objectContaining({ family: "custom_adapter_family" }),
        [project],
      ),
    );

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() =>
      expect(mocks.cancelTrainingRun).toHaveBeenCalledWith(7, 151),
    );
    expect((await screen.findAllByText("Cancelled")).length).toBeGreaterThan(1);
  });

  it("lets administrators register and verify runtimes and add storage", async () => {
    mocks.listWorkspaceTrainingContexts.mockResolvedValue({
      items: [trainingContext(project)],
      next_cursor: null,
    });
    mocks.loadPlatformProject.mockResolvedValue(
      trainingProjectData(project.id, "Training dataset"),
    );

    render(
      <TrainingWorkspace
        {...baseProps}
        view="runtimes"
        initialProjectId={project.id}
        canProvisionRuntimes
      />,
    );

    await screen.findByRole("heading", {
      name: "Provision project infrastructure",
    });
    await userEvent.type(
      screen.getByRole("textbox", { name: "Runtime name" }),
      "Classical CPU",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Immutable image digest" }),
      `sha256:${"a".repeat(64)}`,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Register runtime" }),
    );

    await waitFor(() =>
      expect(mocks.createExecutionEnvironment).toHaveBeenCalledWith(7, {
        name: "Classical CPU",
        environmentClass: "classical-cpu",
        imageDigest: `sha256:${"a".repeat(64)}`,
        packageManifest: {},
        hardwareConstraints: {},
      }),
    );

    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Runtime" }),
      "131",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Record preflight" }),
    );
    await waitFor(() =>
      expect(mocks.verifyExecutionEnvironment).toHaveBeenCalledWith(
        7,
        131,
        "available",
        {},
      ),
    );

    await userEvent.type(
      screen.getByRole("textbox", { name: "Policy name" }),
      "Research artifacts",
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Add storage policy" }),
    );
    await waitFor(() =>
      expect(mocks.createStoragePolicy).toHaveBeenCalledWith(7, {
        name: "Research artifacts",
        backend: "minio",
        artifactPrefix: "workspaces/5/artifact-blobs",
        retentionClass: "indefinite",
        encryption: {},
        cachePolicy: {},
        isDefault: true,
      }),
    );
    expect(
      await screen.findByText("Storage policy added."),
    ).toBeTruthy();
  }, 10_000);

  it("keeps provisioning controls hidden from non-administrators", async () => {
    mocks.listWorkspaceTrainingContexts.mockResolvedValue({
      items: [trainingContext(project)],
      next_cursor: null,
    });

    render(
      <TrainingWorkspace
        {...baseProps}
        view="runtimes"
        initialProjectId={project.id}
      />,
    );

    expect(
      await screen.findByRole("heading", { name: "Execution environments" }),
    ).toBeTruthy();
    expect(
      screen.queryByRole("heading", {
        name: "Provision project infrastructure",
      }),
    ).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Register runtime" }),
    ).toBeNull();
  });

  it("does not apply a late provisioning response to another project", async () => {
    const secondProject = { ...otherProject, workspace_id: 5 };
    const pendingEnvironment = deferred<{
      id: number;
      project_id: number;
      name: string;
      environment_class: string;
      image_digest: string;
      package_manifest: Record<string, unknown>;
      hardware_constraints: Record<string, unknown>;
      verification_report: Record<string, unknown>;
      status: string;
      verified_at: null;
    }>();
    mocks.listWorkspaceTrainingContexts.mockResolvedValue({
      items: [trainingContext(project), trainingContext(secondProject)],
      next_cursor: null,
    });
    mocks.loadPlatformProject.mockImplementation(async (projectId: number) =>
      trainingProjectData(projectId, `Dataset ${projectId}`),
    );
    mocks.createExecutionEnvironment.mockReturnValue(
      pendingEnvironment.promise,
    );

    render(
      <TrainingWorkspace
        {...baseProps}
        projects={[project, secondProject]}
        view="runtimes"
        initialProjectId={project.id}
        canProvisionRuntimes
      />,
    );

    await userEvent.type(
      await screen.findByRole("textbox", { name: "Runtime name" }),
      "Late runtime",
    );
    await userEvent.type(
      screen.getByRole("textbox", { name: "Immutable image digest" }),
      `sha256:${"b".repeat(64)}`,
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Register runtime" }),
    );
    await userEvent.selectOptions(
      screen.getByRole("combobox", { name: "Project context" }),
      String(secondProject.id),
    );

    await act(async () => {
      pendingEnvironment.resolve({
        id: 199,
        project_id: project.id,
        name: "Late runtime",
        environment_class: "classical-cpu",
        image_digest: `sha256:${"b".repeat(64)}`,
        package_manifest: {},
        hardware_constraints: {},
        verification_report: {},
        status: "unavailable",
        verified_at: null,
      });
      await pendingEnvironment.promise;
    });

    expect(screen.queryByText("Runtime registered. Record a worker preflight result before launch.")).toBeNull();
    expect(screen.queryByText("Late runtime")).toBeNull();
  });
});
