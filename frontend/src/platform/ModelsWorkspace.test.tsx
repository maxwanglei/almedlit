// @vitest-environment jsdom

import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { Project } from "@/types/api";

import ModelsWorkspace from "./ModelsWorkspace";
import { EMPTY_PLATFORM_PROJECT_DATA } from "./types";

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
  findWorkspaceModel: vi.fn(),
  listWorkspaceModels: vi.fn(),
  loadPlatformProject: vi.fn(),
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

const modelSummary = {
  id: 81,
  project_id: 7,
  name: "Relevance baseline",
  description: "Screening classifier",
  lifecycle_status: "active",
  created_by_user_id: 12,
  project_name: "Evidence classification",
  project_description: null,
  latest_version: {
    id: 82,
    project_id: 7,
    registered_model_id: 81,
    version_number: 2,
    parent_version_id: 80,
    task_version_id: 12,
    training_dataset_version_id: 51,
    family: "conventional_ml",
    framework: "scikit-learn",
    base_model: {
      base_model_asset_id: 161,
      source_model_id: "medical/abstract-encoder",
      exact_revision: "encoder-commit-42",
    },
    training_method: "full",
    recipe_key: "tfidf_logistic_regression",
    recipe_version: "1",
    parameters: {},
    metrics: {},
    runtime_digest: "a".repeat(64),
    content_hash: "b".repeat(64),
    training_run_id: 151,
    source_dataset_version_id: 31,
    source_dataset_version_number: 3,
    runtime_id: 71,
    runtime_name: "Classical CPU",
    storage_policy_id: 72,
    storage_policy_name: "Research artifacts",
    creator_username: "version-trainer",
    creator_display_name: "Vera Version",
  },
  task_name: "Abstract relevance",
  task_kind: "classification",
  training_dataset_name: "Relevance set",
  evaluation_status: "succeeded",
  evaluation_split: "test",
  evaluation_metrics: { macro_f1: 0.91 },
  creator_username: "trainer",
  creator_display_name: "Tess Trainer",
} as const;

beforeEach(() => {
  vi.clearAllMocks();
  mocks.listWorkspaceModels.mockResolvedValue({
    items: [modelSummary],
    next_cursor: null,
  });
  mocks.findWorkspaceModel.mockResolvedValue(modelSummary);
  mocks.loadPlatformProject.mockResolvedValue({
    ...EMPTY_PLATFORM_PROJECT_DATA,
    models: [modelSummary],
    modelVersions: [modelSummary.latest_version],
    taskDefinitions: [
      {
        id: 11,
        project_id: 7,
        key: "relevance",
        name: "Abstract relevance",
        description: null,
      },
    ],
    taskVersions: [
      {
        id: 12,
        project_id: 7,
        task_definition_id: 11,
        version_number: 4,
        task_kind: "classification",
        input_schema: {},
        output_schema: {},
        label_rules: {},
        annotation_ui: {},
        metrics: ["macro_f1"],
        trainer_compatibility: ["tfidf_logistic_regression"],
        content_hash: "c".repeat(64),
      },
    ],
    trainingDatasets: [
      {
        id: 51,
        project_id: 7,
        name: "Relevance set",
        dataset_version_id: 31,
        task_version_id: 12,
        label_set_version_ids: [41],
        split_map_id: 61,
        composition: [],
        preprocessing: {},
        content_hash: "d".repeat(64),
      },
    ],
    modelEvaluations: [
      {
        id: 91,
        project_id: 7,
        training_run_id: 151,
        model_version_id: 82,
        task_version_id: 12,
        training_dataset_version_id: 51,
        dataset_version_id: 31,
        split_map_id: 61,
        artifact_package_id: null,
        split_name: "test",
        status: "succeeded",
        evaluator_key: "classification",
        evaluator_version: "1",
        row_count: 100,
        requested_metrics: ["macro_f1"],
        metrics: { macro_f1: 0.91 },
        report: {},
        evaluation_plan: {},
        runtime_digest: "e".repeat(64),
        code_digest: "f".repeat(64),
        status_reason: null,
        content_hash: "0".repeat(64),
      },
    ],
  });
});

afterEach(() => cleanup());

describe("ModelsWorkspace", () => {
  it("ignores an older workspace registry response and its completion", async () => {
    const oldPage = deferred<{
      items: Array<typeof modelSummary>;
      next_cursor: number | null;
    }>();
    const currentModel = {
      ...modelSummary,
      id: 91,
      project_id: otherProject.id,
      name: "Current generation model",
      project_name: otherProject.name,
      latest_version: {
        ...modelSummary.latest_version,
        id: 92,
        project_id: otherProject.id,
        registered_model_id: 91,
      },
    };
    const currentPage = deferred<{
      items: Array<typeof currentModel>;
      next_cursor: number | null;
    }>();
    mocks.listWorkspaceModels.mockImplementation(
      async (workspaceId: number) =>
        workspaceId === 5 ? oldPage.promise : currentPage.promise,
    );

    const { rerender } = render(
      <ModelsWorkspace
        workspaceId={5}
        projects={[project]}
        view="registry"
        onNavigate={vi.fn()}
      />,
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceModels).toHaveBeenCalledWith(
        5,
        expect.anything(),
        [project],
      ),
    );

    rerender(
      <ModelsWorkspace
        workspaceId={6}
        projects={[otherProject]}
        view="registry"
        onNavigate={vi.fn()}
      />,
    );
    await waitFor(() =>
      expect(mocks.listWorkspaceModels).toHaveBeenCalledWith(
        6,
        expect.anything(),
        [otherProject],
      ),
    );

    await act(async () => {
      oldPage.resolve({ items: [modelSummary], next_cursor: null });
      await oldPage.promise;
    });
    expect(screen.getByText("Loading models…")).toBeTruthy();
    expect(screen.queryByText("Relevance baseline")).toBeNull();

    await act(async () => {
      currentPage.resolve({ items: [currentModel], next_cursor: null });
      await currentPage.promise;
    });
    expect(
      await screen.findByText("Current generation model"),
    ).toBeTruthy();
    expect(screen.queryByText("Relevance baseline")).toBeNull();
  });

  it("shows model identity, project, family, framework, method, data, evaluation, and creator", async () => {
    render(
      <ModelsWorkspace
        workspaceId={5}
        projects={[project]}
        view="registry"
        onNavigate={vi.fn()}
      />,
    );

    expect(await screen.findByText("Relevance baseline")).toBeTruthy();
    expect(screen.getAllByText("Evidence classification").length).toBeGreaterThan(0);
    expect(screen.getByText(/scikit-learn/i)).toBeTruthy();
    expect(
      screen.getByText(
        /Asset: 161 · Source: medical\/abstract-encoder · Revision: encoder-commit-42/i,
      ),
    ).toBeTruthy();
    expect(screen.getByText(/Abstract relevance/i)).toBeTruthy();
    expect(screen.getByText(/Relevance set/i)).toBeTruthy();
    expect(screen.getByText(/Source dataset v3 · ID 31/i)).toBeTruthy();
    expect(screen.getByText("Classical CPU")).toBeTruthy();
    expect(screen.getByText("Research artifacts")).toBeTruthy();
    expect(screen.getByText(/Macro F1 0.910/i)).toBeTruthy();
    expect(screen.getByText("Tess Trainer")).toBeTruthy();
    expect(screen.getByText(/Latest version: Vera Version/i)).toBeTruthy();
  });

  it("loads project-scoped immutable versions for model detail", async () => {
    render(
      <ModelsWorkspace
        workspaceId={5}
        projects={[project]}
        view="detail"
        modelId={81}
        onNavigate={vi.fn()}
      />,
    );

    expect(
      await screen.findByRole("heading", { name: "Relevance baseline" }),
    ).toBeTruthy();
    expect(screen.getByText("v2")).toBeTruthy();
    expect(screen.getByText("Classical CPU")).toBeTruthy();
    expect(screen.getByText("Research artifacts")).toBeTruthy();
    expect(screen.getByText("Vera Version")).toBeTruthy();
    expect(screen.getByText(/Package|Not recorded/)).toBeTruthy();
    expect(mocks.loadPlatformProject).toHaveBeenCalledWith(7, "models", 5);
  });

  it("shows execution, source dataset, and creator lineage on a version", async () => {
    render(
      <ModelsWorkspace
        workspaceId={5}
        projects={[project]}
        view="version"
        modelId={81}
        versionId={82}
        onNavigate={vi.fn()}
      />,
    );

    expect(
      await screen.findByRole("heading", { name: "Relevance baseline v2" }),
    ).toBeTruthy();
    expect(screen.getByText("Source dataset version")).toBeTruthy();
    expect(screen.getByText(/Source dataset v3 · ID 31/i)).toBeTruthy();
    expect(screen.getByText("Runtime")).toBeTruthy();
    expect(screen.getByText("Classical CPU")).toBeTruthy();
    expect(screen.getByText("Storage policy")).toBeTruthy();
    expect(screen.getByText("Research artifacts")).toBeTruthy();
    expect(screen.getByText("Creator")).toBeTruthy();
    expect(screen.getByText("Vera Version")).toBeTruthy();
  });
});
