// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";

import {
  createDatasetWithVersion,
  createExecutionEnvironment,
  createFeedbackRun,
  createRound,
  createStoragePolicy,
  createTaskWithVersion,
  createTrainingOnlyProject,
  createTrainingDataset,
  getRoundWorkContext,
  getFeedbackRun,
  launchTrainingRun,
  listWorkspaceRoundWorkContexts,
  listWorkspaceModels,
  listWorkspaceTrainingContexts,
  listWorkspaceTrainingRuns,
  loadPlatformProject,
  loadRoundWork,
  materializeFeedbackRun,
  verifyExecutionEnvironment,
} from "./api";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
  type RoundWorkRound,
  type TrainingRecipeDescriptor,
} from "./types";

const mocks = vi.hoisted(() => ({
  listBaseModels: vi.fn(),
  listWorkspaceMembers: vi.fn(),
  request: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  listBaseModels: mocks.listBaseModels,
  listWorkspaceMembers: mocks.listWorkspaceMembers,
  request: mocks.request,
}));

function bodyOf(call: unknown[]): Record<string, unknown> {
  const init = call[1] as RequestInit | undefined;
  return JSON.parse(String(init?.body)) as Record<string, unknown>;
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.listBaseModels.mockResolvedValue([]);
  mocks.listWorkspaceMembers.mockResolvedValue([]);
});

describe("platform API orchestration", () => {
  it("uses project-scoped administrator contracts for runtime and storage setup", async () => {
    mocks.request
      .mockResolvedValueOnce({ id: 131 })
      .mockResolvedValueOnce({ id: 131, status: "available" })
      .mockResolvedValueOnce({ id: 141 });

    await createExecutionEnvironment(7, {
      name: " Classical CPU ",
      environmentClass: "classical-cpu",
      imageDigest: ` sha256:${"a".repeat(64)} `,
      packageManifest: { "scikit-learn": "1.7.1" },
      hardwareConstraints: { device: "cpu" },
    });
    await verifyExecutionEnvironment(7, 131, "available", {
      image_digest_verified: true,
      device: "cpu",
    });
    await createStoragePolicy(7, {
      name: " Research artifacts ",
      backend: "minio",
      artifactPrefix: " workspaces/5/artifact-blobs ",
      retentionClass: "indefinite",
      encryption: { mode: "SSE-S3" },
      cachePolicy: { scope: "workspace" },
      isDefault: true,
    });

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/environments",
      "/environments/131/verification?project_id=7",
      "/storage/policies",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toEqual({
      project_id: 7,
      name: "Classical CPU",
      environment_class: "classical-cpu",
      image_digest: `sha256:${"a".repeat(64)}`,
      package_manifest: { "scikit-learn": "1.7.1" },
      hardware_constraints: { device: "cpu" },
    });
    expect(bodyOf(mocks.request.mock.calls[1])).toEqual({
      status: "available",
      verification_report: {
        image_digest_verified: true,
        device: "cpu",
      },
    });
    expect(bodyOf(mocks.request.mock.calls[2])).toEqual({
      project_id: 7,
      name: "Research artifacts",
      backend: "minio",
      artifact_prefix: "workspaces/5/artifact-blobs",
      retention_class: "indefinite",
      encryption: { mode: "SSE-S3" },
      cache_policy: { scope: "workspace" },
      is_default: true,
    });
  });

  it("uses workspace aggregation reads with explicit paging and filters", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (
        path ===
        "/workspaces/5/training-contexts?project_id=7&limit=100"
      ) {
        return { items: [{ project_id: 7 }], next_cursor: null };
      }
      if (
        path ===
        "/workspaces/5/training-runs?project_id=7&status=running&family=deep_learning&limit=50"
      ) {
        return {
          items: [{ id: 151 }],
          next_cursor: 140,
        };
      }
      if (
        path ===
        "/workspaces/5/models?project_id=7&family=deep_learning&cursor=140&limit=25"
      ) {
        return {
          items: [
            {
              id: 81,
              latest_version: {
                id: 82,
                family: "deep_learning",
                source_dataset_version_number: 3,
                runtime_name: "Transformer CPU",
                storage_policy_name: "Research artifacts",
                creator_display_name: "Vera Version",
              },
            },
          ],
          next_cursor: null,
        };
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    const contexts = await listWorkspaceTrainingContexts(
      5,
      undefined,
      { projectId: 7, limit: 100 },
    );
    const runs = await listWorkspaceTrainingRuns(5, {
      projectId: 7,
      status: "running",
      family: "deep_learning",
    });
    const models = await listWorkspaceModels(5, {
      projectId: 7,
      family: "deep_learning",
      cursor: 140,
      limit: 25,
    });

    expect(contexts.items).toEqual([{ project_id: 7 }]);
    expect(runs.next_cursor).toBe(140);
    expect(models.items[0]?.latest_version).toMatchObject({
      family: "deep_learning",
      source_dataset_version_number: 3,
      runtime_name: "Transformer CPU",
      storage_policy_name: "Research artifacts",
      creator_display_name: "Vera Version",
    });
  });

  it("pages canonical training history", async () => {
    mocks.request.mockResolvedValue({
      items: [],
      next_cursor: null,
    });

    await listWorkspaceTrainingRuns(5, {
      cursor: 140,
      limit: 25,
    });

    expect(mocks.request).toHaveBeenCalledWith(
      "/workspaces/5/training-runs?cursor=140&limit=25",
    );
  });

  it("falls back to authorized project reads only when aggregation is unavailable", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path.startsWith("/workspaces/5/training-contexts?")) {
        throw { status: 404 };
      }
      if (path === "/projects/7/modules") {
        return {
          project_id: 7,
          selected: ["data", "train", "models"],
          effective: ["data", "train", "models"],
          workspace_capabilities: ["training"],
        };
      }
      if (path.startsWith("/tasks?")) return [];
      if (path.startsWith("/datasets?")) return [];
      if (path.startsWith("/models?")) return [];
      if (path.startsWith("/training-runs?")) return [];
      if (path === "/training/recipes") return [];
      if (path.startsWith("/training-recipes?")) return [];
      if (path.startsWith("/environments?")) {
        return [{ id: 131, status: "available" }];
      }
      if (path.startsWith("/storage/policies?")) return [{ id: 141 }];
      if (path.startsWith("/datasets/training-versions?")) return [];
      throw new Error(`Unexpected request: ${path}`);
    });

    const page = await listWorkspaceTrainingContexts(
      5,
      [
        {
          id: 7,
          name: "Standalone baseline",
          description: "Public data",
          annotation_schema: { labels: {} },
          annotation_validation_mode: "strict",
          tasks: [],
          settings: {},
          workspace_id: 5,
        },
      ],
      { limit: 100 },
    );

    expect(page.items).toEqual([
      expect.objectContaining({
        project_id: 7,
        project_name: "Standalone baseline",
        training_only: true,
        available_environment_count: 1,
        storage_policy_count: 1,
      }),
    ]);
  });

  it("creates a training-only provenance project without Annotation", async () => {
    mocks.request.mockResolvedValue({ id: 44, name: "Public baseline" });

    await createTrainingOnlyProject(
      5,
      " Public baseline ",
      " Pinned registry dataset ",
    );

    expect(mocks.request).toHaveBeenCalledWith(
      "/projects",
      expect.objectContaining({ method: "POST" }),
    );
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      name: "Public baseline",
      description: "Pinned registry dataset",
      workspace_id: 5,
      tasks: [],
      settings: {
        modules: ["data", "train", "models", "activity"],
        project_purpose: "training_only",
      },
    });
    expect(
      (
        bodyOf(mocks.request.mock.calls[0]).settings as {
          modules: string[];
        }
      ).modules,
    ).not.toContain("annotate");
  });

  it("loads Data without training, model, round, or future-learning collections", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/projects/7/modules") {
        return {
          project_id: 7,
          selected: ["data", "annotate", "train", "models", "activity"],
          effective: ["data", "annotate", "train", "models", "activity"],
          workspace_capabilities: ["annotation", "training", "lineage"],
        };
      }
      if (path.startsWith("/tasks?")) return [{ id: 11 }];
      if (path.startsWith("/tasks/versions?")) return [{ id: 12 }];
      if (path.startsWith("/datasets?")) return [{ id: 21 }];
      if (path.startsWith("/datasets/versions?")) return [{ id: 22 }];
      if (path.startsWith("/datasets/label-sets?")) return [{ id: 31 }];
      if (path.startsWith("/datasets/split-maps?")) return [{ id: 41 }];
      throw new Error(`Unexpected request: ${path}`);
    });

    const data = await loadPlatformProject(7, "data");
    const paths = mocks.request.mock.calls.map(([path]) => String(path));

    expect(data.taskVersions).toEqual([{ id: 12 }]);
    expect(data.datasetVersions).toEqual([{ id: 22 }]);
    expect(data.labelSets).toEqual([{ id: 31 }]);
    expect(data.splitMaps).toEqual([{ id: 41 }]);
    expect(paths).not.toContain("/rounds?project_id=7");
    expect(paths).not.toContain("/training-runs?project_id=7");
    expect(paths).not.toContain("/models?project_id=7");
    expect(paths.some((path) => path.startsWith("/cycles?"))).toBe(false);
    expect(paths.some((path) => path.startsWith("/feedback-runs"))).toBe(false);
    expect(paths.some((path) => path.startsWith("/workflow-guidelines"))).toBe(false);
    expect(mocks.listBaseModels).not.toHaveBeenCalled();
  });

  it("loads Train dependencies without source corpora or future-learning collections", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/projects/7/modules") {
        return {
          project_id: 7,
          selected: ["data", "annotate", "train", "models", "activity"],
          effective: ["data", "annotate", "train", "models", "activity"],
          workspace_capabilities: ["annotation", "training", "lineage"],
        };
      }
      if (path.startsWith("/tasks?")) return [{ id: 11 }];
      if (path.startsWith("/tasks/versions?")) return [{ id: 12 }];
      if (path.startsWith("/datasets/training-versions?")) return [{ id: 51 }];
      if (path.startsWith("/models?")) return [{ id: 81 }];
      if (path.startsWith("/training-runs?")) return [{ id: 151 }];
      if (path === "/training/recipes") return [{ key: "tfidf_logreg" }];
      if (path.startsWith("/training-recipes/versions?")) return [{ id: 122 }];
      if (path.startsWith("/training-recipes?")) return [{ id: 121 }];
      if (path.startsWith("/environments?")) return [{ id: 131 }];
      if (path.startsWith("/storage/policies?")) return [{ id: 141 }];
      throw new Error(`Unexpected request: ${path}`);
    });

    const data = await loadPlatformProject(7, "train");
    const paths = mocks.request.mock.calls.map(([path]) => String(path));

    expect(data.trainingDatasets).toEqual([{ id: 51 }]);
    expect(data.trainingRuns).toEqual([{ id: 151 }]);
    expect(data.recipes).toEqual([{ key: "tfidf_logreg" }]);
    expect(data.recipeVersions).toEqual([{ id: 122 }]);
    expect(mocks.listBaseModels).toHaveBeenCalledWith(7, {
      readiness: "ready",
    });
    expect(paths).not.toContain("/datasets?project_id=7");
    expect(paths).not.toContain("/rounds?project_id=7");
    expect(paths.some((path) => path.startsWith("/cycles?"))).toBe(false);
    expect(paths.some((path) => path.startsWith("/feedback-runs"))).toBe(false);
    expect(paths.some((path) => path.startsWith("/workflow-guidelines"))).toBe(false);
  });

  it("loads only assignment-scoped dataset records through round work items", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/rounds/71/work-items?project_id=7") {
        return [
          {
            round_item: {
              id: 711,
              annotation_round_id: 71,
              dataset_item_id: 221,
            },
            dataset_item: {
              id: 221,
              stable_key: "selected-pool-item",
              payload: { text: "Assigned text" },
            },
          },
        ];
      }
      if (path === "/rounds/71/decisions?project_id=7") {
        return [{ id: 801, round_item_id: 711 }];
      }
      if (path === "/rounds/71/submissions?project_id=7") {
        return [{ id: 901, annotation_round_id: 71 }];
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    const work = await loadRoundWork(7, {
      id: 71,
      project_id: 7,
      dataset_version_id: 22,
    } as RoundWorkRound);
    const paths = mocks.request.mock.calls.map(([path]) => String(path));

    expect(work.roundItems).toEqual([
      expect.objectContaining({ id: 711, dataset_item_id: 221 }),
    ]);
    expect(work.datasetItems).toEqual([
      expect.objectContaining({
        id: 221,
        stable_key: "selected-pool-item",
        payload: { text: "Assigned text" },
      }),
    ]);
    expect(work.decisions).toEqual([{ id: 801, round_item_id: 711 }]);
    expect(work.submissions).toEqual([
      { id: 901, annotation_round_id: 71 },
    ]);
    expect(paths).toEqual([
      "/rounds/71/work-items?project_id=7",
      "/rounds/71/decisions?project_id=7",
      "/rounds/71/submissions?project_id=7",
    ]);
    expect(paths.some((path) => path.startsWith("/datasets/items?"))).toBe(
      false,
    );
    expect(paths).not.toContain("/rounds/71/items?project_id=7");
  });

  it("loads scoring resources on rounds only when learning and models are enabled", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/projects/7/modules") {
        return {
          project_id: 7,
          selected: ["data", "annotate", "learning", "models"],
          effective: ["data", "annotate", "learning", "models"],
          workspace_capabilities: ["annotation", "active_learning", "training"],
        };
      }
      return [];
    });

    await loadPlatformProject(7, "rounds", 5);
    const paths = mocks.request.mock.calls.map(([path]) => String(path));

    expect(paths).toContain("/cycles?project_id=7");
    expect(paths).toContain("/feedback-runs?project_id=7");
    expect(paths).toContain("/feedback-runs/sets?project_id=7");
    expect(paths).toContain("/datasets/split-maps?project_id=7");
    expect(paths).toContain("/models?project_id=7");
  });

  it("creates, materializes, and polls a project-scoped feedback run", async () => {
    mocks.request
      .mockResolvedValueOnce({ id: 91, status: "planned" })
      .mockResolvedValueOnce({ id: 91, status: "queued" })
      .mockResolvedValueOnce({ id: 91, status: "running" });
    const draft = {
      datasetVersionId: 22,
      taskVersionId: 12,
      cycleId: 5,
      modelVersionId: 81,
    };

    const created = await createFeedbackRun(7, draft);
    await materializeFeedbackRun(7, created.id);
    await getFeedbackRun(7, created.id);

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/feedback-runs",
      "/projects/7/feedback-runs/91/materialize",
      "/projects/7/feedback-runs/91",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      project_id: 7,
      dataset_version_id: 22,
      task_version_id: 12,
      cycle_id: 5,
      producer_type: "registered_model",
      model_version_id: 81,
    });
  });

  it("uses sanitized round context endpoints for My Work navigation", async () => {
    mocks.request
      .mockResolvedValueOnce({ round: { id: 71 } })
      .mockResolvedValueOnce([{ round: { id: 71 } }]);

    await getRoundWorkContext(71, 10);
    await listWorkspaceRoundWorkContexts(10);

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/rounds/71/work-context?workspace_id=10",
      "/workspaces/10/my-work/rounds",
    ]);
  });

  it("keeps Activity on released operational history without future datasets", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/projects/7/modules") {
        return {
          project_id: 7,
          selected: ["activity"],
          effective: ["activity"],
          workspace_capabilities: ["lineage"],
        };
      }
      if (path.startsWith("/rounds?")) return [{ id: 71 }];
      if (path.startsWith("/training-runs/151/evaluations?")) return [{ id: 152 }];
      if (path.startsWith("/training-runs?")) return [{ id: 151 }];
      throw new Error(`Unexpected request: ${path}`);
    });

    const data = await loadPlatformProject(7, "activity");
    const paths = mocks.request.mock.calls.map(([path]) => String(path));

    expect(data.rounds).toEqual([{ id: 71 }]);
    expect(data.trainingRuns).toEqual([{ id: 151 }]);
    expect(data.modelEvaluations).toEqual([{ id: 152 }]);
    expect(data.cycles).toEqual([]);
    expect(data.guidelineProposals).toEqual([]);
    expect(data.guidelineImpacts).toEqual([]);
    expect(data.feedbackEvents).toEqual([]);
    expect(data.reviewCases).toEqual([]);
    expect(paths).not.toContain("/feedback-runs?project_id=7");
    expect(paths).not.toContain("/feedback-runs/sets?project_id=7");
    expect(paths).not.toContain("/workflow-guidelines?project_id=7");
    expect(paths.some((path) => path.startsWith("/cycles?"))).toBe(false);
    expect(paths.some((path) => path.includes("/proposals?"))).toBe(false);
    expect(paths.some((path) => path.includes("/impact-evaluations?"))).toBe(false);
  });

  it.each([
    {
      scope: "overview" as const,
      expected: [
        "/datasets?project_id=7",
        "/datasets/split-maps?project_id=7",
        "/models?project_id=7",
        "/rounds?project_id=7",
        "/training-runs?project_id=7",
      ],
      loadsMembers: false,
    },
    {
      scope: "rounds" as const,
      expected: [
        "/datasets?project_id=7",
        "/workflow-guidelines?project_id=7",
        "/rounds?project_id=7",
        "/tasks?project_id=7",
      ],
      loadsMembers: true,
    },
    {
      scope: "models" as const,
      expected: [
        "/datasets/training-versions?project_id=7",
        "/models?project_id=7",
        "/tasks?project_id=7",
      ],
      loadsMembers: false,
    },
    {
      scope: "settings" as const,
      expected: [
        "/datasets?project_id=7",
        "/environments?project_id=7",
        "/rounds?project_id=7",
        "/storage/policies?project_id=7",
        "/tasks?project_id=7",
      ],
      loadsMembers: false,
    },
  ])(
    "uses the $scope top-level collection boundary",
    async ({ scope, expected, loadsMembers }) => {
      mocks.request.mockImplementation(async (path: string) => {
        if (path === "/projects/7/modules") {
          return {
            project_id: 7,
            selected: ["data", "annotate", "train", "models", "activity"],
            effective: ["data", "annotate", "train", "models", "activity"],
            workspace_capabilities: ["annotation", "training", "lineage"],
          };
        }
        return [];
      });

      await loadPlatformProject(7, scope, 5);
      const collectionPaths = mocks.request.mock.calls
        .map(([path]) => String(path))
        .filter((path) => path !== "/projects/7/modules")
        .sort();

      expect(collectionPaths).toEqual([...expected].sort());
      if (loadsMembers) {
        expect(mocks.listWorkspaceMembers).toHaveBeenCalledWith(5);
      } else {
        expect(mocks.listWorkspaceMembers).not.toHaveBeenCalled();
      }
      expect(
        collectionPaths.some(
          (path) =>
            path.startsWith("/cycles?") ||
            path.startsWith("/feedback-runs") ||
            path.includes("/proposals?") ||
            path.includes("/impact-evaluations?"),
        ),
      ).toBe(false);
    },
  );

  it("creates a task identity before its immutable task version", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/tasks") return { id: 11 };
      if (path === "/tasks/versions") return { id: 12 };
      throw new Error(`Unexpected request: ${path}`);
    });

    await createTaskWithVersion(7, {
      key: "screening",
      name: "Screening",
      description: "Abstract relevance",
      taskKind: "classification",
      labelValues: ["include", "exclude"],
    });

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/tasks",
      "/tasks/versions",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      project_id: 7,
      key: "screening",
      name: "Screening",
    });
    expect(bodyOf(mocks.request.mock.calls[1])).toMatchObject({
      project_id: 7,
      task_definition_id: 11,
      task_kind: "classification",
      label_rules: {
        values: ["include", "exclude"],
        closed_set: true,
      },
      output_schema: {
        type: "string",
        enum: ["include", "exclude"],
      },
    });
  });

  it("routes uploaded data through multipart ingestion", async () => {
    mocks.request.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === "/datasets") return { id: 21 };
      if (
        path === "/projects/7/datasets/21/versions/upload" &&
        init?.body instanceof FormData
      ) {
        return { id: 22 };
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    const file = new File(["id,patient,text\na,p1,Example\n"], "records.csv", {
      type: "text/csv",
    });

    await createDatasetWithVersion(7, {
      name: "Uploaded corpus",
      description: "",
      sourceType: "upload",
      sourceUri: "",
      sourceRevision: "",
      sourceFormat: "csv",
      license: "CC-BY-4.0",
      file,
      stableKeyField: "id",
      groupKeyField: "patient",
      registryDatasetId: "",
      registryConfigName: "",
    });

    const upload = mocks.request.mock.calls[1] as [string, RequestInit];
    expect(upload[0]).toBe("/projects/7/datasets/21/versions/upload");
    expect(upload[1].body).toBeInstanceOf(FormData);
    const form = upload[1].body as FormData;
    expect(form.get("file")).toBe(file);
    expect(form.get("source_format")).toBe("csv");
    expect(form.get("stable_key_field")).toBe("id");
    expect(form.get("group_key_field")).toBe("patient");
  });

  it("materializes an existing project corpus instead of creating an empty version", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/datasets") return { id: 21 };
      if (
        path ===
        "/projects/7/datasets/21/versions/project-corpus"
      ) {
        return { id: 22, item_count: 3 };
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    await createDatasetWithVersion(7, {
      name: "Current project corpus",
      description: "",
      sourceType: "project_corpus",
      sourceUri: "",
      sourceRevision: "",
      sourceFormat: "project_corpus",
      license: "",
      file: null,
      stableKeyField: "",
      groupKeyField: "",
      registryDatasetId: "",
      registryConfigName: "",
    });

    expect(mocks.request).toHaveBeenCalledTimes(2);
    expect(mocks.request.mock.calls[1][0]).toBe(
      "/projects/7/datasets/21/versions/project-corpus",
    );
    expect(bodyOf(mocks.request.mock.calls[1])).toEqual({});
  });

  it("uploads a pinned public-registry snapshot with its immutable revision", async () => {
    mocks.request.mockImplementation(async (path: string, init?: RequestInit) => {
      if (path === "/datasets") return { id: 21 };
      if (
        path ===
          "/projects/7/datasets/21/versions/public-registry-snapshot" &&
        init?.body instanceof FormData
      ) {
        return { id: 22 };
      }
      throw new Error(`Unexpected request: ${path}`);
    });
    const file = new File(["snapshot"], "benchmark.parquet", {
      type: "application/octet-stream",
    });

    await createDatasetWithVersion(7, {
      name: "Public benchmark",
      description: "",
      sourceType: "public_registry",
      sourceUri: "",
      sourceRevision: "a".repeat(40),
      sourceFormat: "parquet",
      license: "Apache-2.0",
      file,
      stableKeyField: "",
      groupKeyField: "",
      registryDatasetId: "owner/benchmark",
      registryConfigName: "default",
    });

    const upload = mocks.request.mock.calls[1] as [string, RequestInit];
    expect(upload[0]).toBe(
      "/projects/7/datasets/21/versions/public-registry-snapshot",
    );
    const form = upload[1].body as FormData;
    expect(form.get("file")).toBe(file);
    expect(form.get("registry_dataset_id")).toBe("owner/benchmark");
    expect(form.get("exact_revision")).toBe("a".repeat(40));
    expect(form.get("config_name")).toBe("default");
    expect(form.get("license_identifier")).toBe("Apache-2.0");
  });

  it("delegates label preservation and protected grouped splits to the server", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/datasets/training-versions/compose") {
        return {
          training_dataset_version: { id: 51 },
          label_set_version_id: 31,
          split_map_id: 41,
          split_counts: { train: 70, validation: 15, test: 15 },
          group_count: 100,
        };
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    await createTrainingDataset(7, {
      name: "Screening train",
      datasetVersionId: 22,
      taskVersionId: 12,
      labelSource: "dataset_field",
      labelSetVersionId: null,
      labelField: "label",
      inputField: "text",
      trainPercent: 70,
      validationPercent: 15,
    });

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/datasets/training-versions/compose",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      project_id: 7,
      name: "Screening train",
      dataset_version_id: 22,
      task_version_id: 12,
      input_field: "text",
      label_field: "label",
      label_set_version_id: null,
      train_percent: 70,
      validation_percent: 15,
      seed: 42,
    });
    expect(bodyOf(mocks.request.mock.calls[0])).not.toHaveProperty("items");
    expect(bodyOf(mocks.request.mock.calls[0])).not.toHaveProperty(
      "assignments",
    );
  });

  it("reuses an immutable label set without reposting annotation labels", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/datasets/training-versions/compose") {
        return {
          training_dataset_version: { id: 51 },
          label_set_version_id: 35,
          split_map_id: 41,
          split_counts: { train: 3, validation: 1, test: 1 },
          group_count: 5,
        };
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    await createTrainingDataset(7, {
      name: "Adjudicated screening train",
      datasetVersionId: 22,
      taskVersionId: 12,
      labelSource: "existing_label_set",
      labelSetVersionId: 35,
      labelField: "",
      inputField: "text",
      trainPercent: 70,
      validationPercent: 15,
    });

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/datasets/training-versions/compose",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      label_field: null,
      label_set_version_id: 35,
      input_field: "text",
    });
  });

  it("materializes targeted selection before creating an annotation round", async () => {
    mocks.request.mockImplementation(async (path: string) => {
      if (path === "/selection-runs") return { id: 61 };
      if (path === "/projects/7/selection-runs/61/materialize") return { id: 62 };
      if (path === "/rounds") return { id: 71 };
      if (path === "/rounds/71/transition?project_id=7") {
        return { id: 71, status: "open" };
      }
      throw new Error(`Unexpected request: ${path}`);
    });

    await createRound(7, {
      name: "False-negative review",
      datasetVersionId: 22,
      taskVersionId: 12,
      cycleId: 5,
      splitMapId: 41,
      guidelineRevisionId: 9,
      feedbackSetVersionId: 73,
      assistancePolicy: "reveal_after_first_pass",
      reannotationMode: "targeted_subset",
      selectionStrategy: "random",
      selectionLimit: 25,
      annotatorUserIds: [7],
      openToAllAnnotators: false,
      reason: "Review model misses",
    });

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/selection-runs",
      "/projects/7/selection-runs/61/materialize",
      "/rounds",
      "/rounds/71/transition?project_id=7",
    ]);
    expect(bodyOf(mocks.request.mock.calls[0])).toMatchObject({
      split_map_id: 41,
      feedback_set_version_id: 73,
      strategy: "random",
      parameters: { limit: 25 },
    });
    expect(bodyOf(mocks.request.mock.calls[2])).toMatchObject({
      selection_set_version_id: 62,
      feedback_set_version_id: 73,
      guideline_revision_id: 9,
      assistance_policy: "reveal_after_first_pass",
      annotator_user_ids: [7],
      open_to_all_annotators: false,
    });
    expect(bodyOf(mocks.request.mock.calls[3])).toEqual({ status: "open" });
  });

  it("validates configuration and pins recipe, model, runtime, and storage at launch", async () => {
    const descriptor: TrainingRecipeDescriptor = {
      schema_version: "training-recipe-descriptor-v1",
      key: "tfidf_logreg",
      version: "1.0.0",
      label: "TF-IDF logistic regression",
      description: "Linear baseline",
      model_family: "conventional_ml",
      architecture_family: "linear_classifier",
      parameterization: "full",
      supported_task_kinds: ["classification"],
      trainer_key: "sklearn_tfidf_logreg",
      implementation_status: "implemented",
      environment: {
        runtime_class: "classical-cpu",
        packages: ["scikit-learn"],
        devices: ["cpu"],
        minimum_memory_gb: 4,
        requires_verified_environment: true,
        setup_hint: "Use verified CPU.",
      },
      config_schema: {},
      artifact_formats: ["joblib"],
    };
    const data: PlatformProjectData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      recipes: [descriptor],
    };
    vi.spyOn(crypto, "randomUUID").mockReturnValue(
      "00000000-0000-4000-8000-000000000000",
    );
    mocks.request.mockImplementation(async (path: string) => {
      if (path.includes("/validate-configuration")) {
        return {
          valid: true,
          normalized_config: { max_features: 1000 },
          errors: [],
        };
      }
      if (path === "/training-recipes/trusted/tfidf_logreg?project_id=7") {
        return { id: 122 };
      }
      if (path === "/models") return { id: 81 };
      if (path === "/training-runs") return { id: 151 };
      throw new Error(`Unexpected request: ${path}`);
    });

    await launchTrainingRun(
      7,
      {
        modelName: "Relevance baseline",
        taskVersionId: 12,
        trainingDatasetVersionId: 51,
        recipeKey: "tfidf_logreg",
        recipeVersionId: null,
        environmentId: 131,
        storagePolicyId: 141,
        seed: 73,
        config: { max_features: 999 },
        evaluationPlan: { splits: ["test"] },
      },
      data,
    );

    expect(mocks.request.mock.calls.map(([path]) => path)).toEqual([
      "/training/recipes/tfidf_logreg/validate-configuration",
      "/training-recipes/trusted/tfidf_logreg?project_id=7",
      "/models",
      "/training-runs",
    ]);
    expect(bodyOf(mocks.request.mock.calls[3])).toMatchObject({
      project_id: 7,
      registered_model_id: 81,
      recipe_version_id: 122,
      environment_id: 131,
      storage_policy_id: 141,
      config: { max_features: 1000 },
      evaluation_plan: { splits: ["test"] },
      idempotency_key: "00000000-0000-4000-8000-000000000000",
    });
  });
});
