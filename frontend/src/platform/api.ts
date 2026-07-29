import {
  listBaseModels,
  listWorkspaceMembers,
  request,
} from "@/api/client";
import type { Project } from "@/types/api";

import type { TrainingLaunchDraft } from "./TrainingScreen";
import type {
  AnnotationRound,
  AnnotationDecision,
  Dataset,
  DatasetItem,
  DatasetVersion,
  ExecutionEnvironment,
  FeedbackEvent,
  FeedbackReveal,
  FeedbackRun,
  FeedbackSetVersion,
  Guideline,
  GuidelineImpact,
  GuidelineProposal,
  GuidelineRevision,
  LabelSetVersion,
  LearningCycle,
  LearningFeedbackProducer,
  LearningFeedbackSource,
  ModelEvaluation,
  ModelVersion,
  PlatformProjectData,
  ProjectModule,
  ProjectModules,
  RegisteredModel,
  ReviewCase,
  RoundWorkContext,
  RoundWorkItemIdentity,
  RoundWorkRound,
  RoundSubmission,
  SplitMap,
  StoragePolicy,
  TaskDefinition,
  TaskKind,
  TaskVersion,
  TrainingDatasetVersion,
  TrainingRecipe,
  TrainingRecipeDescriptor,
  TrainingRecipeVersion,
  TrainingRun,
  WorkspaceModel,
  WorkspacePage,
  WorkspaceTrainingContext,
  WorkspaceTrainingFilters,
  WorkspaceTrainingRun,
  WorkspaceTrainingRunPage,
} from "./types";

export type PlatformLoadScope =
  | "overview"
  | "data"
  | "tasks"
  | "rounds"
  | "quality"
  | "activity"
  | "settings"
  | "train"
  | "models";

type PlatformCollection =
  | "baseModels"
  | "cycles"
  | "datasetVersions"
  | "datasets"
  | "feedbackEvents"
  | "feedbackRuns"
  | "feedbackSets"
  | "guidelineImpacts"
  | "guidelineProposals"
  | "guidelineRevisions"
  | "guidelines"
  | "labelSets"
  | "members"
  | "modelEvaluations"
  | "modelVersions"
  | "models"
  | "projectRecipes"
  | "recipeVersions"
  | "recipes"
  | "reviewCases"
  | "rounds"
  | "splitMaps"
  | "storagePolicies"
  | "taskVersions"
  | "tasks"
  | "trainingDatasets"
  | "trainingRuns"
  | "environments";

function collectionPlan(
  scope: PlatformLoadScope,
  effectiveModules: ProjectModule[],
): Set<PlatformCollection> {
  const plan = new Set<PlatformCollection>();
  const enabled = new Set(effectiveModules);
  const add = (...collections: PlatformCollection[]): void => {
    collections.forEach((collection) => plan.add(collection));
  };

  if (scope === "overview") {
    if (enabled.has("data")) {
      add("datasets", "datasetVersions", "labelSets", "splitMaps");
    }
    if (enabled.has("annotate")) add("rounds");
    if (enabled.has("models")) add("models", "modelVersions");
    if (enabled.has("train")) add("trainingRuns");
    return plan;
  }

  if (scope === "data" && enabled.has("data")) {
    add(
      "tasks",
      "taskVersions",
      "datasets",
      "datasetVersions",
      "labelSets",
      "splitMaps",
    );
    return plan;
  }

  if (scope === "tasks" && enabled.has("data")) {
    add("tasks", "taskVersions");
    return plan;
  }

  if (scope === "rounds" && enabled.has("annotate")) {
    add(
      "tasks",
      "taskVersions",
      "datasets",
      "datasetVersions",
      "rounds",
      "guidelines",
      "guidelineRevisions",
      "members",
    );
    return plan;
  }

  if (scope === "quality" && enabled.has("annotate")) {
    add("tasks", "taskVersions", "rounds");
    return plan;
  }

  if (scope === "train" && enabled.has("train")) {
    add(
      "tasks",
      "taskVersions",
      "trainingDatasets",
      "models",
      "trainingRuns",
      "recipes",
      "baseModels",
      "projectRecipes",
      "recipeVersions",
      "environments",
      "storagePolicies",
    );
    return plan;
  }

  if (scope === "models" && enabled.has("models")) {
    add(
      "tasks",
      "taskVersions",
      "trainingDatasets",
      "models",
      "modelVersions",
      "modelEvaluations",
    );
    return plan;
  }

  if (scope === "activity" && enabled.has("activity")) {
    add(
      "rounds",
      "trainingRuns",
      "modelEvaluations",
    );
    return plan;
  }

  if (scope === "settings") {
    if (enabled.has("data")) {
      add("datasets");
    }
    if (enabled.has("annotate")) {
      add("tasks", "taskVersions", "rounds");
    }
    if (enabled.has("train")) {
      add("environments", "storagePolicies");
    }
    return plan;
  }

  return plan;
}

function projectQuery(projectId: number): string {
  return `project_id=${encodeURIComponent(projectId)}`;
}

function post<T>(path: string, payload: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectModules(
  projectId: number,
  selected: ProjectModule[],
): Promise<ProjectModules> {
  return request<ProjectModules>(`/projects/${projectId}/modules`, {
    method: "PATCH",
    body: JSON.stringify({ selected }),
  });
}

export interface ExecutionEnvironmentDraft {
  name: string;
  environmentClass: string;
  imageDigest: string;
  packageManifest: Record<string, unknown>;
  hardwareConstraints: Record<string, unknown>;
}

export function createExecutionEnvironment(
  projectId: number,
  draft: ExecutionEnvironmentDraft,
): Promise<ExecutionEnvironment> {
  return post<ExecutionEnvironment>("/environments", {
    project_id: projectId,
    name: draft.name.trim(),
    environment_class: draft.environmentClass,
    image_digest: draft.imageDigest.trim(),
    package_manifest: draft.packageManifest,
    hardware_constraints: draft.hardwareConstraints,
  });
}

export function verifyExecutionEnvironment(
  projectId: number,
  environmentId: number,
  status: "available" | "unavailable",
  verificationReport: Record<string, unknown>,
): Promise<ExecutionEnvironment> {
  return post<ExecutionEnvironment>(
    `/environments/${environmentId}/verification?${projectQuery(projectId)}`,
    {
      status,
      verification_report: verificationReport,
    },
  );
}

export interface StoragePolicyDraft {
  name: string;
  backend: "minio" | "local";
  artifactPrefix: string;
  retentionClass: "indefinite" | "resume_14d";
  encryption: Record<string, unknown>;
  cachePolicy: Record<string, unknown>;
  isDefault: boolean;
}

export function createStoragePolicy(
  projectId: number,
  draft: StoragePolicyDraft,
): Promise<StoragePolicy> {
  return post<StoragePolicy>("/storage/policies", {
    project_id: projectId,
    name: draft.name.trim(),
    backend: draft.backend,
    artifact_prefix: draft.artifactPrefix.trim(),
    retention_class: draft.retentionClass,
    encryption: draft.encryption,
    cache_policy: draft.cachePolicy,
    is_default: draft.isDefault,
  });
}

async function loadChildCollections<T>(
  parents: Array<{ id: number }>,
  path: (id: number) => string,
): Promise<T[]> {
  const collections = await Promise.all(parents.map((parent) => request<T[]>(path(parent.id))));
  return collections.flat();
}

export async function loadPlatformProject(
  projectId: number,
  scope: PlatformLoadScope,
  workspaceId?: number,
): Promise<PlatformProjectData> {
  const query = projectQuery(projectId);
  const projectModules = await request<ProjectModules>(
    `/projects/${projectId}/modules`,
  );
  const plan = collectionPlan(scope, projectModules.effective);
  const loadWhen = <T>(
    collection: PlatformCollection,
    loader: () => Promise<T[]>,
  ): Promise<T[]> => plan.has(collection) ? loader() : Promise.resolve([]);
  const [
    taskDefinitions,
    datasets,
    cycles,
    rounds,
    feedbackRuns,
    feedbackSets,
    models,
    guidelines,
    guidelineProposals,
    guidelineImpacts,
    feedbackEvents,
    reviewCases,
    trainingRuns,
    recipes,
    baseModels,
    projectRecipes,
    environments,
    storagePolicies,
    trainingDatasets,
    splitMaps,
  ] = await Promise.all([
    loadWhen("tasks", () => request<TaskDefinition[]>(`/tasks?${query}`)),
    loadWhen("datasets", () => request<Dataset[]>(`/datasets?${query}`)),
    loadWhen("cycles", () =>
      request<LearningCycle[]>(`/cycles?${query}`),
    ),
    loadWhen("rounds", () =>
      request<AnnotationRound[]>(`/rounds?${query}`),
    ),
    loadWhen("feedbackRuns", () =>
      request<FeedbackRun[]>(`/feedback-runs?${query}`),
    ),
    loadWhen("feedbackSets", () =>
      request<FeedbackSetVersion[]>(`/feedback-runs/sets?${query}`),
    ),
    loadWhen("models", () =>
      request<RegisteredModel[]>(`/models?${query}`),
    ),
    loadWhen("guidelines", () =>
      request<Guideline[]>(`/workflow-guidelines?${query}`),
    ),
    loadWhen("guidelineProposals", () =>
      request<GuidelineProposal[]>(`/workflow-guidelines/proposals?${query}`),
    ),
    loadWhen("guidelineImpacts", () =>
      request<GuidelineImpact[]>(`/workflow-guidelines/impact-evaluations?${query}`),
    ),
    loadWhen("feedbackEvents", () =>
      request<FeedbackEvent[]>(`/feedback-runs/events?${query}`),
    ),
    loadWhen("reviewCases", () =>
      request<ReviewCase[]>(`/feedback-runs/review-cases?${query}`),
    ),
    loadWhen("trainingRuns", () =>
      request<TrainingRun[]>(`/training-runs?${query}`),
    ),
    loadWhen("recipes", () =>
      request<TrainingRecipeDescriptor[]>("/training/recipes"),
    ),
    loadWhen("baseModels", () =>
      listBaseModels(projectId, { readiness: "ready" }),
    ),
    loadWhen("projectRecipes", () =>
      request<TrainingRecipe[]>(`/training-recipes?${query}`),
    ),
    loadWhen("environments", () =>
      request<ExecutionEnvironment[]>(`/environments?${query}`),
    ),
    loadWhen("storagePolicies", () =>
      request<StoragePolicy[]>(`/storage/policies?${query}`),
    ),
    loadWhen("trainingDatasets", () =>
      request<TrainingDatasetVersion[]>(`/datasets/training-versions?${query}`),
    ),
    loadWhen("splitMaps", () =>
      request<SplitMap[]>(`/datasets/split-maps?${query}`),
    ),
  ]);

  const [taskVersions, datasetVersions, modelVersions, guidelineRevisions, recipeVersions] =
    await Promise.all([
      loadChildCollections<TaskVersion>(
        plan.has("taskVersions") ? taskDefinitions : [],
        (id) => `/tasks/versions?${query}&task_definition_id=${id}`,
      ),
      loadChildCollections<DatasetVersion>(
        plan.has("datasetVersions") ? datasets : [],
        (id) => `/datasets/versions?${query}&dataset_id=${id}`,
      ),
      loadChildCollections<ModelVersion>(
        plan.has("modelVersions") ? models : [],
        (id) => `/models/versions?${query}&registered_model_id=${id}`,
      ),
      loadChildCollections<GuidelineRevision>(
        plan.has("guidelineRevisions") ? guidelines : [],
        (id) => `/workflow-guidelines/revisions?${query}&guideline_id=${id}`,
      ),
      loadChildCollections<TrainingRecipeVersion>(
        plan.has("recipeVersions") ? projectRecipes : [],
        (id) => `/training-recipes/versions?${query}&training_recipe_id=${id}`,
      ),
    ]);

  const labelSets = await loadChildCollections<LabelSetVersion>(
    plan.has("labelSets") ? datasetVersions : [],
    (id) => `/datasets/label-sets?${query}&dataset_version_id=${id}`,
  );
  const modelEvaluations = plan.has("modelEvaluations")
    ? modelVersions.length
      ? await loadChildCollections<ModelEvaluation>(
          modelVersions,
          (id) => {
            const version = modelVersions.find((item) => item.id === id);
            return `/models/${version?.registered_model_id ?? 0}/versions/${id}/evaluations?${query}`;
          },
        )
      : await loadChildCollections<ModelEvaluation>(
          trainingRuns,
          (id) => `/training-runs/${id}/evaluations?${query}`,
        )
    : [];
  const workspaceMembers = workspaceId && plan.has("members")
    ? await listWorkspaceMembers(workspaceId)
    : [];

  return {
    workspaceMembers,
    projectModules,
    taskDefinitions,
    taskVersions,
    datasets,
    datasetVersions,
    labelSets,
    splitMaps,
    trainingDatasets,
    cycles,
    rounds,
    feedbackRuns,
    feedbackSets,
    models,
    modelVersions,
    modelEvaluations,
    guidelines,
    guidelineRevisions,
    guidelineProposals,
    guidelineImpacts,
    feedbackEvents,
    reviewCases,
    trainingRuns,
    recipes,
    baseModels,
    projectRecipes,
    recipeVersions,
    environments,
    storagePolicies,
  };
}

function workspaceFilterQuery(
  filters: WorkspaceTrainingFilters = {},
): string {
  const query = new URLSearchParams();
  if (filters.projectId !== undefined) {
    query.set("project_id", String(filters.projectId));
  }
  if (filters.status) query.set("status", filters.status);
  if (filters.creatorUserId !== undefined) {
    query.set("creator_user_id", String(filters.creatorUserId));
  }
  if (filters.taskVersionId !== undefined) {
    query.set("task_version_id", String(filters.taskVersionId));
  }
  if (filters.family) query.set("family", filters.family);
  if (filters.cursor !== undefined) query.set("cursor", String(filters.cursor));
  query.set("limit", String(filters.limit ?? 50));
  return query.toString();
}

function unsupportedWorkspaceAggregate(error: unknown): boolean {
  if (!error || typeof error !== "object" || !("status" in error)) {
    return false;
  }
  const status = Number((error as { status: unknown }).status);
  return status === 404 || status === 405 || status === 501;
}

async function workspaceProjects(
  workspaceId: number,
  supplied: Project[] | undefined,
): Promise<Project[]> {
  if (supplied) return supplied;
  return request<Project[]>(
    `/projects?workspace_id=${encodeURIComponent(workspaceId)}`,
  );
}

async function fallbackTrainingProjects(
  workspaceId: number,
  supplied: Project[] | undefined,
  scope: "train" | "models" = "train",
): Promise<Array<{ project: Project; data: PlatformProjectData }>> {
  const projects = await workspaceProjects(workspaceId, supplied);
  const results = await Promise.all(
    projects.map(async (project) => {
      const modules = await request<ProjectModules>(
        `/projects/${project.id}/modules`,
      );
      if (!modules.effective.includes(scope)) return null;
      return {
        project,
        data: await loadPlatformProject(project.id, scope, workspaceId),
      };
    }),
  );
  return results.filter(
    (
      result,
    ): result is { project: Project; data: PlatformProjectData } =>
      result !== null,
  );
}

function taskLabel(
  data: PlatformProjectData,
  taskVersionId: number,
): { name: string; kind: TaskKind } {
  const version = data.taskVersions.find((item) => item.id === taskVersionId);
  const definition = version
    ? data.taskDefinitions.find(
        (item) => item.id === version.task_definition_id,
      )
    : undefined;
  return {
    name:
      definition?.name ??
      version?.task_kind.replace(/_/g, " ") ??
      `Task version ${taskVersionId}`,
    kind: version?.task_kind ?? "classification",
  };
}

function fallbackEvaluation(
  data: PlatformProjectData,
  modelVersionId: number | null,
  trainingRunId?: number,
): ModelEvaluation | undefined {
  return [...data.modelEvaluations]
    .filter(
      (item) =>
        (modelVersionId !== null && item.model_version_id === modelVersionId) ||
        (trainingRunId !== undefined &&
          item.training_run_id === trainingRunId),
    )
    .sort((left, right) => right.id - left.id)[0];
}

export async function listWorkspaceTrainingContexts(
  workspaceId: number,
  projects?: Project[],
  filters: Pick<WorkspaceTrainingFilters, "projectId" | "cursor" | "limit"> = {},
): Promise<WorkspacePage<WorkspaceTrainingContext>> {
  try {
    return await request<WorkspacePage<WorkspaceTrainingContext>>(
      `/workspaces/${workspaceId}/training-contexts?${workspaceFilterQuery(filters)}`,
    );
  } catch (error) {
    if (!unsupportedWorkspaceAggregate(error)) throw error;
  }
  const contexts = await fallbackTrainingProjects(workspaceId, projects);
  return {
    items: contexts.map(({ project, data }) => ({
      project_id: project.id,
      project_name: project.name,
      project_description: project.description,
      training_only:
        data.projectModules.effective.includes("train") &&
        !data.projectModules.effective.includes("annotate"),
      effective_modules: data.projectModules.effective,
      task_version_count: data.taskVersions.length,
      training_dataset_version_count: data.trainingDatasets.length,
      environment_count: data.environments.length,
      available_environment_count: data.environments.filter(
        (item) => item.status === "available",
      ).length,
      storage_policy_count: data.storagePolicies.length,
    })),
    next_cursor: null,
  };
}

function fallbackTrainingRun(
  project: Project,
  data: PlatformProjectData,
  run: TrainingRun,
): WorkspaceTrainingRun {
  const model = data.models.find(
    (item) => item.id === run.registered_model_id,
  );
  const outputVersion = run.output_model_version_id
    ? data.modelVersions.find(
        (item) => item.id === run.output_model_version_id,
      )
    : undefined;
  const recipeVersion = data.recipeVersions.find(
    (item) => item.id === run.recipe_version_id,
  );
  const recipe = data.projectRecipes.find(
    (item) => item.id === recipeVersion?.training_recipe_id,
  );
  const descriptor = data.recipes.find(
    (item) => item.key === recipe?.key,
  );
  const trainingDataset = data.trainingDatasets.find(
    (item) => item.id === run.training_dataset_version_id,
  );
  const datasetVersion = trainingDataset
    ? data.datasetVersions.find(
        (item) => item.id === trainingDataset.dataset_version_id,
      )
    : undefined;
  const task = taskLabel(data, run.task_version_id);
  const evaluation = fallbackEvaluation(
    data,
    run.output_model_version_id,
    run.id,
  );
  return {
    ...run,
    project_name: project.name,
    project_description: project.description,
    model_name: model?.name ?? `Model ${run.registered_model_id}`,
    family: outputVersion?.family ?? descriptor?.model_family ?? "pending",
    framework:
      outputVersion?.framework ??
      descriptor?.trainer_key.replace(/_/g, " ") ??
      "Pending",
    base_model: outputVersion?.base_model ?? {},
    training_method:
      outputVersion?.training_method ?? descriptor?.parameterization ?? "pending",
    task_name: task.name,
    task_kind: task.kind,
    training_dataset_name:
      trainingDataset?.name ??
      `Training dataset ${run.training_dataset_version_id}`,
    dataset_version_number: datasetVersion?.version_number ?? null,
    recipe_name: descriptor?.label ?? recipe?.name ?? `Recipe ${run.recipe_version_id}`,
    runtime_name:
      data.environments.find((item) => item.id === run.environment_id)?.name ??
      `Environment ${run.environment_id}`,
    storage_policy_name:
      data.storagePolicies.find(
        (item) => item.id === run.storage_policy_id,
      )?.name ?? `Storage policy ${run.storage_policy_id}`,
    evaluation_status: evaluation?.status ?? null,
    evaluation_split: evaluation?.split_name ?? null,
    evaluation_metrics: evaluation?.metrics ?? {},
    creator_username: null,
    creator_display_name: null,
  };
}

export async function listWorkspaceTrainingRuns(
  workspaceId: number,
  filters: WorkspaceTrainingFilters = {},
  projects?: Project[],
): Promise<WorkspaceTrainingRunPage> {
  try {
    const page = await request<Partial<WorkspaceTrainingRunPage> & {
      items: WorkspaceTrainingRun[];
      next_cursor: number | null;
    }>(
      `/workspaces/${workspaceId}/training-runs?${workspaceFilterQuery(filters)}`,
    );
    return {
      items: page.items,
      next_cursor: page.next_cursor,
    };
  } catch (error) {
    if (!unsupportedWorkspaceAggregate(error)) throw error;
  }
  const projectData = await fallbackTrainingProjects(workspaceId, projects);
  const items = projectData
    .flatMap(({ project, data }) =>
      data.trainingRuns.map((run) =>
        fallbackTrainingRun(project, data, run),
      ),
    )
    .filter(
      (run) =>
        (filters.projectId === undefined ||
          run.project_id === filters.projectId) &&
        (!filters.status || run.status === filters.status) &&
        (filters.creatorUserId === undefined ||
          run.created_by_user_id === filters.creatorUserId) &&
        (filters.taskVersionId === undefined ||
          run.task_version_id === filters.taskVersionId) &&
        (!filters.family || run.family === filters.family),
    )
    .sort((left, right) => right.id - left.id);
  const limited = items.slice(0, filters.limit ?? 50);
  return {
    items: limited,
    next_cursor: null,
  };
}

function fallbackWorkspaceModel(
  project: Project,
  data: PlatformProjectData,
  model: RegisteredModel,
): WorkspaceModel {
  const latestVersion =
    data.modelVersions
      .filter((item) => item.registered_model_id === model.id)
      .sort(
        (left, right) => right.version_number - left.version_number,
      )[0] ?? null;
  const task = latestVersion
    ? taskLabel(data, latestVersion.task_version_id)
    : null;
  const trainingDataset = latestVersion?.training_dataset_version_id
    ? data.trainingDatasets.find(
        (item) => item.id === latestVersion.training_dataset_version_id,
      )
    : undefined;
  const evaluation = latestVersion
    ? fallbackEvaluation(data, latestVersion.id)
    : undefined;
  return {
    ...model,
    project_name: project.name,
    project_description: project.description,
    latest_version: latestVersion,
    task_name: task?.name ?? null,
    task_kind: task?.kind ?? null,
    training_dataset_name: trainingDataset?.name ?? null,
    evaluation_status: evaluation?.status ?? null,
    evaluation_split: evaluation?.split_name ?? null,
    evaluation_metrics: evaluation?.metrics ?? {},
    creator_username: null,
    creator_display_name: null,
  };
}

export async function listWorkspaceModels(
  workspaceId: number,
  filters: WorkspaceTrainingFilters = {},
  projects?: Project[],
): Promise<WorkspacePage<WorkspaceModel>> {
  try {
    return await request<WorkspacePage<WorkspaceModel>>(
      `/workspaces/${workspaceId}/models?${workspaceFilterQuery(filters)}`,
    );
  } catch (error) {
    if (!unsupportedWorkspaceAggregate(error)) throw error;
  }
  const projectData = await fallbackTrainingProjects(
    workspaceId,
    projects,
    "models",
  );
  const items = projectData
    .flatMap(({ project, data }) =>
      data.models.map((model) =>
        fallbackWorkspaceModel(project, data, model),
      ),
    )
    .filter(
      (model) =>
        (filters.projectId === undefined ||
          model.project_id === filters.projectId) &&
        (!filters.status || model.lifecycle_status === filters.status) &&
        (filters.creatorUserId === undefined ||
          model.created_by_user_id === filters.creatorUserId) &&
        (filters.taskVersionId === undefined ||
          model.latest_version?.task_version_id === filters.taskVersionId) &&
        (!filters.family ||
          model.latest_version?.family === filters.family),
    )
    .sort((left, right) => right.id - left.id);
  return {
    items: items.slice(0, filters.limit ?? 50),
    next_cursor: null,
  };
}

export async function findWorkspaceTrainingRun(
  workspaceId: number,
  runId: number,
  projects?: Project[],
): Promise<WorkspaceTrainingRun | null> {
  let cursor: number | undefined;
  do {
    const page = await listWorkspaceTrainingRuns(
      workspaceId,
      { cursor, limit: 100 },
      projects,
    );
    const match = page.items.find((item) => item.id === runId);
    if (match) return match;
    cursor = page.next_cursor ?? undefined;
  } while (cursor !== undefined);
  return null;
}

export async function findWorkspaceModel(
  workspaceId: number,
  modelId: number,
  projects?: Project[],
): Promise<WorkspaceModel | null> {
  let cursor: number | undefined;
  do {
    const page = await listWorkspaceModels(
      workspaceId,
      { cursor, limit: 100 },
      projects,
    );
    const match = page.items.find((item) => item.id === modelId);
    if (match) return match;
    cursor = page.next_cursor ?? undefined;
  } while (cursor !== undefined);
  return null;
}

export async function createTrainingOnlyProject(
  workspaceId: number,
  name: string,
  description: string,
): Promise<Project> {
  return post<Project>("/projects", {
    name: name.trim(),
    description: description.trim() || null,
    annotation_schema: { labels: {} },
    annotation_validation_mode: "strict",
    tasks: [],
    settings: {
      modules: ["data", "train", "models", "activity"],
      project_purpose: "training_only",
    },
    workspace_id: workspaceId,
  });
}

export function cancelTrainingRun(
  projectId: number,
  trainingRunId: number,
): Promise<TrainingRun> {
  return post<TrainingRun>(
    `/training-runs/${trainingRunId}/transition?${projectQuery(projectId)}`,
    { status: "cancelled" },
  );
}

export interface DatasetDraft {
  name: string;
  description: string;
  sourceType: Dataset["source_type"];
  sourceUri: string;
  sourceRevision: string;
  sourceFormat: "csv" | "jsonl" | "parquet" | "project_corpus" | "other";
  license: string;
  file: File | null;
  stableKeyField: string;
  groupKeyField: string;
  registryDatasetId: string;
  registryConfigName: string;
}

export async function createDatasetWithVersion(
  projectId: number,
  draft: DatasetDraft,
): Promise<DatasetVersion> {
  const dataset = await post<Dataset>("/datasets", {
    project_id: projectId,
    name: draft.name.trim(),
    description: draft.description.trim() || null,
    source_type: draft.sourceType,
  });
  if (draft.sourceType === "project_corpus") {
    return post<DatasetVersion>(
      `/projects/${projectId}/datasets/${dataset.id}/versions/project-corpus`,
      {},
    );
  }
  if (draft.sourceType === "upload") {
    if (!draft.file) throw new Error("Choose a CSV, JSONL, or Parquet file.");
    const form = new FormData();
    form.set("file", draft.file);
    form.set("source_format", draft.sourceFormat === "other" ? "auto" : draft.sourceFormat);
    if (draft.stableKeyField.trim()) form.set("stable_key_field", draft.stableKeyField.trim());
    if (draft.groupKeyField.trim()) form.set("group_key_field", draft.groupKeyField.trim());
    return request<DatasetVersion>(
      `/projects/${projectId}/datasets/${dataset.id}/versions/upload`,
      { method: "POST", body: form },
    );
  }
  if (draft.sourceType === "public_registry") {
    if (!draft.file) {
      throw new Error("Choose an exported snapshot of the pinned public dataset.");
    }
    if (draft.sourceFormat === "other") {
      throw new Error("Public dataset snapshots must be CSV, JSONL, or Parquet.");
    }
    const form = new FormData();
    form.set("file", draft.file);
    form.set("registry_dataset_id", draft.registryDatasetId.trim());
    form.set("exact_revision", draft.sourceRevision.trim());
    form.set("source_format", draft.sourceFormat);
    if (draft.registryConfigName.trim()) {
      form.set("config_name", draft.registryConfigName.trim());
    }
    if (draft.license.trim()) {
      form.set("license_identifier", draft.license.trim());
    }
    if (draft.stableKeyField.trim()) {
      form.set("stable_key_field", draft.stableKeyField.trim());
    }
    if (draft.groupKeyField.trim()) {
      form.set("group_key_field", draft.groupKeyField.trim());
    }
    return request<DatasetVersion>(
      `/projects/${projectId}/datasets/${dataset.id}/versions/public-registry-snapshot`,
      { method: "POST", body: form },
    );
  }
  return post<DatasetVersion>("/datasets/versions", {
    project_id: projectId,
    dataset_id: dataset.id,
    source_uri: draft.sourceUri.trim() || null,
    source_revision: draft.sourceRevision.trim(),
    source_format: draft.sourceFormat,
    data_schema: {},
    provenance: {
      import_method: draft.sourceType,
      pinned: Boolean(draft.sourceRevision.trim()),
    },
    license_info: draft.license.trim() ? { identifier: draft.license.trim() } : {},
    items: [],
  });
}

export interface TaskDraft {
  key: string;
  name: string;
  description: string;
  taskKind: TaskKind;
  labelValues: string[];
}

export async function createTaskWithVersion(
  projectId: number,
  draft: TaskDraft,
): Promise<TaskVersion> {
  const task = await post<TaskDefinition>("/tasks", {
    project_id: projectId,
    key: draft.key.trim(),
    name: draft.name.trim(),
    description: draft.description.trim() || null,
  });
  const payload = defaultTaskVersionPayload(projectId, task.id, draft.taskKind);
  payload.label_rules = {
    values: draft.labelValues,
    closed_set: draft.labelValues.length > 0,
  };
  if (draft.labelValues.length && draft.taskKind === "classification") {
    payload.output_schema = { type: "string", enum: draft.labelValues };
  } else if (
    draft.labelValues.length &&
    draft.taskKind === "multilabel_classification"
  ) {
    payload.output_schema = {
      type: "array",
      items: { type: "string", enum: draft.labelValues },
      uniqueItems: true,
    };
  }
  return post<TaskVersion>("/tasks/versions", payload);
}

export interface TrainingDataDraft {
  name: string;
  datasetVersionId: number;
  taskVersionId: number;
  labelSource: "existing_label_set" | "dataset_field";
  labelSetVersionId: number | null;
  labelField: string;
  inputField: string;
  trainPercent: number;
  validationPercent: number;
}

export async function createTrainingDataset(
  projectId: number,
  draft: TrainingDataDraft,
): Promise<TrainingDatasetVersion> {
  const response = await post<{
    training_dataset_version: TrainingDatasetVersion;
    label_set_version_id: number;
    split_map_id: number;
    split_counts: Record<string, number>;
    group_count: number;
  }>("/datasets/training-versions/compose", {
    project_id: projectId,
    name: draft.name.trim(),
    dataset_version_id: draft.datasetVersionId,
    task_version_id: draft.taskVersionId,
    input_field: draft.inputField.trim(),
    label_field:
      draft.labelSource === "dataset_field"
        ? draft.labelField.trim()
        : null,
    label_set_version_id:
      draft.labelSource === "existing_label_set"
        ? draft.labelSetVersionId
        : null,
    train_percent: draft.trainPercent,
    validation_percent: draft.validationPercent,
    seed: 42,
  });
  return response.training_dataset_version;
}

export function getAnnotationRound(roundId: number): Promise<AnnotationRound> {
  return request<AnnotationRound>(`/rounds/${roundId}`);
}

export function getRoundWorkContext(
  roundId: number,
  workspaceId?: number,
): Promise<RoundWorkContext> {
  const query =
    workspaceId === undefined ? "" : `?workspace_id=${workspaceId}`;
  return request<RoundWorkContext>(`/rounds/${roundId}/work-context${query}`);
}

export function listWorkspaceRoundWorkContexts(
  workspaceId: number,
): Promise<RoundWorkContext[]> {
  return request<RoundWorkContext[]>(
    `/workspaces/${workspaceId}/my-work/rounds`,
  );
}

export interface CycleDraft {
  name: string;
  goal: "expand_pool" | "reannotate" | "guideline_pilot" | "error_remediation";
  taskVersionId: number;
  datasetVersionId: number;
  baselineModelVersionId: number | null;
  feedbackSources: Array<
    Omit<
      LearningFeedbackSource,
      | "provider"
      | "external_model_id"
      | "exact_revision"
      | "configuration"
      | "data_egress_policy"
    > & {
      producer_type: LearningFeedbackProducer;
      provider?: string | null;
      external_model_id?: string | null;
      exact_revision?: string | null;
      configuration?: Record<string, unknown>;
      data_egress_policy?: Record<string, unknown>;
    }
  >;
}

export function createCycle(projectId: number, draft: CycleDraft): Promise<LearningCycle> {
  return post<LearningCycle>("/cycles", {
    project_id: projectId,
    name: draft.name.trim(),
    goal: draft.goal,
    task_version_id: draft.taskVersionId,
    source_dataset_version_id: draft.datasetVersionId,
    baseline_model_version_id: draft.baselineModelVersionId,
    feedback_sources: draft.feedbackSources,
    metadata: {},
  });
}

export interface RoundDraft {
  name: string;
  datasetVersionId: number;
  taskVersionId: number;
  cycleId: number | null;
  guidelineRevisionId: number | null;
  feedbackSetVersionId: number | null;
  assistancePolicy: AnnotationRound["assistance_policy"];
  reannotationMode: AnnotationRound["reannotation_mode"];
  selectionStrategy:
    | "all"
    | "random"
    | "uncertainty"
    | "diversity"
    | "disagreement"
    | "error_based"
    | "hybrid_uncertainty_diversity";
  selectionLimit: number;
  annotatorUserIds: number[];
  openToAllAnnotators: boolean;
  reason: string;
}

export async function createRound(
  projectId: number,
  draft: RoundDraft,
): Promise<AnnotationRound> {
  let selectionSetVersionId: number | null = null;
  if (draft.reannotationMode === "targeted_subset") {
    const splitMaps = await request<SplitMap[]>(
      `/datasets/split-maps?${projectQuery(projectId)}&dataset_version_id=${draft.datasetVersionId}`,
    );
    const splitMap = splitMaps[0];
    if (!splitMap) {
      throw new Error(
        "Create a governed train, validation, test, and pool split before selecting annotation items.",
      );
    }
    const selectionRun = await post<{ id: number }>("/selection-runs", {
      project_id: projectId,
      dataset_version_id: draft.datasetVersionId,
      task_version_id: draft.taskVersionId,
      cycle_id: draft.cycleId,
      feedback_set_version_id: draft.feedbackSetVersionId,
      split_map_id: splitMap.id,
      strategy: draft.selectionStrategy,
      parameters: { limit: draft.selectionLimit },
      eligibility_filter: {},
      seed: 42,
    });
    const selectionSet = await post<{ id: number }>(
      `/projects/${projectId}/selection-runs/${selectionRun.id}/materialize`,
      {},
    );
    selectionSetVersionId = selectionSet.id;
  }
  const annotationRound = await post<AnnotationRound>("/rounds", {
    project_id: projectId,
    name: draft.name.trim(),
    dataset_version_id: draft.datasetVersionId,
    task_version_id: draft.taskVersionId,
    cycle_id: draft.cycleId,
    guideline_revision_id: draft.guidelineRevisionId,
    selection_set_version_id: selectionSetVersionId,
    feedback_set_version_id: draft.feedbackSetVersionId,
    assistance_policy: draft.assistancePolicy,
    reannotation_mode: draft.reannotationMode,
    annotator_user_ids: draft.annotatorUserIds,
    open_to_all_annotators: draft.openToAllAnnotators,
    reason: draft.reason.trim() || null,
  });
  return post<AnnotationRound>(
    `/rounds/${annotationRound.id}/transition?${projectQuery(projectId)}`,
    { status: "open" },
  );
}

export interface RoundWorkData {
  roundItems: RoundWorkItemIdentity[];
  datasetItems: DatasetItem[];
  decisions: AnnotationDecision[];
  submissions: RoundSubmission[];
}

interface RoundWorkItem {
  round_item: RoundWorkItemIdentity;
  dataset_item: DatasetItem;
}

export async function loadRoundWork(
  projectId: number,
  round: RoundWorkRound,
): Promise<RoundWorkData> {
  const query = projectQuery(projectId);
  const [workItems, decisions, submissions] = await Promise.all([
    request<RoundWorkItem[]>(`/rounds/${round.id}/work-items?${query}`),
    request<AnnotationDecision[]>(`/rounds/${round.id}/decisions?${query}`),
    request<RoundSubmission[]>(`/rounds/${round.id}/submissions?${query}`),
  ]);
  return {
    roundItems: workItems.map((item) => item.round_item),
    datasetItems: workItems.map((item) => item.dataset_item),
    decisions,
    submissions,
  };
}

export function addRoundDecision(
  projectId: number,
  payload: {
    roundItemId: number;
    output: unknown;
    supersedesDecisionId: number | null;
    decisionKind: AnnotationDecision["decision_kind"];
    isInitialCheckpoint: boolean;
    rationale: string;
  },
): Promise<AnnotationDecision> {
  return post<AnnotationDecision>("/rounds/decisions", {
    project_id: projectId,
    round_item_id: payload.roundItemId,
    supersedes_decision_id: payload.supersedesDecisionId,
    output: payload.output,
    decision_kind: payload.decisionKind,
    is_initial_checkpoint: payload.isInitialCheckpoint,
    rationale: payload.rationale.trim() || null,
  });
}

export function submitRoundDecisions(
  projectId: number,
  roundId: number,
  decisionIds: number[],
): Promise<RoundSubmission> {
  return post<RoundSubmission>("/rounds/submissions", {
    project_id: projectId,
    annotation_round_id: roundId,
    decision_ids: decisionIds,
  });
}

export function revealRoundFeedback(
  projectId: number,
  roundId: number,
  roundItemId: number,
): Promise<FeedbackReveal> {
  return post<FeedbackReveal>(
    `/rounds/${roundId}/items/${roundItemId}/feedback-reveal`,
    {
      project_id: projectId,
      candidate_key: "primary",
      context: { surface: "project_annotation_workbench" },
    },
  );
}

export function recordFeedbackDecision(
  projectId: number,
  exposureId: number,
  annotationDecisionId: number,
  decision: "accepted" | "modified" | "rejected" | "ignored",
): Promise<void> {
  return post<void>("/feedback-runs/decisions", {
    project_id: projectId,
    feedback_exposure_id: exposureId,
    round_annotation_decision_id: annotationDecisionId,
    decision,
    details: {},
  });
}

export function transitionRound(
  projectId: number,
  roundId: number,
  status: "open" | "closed" | "cancelled",
): Promise<AnnotationRound> {
  return post<AnnotationRound>(
    `/rounds/${roundId}/transition?${projectQuery(projectId)}`,
    { status },
  );
}

export function publishRoundLabelSet(
  projectId: number,
  roundId: number,
  payload: {
    name: string;
    sourceKind: "human" | "adjudicated";
    submissionIds: number[];
  },
): Promise<LabelSetVersion> {
  return post<LabelSetVersion>(`/rounds/${roundId}/label-sets`, {
    project_id: projectId,
    name: payload.name,
    source_kind: payload.sourceKind,
    submission_ids: payload.submissionIds,
    parent_version_id: null,
    composition_policy: "replace",
  });
}

export interface GuidelineDraft {
  name: string;
  description: string;
  taskDefinitionId: number;
  taskVersionId: number;
  markdown: string;
  rationale: string;
}

export async function createGuidelineWithRevision(
  projectId: number,
  draft: GuidelineDraft,
): Promise<GuidelineRevision> {
  const guideline = await post<Guideline>("/workflow-guidelines", {
    project_id: projectId,
    task_definition_id: draft.taskDefinitionId,
    name: draft.name.trim(),
    description: draft.description.trim() || null,
  });
  return post<GuidelineRevision>("/workflow-guidelines/revisions", {
    project_id: projectId,
    guideline_id: guideline.id,
    task_version_id: draft.taskVersionId,
    markdown: draft.markdown,
    rationale: draft.rationale.trim() || null,
    diff_summary: {},
    source_proposal_ids: [],
  });
}

async function ensureRecipeVersion(
  projectId: number,
  descriptor: TrainingRecipeDescriptor,
): Promise<TrainingRecipeVersion> {
  return post<TrainingRecipeVersion>(
    `/training-recipes/trusted/${encodeURIComponent(descriptor.key)}?${projectQuery(projectId)}`,
    {},
  );
}

export async function launchTrainingRun(
  projectId: number,
  draft: TrainingLaunchDraft,
  data: PlatformProjectData,
): Promise<TrainingRun> {
  const descriptor = data.recipes.find((item) => item.key === draft.recipeKey);
  if (!descriptor) throw new Error("The selected training recipe is no longer available.");
  const validation = await post<{
    valid: boolean;
    normalized_config: Record<string, unknown> | null;
    errors: string[];
  }>(`/training/recipes/${encodeURIComponent(draft.recipeKey)}/validate-configuration`, {
    config: draft.config,
  });
  if (!validation.valid) {
    throw new Error(validation.errors.join("; ") || "The recipe configuration is invalid.");
  }

  const recipeVersion = await ensureRecipeVersion(projectId, descriptor);

  let model = data.models.find(
    (item) => item.name.localeCompare(draft.modelName, undefined, { sensitivity: "accent" }) === 0,
  );
  if (!model) {
    model = await post<RegisteredModel>("/models", {
      project_id: projectId,
      name: draft.modelName,
      description: null,
    });
  }

  return post<TrainingRun>("/training-runs", {
    project_id: projectId,
    registered_model_id: model.id,
    task_version_id: draft.taskVersionId,
    training_dataset_version_id: draft.trainingDatasetVersionId,
    recipe_version_id: recipeVersion.id,
    environment_id: draft.environmentId,
    storage_policy_id: draft.storagePolicyId,
    idempotency_key: crypto.randomUUID(),
    parent_model_version_id: null,
    evaluation_plan: draft.evaluationPlan,
    config: validation.normalized_config ?? draft.config,
    seed: draft.seed,
  });
}

export function defaultTaskVersionPayload(
  projectId: number,
  taskDefinitionId: number,
  taskKind: TaskKind,
): Record<string, unknown> {
  const contracts: Record<TaskKind, {
    output_schema: Record<string, unknown>;
    metrics: string[];
    trainers: string[];
  }> = {
    regression: {
      output_schema: { type: "number" },
      metrics: ["mae", "rmse", "r2"],
      trainers: ["tfidf_linear_regression"],
    },
    classification: {
      output_schema: { type: "string" },
      metrics: ["macro_f1", "precision", "recall", "accuracy"],
      trainers: ["tfidf_logistic_regression", "transformer_sequence_classification"],
    },
    multilabel_classification: {
      output_schema: { type: "array", items: { type: "string" }, uniqueItems: true },
      metrics: ["micro_f1", "macro_f1", "precision", "recall"],
      trainers: ["tfidf_logistic_regression", "transformer_sequence_classification"],
    },
    token_labeling: {
      output_schema: {
        type: "array",
        items: { type: "string" },
      },
      metrics: ["entity_f1", "token_f1"],
      trainers: ["transformer_token_classification"],
    },
    span_extraction: {
      output_schema: {
        type: "object",
        required: ["start", "end"],
        properties: {
          start: { type: "integer", minimum: 0 },
          end: { type: "integer", minimum: 1 },
          label: { type: "string" },
        },
      },
      metrics: ["span_f1", "exact_match"],
      trainers: ["transformer_span_extraction"],
    },
    relation_extraction: {
      output_schema: {
        type: "array",
        items: {
          type: "object",
          required: ["head", "tail", "type"],
          properties: {
            head: { type: "string" },
            tail: { type: "string" },
            type: { type: "string" },
          },
        },
      },
      metrics: ["relation_f1", "precision", "recall"],
      trainers: [],
    },
    ranking: {
      output_schema: {
        type: "array",
        items: { type: "string" },
      },
      metrics: ["ndcg", "mrr", "map"],
      trainers: [],
    },
    generation: {
      output_schema: {
        type: "object",
        required: ["completion"],
        properties: { completion: { type: "string" } },
      },
      metrics: ["rouge_l", "bleu", "bertscore"],
      trainers: [
        "causal_lm_sft_full",
        "causal_lm_sft_lora",
        "causal_lm_sft_qlora",
        "seq2seq_lm_sft_full",
        "seq2seq_lm_sft_lora",
        "seq2seq_lm_sft_qlora",
      ],
    },
    instruction_tuning: {
      output_schema: {
        type: "object",
        required: ["completion"],
        properties: { completion: { type: "string" } },
      },
      metrics: ["validation_loss", "perplexity"],
      trainers: [
        "causal_lm_sft_full",
        "causal_lm_sft_lora",
        "causal_lm_sft_qlora",
        "seq2seq_lm_sft_full",
        "seq2seq_lm_sft_lora",
        "seq2seq_lm_sft_qlora",
      ],
    },
  };
  const contract = contracts[taskKind];
  const inputSchema =
    taskKind === "token_labeling"
      ? {
          type: "object",
          required: ["tokens"],
          properties: {
            tokens: {
              type: "array",
              minItems: 1,
              items: { type: "string", minLength: 1 },
            },
          },
        }
      : taskKind === "generation" || taskKind === "instruction_tuning"
        ? {
            type: "object",
            required: ["prompt"],
            properties: { prompt: { type: "string", minLength: 1 } },
          }
        : {
            type: "object",
            required: ["text"],
            properties: { text: { type: "string", minLength: 1 } },
          };
  return {
    project_id: projectId,
    task_definition_id: taskDefinitionId,
    task_kind: taskKind,
    input_schema: inputSchema,
    output_schema: contract.output_schema,
    label_rules: {},
    annotation_ui: { preset: taskKind },
    metrics: contract.metrics,
    trainer_compatibility: contract.trainers,
  };
}
