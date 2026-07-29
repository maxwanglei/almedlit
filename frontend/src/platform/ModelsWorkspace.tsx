import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import type { Project } from "@/types/api";

import {
  findWorkspaceModel,
  listWorkspaceModels,
  loadPlatformProject,
} from "./api";
import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformSection,
  PlatformStatus,
  formatStatus,
  shortHash,
} from "./components";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type ModelEvaluation,
  type ModelVersion,
  type PlatformProjectData,
  type WorkspaceModel,
  type WorkspacePage,
  type WorkspaceTrainingFilters,
} from "./types";

export type ModelsWorkspaceView = "registry" | "detail" | "version";

export interface ModelsWorkspaceProps {
  workspaceId: number;
  projects: Project[];
  view: ModelsWorkspaceView;
  modelId?: number | null;
  versionId?: number | null;
  initialProjectId?: number | null;
  onNavigate: (path: string, mode?: "push" | "replace") => void;
}

function errorMessage(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "Unable to load the model registry.";
}

function baseModelName(baseModel: Record<string, unknown>): string {
  for (const key of ["display_name", "source_model_id", "model_id", "name"]) {
    const value = baseModel[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return Object.keys(baseModel).length ? "Recorded in manifest" : "From scratch";
}

function baseModelLineage(baseModel: Record<string, unknown>): {
  asset: string;
  source: string;
  revision: string;
} {
  const first = (...keys: string[]): string | null => {
    for (const key of keys) {
      const value = baseModel[key];
      if (
        (typeof value === "string" || typeof value === "number") &&
        String(value).trim()
      ) {
        return String(value);
      }
    }
    return null;
  };
  const empty = Object.keys(baseModel).length === 0;
  return {
    asset:
      first("base_model_asset_id", "asset_id", "id") ??
      (empty ? "Not required" : "Not recorded"),
    source:
      first("source_model_id", "source", "model_id", "name") ??
      (empty ? "From scratch" : "Not recorded"),
    revision:
      first("exact_revision", "revision", "commit") ??
      (empty ? "Not applicable" : "Not recorded"),
  };
}

function metricSummary(metrics: Record<string, number | null>): string {
  const values = Object.entries(metrics)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .slice(0, 3)
    .map(([name, value]) => `${formatStatus(name)} ${value.toFixed(3)}`);
  return values.join(" · ") || "No aggregate metrics";
}

function creatorName(model: WorkspaceModel): string {
  return (
    model.creator_display_name?.trim() ||
    model.creator_username?.trim() ||
    (model.created_by_user_id ? `User ${model.created_by_user_id}` : "System")
  );
}

function versionCreatorName(version: ModelVersion): string {
  return (
    version.creator_display_name?.trim() ||
    version.creator_username?.trim() ||
    (version.created_by_user_id
      ? `User ${version.created_by_user_id}`
      : "System")
  );
}

function sourceDatasetVersion(version: ModelVersion): string {
  if (version.source_dataset_version_number !== null &&
      version.source_dataset_version_number !== undefined) {
    return version.source_dataset_version_id
      ? `Source dataset v${version.source_dataset_version_number} · ID ${version.source_dataset_version_id}`
      : `Source dataset v${version.source_dataset_version_number}`;
  }
  return version.source_dataset_version_id
    ? `Source dataset version ${version.source_dataset_version_id}`
    : "Source dataset version not recorded";
}

function versionEvaluation(
  data: PlatformProjectData,
  versionId: number,
): ModelEvaluation | undefined {
  return [...data.modelEvaluations]
    .filter((item) => item.model_version_id === versionId)
    .sort((left, right) => right.id - left.id)[0];
}

function taskName(
  data: PlatformProjectData,
  version: ModelVersion,
): string {
  const taskVersion = data.taskVersions.find(
    (item) => item.id === version.task_version_id,
  );
  const task = taskVersion
    ? data.taskDefinitions.find(
        (item) => item.id === taskVersion.task_definition_id,
      )
    : undefined;
  return (
    task?.name ??
    taskVersion?.task_kind.replace(/_/g, " ") ??
    `Task version ${version.task_version_id}`
  );
}

export default function ModelsWorkspace(
  props: ModelsWorkspaceProps,
): React.ReactElement {
  return (
    <main
      id="main-content"
      className="module-workspace-main"
      tabIndex={-1}
    >
      {props.view === "registry" ? (
        <ModelRegistry {...props} />
      ) : (
        <ModelDetail {...props} />
      )}
    </main>
  );
}

function ModelRegistry({
  workspaceId,
  projects,
  initialProjectId,
  onNavigate,
}: ModelsWorkspaceProps): React.ReactElement {
  const [filters, setFilters] = useState<WorkspaceTrainingFilters>({
    projectId: initialProjectId ?? undefined,
    limit: 50,
  });
  const [page, setPage] = useState<WorkspacePage<WorkspaceModel>>({
    items: [],
    next_cursor: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);

  const load = useCallback(
    async (
      nextFilters: WorkspaceTrainingFilters,
      append = false,
    ): Promise<void> => {
      const requestId = ++loadRequestIdRef.current;
      setLoading(true);
      setError(null);
      try {
        const result = await listWorkspaceModels(
          workspaceId,
          nextFilters,
          projects,
        );
        if (requestId !== loadRequestIdRef.current) return;
        setPage((current) => ({
          items: append ? [...current.items, ...result.items] : result.items,
          next_cursor: result.next_cursor,
        }));
      } catch (caught) {
        if (requestId === loadRequestIdRef.current) {
          setError(errorMessage(caught));
        }
      } finally {
        if (requestId === loadRequestIdRef.current) {
          setLoading(false);
        }
      }
    },
    [projects, workspaceId],
  );

  useEffect(() => {
    const initialFilters = {
      projectId: initialProjectId ?? undefined,
      limit: 50,
    };
    setFilters(initialFilters);
    setPage({ items: [], next_cursor: null });
    void load(initialFilters);
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [initialProjectId, load, workspaceId]);

  const familyOptions = [
    ...new Set(
      page.items
        .map((model) => model.latest_version?.family)
        .filter((family): family is string => Boolean(family)),
    ),
  ].sort();

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Models"
        description="Workspace-wide named model identities with immutable versions and evaluation lineage."
        actionLabel="Train model"
        onAction={() => onNavigate("/training/new")}
      />
      <PlatformSection title="Registry filters">
        <form
          className="platform-form-grid"
          onSubmit={(event) => {
            event.preventDefault();
            void load(filters);
          }}
        >
          <label>
            <span>Project</span>
            <select
              value={filters.projectId ?? ""}
              onChange={(event) =>
                setFilters((current) => ({
                  ...current,
                  projectId: event.target.value
                    ? Number(event.target.value)
                    : undefined,
                  cursor: undefined,
                }))
              }
            >
              <option value="">All model projects</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Lifecycle status</span>
            <select
              value={filters.status ?? ""}
              onChange={(event) =>
                setFilters((current) => ({
                  ...current,
                  status: event.target.value || undefined,
                  cursor: undefined,
                }))
              }
            >
              <option value="">All statuses</option>
              <option value="candidate">Candidate</option>
              <option value="active">Active</option>
              <option value="archived">Archived</option>
            </select>
          </label>
          <label>
            <span>Model family</span>
            <input
              type="text"
              list="model-registry-family-options"
              value={filters.family ?? ""}
              onChange={(event) =>
                setFilters((current) => ({
                  ...current,
                  family: event.target.value || undefined,
                  cursor: undefined,
                }))
              }
              placeholder="All families"
            />
            <datalist id="model-registry-family-options">
              {familyOptions.map((family) => (
                <option key={family} value={family}>
                  {formatStatus(family)}
                </option>
              ))}
            </datalist>
          </label>
          <div className="platform-wizard-actions">
            <Button
              type="submit"
              label={loading ? "Loading…" : "Apply filters"}
              isDisabled={loading}
            />
          </div>
        </form>
      </PlatformSection>
      {error ? <p className="platform-form-warning" role="alert">{error}</p> : null}
      <PlatformSection
        title="Model registry"
        description="Project remains the provenance owner while this registry spans the active workspace."
      >
        {loading && !page.items.length ? (
          <p className="platform-inline-empty" role="status">Loading models…</p>
        ) : page.items.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Workspace model registry"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Model and project</th>
                  <th scope="col">Latest version</th>
                  <th scope="col">Family and framework</th>
                  <th scope="col">Base model and method</th>
                  <th scope="col">Task and data</th>
                  <th scope="col">Runtime and storage</th>
                  <th scope="col">Evaluation</th>
                  <th scope="col">Creators</th>
                  <th scope="col">Status</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((model) => {
                  const version = model.latest_version;
                  const baseModel = baseModelLineage(
                    version?.base_model ?? {},
                  );
                  return (
                    <tr key={model.id}>
                      <td
                        data-label="Model and project"
                        data-priority="identity"
                      >
                        <PlatformRouteLink
                          href={`/models/${model.id}`}
                          className="platform-text-action"
                          onNavigate={onNavigate}
                        >
                          {model.name}
                        </PlatformRouteLink>
                        <span>{model.project_name}</span>
                      </td>
                      <td data-label="Latest version">
                        {version ? (
                          <>
                            v{version.version_number}
                            <span><code>{shortHash(version.content_hash)}</code></span>
                          </>
                        ) : (
                          "Not trained"
                        )}
                      </td>
                      <td data-label="Family and framework">
                        {version ? formatStatus(version.family) : "Pending"}
                        <span>{version?.framework ?? "Not recorded"}</span>
                      </td>
                      <td data-label="Base model and method">
                        {version ? baseModelName(version.base_model) : "Pending"}
                        <span>
                          {version
                            ? formatStatus(version.training_method)
                            : "Not trained"}
                        </span>
                        <span>
                          Asset: {baseModel.asset} · Source: {baseModel.source} ·
                          Revision: {baseModel.revision}
                        </span>
                      </td>
                      <td data-label="Task and data">
                        {model.task_name ?? "Not bound"}
                        <span>
                          {model.task_kind
                            ? `${formatStatus(model.task_kind)} · `
                            : ""}
                          {model.training_dataset_name ?? "No training dataset"}
                        </span>
                        {version ? (
                          <span>{sourceDatasetVersion(version)}</span>
                        ) : null}
                      </td>
                      <td data-label="Runtime and storage">
                        {version?.runtime_name ?? "Runtime not linked"}
                        <span>
                          {version?.storage_policy_name ??
                            "Storage policy not linked"}
                        </span>
                      </td>
                      <td data-label="Evaluation">
                        {model.evaluation_status ? (
                          <>
                            <PlatformStatus value={model.evaluation_status} />
                            <span>
                              {model.evaluation_split
                                ? `${model.evaluation_split} · `
                                : ""}
                              {metricSummary(model.evaluation_metrics)}
                            </span>
                          </>
                        ) : (
                          "Not evaluated"
                        )}
                      </td>
                      <td data-label="Creators">
                        {creatorName(model)}
                        <span>
                          {version
                            ? `Latest version: ${versionCreatorName(version)}`
                            : "No version creator"}
                        </span>
                      </td>
                      <td data-label="Status" data-priority="status">
                        <PlatformStatus value={model.lifecycle_status} />
                      </td>
                      <td data-label="Actions" data-priority="action">
                        <PlatformRouteLink
                          href={`/models/${model.id}`}
                          className="platform-text-action"
                          onNavigate={onNavigate}
                        >
                          View
                        </PlatformRouteLink>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No named models"
            detail="Launch a training run with a stable model name to create its first immutable version."
            actionLabel="Train first model"
            onAction={() => onNavigate("/training/new")}
          />
        )}
        {page.next_cursor !== null ? (
          <div className="platform-wizard-actions">
            <Button
              label={loading ? "Loading…" : "Load more"}
              isDisabled={loading}
              onClick={() =>
                void load(
                  { ...filters, cursor: page.next_cursor ?? undefined },
                  true,
                )
              }
            />
          </div>
        ) : null}
      </PlatformSection>
    </div>
  );
}

function ModelDetail({
  workspaceId,
  projects,
  view,
  modelId,
  versionId,
  onNavigate,
}: ModelsWorkspaceProps): React.ReactElement {
  const [model, setModel] = useState<WorkspaceModel | null>(null);
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setModel(null);
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoading(true);
    setError(null);
    if (!modelId) {
      setLoading(false);
      return () => {
        active = false;
      };
    }
    void findWorkspaceModel(workspaceId, modelId, projects)
      .then(async (result) => {
        if (!active || !result) return;
        setModel(result);
        const projectData = await loadPlatformProject(
          result.project_id,
          "models",
          workspaceId,
        );
        if (active) setData(projectData);
      })
      .catch((caught: unknown) => {
        if (active) setError(errorMessage(caught));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [modelId, projects, workspaceId]);

  if (loading) {
    return (
      <div className="platform-page">
        <PlatformPageHeader
          title="Model registry"
          description="Loading model lineage…"
        />
      </div>
    );
  }
  if (!model) {
    return (
      <div className="platform-page">
        <PlatformPageHeader
          title="Model not found"
          description="This model is unavailable in the active workspace."
          actionLabel="Back to models"
          onAction={() => onNavigate("/models", "replace")}
        />
        {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      </div>
    );
  }

  const versions = data.modelVersions
    .filter((item) => item.registered_model_id === model.id)
    .sort((left, right) => right.version_number - left.version_number);
  const selectedVersion =
    view === "version"
      ? versions.find((item) => item.id === versionId) ?? null
      : null;

  if (view === "version") {
    return (
      <ModelVersionDetail
        model={model}
        data={data}
        version={selectedVersion}
        onNavigate={onNavigate}
      />
    );
  }

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title={model.name}
        description={`${model.project_name} · named model identity`}
        actionLabel="Train next version"
        onAction={() =>
          onNavigate(`/training/new?projectId=${model.project_id}`)
        }
        secondary={
          <Button label="Back to models" onClick={() => onNavigate("/models")} />
        }
      />
      <PlatformSection title="Identity">
        <dl className="platform-review-grid">
          <div><dt>Project</dt><dd>{model.project_name}</dd></div>
          <div><dt>Status</dt><dd><PlatformStatus value={model.lifecycle_status} /></dd></div>
          <div><dt>Creator</dt><dd>{creatorName(model)}</dd></div>
          <div><dt>Versions</dt><dd>{versions.length}</dd></div>
          <div>
            <dt>Description</dt>
            <dd>{model.description ?? "No description"}</dd>
          </div>
        </dl>
      </PlatformSection>
      <PlatformSection
        title="Immutable versions"
        description="Each version pins task, training data, recipe, runtime, and checkpoint lineage."
      >
        {versions.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label={`${model.name} versions`}
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Version</th>
                  <th scope="col">Family and framework</th>
                  <th scope="col">Base model and method</th>
                  <th scope="col">Task</th>
                  <th scope="col">Training data</th>
                  <th scope="col">Runtime and storage</th>
                  <th scope="col">Creator</th>
                  <th scope="col">Evaluation</th>
                  <th scope="col">Checkpoint</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {versions.map((version) => {
                  const evaluation = versionEvaluation(data, version.id);
                  const trainingDataset = version.training_dataset_version_id
                    ? data.trainingDatasets.find(
                        (item) =>
                          item.id === version.training_dataset_version_id,
                      )
                    : undefined;
                  const baseModel = baseModelLineage(version.base_model);
                  return (
                    <tr key={version.id}>
                      <td data-label="Version" data-priority="identity">
                        <PlatformRouteLink
                          href={`/models/${model.id}/versions/${version.id}`}
                          className="platform-text-action"
                          onNavigate={onNavigate}
                        >
                          v{version.version_number}
                        </PlatformRouteLink>
                        <span><code>{shortHash(version.content_hash)}</code></span>
                      </td>
                      <td data-label="Family and framework">
                        {formatStatus(version.family)}
                        <span>{version.framework}</span>
                      </td>
                      <td data-label="Base model and method">
                        {baseModelName(version.base_model)}
                        <span>{formatStatus(version.training_method)}</span>
                        <span>
                          Asset: {baseModel.asset} · Source: {baseModel.source} ·
                          Revision: {baseModel.revision}
                        </span>
                      </td>
                      <td data-label="Task">{taskName(data, version)}</td>
                      <td data-label="Training data">
                        {trainingDataset?.name ?? "External"}
                        <span>{sourceDatasetVersion(version)}</span>
                      </td>
                      <td data-label="Runtime and storage">
                        {version.runtime_name ?? "Runtime not linked"}
                        <span>
                          {version.storage_policy_name ??
                            "Storage policy not linked"}
                        </span>
                      </td>
                      <td data-label="Creator">{versionCreatorName(version)}</td>
                      <td data-label="Evaluation">
                        {evaluation ? (
                          <>
                            <PlatformStatus value={evaluation.status} />
                            <span>{metricSummary(evaluation.metrics)}</span>
                          </>
                        ) : (
                          "Not evaluated"
                        )}
                      </td>
                      <td data-label="Checkpoint">
                        {version.checkpoint_package_id
                          ? `Package ${version.checkpoint_package_id}`
                          : "Not recorded"}
                      </td>
                      <td data-label="Actions" data-priority="action">
                        <PlatformRouteLink
                          href={`/models/${model.id}/versions/${version.id}`}
                          className="platform-text-action"
                          onNavigate={onNavigate}
                        >
                          View
                        </PlatformRouteLink>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No model versions"
            detail="A successful training run will create the first immutable version."
            actionLabel="Train model"
            onAction={() =>
              onNavigate(`/training/new?projectId=${model.project_id}`)
            }
          />
        )}
      </PlatformSection>
    </div>
  );
}

function ModelVersionDetail({
  model,
  data,
  version,
  onNavigate,
}: {
  model: WorkspaceModel;
  data: PlatformProjectData;
  version: ModelVersion | null;
  onNavigate: ModelsWorkspaceProps["onNavigate"];
}): React.ReactElement {
  if (!version) {
    return (
      <div className="platform-page">
        <PlatformPageHeader
          title="Model version not found"
          description={`No matching immutable version exists for ${model.name}.`}
          actionLabel="Back to model"
          onAction={() => onNavigate(`/models/${model.id}`, "replace")}
        />
      </div>
    );
  }
  const evaluation = versionEvaluation(data, version.id);
  const taskVersion = data.taskVersions.find(
    (item) => item.id === version.task_version_id,
  );
  const trainingDataset = version.training_dataset_version_id
    ? data.trainingDatasets.find(
        (item) => item.id === version.training_dataset_version_id,
      )
    : undefined;
  const baseModel = baseModelLineage(version.base_model);
  return (
    <div className="platform-page">
      <PlatformPageHeader
        title={`${model.name} v${version.version_number}`}
        description={`${model.project_name} · immutable model version`}
        actionLabel="Back to model"
        onAction={() => onNavigate(`/models/${model.id}`)}
      />
      <PlatformSection title="Version lineage">
        <dl className="platform-review-grid">
          <div><dt>Project</dt><dd>{model.project_name}</dd></div>
          <div><dt>Model</dt><dd>{model.name}</dd></div>
          <div><dt>Family</dt><dd>{formatStatus(version.family)}</dd></div>
          <div><dt>Framework</dt><dd>{version.framework}</dd></div>
          <div><dt>Base model</dt><dd>{baseModelName(version.base_model)}</dd></div>
          <div><dt>Base model asset</dt><dd>{baseModel.asset}</dd></div>
          <div><dt>Base model source</dt><dd>{baseModel.source}</dd></div>
          <div><dt>Base model revision</dt><dd>{baseModel.revision}</dd></div>
          <div><dt>Method</dt><dd>{formatStatus(version.training_method)}</dd></div>
          <div>
            <dt>Task</dt>
            <dd>
              {taskName(data, version)}
              {taskVersion ? ` · v${taskVersion.version_number}` : ""}
            </dd>
          </div>
          <div>
            <dt>Training dataset</dt>
            <dd>{trainingDataset?.name ?? "External"}</dd>
          </div>
          <div>
            <dt>Source dataset version</dt>
            <dd>{sourceDatasetVersion(version)}</dd>
          </div>
          <div>
            <dt>Runtime</dt>
            <dd>{version.runtime_name ?? "Not linked to a training run"}</dd>
          </div>
          <div>
            <dt>Storage policy</dt>
            <dd>
              {version.storage_policy_name ?? "Not linked to a training run"}
            </dd>
          </div>
          <div><dt>Creator</dt><dd>{versionCreatorName(version)}</dd></div>
          <div>
            <dt>Training run</dt>
            <dd>{version.training_run_id ?? "Not linked"}</dd>
          </div>
          <div><dt>Recipe</dt><dd>{version.recipe_key} · {version.recipe_version}</dd></div>
          <div><dt>Seed</dt><dd>{version.seed ?? "Not recorded"}</dd></div>
          <div>
            <dt>Checkpoint</dt>
            <dd>
              {version.checkpoint_package_id
                ? `Artifact package ${version.checkpoint_package_id}`
                : "Not recorded"}
            </dd>
          </div>
          <div>
            <dt>Parent version</dt>
            <dd>{version.parent_version_id ?? "None"}</dd>
          </div>
          <div><dt>Runtime digest</dt><dd><code>{version.runtime_digest}</code></dd></div>
          <div><dt>Code digest</dt><dd><code>{version.code_digest ?? "Not recorded"}</code></dd></div>
          <div><dt>Content hash</dt><dd><code>{version.content_hash}</code></dd></div>
          <div>
            <dt>Evaluation</dt>
            <dd>
              {evaluation ? (
                <>
                  <PlatformStatus value={evaluation.status} />
                  <span className="platform-review-detail">
                    {evaluation.split_name} · {metricSummary(evaluation.metrics)}
                  </span>
                </>
              ) : (
                "Not evaluated"
              )}
            </dd>
          </div>
        </dl>
      </PlatformSection>
      <PlatformSection title="Parameters">
        <pre className="platform-code-block">
          {JSON.stringify(version.parameters, null, 2)}
        </pre>
      </PlatformSection>
    </div>
  );
}
