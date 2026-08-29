import { useCallback, useEffect, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import type { Project } from "@/types/api";

import {
  cancelTrainingRun,
  createDatasetWithVersion,
  createExecutionEnvironment,
  createStoragePolicy,
  createTaskWithVersion,
  createTrainingDataset,
  createTrainingOnlyProject,
  findWorkspaceTrainingRun,
  launchTrainingRun,
  listWorkspaceTrainingContexts,
  listWorkspaceTrainingRuns,
  loadPlatformProject,
  verifyExecutionEnvironment,
} from "./api";
import DataScreen from "./DataScreen";
import PlatformDialog from "./PlatformDialog";
import TrainingScreen, { type TrainingLaunchDraft } from "./TrainingScreen";
import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformSection,
  PlatformStatus,
  formatStatus,
} from "./components";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type WorkspaceTrainingContext,
  type WorkspaceTrainingFilters,
  type WorkspaceTrainingRun,
  type WorkspaceTrainingRunPage,
} from "./types";

export type TrainingWorkspaceView =
  | "runs"
  | "new"
  | "data"
  | "runtimes"
  | "run-detail";

export interface TrainingWorkspaceProps {
  workspaceId: number;
  projects: Project[];
  view: TrainingWorkspaceView;
  runId?: number | null;
  initialProjectId?: number | null;
  initialDatasetId?: number | null;
  currentUserId: number;
  currentUserName: string;
  canCreateTrainingProject: boolean;
  canCreateTask: boolean;
  canProvisionRuntimes: boolean;
  onNavigate: (path: string, mode?: "push" | "replace") => void;
  onProjectCreated?: (project: Project) => void | Promise<void>;
}

type TrainingDialog = "dataset" | "task" | "trainingData" | null;

export function trainingDraftKey(
  workspaceId: number,
  projectId: number,
  datasetId?: number | null,
): string {
  return `${workspaceId}:${projectId}:${datasetId ?? "none"}`;
}

const NAV_ITEMS: Array<{
  view: Exclude<TrainingWorkspaceView, "run-detail">;
  path: string;
  label: string;
}> = [
  { view: "runs", path: "/training", label: "Runs" },
  { view: "new", path: "/training/new", label: "New training" },
  { view: "data", path: "/training/data", label: "Training data" },
  { view: "runtimes", path: "/training/runtimes", label: "Runtimes" },
];

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : fallback;
}

function parseJsonObject(value: string, label: string): Record<string, unknown> {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value);
  } catch {
    throw new Error(`${label} must be valid JSON.`);
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return parsed as Record<string, unknown>;
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
  const entries = Object.entries(metrics)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .slice(0, 3)
    .map(([name, value]) => `${formatStatus(name)} ${value.toFixed(3)}`);
  return entries.join(" · ") || "No aggregate metrics";
}

function creatorName(
  run: Pick<
    WorkspaceTrainingRun,
    "creator_display_name" | "creator_username" | "created_by_user_id"
  >,
): string {
  return (
    run.creator_display_name?.trim() ||
    run.creator_username?.trim() ||
    (run.created_by_user_id ? `User ${run.created_by_user_id}` : "System")
  );
}

function WorkspaceNavigation({
  current,
  onNavigate,
}: {
  current: TrainingWorkspaceView;
  onNavigate: TrainingWorkspaceProps["onNavigate"];
}): React.ReactElement {
  return (
    <nav className="platform-page-actions" aria-label="Training sections">
      {NAV_ITEMS.map((item) => (
        <PlatformRouteLink
          key={item.view}
          href={item.path}
          className="platform-text-action"
          ariaCurrent={
            current === item.view ||
            (current === "run-detail" && item.view === "runs")
              ? "page"
              : undefined
          }
          onNavigate={onNavigate}
        >
          {item.label}
        </PlatformRouteLink>
      ))}
    </nav>
  );
}

function ProjectContextPicker({
  contexts,
  projectId,
  label = "Project context",
  onChange,
}: {
  contexts: WorkspaceTrainingContext[];
  projectId: number;
  label?: string;
  onChange: (projectId: number) => void;
}): React.ReactElement {
  return (
    <label>
      <span>{label}</span>
      <select
        value={projectId}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        <option value={0}>Choose a project</option>
        {contexts.map((context) => (
          <option key={context.project_id} value={context.project_id}>
            {context.project_name}
            {context.effective_modules.includes("annotate")
              ? " · annotation enabled"
              : " · training-only"}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function TrainingWorkspace(
  props: TrainingWorkspaceProps,
): React.ReactElement {
  return (
    <main
      id="main-content"
      className="module-workspace-main"
      tabIndex={-1}
    >
      <div className="platform-page">
        <WorkspaceNavigation current={props.view} onNavigate={props.onNavigate} />
        {props.view === "runs" ? <TrainingRuns {...props} /> : null}
        {props.view === "new" ? <NewTraining {...props} /> : null}
        {props.view === "data" ? <TrainingData {...props} /> : null}
        {props.view === "runtimes" ? <TrainingRuntimes {...props} /> : null}
        {props.view === "run-detail" ? <TrainingRunDetail {...props} /> : null}
      </div>
    </main>
  );
}

function TrainingRuns({
  workspaceId,
  projects,
  initialProjectId,
  currentUserId,
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [filters, setFilters] = useState<WorkspaceTrainingFilters>({
    projectId: initialProjectId ?? undefined,
    limit: 50,
  });
  const [page, setPage] = useState<WorkspaceTrainingRunPage>({
    items: [],
    next_cursor: null,
  });
  const [loading, setLoading] = useState(true);
  const [busyRunId, setBusyRunId] = useState<number | null>(null);
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
        const result = await listWorkspaceTrainingRuns(
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
          setError(errorMessage(caught, "Unable to load training runs."));
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
    setFilters({
      projectId: initialProjectId ?? undefined,
      limit: 50,
    });
    setPage({
      items: [],
      next_cursor: null,
    });
    void load(
      { projectId: initialProjectId ?? undefined, limit: 50 },
      false,
    );
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [initialProjectId, load, workspaceId]);

  async function cancel(run: WorkspaceTrainingRun): Promise<void> {
    setBusyRunId(run.id);
    setError(null);
    try {
      const updated = await cancelTrainingRun(run.project_id, run.id);
      setPage((current) => ({
        ...current,
        items: current.items.map((item) =>
          item.id === run.id ? { ...item, ...updated } : item,
        ),
      }));
    } catch (caught) {
      setError(errorMessage(caught, "Unable to cancel this training run."));
    } finally {
      setBusyRunId(null);
    }
  }

  const historyRuns = [...page.items].sort((left, right) => {
    const leftTime = left.created_at ? Date.parse(left.created_at) : 0;
    const rightTime = right.created_at ? Date.parse(right.created_at) : 0;
    return rightTime - leftTime;
  });
  const familyOptions = [
    ...new Set(historyRuns.map((run) => run.family).filter(Boolean)),
  ].sort();

  return (
    <>
      <PlatformPageHeader
        title="Training"
        description="Workspace-wide runs across annotation and training-only projects."
        actionLabel="New training"
        onAction={() => onNavigate("/training/new")}
      />
      <PlatformSection title="Run filters">
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
              <option value="">All training projects</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Status</span>
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
              <option value="queued">Queued</option>
              <option value="running">Running</option>
              <option value="succeeded">Succeeded</option>
              <option value="failed">Failed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </label>
          <label>
            <span>Model family</span>
            <input
              type="text"
              list="training-run-family-options"
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
            <datalist id="training-run-family-options">
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
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      <PlatformSection
        title="Runs"
        description="Each run remains bound to its project provenance and immutable inputs."
      >
        {loading && !historyRuns.length ? (
          <p className="platform-inline-empty" role="status">Loading training runs…</p>
        ) : historyRuns.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Workspace training runs"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Run and project</th>
                  <th scope="col">Model</th>
                  <th scope="col">Task and data</th>
                  <th scope="col">Recipe</th>
                  <th scope="col">Runtime and storage</th>
                  <th scope="col">Evaluation</th>
                  <th scope="col">Creator</th>
                  <th scope="col">Status</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {historyRuns.map((run) => {
                  const cancellable =
                    run.created_by_user_id === currentUserId &&
                    ["queued", "running"].includes(run.status);
                  const baseModel = baseModelLineage(run.base_model);
                  return (
                    <tr key={run.id}>
                      <td data-label="Run and project" data-priority="identity">
                        <PlatformRouteLink
                          href={`/training/runs/${run.id}`}
                          className="platform-text-action"
                          onNavigate={onNavigate}
                        >
                          Run {run.id}
                        </PlatformRouteLink>
                        <span>{run.project_name}</span>
                      </td>
                      <td data-label="Model">
                        <strong>{run.model_name}</strong>
                        <span>
                          {formatStatus(run.family)} · {run.framework} ·{" "}
                          {formatStatus(run.training_method)}
                        </span>
                        <span>
                          {baseModelName(run.base_model)} · Asset:{" "}
                          {baseModel.asset} · Source: {baseModel.source} ·
                          Revision: {baseModel.revision}
                        </span>
                      </td>
                      <td data-label="Task and data">
                        {run.task_name}
                        <span>
                          {formatStatus(run.task_kind)} ·{" "}
                          {run.training_dataset_name}
                          {run.dataset_version_number
                            ? ` v${run.dataset_version_number}`
                            : ""}
                        </span>
                      </td>
                      <td data-label="Recipe">{run.recipe_name}</td>
                      <td data-label="Runtime and storage">
                        {run.runtime_name}
                        <span>{run.storage_policy_name}</span>
                      </td>
                      <td data-label="Evaluation">
                        {run.evaluation_status ? (
                          <>
                            <PlatformStatus value={run.evaluation_status} />
                            <span>
                              {run.evaluation_split
                                ? `${run.evaluation_split} · `
                                : ""}
                              {metricSummary(run.evaluation_metrics)}
                            </span>
                          </>
                        ) : (
                          "Pending"
                        )}
                      </td>
                      <td data-label="Creator">{creatorName(run)}</td>
                      <td data-label="Status" data-priority="status">
                        <PlatformStatus value={run.status} />
                        {run.failure_reason ? (
                          <span className="platform-run-failure">
                            {run.failure_reason}
                          </span>
                        ) : null}
                      </td>
                      <td data-label="Actions" data-priority="action">
                        {cancellable ? (
                          <Button
                            label={
                              busyRunId === run.id ? "Cancelling…" : "Cancel"
                            }
                            size="sm"
                            isDisabled={busyRunId !== null}
                            onClick={() => void cancel(run)}
                          />
                        ) : (
                          <PlatformRouteLink
                            href={`/training/runs/${run.id}`}
                            className="platform-text-action"
                            onNavigate={onNavigate}
                          >
                            View
                          </PlatformRouteLink>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No training runs"
            detail="Launch from an existing annotation project or a training-only project."
            actionLabel="New training"
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
                  {
                    ...filters,
                    cursor: page.next_cursor ?? undefined,
                  },
                  true,
                )
              }
            />
          </div>
        ) : null}
      </PlatformSection>
    </>
  );
}

function NewTraining({
  workspaceId,
  projects,
  initialProjectId,
  initialDatasetId,
  currentUserName,
  canCreateTrainingProject,
  onNavigate,
  onProjectCreated,
}: TrainingWorkspaceProps): React.ReactElement {
  const [contexts, setContexts] = useState<WorkspaceTrainingContext[]>([]);
  const [contextsWorkspaceId, setContextsWorkspaceId] = useState<number | null>(
    null,
  );
  const [contextKind, setContextKind] = useState<
    "existing" | "training_only"
  >("existing");
  const [projectId, setProjectId] = useState(initialProjectId ?? 0);
  const [contextConfirmed, setContextConfirmed] = useState(false);
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const contextsRequestIdRef = useRef(0);
  const projectDataRequestIdRef = useRef(0);

  const loadContexts = useCallback(async (): Promise<void> => {
    const requestId = ++contextsRequestIdRef.current;
    projectDataRequestIdRef.current += 1;
    setLoading(true);
    setError(null);
    try {
      const result = await listWorkspaceTrainingContexts(
        workspaceId,
        projects,
        { limit: 100 },
      );
      if (requestId !== contextsRequestIdRef.current) return;
      setContexts(result.items);
      setContextsWorkspaceId(workspaceId);
      if (initialProjectId) {
        const initial = result.items.find(
          (item) => item.project_id === initialProjectId,
        );
        if (initial) {
          setProjectId(initial.project_id);
          setContextKind(
            initial.effective_modules.includes("annotate")
              ? "existing"
              : "training_only",
          );
          setContextConfirmed(true);
        } else {
          setProjectId(0);
          setContextConfirmed(false);
        }
      }
    } catch (caught) {
      if (requestId === contextsRequestIdRef.current) {
        setError(errorMessage(caught, "Unable to load training contexts."));
      }
    } finally {
      if (requestId === contextsRequestIdRef.current) {
        setLoading(false);
      }
    }
  }, [initialProjectId, projects, workspaceId]);

  useEffect(() => {
    setContexts([]);
    setContextsWorkspaceId(null);
    setProjectId(initialProjectId ?? 0);
    setContextConfirmed(false);
    setContextKind("existing");
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setProjectName("");
    setProjectDescription("");
    setError(null);
    setBusy(false);
  }, [initialDatasetId, initialProjectId, workspaceId]);

  useEffect(() => {
    void loadContexts();
    return () => {
      contextsRequestIdRef.current += 1;
      projectDataRequestIdRef.current += 1;
    };
  }, [loadContexts]);

  useEffect(() => {
    if (
      !contextConfirmed ||
      !projectId ||
      contextsWorkspaceId !== workspaceId ||
      !contexts.some((item) => item.project_id === projectId)
    ) {
      setData(EMPTY_PLATFORM_PROJECT_DATA);
      return;
    }
    const requestId = ++projectDataRequestIdRef.current;
    setLoading(true);
    setError(null);
    const loaders = [
      loadPlatformProject(projectId, "train", workspaceId),
      loadPlatformProject(projectId, "data", workspaceId),
    ];
    void Promise.all(loaders)
      .then(([trainingData, sourceData]) => {
        if (requestId !== projectDataRequestIdRef.current) return;
        setData(
          sourceData
            ? {
                ...trainingData,
                datasets: sourceData.datasets,
                datasetVersions: sourceData.datasetVersions,
                labelSets: sourceData.labelSets,
                splitMaps: sourceData.splitMaps,
              }
            : trainingData,
        );
      })
      .catch((caught: unknown) => {
        if (requestId === projectDataRequestIdRef.current) {
          setError(
            errorMessage(caught, "Unable to load this training project."),
          );
        }
      })
      .finally(() => {
        if (requestId === projectDataRequestIdRef.current) {
          setLoading(false);
        }
      });
    return () => {
      if (requestId === projectDataRequestIdRef.current) {
        projectDataRequestIdRef.current += 1;
      }
    };
  }, [
    contextConfirmed,
    contexts,
    contextsWorkspaceId,
    initialDatasetId,
    projectId,
    workspaceId,
  ]);

  const context = contexts.find((item) => item.project_id === projectId);
  const contextProject = projects.find((item) => item.id === projectId);
  const matchingContexts = contexts.filter((item) =>
    contextKind === "training_only"
      ? !item.effective_modules.includes("annotate")
      : item.effective_modules.includes("annotate"),
  );

  async function createProject(): Promise<void> {
    if (!projectName.trim()) {
      setError("Enter a name for the training-only project.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const project = await createTrainingOnlyProject(
        workspaceId,
        projectName,
        projectDescription,
      );
      await onProjectCreated?.(project);
      setContexts((current) => [
        ...current.filter((item) => item.project_id !== project.id),
        {
          project_id: project.id,
          project_name: project.name,
          project_description: project.description,
          training_only: true,
          effective_modules: ["data", "train", "models", "activity"],
          task_version_count: 0,
          training_dataset_version_count: 0,
          environment_count: 0,
          available_environment_count: 0,
          storage_policy_count: 0,
        },
      ]);
      setProjectId(project.id);
      setContextKind("training_only");
      setContextConfirmed(true);
      setProjectName("");
      setProjectDescription("");
    } catch (caught) {
      setError(
        errorMessage(caught, "Unable to create the training-only project."),
      );
    } finally {
      setBusy(false);
    }
  }

  async function launch(draft: TrainingLaunchDraft): Promise<void> {
    if (!projectId) throw new Error("Choose a training context first.");
    setBusy(true);
    setError(null);
    try {
      const run = await launchTrainingRun(projectId, draft, data);
      onNavigate(`/training/runs/${run.id}`);
    } catch (caught) {
      const message = errorMessage(caught, "Unable to launch training.");
      setError(message);
      throw caught;
    } finally {
      setBusy(false);
    }
  }

  if (!contextConfirmed) {
    return (
      <>
        <PlatformPageHeader
          title="New training"
          description="Choose where task, data, model, and artifact lineage will be owned."
        />
        {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
        <PlatformSection
          title="Training context"
          description="Standalone work uses a lightweight training-only project; annotation projects remain fully supported."
        >
          <div
            className="platform-option-list"
            role="radiogroup"
            aria-label="Training context type"
          >
            <label>
              <input
                type="radio"
                name="training-context"
                checked={contextKind === "existing"}
                onChange={() => {
                  setContextKind("existing");
                  setProjectId(
                    contexts.find((item) => !item.training_only)?.project_id ??
                      0,
                  );
                }}
              />
              <span>
                <strong>Existing project</strong>
                <small>
                  Use an annotation-enabled project and its versioned task and
                  data.
                </small>
              </span>
            </label>
            <label>
              <input
                type="radio"
                name="training-context"
                checked={contextKind === "training_only"}
                onChange={() => {
                  setContextKind("training_only");
                  setProjectId(
                    contexts.find((item) => item.training_only)?.project_id ??
                      0,
                  );
                }}
              />
              <span>
                <strong>Training-only project</strong>
                <small>Use uploaded or pinned public data without Annotation.</small>
              </span>
            </label>
          </div>
          <div className="platform-form-grid">
            <ProjectContextPicker
              contexts={matchingContexts}
              projectId={projectId}
              label={
                contextKind === "training_only"
                  ? "Existing training-only project"
                  : "Existing project"
              }
              onChange={setProjectId}
            />
          </div>
          <div className="platform-wizard-actions">
            <Button
              label={loading ? "Loading…" : "Use selected project"}
              variant="primary"
              isDisabled={!projectId || loading}
              onClick={() => setContextConfirmed(true)}
            />
          </div>
        </PlatformSection>
        {contextKind === "training_only" && canCreateTrainingProject ? (
          <PlatformSection
          title="Create training-only project"
            description="It enables Data, Training, Models, and Activity without enabling Annotation."
          >
            <form
              className="platform-dialog-form"
              onSubmit={(event) => {
                event.preventDefault();
                void createProject();
              }}
            >
              <label>
                <span>Project name</span>
                <input
                  required
                  maxLength={255}
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                />
              </label>
              <label>
                <span>Description</span>
                <textarea
                  rows={3}
                  value={projectDescription}
                  onChange={(event) =>
                    setProjectDescription(event.target.value)
                  }
                />
              </label>
              <div className="platform-dialog-actions">
                <Button
                  type="submit"
                  label={busy ? "Creating…" : "Create project"}
                  variant="primary"
                  isDisabled={busy}
                />
              </div>
            </form>
          </PlatformSection>
        ) : null}
        {contextKind === "training_only" &&
        !canCreateTrainingProject &&
        !matchingContexts.length ? (
          <p className="platform-form-warning" role="status">
            Ask a workspace manager to create a training-only project.
          </p>
        ) : null}
      </>
    );
  }

  if (loading && !data.taskVersions.length) {
    return (
      <>
        <PlatformPageHeader
          title="New training"
          description={`Loading ${context?.project_name ?? "project"}…`}
        />
      </>
    );
  }

  return (
    <>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      <div className="platform-section-header">
        <p>
          Project context: <strong>{context?.project_name ?? `Project ${projectId}`}</strong>
        </p>
        <Button
          label="Change context"
          size="sm"
          onClick={() => setContextConfirmed(false)}
        />
      </div>
      <TrainingScreen
        key={trainingDraftKey(workspaceId, projectId, initialDatasetId)}
        data={data}
        busy={busy}
        contextLabel={
          context
            ? `${context.project_name}${
                context.effective_modules.includes("annotate")
                  ? " · annotation enabled"
                  : " · training-only"
              }`
            : `Project ${projectId}`
        }
        creatorLabel={currentUserName}
        title="New training"
        showRecentRuns={false}
        initialDatasetId={initialDatasetId}
        initialEnvironmentId={
          typeof contextProject?.settings.default_environment_id === "number"
            ? contextProject.settings.default_environment_id
            : null
        }
        initialStoragePolicyId={
          typeof contextProject?.settings.default_storage_policy_id ===
          "number"
            ? contextProject.settings.default_storage_policy_id
            : null
        }
        initialEvaluationSplit={
          typeof contextProject?.settings.default_evaluation_split === "string"
            ? contextProject.settings.default_evaluation_split
            : null
        }
        onOpenTrainingData={() =>
          onNavigate(
            `/training/data?projectId=${projectId}${
              initialDatasetId ? `&datasetId=${initialDatasetId}` : ""
            }`,
          )
        }
        onLaunch={launch}
      />
    </>
  );
}

function TrainingData({
  workspaceId,
  projects,
  initialProjectId,
  initialDatasetId,
  canCreateTask,
}: TrainingWorkspaceProps): React.ReactElement {
  const [contexts, setContexts] = useState<WorkspaceTrainingContext[]>([]);
  const [contextsWorkspaceId, setContextsWorkspaceId] = useState<number | null>(
    null,
  );
  const [projectId, setProjectId] = useState(0);
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState<TrainingDialog>(null);
  const [dialogDatasetId, setDialogDatasetId] = useState<number | null>(null);
  const [requestedDatasetHandled, setRequestedDatasetHandled] =
    useState(false);
  const reloadRequestIdRef = useRef(0);
  const reloadContextKey = `${workspaceId}:${contextsWorkspaceId ?? "none"}:${projectId}`;
  const reloadContextKeyRef = useRef(reloadContextKey);
  reloadContextKeyRef.current = reloadContextKey;

  const reload = useCallback(async (): Promise<void> => {
    if (reloadContextKeyRef.current !== reloadContextKey) return;
    const requestId = ++reloadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === reloadRequestIdRef.current &&
      reloadContextKey === reloadContextKeyRef.current;
    if (
      !projectId ||
      contextsWorkspaceId !== workspaceId ||
      !contexts.some((item) => item.project_id === projectId)
    ) {
      if (isCurrentRequest()) {
        setData(EMPTY_PLATFORM_PROJECT_DATA);
      }
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const [sourceData, trainingData] = await Promise.all([
        loadPlatformProject(projectId, "data", workspaceId),
        loadPlatformProject(projectId, "train", workspaceId),
      ]);
      if (isCurrentRequest()) {
        setData({
          ...sourceData,
          trainingDatasets: trainingData.trainingDatasets,
          models: trainingData.models,
          trainingRuns: trainingData.trainingRuns,
          recipes: trainingData.recipes,
          baseModels: trainingData.baseModels,
          projectRecipes: trainingData.projectRecipes,
          recipeVersions: trainingData.recipeVersions,
          environments: trainingData.environments,
          storagePolicies: trainingData.storagePolicies,
        });
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(errorMessage(caught, "Unable to load training data."));
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false);
      }
    }
  }, [
    contexts,
    contextsWorkspaceId,
    projectId,
    reloadContextKey,
    workspaceId,
  ]);

  useEffect(() => {
    let active = true;
    setContexts([]);
    setContextsWorkspaceId(null);
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setProjectId(0);
    setLoading(true);
    void listWorkspaceTrainingContexts(workspaceId, projects, { limit: 100 })
      .then((result) => {
        if (!active) return;
        setContexts(result.items);
        setContextsWorkspaceId(workspaceId);
        setProjectId(
          result.items.some((item) => item.project_id === initialProjectId)
            ? initialProjectId ?? 0
            : result.items[0]?.project_id ?? 0,
        );
      })
      .catch((caught: unknown) => {
        if (active) {
          setError(errorMessage(caught, "Unable to load training contexts."));
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [initialProjectId, projects, workspaceId]);

  useEffect(() => {
    setRequestedDatasetHandled(false);
  }, [initialDatasetId, projectId, workspaceId]);

  useEffect(() => {
    void reload();
    return () => {
      reloadRequestIdRef.current += 1;
    };
  }, [reload]);

  useEffect(() => {
    if (
      requestedDatasetHandled ||
      !initialDatasetId ||
      loading ||
      !data.datasets.some((dataset) => dataset.id === initialDatasetId)
    ) {
      return;
    }
    setDialogDatasetId(initialDatasetId);
    setDialog("trainingData");
    setRequestedDatasetHandled(true);
  }, [
    data.datasets,
    initialDatasetId,
    loading,
    requestedDatasetHandled,
  ]);

  const context = contexts.find((item) => item.project_id === projectId);

  async function mutate(operation: () => Promise<unknown>): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await operation();
      await reload();
    } catch (caught) {
      setError(errorMessage(caught, "Unable to update training data."));
      throw caught;
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <PlatformSection title="Project context">
        <div className="platform-form-grid">
          <ProjectContextPicker
            contexts={contexts}
            projectId={projectId}
            onChange={setProjectId}
          />
        </div>
      </PlatformSection>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      {loading && !projectId ? (
        <p className="platform-inline-empty" role="status">Loading training contexts…</p>
      ) : projectId ? (
        <DataScreen
          data={data}
          legacyDocumentCount={0}
          title="Training data"
          description={`Versioned source data and label composition for ${
            context?.project_name ?? `Project ${projectId}`
          }.`}
          secondary={
            canCreateTask ? (
              <Button
                label="Define task"
                onClick={() => setDialog("task")}
              />
            ) : undefined
          }
          onCreate={() => setDialog("dataset")}
          onPrepareTraining={(dataset) => {
            setDialogDatasetId(dataset.id);
            setDialog("trainingData");
          }}
        />
      ) : (
        <PlatformEmpty
          title="No training project"
          detail="A manager must create a project with Training enabled before data can be imported."
        />
      )}
      {projectId ? (
        <PlatformSection
          title="Composed training datasets"
          description="Each entry pins source data, labels, preprocessing, and a protected split map."
        >
          {data.trainingDatasets.length ? (
            <div
              className="platform-table-scroll platform-table-scroll--summary"
              role="region"
              aria-label="Composed training datasets"
              tabIndex={0}
            >
              <table className="platform-table platform-table--summary">
                <thead>
                  <tr>
                    <th scope="col">Training dataset</th>
                    <th scope="col">Project</th>
                    <th scope="col">Task</th>
                    <th scope="col">Source version</th>
                    <th scope="col">Label layers</th>
                    <th scope="col">Split map</th>
                    <th scope="col">Fingerprint</th>
                  </tr>
                </thead>
                <tbody>
                  {data.trainingDatasets.map((dataset) => {
                    const taskVersion = data.taskVersions.find(
                      (item) => item.id === dataset.task_version_id,
                    );
                    const task = taskVersion
                      ? data.taskDefinitions.find(
                          (item) =>
                            item.id === taskVersion.task_definition_id,
                        )
                      : undefined;
                    const sourceVersion = data.datasetVersions.find(
                      (item) => item.id === dataset.dataset_version_id,
                    );
                    const split = data.splitMaps.find(
                      (item) => item.id === dataset.split_map_id,
                    );
                    return (
                      <tr key={dataset.id}>
                        <td
                          data-label="Training dataset"
                          data-priority="identity"
                        >
                          <strong>{dataset.name}</strong>
                        </td>
                        <td data-label="Project">{context?.project_name}</td>
                        <td data-label="Task">
                          {task?.name ??
                            taskVersion?.task_kind.replace(/_/g, " ") ??
                            `Task version ${dataset.task_version_id}`}
                          {taskVersion ? (
                            <span>v{taskVersion.version_number}</span>
                          ) : null}
                        </td>
                        <td data-label="Source version">
                          {sourceVersion
                            ? `v${sourceVersion.version_number}`
                            : `ID ${dataset.dataset_version_id}`}
                        </td>
                        <td data-label="Label layers">
                          {dataset.label_set_version_ids.length}
                        </td>
                        <td data-label="Split map">
                          {split?.name ?? `Split ${dataset.split_map_id}`}
                        </td>
                        <td data-label="Fingerprint">
                          <code>{dataset.content_hash.slice(0, 10)}</code>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="platform-inline-empty">
              No composed training dataset yet. Prepare one from a source
              dataset after defining its task and label source.
            </p>
          )}
        </PlatformSection>
      ) : null}
      {dialog && projectId ? (
        <PlatformDialog
          kind={dialog}
          data={data}
          busy={busy}
          initialDatasetId={dialogDatasetId}
          onClose={() => {
            setDialog(null);
            setDialogDatasetId(null);
          }}
          onCreateDataset={(draft) =>
            mutate(() => createDatasetWithVersion(projectId, draft))
          }
          onCreateTask={(draft) =>
            mutate(() => createTaskWithVersion(projectId, draft))
          }
          onPrepareTrainingData={(draft) =>
            mutate(() => createTrainingDataset(projectId, draft))
          }
          onCreateCycle={async () => {
            throw new Error("Learning cycles are planned for a later release.");
          }}
          onCreateRound={async () => {
            throw new Error("Rounds are managed from Projects.");
          }}
          onScoreFeedback={async () => {
            throw new Error("Model scoring is managed from Projects.");
          }}
          onCreateGuideline={async () => {
            throw new Error("Guideline Learning is planned for a later release.");
          }}
        />
      ) : null}
    </>
  );
}

function TrainingRuntimes({
  workspaceId,
  projects,
  initialProjectId,
  canProvisionRuntimes,
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [contexts, setContexts] = useState<WorkspaceTrainingContext[]>([]);
  const [contextsWorkspaceId, setContextsWorkspaceId] = useState<number | null>(
    null,
  );
  const [projectId, setProjectId] = useState(0);
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(true);
  const [provisioning, setProvisioning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [runtimeName, setRuntimeName] = useState("");
  const [runtimeClass, setRuntimeClass] = useState("classical-cpu");
  const [runtimeDigest, setRuntimeDigest] = useState("");
  const [runtimePackages, setRuntimePackages] = useState("{}");
  const [runtimeHardware, setRuntimeHardware] = useState("{}");
  const [verificationEnvironmentId, setVerificationEnvironmentId] = useState(0);
  const [verificationStatus, setVerificationStatus] =
    useState<"available" | "unavailable">("available");
  const [verificationReport, setVerificationReport] = useState("{}");
  const [storageName, setStorageName] = useState("");
  const [storageBackend, setStorageBackend] =
    useState<"minio" | "local">("minio");
  const [artifactPrefix, setArtifactPrefix] = useState("");
  const [retentionClass, setRetentionClass] =
    useState<"indefinite" | "resume_14d">("indefinite");
  const [storageDefault, setStorageDefault] = useState(true);
  const [storageEncryption, setStorageEncryption] = useState("{}");
  const [storageCache, setStorageCache] = useState("{}");
  const operationContextRef = useRef({ workspaceId, projectId });
  operationContextRef.current = { workspaceId, projectId };

  useEffect(() => {
    let active = true;
    setContexts([]);
    setContextsWorkspaceId(null);
    setProjectId(0);
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoading(true);
    void listWorkspaceTrainingContexts(workspaceId, projects, { limit: 100 })
      .then((result) => {
        if (!active) return;
        setContexts(result.items);
        setContextsWorkspaceId(workspaceId);
        setProjectId(
          result.items.some((item) => item.project_id === initialProjectId)
            ? initialProjectId ?? 0
            : result.items[0]?.project_id ?? 0,
        );
      })
      .catch((caught: unknown) => {
        if (active) {
          setError(errorMessage(caught, "Unable to load runtime contexts."));
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [initialProjectId, projects, workspaceId]);

  useEffect(() => {
    if (
      !projectId ||
      contextsWorkspaceId !== workspaceId ||
      !contexts.some((item) => item.project_id === projectId)
    ) {
      setData(EMPTY_PLATFORM_PROJECT_DATA);
      return;
    }
    let active = true;
    setLoading(true);
    setError(null);
    void loadPlatformProject(projectId, "train", workspaceId)
      .then((result) => {
        if (active) setData(result);
      })
      .catch((caught: unknown) => {
        if (active) {
          setError(errorMessage(caught, "Unable to load runtimes."));
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [contexts, contextsWorkspaceId, projectId, workspaceId]);

  useEffect(() => {
    setStatusMessage(null);
    setError(null);
    setProvisioning(false);
    setVerificationEnvironmentId(0);
    setArtifactPrefix(
      projectId ? `workspaces/${workspaceId}/artifact-blobs` : "",
    );
  }, [projectId, workspaceId]);

  async function provision(
    action: () => Promise<void>,
    successMessage: string,
    expectedContext: { workspaceId: number; projectId: number },
  ): Promise<void> {
    setProvisioning(true);
    setError(null);
    setStatusMessage(null);
    try {
      await action();
      if (
        operationContextRef.current.workspaceId ===
          expectedContext.workspaceId &&
        operationContextRef.current.projectId === expectedContext.projectId
      ) {
        setStatusMessage(successMessage);
      }
    } catch (caught) {
      if (
        operationContextRef.current.workspaceId ===
          expectedContext.workspaceId &&
        operationContextRef.current.projectId === expectedContext.projectId
      ) {
        setError(errorMessage(caught, "Provisioning could not be completed."));
      }
    } finally {
      if (
        operationContextRef.current.workspaceId ===
          expectedContext.workspaceId &&
        operationContextRef.current.projectId === expectedContext.projectId
      ) {
        setProvisioning(false);
      }
    }
  }

  async function registerRuntime(): Promise<void> {
    if (!projectId) return;
    const expectedContext = { workspaceId, projectId };
    await provision(async () => {
      const environment = await createExecutionEnvironment(projectId, {
        name: runtimeName,
        environmentClass: runtimeClass,
        imageDigest: runtimeDigest,
        packageManifest: parseJsonObject(runtimePackages, "Package manifest"),
        hardwareConstraints: parseJsonObject(
          runtimeHardware,
          "Hardware constraints",
        ),
      });
      if (
        operationContextRef.current.workspaceId !== workspaceId ||
        operationContextRef.current.projectId !== projectId
      ) {
        return;
      }
      setData((current) => ({
        ...current,
        environments: [...current.environments, environment],
      }));
      setVerificationEnvironmentId(environment.id);
      setRuntimeName("");
      setRuntimeDigest("");
      setRuntimePackages("{}");
      setRuntimeHardware("{}");
    }, "Runtime registered. Record a worker preflight result before launch.", expectedContext);
  }

  async function recordVerification(): Promise<void> {
    if (!projectId || !verificationEnvironmentId) return;
    const expectedContext = { workspaceId, projectId };
    await provision(async () => {
      const environment = await verifyExecutionEnvironment(
        projectId,
        verificationEnvironmentId,
        verificationStatus,
        parseJsonObject(verificationReport, "Verification report"),
      );
      if (
        operationContextRef.current.workspaceId !== workspaceId ||
        operationContextRef.current.projectId !== projectId
      ) {
        return;
      }
      setData((current) => ({
        ...current,
        environments: current.environments.map((item) =>
          item.id === environment.id ? environment : item,
        ),
      }));
      setVerificationReport("{}");
    }, "Worker preflight result recorded.", expectedContext);
  }

  async function addStoragePolicy(): Promise<void> {
    if (!projectId) return;
    const expectedContext = { workspaceId, projectId };
    await provision(async () => {
      const policy = await createStoragePolicy(projectId, {
        name: storageName,
        backend: storageBackend,
        artifactPrefix,
        retentionClass,
        encryption: parseJsonObject(storageEncryption, "Encryption settings"),
        cachePolicy: parseJsonObject(storageCache, "Cache policy"),
        isDefault: storageDefault,
      });
      if (
        operationContextRef.current.workspaceId !== workspaceId ||
        operationContextRef.current.projectId !== projectId
      ) {
        return;
      }
      setData((current) => ({
        ...current,
        storagePolicies: [
          ...current.storagePolicies.map((item) =>
            policy.is_default ? { ...item, is_default: false } : item,
          ),
          policy,
        ],
      }));
      setStorageName("");
      setStorageEncryption("{}");
      setStorageCache("{}");
    }, "Storage policy added.", expectedContext);
  }

  const context = contexts.find((item) => item.project_id === projectId);
  return (
    <>
      <PlatformPageHeader
        title="Runtimes"
        description="Verified worker environments and administrator-managed artifact storage."
        actionLabel={
          canProvisionRuntimes && projectId ? "Project settings" : undefined
        }
        onAction={
          canProvisionRuntimes && projectId
            ? () => onNavigate(`/projects/${projectId}/settings`)
            : undefined
        }
      />
      <PlatformSection title="Project context">
        <div className="platform-form-grid">
          <ProjectContextPicker
            contexts={contexts}
            projectId={projectId}
            onChange={setProjectId}
          />
        </div>
      </PlatformSection>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      {statusMessage ? (
        <p role="status" className="platform-form-success">
          {statusMessage}
        </p>
      ) : null}
      {canProvisionRuntimes ? (
        <PlatformSection
          title="Provision project infrastructure"
          description="Register externally prepared workers and storage. Package installation and worker preflight stay outside the browser."
        >
          <div className="platform-provision-grid">
            <form
              className="platform-dialog-form"
              onSubmit={(event) => {
                event.preventDefault();
                void registerRuntime();
              }}
            >
              <h3>Register runtime</h3>
              <label>
                <span>Runtime name</span>
                <input
                  required
                  maxLength={255}
                  value={runtimeName}
                  onChange={(event) => setRuntimeName(event.target.value)}
                />
              </label>
              <label>
                <span>Runtime class</span>
                <select
                  value={runtimeClass}
                  onChange={(event) => setRuntimeClass(event.target.value)}
                >
                  <option value="classical-cpu">Classical CPU</option>
                  <option value="torch-cpu">Torch CPU</option>
                  <option value="transformer-cpu">Transformer CPU</option>
                  <option value="peft-accelerator">PEFT accelerator</option>
                  <option value="qlora-cuda">QLoRA CUDA</option>
                </select>
              </label>
              <label>
                <span>Immutable image digest</span>
                <input
                  required
                  minLength={64}
                  maxLength={71}
                  pattern="(?:sha256:)?[0-9a-fA-F]{64}"
                  placeholder="sha256:…"
                  value={runtimeDigest}
                  onChange={(event) => setRuntimeDigest(event.target.value)}
                />
              </label>
              <details>
                <summary>Runtime manifest</summary>
                <label>
                  <span>Package manifest JSON</span>
                  <textarea
                    rows={4}
                    value={runtimePackages}
                    onChange={(event) => setRuntimePackages(event.target.value)}
                  />
                </label>
                <label>
                  <span>Hardware constraints JSON</span>
                  <textarea
                    rows={4}
                    value={runtimeHardware}
                    onChange={(event) => setRuntimeHardware(event.target.value)}
                  />
                </label>
              </details>
              <Button
                type="submit"
                label={provisioning ? "Registering…" : "Register runtime"}
                variant="primary"
                isDisabled={provisioning || !projectId}
              />
            </form>

            <form
              className="platform-dialog-form"
              onSubmit={(event) => {
                event.preventDefault();
                void recordVerification();
              }}
            >
              <h3>Record worker preflight</h3>
              <label>
                <span>Runtime</span>
                <select
                  required
                  value={verificationEnvironmentId || ""}
                  onChange={(event) =>
                    setVerificationEnvironmentId(Number(event.target.value))
                  }
                >
                  <option value="">Select runtime</option>
                  {data.environments.map((environment) => (
                    <option key={environment.id} value={environment.id}>
                      {environment.name}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Availability</span>
                <select
                  value={verificationStatus}
                  onChange={(event) =>
                    setVerificationStatus(
                      event.target.value as "available" | "unavailable",
                    )
                  }
                >
                  <option value="available">Available</option>
                  <option value="unavailable">Unavailable</option>
                </select>
              </label>
              <label>
                <span>Preflight report JSON</span>
                <textarea
                  rows={7}
                  value={verificationReport}
                  onChange={(event) => setVerificationReport(event.target.value)}
                />
              </label>
              <Button
                type="submit"
                label={provisioning ? "Recording…" : "Record preflight"}
                isDisabled={
                  provisioning || !projectId || !verificationEnvironmentId
                }
              />
            </form>

            <form
              className="platform-dialog-form"
              onSubmit={(event) => {
                event.preventDefault();
                void addStoragePolicy();
              }}
            >
              <h3>Add storage policy</h3>
              <label>
                <span>Policy name</span>
                <input
                  required
                  maxLength={255}
                  value={storageName}
                  onChange={(event) => setStorageName(event.target.value)}
                />
              </label>
              <div className="platform-form-grid">
                <label>
                  <span>Backend</span>
                  <select
                    value={storageBackend}
                    onChange={(event) =>
                      setStorageBackend(
                        event.target.value as "minio" | "local",
                      )
                    }
                  >
                    <option value="minio">S3 / MinIO</option>
                    <option value="local">Local development</option>
                  </select>
                </label>
                <label>
                  <span>Retention</span>
                  <select
                    value={retentionClass}
                    onChange={(event) =>
                      setRetentionClass(
                        event.target.value as "indefinite" | "resume_14d",
                      )
                    }
                  >
                    <option value="indefinite">Indefinite</option>
                    <option value="resume_14d">Resume artifacts · 14 days</option>
                  </select>
                </label>
              </div>
              <label>
                <span>Artifact prefix</span>
                <input
                  required
                  readOnly
                  maxLength={512}
                  value={artifactPrefix}
                />
              </label>
              <label className="platform-checkbox-row">
                <input
                  type="checkbox"
                  checked={storageDefault}
                  onChange={(event) => setStorageDefault(event.target.checked)}
                />
                <span>Use as the project default</span>
              </label>
              <details>
                <summary>Storage controls</summary>
                <label>
                  <span>Encryption settings JSON</span>
                  <textarea
                    rows={4}
                    value={storageEncryption}
                    onChange={(event) => setStorageEncryption(event.target.value)}
                  />
                </label>
                <label>
                  <span>Cache policy JSON</span>
                  <textarea
                    rows={4}
                    value={storageCache}
                    onChange={(event) => setStorageCache(event.target.value)}
                  />
                </label>
              </details>
              <Button
                type="submit"
                label={provisioning ? "Adding…" : "Add storage policy"}
                variant="primary"
                isDisabled={provisioning || !projectId}
              />
            </form>
          </div>
        </PlatformSection>
      ) : null}
      <PlatformSection
        title="Execution environments"
        description={`Availability for ${
          context?.project_name ?? "the selected project"
        }. Package installation is never initiated here.`}
      >
        {loading && !data.environments.length ? (
          <p className="platform-inline-empty" role="status">Loading runtimes…</p>
        ) : data.environments.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Execution environments"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Runtime</th>
                  <th scope="col">Project</th>
                  <th scope="col">Class</th>
                  <th scope="col">Image digest</th>
                  <th scope="col">Verification</th>
                  <th scope="col">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.environments.map((environment) => (
                  <tr key={environment.id}>
                    <td data-label="Runtime" data-priority="identity">
                      <strong>{environment.name}</strong>
                    </td>
                    <td data-label="Project">{context?.project_name}</td>
                    <td data-label="Class">{environment.environment_class}</td>
                    <td data-label="Image digest">
                      <code>{environment.image_digest ?? "Not bound"}</code>
                    </td>
                    <td data-label="Verification">
                      {environment.verified_at
                        ? new Date(environment.verified_at).toLocaleString()
                        : "Not verified"}
                    </td>
                    <td data-label="Status" data-priority="status">
                      <PlatformStatus value={environment.status} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No runtime enabled"
            detail="A workspace administrator must enable and verify a worker environment."
          />
        )}
      </PlatformSection>
      <PlatformSection title="Storage policies">
        {data.storagePolicies.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Training storage policies"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Policy</th>
                  <th scope="col">Project</th>
                  <th scope="col">Backend</th>
                  <th scope="col">Artifact prefix</th>
                  <th scope="col">Retention</th>
                  <th scope="col">Default</th>
                </tr>
              </thead>
              <tbody>
                {data.storagePolicies.map((policy) => (
                  <tr key={policy.id}>
                    <td data-label="Policy" data-priority="identity">
                      <strong>{policy.name}</strong>
                    </td>
                    <td data-label="Project">{context?.project_name}</td>
                    <td data-label="Backend">{policy.backend}</td>
                    <td data-label="Artifact prefix"><code>{policy.artifact_prefix}</code></td>
                    <td data-label="Retention">{policy.retention_class}</td>
                    <td data-label="Default">{policy.is_default ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No storage policy"
            detail="An administrator-managed policy is required before training can launch."
          />
        )}
      </PlatformSection>
    </>
  );
}

function TrainingRunDetail({
  workspaceId,
  projects,
  runId,
  currentUserId,
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [run, setRun] = useState<WorkspaceTrainingRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);

  const load = useCallback(async (): Promise<void> => {
    const requestId = ++loadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === loadRequestIdRef.current;
    if (!runId) {
      if (isCurrentRequest()) {
        setRun(null);
        setLoading(false);
        setError(null);
      }
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await findWorkspaceTrainingRun(
        workspaceId,
        runId,
        projects,
      );
      if (isCurrentRequest()) {
        setRun(result);
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(errorMessage(caught, "Unable to load this training run."));
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false);
      }
    }
  }, [projects, runId, workspaceId]);

  useEffect(() => {
    setRun(null);
    void load();
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [load, workspaceId]);

  async function cancel(): Promise<void> {
    if (!run) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await cancelTrainingRun(run.project_id, run.id);
      setRun({ ...run, ...updated });
    } catch (caught) {
      setError(errorMessage(caught, "Unable to cancel this training run."));
    } finally {
      setBusy(false);
    }
  }

  if (loading) {
    return (
      <PlatformPageHeader
        title="Training run"
        description="Loading the immutable run manifest…"
      />
    );
  }
  if (!run) {
    return (
      <>
        <PlatformPageHeader
          title="Training run not found"
          description="This run is unavailable in the active workspace."
          actionLabel="Back to runs"
          onAction={() => onNavigate("/training", "replace")}
        />
        {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      </>
    );
  }

  const cancellable =
    run.created_by_user_id === currentUserId &&
    ["queued", "running"].includes(run.status);
  const baseModel = baseModelLineage(run.base_model);
  return (
    <>
      <PlatformPageHeader
        title={`Training run ${run.id}`}
        description={`${run.project_name} · ${run.model_name}`}
        actionLabel="Back to runs"
        onAction={() => onNavigate("/training")}
        secondary={
          cancellable ? (
            <Button
              label={busy ? "Cancelling…" : "Cancel run"}
              isDisabled={busy}
              onClick={() => void cancel()}
            />
          ) : undefined
        }
      />
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      <PlatformSection title="Run manifest">
        <dl className="platform-review-grid">
          <div><dt>Project context</dt><dd>{run.project_name}</dd></div>
          <div><dt>Model name</dt><dd>{run.model_name}</dd></div>
          <div><dt>Family</dt><dd>{formatStatus(run.family)}</dd></div>
          <div><dt>Framework</dt><dd>{run.framework}</dd></div>
          <div><dt>Base model</dt><dd>{baseModelName(run.base_model)}</dd></div>
          <div><dt>Base model asset</dt><dd>{baseModel.asset}</dd></div>
          <div><dt>Base model source</dt><dd>{baseModel.source}</dd></div>
          <div><dt>Base model revision</dt><dd>{baseModel.revision}</dd></div>
          <div><dt>Training method</dt><dd>{formatStatus(run.training_method)}</dd></div>
          <div><dt>Task</dt><dd>{run.task_name} · {formatStatus(run.task_kind)}</dd></div>
          <div>
            <dt>Training dataset</dt>
            <dd>
              {run.training_dataset_name}
              {run.dataset_version_number
                ? ` · source v${run.dataset_version_number}`
                : ""}
            </dd>
          </div>
          <div><dt>Recipe</dt><dd>{run.recipe_name}</dd></div>
          <div><dt>Runtime</dt><dd>{run.runtime_name}</dd></div>
          <div><dt>Storage</dt><dd>{run.storage_policy_name}</dd></div>
          <div>
            <dt>Evaluation</dt>
            <dd>
              {run.evaluation_status
                ? `${formatStatus(run.evaluation_status)}${
                    run.evaluation_split ? ` · ${run.evaluation_split}` : ""
                  }`
                : "Pending"}
              <span className="platform-review-detail">
                {metricSummary(run.evaluation_metrics)}
              </span>
            </dd>
          </div>
          <div><dt>Creator</dt><dd>{creatorName(run)}</dd></div>
          <div><dt>Seed</dt><dd>{run.seed}</dd></div>
          <div><dt>Status</dt><dd><PlatformStatus value={run.status} /></dd></div>
        </dl>
      </PlatformSection>
      <PlatformSection title="Pinned configuration">
        <pre className="platform-code-block">
          {JSON.stringify(run.config, null, 2)}
        </pre>
      </PlatformSection>
    </>
  );
}
