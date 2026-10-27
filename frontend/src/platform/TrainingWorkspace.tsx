import { NativeInput, NativeSelect, NativeTextarea } from "@/components/NativeField";
import ValidatedForm from "@/components/ValidatedForm";
import LoadingState from "@/components/LoadingState";
import { useCallback, useEffect, useId, useRef, useState, type FormEvent } from "react";
import { Button } from "@astryxdesign/core/Button";

import ConfirmDialog from "@/components/ConfirmDialog";

import type { Project } from "@/types/api";
import { request } from "@/api/client";
import { getSessionGeneration } from "@/auth/session";

import {
  type TrainingLaunchAttempt,
  cancelTrainingRun,
  createExecutionEnvironment,
  createStoragePolicy,
  createTrainingOnlyProject,
  findWorkspaceTrainingRun,
  launchTrainingRun,
  listWorkspaceTrainingContexts,
  listWorkspaceTrainingRuns,
  loadPlatformProject,
  verifyExecutionEnvironment,
} from "./api";
import DataScreen from "./DataScreen";
import { baseModelLineage, baseModelName, creatorName, metricSummary } from "./modelPresentation";
import TrainingDatasetBuilder from "./TrainingDatasetBuilder";
import PlatformDialog from "./PlatformDialog";
import TrainingScreen, { type TrainingLaunchDraft } from "./TrainingScreen";
import { useResourceCreation } from "./useResourceCreation";
import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformSection,
  PlatformStatus,
  formatStatus,
  formatDateTime,
} from "./components";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type ModelEvaluation,
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
  initialDatasetVersionId?: number | null;
  initialTaskVersionId?: number | null;
  initialLabelSetVersionId?: number | null;
  initialTrainingDatasetVersionId?: number | null;
  search?: string;
  currentUserId: number;
  currentUserName: string;
  canCreateTrainingProject: boolean;
  canCreateTask: boolean;
  canProvisionRuntimes: boolean;
  canCancelAnyRun?: boolean;
  onNavigate: (path: string, mode?: "push" | "replace") => void;
  onProjectCreated?: (project: Project) => void | Promise<void>;
}

type TrainingDialog = "dataset" | "task" | null;

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

function jsonObjectError(value: string, label: string): string {
  try {
    parseJsonObject(value, label);
    return "";
  } catch (caught) {
    return errorMessage(caught, "Enter a JSON object.");
  }
}

function JsonObjectField({ label, value, onChange, rows = 4 }: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  rows?: number;
}): React.ReactElement {
  const errorId = useId();
  const message = jsonObjectError(value, label);
  return <label>
    <span>{label} JSON</span>
    <NativeTextarea
      rows={rows}
      value={value}
      aria-invalid={message ? true : undefined}
      aria-describedby={message ? errorId : undefined}
      onChange={(event) => {
        event.currentTarget.setCustomValidity(jsonObjectError(event.currentTarget.value, label));
        onChange(event.currentTarget.value);
      }}
    />
    {message ? <small id={errorId} className="platform-field-error" aria-hidden="true">{message}</small> : null}
  </label>;
}

function revealInvalidField(event: FormEvent<HTMLFormElement>): void {
  for (const details of event.currentTarget.querySelectorAll("details")) {
    if (event.target instanceof Node && details.contains(event.target)) details.open = true;
  }
}

function WorkspaceNavigation({
  current,
  projectId,
  onNavigate,
}: {
  current: TrainingWorkspaceView;
  projectId?: number | null;
  onNavigate: TrainingWorkspaceProps["onNavigate"];
}): React.ReactElement {
  return (
    <nav className="platform-page-actions" aria-label="Training sections">
      {NAV_ITEMS.map((item) => (
        <PlatformRouteLink
          key={item.view}
          href={`${item.path}${projectId ? `?projectId=${projectId}` : ""}`}
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
      <NativeSelect
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
      </NativeSelect>
    </label>
  );
}

async function loadAllTrainingContexts(workspaceId: number, projects: Project[]) {
  const result = await listWorkspaceTrainingContexts(workspaceId, projects, { limit: 100 });
  const items = [...result.items];
  let cursor = result.next_cursor;
  while (cursor !== null) {
    const page = await listWorkspaceTrainingContexts(workspaceId, projects, { limit: 100, cursor });
    items.push(...page.items);
    cursor = page.next_cursor;
  }
  return { items, next_cursor: null };
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
        <WorkspaceNavigation current={props.view} projectId={props.initialProjectId} onNavigate={props.onNavigate} />
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
  canCancelAnyRun = false,
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [filters, setFilters] = useState<WorkspaceTrainingFilters>({
    projectId: initialProjectId ?? undefined,
    limit: 50,
  });
  const appliedFilters = useRef(filters);
  const currentFilters = useRef(filters);
  currentFilters.current = filters;
  const [page, setPage] = useState<WorkspaceTrainingRunPage>({
    items: [],
    next_cursor: null,
  });
  const [loading, setLoading] = useState(true);
  const [busyRunId, setBusyRunId] = useState<number | null>(null);
  const [runToCancel, setRunToCancel] = useState<WorkspaceTrainingRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);
  const projectsRef = useRef(projects);
  projectsRef.current = projects;
  const filterWorkspaceRef = useRef(workspaceId);

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
          projectsRef.current,
        );
        if (requestId !== loadRequestIdRef.current) return;
        appliedFilters.current = nextFilters;
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
    [workspaceId],
  );

  useEffect(() => {
    const nextFilters: WorkspaceTrainingFilters = {
      ...(filterWorkspaceRef.current === workspaceId ? currentFilters.current : {}),
      projectId: initialProjectId ?? undefined,
      cursor: undefined,
      limit: 50,
    };
    filterWorkspaceRef.current = workspaceId;
    setFilters(nextFilters);
    setPage({
      items: [],
      next_cursor: null,
    });
    void load(
      nextFilters,
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
      setRunToCancel(null);
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
        actionLabel={historyRuns.length ? "Train model" : undefined}
        onAction={() => onNavigate(`/training/new${initialProjectId ? `?projectId=${initialProjectId}` : ""}`)}
      />
      <PlatformSection title="Run filters">
        <ValidatedForm
          className="platform-form-grid"
          onSubmit={(event) => {
            event.preventDefault();
            void load(filters);
          }}
        >
          <label>
            <span>Project</span>
            <NativeSelect
              value={filters.projectId ?? ""}
              onChange={(event) => {
                const id = event.target.value ? Number(event.target.value) : undefined;
                setFilters((current) => ({ ...current, projectId: id, cursor: undefined }));
                onNavigate(`/training${id ? `?projectId=${id}` : ""}`, "replace");
              }}
            >
              <option value="">All training projects</option>
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </NativeSelect>
          </label>
          <label>
            <span>Status</span>
            <NativeSelect
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
            </NativeSelect>
          </label>
          <label>
            <span>Model family</span>
            <NativeInput
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
        </ValidatedForm>
      </PlatformSection>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      <PlatformSection
        title="Runs"
        description="Each run remains bound to its project provenance and immutable inputs."
      >
        {loading && !historyRuns.length ? (
          <LoadingState label="Loading training runs…" />
        ) : error && !historyRuns.length ? (
          <Button label="Retry loading runs" onClick={() => void load(appliedFilters.current)} />
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
                    (canCancelAnyRun || run.created_by_user_id === currentUserId) &&
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
                            onClick={() => { setError(null); setRunToCancel(run); }}
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
            actionLabel="Train model"
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
                    ...appliedFilters.current,
                    cursor: page.next_cursor ?? undefined,
                  },
                  true,
                )
              }
            />
          </div>
        ) : null}
      </PlatformSection>
      <ConfirmDialog open={runToCancel !== null} title="Cancel training run?" description="This stops the run. A cancelled run cannot be resumed. Launch a new run to train again." confirmLabel="Cancel run" busy={busyRunId !== null} error={error} onCancel={() => { setRunToCancel(null); setError(null); }} onConfirm={async () => { if (runToCancel) await cancel(runToCancel); }} />
    </>
  );
}

function NewTraining({
  workspaceId,
  projects,
  initialProjectId,
  initialDatasetId,
  initialTrainingDatasetVersionId,
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
  const [loadedProjectId, setLoadedProjectId] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const contextsRequestIdRef = useRef(0);
  const projectDataRequestIdRef = useRef(0);
  const selectedContextRef = useRef(initialProjectId ?? 0);
  const launchAttempt = useRef<{ signature: string; attempt: TrainingLaunchAttempt } | null>(null);

  const loadContexts = useCallback(async (): Promise<void> => {
    const requestId = ++contextsRequestIdRef.current;
    projectDataRequestIdRef.current += 1;
    setLoading(true);
    setError(null);
    try {
      const result = await loadAllTrainingContexts(workspaceId, projects);
      if (requestId !== contextsRequestIdRef.current) return;
      setContexts(result.items);
      setContextsWorkspaceId(workspaceId);
      if (initialProjectId && selectedContextRef.current === initialProjectId) {
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
    selectedContextRef.current = initialProjectId ?? 0;
    setProjectId(initialProjectId ?? 0);
    setContextConfirmed(false);
    setContextKind("existing");
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoadedProjectId(null);
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
        setLoadedProjectId(projectId);
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
      selectedContextRef.current = project.id;
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
      const signature = JSON.stringify({ projectId, draft });
      if (launchAttempt.current?.signature !== signature) {
        launchAttempt.current = { signature, attempt: { key: crypto.randomUUID(), modelId: null, recipeVersionId: null } };
      }
      const run = await launchTrainingRun(projectId, draft, data, launchAttempt.current.attempt);
      onNavigate(`/training/runs/${run.id}`);
    } catch (caught) {
      const message = errorMessage(caught, "Unable to launch training.");
      setError(message);
    } finally {
      setBusy(false);
    }
  }

  const contextChooser = !contextConfirmed ? (
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
              <NativeInput
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
              <NativeInput
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
              onChange={(id) => { selectedContextRef.current = id; setProjectId(id); }}
            />
          </div>
          <div className="platform-wizard-actions">
            <Button
              label={loading ? "Loading…" : "Use selected project"}
              variant="primary"
              isDisabled={!projectId || loading}
              onClick={() => { setContextConfirmed(true); onNavigate(`/training/new?projectId=${projectId}`, "replace"); }}
            />
          </div>
        </PlatformSection>
        {contextKind === "training_only" && canCreateTrainingProject ? (
          <PlatformSection
          title="Create training-only project"
            description="It enables Data, Training, Models, and Activity without enabling Annotation."
          >
            <ValidatedForm
              className="platform-dialog-form"
              onSubmit={(event) => {
                event.preventDefault();
                void createProject();
              }}
            >
              <label>
                <span>Project name</span>
                <NativeInput
                  required
                  maxLength={255}
                  value={projectName}
                  onChange={(event) => setProjectName(event.target.value)}
                />
              </label>
              <label>
                <span>Description</span>
                <NativeTextarea
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
            </ValidatedForm>
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
    ) : null;

  if (contextConfirmed && loadedProjectId !== projectId && loading && !error) {
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
      {contextChooser}
      <div hidden={!contextConfirmed}>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      <div className="platform-section-header">
        {context && contextsWorkspaceId === workspaceId ? <p>
          Project context: <strong>{context.project_name}</strong>
        </p> : null}
        <Button
          label="Change context"
          size="sm"
          onClick={() => setContextConfirmed(false)}
        />
      </div>
      {loadedProjectId === projectId ? <TrainingScreen
        key={`${trainingDraftKey(workspaceId, projectId, initialDatasetId)}:${initialTrainingDatasetVersionId ?? "none"}`}
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
        initialDatasetId={projectId === initialProjectId ? initialDatasetId : null}
        initialTrainingDatasetVersionId={projectId === initialProjectId ? initialTrainingDatasetVersionId : null}
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
              projectId === initialProjectId && initialDatasetId ? `&datasetId=${initialDatasetId}` : ""
            }`,
          )
        }
        onLaunch={launch}
      /> : null}
      </div>
    </>
  );
}

function TrainingData({
  workspaceId,
  projects,
  initialProjectId,
  initialDatasetId,
  canCreateTask,
  canCreateTrainingProject,
  initialDatasetVersionId,
  initialTaskVersionId,
  initialLabelSetVersionId,
  initialTrainingDatasetVersionId,
  search = "",
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [contexts, setContexts] = useState<WorkspaceTrainingContext[]>([]);
  const [contextsWorkspaceId, setContextsWorkspaceId] = useState<number | null>(
    null,
  );
  const projectId = contextsWorkspaceId !== workspaceId ? 0 :
    initialProjectId ? contexts.find((item) => item.project_id === initialProjectId)?.project_id ?? 0 : contexts[0]?.project_id ?? 0;
  const resourceCreation = useResourceCreation(projectId || null, workspaceId);
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState<TrainingDialog>(null);
  const builderOpen = new URLSearchParams(search).get("flow") === "prepare" || Boolean(initialDatasetId);
  const [loadedProjectId, setLoadedProjectId] = useState<number | null>(null);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  useEffect(() => {
    if (!savedMessage) return;
    const timer = setTimeout(() => setSavedMessage(null), 5_000);
    return () => clearTimeout(timer);
  }, [savedMessage]);
  const dataTab = new URLSearchParams(search).get("tab") === "source" ? "source" : "training";
  const reloadRequestIdRef = useRef(0);
  const sessionGeneration = useRef(getSessionGeneration()).current;
  const reloadContextKey = `${workspaceId}:${contextsWorkspaceId ?? "none"}:${projectId}`;
  const reloadContextKeyRef = useRef(reloadContextKey);
  reloadContextKeyRef.current = reloadContextKey;

  const reload = useCallback(async (): Promise<boolean> => {
    if (reloadContextKeyRef.current !== reloadContextKey || sessionGeneration !== getSessionGeneration()) return false;
    const requestId = ++reloadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === reloadRequestIdRef.current &&
      reloadContextKey === reloadContextKeyRef.current &&
      sessionGeneration === getSessionGeneration();
    if (
      !projectId ||
      contextsWorkspaceId !== workspaceId ||
      !contexts.some((item) => item.project_id === projectId)
    ) {
      if (isCurrentRequest()) {
        setData(EMPTY_PLATFORM_PROJECT_DATA);
      }
      return false;
    }
    setLoading(true);
    setError(null);
    try {
      const [sourceData, trainingData] = await Promise.all([
        loadPlatformProject(projectId, "data", workspaceId),
        loadPlatformProject(projectId, "train", workspaceId),
      ]);
      if (isCurrentRequest()) {
        setLoadedProjectId(projectId);
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
        return true;
      }
      return false;
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(errorMessage(caught, "Unable to load training data."));
      }
      throw caught;
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
    sessionGeneration,
    workspaceId,
  ]);

  useEffect(() => {
    let active = true;
    setContexts([]);
    setContextsWorkspaceId(null);
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoading(true);
    void loadAllTrainingContexts(workspaceId, projects)
      .then((result) => {
        if (!active) return;
        setContexts(result.items);
        setContextsWorkspaceId(workspaceId);
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
  }, [projects, workspaceId]);

  useEffect(() => {
    void reload().catch(() => undefined);
    return () => {
      reloadRequestIdRef.current += 1;
    };
  }, [reload]);

  useEffect(() => {
    if (!initialProjectId && projectId && contextsWorkspaceId === workspaceId) {
      const params = new URLSearchParams(search);
      params.set("projectId", String(projectId));
      onNavigate(`/training/data?${params}`, "replace");
    }
  }, [contextsWorkspaceId, initialProjectId, onNavigate, projectId, search, workspaceId]);

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
      <PlatformPageHeader
        title="Training data"
        description="Prepare and manage training datasets from submitted annotations or external labels."
        actionLabel={!builderOpen && !loading && !error && projectId > 0 && loadedProjectId === projectId && (dataTab === "training" ? data.trainingDatasets.length > 0 : data.datasets.length > 0)
          ? dataTab === "training" ? "Create training dataset" : "Import source data"
          : undefined}
        onAction={() => dataTab === "training"
          ? onNavigate(`/training/data?projectId=${projectId}&flow=prepare`)
          : setDialog("dataset")}
      />
      <PlatformSection title="Project context">
        <div className="platform-form-grid">
          <ProjectContextPicker
            contexts={contexts}
            projectId={projectId}
            onChange={(nextProjectId) => {
              setDialog(null);
              setSavedMessage(null);
              onNavigate(`/training/data?projectId=${nextProjectId}${builderOpen ? "&flow=prepare" : `&tab=${dataTab}`}`);
            }}
          />
        </div>
      </PlatformSection>
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      {savedMessage ? <p role="status">{savedMessage}</p> : null}
      {builderOpen && projectId && loadedProjectId === projectId ? <TrainingDatasetBuilder key={`${workspaceId}:${projectId}:${initialTrainingDatasetVersionId ?? "new"}:${initialDatasetVersionId ?? initialDatasetId ?? "all"}:${initialTaskVersionId ?? "task"}`}
        projectId={projectId} data={data} initialDatasetId={initialDatasetId}
        canUseRoundAnnotations={canCreateTask}
        initialDatasetVersionId={initialDatasetVersionId} initialTaskVersionId={initialTaskVersionId} initialLabelSetVersionId={initialLabelSetVersionId}
        parent={data.trainingDatasets.find((version) => version.id === initialTrainingDatasetVersionId)}
        onImport={() => setDialog("dataset")} onDefineTask={canCreateTask ? () => setDialog("task") : undefined}
        onCancel={() => onNavigate(`/training/data?projectId=${projectId}&tab=training`)}
        onCreated={async (version) => {
          if (reloadContextKeyRef.current !== reloadContextKey || sessionGeneration !== getSessionGeneration()) throw new Error("The training context changed. Your dataset remains saved in its original project.");
          setSavedMessage(`${version.name} · v${version.version_number ?? 1} saved. Choose Train model to launch training.`);
          const refreshed = await reload();
          if (!refreshed || reloadContextKeyRef.current !== reloadContextKey || sessionGeneration !== getSessionGeneration()) throw new Error("The training context changed. Your dataset remains saved in its original project.");
          onNavigate(`/training/data?projectId=${projectId}&tab=training`);
        }} /> : null}
      {(loading || (projectId > 0 && loadedProjectId !== projectId && !error)) && !(builderOpen && loadedProjectId === projectId) ? (
        <LoadingState label="Loading training data…" />
      ) : error ? null : projectId ? (builderOpen ? null : (
        <DataScreen
          data={loadedProjectId === projectId ? data : EMPTY_PLATFORM_PROJECT_DATA}
          legacyDocumentCount={0}
          showHeader={false}
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
          activeTab={dataTab}
          onTabChange={(tab) => onNavigate(`/training/data?projectId=${projectId}&tab=${tab}`)}
          onCreateTraining={() => onNavigate(`/training/data?projectId=${projectId}&flow=prepare`)}
          onTrain={(version) => onNavigate(`/training/new?projectId=${projectId}&trainingDatasetVersionId=${version.id}`)}
          onNewTrainingVersion={(version) => onNavigate(`/training/data?projectId=${projectId}&flow=prepare&trainingDatasetId=${version.training_dataset_id}&trainingDatasetVersionId=${version.id}`)}
          onPrepareTraining={(dataset) => {
            onNavigate(`/training/data?projectId=${projectId}&flow=prepare&datasetId=${dataset.id}`);
          }}
        />
      )) : (
        <PlatformEmpty
          title="No training project"
          detail={canCreateTrainingProject ? "Create a project with Training enabled before importing training data." : "Ask a workspace manager to create a project with Training enabled before importing training data."}
          actionLabel="Open projects"
          onAction={() => onNavigate("/projects")}
        />
      )}
      {dialog && projectId ? (
        <PlatformDialog
          kind={dialog}
          taskPurpose="training"
          data={data}
          busy={busy}
          onClose={() => {
            setDialog(null);
          }}
          onCreateDataset={(draft) =>
            mutate(() => resourceCreation.createDataset(draft))
          }
          onCreateTask={(draft) =>
            mutate(() => resourceCreation.createTask(draft))
          }
          onCreateRound={async () => {
            throw new Error("Rounds are managed from Projects.");
          }}
          onScoreFeedback={async () => {
            throw new Error("Model scoring is managed from Projects.");
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
  const mutationRevision = useRef(0);
  const operationContextRef = useRef({ workspaceId, projectId });
  operationContextRef.current = { workspaceId, projectId };

  useEffect(() => {
    let active = true;
    setContexts([]);
    setContextsWorkspaceId(null);
    setProjectId(0);
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoading(true);
    void loadAllTrainingContexts(workspaceId, projects)
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
    const revision = mutationRevision.current;
    setData(EMPTY_PLATFORM_PROJECT_DATA);
    setLoading(true);
    setError(null);
    void loadPlatformProject(projectId, "train", workspaceId)
      .then((result) => {
        if (active) setData((current) => revision === mutationRevision.current ? result : {
          ...result,
          environments: [...new Map([...result.environments, ...current.environments].map((item) => [item.id, item])).values()],
          storagePolicies: [...new Map([...result.storagePolicies, ...current.storagePolicies].map((item) => [item.id, item])).values()],
        });
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
      mutationRevision.current += 1;
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
      mutationRevision.current += 1;
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
      mutationRevision.current += 1;
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

  useEffect(() => {
    if (!initialProjectId && projectId && contextsWorkspaceId === workspaceId) {
      onNavigate(`/training/runtimes?projectId=${projectId}`, "replace");
    }
  }, [contextsWorkspaceId, initialProjectId, onNavigate, projectId, workspaceId]);

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
            onChange={(nextProjectId) => { setProjectId(nextProjectId); onNavigate(`/training/runtimes?projectId=${nextProjectId}`, "replace"); }}
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
            <ValidatedForm
              className="platform-dialog-form"
              onInvalidCapture={revealInvalidField}
              onSubmit={(event) => {
                event.preventDefault();
                void registerRuntime();
              }}
            >
              <h3>Register runtime</h3>
              <label>
                <span>Runtime name</span>
                <NativeInput
                  required
                  maxLength={255}
                  value={runtimeName}
                  onChange={(event) => setRuntimeName(event.target.value)}
                />
              </label>
              <label>
                <span>Runtime class</span>
                <NativeSelect
                  value={runtimeClass}
                  onChange={(event) => setRuntimeClass(event.target.value)}
                >
                  <option value="classical-cpu">Classical CPU</option>
                  <option value="torch-cpu">Torch CPU</option>
                  <option value="transformer-cpu">Transformer CPU</option>
                  <option value="peft-accelerator">PEFT accelerator</option>
                  <option value="qlora-cuda">QLoRA CUDA</option>
                </NativeSelect>
              </label>
              <label>
                <span>Immutable image digest</span>
                <NativeInput
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
                <JsonObjectField label="Package manifest" value={runtimePackages} onChange={setRuntimePackages} />
                <JsonObjectField label="Hardware constraints" value={runtimeHardware} onChange={setRuntimeHardware} />
              </details>
              <Button
                type="submit"
                label={provisioning ? "Registering…" : "Register runtime"}
                variant="primary"
                isDisabled={provisioning || !projectId}
              />
            </ValidatedForm>

            <ValidatedForm
              className="platform-dialog-form"
              onInvalidCapture={revealInvalidField}
              onSubmit={(event) => {
                event.preventDefault();
                void recordVerification();
              }}
            >
              <h3>Record worker preflight</h3>
              <label>
                <span>Runtime</span>
                <NativeSelect
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
                </NativeSelect>
              </label>
              <label>
                <span>Availability</span>
                <NativeSelect
                  value={verificationStatus}
                  onChange={(event) =>
                    setVerificationStatus(
                      event.target.value as "available" | "unavailable",
                    )
                  }
                >
                  <option value="available">Available</option>
                  <option value="unavailable">Unavailable</option>
                </NativeSelect>
              </label>
              <JsonObjectField label="Preflight report" value={verificationReport} onChange={setVerificationReport} rows={7} />
              <Button
                type="submit"
                label={provisioning ? "Recording…" : "Record preflight"}
                isDisabled={
                  provisioning || !projectId || !verificationEnvironmentId
                }
              />
            </ValidatedForm>

            <ValidatedForm
              className="platform-dialog-form"
              onInvalidCapture={revealInvalidField}
              onSubmit={(event) => {
                event.preventDefault();
                void addStoragePolicy();
              }}
            >
              <h3>Add storage policy</h3>
              <label>
                <span>Policy name</span>
                <NativeInput
                  required
                  maxLength={255}
                  value={storageName}
                  onChange={(event) => setStorageName(event.target.value)}
                />
              </label>
              <div className="platform-form-grid">
                <label>
                  <span>Backend</span>
                  <NativeSelect
                    value={storageBackend}
                    onChange={(event) =>
                      setStorageBackend(
                        event.target.value as "minio" | "local",
                      )
                    }
                  >
                    <option value="minio">S3 / MinIO</option>
                    <option value="local">Local development</option>
                  </NativeSelect>
                </label>
                <label>
                  <span>Retention</span>
                  <NativeSelect
                    value={retentionClass}
                    onChange={(event) =>
                      setRetentionClass(
                        event.target.value as "indefinite" | "resume_14d",
                      )
                    }
                  >
                    <option value="indefinite">Indefinite</option>
                    <option value="resume_14d">Resume artifacts · 14 days</option>
                  </NativeSelect>
                </label>
              </div>
              <label>
                <span>Artifact prefix</span>
                <NativeInput
                  required
                  readOnly
                  maxLength={512}
                  value={artifactPrefix}
                />
              </label>
              <label className="platform-checkbox-row">
                <NativeInput
                  type="checkbox"
                  checked={storageDefault}
                  onChange={(event) => setStorageDefault(event.target.checked)}
                />
                <span>Use as the project default</span>
              </label>
              <details>
                <summary>Storage controls</summary>
                <JsonObjectField label="Encryption settings" value={storageEncryption} onChange={setStorageEncryption} />
                <JsonObjectField label="Cache policy" value={storageCache} onChange={setStorageCache} />
              </details>
              <Button
                type="submit"
                label={provisioning ? "Adding…" : "Add storage policy"}
                variant="primary"
                isDisabled={provisioning || !projectId}
              />
            </ValidatedForm>
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
          <LoadingState label="Loading runtimes…" />
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
                      {environment.image_digest ? <details><summary>View image digest</summary><code>{environment.image_digest}</code></details> : "Not bound"}
                    </td>
                    <td data-label="Verification">
                      {formatDateTime(environment.verified_at, "Not verified")}
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
                    <td data-label="Artifact prefix"><details><summary>View storage path</summary><code>{policy.artifact_prefix}</code></details></td>
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
            actionLabel={canProvisionRuntimes ? "Configure storage policy" : "Open workspace settings"}
            onAction={canProvisionRuntimes ? () => window.document.querySelector<HTMLFormElement>(".platform-provision-grid form:last-child")?.querySelector<HTMLInputElement>("input")?.focus() : () => onNavigate("/workspace-settings")}
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
  canCancelAnyRun = false,
  onNavigate,
}: TrainingWorkspaceProps): React.ReactElement {
  const [run, setRun] = useState<WorkspaceTrainingRun | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>([]);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);

  const load = useCallback(async (background = false): Promise<void> => {
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
    if (!background) setLoading(true);
    setError(null);
    try {
      const result = await findWorkspaceTrainingRun(
        workspaceId,
        runId,
        projects,
      );
      if (!isCurrentRequest()) return;
      setRun(result);
      setEvaluationError(null);
      if (result) {
        try {
          const reports = await request<ModelEvaluation[]>(`/training-runs/${result.id}/evaluations?project_id=${result.project_id}`);
          if (isCurrentRequest()) setEvaluations(reports);
        } catch (caught) {
          if (isCurrentRequest()) setEvaluationError(errorMessage(caught, "Evaluation details could not be loaded."));
        }
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
    setEvaluations([]);
    setEvaluationError(null);
    setConfirmCancel(false);
    setBusy(false);
    void load();
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [load, workspaceId]);

  const runStatus = run?.status;
  useEffect(() => {
    if (!runStatus || busy || !["queued", "running"].includes(runStatus)) return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async (): Promise<void> => {
      await load(true);
      if (active) timer = setTimeout(() => void poll(), 3_000);
    };
    timer = setTimeout(() => void poll(), 3_000);
    return () => { active = false; clearTimeout(timer); };
  }, [busy, load, runStatus]);

  async function cancel(): Promise<void> {
    if (!run) return;
    const requestId = ++loadRequestIdRef.current;
    setBusy(true);
    setError(null);
    try {
      const updated = await cancelTrainingRun(run.project_id, run.id);
      if (requestId === loadRequestIdRef.current) { setRun({ ...run, ...updated }); setConfirmCancel(false); }
    } catch (caught) {
      if (requestId === loadRequestIdRef.current) setError(errorMessage(caught, "Unable to cancel this training run."));
    } finally {
      if (requestId === loadRequestIdRef.current) setBusy(false);
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
  if (!run && error) {
    return <>
      <PlatformPageHeader title="Training run could not be loaded" description="Try loading this run again." actionLabel="Retry loading run" onAction={() => void load()} />
      <p role="alert" className="platform-form-warning">{error}</p>
      <Button label="Back to runs" onClick={() => onNavigate("/training")} />
    </>;
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
    (canCancelAnyRun || run.created_by_user_id === currentUserId) &&
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
              onClick={() => { setError(null); setConfirmCancel(true); }}
            />
          ) : undefined
        }
      />
      {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
      {run.failure_reason ? <p role="alert" className="platform-form-warning">{run.failure_reason}</p> : null}
      <PlatformSection title="Run manifest">
        <dl className="platform-review-grid">
          <div><dt>Project context</dt><dd>{run.project_name}</dd></div>
          <div><dt>Model name</dt><dd>{run.model_name}</dd></div>
          <div><dt>Family</dt><dd>{formatStatus(run.family)}</dd></div>
          <div><dt>Framework</dt><dd>{run.framework}</dd></div>
          <div><dt>Base model</dt><dd>{baseModelName(run.base_model)}</dd></div>
          <div><dt>Base model details</dt><dd><details><summary>View provenance</summary><p>Asset {baseModel.asset} · Source {baseModel.source} · Revision <code>{baseModel.revision}</code></p></details></dd></div>
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
          <div><dt>Created</dt><dd>{formatDateTime(run.created_at, "Not recorded")}</dd></div>
          <div><dt>Started</dt><dd>{formatDateTime(run.started_at, "Not started")}</dd></div>
          <div><dt>Completed</dt><dd>{formatDateTime(run.completed_at, "Not completed")}</dd></div>
        </dl>
      </PlatformSection>
      <PlatformSection title="Evaluation details">
        {evaluationError ? <><p role="alert" className="platform-form-warning">{evaluationError}</p><Button label="Retry evaluation details" onClick={() => void load(true)} /></> : evaluations.length ? <ul>{evaluations.map((evaluation) => <li key={evaluation.id}><strong>{formatStatus(evaluation.split_name)}</strong> · <PlatformStatus value={evaluation.status} />{evaluation.status_reason ? <p>{evaluation.status_reason}</p> : null}<p>{metricSummary(evaluation.metrics)}</p></li>)}</ul> : <p>No evaluation reports are available for this run.</p>}
      </PlatformSection>
      <PlatformSection title="Pinned configuration">
        <details><summary>View configuration</summary><pre className="platform-code-block">{JSON.stringify(run.config, null, 2)}</pre></details>
      </PlatformSection>
      <ConfirmDialog open={confirmCancel} title="Cancel training run?" description="This stops the run. A cancelled run cannot be resumed. Launch a new run to train again." confirmLabel="Cancel run" busy={busy} error={error} onCancel={() => { setConfirmCancel(false); setError(null); }} onConfirm={cancel} />
    </>
  );
}
