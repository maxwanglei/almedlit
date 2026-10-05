import { useEffect, useId, useMemo, useState } from "react";

import { request } from "@/api/client";
import type { ProjectTask } from "@/types/api";

import { PlatformRouteLink } from "./components";
import { paperAnnotationCompatibility } from "./paperTaskContracts";
import type { Dataset, DatasetVersion, RoundWorkContext, TaskDefinition, TaskVersion } from "./types";

interface TaskInventoryProject {
  id: number;
  tasks?: ProjectTask[];
  workflow_task_count?: number;
}

interface SavedTasks {
  projectId: number;
  definitions: TaskDefinition[];
  versions: TaskVersion[];
  datasetVersions: DatasetVersion[];
}

/** Saved tasks and assigned rounds are separate inventories: a task survives before its first round. */
export default function MyWorkTaskInventory({ project, contexts, canReadAllTasks, canManage, onNavigate }: {
  project: TaskInventoryProject;
  contexts: RoundWorkContext[];
  canReadAllTasks: boolean;
  canManage: boolean;
  onNavigate: (path: string) => void;
}): React.ReactElement {
  const headingId = useId();
  const [saved, setSaved] = useState<SavedTasks | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const projectId = project.id;
  useEffect(() => {
    let active = true;
    setSaved(null);
    setError(null);
    if (!canReadAllTasks) return;
    void Promise.all([
      request<TaskDefinition[]>(`/tasks?project_id=${projectId}`),
      request<TaskVersion[]>(`/tasks/versions?project_id=${projectId}`),
      request<Dataset[]>(`/datasets?project_id=${projectId}`),
    ]).then(async ([definitions, versions, datasets]) => {
      if (!active) return;
      if (definitions.some((task) => task.project_id !== projectId) || versions.some((version) => version.project_id !== projectId) || datasets.some((dataset) => dataset.project_id !== projectId)) {
        throw new Error("The server returned tasks or source collections from a different project. Reload the task list.");
      }
      // The API requires dataset_id for version listing; project scope alone is insufficient.
      const datasetVersions = (await Promise.all(datasets.map(async (dataset) => {
        const children = await request<DatasetVersion[]>(`/datasets/versions?project_id=${projectId}&dataset_id=${dataset.id}`);
        if (children.some((version) => version.project_id !== projectId || version.dataset_id !== dataset.id)) {
          throw new Error("The server returned a source version from a different collection. Reload the task list.");
        }
        return children;
      }))).flat();
      if (active) setSaved({ projectId, definitions, versions, datasetVersions });
    }).catch((caught: unknown) => {
      if (active) setError(caught instanceof Error ? caught.message : "Saved annotation tasks could not be loaded.");
    });
    return () => { active = false; };
  }, [attempt, canReadAllTasks, projectId, project.workflow_task_count]);

  const projectContexts = contexts.filter((context) => context.project.id === projectId);
  const loaded = canReadAllTasks && saved?.projectId === projectId ? saved : null;
  const assignedDefinitions = new Map(projectContexts.map((context) => [context.task.id, context.task]));
  const definitions = loaded?.definitions ?? [...assignedDefinitions.values()];
  const versions = loaded?.versions ?? projectContexts.map((context) => context.task_version);
  const latestVersions = useMemo(() => {
    const latest = new Map<number, TaskVersion>();
    for (const version of versions) {
      if ((latest.get(version.task_definition_id)?.version_number ?? 0) < version.version_number) latest.set(version.task_definition_id, version);
    }
    return latest;
  }, [versions]);
  const documentTasks = (project.tasks ?? []).filter((task) => task.enabled);
  const loading = canReadAllTasks && !loaded && !error;
  const savedTaskCount = canReadAllTasks
    ? (loaded?.definitions.length ?? project.workflow_task_count ?? definitions.length) + documentTasks.length
    : definitions.length + documentTasks.length;
  const pinnedSourceIds = new Set(projectContexts.map((context) => context.round.dataset_version_id));
  const pinnedSource = pinnedSourceIds.size === 1 ? loaded?.datasetVersions.find((version) => pinnedSourceIds.has(version.id)) : undefined;
  const setupPath = (version: TaskVersion | undefined): string => {
    if (!version) return `/projects/${projectId}/tasks`;
    if (pinnedSource) return `/projects/${projectId}/data?tab=source&flow=import&datasetId=${pinnedSource.dataset_id}&datasetVersionId=${pinnedSource.id}&taskVersionId=${version.id}`;
    return `/projects/${projectId}/data?tab=source&taskVersionId=${version.id}`;
  };

  return <section className="aw-panel paper-task-inventory" aria-labelledby={headingId}>
    <header>
      <h2 id={headingId}>{canReadAllTasks ? "Annotation tasks" : "Assigned annotation tasks"} <span>{savedTaskCount}</span></h2>
    </header>
    {loading ? <p role="status">Loading saved annotation tasks…</p> : null}
    {error ? <p role="alert">{error} <button type="button" onClick={() => setAttempt((current) => current + 1)}>Retry loading tasks</button></p> : null}
    <div className="paper-task-list">
      {definitions.map((task) => {
        const version = latestVersions.get(task.id);
        const openRounds = projectContexts.filter((context) => context.task.id === task.id && context.round.status === "open");
        const needsPaperSetup = version && !paperAnnotationCompatibility(version).supported && version.task_kind === "token_labeling";
        return <article className="paper-task-row" key={`round-task-${task.id}`}>
          <div className="paper-task-summary">
            <h3>{task.name}</h3>
            <span className="paper-task-status" data-status={openRounds.length ? "ready" : "setup"}>{openRounds.length ? "Ready for annotation" : "Needs setup"}</span>
            {!openRounds.length ? <p>{needsPaperSetup ? "Set up this task for paper annotation." : version ? "No open annotation round is assigned to you for this task." : "Finish task setup to begin."}</p> : null}
          </div>
          <div className="paper-task-actions">{openRounds.map((context) => <PlatformRouteLink key={context.round.id} className="paper-task-action"
            href={`/my-work/rounds/${context.round.id}?view=annotate`} onNavigate={() => onNavigate(`/my-work/rounds/${context.round.id}?view=annotate`)}>
            Continue {task.name}{openRounds.length > 1 ? ` · Round ${context.round.sequence}` : ""}
          </PlatformRouteLink>)}
            {!openRounds.length && canManage ? <PlatformRouteLink className="paper-task-action" href={setupPath(version)} onNavigate={() => onNavigate(setupPath(version))}>Set up {task.name}</PlatformRouteLink> : null}
          </div>
        </article>;
      })}
      {documentTasks.map((task) => <article className="paper-task-row" key={`document-task-${task.id}`}>
        <div className="paper-task-summary"><h3>{task.display_name}</h3><span className="paper-task-status" data-status="ready">Document annotation task</span></div>
        <div className="paper-task-actions"><PlatformRouteLink className="paper-task-action" href={`/my-work?project=${projectId}&view=annotate&task=${task.id}`} onNavigate={() => onNavigate(`/my-work?project=${projectId}&view=annotate&task=${task.id}`)}>Open {task.display_name}</PlatformRouteLink></div>
      </article>)}
    </div>
    {!loading && !error && savedTaskCount === 0 ? <p>No annotation tasks have been saved for this project.</p> : null}
  </section>;
}
