import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import PubmedImportPanel from "@/components/PubmedImportPanel";
import { request } from "@/api/client";
import { createRound, createTaskWithVersion } from "./api";
import { PlatformRouteLink, PlatformSection } from "./components";
import { derivePaperEntityTaskVersionPayload, paperAnnotationCompatibility, paperTaskLabels, supportsPaperAnnotation } from "./paperTaskContracts";
import type { Dataset, DatasetVersion, PlatformProjectData, TaskVersion } from "./types";

type Purpose = "annotation" | "inference" | "training_source";
interface SourceSetupDraft {
  name?: string; documentIds?: number[]; taskName?: string; labels?: string;
  datasetId?: number | null; version?: DatasetVersion | null; taskDefinitionId?: number | null;
  taskVersion?: TaskVersion | null; roundId?: number | null; message?: string | null;
  selectedTaskId?: number; taskMode?: "classification" | "entities";
  roundTaskVersionId?: number; roundDatasetVersionId?: number;
}
function saveDraft(projectId: number, draft: SourceSetupDraft): void {
  try {
    sessionStorage.setItem(`al-medlit:source-setup:${projectId}:${draft.datasetId ?? "new"}`, JSON.stringify(draft));
  } catch { /* The saved backend resources remain the recovery source. */ }
}
export default function SourceDatasetSetup({ projectId, projectName, data, currentUserId,
  datasetId, datasetVersionId, initialTaskVersionId, canAnnotate, canInfer, onRefresh, onImported, onNavigate,
}: {
  projectId: number; projectName: string; data: PlatformProjectData; currentUserId: number | null;
  datasetId?: number | null; datasetVersionId?: number | null; canAnnotate: boolean; canInfer: boolean;
  initialTaskVersionId?: number | null;
  onRefresh: () => Promise<void>; onImported?: () => Promise<unknown>;
  onNavigate: (path: string, mode?: "push" | "replace") => void;
}): React.ReactElement {
  const storageKey = `al-medlit:source-setup:${projectId}:${datasetId ?? "new"}`;
  const restored = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey) ?? "null") as SourceSetupDraft | null; }
    catch { return null; }
  }, [storageKey]);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const existing = data.datasets.find((item) => item.id === datasetId);
  const previous = data.datasetVersions.filter((item) => item.dataset_id === datasetId)
    .sort((a, b) => b.version_number - a.version_number)[0];
  const [name, setName] = useState(restored?.name ?? existing?.name ?? `${projectName} papers`);
  const [purposes, setPurposes] = useState<Purpose[]>(existing?.purposes?.length ? existing.purposes : [canAnnotate ? "annotation" : "inference"]);
  const [documentIds, setDocumentIds] = useState<number[]>(restored?.version && !datasetVersionId ? [] : restored?.documentIds ?? []);
  const [version, setVersion] = useState<DatasetVersion | null>(data.datasetVersions.find((item) => item.id === datasetVersionId) ?? (datasetVersionId === restored?.version?.id ? restored?.version ?? null : null));
  const savedDatasetId = useRef<number | null>(datasetId ?? restored?.datasetId ?? null);
  const resumeAnnotation = Boolean(datasetVersionId && datasetVersionId === restored?.version?.id &&
    (!initialTaskVersionId || initialTaskVersionId === (restored?.selectedTaskId ?? restored?.taskVersion?.id)));
  const latestTasks = useMemo(() => {
    const latest = new Map<number, TaskVersion>();
    for (const task of data.taskVersions) {
      if (!latest.has(task.task_definition_id) || latest.get(task.task_definition_id)!.version_number < task.version_number) latest.set(task.task_definition_id, task);
    }
    return [...latest.values()];
  }, [data.taskVersions]);
  const [taskId, setTaskId] = useState(initialTaskVersionId ?? (resumeAnnotation ? restored?.selectedTaskId ?? restored?.taskVersion?.id : undefined) ?? latestTasks.find(supportsPaperAnnotation)?.id ?? 0);
  const [taskMode, setTaskMode] = useState<"classification" | "entities">(restored?.taskMode ?? "classification");
  const [taskName, setTaskName] = useState(restored?.taskName ?? "Paper relevance");
  const [labels, setLabels] = useState(restored?.labels ?? "");
  const [createdTask, setCreatedTask] = useState<TaskVersion | null>(resumeAnnotation ? restored?.taskVersion ?? null : null);
  const savedTaskDefinitionId = useRef<number | null>(resumeAnnotation ? restored?.taskDefinitionId ?? null : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(restored?.version && !datasetVersionId ? null : restored?.message ?? null);
  const continuationRef = useRef<HTMLDivElement>(null);
  const [focusContinuation, setFocusContinuation] = useState(false);
  const savedRoundId = useRef<number | null>(resumeAnnotation ? restored?.roundId ?? null : null);
  const savedRoundTaskVersionId = useRef(resumeAnnotation ? restored?.roundTaskVersionId ?? restored?.taskVersion?.id : undefined);
  const savedRoundDatasetVersionId = useRef(resumeAnnotation ? restored?.roundDatasetVersionId ?? restored?.version?.id : undefined);
  const previousRoute = useRef({ datasetId, datasetVersionId });
  const labelValues = [...new Set(labels.split(",").map((label) => label.trim()).filter(Boolean))];
  const selectedTask = createdTask && (!taskId || createdTask.id === taskId) ? createdTask : data.taskVersions.find((item) => item.id === taskId);
  const compatibility = selectedTask ? paperAnnotationCompatibility(selectedTask) : null;
  const availableTasks = latestTasks.filter(supportsPaperAnnotation);
  if (selectedTask && !availableTasks.some((item) => item.id === selectedTask.id)) availableTasks.push(selectedTask);
  const existingRound = selectedTask && version && currentUserId !== null ? data.rounds
    .filter((round) => round.status === "open" && round.dataset_version_id === version.id && round.task_version_id === selectedTask.id &&
      (round.open_to_all_annotators || round.annotator_user_ids.includes(currentUserId)))
    .sort((a, b) => b.id - a.id)[0] : undefined;
  const explicitTaskRoute = useRef(initialTaskVersionId);
  useEffect(() => {
    if (explicitTaskRoute.current === initialTaskVersionId) return;
    explicitTaskRoute.current = initialTaskVersionId;
    if (!initialTaskVersionId) return;
    setTaskId(initialTaskVersionId);
    setCreatedTask((current) => current?.id === initialTaskVersionId ? current : null);
    savedRoundId.current = null;
    savedTaskDefinitionId.current = null;
    setError(null);
  }, [initialTaskVersionId]);
  useEffect(() => {
    const priorRoute = previousRoute.current;
    previousRoute.current = { datasetId, datasetVersionId };
    const changedCollection = Boolean(priorRoute.datasetId && priorRoute.datasetId !== datasetId);
    const changedSavedVersion = Boolean(priorRoute.datasetVersionId && priorRoute.datasetVersionId !== datasetVersionId);
    if (!changedCollection && !changedSavedVersion) return;
    // Back/forward and "Add papers" can reuse this mounted component. A saved round
    // only belongs to its original source version, never to the new selection.
    const resumedVersion = datasetVersionId && restored?.version?.id === datasetVersionId ? restored.version : null;
    savedDatasetId.current = datasetId ?? null;
    savedTaskDefinitionId.current = resumedVersion ? restored?.taskDefinitionId ?? null : null;
    savedRoundId.current = resumedVersion ? restored?.roundId ?? null : null;
    savedRoundTaskVersionId.current = resumedVersion ? restored?.roundTaskVersionId ?? restored?.taskVersion?.id : undefined;
    savedRoundDatasetVersionId.current = resumedVersion ? restored?.roundDatasetVersionId ?? restored?.version?.id : undefined;
    setVersion(data.datasetVersions.find((item) => item.id === datasetVersionId) ?? resumedVersion);
    setCreatedTask(resumedVersion ? restored?.taskVersion ?? null : null);
    setDocumentIds(restored?.version && !datasetVersionId ? [] : restored?.documentIds ?? []);
    setMessage(restored?.version && !datasetVersionId ? null : restored?.message ?? null);
    setName(restored?.name ?? existing?.name ?? `${projectName} papers`);
    setError(null);
  }, [datasetId, datasetVersionId, data.datasetVersions, existing?.name, projectName, restored]);
  useEffect(() => {
    if (!datasetVersionId) return;
    const selected = data.datasetVersions.find((item) => item.id === datasetVersionId);
    if (selected) setVersion(selected);
  }, [data.datasetVersions, datasetVersionId]);
  useEffect(() => {
    saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: savedDatasetId.current, version, taskDefinitionId: savedTaskDefinitionId.current, taskVersion: createdTask, roundId: savedRoundId.current, message, selectedTaskId: taskId, taskMode, roundTaskVersionId: savedRoundTaskVersionId.current, roundDatasetVersionId: savedRoundDatasetVersionId.current });
  }, [projectId, storageKey, name, documentIds, taskName, labels, version, createdTask, message, taskId, taskMode]);
  useEffect(() => {
    if (focusContinuation) {
      continuationRef.current?.focus();
      setFocusContinuation(false);
    }
  }, [focusContinuation]);

  function selectAnnotationTask(nextTaskId: number): void {
    setTaskId(nextTaskId);
    setCreatedTask(null);
    savedRoundId.current = null;
    savedRoundTaskVersionId.current = undefined;
    savedRoundDatasetVersionId.current = undefined;
    savedTaskDefinitionId.current = null;
    setLabels("");
    setError(null);
    if (version) onNavigate(`/projects/${projectId}/data?tab=source&flow=import&datasetId=${version.dataset_id}&datasetVersionId=${version.id}${nextTaskId ? `&taskVersionId=${nextTaskId}` : ""}`, "replace");
  }

  async function enablePaperEntities(): Promise<void> {
    if (!selectedTask || !version) return;
    setBusy(true); setError(null);
    try {
      const payload = derivePaperEntityTaskVersionPayload(selectedTask, labelValues.length ? labelValues : undefined);
      // Refresh the immutable version list before creation, including after a lost response.
      const currentVersions = await request<TaskVersion[]>(`/tasks/versions?project_id=${projectId}&task_definition_id=${selectedTask.task_definition_id}`);
      const targetLabels = [...payload.label_rules.values as string[]].sort();
      const matching = currentVersions.find((task) => task.task_definition_id === selectedTask.task_definition_id &&
        task.annotation_ui.preset === "document_entities" && supportsPaperAnnotation(task) &&
        JSON.stringify(paperTaskLabels(task).sort()) === JSON.stringify(targetLabels));
      const task = matching ?? await request<TaskVersion>("/tasks/versions", { method: "POST", body: JSON.stringify(payload) });
      const success = `NER is ready for these papers: ${data.taskDefinitions.find((item) => item.id === task.task_definition_id)?.name ?? "Entity annotation"} · v${task.version_number}. The previous task version is preserved.`;
      saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: version.dataset_id, version, taskVersion: task, selectedTaskId: task.id, taskMode: "entities", message: success });
      if (!mounted.current) return;
      selectAnnotationTask(task.id);
      setCreatedTask(task);
      setMessage(success);
      try { await onRefresh(); } catch { setError("The NER task version was saved, but the project summary could not refresh. You can start annotation below."); }
    } catch (caught) { setError(caught instanceof Error ? caught.message : "The NER task version could not be created. Your source dataset is unchanged."); }
    finally { setBusy(false); }
  }

  async function saveDataset(): Promise<void> {
    setBusy(true); setError(null);
    try {
      if (!savedDatasetId.current) {
        const dataset = await request<Dataset>("/datasets", { method: "POST", body: JSON.stringify({
          project_id: projectId, name: name.trim(), source_type: "project_corpus", purposes,
        }) });
        savedDatasetId.current = dataset.id;
        if (!mounted.current) return;
        try { sessionStorage.removeItem(`al-medlit:source-setup:${projectId}:new`); } catch { /* Draft clearing is best effort. */ }
        saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: dataset.id, message, selectedTaskId: taskId, taskMode });
        onNavigate(`/projects/${projectId}/data?tab=source&flow=import&datasetId=${dataset.id}${taskId ? `&taskVersionId=${taskId}` : ""}`, "replace");
      }
      const prior = Array.isArray(previous?.provenance.source_document_ids) ? previous.provenance.source_document_ids.filter((id): id is number => typeof id === "number") : [];
      const selected = [...new Set([...prior, ...documentIds])];
      const created = await request<DatasetVersion>(`/projects/${projectId}/datasets/${savedDatasetId.current}/versions/project-corpus`, {
        method: "POST", body: JSON.stringify({ document_ids: selected }),
      });
      const success = `Source dataset saved: ${name} · v${created.version_number} · ${created.item_count} papers. Choose your next action below.`;
      savedRoundId.current = null;
      savedTaskDefinitionId.current = null;
      // Persist the completed step before navigation or a parent refresh can remount us.
      saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: created.dataset_id, version: created, message: success, selectedTaskId: taskId, taskMode });
      if (!mounted.current) return;
      setVersion(created);
      setCreatedTask(null);
      setMessage(success);
      setFocusContinuation(true);
      onNavigate(`/projects/${projectId}/data?tab=source&flow=import&datasetId=${created.dataset_id}&datasetVersionId=${created.id}${taskId ? `&taskVersionId=${taskId}` : ""}`, "replace");
      try { await onRefresh(); } catch { setError("Your source dataset was saved, but the project summary could not refresh. You can continue below or reload the page."); }
    } catch (caught) { setError(caught instanceof Error ? caught.message : "The dataset could not be saved. Imported papers are retained; retry this step."); }
    finally { setBusy(false); }
  }

  async function startAnnotation(): Promise<void> {
    if (!version || currentUserId === null) return;
    if (taskId && !selectedTask) { setError("This task version is not available in the current project. Refresh the project and choose it again."); return; }
    if (selectedTask && !compatibility?.supported) {
      setError(compatibility?.reason ?? "Choose a compatible annotation task."); return;
    }
    if (existingRound) { onNavigate(`/my-work/rounds/${existingRound.id}?view=annotate`); return; }
    setBusy(true); setError(null);
    try {
      let task = selectedTask;
      if (!task) {
        task = await createTaskWithVersion(projectId, {
          key: `paper_${taskMode}_${version.id}_${taskName.toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 60) || "task"}`, name: taskName.trim(),
          description: taskMode === "entities" ? "Named entities in paper text" : "Paper classification",
          taskKind: taskMode === "entities" ? "span_extraction" : "classification", labelValues,
          ...(taskMode === "entities" ? { annotationMode: "paper_entities" as const } : {}),
        }, (definition) => {
          savedTaskDefinitionId.current = definition.id;
          saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: savedDatasetId.current, version, taskDefinitionId: definition.id, taskMode, selectedTaskId: 0 });
        }, savedTaskDefinitionId.current);
        if (!mounted.current) return;
        saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: savedDatasetId.current, version, taskDefinitionId: savedTaskDefinitionId.current, taskVersion: task, taskMode, selectedTaskId: task.id });
        setCreatedTask(task);
        setTaskId(task.id);
      }
      const savedRound = data.rounds.find((round) => round.id === savedRoundId.current);
      if (savedRoundId.current && savedRoundTaskVersionId.current === task.id && savedRoundDatasetVersionId.current === version.id &&
        (!savedRound || savedRound.status === "draft")) {
        await request(`/rounds/${savedRoundId.current}/transition?project_id=${projectId}`, { method: "POST", body: JSON.stringify({ status: "open" }) });
        if (mounted.current) onNavigate(`/my-work/rounds/${savedRoundId.current}?view=annotate`); return;
      }
      const round = await createRound(projectId, {
        name: `${name} · ${data.taskDefinitions.find((item) => item.id === task.task_definition_id)?.name ?? taskName.trim()}`, datasetVersionId: version.id, taskVersionId: task.id,
        cycleId: null, splitMapId: null, guidelineRevisionId: null, feedbackSetVersionId: null,
        assistancePolicy: "blind", reannotationMode: "full_dataset", selectionStrategy: "all", selectionLimit: version.item_count,
        annotatorUserIds: [currentUserId], openToAllAnnotators: false, reason: "Initial annotation",
      }, (created) => {
        savedRoundId.current = created.id;
        savedRoundTaskVersionId.current = task.id;
        savedRoundDatasetVersionId.current = version.id;
        saveDraft(projectId, { name, documentIds, taskName, labels, datasetId: savedDatasetId.current, version, taskDefinitionId: savedTaskDefinitionId.current, taskVersion: task, roundId: created.id, selectedTaskId: task.id, taskMode, roundTaskVersionId: task.id, roundDatasetVersionId: version.id });
      });
      if (mounted.current) onNavigate(`/my-work/rounds/${round.id}?view=annotate`);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Annotation setup failed. Your source dataset remains saved."); }
    finally { setBusy(false); }
  }

  return <PlatformSection title={version ? "Choose the next action" : existing ? "Add papers to this collection" : "Import PubMed papers"}
    description="Import papers once. Use a saved source dataset for annotation, predictions, or training preparation.">
    <p><PlatformRouteLink href={`/projects/${projectId}/data?tab=source`} onNavigate={() => onNavigate(`/projects/${projectId}/data?tab=source`)}>Back to source datasets</PlatformRouteLink></p>
    <ol className="platform-page-actions" aria-label="Source setup steps"><li aria-current={!version && !documentIds.length ? "step" : undefined}>1. Import papers</li><li aria-current={!version && documentIds.length ? "step" : undefined}>2. Save collection</li><li aria-current={version ? "step" : undefined}>3. Choose next action</li></ol>
    {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
    {message ? <p role="status" aria-label="Source collection status">{message}</p> : null}
    {!version ? <>
      {documentIds.length > 0 ? <div ref={continuationRef} tabIndex={-1}>
        <h3>Save this collection</h3>
        <p>{documentIds.length} papers are ready. Save a named source dataset to choose an annotation task or run predictions.</p>
        {previous ? <p>The {previous.item_count} papers in the previous version will also be retained.</p> : null}
        <Button label={busy ? "Saving dataset…" : existing ? "Save new source version" : "Create source dataset"} variant="primary" isDisabled={busy || !name.trim()} onClick={() => void saveDataset()} />
      </div> : null}
      <div className="platform-dialog-form">
        <label><span>Collection name</span><input value={name} disabled={Boolean(existing) || busy} onChange={(event) => setName(event.target.value)} required maxLength={255} /></label>
        <fieldset className="platform-fieldset"><legend>Use this source for</legend>
          {([["annotation", "Annotation", canAnnotate], ["inference", "Predictions", canInfer], ["training_source", "Training preparation", data.projectModules.effective.includes("train")]] as const).filter(([, , enabled]) => enabled).map(([purpose, label]) => <label className="platform-checkbox-row" key={purpose}>
            <input type="checkbox" checked={purposes.includes(purpose)} disabled={Boolean(existing)} onChange={(event) => setPurposes((current) => event.target.checked ? [...current, purpose] : current.filter((item) => item !== purpose))} />{label}
          </label>)}
        </fieldset>
      </div>
      <PubmedImportPanel projectId={projectId} onImported={async (result) => {
        if (!mounted.current) return;
        const ids = [...result.created, ...result.skipped].map((item) => item.document_id).filter((id): id is number => typeof id === "number");
        const selected = [...new Set([...documentIds, ...ids])];
        const createdCount = result.created.length;
        const reusedCount = result.skipped.filter((item) => typeof item.document_id === "number").length;
        const imported = `${createdCount} ${createdCount === 1 ? "paper" : "papers"} imported; ${reusedCount} existing ${reusedCount === 1 ? "paper" : "papers"} reused. ${selected.length ? "Save this collection to continue." : "No usable papers were selected."}`;
        saveDraft(projectId, { name, documentIds: selected, taskName, labels, datasetId: savedDatasetId.current, version, message: imported, selectedTaskId: taskId, taskMode });
        setDocumentIds(selected);
        setMessage(imported);
        setFocusContinuation(selected.length > 0);
        await onImported?.();
      }} />
    </> : <>
      <div ref={continuationRef} tabIndex={-1}><h3>Source dataset saved</h3>
        <p><strong>{name}</strong> · v{version.version_number} · {version.item_count} papers</p>
      </div>
      {canAnnotate ? <div className="platform-dialog-form">
        <label><span>Annotation task</span><select value={taskId} disabled={busy} onChange={(event) => selectAnnotationTask(Number(event.target.value))}>
          <option value={0}>Create a new annotation task</option>
          {availableTasks.map((task) => <option key={task.id} value={task.id}>{data.taskDefinitions.find((item) => item.id === task.task_definition_id)?.name ?? (task.task_kind === "classification" ? "Classification" : "Named entities")} · v{task.version_number}</option>)}
        </select></label>
        {!taskId ? <>
          <label><span>Task type</span><select value={taskMode} disabled={busy} onChange={(event) => {
            const nextMode = event.target.value as "classification" | "entities";
            setTaskMode(nextMode); setCreatedTask(null); savedRoundId.current = null; savedTaskDefinitionId.current = null; setLabels("");
            if (taskName === "Paper relevance" || taskName === "Named entities") setTaskName(nextMode === "entities" ? "Named entities" : "Paper relevance");
          }}><option value="classification">Document classification</option><option value="entities">Named entities in papers (NER)</option></select></label>
          <label><span>Task name</span><input value={taskName} disabled={busy} onChange={(event) => setTaskName(event.target.value)} required /></label>
          <label><span>Labels</span><input value={labels} disabled={busy} onChange={(event) => setLabels(event.target.value)} placeholder={taskMode === "entities" ? "For example: Drug, Gene, Disease" : "For example: relevant, not relevant"} required /></label>
          <p>{taskMode === "entities" ? "Enter one or more entity labels, separated by commas. Select spans in the paper text to annotate these entities." : "Enter at least two distinct labels, separated by commas. Choose labels that match your research question."}</p>
        </> : null}
        {selectedTask && compatibility?.supported ? <p>Task type: {compatibility.editor === "entities" ? "Named entities in papers (NER)" : "Document classification"}. Labels: {paperTaskLabels(selectedTask).join(", ") || "Defined by this task's output schema"}.</p> : null}
        {selectedTask && !compatibility?.supported ? <div>
          <p role="status">{compatibility?.reason}</p>
          {selectedTask.task_kind === "token_labeling" || selectedTask.task_kind === "span_extraction" ? <>
            <label><span>Entity labels (optional override)</span><input value={labels} disabled={busy} onChange={(event) => setLabels(event.target.value)} placeholder="Use the existing task labels, or enter Drug, Gene, Disease" /></label>
            <Button label={busy ? "Enabling NER…" : "Enable NER for these papers"} variant="secondary" isDisabled={busy} onClick={() => void enablePaperEntities()} />
          </> : null}
        </div> : null}
        <p>{existingRound ? `Continue your open round: ${existingRound.name}.` : "This starts a blind annotation round assigned to you. Predictions are not shown."}</p>
        <Button label={busy ? "Starting annotation…" : existingRound ? "Continue annotation" : "Start annotating"} variant="primary" isDisabled={busy || !version.item_count || currentUserId === null || Boolean(taskId && !compatibility?.supported) || (!taskId && (!taskName.trim() || labelValues.length < (taskMode === "entities" ? 1 : 2)))} onClick={() => void startAnnotation()} />
      </div> : null}
      {canInfer ? <p><PlatformRouteLink href={`/projects/${projectId}/inference?datasetVersionId=${version.id}`} onNavigate={() => onNavigate(`/projects/${projectId}/inference?datasetVersionId=${version.id}`)}>Run predictions on this source</PlatformRouteLink></p> : null}
      {data.projectModules.effective.includes("train") ? <p><PlatformRouteLink href={`/training/data?projectId=${projectId}&flow=prepare&datasetVersionId=${version.id}`} onNavigate={() => onNavigate(`/training/data?projectId=${projectId}&flow=prepare&datasetVersionId=${version.id}`)}>Prepare training data</PlatformRouteLink></p> : null}
    </>}
  </PlatformSection>;
}
