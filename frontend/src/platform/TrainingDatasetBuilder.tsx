import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { PlatformEmpty, PlatformSection } from "./components";
import { prepareTrainingDataset, previewTrainingDataset, type TrainingPreparationDraft, type TrainingPreview, type TrainingSource } from "./trainingPreparationApi";
import type { PlatformProjectData, TrainingDatasetVersion } from "./types";

export interface TrainingDatasetBuilderProps {
  projectId: number; data: PlatformProjectData; initialDatasetId?: number | null;
  initialDatasetVersionId?: number | null; initialTaskVersionId?: number | null; initialLabelSetVersionId?: number | null;
  parent?: TrainingDatasetVersion; onImport: () => void; onDefineTask?: () => void;
  onCreated: (version: TrainingDatasetVersion) => Promise<void>; onCancel: () => void;
}

export function inheritedTrainingSource(source: TrainingSource): TrainingSource {
  return { dataset_version_id: source.dataset_version_id, input_mapping: source.input_mapping ?? { text: "text" },
    ...(source.split_map_id ? { split_map_id: source.split_map_id } : {}),
    ...(source.label_set_version_id ? { label_set_version_id: source.label_set_version_id } : source.annotation_round_id ? {
      annotation_round_id: source.annotation_round_id, ...(source.submission_ids ? { submission_ids: source.submission_ids } : {}),
    } : { label_field: source.label_field ?? "label" }),
  };
}

function taskInputMapping(data: PlatformProjectData, taskId: number): Record<string, string> {
  const task = data.taskVersions.find((item) => item.id === taskId);
  const fields = Object.keys((task?.input_schema.properties as Record<string, unknown> | undefined) ?? { text: {} });
  return Object.fromEntries((fields.length ? fields : ["text"]).map((field) => [field, field]));
}

export default function TrainingDatasetBuilder({ projectId, data, initialDatasetId, initialDatasetVersionId, initialTaskVersionId, initialLabelSetVersionId, parent, onImport, onDefineTask, onCreated, onCancel }: TrainingDatasetBuilderProps): React.ReactElement {
  const storageKey = `al-medlit:training-preparation:${projectId}:${parent?.id ?? "new"}:${initialDatasetVersionId ?? initialDatasetId ?? "all"}`;
  const restored = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey) ?? "null") as { draft?: TrainingPreparationDraft; preview?: TrainingPreview; attempt?: { signature: string; key: string }; startChoice?: "annotations" | "external" | "combine" } | null; } catch { return null; }
  }, [storageKey]);
  const sourceVersions = data.datasetVersions.filter((version) => version.provenance?.ingestion !== "training_preparation_v1");
  const initialVersion = initialDatasetVersionId ?? sourceVersions.filter((version) => !initialDatasetId || version.dataset_id === initialDatasetId).sort((a, b) => b.version_number - a.version_number)[0]?.id ?? 0;
  const initialTask = parent?.task_version_id ?? initialTaskVersionId ?? data.taskVersions[0]?.id ?? 0;
  const [name, setName] = useState(restored?.draft?.name ?? parent?.name ?? "");
  const [taskId, setTaskId] = useState(restored?.draft?.task_version_id ?? initialTask);
  const [sources, setSources] = useState<TrainingSource[]>(() => {
    if (restored?.draft?.sources) return restored.draft.sources;
    const inherited = parent?.preparation_manifest?.sources;
    if (Array.isArray(inherited) && inherited.length) return (inherited as TrainingSource[]).map(inheritedTrainingSource);
    const layer = data.labelSets.find((item) => item.id === initialLabelSetVersionId) ?? data.labelSets.find((item) => item.dataset_version_id === initialVersion && item.task_version_id === initialTask);
    const round = data.rounds.find((item) => item.dataset_version_id === initialVersion && item.task_version_id === initialTask);
    return [{ dataset_version_id: initialVersion, input_mapping: taskInputMapping(data, initialTask), ...(layer ? { label_set_version_id: layer.id } : round ? { annotation_round_id: round.id } : { label_field: "label" }) }];
  });
  const [startChoice, setStartChoice] = useState<"annotations" | "external" | "combine">(restored?.startChoice ?? (parent ? "combine" : initialLabelSetVersionId || data.rounds.some((round) => round.dataset_version_id === initialVersion) ? "annotations" : "external"));
  const [trainPercent, setTrainPercent] = useState(restored?.draft?.train_percent ?? 80);
  const [validationPercent, setValidationPercent] = useState(restored?.draft?.validation_percent ?? 10);
  const [preview, setPreview] = useState<TrainingPreview | null>(() => restored?.attempt?.signature === JSON.stringify(restored?.draft) ? restored?.preview ?? null : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<TrainingDatasetVersion | null>(null);
  const attempt = useRef<{ signature: string; key: string } | null>(restored?.attempt ?? null);
  const requestId = useRef(0);
  const task = data.taskVersions.find((item) => item.id === taskId);
  const taskFields = Object.keys((task?.input_schema.properties as Record<string, unknown> | undefined) ?? { text: {} });
  const draft = useMemo<TrainingPreparationDraft>(() => ({ name: name.trim(), task_version_id: taskId, sources,
    train_percent: trainPercent, validation_percent: validationPercent, seed: 42,
    ...(parent?.training_dataset_id ? { training_dataset_id: parent.training_dataset_id, parent_version_id: parent.id } : {}),
  }), [name, taskId, sources, trainPercent, validationPercent, parent]);
  const signature = JSON.stringify(draft);
  const signatureRef = useRef(signature); signatureRef.current = signature;
  const previousSignature = useRef(signature);
  useEffect(() => {
    if (signature === previousSignature.current) return;
    previousSignature.current = signature;
    setPreview(null); setError(null); requestId.current += 1;
  }, [signature]);
  useEffect(() => () => { requestId.current += 1; }, []);
  useEffect(() => {
    if (saved) return;
    try { sessionStorage.setItem(storageKey, JSON.stringify({ draft, preview, attempt: attempt.current, startChoice })); } catch { /* Backend versions remain available if session storage is unavailable. */ }
  }, [storageKey, draft, preview, saved, startChoice]);

  function updateSource(index: number, change: Partial<TrainingSource>): void {
    setSources((current) => current.map((source, row) => row === index ? { ...source, ...change } : source));
  }
  function labelChoice(index: number, choice: string): void {
    const [kind, value] = choice.split(":");
    setSources((current) => current.map((source, row) => {
      if (row !== index) return source;
      const next: TrainingSource = { dataset_version_id: source.dataset_version_id, input_mapping: source.input_mapping, split_map_id: source.split_map_id };
      return { ...next, ...(kind === "round" ? { annotation_round_id: Number(value) } : kind === "layer" ? { label_set_version_id: Number(value) } : { label_field: "label" }) };
    }));
  }
  function annotationSource(selectedTaskId: number): TrainingSource[] {
    const mapping = taskInputMapping(data, selectedTaskId);
    const round = data.rounds.find((item) => item.task_version_id === selectedTaskId);
    const layer = data.labelSets.find((item) => item.task_version_id === selectedTaskId && item.source_kind !== "imported" && item.composition_policy !== "exclude");
    return round ? [{ dataset_version_id: round.dataset_version_id, input_mapping: mapping, annotation_round_id: round.id }] : layer ? [{ dataset_version_id: layer.dataset_version_id, input_mapping: mapping, label_set_version_id: layer.id }] : [];
  }
  function chooseStart(choice: "annotations" | "external" | "combine"): void {
    setStartChoice(choice);
    const mapping = taskInputMapping(data, taskId);
    if (choice === "combine") { if (sources.length < 2) setSources((current) => [...current, { dataset_version_id: 0, input_mapping: mapping, label_field: "label" }]); return; }
    if (choice === "annotations") {
      setSources(annotationSource(taskId));
      return;
    }
    const external = sourceVersions.find((version) => data.datasets.some((dataset) => dataset.id === version.dataset_id && ["upload", "public_registry"].includes(dataset.source_type)));
    setSources([{ dataset_version_id: external?.id ?? 0, input_mapping: mapping, label_field: "label" }]);
    if (!external) onImport();
  }
  async function review(): Promise<void> {
    const id = ++requestId.current; setBusy(true); setError(null);
    try { const result = await previewTrainingDataset(projectId, draft); if (id === requestId.current && signature === signatureRef.current) setPreview(result); }
    catch (caught) { if (id === requestId.current) setError(caught instanceof Error ? caught.message : "Could not preview training data."); }
    finally { if (id === requestId.current) setBusy(false); }
  }
  async function prepare(): Promise<void> {
    if (!preview?.ready) return;
    const id = ++requestId.current; setBusy(true); setError(null);
    if (attempt.current?.signature !== signature) attempt.current = { signature, key: crypto.randomUUID() };
    try { sessionStorage.setItem(storageKey, JSON.stringify({ draft, preview, attempt: attempt.current, startChoice })); } catch { /* Keep the current attempt in memory. */ }
    try {
      const result = await prepareTrainingDataset(projectId, draft, preview, attempt.current.key);
      if (id !== requestId.current) return;
      setSaved(result.training_dataset_version);
      try { sessionStorage.removeItem(storageKey); } catch { /* The saved version is authoritative. */ }
      await onCreated(result.training_dataset_version);
    } catch (caught) { if (id === requestId.current) setError(caught instanceof Error ? caught.message : "Could not save training dataset. Retry with the same reviewed selection."); }
    finally { if (id === requestId.current) setBusy(false); }
  }
  if (saved) return <PlatformSection title="Training dataset saved"><p role="status">{saved.name} · v{saved.version_number ?? 1} is ready. Its source records and labels remain unchanged.</p><Button label="Back to training datasets" onClick={onCancel} /></PlatformSection>;
  return <PlatformSection title={parent ? `Create a new version of ${parent.name}` : "Create training dataset"} description="Use submitted annotations, external labeled data, or combine several sources. Preview the records and splits before saving.">
    {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
    {!sourceVersions.length ? <PlatformEmpty title="Import a source dataset first" detail="Upload labeled records or import papers for annotation." actionLabel="Import source data" onAction={onImport} /> : null}
    {!data.taskVersions.length ? <PlatformEmpty title="Define the prediction task" detail="A shared task defines the input fields and allowed labels for every source." actionLabel={onDefineTask ? "Define task" : undefined} onAction={onDefineTask} /> : null}
    <form className="platform-dialog-form" onSubmit={(event) => { event.preventDefault(); void review(); }}>
      {!parent ? <fieldset className="platform-fieldset" disabled={busy}><legend>Start with</legend><div className="platform-option-list">
        {([["annotations", "Use project annotations"], ["external", "Import external labeled data"], ["combine", "Combine sources"]] as const).map(([choice, label]) => <label key={choice}><input type="radio" name="training-start" value={choice} checked={startChoice === choice} onChange={() => chooseStart(choice)} /><span>{label}</span></label>)}
      </div></fieldset> : null}
      <label><span>Training dataset name</span><input required value={name} onChange={(event) => setName(event.target.value)} disabled={busy || Boolean(parent)} maxLength={255} /></label>
      <label><span>Task</span><select required value={taskId} disabled={busy || Boolean(parent)} onChange={(event) => { const id = Number(event.target.value); setTaskId(id); setSources((current) => startChoice === "annotations" ? annotationSource(id) : current.map((source) => ({ dataset_version_id: source.dataset_version_id, input_mapping: taskInputMapping(data, id), label_field: "label" }))); }}>
        <option value={0}>Select a task</option>{data.taskVersions.map((item) => <option key={item.id} value={item.id}>{data.taskDefinitions.find((definition) => definition.id === item.task_definition_id)?.name ?? item.task_kind} · v{item.version_number}</option>)}
      </select></label>
      {startChoice === "annotations" && !sources.length ? <p role="status">No submitted project annotations are available for this task. Annotate a source dataset first, or choose external labeled data.</p> : null}
      {sources.map((source, index) => <fieldset className="platform-fieldset" key={index} disabled={busy}>
        <legend>Source {index + 1}</legend>
        <label><span>Source dataset version</span><select required value={source.dataset_version_id} onChange={(event) => updateSource(index, { dataset_version_id: Number(event.target.value), label_field: "label", label_set_version_id: undefined, annotation_round_id: undefined, submission_ids: undefined, split_map_id: undefined })}>
          <option value={0}>Select a source</option>{sourceVersions.filter((version) => version.item_count > 0).map((version) => <option key={version.id} value={version.id}>{data.datasets.find((dataset) => dataset.id === version.dataset_id)?.name ?? "Source"} · v{version.version_number} · {version.item_count} records</option>)}
        </select></label>
        <div className="platform-form-grid">{taskFields.map((field) => <label key={field}><span>Source field for {field}</span><input required value={source.input_mapping[field] ?? field} onChange={(event) => updateSource(index, { input_mapping: { ...source.input_mapping, [field]: event.target.value } })} /></label>)}</div>
        <label><span>Labels from</span><select value={source.annotation_round_id ? `round:${source.annotation_round_id}` : source.label_set_version_id ? `layer:${source.label_set_version_id}` : "field"} onChange={(event) => labelChoice(index, event.target.value)}>
          <option value="field">A field in the imported dataset</option>
          {data.labelSets.filter((layer) => layer.dataset_version_id === source.dataset_version_id && layer.task_version_id === taskId && layer.composition_policy !== "exclude").map((layer) => <option key={`layer:${layer.id}`} value={`layer:${layer.id}`}>{layer.name} · v{layer.version_number} · {layer.source_kind}</option>)}
          {data.rounds.filter((round) => round.dataset_version_id === source.dataset_version_id && round.task_version_id === taskId).map((round) => <option key={`round:${round.id}`} value={`round:${round.id}`}>Submitted annotations: {round.name}</option>)}
        </select></label>
        {source.label_field !== undefined ? <label><span>Label field</span><input required value={source.label_field} onChange={(event) => updateSource(index, { label_field: event.target.value })} /></label> : null}
        {source.annotation_round_id ? <p>Only finalized submissions enter the snapshot. You can continue annotating this source after creating the training dataset.</p> : null}
        <label><span>Existing evaluation splits</span><select value={source.split_map_id ?? 0} onChange={(event) => updateSource(index, { split_map_id: Number(event.target.value) || undefined })}><option value={0}>Use protected splits when present; split new records below</option>{data.splitMaps.filter((split) => split.dataset_version_id === source.dataset_version_id).map((split) => <option key={split.id} value={split.id}>{split.name}</option>)}</select></label>
        {sources.length > 1 ? <Button label={`Remove source ${index + 1}`} variant="ghost" onClick={() => setSources((current) => current.filter((_, row) => row !== index))} /> : null}
      </fieldset>)}
      <div className="platform-row-actions"><Button label="Add another source" isDisabled={busy} onClick={() => setSources((current) => [...current, { dataset_version_id: 0, input_mapping: taskInputMapping(data, taskId), label_field: "label" }])} /><Button label="Import external data" isDisabled={busy} onClick={onImport} /></div>
      <div className="platform-form-grid"><label><span>Train percent for new groups</span><input type="number" min={1} max={98} required value={trainPercent} disabled={busy} onChange={(event) => setTrainPercent(Number(event.target.value))} /></label><label><span>Validation percent for new groups</span><input type="number" min={1} max={98} required value={validationPercent} disabled={busy} onChange={(event) => setValidationPercent(Number(event.target.value))} /></label></div>
      <p>Test: {100 - trainPercent - validationPercent}%. Existing protected evaluation groups stay protected. Unsubmitted or unlabeled records are excluded.</p>
      <div className="platform-row-actions"><Button label={busy ? "Checking…" : "Preview training dataset"} type="submit" variant="primary" isDisabled={busy || !taskId || !sources.length || sources.some((source) => !source.dataset_version_id) || trainPercent + validationPercent >= 100} /><Button label="Cancel" isDisabled={busy} onClick={onCancel} /></div>
    </form>
    {preview ? <section aria-label="Training dataset preview"><h3>Review included data</h3>
      <p role="status">{preview.item_count} records · {preview.group_count} groups · {preview.excluded_unlabeled_count} unlabeled records excluded · {preview.duplicate_count} duplicates removed</p>
      <p>{Object.entries(preview.split_counts).map(([split, count]) => `${split}: ${count}`).join(" · ")}</p>
      <ul>{preview.source_counts.map((source, index) => <li key={`${source.dataset_version_id}:${index}`}>Source {index + 1}: {source.total_count} records, {source.labeled_count} labeled, {source.excluded_unlabeled_count} unlabeled excluded</li>)}</ul>
      {preview.issues.length ? <ul>{preview.issues.map((issue, index) => <li key={index}>{issue.source_index !== undefined ? `Source ${issue.source_index + 1}: ` : ""}{issue.message}</li>)}</ul> : null}
      <Button label={busy ? "Saving…" : parent ? "Create new training version" : "Create training dataset"} variant="primary" isDisabled={busy || !preview.ready} onClick={() => void prepare()} />
    </section> : null}
  </PlatformSection>;
}
