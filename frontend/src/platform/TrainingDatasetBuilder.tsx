import { NativeInput, NativeSelect } from "@/components/NativeField";
import ValidatedForm from "@/components/ValidatedForm";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { PlatformEmpty, PlatformSection, formatCount } from "./components";
import { prepareTrainingDataset, previewTrainingDataset, type TrainingPreparationDraft, type TrainingPreview, type TrainingSource } from "./trainingPreparationApi";
import type { PlatformProjectData, TrainingDatasetVersion } from "./types";
import { getSessionGeneration } from "@/auth/session";

export interface TrainingDatasetBuilderProps {
  projectId: number; data: PlatformProjectData; initialDatasetId?: number | null;
  initialDatasetVersionId?: number | null; initialTaskVersionId?: number | null; initialLabelSetVersionId?: number | null;
  parent?: TrainingDatasetVersion; onImport: () => void; onDefineTask?: () => void;
  onCreated: (version: TrainingDatasetVersion) => Promise<void>; onCancel: () => void;
  canUseRoundAnnotations?: boolean;
}

export function inheritedTrainingSource(source: TrainingSource): TrainingSource {
  return { dataset_version_id: source.dataset_version_id, input_mapping: source.input_mapping ?? { text: "text" },
    ...(source.split_map_id ? { split_map_id: source.split_map_id } : {}),
    ...(source.label_set_version_id ? { label_set_version_id: source.label_set_version_id } : source.annotation_round_id ? {
      annotation_round_id: source.annotation_round_id, ...(source.submission_ids ? { submission_ids: source.submission_ids } : {}),
    } : { label_field: source.label_field ?? "" }),
  };
}

function taskInputMapping(data: PlatformProjectData, taskId: number): Record<string, string> {
  const task = data.taskVersions.find((item) => item.id === taskId);
  const fields = Object.keys((task?.input_schema.properties as Record<string, unknown> | undefined) ?? { text: {} });
  return Object.fromEntries((fields.length ? fields : ["text"]).map((field) => [field, field]));
}

export default function TrainingDatasetBuilder({ projectId, data, initialDatasetId, initialDatasetVersionId, initialTaskVersionId, initialLabelSetVersionId, parent, onImport, onDefineTask, onCreated, onCancel, canUseRoundAnnotations = true }: TrainingDatasetBuilderProps): React.ReactElement {
  const storageKey = `al-medlit:training-preparation:${projectId}:${parent?.id ?? "new"}:${initialDatasetVersionId ?? initialDatasetId ?? "all"}`;
  const restored = useMemo(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey) ?? "null") as { draft?: TrainingPreparationDraft; preview?: TrainingPreview; attempt?: { signature: string; key: string }; startChoice?: "annotations" | "external" | "combine"; saved?: TrainingDatasetVersion } | null; } catch { return null; }
  }, [storageKey]);
  const sourceVersions = data.datasetVersions.filter((version) => version.provenance?.ingestion !== "training_preparation_v1");
  const initialVersion = initialDatasetVersionId ?? sourceVersions.filter((version) => !initialDatasetId || version.dataset_id === initialDatasetId).sort((a, b) => b.version_number - a.version_number)[0]?.id ?? 0;
  const initialTask = parent?.task_version_id ?? initialTaskVersionId ?? [...data.taskVersions].sort((a, b) => b.id - a.id)[0]?.id ?? 0;
  const [name, setName] = useState(restored?.draft?.name ?? parent?.name ?? "");
  const [taskId, setTaskId] = useState(restored?.draft?.task_version_id ?? initialTask);
  function importedLabelField(versionId: number): string {
    const properties = data.datasetVersions.find((version) => version.id === versionId)?.data_schema?.properties;
    return properties && typeof properties === "object" && "label" in properties ? "label" : "";
  }
  function annotationSource(selectedTaskId: number, versionId?: number): TrainingSource[] {
    const mapping = taskInputMapping(data, selectedTaskId);
    const eligibleVersion = (id: number): boolean => versionId !== undefined ? id === versionId :
      initialDatasetVersionId ? id === initialDatasetVersionId :
      !initialDatasetId || sourceVersions.some((version) => version.id === id && version.dataset_id === initialDatasetId);
    const layer = data.labelSets.filter((item) => item.task_version_id === selectedTaskId && eligibleVersion(item.dataset_version_id) && item.source_kind !== "imported" && item.composition_policy !== "exclude")
      .sort((a, b) => b.id - a.id)[0];
    const round = canUseRoundAnnotations ? data.rounds.filter((item) => item.task_version_id === selectedTaskId && eligibleVersion(item.dataset_version_id) && (item.status === "open" || item.status === "closed"))
      .sort((a, b) => b.id - a.id)[0] : undefined;
    return layer ? [{ dataset_version_id: layer.dataset_version_id, input_mapping: mapping, label_set_version_id: layer.id }] :
      round ? [{ dataset_version_id: round.dataset_version_id, input_mapping: mapping, annotation_round_id: round.id }] : [];
  }
  const [sources, setSources] = useState<TrainingSource[]>(() => {
    if (restored?.draft?.sources) return restored.draft.sources.filter((source) => canUseRoundAnnotations || !source.annotation_round_id);
    const inherited = parent?.preparation_manifest?.sources;
    if (Array.isArray(inherited) && inherited.length) return (inherited as TrainingSource[]).map(inheritedTrainingSource).filter((source) => canUseRoundAnnotations || !source.annotation_round_id);
    const layer = data.labelSets.find((item) => item.id === initialLabelSetVersionId && item.dataset_version_id === initialVersion && item.task_version_id === initialTask && item.composition_policy !== "exclude");
    if (layer) return [{ dataset_version_id: initialVersion, input_mapping: taskInputMapping(data, initialTask), label_set_version_id: layer.id }];
    const annotations = annotationSource(initialTask, initialVersion);
    return annotations.length ? annotations : [{ dataset_version_id: initialVersion, input_mapping: taskInputMapping(data, initialTask), label_field: importedLabelField(initialVersion) }];
  });
  const [startChoice, setStartChoice] = useState<"annotations" | "external" | "combine">(restored?.startChoice ?? (parent ? "combine" : sources.some((source) => source.label_set_version_id || source.annotation_round_id) ? "annotations" : "external"));
  const [trainPercent, setTrainPercent] = useState(restored?.draft?.train_percent ?? 80);
  const [validationPercent, setValidationPercent] = useState(restored?.draft?.validation_percent ?? 10);
  const [preview, setPreview] = useState<TrainingPreview | null>(() => restored?.attempt?.signature === JSON.stringify(restored?.draft) ? restored?.preview ?? null : null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showValidation, setShowValidation] = useState(false);
  const [saved, setSaved] = useState<TrainingDatasetVersion | null>(restored?.saved ?? null);
  const attempt = useRef<{ signature: string; key: string } | null>(restored?.attempt ?? null);
  const requestId = useRef(0);
  const sessionGeneration = useRef(getSessionGeneration()).current;
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
    if (saved || sessionGeneration !== getSessionGeneration()) return;
    try { sessionStorage.setItem(storageKey, JSON.stringify({ draft, preview, attempt: attempt.current, startChoice })); } catch { /* Backend versions remain available if session storage is unavailable. */ }
  }, [storageKey, draft, preview, saved, startChoice, sessionGeneration]);

  function updateSource(index: number, change: Partial<TrainingSource>): void {
    setSources((current) => current.map((source, row) => row === index ? { ...source, ...change } : source));
  }
  function labelChoice(index: number, choice: string): void {
    const [kind, value] = choice.split(":");
    setSources((current) => current.map((source, row) => {
      if (row !== index) return source;
      const next: TrainingSource = { dataset_version_id: source.dataset_version_id, input_mapping: source.input_mapping, split_map_id: source.split_map_id };
      return { ...next, ...(kind === "round" ? { annotation_round_id: Number(value) } : kind === "layer" ? { label_set_version_id: Number(value) } : { label_field: importedLabelField(source.dataset_version_id) }) };
    }));
  }
  function chooseStart(choice: "annotations" | "external" | "combine"): void {
    setStartChoice(choice);
    const mapping = taskInputMapping(data, taskId);
    if (choice === "combine") { if (sources.length < 2) setSources((current) => [...current, { dataset_version_id: 0, input_mapping: mapping, label_field: "" }]); return; }
    if (choice === "annotations") {
      setSources(annotationSource(taskId));
      return;
    }
    const external = sourceVersions.find((version) => data.datasets.some((dataset) => dataset.id === version.dataset_id && ["upload", "public_registry"].includes(dataset.source_type)));
    setSources([{ dataset_version_id: external?.id ?? 0, input_mapping: mapping, label_field: importedLabelField(external?.id ?? 0) }]);
    if (!external) onImport();
  }
  const duplicateSources = new Set(sources.map((source) => source.dataset_version_id)).size !== sources.length;
  async function review(): Promise<void> {
    if (sessionGeneration !== getSessionGeneration()) return;
    if (sources.length > 30 || duplicateSources) { setError("Choose up to 30 distinct source dataset versions."); return; }
    const id = ++requestId.current; setBusy(true); setError(null);
    try { const result = await previewTrainingDataset(projectId, draft); if (id === requestId.current && signature === signatureRef.current && sessionGeneration === getSessionGeneration()) setPreview(result); }
    catch (caught) { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setError(caught instanceof Error ? caught.message : "Could not preview training data."); }
    finally { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setBusy(false); }
  }
  async function continueSaved(version: TrainingDatasetVersion): Promise<void> {
    if (sessionGeneration !== getSessionGeneration()) return;
    const id = ++requestId.current;
    let savedDraft: string | null = null;
    try { savedDraft = sessionStorage.getItem(storageKey); } catch { /* Opening a saved dataset does not require browser storage. */ }
    setBusy(true); setError(null);
    try {
      await onCreated(version);
      if (sessionGeneration !== getSessionGeneration()) return;
      try { if (savedDraft !== null && sessionStorage.getItem(storageKey) === savedDraft) sessionStorage.removeItem(storageKey); } catch { /* Browser storage can be blocked after the dataset is saved. */ }
    } catch (caught) { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setError(caught instanceof Error ? caught.message : "The dataset was saved, but could not be opened. Retry opening the saved dataset."); }
    finally { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setBusy(false); }
  }
  async function prepare(): Promise<void> {
    if (!preview?.ready || sessionGeneration !== getSessionGeneration()) return;
    const id = ++requestId.current; setBusy(true); setError(null);
    if (attempt.current?.signature !== signature) attempt.current = { signature, key: crypto.randomUUID() };
    try { sessionStorage.setItem(storageKey, JSON.stringify({ draft, preview, attempt: attempt.current, startChoice })); } catch { /* Keep the current attempt in memory. */ }
    try {
      const result = await prepareTrainingDataset(projectId, draft, preview, attempt.current.key);
      if (id !== requestId.current || sessionGeneration !== getSessionGeneration()) return;
      setSaved(result.training_dataset_version);
      try { sessionStorage.setItem(storageKey, JSON.stringify({ draft, preview, attempt: attempt.current, startChoice, saved: result.training_dataset_version })); } catch { /* Browser storage can be blocked after the dataset is saved. */ }
      await continueSaved(result.training_dataset_version);
    } catch (caught) { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setError(caught instanceof Error ? caught.message : "Could not save training dataset. Retry with the same reviewed selection."); }
    finally { if (id === requestId.current && sessionGeneration === getSessionGeneration()) setBusy(false); }
  }
  if (saved) return <PlatformSection title="Training dataset saved">{error ? <p role="alert">{error}</p> : null}<p role="status">{saved.name} · v{saved.version_number ?? 1} is ready. Its source records and labels remain unchanged.</p><Button label={error ? "Retry opening saved dataset" : "Open saved dataset"} isDisabled={busy} onClick={() => void continueSaved(saved)} /><Button label="Back to training datasets" onClick={onCancel} /></PlatformSection>;
  return <PlatformSection title={parent ? `Create a new version of ${parent.name}` : "Create training dataset"} description="Use submitted annotations, external labeled data, or combine several sources. Preview the records and splits before saving.">
    {error ? <p role="alert" className="platform-form-warning">{error}</p> : null}
    {sources.length > 30 || duplicateSources ? <p role="alert">Choose up to 30 distinct source dataset versions.</p> : null}
    {!sourceVersions.length ? <PlatformEmpty title="Import a source dataset first" detail="Upload labeled records or import papers for annotation." actionLabel="Import source data" onAction={onImport} /> : null}
    {!data.taskVersions.length ? <PlatformEmpty title="Define the prediction task" detail="A shared task defines the input fields and allowed labels for every source." actionLabel={onDefineTask ? "Define task" : undefined} onAction={onDefineTask} /> : null}
    <ValidatedForm className="platform-dialog-form" onBlur={() => setShowValidation(true)} onSubmit={(event) => { event.preventDefault(); void review(); }}>
      {!parent ? <fieldset className="platform-fieldset" disabled={busy}><legend>Start with</legend><div className="platform-option-list">
        {([["annotations", "Use project annotations"], ["external", "Import external labeled data"], ["combine", "Combine sources"]] as const).map(([choice, label]) => <label key={choice}><NativeInput type="radio" name="training-start" value={choice} checked={startChoice === choice} onChange={() => chooseStart(choice)} /><span>{label}</span></label>)}
      </div></fieldset> : null}
      <label><span>Training dataset name</span><NativeInput required aria-invalid={showValidation && !name.trim() ? true : undefined} aria-describedby={showValidation && !name.trim() ? "training-data-name-error" : undefined} value={name} onChange={(event) => setName(event.target.value)} disabled={busy || Boolean(parent)} maxLength={255} />{showValidation && !name.trim() ? <small id="training-data-name-error" aria-hidden="true">Enter a training dataset name.</small> : null}</label>
      <label><span>Task</span><NativeSelect required aria-invalid={showValidation && !taskId ? true : undefined} aria-describedby={showValidation && !taskId ? "training-data-task-error" : undefined} value={taskId} disabled={busy || Boolean(parent)} onChange={(event) => { const id = Number(event.target.value); setTaskId(id); setSources((current) => startChoice === "annotations" ? annotationSource(id) : current.map((source) => ({ dataset_version_id: source.dataset_version_id, input_mapping: taskInputMapping(data, id), label_field: importedLabelField(source.dataset_version_id) }))); }}>
        <option value={0}>Select a task</option>{data.taskVersions.map((item) => <option key={item.id} value={item.id}>{data.taskDefinitions.find((definition) => definition.id === item.task_definition_id)?.name ?? item.task_kind} · v{item.version_number}</option>)}
      </NativeSelect>{showValidation && !taskId ? <small id="training-data-task-error" aria-hidden="true">Choose a task.</small> : null}</label>
      {startChoice === "annotations" && !sources.length ? <p role="status">{canUseRoundAnnotations ? "No submitted project annotations are available for this task. Annotate a source dataset first, or choose external labeled data." : "No published annotation labels are available for this task. Ask a manager to publish a label set, or choose external labeled data."}</p> : null}
      {sources.map((source, index) => <fieldset className="platform-fieldset" key={index} disabled={busy}>
        <legend>Source {index + 1}</legend>
        <label><span>Source dataset version</span><NativeSelect required aria-invalid={showValidation && !source.dataset_version_id ? true : undefined} aria-describedby={showValidation && !source.dataset_version_id ? `training-data-source-${index}-error` : undefined} value={source.dataset_version_id} onChange={(event) => updateSource(index, { dataset_version_id: Number(event.target.value), label_field: importedLabelField(Number(event.target.value)), label_set_version_id: undefined, annotation_round_id: undefined, submission_ids: undefined, split_map_id: undefined })}>
          <option value={0}>Select a source</option>{sourceVersions.filter((version) => version.item_count > 0).map((version) => <option key={version.id} value={version.id} disabled={sources.some((other, row) => row !== index && other.dataset_version_id === version.id)}>{data.datasets.find((dataset) => dataset.id === version.dataset_id)?.name ?? "Source"} · v{version.version_number} · {formatCount(version.item_count, "record")}</option>)}
        </NativeSelect>{showValidation && !source.dataset_version_id ? <small id={`training-data-source-${index}-error`} aria-hidden="true">Choose a source dataset version.</small> : null}</label>
        <div className="platform-form-grid">{taskFields.map((field) => <label key={field}><span>Source field for {field}</span><NativeInput required aria-invalid={showValidation && !(source.input_mapping[field] ?? field).trim() ? true : undefined} aria-describedby={showValidation && !(source.input_mapping[field] ?? field).trim() ? `training-data-mapping-${index}-${encodeURIComponent(field)}-error` : undefined} value={source.input_mapping[field] ?? field} onChange={(event) => updateSource(index, { input_mapping: { ...source.input_mapping, [field]: event.target.value } })} />{showValidation && !(source.input_mapping[field] ?? field).trim() ? <small id={`training-data-mapping-${index}-${encodeURIComponent(field)}-error`} aria-hidden="true">Enter the source field for {field}.</small> : null}</label>)}</div>
        <label><span>Labels from</span><NativeSelect value={source.annotation_round_id ? `round:${source.annotation_round_id}` : source.label_set_version_id ? `layer:${source.label_set_version_id}` : "field"} onChange={(event) => labelChoice(index, event.target.value)}>
          <option value="field">A field in the imported dataset</option>
          {data.labelSets.filter((layer) => layer.dataset_version_id === source.dataset_version_id && layer.task_version_id === taskId && layer.composition_policy !== "exclude").map((layer) => <option key={`layer:${layer.id}`} value={`layer:${layer.id}`}>{layer.name} · v{layer.version_number} · {layer.source_kind}</option>)}
          {canUseRoundAnnotations ? data.rounds.filter((round) => round.dataset_version_id === source.dataset_version_id && round.task_version_id === taskId && (round.status === "open" || round.status === "closed")).map((round) => <option key={`round:${round.id}`} value={`round:${round.id}`}>Submitted annotations: {round.name}</option>) : null}
        </NativeSelect></label>
        {source.label_field !== undefined ? <label><span>Label field</span><NativeInput required aria-invalid={showValidation && !source.label_field.trim() ? true : undefined} aria-describedby={showValidation && !source.label_field.trim() ? `training-data-label-${index}-error` : undefined} value={source.label_field} onChange={(event) => updateSource(index, { label_field: event.target.value })} />{showValidation && !source.label_field.trim() ? <small id={`training-data-label-${index}-error`} aria-hidden="true">Enter the field containing labels.</small> : null}</label> : null}
        {source.annotation_round_id ? <p>Only finalized submissions enter the snapshot. You can continue annotating this source after creating the training dataset.</p> : null}
        <label><span>Existing evaluation splits</span><NativeSelect value={source.split_map_id ?? 0} onChange={(event) => updateSource(index, { split_map_id: Number(event.target.value) || undefined })}><option value={0}>Use protected splits when present; split new records below</option>{data.splitMaps.filter((split) => split.dataset_version_id === source.dataset_version_id).map((split) => <option key={split.id} value={split.id}>{split.name}</option>)}</NativeSelect></label>
        {sources.length > 1 ? <Button label={`Remove source ${index + 1}`} variant="ghost" onClick={() => setSources((current) => current.filter((_, row) => row !== index))} /> : null}
      </fieldset>)}
      <div className="platform-row-actions"><Button label="Add another source" isDisabled={busy || sources.length >= 30} onClick={() => setSources((current) => [...current, { dataset_version_id: 0, input_mapping: taskInputMapping(data, taskId), label_field: "" }])} /><Button label="Import external data" isDisabled={busy} onClick={onImport} /></div>
      <div className="platform-form-grid"><label><span>Train percent for new groups</span><NativeInput type="number" min={1} max={98} required aria-invalid={showValidation && (trainPercent < 1 || trainPercent > 98 || trainPercent + validationPercent >= 100) ? true : undefined} aria-describedby={showValidation && (trainPercent < 1 || trainPercent > 98 || trainPercent + validationPercent >= 100) ? "training-data-split-error" : undefined} value={trainPercent} disabled={busy} onChange={(event) => setTrainPercent(Number(event.target.value))} /></label><label><span>Validation percent for new groups</span><NativeInput type="number" min={1} max={98} required aria-invalid={showValidation && (validationPercent < 1 || validationPercent > 98 || trainPercent + validationPercent >= 100) ? true : undefined} aria-describedby={showValidation && (validationPercent < 1 || validationPercent > 98 || trainPercent + validationPercent >= 100) ? "training-data-split-error" : undefined} value={validationPercent} disabled={busy} onChange={(event) => setValidationPercent(Number(event.target.value))} /></label></div>
      {showValidation && (trainPercent < 1 || validationPercent < 1 || trainPercent > 98 || validationPercent > 98 || trainPercent + validationPercent >= 100) ? <small id="training-data-split-error" aria-hidden="true">Use train and validation percentages from 1 to 98, with a total below 100 to retain a test split.</small> : null}
      <p>Test: {100 - trainPercent - validationPercent}%. Existing protected evaluation groups stay protected. Unsubmitted or unlabeled records are excluded.</p>
      <div className="platform-row-actions"><Button label="Cancel" isDisabled={busy} onClick={onCancel} /><Button label={busy ? "Checking…" : "Preview training dataset"} type="submit" variant={preview ? "secondary" : "primary"} isDisabled={busy || sources.length > 30 || duplicateSources || !taskId || !sources.length || sources.some((source) => !source.dataset_version_id) || trainPercent + validationPercent >= 100} /></div>
    </ValidatedForm>
    {preview ? <section aria-label="Training dataset preview"><h3>Review included data</h3>
      <p role="status">{formatCount(preview.item_count, "record")} · {formatCount(preview.group_count, "group")} · {formatCount(preview.excluded_unlabeled_count, "unlabeled record")} excluded · {formatCount(preview.duplicate_count, "duplicate")} removed</p>
      <p>{Object.entries(preview.split_counts).map(([split, count]) => `${split}: ${count.toLocaleString()}`).join(" · ")}</p>
      <ul>{preview.source_counts.map((source, index) => <li key={`${source.dataset_version_id}:${index}`}>Source {index + 1}: {formatCount(source.total_count, "record")}, {source.labeled_count.toLocaleString()} labeled, {source.excluded_unlabeled_count.toLocaleString()} unlabeled excluded</li>)}</ul>
      {preview.issues.length ? <ul>{preview.issues.map((issue, index) => <li key={index}>{issue.source_index != null ? `Source ${issue.source_index + 1}: ` : ""}{issue.message}</li>)}</ul> : null}
      {preview.warnings?.length ? <div role="status" aria-label="Training dataset warnings" className="platform-form-warning">
        <strong>Review warnings</strong>
        <ul>{preview.warnings.map((warning, index) => <li key={index}>{warning.source_index != null ? `Source ${warning.source_index + 1}: ` : ""}{warning.message}</li>)}</ul>
      </div> : null}
      <Button label={busy ? "Saving…" : parent ? "Create new training version" : "Create training dataset"} variant="primary" isDisabled={busy || !preview.ready} onClick={() => void prepare()} />
    </section> : null}
  </PlatformSection>;
}
