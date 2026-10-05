import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import { PlatformEmpty, PlatformPageHeader, PlatformSection, PlatformStatus } from "./components";
import {
  createPredictionReview, createPredictionRun, getPredictionResults, downloadPredictions,
  listPredictionRuns, predictionDownloadUrl, retryPredictionRun,
  type PredictionResultPage, type PredictionRun,
} from "./predictionApi";
import type { PlatformProjectData } from "./types";

const message = (error: unknown): string => error instanceof Error ? error.message : "The request failed. Try again.";
const predictionLabel = (value: unknown): string => typeof value === "string" ? value : JSON.stringify(value);
const draftStorageKey = (projectId: number) => `prediction-draft:${projectId}`;
interface SavedDraft { name: string; datasetVersionId: number; modelVersionId: number; step: number; request: {fingerprint: string; key: string} }
function readDraft(projectId: number): SavedDraft {
  let stored: Partial<SavedDraft> = {};
  try { const parsed = JSON.parse(sessionStorage.getItem(draftStorageKey(projectId)) || "{}"); if (parsed && typeof parsed === "object") stored = parsed; } catch { /* A fresh draft remains usable when storage is unavailable. */ }
  return {
    name: typeof stored.name === "string" ? stored.name : "",
    datasetVersionId: Number(new URLSearchParams(window.location.search).get("datasetVersionId")) || Number(stored.datasetVersionId) || 0,
    modelVersionId: Number(stored.modelVersionId) || 0,
    step: typeof stored.step === "number" && stored.step >= 0 && stored.step <= 2 ? stored.step : 0,
    request: stored.request && typeof stored.request.key === "string" && typeof stored.request.fingerprint === "string" ? stored.request : { fingerprint: "", key: "" },
  };
}

export default function InferenceScreen({ projectId, data, currentUserId, canReview, onOpenRound, onRefresh, onImport, onOpenModels }: {
  projectId: number;
  data: PlatformProjectData;
  currentUserId: number | null;
  canReview: boolean;
  onOpenRound: (roundId: number) => void;
  onRefresh: () => Promise<void>;
  onImport?: () => void;
  onOpenModels?: () => void;
}): React.ReactElement {
  const [runs, setRuns] = useState<PredictionRun[]>([]);
  const [runId, setRunId] = useState(() => Number(new URLSearchParams(window.location.search).get("predictionRunId")) || 0);
  const [name, setName] = useState(() => readDraft(projectId).name);
  const [datasetVersionId, setDatasetVersionId] = useState(() => readDraft(projectId).datasetVersionId);
  const [modelVersionId, setModelVersionId] = useState(() => readDraft(projectId).modelVersionId);
  const [step, setStep] = useState(() => readDraft(projectId).step);
  const [busyAction, setBusyAction] = useState<"create" | "retry" | "review" | null>(null);
  const busy = busyAction !== null;
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [page, setPage] = useState<PredictionResultPage | null>(null);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [includeSubmitted, setIncludeSubmitted] = useState(false);
  const [reviewName, setReviewName] = useState("");
  const [pollPaused, setPollPaused] = useState(false);
  const [pollGeneration, setPollGeneration] = useState(0);
  const projectRef = useRef(projectId);
  projectRef.current = projectId;
  const createKey = useRef(readDraft(projectId).request);
  const reviewKey = useRef({ fingerprint: "", key: "" });
  const compatibleModels = useMemo(() => data.modelVersions.filter((model) =>
    model.recipe_key === "tfidf_logistic_regression" && model.framework === "scikit-learn" && model.checkpoint_package_id &&
    data.taskVersions.some((task) => task.id === model.task_version_id && task.task_kind === "classification"),
  ), [data.modelVersions, data.taskVersions]);
  const datasets = data.datasetVersions.filter((version) => version.item_count > 0 && version.provenance?.ingestion !== "training_preparation_v1");
  const run = runs.find((item) => item.id === runId);
  const activeKey = runs.filter((item) => item.status === "planned" || item.status === "queued" || item.status === "running").map((item) => item.id).join(":");

  function saveDraft(next: Partial<SavedDraft> = {}): void {
    try { sessionStorage.setItem(draftStorageKey(projectId), JSON.stringify({ name, datasetVersionId, modelVersionId, step, request: createKey.current, ...next })); } catch { /* Network requests remain available without browser storage. */ }
  }
  function moveStep(next: number): void { setStep(next); saveDraft({ step: next }); setError(null); }
  function saveReview(next: Record<string, unknown> = {}): void {
    try { sessionStorage.setItem(`prediction-review:${projectId}:${runId}`, JSON.stringify({ name: reviewName, selected: [...selected], includeSubmitted, request: reviewKey.current, ...next })); } catch { /* Keep review usable without storage. */ }
  }

  const refreshRuns = useCallback(async (): Promise<void> => {
    const refreshed = await listPredictionRuns(projectId);
    if (projectRef.current === projectId) setRuns(refreshed);
  }, [projectId]);

  useEffect(() => {
    let cancelled = false;
    setRuns([]);
    setPage(null);
    setSelected(new Set());
    setRunId(Number(new URLSearchParams(window.location.search).get("predictionRunId")) || 0);
    const draft = readDraft(projectId);
    setDatasetVersionId(draft.datasetVersionId);
    setModelVersionId(draft.modelVersionId);
    setName(draft.name);
    setStep(draft.step);
    createKey.current = draft.request;
    setReviewName("");
    setOffset(0);
    setBusyAction(null);
    setPollPaused(false);
    setLoading(true);
    setError(null);
    void refreshRuns().catch((caught) => { if (!cancelled) setError(message(caught)); }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refreshRuns, projectId]);

  useEffect(() => {
    const restore = (): void => {
      setRunId(Number(new URLSearchParams(window.location.search).get("predictionRunId")) || 0);
      setOffset(0); setSelected(new Set()); setReviewName("");
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);

  useEffect(() => {
    if (!run?.id) return;
    let saved: Record<string, unknown> = {};
    try { const parsed = JSON.parse(sessionStorage.getItem(`prediction-review:${projectId}:${run.id}`) || "{}"); if (parsed && typeof parsed === "object") saved = parsed; } catch { /* Start a fresh review when storage is unavailable. */ }
    setReviewName(typeof saved.name === "string" ? saved.name : `${run.name} review`.slice(0, 255));
    setSelected(new Set(Array.isArray(saved.selected) ? saved.selected.filter((item): item is number => typeof item === "number" && Number.isInteger(item) && item > 0) : []));
    setIncludeSubmitted(saved.includeSubmitted === true);
    const request = saved.request as {fingerprint?: unknown; key?: unknown} | undefined;
    reviewKey.current = request && typeof request.fingerprint === "string" && typeof request.key === "string" ? { fingerprint: request.fingerprint, key: request.key } : { fingerprint: "", key: "" };
  }, [projectId, run?.id, run?.name]);

  useEffect(() => {
    if (!activeKey || pollPaused) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const deadline = Date.now() + 120_000;
    const poll = async (): Promise<void> => {
      if (cancelled) return;
      if (Date.now() >= deadline) { setPollPaused(true); return; }
      try { await refreshRuns(); } catch { if (!cancelled) setNotice("Status could not be refreshed. Automatic updates will retry."); }
      if (!cancelled) timer = setTimeout(() => void poll(), 3_000);
    };
    timer = setTimeout(() => void poll(), 3_000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [activeKey, pollPaused, pollGeneration, refreshRuns]);

  useEffect(() => {
    let cancelled = false;
    setPage(null);
    if (run?.status === "completed") {
      void getPredictionResults(projectId, run.id, offset).then((result) => { if (!cancelled) setPage(result); }).catch((caught) => { if (!cancelled) setError(message(caught)); });
    }
    return () => { cancelled = true; };
  }, [projectId, run?.id, run?.status, offset, pollGeneration]);

  function openRun(next: PredictionRun): void {
    setRunId(next.id);
    setOffset(0);
    setSelected(new Set());
    setIncludeSubmitted(false);
    setReviewName(`${next.name} review`.slice(0, 255));
    const url = new URL(window.location.href);
    url.searchParams.set("predictionRunId", String(next.id));
    if (url.href !== window.location.href) window.history.pushState(window.history.state, "", url);
  }

  async function create(): Promise<void> {
    if (!name.trim() || !datasetVersionId || !modelVersionId) {
      setError("Enter a prediction name and choose a dataset and compatible model.");
      return;
    }
    const draft = { name: name.trim(), dataset_version_id: datasetVersionId, model_version_id: modelVersionId };
    const fingerprint = JSON.stringify(draft);
    if (createKey.current.fingerprint !== fingerprint) createKey.current = { fingerprint, key: crypto.randomUUID() };
    saveDraft();
    setBusyAction("create"); setError(null);
    try {
      const created = await createPredictionRun(projectId, { ...draft, request_key: createKey.current.key });
      if (projectRef.current !== projectId) return;
      setRuns((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      openRun(created);
      setName(""); createKey.current = { fingerprint: "", key: "" };
      setStep(0); saveDraft({ name: "", step: 0, request: createKey.current });
      setPollPaused(false); setPollGeneration((value) => value + 1);
      setNotice(`Prediction run “${created.name}” ${created.status === "completed" ? "completed" : "created"}.`);
    } catch (caught) { if (projectRef.current === projectId) setError(message(caught)); }
    finally { if (projectRef.current === projectId) setBusyAction(null); }
  }

  async function retry(target: PredictionRun): Promise<void> {
    setBusyAction("retry"); setError(null);
    try {
      const refreshed = await retryPredictionRun(projectId, target.id);
      if (projectRef.current !== projectId) return;
      setRuns((current) => current.map((item) => item.id === refreshed.id ? refreshed : item));
      setPollPaused(false); setPollGeneration((value) => value + 1);
    } catch (caught) { if (projectRef.current === projectId) setError(message(caught)); } finally { if (projectRef.current === projectId) setBusyAction(null); }
  }

  async function review(): Promise<void> {
    if (!run || !selected.size || !reviewName.trim()) { setError("Name the review round and select at least 1 eligible result."); return; }
    const draft = { name: reviewName.trim(), dataset_item_ids: [...selected].sort((a, b) => a - b), include_submitted: includeSubmitted };
    const fingerprint = JSON.stringify({ ...draft, runId });
    if (reviewKey.current.fingerprint !== fingerprint) reviewKey.current = { fingerprint, key: crypto.randomUUID() };
    saveReview();
    setBusyAction("review"); setError(null);
    try {
      const result = await createPredictionReview(projectId, run.id, { ...draft, request_key: reviewKey.current.key });
      if (projectRef.current !== projectId) return;
      setNotice(`Created a review round with ${result.item_count} items. ${result.excluded_protected + result.excluded_submitted} ineligible items excluded.`);
      await onRefresh();
      if (projectRef.current === projectId) onOpenRound(result.round_id);
    } catch (caught) { if (projectRef.current === projectId) setError(message(caught)); } finally { if (projectRef.current === projectId) setBusyAction(null); }
  }

  const datasetName = (versionId: number): string => {
    const version = data.datasetVersions.find((item) => item.id === versionId);
    return `${data.datasets.find((item) => item.id === version?.dataset_id)?.name ?? "Dataset"} · v${version?.version_number ?? "?"}`;
  };
  async function download(format: "csv" | "jsonl"): Promise<void> {
    if (!run) return;
    setError(null);
    setNotice("Preparing prediction export…");
    try {
      const blob = await downloadPredictions(projectId, run.id, format);
      if (projectRef.current !== projectId) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `predictions-${run.id}.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1_000);
      setNotice("Prediction export downloaded.");
    } catch (caught) {
      if (projectRef.current === projectId) { setError(message(caught)); setNotice(""); }
    }
  }
  const modelName = (versionId: number): string => {
    const version = data.modelVersions.find((item) => item.id === versionId);
    return `${data.models.find((item) => item.id === version?.registered_model_id)?.name ?? "Model"} · v${version?.version_number ?? "?"}`;
  };

  return <div className="platform-page">
    <PlatformPageHeader title="Inference" description="Predict labels with a trained model, download results, or select examples for manual review." />
    {error ? <p className="platform-dialog-error" role="alert">{error}</p> : null}
    <p role="status" aria-live="polite">{notice}</p>
    <PlatformSection title="New prediction run" description="Use a TF-IDF single-label classifier trained in this project. Predictions stay separate from human annotations.">
      {!compatibleModels.length || !datasets.length ? <PlatformEmpty title={!datasets.length ? "Choose source data for predictions" : "A compatible trained model is needed"} detail={!datasets.length ? "Import a named collection of papers. Annotation and training splits are optional for predictions." : "Select a TF-IDF single-label classification model trained in this project."} actionLabel={!datasets.length && onImport ? "Import source dataset" : onOpenModels ? "Open models" : undefined} onAction={!datasets.length ? onImport : onOpenModels} /> :
        <form className="platform-dialog-form" onSubmit={(event) => { event.preventDefault(); if (step < 2) moveStep(step + 1); else void create(); }}>
          <nav className="platform-wizard-steps" aria-label="Prediction setup">{["Select dataset", "Select model", "Review"].map((label, index) => <button type="button" key={label} aria-current={step === index ? "step" : undefined} disabled={busy || (index > 0 && !datasetVersionId) || (index > 1 && (!modelVersionId || !name.trim()))} onClick={() => moveStep(index)}>{index + 1}. {label}</button>)}</nav>
          {step === 0 ? <>
            <label><span>Source dataset</span><select required name="prediction-dataset" value={datasetVersionId || ""} onChange={(event) => { const next = Number(event.target.value); setDatasetVersionId(next); saveDraft({ datasetVersionId: next }); }}>
              <option value="">Select a dataset</option>{datasets.map((version) => <option key={version.id} value={version.id}>{datasetName(version.id)} · {version.item_count} items</option>)}
            </select></label>
            <p>Every example in this dataset version will receive a prediction.</p>
            {onImport ? <Button label="Import new source dataset" type="button" variant="ghost" onClick={onImport} /> : null}
          </> : step === 1 ? <>
            <label><span>Prediction name</span><input name="prediction-name" autoComplete="off" required maxLength={255} value={name} onChange={(event) => { setName(event.target.value); saveDraft({ name: event.target.value }); }} placeholder="Tucatinib predictions…" /></label>
            <label><span>Trained model</span><select required name="prediction-model" value={modelVersionId || ""} onChange={(event) => { const next = Number(event.target.value); setModelVersionId(next); saveDraft({ modelVersionId: next }); }}>
              <option value="">Select a model</option>{compatibleModels.map((version) => <option key={version.id} value={version.id}>{modelName(version.id)}</option>)}
            </select></label>
          </> : <dl><dt>Prediction run</dt><dd>{name}</dd><dt>Source dataset</dt><dd>{datasetName(datasetVersionId)}</dd><dt>Trained model</dt><dd>{modelName(modelVersionId)}</dd><dt>Examples to predict</dt><dd>{datasets.find((item) => item.id === datasetVersionId)?.item_count ?? 0}</dd></dl>}
          <div className="platform-dialog-actions">{step > 0 ? <Button label="Back" type="button" isDisabled={busy} onClick={() => moveStep(step - 1)} /> : null}<Button label={busyAction === "create" ? "Starting predictions…" : step === 0 ? "Continue to model" : step === 1 ? "Review predictions" : "Run predictions"} type="submit" variant="primary" isDisabled={busy} isLoading={busyAction === "create"} /></div>
        </form>}
    </PlatformSection>
    <PlatformSection title="Prediction runs" description="Runs keep their selected dataset and model versions." action={<Button label="Refresh status" variant="secondary" onClick={() => { setPollPaused(false); setPollGeneration((value) => value + 1); void refreshRuns().catch((caught) => setError(message(caught))); }} />}>
      {pollPaused ? <p role="status">Automatic updates paused after 2 minutes. Refresh status to resume.</p> : null}
      {loading ? <p role="status">Loading prediction runs…</p> : runs.length ? <div className="platform-table-scroll" role="region" aria-label="Prediction runs" tabIndex={0}><table className="platform-table"><thead><tr><th scope="col">Run</th><th scope="col">Dataset</th><th scope="col">Model</th><th scope="col">Status</th><th scope="col">Actions</th></tr></thead><tbody>
        {runs.map((item) => <tr key={item.id}><td><strong>{item.name}</strong>{item.failure_reason ? <span>{item.failure_reason}</span> : null}</td><td>{datasetName(item.dataset_version_id)}</td><td>{modelName(item.model_version_id)}</td><td><PlatformStatus value={item.status} /></td><td><div className="platform-row-actions"><Button label={item.status === "completed" ? "View results" : "View run"} size="sm" onClick={() => openRun(item)} />{item.status === "failed" || item.status === "queued" ? <Button label="Retry run" size="sm" isDisabled={busy} onClick={() => void retry(item)} /> : null}</div></td></tr>)}
      </tbody></table></div> : <PlatformEmpty title="No prediction runs yet" detail="Choose source data and a trained model to make your first predictions." />}
    </PlatformSection>
    {run ? <PlatformSection title={run.name} description={`${datasetName(run.dataset_version_id)} · ${modelName(run.model_version_id)}`}>
      {run.status !== "completed" ? <p role="status">{run.status === "failed" ? run.failure_reason ?? "Prediction failed. Retry the run." : "Predictions are being prepared. You can return to this run later."}</p> : <>
        <div className="platform-page-actions"><a className="platform-text-action" href={predictionDownloadUrl(projectId, run.id, "csv")} download onClick={(event) => { event.preventDefault(); void download("csv"); }}>Download CSV</a><a className="platform-text-action" href={predictionDownloadUrl(projectId, run.id, "jsonl")} download onClick={(event) => { event.preventDefault(); void download("jsonl"); }}>Download JSONL</a></div>
        {canReview && currentUserId ? <label className="platform-checkbox-row platform-prediction-review-toggle"><input type="checkbox" checked={includeSubmitted} onChange={(event) => { setIncludeSubmitted(event.target.checked); setSelected(new Set()); saveReview({ includeSubmitted: event.target.checked, selected: [] }); }} /><span>Include previously submitted annotations for another review</span></label> : null}
        {!page ? <p role="status">Loading results…</p> : <>
          <p aria-live="polite">{page.total} predictions · {selected.size} selected for review</p>
          <div className="platform-table-scroll" role="region" aria-label="Prediction results" tabIndex={0}><table className="platform-table"><thead><tr>{canReview ? <th scope="col">Select</th> : null}<th scope="col">Example</th><th scope="col">Prediction</th><th scope="col">Confidence</th><th scope="col">Review status</th></tr></thead><tbody>{page.items.map((item) => <tr key={item.dataset_item_id}>
            {canReview ? <td><input type="checkbox" aria-label={`Review ${item.stable_key}`} checked={selected.has(item.dataset_item_id)} disabled={item.protected || (item.already_submitted && !includeSubmitted) || busy} onChange={(event) => { const next = new Set(selected); if (event.target.checked) next.add(item.dataset_item_id); else next.delete(item.dataset_item_id); setSelected(next); saveReview({ selected: [...next] }); }} /></td> : null}
            <td><strong>{item.title}</strong><details><summary>Read example</summary><p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{item.text}</p></details></td><td>{predictionLabel(item.prediction)}</td><td>{item.confidence === null ? "Unavailable" : new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 }).format(item.confidence)}</td><td>{item.protected ? "Protected holdout — excluded" : item.already_submitted ? "Previously submitted" : "Available for review"}</td>
          </tr>)}</tbody></table></div>
          <div className="platform-row-actions platform-prediction-pagination"><Button label="Previous results" isDisabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 50))} /><span>{page.total ? `${offset + 1}–${Math.min(offset + 50, page.total)} of ${page.total}` : "0 results"}</span><Button label="Next results" isDisabled={offset + 50 >= page.total} onClick={() => setOffset(offset + 50)} />{canReview ? <Button label="Select eligible results on this page" onClick={() => { const next = new Set([...selected, ...page.items.filter((item) => !item.protected && (!item.already_submitted || includeSubmitted)).map((item) => item.dataset_item_id)]); setSelected(next); saveReview({ selected: [...next] }); }} /> : null}</div>
        </>}
        {canReview && currentUserId ? <form className="platform-dialog-form" onSubmit={(event) => { event.preventDefault(); void review(); }}><label><span>Review round name</span><input required name="prediction-review-name" autoComplete="off" maxLength={255} value={reviewName} onChange={(event) => { setReviewName(event.target.value); saveReview({ name: event.target.value }); }} /></label><p>You will review the selected original examples with model suggestions visible. Protected holdout examples remain excluded.</p><div className="platform-dialog-actions"><Button label={busyAction === "review" ? "Creating review round…" : "Create review round"} type="submit" variant="primary" isDisabled={busy} isLoading={busyAction === "review"} /></div></form> : null}
      </>}
    </PlatformSection> : null}
  </div>;
}
