import { NativeInput, NativeSelect, NativeTextarea } from "@/components/NativeField";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Eye,
  Save,
  Send,
  X,
} from "lucide-react";

import ConfirmDialog from "@/components/ConfirmDialog";
import { IconButton } from "@astryxdesign/core/IconButton";

import {
  addRoundDecision,
  loadRoundWork,
  publishRoundLabelSet,
  revealRoundFeedback,
  submitRoundDecisions,
  transitionRound,
  type RoundWorkData,
} from "./api";
import { formatStatus, PlatformStatus } from "./components";
import type {
  AnnotationDecision,
  FeedbackReveal,
  RoundWorkRound,
  TaskVersion,
} from "./types";

interface RoundWorkbenchProps {
  round: RoundWorkRound;
  task: TaskVersion;
  currentUserId: number | null;
  canManage: boolean;
  onClose: () => void;
  onRefresh: () => Promise<void>;
}

type FeedbackDisposition = "accepted" | "modified" | "rejected" | "ignored";

function latestDecisions(
  decisions: AnnotationDecision[],
  annotatorUserId: number | null,
): Map<number, AnnotationDecision> {
  const own = decisions.filter(
    (decision) =>
      annotatorUserId === null || decision.annotator_user_id === annotatorUserId,
  );
  const superseded = new Set(
    own
      .map((decision) => decision.supersedes_decision_id)
      .filter((id): id is number => id !== null),
  );
  return new Map(
    own
      .filter((decision) => !superseded.has(decision.id))
      .map((decision) => [decision.round_item_id, decision]),
  );
}

function enumValues(task: TaskVersion): string[] {
  const direct = task.output_schema.enum;
  if (Array.isArray(direct) && direct.every((value) => typeof value === "string")) {
    return direct;
  }
  const properties = task.output_schema.properties;
  if (properties && typeof properties === "object" && !Array.isArray(properties)) {
    const label = (properties as Record<string, unknown>).label;
    if (label && typeof label === "object" && !Array.isArray(label)) {
      const nested = (label as Record<string, unknown>).enum;
      if (Array.isArray(nested) && nested.every((value) => typeof value === "string")) {
        return nested;
      }
    }
  }
  const configured = task.label_rules.values;
  return Array.isArray(configured) &&
    configured.every((value) => typeof value === "string")
    ? configured
    : [];
}

function editorValue(task: TaskVersion, output: unknown): string {
  if (output === undefined || output === null) return "";
  if (task.task_kind === "regression") return String(output);
  if (task.task_kind === "classification") {
    if (typeof output === "string") return output;
    if (
      output &&
      typeof output === "object" &&
      !Array.isArray(output) &&
      typeof (output as Record<string, unknown>).label === "string"
    ) {
      return String((output as Record<string, unknown>).label);
    }
  }
  if (task.task_kind === "multilabel_classification" && Array.isArray(output)) {
    return output.filter((value) => typeof value === "string").join("\n");
  }
  if (
    ["generation", "instruction_tuning"].includes(task.task_kind) &&
    output &&
    typeof output === "object" &&
    !Array.isArray(output) &&
    typeof (output as Record<string, unknown>).completion === "string"
  ) {
    return String((output as Record<string, unknown>).completion);
  }
  return JSON.stringify(output, null, 2);
}

function parsedOutput(task: TaskVersion, value: string): unknown {
  if (task.task_kind === "regression") {
    const number = Number(value);
    if (!value.trim() || !Number.isFinite(number)) throw new Error("Enter a valid numeric value.");
    return number;
  }
  if (task.task_kind === "classification") {
    if (!value.trim()) throw new Error("Enter a label.");
    return task.output_schema.type === "object" ? { label: value } : value;
  }
  if (task.task_kind === "multilabel_classification") {
    return [...new Set(value.split(/\r?\n|,/).map((item) => item.trim()).filter(Boolean))];
  }
  if (["generation", "instruction_tuning"].includes(task.task_kind)) {
    return task.output_schema.type === "object" ? { completion: value } : value;
  }
  try {
    return JSON.parse(value);
  } catch {
    throw new Error("The annotation must be valid JSON for this task.");
  }
}

function AnnotationEditor({
  task,
  value,
  disabled,
  error,
  onChange,
}: {
  task: TaskVersion;
  value: string;
  disabled: boolean;
  error: string | null;
  onChange: (value: string) => void;
}): React.ReactElement {
  const values = enumValues(task);
  if (task.task_kind === "classification" && values.length) {
    return (
      <label className="platform-workbench-field">
        <span>Label</span>
        <NativeSelect
          aria-label="Label"
          id="round-annotation-input"
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "round-annotation-error" : undefined}
          required
          disabled={disabled}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">Select a label</option>
          {values.map((option) => <option key={option}>{option}</option>)}
        </NativeSelect>
        {error ? <span id="round-annotation-error" className="platform-dialog-error" role="alert">{error}</span> : null}
      </label>
    );
  }
  if (task.task_kind === "regression") {
    return (
      <label className="platform-workbench-field">
        <span>Value</span>
        <NativeInput
          type="number"
          step="any"
          aria-label="Value"
          id="round-annotation-input"
          aria-invalid={Boolean(error)}
          aria-describedby={error ? "round-annotation-error" : undefined}
          required
          disabled={disabled}
          value={value}
          onChange={(event) => onChange(event.target.value)}
        />
        {error ? <span id="round-annotation-error" className="platform-dialog-error" role="alert">{error}</span> : null}
      </label>
    );
  }
  const naturalText = [
    "classification",
    "multilabel_classification",
    "generation",
    "instruction_tuning",
  ].includes(task.task_kind);
  return (
    <label className="platform-workbench-field">
      <span>
        {task.task_kind === "classification"
          ? "Label"
          : task.task_kind === "multilabel_classification"
            ? "Labels, one per line"
            : naturalText
              ? "Response"
              : "Structured annotation"}
      </span>
      <NativeTextarea
        aria-label={task.task_kind === "classification" ? "Label" : task.task_kind === "multilabel_classification" ? "Labels, one per line" : naturalText ? "Response" : "Structured annotation"}
        rows={naturalText ? 7 : 10}
        id="round-annotation-input"
        aria-invalid={Boolean(error)}
        aria-describedby={error ? "round-annotation-error" : undefined}
        required
        disabled={disabled}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        spellCheck={naturalText}
      />
      {error ? <span id="round-annotation-error" className="platform-dialog-error" role="alert">{error}</span> : null}
    </label>
  );
}

function ItemPayload({
  payload,
}: {
  payload: Record<string, unknown>;
}): React.ReactElement {
  return (
    <dl className="platform-workbench-payload">
      {Object.entries(payload).map(([name, value]) => (
        <div key={name}>
          <dt>{name.replace(/_/g, " ")}</dt>
          <dd>
            {typeof value === "string"
              ? value
              : typeof value === "object" && value !== null ? <details><summary>View {name.replace(/_/g, " ")}</summary><pre className="platform-code-block">{JSON.stringify(value, null, 2)}</pre></details> : String(value)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

export default function RoundWorkbench({
  round,
  task,
  currentUserId,
  canManage,
  onClose,
  onRefresh,
}: RoundWorkbenchProps): React.ReactElement {
  const [work, setWork] = useState<RoundWorkData | null>(null);
  const [itemIndex, setItemIndex] = useState(0);
  const [editorText, setEditorText] = useState("");
  const [rationale, setRationale] = useState("");
  const [decisionKind, setDecisionKind] =
    useState<AnnotationDecision["decision_kind"]>("annotation");
  const [feedbackByItem, setFeedbackByItem] =
    useState<Record<number, FeedbackReveal>>({});
  const [feedbackDisposition, setFeedbackDisposition] =
    useState<FeedbackDisposition | "">("");
  const [labelSourceKind, setLabelSourceKind] =
    useState<"human" | "adjudicated">("human");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fieldError, setFieldError] = useState<{ field: "annotation" | "feedback"; message: string } | null>(null);
  const [pendingAction, setPendingAction] = useState<"finalize" | "close" | "publish" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [publishedSnapshots, setPublishedSnapshots] = useState<string[]>([]);
  const publicationKey = JSON.stringify([
    round.project_id,
    round.id,
    labelSourceKind,
    work?.submissions.map((submission) => submission.id).sort((a, b) => a - b),
  ]);
  const autoRevealStarted = useRef(new Set<number>());
  const reloadRequestIdRef = useRef(0);
  const roundIdentity = `${round.project_id}:${round.id}`;
  const roundIdentityRef = useRef(roundIdentity);
  roundIdentityRef.current = roundIdentity;

  const ownLatest = useMemo(
    () => latestDecisions(work?.decisions ?? [], currentUserId),
    [currentUserId, work?.decisions],
  );
  const currentRoundItem = work?.roundItems[itemIndex] ?? null;
  const currentDatasetItem = currentRoundItem
    ? work?.datasetItems.find(
        (item) => item.id === currentRoundItem.dataset_item_id,
      ) ?? null
    : null;
  const currentDecision = currentRoundItem
    ? ownLatest.get(currentRoundItem.id) ?? null
    : null;
  const currentFeedback = currentRoundItem && round.status === "open"
    ? feedbackByItem[currentRoundItem.id]
    : undefined;
  const currentSelectionStrategy = currentRoundItem &&
    typeof currentRoundItem.selection_reason?.strategy === "string"
      ? currentRoundItem.selection_reason.strategy
      : null;
  const currentUserSubmitted = work?.submissions
    .filter((submission) => submission.annotator_user_id === currentUserId)
    .sort((left, right) => right.sequence - left.sequence)[0];
  const canAdjudicateClosedRound = canManage && round.status === "closed";
  const canWriteDecision =
    round.status === "open" ||
    (canAdjudicateClosedRound && decisionKind === "adjudication");
  const canFinalizeDecisions =
    round.status === "open" ||
    (canAdjudicateClosedRound &&
      ownLatest.size > 0 &&
      [...ownLatest.values()].every(
        (decision) => decision.decision_kind === "adjudication",
      ));

  const reloadWork = useCallback(async (): Promise<void> => {
    if (roundIdentityRef.current !== roundIdentity) return;
    const requestId = ++reloadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === reloadRequestIdRef.current &&
      roundIdentity === roundIdentityRef.current;
    setBusy(true);
    setError(null);
    try {
      const result = await loadRoundWork(round.project_id, round);
      if (isCurrentRequest()) {
        setWork(result);
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Unable to load round work.",
        );
      }
    } finally {
      if (isCurrentRequest()) {
        setBusy(false);
      }
    }
  }, [round, roundIdentity]);

  useEffect(() => { setMessage(null); setPendingAction(null); }, [roundIdentity]);

  useEffect(() => {
    if (fieldError) document.getElementById(fieldError.field === "annotation" ? "round-annotation-input" : "round-feedback-input")?.focus();
  }, [fieldError]);

  useEffect(() => {
    setWork(null);
    setItemIndex(0);
    setFeedbackByItem({});
    autoRevealStarted.current.clear();
    void reloadWork();
    return () => {
      reloadRequestIdRef.current += 1;
    };
  }, [reloadWork]);

  useEffect(() => {
    setEditorText(editorValue(task, currentDecision?.output));
    setRationale(currentDecision?.rationale ?? "");
    setFeedbackDisposition("");
    setFieldError(null);
    setError(null);
  }, [currentDecision, currentRoundItem?.id, task]);

  const revealFeedback = useCallback(async (): Promise<void> => {
    if (!currentRoundItem || !round.feedback_available || round.status !== "open") return;
    setBusy(true);
    setError(null);
    try {
      const revealed = await revealRoundFeedback(
        round.project_id,
        round.id,
        currentRoundItem.id,
      );
      setFeedbackByItem((current) => ({
        ...current,
        [currentRoundItem.id]: revealed,
      }));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to reveal feedback.");
    } finally {
      setBusy(false);
    }
  }, [
    currentRoundItem,
    round.feedback_available,
    round.id,
    round.project_id,
    round.status,
  ]);

  useEffect(() => {
    if (
      !currentRoundItem ||
      round.status !== "open" ||
      !round.feedback_available ||
      !["immediate_suggestions", "critique", "micro_question"].includes(
        round.assistance_policy,
      ) ||
      feedbackByItem[currentRoundItem.id] ||
      autoRevealStarted.current.has(currentRoundItem.id)
    ) {
      return;
    }
    autoRevealStarted.current.add(currentRoundItem.id);
    void revealFeedback();
  }, [
    currentRoundItem,
    feedbackByItem,
    revealFeedback,
    round.assistance_policy,
    round.feedback_available,
    round.status,
  ]);

  async function saveDecision(): Promise<void> {
    if (!canWriteDecision || !task || !currentRoundItem || currentUserId === null) return;
    let output: unknown;
    try {
      output = parsedOutput(task, editorText);
    } catch (caught) {
      setFieldError({ field: "annotation", message: caught instanceof Error ? caught.message : "Invalid annotation." });
      return;
    }
    if (currentFeedback && !feedbackDisposition) {
      setFieldError({ field: "feedback", message: "Record how the feedback affected this decision." });
      return;
    }
    setBusy(true);
    setError(null);
    setFieldError(null);
    setMessage(null);
    try {
      const decision = await addRoundDecision(round.project_id, {
        roundItemId: currentRoundItem.id,
        output,
        supersedesDecisionId: currentDecision?.id ?? null,
        decisionKind,
        isInitialCheckpoint:
          !currentDecision &&
          round.assistance_policy === "reveal_after_first_pass" &&
          !currentFeedback,
        rationale,
        ...(currentFeedback && feedbackDisposition ? { feedbackDecision: {
          exposureId: currentFeedback.exposure.id,
          decision: feedbackDisposition,
        } } : {}),
      });
      setWork((current) =>
        current
          ? { ...current, decisions: [...current.decisions, decision] }
          : current,
      );
      setFeedbackDisposition("");
      setMessage("Decision saved.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to save decision.");
    } finally {
      setBusy(false);
    }
  }

  async function submit(): Promise<void> {
    if (!canFinalizeDecisions || !work || currentUserId === null) return;
    const decisions = latestDecisions(work.decisions, currentUserId);
    const decisionIds = work.roundItems
      .map((item) => decisions.get(item.id)?.id)
      .filter((id): id is number => id !== undefined);
    if (decisionIds.length !== work.roundItems.length) {
      setError("Complete one current decision for every item before submitting.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const submission = await submitRoundDecisions(
        round.project_id,
        round.id,
        decisionIds,
      );
      setWork((current) =>
        current
          ? { ...current, submissions: [...current.submissions, submission] }
          : current,
      );
      setMessage(`Submission ${submission.sequence} finalized.`);
      setPendingAction(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to submit the round.");
    } finally {
      setBusy(false);
    }
  }

  async function closeRound(): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      await transitionRound(round.project_id, round.id, "closed");
      await onRefresh();
      setMessage("Round closed.");
      setPendingAction(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to close the round.");
    } finally {
      setBusy(false);
    }
  }

  async function publishLabels(): Promise<void> {
    if (!work?.submissions.length || busy || publishedSnapshots.includes(publicationKey)) return;
    setBusy(true);
    setError(null);
    try {
      const labelSet = await publishRoundLabelSet(round.project_id, round.id, {
        name: `${round.name} finalized labels`,
        sourceKind: labelSourceKind,
        submissionIds: work.submissions.map((submission) => submission.id),
      });
      setPublishedSnapshots((current) => [...current, publicationKey]);
      await onRefresh();
      setMessage(`${labelSet.name} v${labelSet.version_number} published.`);
      setPendingAction(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to publish labels.");
    } finally {
      setBusy(false);
    }
  }

  const canReveal =
    round.status === "open" &&
    round.feedback_available &&
    round.assistance_policy !== "blind" &&
    (round.assistance_policy !== "reveal_after_first_pass" || Boolean(currentDecision));

  return (
    <section className="platform-workbench" aria-labelledby="round-workbench-title">
      <header>
        <div>
          <span>Round {round.sequence}</span>
          <h2 id="round-workbench-title">{round.name}</h2>
        </div>
        <div className="platform-workbench-header-actions">
          <PlatformStatus value={round.status} />
          <IconButton label="Close workbench" tooltip="Close workbench" variant="ghost" icon={<X size={18} aria-hidden="true" />} onClick={onClose} />
        </div>
      </header>

      {error ? <p className="platform-dialog-error" role="alert">{error}</p> : null}
      {message ? <p className="platform-workbench-message" role="status">{message}</p> : null}

      {busy && !work ? <p className="platform-inline-empty">Loading round…</p> : null}
      {work && currentRoundItem && currentDatasetItem ? (
        <>
          <div className="platform-workbench-progress">
            <strong>{ownLatest.size} of {work.roundItems.length} decided</strong>
            <span>
              Item {itemIndex + 1} of {work.roundItems.length}
              {currentUserSubmitted
                ? ` · submitted snapshot ${currentUserSubmitted.sequence}`
                : ""}
            </span>
            <span className="platform-workbench-priority">
              {currentRoundItem.selection_rank != null
                ? `Priority #${currentRoundItem.selection_rank}`
                : "Full-dataset item"}
              {currentRoundItem.selection_score != null
                ? ` · ${currentSelectionStrategy === "uncertainty" ? "Uncertainty" : "Selection score"} ${currentRoundItem.selection_score.toFixed(3)}`
                : ""}
              {currentSelectionStrategy ? (
                <span className="platform-strategy-badge">
                  {formatStatus(currentSelectionStrategy)}
                </span>
              ) : null}
            </span>
          </div>

          <div className="platform-workbench-layout">
            <section aria-labelledby="round-source-title">
              <h3 id="round-source-title">Source record</h3>
              <ItemPayload payload={currentDatasetItem.payload} />
            </section>
            <section aria-labelledby="round-decision-title">
              <h3 id="round-decision-title">
                {formatStatus(task.task_kind)}
              </h3>
              {canManage ? (
                <label className="platform-workbench-field">
                  <span>Decision type</span>
                  <NativeSelect
                    value={decisionKind}
                    disabled={busy || (round.status !== "open" && !canAdjudicateClosedRound)}
                    onChange={(event) =>
                      setDecisionKind(
                        event.target.value as AnnotationDecision["decision_kind"],
                      )
                    }
                  >
                    <option value="annotation" disabled={round.status === "closed"}>Annotation</option>
                    <option value="adjudication">Adjudication</option>
                    <option value="correction" disabled={round.status === "closed"}>Correction</option>
                  </NativeSelect>
                </label>
              ) : null}
              <AnnotationEditor
                task={task}
                value={editorText}
                disabled={busy || !canWriteDecision}
                error={fieldError?.field === "annotation" ? fieldError.message : null}
                onChange={(value) => { setEditorText(value); setFieldError(null); }}
              />
              <label className="platform-workbench-field">
                <span>Rationale</span>
                <NativeTextarea
                  rows={3}
                  value={rationale}
                  disabled={busy || !canWriteDecision}
                  onChange={(event) => setRationale(event.target.value)}
                />
              </label>

              {currentFeedback ? (
                <div className="platform-feedback-reveal" role="region" aria-label="Revealed feedback">
                  <div>
                    <strong>Feedback</strong>
                    {currentFeedback.candidate.score !== null ? (
                      <span>Score {currentFeedback.candidate.score.toFixed(3)}</span>
                    ) : null}
                  </div>
                  <details><summary>Feedback output</summary><pre className="platform-code-block">{JSON.stringify(currentFeedback.candidate.output, null, 2)}</pre></details>
                  <button
                    type="button"
                    className="platform-text-action"
                    onClick={() => {
                      setEditorText(editorValue(task, currentFeedback.candidate.output));
                      setFeedbackDisposition("accepted");
                    }}
                  >
                    <Check size={16} aria-hidden="true" />
                    Use suggestion
                  </button>
                  <label className="platform-workbench-field">
                    <span>Feedback decision</span>
                    <NativeSelect
                      aria-label="Feedback decision"
                      id="round-feedback-input"
                      aria-invalid={fieldError?.field === "feedback"}
                      aria-describedby={fieldError?.field === "feedback" ? "round-feedback-error" : undefined}
                      value={feedbackDisposition}
                      onChange={(event) =>
                        setFeedbackDisposition(
                          event.target.value as FeedbackDisposition | "",
                        )
                      }
                    >
                      <option value="">Select</option>
                      <option value="accepted">Accepted</option>
                      <option value="modified">Modified</option>
                      <option value="rejected">Rejected</option>
                      <option value="ignored">Ignored</option>
                    </NativeSelect>
                    {fieldError?.field === "feedback" ? <span id="round-feedback-error" className="platform-dialog-error" role="alert">{fieldError.message}</span> : null}
                  </label>
                </div>
              ) : null}

              <div className="platform-workbench-actions">
                {canReveal && !currentFeedback ? (
                  <button type="button" onClick={() => void revealFeedback()} disabled={busy}>
                    <Eye size={17} aria-hidden="true" />
                    Reveal feedback
                  </button>
                ) : null}
                <button
                  type="button"
                  onClick={() => void saveDecision()}
                  disabled={busy || !canWriteDecision || currentUserId === null}
                >
                  <Save size={17} aria-hidden="true" />
                  Save decision
                </button>
              </div>
            </section>
          </div>

          <footer className="platform-workbench-footer">
            <div>
              <IconButton
                label="Previous item"
                tooltip="Previous item"
                icon={<ChevronLeft size={18} aria-hidden="true" />}
                isDisabled={itemIndex === 0}
                onClick={() => setItemIndex((current) => Math.max(0, current - 1))}
              />
              <IconButton
                label="Next item"
                tooltip="Next item"
                icon={<ChevronRight size={18} aria-hidden="true" />}
                isDisabled={itemIndex >= work.roundItems.length - 1}
                onClick={() => setItemIndex((current) => Math.min(work.roundItems.length - 1, current + 1))}
              />
            </div>
            <button
              type="button"
              onClick={() => { setError(null); setPendingAction("finalize"); }}
              disabled={
                busy ||
                !canFinalizeDecisions ||
                currentUserId === null ||
                ownLatest.size !== work.roundItems.length
              }
            >
              <Send size={17} aria-hidden="true" />
              Finalize submission
            </button>
          </footer>

          {canManage ? (
            <div className="platform-workbench-manager">
              {round.status === "open" ? (
                <button type="button" onClick={() => { setError(null); setPendingAction("close"); }} disabled={busy}>
                  Close round
                </button>
              ) : null}
              {round.status === "closed" ? (
                <>
                  <label className="platform-workbench-field">
                    <span>Label set source</span>
                    <NativeSelect
                      value={labelSourceKind}
                      onChange={(event) =>
                        setLabelSourceKind(
                          event.target.value as "human" | "adjudicated",
                        )
                      }
                    >
                      <option value="human">Human consensus</option>
                      <option value="adjudicated">Adjudicated</option>
                    </NativeSelect>
                  </label>
                  <button
                    type="button"
                    onClick={() => { setError(null); setPendingAction("publish"); }}
                    disabled={busy || !work.submissions.length || publishedSnapshots.includes(publicationKey)}
                  >
                    Publish label set
                  </button>
                </>
              ) : null}
            </div>
          ) : null}
        </>
      ) : work ? (
        <p className="platform-inline-empty">This round has no materialized items.</p>
      ) : null}
      <ConfirmDialog
        open={pendingAction !== null}
        title={pendingAction === "close" ? "Close round?" : pendingAction === "publish" ? "Publish label set?" : "Finalize submission?"}
        description={pendingAction === "close" ? "Closing this round stops further annotation. This action cannot be undone." : pendingAction === "publish" ? "This publishes a permanent label set from the submitted decisions." : "This creates a permanent submission snapshot of your current decisions."}
        confirmLabel={pendingAction === "close" ? "Close round" : pendingAction === "publish" ? "Publish label set" : "Finalize submission"}
        busy={busy}
        error={error}
        onCancel={() => setPendingAction(null)}
        onConfirm={async () => {
          if (pendingAction === "close") await closeRound();
          else if (pendingAction === "publish") await publishLabels();
          else if (pendingAction === "finalize") await submit();
        }}
      />
    </section>
  );
}
