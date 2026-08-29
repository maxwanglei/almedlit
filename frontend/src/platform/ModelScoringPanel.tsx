import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import { RefreshCw } from "lucide-react";

import { getFeedbackRun } from "./api";
import {
  PlatformEmpty,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type { FeedbackRun, PlatformProjectData } from "./types";

const POLL_INTERVAL_MS = 3_000;
const POLL_TIMEOUT_MS = 120_000;

function mergeRuns(
  current: FeedbackRun[],
  refreshed: FeedbackRun[],
): FeedbackRun[] {
  const byId = new Map(current.map((run) => [run.id, run]));
  refreshed.forEach((run) => byId.set(run.id, run));
  return [...byId.values()].sort((left, right) => right.id - left.id);
}

export default function ModelScoringPanel({
  projectId,
  data,
  onCreate,
  onRefresh,
}: {
  projectId: number;
  data: PlatformProjectData;
  onCreate: () => void;
  onRefresh: () => Promise<void>;
}): React.ReactElement {
  const [runs, setRuns] = useState<FeedbackRun[]>(() =>
    data.feedbackRuns.filter((run) => run.producer_type === "registered_model"),
  );
  const [pollExpired, setPollExpired] = useState(false);
  const [pollGeneration, setPollGeneration] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const projectRef = useRef(projectId);
  const runsRef = useRef(runs);
  projectRef.current = projectId;
  runsRef.current = runs;

  useEffect(() => {
    setRuns(
      data.feedbackRuns.filter((run) => run.producer_type === "registered_model"),
    );
  }, [data.feedbackRuns, projectId]);

  const activeRuns = useMemo(
    () => runs.filter((run) => run.status === "queued" || run.status === "running"),
    [runs],
  );
  const activeRunKey = activeRuns.map((run) => run.id).join(":");

  const refreshActive = useCallback(async (): Promise<void> => {
    const currentActiveRuns = runsRef.current.filter(
      (run) => run.status === "queued" || run.status === "running",
    );
    if (!currentActiveRuns.length) {
      await onRefresh();
      return;
    }
    const requestedProjectId = projectId;
    const refreshed = await Promise.all(
      currentActiveRuns.map((run) => getFeedbackRun(projectId, run.id)),
    );
    if (projectRef.current !== requestedProjectId) return;
    setRuns((current) => mergeRuns(current, refreshed));
    if (refreshed.some((run) => run.status === "completed" || run.status === "failed")) {
      await onRefresh();
    }
  }, [onRefresh, projectId]);

  useEffect(() => {
    if (!activeRunKey || pollExpired) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const deadline = Date.now() + POLL_TIMEOUT_MS;

    const poll = async (): Promise<void> => {
      if (cancelled) return;
      if (Date.now() >= deadline) {
        setPollExpired(true);
        return;
      }
      try {
        await refreshActive();
      } catch {
        // A transient polling failure leaves the last known state visible; the
        // normal project error surface handles explicit refresh failures.
      } finally {
        if (!cancelled) timer = setTimeout(() => void poll(), POLL_INTERVAL_MS);
      }
    };

    timer = setTimeout(() => void poll(), POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [activeRunKey, pollExpired, pollGeneration, refreshActive]);

  async function manualRefresh(): Promise<void> {
    setRefreshing(true);
    setPollExpired(false);
    setPollGeneration((value) => value + 1);
    try {
      await refreshActive();
    } finally {
      setRefreshing(false);
    }
  }

  return (
    <PlatformSection
      title="Model scoring"
      description="Score every item with a compatible TF-IDF classifier, then use the completed feedback set for uncertainty selection. Protected items are scored but remain excluded when a round is selected."
      action={
        <Button
          label="Score with model"
          variant="secondary"
          onClick={onCreate}
        />
      }
    >
      {pollExpired ? (
        <p className="platform-scoring-notice" role="status">
          Automatic updates paused after two minutes. The scoring job is still active.
          <Button
            label="Refresh status"
            icon={<RefreshCw size={16} />}
            variant="ghost"
            isDisabled={refreshing}
            isLoading={refreshing}
            onClick={() => void manualRefresh()}
          />
        </p>
      ) : null}
      {runs.length ? (
        <section
          className="platform-table-scroll platform-table-scroll--summary"
          role="region"
          aria-label="Model scoring runs"
          tabIndex={0}
        >
          <table className="platform-table platform-table--summary">
            <thead>
              <tr>
                <th scope="col">Model</th>
                <th scope="col">Status</th>
                <th scope="col">Dataset</th>
                <th scope="col">Candidates</th>
                <th scope="col">Result</th>
                <th scope="col">Details</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const version = data.modelVersions.find(
                  (item) => item.id === run.model_version_id,
                );
                const model = data.models.find(
                  (item) => item.id === version?.registered_model_id,
                );
                const feedbackSet = data.feedbackSets.find(
                  (item) => item.id === run.output_feedback_set_version_id,
                );
                return (
                  <tr key={run.id}>
                    <td data-label="Model" data-priority="identity">
                      <strong>{model?.name ?? `Model version ${run.model_version_id}`}</strong>
                      <span>{version ? `v${version.version_number}` : `Run ${run.id}`}</span>
                    </td>
                    <td data-label="Status" data-priority="status">
                      <PlatformStatus value={run.status} />
                    </td>
                    <td data-label="Dataset">v{run.dataset_version_id}</td>
                    <td data-label="Candidates">{feedbackSet?.candidate_count ?? "—"}</td>
                    <td data-label="Result">
                      {feedbackSet ? (
                        <a
                          className="platform-text-action"
                          href={`/projects/${projectId}/rounds?feedbackSetId=${feedbackSet.id}`}
                        >
                          Feedback set v{feedbackSet.version_number}
                        </a>
                      ) : "—"}
                    </td>
                    <td data-label="Details">
                      {run.failure_reason ?? (run.cycle_id ? `Cycle ${run.cycle_id}` : "Standalone")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ) : (
        <PlatformEmpty
          title="No model scores yet"
          detail="Create a reusable feedback set from a completed TF-IDF classification model."
        />
      )}
    </PlatformSection>
  );
}
