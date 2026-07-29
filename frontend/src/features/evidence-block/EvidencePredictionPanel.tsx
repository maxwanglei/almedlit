import type {
  EvidenceCandidatePrediction,
  PredictionReviewAction,
} from "@/types/api";

interface EvidencePredictionPanelProps {
  predictions: EvidenceCandidatePrediction[];
  loading: boolean;
  busy: boolean;
  selectionAvailable: boolean;
  selectedPredictionId: number | null;
  onSelect: (prediction: EvidenceCandidatePrediction) => void;
  onReview: (
    prediction: EvidenceCandidatePrediction,
    action: PredictionReviewAction,
  ) => void | Promise<void>;
  onRefresh: () => void | Promise<void>;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export default function EvidencePredictionPanel({
  predictions,
  loading,
  busy,
  selectionAvailable,
  selectedPredictionId,
  onSelect,
  onReview,
  onRefresh,
}: EvidencePredictionPanelProps): React.ReactElement {
  const pendingCount = predictions.filter(
    (prediction) => prediction.review_status === "pending",
  ).length;
  return (
    <section className="eb-predictions" aria-label="Model prediction candidates">
      <header>
        <div>
          <strong>{predictions.length} model candidates</strong>
          <span>{pendingCount} pending review</span>
        </div>
        <button type="button" onClick={() => void onRefresh()} disabled={loading || busy}>
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </header>
      {predictions.length === 0 ? (
        <p className="aw-empty">No candidates for this run, document, and target.</p>
      ) : null}
      {predictions.map((prediction) => {
        const pending = prediction.review_status === "pending";
        const selected = selectedPredictionId === prediction.id;
        const latestReview = prediction.reviews[prediction.reviews.length - 1];
        return (
          <article
            className={`eb-prediction-card ${selected ? "active" : ""} ${prediction.review_status}`}
            key={prediction.id}
          >
            <button
              className="eb-prediction-select"
              type="button"
              onClick={() => onSelect(prediction)}
              aria-pressed={selected}
            >
              <strong>
                Sentences {prediction.start_sentence_ordinal + 1}–
                {prediction.end_sentence_ordinal + 1}
              </strong>
              <span>
                {percent(prediction.block_confidence)} confidence · {percent(prediction.uncertainty)}{" "}
                uncertainty
              </span>
              <small>
                {prediction.review_status} for you · run #{prediction.run_id} · checkpoint #{prediction.checkpoint_id}
              </small>
            </button>
            <div className="eb-prediction-actions">
              <button
                type="button"
                onClick={() => void onReview(prediction, "accept")}
                disabled={!pending || busy}
                aria-label={`Accept prediction ${prediction.id}`}
              >
                Accept
              </button>
              <button
                type="button"
                onClick={() => void onReview(prediction, "modify")}
                disabled={!pending || busy || !selectionAvailable}
                aria-label={`Modify prediction ${prediction.id} using selected sentences`}
              >
                Use selected bounds
              </button>
              <button
                className="danger"
                type="button"
                onClick={() => void onReview(prediction, "reject")}
                disabled={!pending || busy}
                aria-label={`Reject prediction ${prediction.id}`}
              >
                Reject
              </button>
            </div>
            <details>
              <summary>Traceability</summary>
              <dl>
                <div>
                  <dt>Candidate</dt>
                  <dd>#{prediction.id}</dd>
                </div>
                <div>
                  <dt>Decoder</dt>
                  <dd>{prediction.decoder_version}</dd>
                </div>
                <div>
                  <dt>Windows</dt>
                  <dd>{prediction.source_window_ids.join(", ") || "—"}</dd>
                </div>
                <div>
                  <dt>Structure</dt>
                  <dd>#{prediction.structure_version_id}</dd>
                </div>
                <div>
                  <dt>Diagnostics</dt>
                  <dd>
                    {prediction.diagnostics_artifact_id
                      ? `Artifact #${prediction.diagnostics_artifact_id}`
                      : "Run-level artifact"}
                  </dd>
                </div>
              </dl>
              {latestReview ? (
                <p>
                  Review r{latestReview.revision}: {latestReview.action}
                  {latestReview.resulting_annotation_id
                    ? ` → annotation #${latestReview.resulting_annotation_id}`
                    : ""}
                </p>
              ) : null}
            </details>
          </article>
        );
      })}
    </section>
  );
}
