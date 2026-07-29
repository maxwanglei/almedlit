import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type { LearningCycle, PlatformProjectData } from "./types";

const CYCLE_STAGES = [
  "pool",
  "select",
  "annotate",
  "resolve",
  "reflect",
  "publish_guideline",
  "train",
  "evaluate",
] as const;

function stageIndex(cycle: LearningCycle): number {
  const index = CYCLE_STAGES.indexOf(cycle.current_stage as (typeof CYCLE_STAGES)[number]);
  return index < 0 ? 0 : index;
}

export default function LearningScreen({
  data,
  onCreate,
  onOpenRound,
}: {
  data: PlatformProjectData;
  onCreate: () => void;
  onOpenRound: () => void;
}): React.ReactElement {
  const latest = [...data.cycles].sort((left, right) => right.sequence - left.sequence)[0];
  const currentIndex = latest ? stageIndex(latest) : -1;
  const baselineVersion = latest?.baseline_model_version_id
    ? data.modelVersions.find((version) => version.id === latest.baseline_model_version_id)
    : null;
  const baselineModel = baselineVersion
    ? data.models.find((model) => model.id === baselineVersion.registered_model_id)
    : null;
  const feedbackSources = latest
    ? [
        ...(baselineVersion
          ? [`${baselineModel?.name ?? "Registered model"} v${baselineVersion.version_number}`]
          : []),
        ...(latest.feedback_sources ?? []).map((source) => source.name),
      ]
    : [];

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Learning Loop"
        description="Repeatable selection, annotation, guideline, training, and evaluation cycles."
        actionLabel="New cycle"
        actionDisabled={!data.datasetVersions.length || !data.taskVersions.length}
        onAction={onCreate}
      />

      <PlatformSection
        title={latest ? `Cycle ${latest.sequence}: ${latest.name}` : "Current cycle"}
        description="Every stage records immutable inputs and outputs."
      >
        {latest ? (
          <>
            <ol className="platform-cycle-track" aria-label="Learning cycle stages">
              {CYCLE_STAGES.map((stage, index) => {
                const state =
                  index < currentIndex ? "complete" : index === currentIndex ? "current" : "future";
                return (
                  <li key={stage} data-state={state}>
                    <span aria-hidden="true">{index + 1}</span>
                    <strong>{stage.replace(/_/g, " ")}</strong>
                  </li>
                );
              })}
            </ol>
            <div className="platform-cycle-meta">
              <PlatformStatus value={latest.status} />
              <span>Task version {latest.task_version_id}</span>
              <span>Dataset version {latest.source_dataset_version_id}</span>
              <span>Feedback: {feedbackSources.join(", ")}</span>
            </div>
          </>
        ) : (
          <PlatformEmpty
            title="No learning cycle yet"
            detail="A cycle may start with an internal model, an external LLM, rules, dictionaries, or human disagreement."
            actionLabel="Plan first cycle"
            onAction={onCreate}
          />
        )}
      </PlatformSection>

      <PlatformSection
        title="Annotation rounds"
        description="The same source items can appear in multiple isolated rounds."
      >
        {data.rounds.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Learning-cycle annotation rounds table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Round</th>
                  <th scope="col">Status</th>
                  <th scope="col">Scope</th>
                  <th scope="col">Assistance</th>
                  <th scope="col">Guideline</th>
                  <th scope="col">Feedback</th>
                </tr>
              </thead>
              <tbody>
                {[...data.rounds]
                  .sort((left, right) => right.sequence - left.sequence)
                  .map((round) => (
                    <tr key={round.id}>
                      <td><strong>{round.name}</strong><span>Round {round.sequence}</span></td>
                      <td><PlatformStatus value={round.status} /></td>
                      <td>{round.reannotation_mode.replace(/_/g, " ")}</td>
                      <td>{round.assistance_policy.replace(/_/g, " ")}</td>
                      <td>{round.guideline_revision_id ? `v${round.guideline_revision_id}` : "None"}</td>
                      <td>{round.feedback_set_version_id ? `Set ${round.feedback_set_version_id}` : "Blind"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No annotation rounds"
            detail="Create a targeted active-learning batch or a full-dataset re-annotation round."
            actionLabel="Create round"
            onAction={onOpenRound}
          />
        )}
      </PlatformSection>

      <PlatformSection
        title="Cycle history"
        description="Earlier cycles remain available for comparison and audit."
      >
        {data.cycles.length ? (
          <div className="platform-history-list">
            {[...data.cycles]
              .sort((left, right) => right.sequence - left.sequence)
              .map((cycle) => (
                <div key={cycle.id}>
                  <span>Cycle {cycle.sequence}</span>
                  <strong>{cycle.name}</strong>
                  <PlatformStatus value={cycle.current_stage} />
                </div>
              ))}
          </div>
        ) : (
          <p className="platform-inline-empty">Cycle history will appear here.</p>
        )}
      </PlatformSection>
    </div>
  );
}
