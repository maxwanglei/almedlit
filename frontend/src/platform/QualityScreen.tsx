import type {
  Document,
  ProjectProgress,
  TaskAssignment,
} from "@/types/api";

import EvidenceAdjudicationPanel from "./EvidenceAdjudicationPanel";
import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStats,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData } from "./types";

export default function QualityScreen({
  data,
  progress,
  projectId,
  documents,
  assignments,
  allowSoloGold,
}: {
  data: PlatformProjectData;
  progress: ProjectProgress | null;
  projectId: number;
  documents: Document[];
  assignments: TaskAssignment[];
  allowSoloGold: boolean;
}): React.ReactElement {
  const finalRounds = data.rounds.filter((round) =>
    ["closed", "completed"].includes(round.status),
  );
  const reviewableRounds = data.rounds.filter((round) =>
    ["submitted", "adjudication_ready", "adjudicated"].includes(round.status),
  );

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Quality & Review"
        description="Review round readiness, adjudication state, and quality evidence without exposing private annotator work."
      />

      <PlatformStats
        items={[
          {
            label: "Reviewable rounds",
            value: reviewableRounds.length,
            detail: "Submitted or awaiting adjudication",
          },
          {
            label: "Final rounds",
            value: finalRounds.length,
            detail: "Closed or completed",
          },
          {
            label: "Task contracts",
            value: data.taskVersions.length,
            detail: "Version-pinned quality scopes",
          },
        ]}
      />

      <PlatformSection
        title="Round review"
        description="Comparison and adjudication must remain pinned to one task, dataset, guideline, and assignment round."
      >
        {data.rounds.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Round quality table"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Round</th>
                  <th scope="col">Status</th>
                  <th scope="col">Task</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Guideline</th>
                </tr>
              </thead>
              <tbody>
                {[...data.rounds]
                  .sort((left, right) => right.sequence - left.sequence)
                  .map((round) => (
                    <tr key={round.id}>
                      <td data-label="Round" data-priority="identity">
                        <strong>{round.name}</strong>
                        <span>Round {round.sequence}</span>
                      </td>
                      <td data-label="Status" data-priority="status">
                        <PlatformStatus value={round.status} />
                      </td>
                      <td data-label="Task">v{round.task_version_id}</td>
                      <td data-label="Dataset">v{round.dataset_version_id}</td>
                      <td data-label="Guideline">
                        {round.guideline_revision_id
                          ? `v${round.guideline_revision_id}`
                          : "Not pinned"}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No review scope yet"
            detail="Quality and adjudication records appear after an annotation round is created."
          />
        )}
      </PlatformSection>

      <EvidenceAdjudicationPanel
        projectId={projectId}
        documents={documents}
        assignments={assignments}
        allowSoloGold={allowSoloGold}
      />

      {progress ? (
        <p className="platform-footnote">
          Legacy progress remains available while existing assignment cohorts are migrated.
        </p>
      ) : null}
    </div>
  );
}
