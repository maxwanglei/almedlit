import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData } from "./types";

export default function GuidelinesScreen({
  data,
  onCreate,
}: {
  data: PlatformProjectData;
  onCreate: () => void;
}): React.ReactElement {
  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Guidelines"
        description="Feedback-linked proposals, immutable revisions, pilots, and controlled activation."
        actionLabel="New guideline"
        actionDisabled={!data.taskDefinitions.length}
        onAction={onCreate}
      />

      <PlatformSection
        title="Guideline versions"
        description="Activating a new version never changes an open annotation round."
      >
        {data.guidelines.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Guideline versions table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Guideline</th>
                  <th scope="col">Revision</th>
                  <th scope="col">Status</th>
                  <th scope="col">Task version</th>
                  <th scope="col">Rationale</th>
                  <th scope="col">Impact check</th>
                </tr>
              </thead>
              <tbody>
                {data.guidelines.flatMap((guideline) => {
                  const revisions = data.guidelineRevisions
                    .filter((revision) => revision.guideline_id === guideline.id)
                    .sort((left, right) => right.version_number - left.version_number);
                  if (!revisions.length) {
                    return [
                      <tr key={`${guideline.id}:empty`}>
                        <td><strong>{guideline.name}</strong></td>
                        <td colSpan={5}>No revision published</td>
                      </tr>,
                    ];
                  }
                  return revisions.map((revision, index) => {
                    const impact = data.guidelineImpacts.find(
                      (item) => item.guideline_revision_id === revision.id,
                    );
                    const taskVersion = data.taskVersions.find(
                      (item) => item.id === revision.task_version_id,
                    );
                    return (
                      <tr key={revision.id}>
                        <td><strong>{index === 0 ? guideline.name : ""}</strong></td>
                        <td>v{revision.version_number}</td>
                        <td><PlatformStatus value={revision.status} /></td>
                        <td>
                          {taskVersion
                            ? `${taskVersion.task_kind.replace(/_/g, " ")} · v${taskVersion.version_number}`
                            : `Version ${revision.task_version_id}`}
                        </td>
                        <td>{revision.rationale ?? "No rationale recorded"}</td>
                        <td>
                          {impact ? (
                            <PlatformStatus
                              value={
                                impact.status === "completed"
                                  ? impact.passed
                                    ? "passed"
                                    : "failed"
                                  : impact.status
                              }
                            />
                          ) : (
                            "Required before activation"
                          )}
                        </td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No task-scoped guidelines"
            detail="Create a guideline after defining a task. Generated suggestions remain proposals until reviewed."
            actionLabel="New guideline"
            onAction={onCreate}
          />
        )}
      </PlatformSection>

      <PlatformSection
        title="Change proposals"
        description="Every proposal cites the feedback events that motivated it."
      >
        {data.guidelineProposals.length ? (
          <div className="platform-proposal-list">
            {data.guidelineProposals.map((proposal) => (
              <article key={proposal.id}>
                <div>
                  <strong>{proposal.rationale}</strong>
                  <span>{proposal.feedback_event_ids.length} supporting events</span>
                </div>
                <PlatformStatus value={proposal.status} />
              </article>
            ))}
          </div>
        ) : (
          <p className="platform-inline-empty">
            No proposals. Corrections, disagreements, evaluation failures, and LLM critiques can feed this queue.
          </p>
        )}
      </PlatformSection>
    </div>
  );
}
