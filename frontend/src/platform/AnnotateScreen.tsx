import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData } from "./types";

export default function AnnotateScreen({
  view,
  data,
  onCreateRound,
  onCreateTask,
  onOpenRound,
  currentUserId,
  canManage,
}: {
  view: "tasks" | "rounds";
  data: PlatformProjectData;
  onCreateRound: () => void;
  onCreateTask: () => void;
  onOpenRound: (roundId: number) => void;
  currentUserId: number | null;
  canManage: boolean;
}): React.ReactElement {
  const showingTasks = view === "tasks";

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title={showingTasks ? "Tasks" : "Team & Rounds"}
        description={
          showingTasks
            ? "Define immutable task contracts for the annotation interfaces enabled in this project."
            : "Organize assignments, monitor rounds, and retain immutable submission history."
        }
        actionLabel={
          canManage ? (showingTasks ? "New task" : "New round") : undefined
        }
        actionDisabled={
          !showingTasks &&
          (!data.datasetVersions.length || !data.taskVersions.length)
        }
        onAction={
          canManage
            ? showingTasks
              ? onCreateTask
              : onCreateRound
            : undefined
        }
      />

      {showingTasks ? <PlatformSection
        title="Task contracts"
        description="Each version pins the NLP task kind, schemas, label rules, annotation UI, metrics, and compatible trainers."
      >
        {data.taskDefinitions.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Task contracts table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Task</th>
                  <th scope="col">Version</th>
                  <th scope="col">Kind</th>
                  <th scope="col">Metrics</th>
                  <th scope="col">Fingerprint</th>
                </tr>
              </thead>
              <tbody>
                {data.taskDefinitions.flatMap((task) =>
                  data.taskVersions
                    .filter((version) => version.task_definition_id === task.id)
                    .map((version, index) => (
                      <tr key={version.id}>
                        <td><strong>{index === 0 ? task.name : ""}</strong></td>
                        <td>v{version.version_number}</td>
                        <td>{version.task_kind.replace(/_/g, " ")}</td>
                        <td>{version.metrics.join(", ") || "Not set"}</td>
                        <td><code>{version.content_hash.slice(0, 10)}</code></td>
                      </tr>
                    )),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No task contract"
            detail="Define classification, regression, token labeling, extraction, ranking, generation, or instruction tuning."
            actionLabel={canManage ? "Create task" : undefined}
            onAction={canManage ? onCreateTask : undefined}
          />
        )}
      </PlatformSection> : null}

      {!showingTasks ? <PlatformSection
        title="Annotation rounds"
        description="Each round pins its task, dataset, assignments, and submitted decisions."
      >
        {data.rounds.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Annotation rounds table"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Round</th>
                  <th scope="col">Status</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Task</th>
                  <th scope="col">Guideline</th>
                  <th scope="col">Reason</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {[...data.rounds]
                  .sort((left, right) => right.sequence - left.sequence)
                  .map((round) => {
                    const assigned =
                      round.open_to_all_annotators ||
                      (currentUserId !== null &&
                        round.annotator_user_ids.includes(currentUserId));
                    const canOpen = canManage || assigned;
                    return (
                    <tr key={round.id}>
                      <td data-label="Round" data-priority="identity">
                        <strong>{round.name}</strong>
                        <span>Round {round.sequence}</span>
                      </td>
                      <td data-label="Status" data-priority="status">
                        <PlatformStatus value={round.status} />
                      </td>
                      <td data-label="Dataset">v{round.dataset_version_id}</td>
                      <td data-label="Task">v{round.task_version_id}</td>
                      <td data-label="Guideline">
                        {round.guideline_revision_id
                          ? `v${round.guideline_revision_id}`
                          : "Not pinned"}
                      </td>
                      <td data-label="Reason">{round.reason ?? "Initial annotation"}</td>
                      <td data-label="Action" data-priority="action">
                        <button
                          type="button"
                          className="platform-text-action"
                          disabled={!canOpen}
                          onClick={() => onOpenRound(round.id)}
                        >
                          {round.status === "open" && assigned ? "Annotate" : "Review"}
                        </button>
                      </td>
                    </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No rounds yet"
            detail={
              canManage
                ? "Create a blind annotation round after adding a task and dataset."
                : "No annotation rounds have been assigned to this project."
            }
            actionLabel={canManage ? "Create round" : undefined}
            onAction={canManage ? onCreateRound : undefined}
          />
        )}
      </PlatformSection> : null}
    </div>
  );
}
