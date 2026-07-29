import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData } from "./types";

interface ActivityItem {
  id: string;
  when: string | null | undefined;
  kind: string;
  name: string;
  status: string;
}

export default function ActivityScreen({
  data,
}: {
  data: PlatformProjectData;
}): React.ReactElement {
  const items: ActivityItem[] = [
    ...data.rounds.map((round) => ({
      id: `round:${round.id}`,
      when: round.updated_at ?? round.created_at,
      kind: "Annotation round",
      name: round.name,
      status: round.status,
    })),
    ...data.trainingRuns.map((run) => ({
      id: `run:${run.id}`,
      when: run.updated_at ?? run.created_at,
      kind: "Training run",
      name: `Training run ${run.id}`,
      status: run.status,
    })),
    ...data.modelEvaluations.map((evaluation) => ({
      id: `evaluation:${evaluation.id}`,
      when: evaluation.updated_at ?? evaluation.created_at,
      kind: "Model evaluation",
      name: `${evaluation.split_name} · ${evaluation.row_count} records`,
      status: evaluation.status,
    })),
  ].sort((left, right) => String(right.when ?? "").localeCompare(String(left.when ?? "")));

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Activity"
        description="Released annotation and model-development history for this project."
      />
      <PlatformSection title="Recent activity">
        {items.length ? (
          <div className="platform-activity-list">
            {items.map((item) => (
              <div key={item.id}>
                <time dateTime={item.when ?? undefined}>
                  {item.when ? new Date(item.when).toLocaleString() : "Time unavailable"}
                </time>
                <span>{item.kind}</span>
                <strong>{item.name}</strong>
                <PlatformStatus value={item.status} />
              </div>
            ))}
          </div>
        ) : (
          <PlatformEmpty
            title="No versioned activity yet"
            detail="New annotation rounds, training runs, and model evaluations will appear here."
          />
        )}
      </PlatformSection>
    </div>
  );
}
