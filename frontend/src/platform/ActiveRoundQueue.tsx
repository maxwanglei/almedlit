import { PlatformRouteLink, PlatformStatus } from "./components";
import type { RoundWorkContext } from "./types";

export default function ActiveRoundQueue({
  contexts,
  onOpenRound,
}: {
  contexts: RoundWorkContext[];
  onOpenRound: (roundId: number) => void;
}): React.ReactElement | null {
  const rounds = [...contexts].sort(
    (left, right) => right.round.sequence - left.round.sequence,
  );

  if (!rounds.length) return null;

  return (
    <section className="aw-panel aw-active-rounds" aria-labelledby="aw-active-rounds-title">
      <header className="aw-active-rounds-header">
        <div>
          <h2 id="aw-active-rounds-title">
            Annotation rounds <span>{rounds.length}</span>
          </h2>
          <p>Assigned project rounds ready for your work.</p>
        </div>
      </header>
      <div className="aw-active-round-list">
        {rounds.map((context) => {
          const { round } = context;
          return (
            <article className="aw-active-round-row" key={round.id}>
              <div className="aw-active-round-name">
                <strong>{round.name}</strong>
                <span>{context.project.name} · Round {round.sequence}</span>
              </div>
              <dl className="aw-active-round-meta">
                <div>
                  <dt>Task</dt>
                  <dd>
                    {context.task.name} · v{context.task_version.version_number}
                  </dd>
                </div>
                <div>
                  <dt>Guideline</dt>
                  <dd>
                    {context.guideline
                      ? `${context.guideline.name} · v${context.guideline.version_number}`
                      : "Not pinned"}
                  </dd>
                </div>
              </dl>
              <PlatformStatus value={round.status} />
              <PlatformRouteLink
                href={`/my-work/rounds/${round.id}`}
                className="aw-secondary-action"
                onNavigate={() => onOpenRound(round.id)}
              >
                Open round
              </PlatformRouteLink>
            </article>
          );
        })}
      </div>
    </section>
  );
}
