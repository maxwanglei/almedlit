import { ArrowRight } from "lucide-react";

import type { Document, ProjectProgress, TaskAssignment } from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformSection,
  PlatformStats,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData } from "./types";

interface OverviewScreenProps {
  data: PlatformProjectData;
  documents: Document[];
  assignments: TaskAssignment[];
  progress: ProjectProgress | null;
  onOpenData: () => void;
  onOpenTraining: () => void;
  onOpenModels: () => void;
}

export default function OverviewScreen({
  data,
  documents,
  assignments,
  progress,
  onOpenData,
  onOpenTraining,
  onOpenModels,
}: OverviewScreenProps): React.ReactElement {
  const latestModels = data.modelVersions.length;
  const labeled = data.labelSets.reduce((total, item) => total + item.label_count, 0);
  const effectiveModules = new Set(data.projectModules.effective);
  const overviewStats = [
    effectiveModules.has("data")
      ? {
          label: "Source records",
          value:
            data.datasetVersions.reduce((total, item) => total + item.item_count, 0) ||
            documents.length,
          detail: `${data.datasets.length || (documents.length ? 1 : 0)} datasets`,
        }
      : null,
    effectiveModules.has("annotate")
      ? {
          label: "Labels",
          value: labeled,
          detail: `${data.labelSets.length} immutable layers`,
        }
      : null,
    effectiveModules.has("annotate")
      ? {
          label: "Open work",
          value:
            data.rounds.filter((round) => !["closed", "completed"].includes(round.status))
              .length || assignments.filter((item) => item.status !== "completed").length,
          detail: `${data.rounds.length} annotation rounds`,
        }
      : null,
    effectiveModules.has("models")
      ? {
          label: "Model versions",
          value: latestModels,
          detail: `${data.models.length} named models`,
        }
      : null,
  ].filter((item) => item !== null);

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Project overview"
        description="Current data, annotation, training, and model state in one place."
      />
      {overviewStats.length ? <PlatformStats items={overviewStats} /> : null}

      <PlatformSection
        title="Future modules"
        description="Planned after the core model-training workflow is released."
      >
        <div className="platform-planned-state" aria-label="Planned modules">
          <PlatformStatus value="planned" />
          <div>
            <strong>Learning Loop and Guideline Learning</strong>
            <p>
              Future releases will connect model feedback, repeated annotation,
              error analysis, and guideline revision.
            </p>
          </div>
        </div>
      </PlatformSection>

      {effectiveModules.has("data") ? (
        <PlatformSection
          title="Data readiness"
          description="Training uses immutable source, label, and split versions."
        >
          {data.datasetVersions.length || documents.length ? (
            <div className="platform-readiness-list">
              <div>
                <span className="platform-readiness-mark complete" aria-hidden="true" />
                <div>
                  <strong>Source data available</strong>
                  <p>{data.datasetVersions.length || 1} versioned source snapshots</p>
                </div>
              </div>
              <div>
                <span
                  className={`platform-readiness-mark ${data.labelSets.length ? "complete" : ""}`}
                  aria-hidden="true"
                />
                <div>
                  <strong>Label layer</strong>
                  <p>
                    {data.labelSets.length
                      ? `${data.labelSets.length} label versions retained`
                      : "Optional for unlabeled or annotation-first work"}
                  </p>
                </div>
              </div>
              <div>
                <span
                  className={`platform-readiness-mark ${data.splitMaps.length ? "complete" : ""}`}
                  aria-hidden="true"
                />
                <div>
                  <strong>Protected split policy</strong>
                  <p>
                    {data.splitMaps.length
                      ? "Stable train, validation, test, and pool assignments"
                      : "Create before comparative training"}
                  </p>
                </div>
              </div>
              <PlatformRouteLink
                href={`/projects/${data.projectModules.project_id}/data`}
                className="platform-text-action"
                onNavigate={onOpenData}
              >
                Review data
              </PlatformRouteLink>
            </div>
          ) : (
            <PlatformEmpty
              title="No dataset yet"
              detail="Import a public dataset, upload structured files, or snapshot the project corpus."
              actionLabel="Add data"
              onAction={onOpenData}
            />
          )}
        </PlatformSection>
      ) : null}

      {effectiveModules.has("train") || effectiveModules.has("models") ? (
        <PlatformSection
          title="Model development"
          description="Training and model registry work open in dedicated workspaces while retaining this project as context."
        >
          <div className="platform-module-links">
            {effectiveModules.has("train") ? (
              <PlatformRouteLink
                href={`/training?projectId=${data.projectModules.project_id}`}
                onNavigate={onOpenTraining}
              >
                <span>
                  <strong>Training</strong>
                  <small>
                    {data.trainingRuns.length
                      ? `${data.trainingRuns.length} recorded runs`
                      : "Prepare data and launch a run"}
                  </small>
                </span>
                <ArrowRight size={17} aria-hidden="true" />
              </PlatformRouteLink>
            ) : null}
            {effectiveModules.has("models") ? (
              <PlatformRouteLink
                href={`/models?projectId=${data.projectModules.project_id}`}
                onNavigate={onOpenModels}
              >
                <span>
                  <strong>Models</strong>
                  <small>
                    {data.models.length
                      ? `${data.models.length} named models`
                      : "Open the workspace model registry"}
                  </small>
                </span>
                <ArrowRight size={17} aria-hidden="true" />
              </PlatformRouteLink>
            ) : null}
          </div>
        </PlatformSection>
      ) : null}

      {progress ? (
        <p className="platform-footnote">
          Legacy assignment progress remains available while existing projects are migrated.
        </p>
      ) : null}
    </div>
  );
}
