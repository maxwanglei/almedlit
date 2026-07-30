import {
  ArrowRight,
  BookOpenCheck,
  BrainCircuit,
  Cpu,
  GitBranch,
  Radar,
  ScanSearch,
  type LucideIcon,
} from "lucide-react";

import type { CapabilityKey } from "@/auth/capabilities";
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

interface RoadmapCapability {
  id: string;
  title: string;
  status: "foundation" | "partial" | "planned";
  description: string;
  milestones: readonly string[];
  capabilities: readonly {
    key: CapabilityKey;
    label: string;
  }[];
  icon: LucideIcon;
}

const ROADMAP_CAPABILITIES = [
  {
    id: "inference",
    title: "Inference",
    status: "foundation",
    description:
      "Versioned batch runs and assignment-scoped prediction review exist; the dedicated manager workflow is staged.",
    milestones: [
      "Batch prediction from immutable checkpoints",
      "Candidate windows and append-only human review",
      "Prompt-based LLM inference after endpoint integration",
    ],
    capabilities: [{ key: "inference", label: "Inference" }],
    icon: ScanSearch,
  },
  {
    id: "active-learning",
    title: "Active Learning",
    status: "planned",
    description:
      "Strategy execution, pool management, and stopping criteria will close the annotation-to-model loop.",
    milestones: [
      "Uncertainty, committee, diversity, and hybrid ranking",
      "Document-, sentence-, and span-level selection",
      "Budgets, convergence, and prioritized annotation rounds",
    ],
    capabilities: [{ key: "active_learning", label: "Active learning" }],
    icon: Radar,
  },
  {
    id: "co-learning",
    title: "Co-learning",
    status: "partial",
    description:
      "Correction-derived error and guideline records exist; the wider human-learning feedback system remains staged.",
    milestones: [
      "Confident-disagreement review queue",
      "Post-annotation critique and similar cases",
      "Personal error analysis and calibration",
    ],
    capabilities: [{ key: "co_learning", label: "Co-learning" }],
    icon: BrainCircuit,
  },
  {
    id: "lineage-export",
    title: "Lineage & Export",
    status: "foundation",
    description:
      "Immutable snapshots, artifact lineage, and authenticated exports exist; reproduction and diff reporting UI is staged.",
    milestones: [
      "Corpus and annotation-set snapshots",
      "Training and inference provenance graphs",
      "Versioned exports and paper-ready reports",
    ],
    capabilities: [
      { key: "lineage", label: "Lineage" },
      { key: "export", label: "Export" },
    ],
    icon: GitBranch,
  },
  {
    id: "guideline-learning",
    title: "Guideline Learning",
    status: "foundation",
    description:
      "Versioned guidelines and correction-derived learning records exist; automated proposals and richer collaborative editing are staged.",
    milestones: [
      "Correction and disagreement clustering",
      "Manager-reviewed clarification proposals",
      "Micro-training, impact checks, and retraining actions",
    ],
    capabilities: [{ key: "co_learning", label: "Co-learning" }],
    icon: BookOpenCheck,
  },
  {
    id: "compute-llm",
    title: "HPC & LLM Serving",
    status: "partial",
    description:
      "Local and minimal SSH/SLURM execution exist; live cluster hardening and managed vLLM serving come later.",
    milestones: [
      "Verified, image-bound runtime profiles",
      "Remote submit, poll, cancel, and retrieval",
      "External then managed vLLM endpoints",
    ],
    capabilities: [
      { key: "hpc_training", label: "HPC training" },
      { key: "llm_serving", label: "LLM serving" },
    ],
    icon: Cpu,
  },
] as const satisfies readonly RoadmapCapability[];

function capabilityAvailability(
  item: RoadmapCapability,
  workspaceCapabilities: ReadonlySet<string>,
): {
  available: boolean;
  label: string;
} {
  const available = item.capabilities.filter((capability) =>
    workspaceCapabilities.has(capability.key),
  );
  if (!available.length) {
    return {
      available: false,
      label: "Not available in this workspace",
    };
  }
  return {
    available: true,
    label: `${available.map((capability) => capability.label).join(" and ")} ${
      available.length === 1 ? "capability" : "capabilities"
    } available`,
  };
}

function ResearchLoopRoadmap({
  workspaceCapabilities,
}: {
  workspaceCapabilities: ReadonlySet<string>;
}): React.ReactElement {
  return (
    <PlatformSection
      title="Research loop roadmap"
      description="Current foundations and planned UI remain visible without exposing unfinished controls."
    >
      <div
        className="platform-roadmap-grid"
        aria-label="Research loop capability roadmap"
      >
        {ROADMAP_CAPABILITIES.map((item) => {
          const Icon = item.icon;
          const availability = capabilityAvailability(
            item,
            workspaceCapabilities,
          );
          return (
            <article
              key={item.id}
              className="platform-roadmap-card"
              aria-label={`${item.title} roadmap`}
            >
              <header>
                <span className="platform-roadmap-icon" aria-hidden="true">
                  <Icon size={19} strokeWidth={1.8} />
                </span>
                <div>
                  <h3>{item.title}</h3>
                  <PlatformStatus value={item.status} />
                </div>
              </header>
              <p>{item.description}</p>
              <ul>
                {item.milestones.map((milestone) => (
                  <li key={milestone}>{milestone}</li>
                ))}
              </ul>
              <footer>
                <span
                  className="platform-capability-availability"
                  data-available={availability.available}
                >
                  <span aria-hidden="true" />
                  {availability.label}
                </span>
              </footer>
            </article>
          );
        })}
      </div>
    </PlatformSection>
  );
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
  const workspaceCapabilities = new Set(
    data.projectModules.workspace_capabilities,
  );
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

      <ResearchLoopRoadmap workspaceCapabilities={workspaceCapabilities} />

      {progress ? (
        <p className="platform-footnote">
          Legacy assignment progress remains available while existing projects are migrated.
        </p>
      ) : null}
    </div>
  );
}
