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
import type { Document, ProjectProgress, ProjectTask, TaskAssignment } from "@/types/api";

import {
  PlatformPageHeader,
  PlatformRouteLink,
  PlatformSection,
  PlatformStats,
  PlatformStatus,
} from "./components";
import type { PlatformProjectData, TaskKind } from "./types";

interface OverviewScreenProps {
  data: PlatformProjectData;
  documents: Document[];
  assignments: TaskAssignment[];
  progress: ProjectProgress | null;
  onOpenData: () => void;
  onOpenTraining?: () => void;
  onOpenModels?: () => void;
  onImport?: () => void;
  onOpenInference?: () => void;
  onCreateTraining?: () => void;
  currentUserId?: number | null;
  onContinueAnnotation?: (roundId: number) => void;
  onSetupAnnotation?: () => void;
  onSetupTask?: (taskVersionId: number) => void;
  onTrainModel?: (trainingDatasetVersionId: number) => void;
  projectTasks?: ProjectTask[];
  onOpenDocumentTask?: (task: ProjectTask) => void;
}

function WorkflowCard({ title, readiness, description, action }: {
  title: string; readiness: string; description: string;
  action?: { href: string; label: string; onNavigate: () => void };
}): React.ReactElement {
  return <article className="platform-roadmap-card" aria-label={`${title} readiness`}>
    <header><div><h3>{title}</h3></div></header>
    <p><strong>{readiness}</strong></p><p>{description}</p>
    {action ? <footer><PlatformRouteLink href={action.href} onNavigate={action.onNavigate}>{action.label} →</PlatformRouteLink></footer> : null}
  </article>;
}

function taskKindLabel(kind: TaskKind, preset?: unknown): string {
  if (kind === "token_labeling") return "Token labeling (tokenized data)";
  if (kind === "span_extraction" && preset === "document_entities") return "Named entities (NER)";
  return kind.replace(/_/g, " ").replace(/^./, (letter) => letter.toUpperCase());
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
      "Versioned batch predictions, exports, and selected prediction review are available for saved TF-IDF models.",
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
  onImport,
  onOpenInference,
  onCreateTraining,
  currentUserId = null,
  onContinueAnnotation,
  onSetupAnnotation,
  onSetupTask,
  onTrainModel,
  projectTasks = [],
  onOpenDocumentTask,
}: OverviewScreenProps): React.ReactElement {
  const latestModels = data.modelVersions.length;
  const labeled = data.labelSets.reduce((total, item) => total + item.label_count, 0);
  const effectiveModules = new Set(data.projectModules.effective);
  const workspaceCapabilities = new Set(
    data.projectModules.workspace_capabilities,
  );
  const latestSources = new Map<number, (typeof data.datasetVersions)[number]>();
  for (const version of data.datasetVersions) {
    if (version.provenance?.ingestion === "training_preparation_v1") continue;
    if ((latestSources.get(version.dataset_id)?.version_number ?? 0) < version.version_number) latestSources.set(version.dataset_id, version);
  }
  const sourceCount = [...latestSources.values()].filter((version) => version.item_count > 0).length;
  const projectId = data.projectModules.project_id;
  const openRounds = onContinueAnnotation && currentUserId !== null ? [...data.rounds]
    .filter((round) => round.status === "open" && (round.open_to_all_annotators || round.annotator_user_ids.includes(currentUserId)))
    .sort((left, right) => right.sequence - left.sequence || right.id - left.id) : [];
  const enabledDocumentTasks = projectTasks.filter((task) => task.enabled);
  const latestTasks = new Map<number, (typeof data.taskVersions)[number]>();
  for (const version of data.taskVersions) {
    if ((latestTasks.get(version.task_definition_id)?.version_number ?? 0) < version.version_number) latestTasks.set(version.task_definition_id, version);
  }
  const tasksNeedingSetup = onSetupTask ? data.taskDefinitions.filter((task) => latestTasks.has(task.id) && !openRounds.some((round) => data.taskVersions.some((version) => version.id === round.task_version_id && version.task_definition_id === task.id))) : [];
  const annotationChoiceCount = openRounds.length + tasksNeedingSetup.length + (onOpenDocumentTask && documents.length ? enabledDocumentTasks.length : 0);
  const openRound = annotationChoiceCount === 1 ? openRounds[0] : undefined;
  const onlySetupTask = annotationChoiceCount === 1 ? tasksNeedingSetup[0] : undefined;
  const onlyDocumentTask = annotationChoiceCount === 1 && !openRound && !onlySetupTask ? enabledDocumentTasks[0] : undefined;
  const roundTaskName = (taskVersionId: number): string | undefined => {
    const version = data.taskVersions.find((item) => item.id === taskVersionId);
    return data.taskDefinitions.find((task) => task.id === version?.task_definition_id)?.name;
  };
  const taskCount = data.taskDefinitions.length + projectTasks.length;
  const preparedVersion = [...data.trainingDatasets].sort((left, right) => right.id - left.id)[0];
  const compatibleModelCount = data.modelVersions.filter((model) => model.recipe_key === "tfidf_logistic_regression" && model.framework === "scikit-learn" && model.checkpoint_package_id &&
    data.taskVersions.some((task) => task.id === model.task_version_id && task.task_kind === "classification")).length;
  const annotationAction = annotationChoiceCount > 1 ? {
    href: "#project-annotation-tasks", label: "Choose annotation task", onNavigate: () => {
      const choices = window.document.getElementById("project-annotation-tasks");
      choices?.scrollIntoView?.({ block: "start" });
      choices?.focus({ preventScroll: true });
    },
  } : openRound && onContinueAnnotation ? {
    href: `/my-work/rounds/${openRound.id}`, label: roundTaskName(openRound.task_version_id) ? `Continue ${roundTaskName(openRound.task_version_id)}` : "Continue annotation", onNavigate: () => onContinueAnnotation(openRound.id),
  } : onlyDocumentTask && onOpenDocumentTask ? {
    href: `/my-work?project=${projectId}&view=annotate&task=${onlyDocumentTask.id}`, label: `Open ${onlyDocumentTask.display_name}`, onNavigate: () => onOpenDocumentTask(onlyDocumentTask),
  } : onlySetupTask && onSetupTask ? {
    href: `/projects/${projectId}/data?tab=source&taskVersionId=${latestTasks.get(onlySetupTask.id)!.id}`, label: `Set up ${onlySetupTask.name}`, onNavigate: () => onSetupTask(latestTasks.get(onlySetupTask.id)!.id),
  } : sourceCount && onSetupAnnotation ? {
    href: `/projects/${projectId}/data?tab=source`, label: "Set up annotation", onNavigate: onSetupAnnotation,
  } : onSetupAnnotation && onImport ? {
    href: `/projects/${projectId}/data?tab=source&flow=import`, label: "Import PMIDs", onNavigate: onImport,
  } : undefined;
  const trainingAction = preparedVersion && onTrainModel ? {
    href: `/training/new?projectId=${projectId}&trainingDatasetVersionId=${preparedVersion.id}`, label: "Train model", onNavigate: () => onTrainModel(preparedVersion.id),
  } : onCreateTraining ? {
    href: `/training/data?projectId=${projectId}&flow=prepare`, label: "Create training dataset", onNavigate: onCreateTraining,
  } : onOpenTraining ? { href: `/training?projectId=${projectId}`, label: "Open training", onNavigate: onOpenTraining } : undefined;
  const overviewStats = [
    effectiveModules.has("annotate") ? {
      label: "Annotation tasks", value: taskCount,
      detail: projectTasks.length ? `${data.taskDefinitions.length} round tasks · ${projectTasks.length} document tasks` : `${data.taskVersions.length} saved ${data.taskVersions.length === 1 ? "task version" : "task versions"}`,
    } : null,
    effectiveModules.has("data")
      ? {
          label: "Source records",
          value:
            [...latestSources.values()].reduce((total, item) => total + item.item_count, 0) ||
            documents.length,
          detail: `${latestSources.size} source datasets`,
        }
      : null,
    effectiveModules.has("train") ? {
      label: "Training datasets", value: new Set(data.trainingDatasets.map((version) => version.training_dataset_id ?? `legacy:${version.id}`)).size,
      detail: `${data.trainingDatasets.length} saved versions`,
    } : null,
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

      <PlatformSection title="Choose your next step" description="Source datasets can be used for annotation and predictions. Training datasets combine source records with labels and evaluation splits.">
        <div className="platform-roadmap-grid">
          {effectiveModules.has("annotate") ? <WorkflowCard title="Annotate"
            readiness={annotationChoiceCount > 1 ? `${annotationChoiceCount} annotation choices available` : openRound ? `${openRound.name} is open for you` : onlyDocumentTask ? `${onlyDocumentTask.display_name} is ready` : sourceCount ? `${sourceCount} source ${sourceCount === 1 ? "dataset" : "datasets"} ready` : "No source dataset yet"}
            description={annotationChoiceCount > 1 ? "Choose a task below. Each annotation round keeps its own source, task, and decisions." : openRound ? "Continue your assigned round using its saved source and task." : onlyDocumentTask ? "Open this task in the document editor." : annotationAction ? "Choose a source and task to start labeling papers." : onContinueAnnotation ? "No annotation work is assigned to you. A project manager can set up a round." : "Annotation work is not available for your role in this workspace."}
            action={annotationAction} /> : null}
          {effectiveModules.has("train") ? <WorkflowCard title="Train"
            readiness={preparedVersion ? `${preparedVersion.name} · v${preparedVersion.version_number ?? 1} ready` : "No training dataset yet"}
            description={trainingAction ? "Use submitted annotations, external labeled data, or both to train a model." : "Your role does not have access to training."}
            action={trainingAction} /> : null}
          {effectiveModules.has("data") && effectiveModules.has("models") && workspaceCapabilities.has("inference") ? <WorkflowCard title="Run predictions"
            readiness={`${sourceCount} source ${sourceCount === 1 ? "dataset" : "datasets"} · ${compatibleModelCount} compatible TF-IDF ${compatibleModelCount === 1 ? "model" : "models"}`}
            description={!onOpenInference ? "Your role does not have access to predictions." : !sourceCount ? "Import a source dataset, then choose a saved model for predictions." : !compatibleModelCount ? "A saved TF-IDF classification model with a checkpoint is needed." : "Apply a saved model, export predictions, or review selected results."}
            action={onOpenInference ? { href: `/projects/${projectId}/inference`, label: sourceCount && compatibleModelCount ? "Run predictions" : "Set up predictions", onNavigate: onOpenInference } : undefined} /> : null}
        </div>
        {effectiveModules.has("data") ? <p><PlatformRouteLink href={`/projects/${projectId}/data?tab=source`} onNavigate={onOpenData}>View source datasets and imports</PlatformRouteLink></p> : null}
      </PlatformSection>

      {effectiveModules.has("annotate") && taskCount > 0 ? (
        <div id="project-annotation-tasks" tabIndex={-1}>
          <PlatformSection title="Annotation tasks" description={projectTasks.length ? "Choose the task you want to work on. Round tasks use a saved dataset version; document tasks open the project papers in the document editor." : "Choose a task and its annotation round. Adding a task does not change the task in an existing round."}>
            <div className="platform-table-scroll platform-table-scroll--summary" role="region" aria-label="Project annotation tasks" tabIndex={0}>
              <table className="platform-table platform-table--summary">
                <thead><tr><th scope="col">Task</th><th scope="col">Annotation work</th><th scope="col">Actions</th></tr></thead>
                <tbody>
                  {data.taskDefinitions.map((task) => {
                    const latestVersion = latestTasks.get(task.id);
                    const versions = new Set(data.taskVersions.filter((version) => version.task_definition_id === task.id).map((version) => version.id));
                    const assignedRounds = openRounds.filter((round) => versions.has(round.task_version_id));
                    return <tr key={`round-task:${task.id}`}>
                      <td data-label="Task" data-priority="identity"><strong>{task.name}</strong><span>{latestVersion ? `${taskKindLabel(latestVersion.task_kind, latestVersion.annotation_ui?.preset)} · v${latestVersion.version_number}` : "Task setup incomplete"}</span></td>
                      <td data-label="Annotation work">{assignedRounds.length ? `${assignedRounds.length} open ${assignedRounds.length === 1 ? "round" : "rounds"} for you` : data.rounds.some((round) => versions.has(round.task_version_id)) ? "No open round assigned to you" : "No annotation round yet"}</td>
                      <td data-label="Actions" data-priority="action">{assignedRounds.length ? assignedRounds.map((round) => <p key={round.id}><PlatformRouteLink href={`/my-work/rounds/${round.id}`} onNavigate={() => onContinueAnnotation?.(round.id)}>Continue {task.name} — {round.name}</PlatformRouteLink></p>) : latestVersion && onSetupTask ? <PlatformRouteLink href={`/projects/${projectId}/data?tab=source&taskVersionId=${latestVersion.id}`} onNavigate={() => onSetupTask(latestVersion.id)}>Set up {task.name}</PlatformRouteLink> : null}</td>
                    </tr>;
                  })}
                  {projectTasks.map((task) => <tr key={`document-task:${task.id}`}>
                    <td data-label="Task" data-priority="identity"><strong>{task.display_name}</strong><span>{task.annotation_type === "entity" ? "Named entity recognition" : task.annotation_type.replace(/_/g, " ")}</span></td>
                    <td data-label="Annotation work">{!task.enabled ? "Disabled" : documents.length ? `${documents.length} project papers · document editor` : "Import papers to start"}</td>
                    <td data-label="Actions" data-priority="action">{task.enabled && documents.length && onOpenDocumentTask ? <PlatformRouteLink href={`/my-work?project=${projectId}&view=annotate&task=${task.id}`} onNavigate={() => onOpenDocumentTask(task)}>Open {task.display_name}</PlatformRouteLink> : null}</td>
                  </tr>)}
                </tbody>
              </table>
            </div>
          </PlatformSection>
        </div>
      ) : null}

      {(effectiveModules.has("train") && onOpenTraining) || (effectiveModules.has("models") && onOpenModels) ? (
        <PlatformSection
          title="Model development"
          description="Training and model registry work open in dedicated workspaces while retaining this project as context."
        >
          <div className="platform-module-links">
            {effectiveModules.has("train") && onOpenTraining ? (
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
            {effectiveModules.has("models") && onOpenModels ? (
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
