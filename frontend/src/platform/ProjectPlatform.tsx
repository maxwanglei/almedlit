import { useEffect, type ReactNode } from "react";
import { Banner } from "@astryxdesign/core/Banner";

import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";
import {
  canPerform,
  effectiveRoleLabel,
  type AccessSnapshot,
} from "@/navigation/AccessContext";
import { canAccessProjectSection } from "@/navigation/moduleNavigation";
import type {
  Document,
  Project,
  ProjectProgress,
  ProjectUpdate,
  TaskAssignment,
} from "@/types/api";

import ActivityScreen from "./ActivityScreen";
import AnnotateScreen from "./AnnotateScreen";
import DataScreen from "./DataScreen";
import SourceDatasetSetup from "./SourceDatasetSetup";
import InferenceScreen from "./InferenceScreen";
import {
  parseProjectPlatformRoute,
  PROJECT_ROUTE_REGISTRY,
  projectPlatformPath,
  projectSupportsSection,
  type ProjectPlatformTab,
} from "./navigation";
import OverviewScreen from "./OverviewScreen";
import QualityScreen from "./QualityScreen";
import SettingsScreen from "./SettingsScreen";
import type { Dataset, PlatformProjectData, ProjectModule } from "./types";

export type PlatformDialogKind =
  | "dataset"
  | "task"
  | "trainingData"
  | "cycle"
  | "round"
  | "feedbackScore"
  | "guideline"
  | null;

interface ProjectPlatformProps {
  pathname: string;
  search?: string;
  project: Project;
  projects: Project[];
  documents: Document[];
  assignments: TaskAssignment[];
  progress: ProjectProgress | null;
  data: PlatformProjectData;
  loading: boolean;
  busy: boolean;
  error: string | null;
  access: AccessSnapshot;
  currentUserId: number | null;
  dialog: PlatformDialogKind;
  onDialogChange: (dialog: PlatformDialogKind) => void;
  onProjectSelect: (projectId: number, tab: ProjectPlatformTab) => void;
  onNavigate: (path: string, mode?: "push" | "replace") => void;
  onOpenRound: (roundId: number) => void;
  onUpdateProject: (payload: ProjectUpdate) => Promise<void>;
  onUpdateModules: (selected: ProjectModule[]) => Promise<void>;
  onRefresh: () => Promise<void>;
  onDocumentsImported?: () => Promise<unknown>;
  dialogContent?: ReactNode;
}

export default function ProjectPlatform({
  pathname,
  search = "",
  project,
  projects,
  documents,
  assignments,
  progress,
  data,
  loading,
  busy,
  error,
  access,
  currentUserId,
  dialog,
  onDialogChange,
  onProjectSelect,
  onNavigate,
  onOpenRound,
  onUpdateProject,
  onUpdateModules,
  onRefresh,
  onDocumentsImported,
  dialogContent,
}: ProjectPlatformProps): React.ReactElement {
  const route = parseProjectPlatformRoute(pathname);
  const params = new URLSearchParams(search);
  const sourceTab = params.get("tab") === "training" ? "training" : "source";
  const queryId = (key: string): number | null => { const value = Number(params.get(key)); return Number.isSafeInteger(value) && value > 0 ? value : null; };
  const requestedTab = route?.tab ?? "overview";
  const effectiveModules = new Set(data.projectModules.effective);
  const moduleConfigReady = data.projectModules.project_id === project.id;
  const visibleTabs = PROJECT_ROUTE_REGISTRY.filter(
    (item) =>
      canAccessProjectSection(access, item.id) &&
      (item.id !== "inference" || effectiveModules.has("data")) &&
      (item.backendModule === null || effectiveModules.has(item.backendModule)),
  );
  const canManageAnnotation = canAccessProjectSection(access, "tasks");
  const canInfer = canPerform(access, "inference:run") && effectiveModules.has("data") && effectiveModules.has("models");
  const canManageSources = canPerform(access, "projects:create");
  const openImport = (dataset?: Dataset): void => onNavigate(`/projects/${project.id}/data?tab=source&flow=import${dataset ? `&datasetId=${dataset.id}` : ""}`);
  const setupTask = (taskVersionId: number): void => {
    const collections = data.datasets.filter((dataset) => dataset.source_type === "project_corpus");
    const versions = data.datasetVersions.filter((version) => collections.some((dataset) => dataset.id === version.dataset_id));
    const latest = collections.length === 1 ? [...versions].sort((a, b) => b.version_number - a.version_number)[0] : null;
    onNavigate(latest
      ? `/projects/${project.id}/data?tab=source&flow=import&datasetId=${latest.dataset_id}&datasetVersionId=${latest.id}&taskVersionId=${taskVersionId}`
      : `/projects/${project.id}/data?tab=source&taskVersionId=${taskVersionId}`);
  };
  const canScore =
    canPerform(access, "learning:score") &&
    effectiveModules.has("learning") &&
    effectiveModules.has("models");
  const tab =
    !moduleConfigReady ||
    visibleTabs.some((item) => item.id === requestedTab)
      ? requestedTab
      : "overview";

  useEffect(() => {
    if (
      !loading &&
      moduleConfigReady &&
      route &&
      route.tab !== tab
    ) {
      onNavigate(projectPlatformPath(project.id, tab), "replace");
    }
  }, [
    loading,
    moduleConfigReady,
    onNavigate,
    project.id,
    route,
    tab,
  ]);

  function navigateTab(nextTab: ProjectPlatformTab): void {
    onNavigate(projectPlatformPath(project.id, nextTab));
  }

  let content: React.ReactElement;
  switch (tab) {
    case "data":
      content = (
        params.get("flow") === "import" ? <SourceDatasetSetup key={project.id} projectId={project.id} projectName={project.name} data={data} currentUserId={currentUserId}
          datasetId={queryId("datasetId")} datasetVersionId={queryId("datasetVersionId")} initialTaskVersionId={queryId("taskVersionId")} canAnnotate={canManageAnnotation && effectiveModules.has("annotate")} canInfer={canInfer}
          onRefresh={onRefresh} onImported={onDocumentsImported} onNavigate={onNavigate} /> : <>
          {queryId("taskVersionId") ? <p role="status">Choose a source collection below for the selected annotation task.</p> : null}
          <DataScreen
          data={data}
          legacyDocumentCount={documents.length}
          onCreate={() => onDialogChange("dataset")}
          activeTab={sourceTab}
          onTabChange={(next) => onNavigate(`/projects/${project.id}/data?tab=${next}`)}
          onImport={canManageSources ? openImport : undefined}
          onAnnotate={canManageAnnotation ? (dataset, version) => onNavigate(`/projects/${project.id}/data?tab=source&flow=import&datasetId=${dataset.id}&datasetVersionId=${version.id}${queryId("taskVersionId") ? `&taskVersionId=${queryId("taskVersionId")}` : ""}`) : undefined}
          onPredict={canInfer ? (version) => onNavigate(`/projects/${project.id}/inference?datasetVersionId=${version.id}`) : undefined}
          onCreateTraining={effectiveModules.has("train") ? () => onNavigate(`/training/data?projectId=${project.id}&flow=prepare`) : undefined}
          onTrain={(version) => onNavigate(`/training/new?projectId=${project.id}&trainingDatasetVersionId=${version.id}`)}
          onNewTrainingVersion={(version) => onNavigate(`/training/data?projectId=${project.id}&flow=prepare&trainingDatasetId=${version.training_dataset_id}&trainingDatasetVersionId=${version.id}`)}
          onPrepareTraining={
            effectiveModules.has("train")
              ? (dataset: Dataset) =>
                  onNavigate(
                    `/training/data?projectId=${project.id}&flow=prepare&datasetId=${dataset.id}`,
                  )
              : undefined
          }
        /></>
      );
      break;
    case "inference":
      content = <InferenceScreen projectId={project.id} data={data} currentUserId={currentUserId} canReview={canManageAnnotation && effectiveModules.has("annotate")} onOpenRound={onOpenRound} onRefresh={onRefresh} onImport={canManageSources ? () => openImport() : () => onDialogChange("dataset")} onOpenModels={() => onNavigate(`/models?projectId=${project.id}`)} />;
      break;
    case "tasks":
      content = (
        <AnnotateScreen
          view="tasks"
          data={data}
          onCreateRound={() => onDialogChange("round")}
          onCreateTask={() => onDialogChange("task")}
          onSetupTask={canManageAnnotation ? setupTask : undefined}
          onOpenRound={onOpenRound}
          currentUserId={currentUserId}
          canManage={canManageAnnotation}
          canScore={false}
          projectId={project.id}
          onCreateScore={() => onDialogChange("feedbackScore")}
          onRefresh={onRefresh}
        />
      );
      break;
    case "rounds":
      content = (
        <AnnotateScreen
          view="rounds"
          data={data}
          onCreateRound={() => onDialogChange("round")}
          onCreateTask={() => onDialogChange("task")}
          onOpenRound={onOpenRound}
          currentUserId={currentUserId}
          canManage={canManageAnnotation}
          canScore={canScore}
          projectId={project.id}
          onCreateScore={() => onDialogChange("feedbackScore")}
          onRefresh={onRefresh}
        />
      );
      break;
    case "quality":
      content = (
        <QualityScreen
          data={data}
          progress={progress}
          projectId={project.id}
          documents={documents}
          assignments={assignments}
          allowSoloGold={access.isWorkspaceOwner}
        />
      );
      break;
    case "activity":
      content = <ActivityScreen data={data} />;
      break;
    case "settings":
      content = (
        <SettingsScreen
          project={project}
          data={data}
          busy={busy}
          onUpdateProject={onUpdateProject}
          onUpdateModules={onUpdateModules}
        />
      );
      break;
    default:
      content = (
        <OverviewScreen
          data={data}
          projectTasks={project.tasks}
          onSetupTask={canManageAnnotation ? setupTask : undefined}
          documents={documents}
          assignments={assignments}
          progress={progress}
          currentUserId={currentUserId}
          onOpenData={() => onNavigate(`/projects/${project.id}/data?tab=source`)}
          onImport={canManageSources && effectiveModules.has("data") ? () => openImport() : undefined}
          onContinueAnnotation={canPerform(access, "annotation:work") && effectiveModules.has("annotate") ? onOpenRound : undefined}
          onSetupAnnotation={canManageAnnotation && effectiveModules.has("data") && effectiveModules.has("annotate") ? () => onNavigate(`/projects/${project.id}/data?tab=source`) : undefined}
          onOpenInference={canInfer ? () => navigateTab("inference") : undefined}
          onCreateTraining={canPerform(access, "training:launch") && effectiveModules.has("train") ? () => onNavigate(`/training/data?projectId=${project.id}&flow=prepare`) : undefined}
          onTrainModel={canPerform(access, "training:launch") && effectiveModules.has("train") ? (versionId) => onNavigate(`/training/new?projectId=${project.id}&trainingDatasetVersionId=${versionId}`) : undefined}
          onOpenTraining={canPerform(access, "training:read") ? () =>
            onNavigate(`/training?projectId=${project.id}`)
          : undefined}
          onOpenModels={canPerform(access, "models:read") ? () =>
            onNavigate(`/models?projectId=${project.id}`)
          : undefined}
        />
      );
  }

  return (
    <div className="platform-shell">
      <div className="platform-context-bar">
        <label>
          <span>Project</span>
          <select
            value={project.id}
            onChange={(event) => {
              const nextProjectId = Number(event.target.value);
              const nextProject = projects.find(
                (item) => item.id === nextProjectId,
              );
              const nextTab =
                nextProject &&
                canAccessProjectSection(access, tab) &&
                projectSupportsSection(nextProject, tab)
                  ? tab
                  : "overview";
              onProjectSelect(nextProjectId, nextTab);
            }}
          >
            {projects.map((item) => (
              <option key={item.id} value={item.id}>{item.name}</option>
            ))}
          </select>
        </label>
        <div>
          <strong>{project.name}</strong>
          <span>{project.description ?? "No project description"}</span>
        </div>
        <span className="platform-role">
          {effectiveModules.has("train") && !effectiveModules.has("annotate")
            ? "Training-only"
            : effectiveRoleLabel(access)}
        </span>
      </div>

      <div className="platform-mobile-tab">
        <label>
          <span>Project section</span>
          <select value={tab} onChange={(event) => navigateTab(event.target.value as ProjectPlatformTab)}>
            {visibleTabs.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="platform-workspace">
        <nav className="platform-sidebar" aria-label="Project">
          {visibleTabs.map((item) => (
            <a
              key={item.id}
              href={projectPlatformPath(project.id, item.id)}
              aria-current={tab === item.id ? "page" : undefined}
              onClick={(event) => {
                if (!shouldHandleSpaClick(event)) {
                  return;
                }
                event.preventDefault();
                navigateTab(item.id);
              }}
            >
              {item.label}
            </a>
          ))}
        </nav>
        <main id="main-content" className="platform-main" tabIndex={-1}>
          {error ? (
            <Banner
              status="error"
              title="Project data could not be loaded"
              description={error}
              container="section"
            />
          ) : null}
          {loading && !moduleConfigReady ? (
            <div className="platform-loading" role="status" aria-live="polite">
              <span aria-hidden="true" />
              Loading project resources…
            </div>
          ) : <div key={project.id} aria-busy={loading}>{content}</div>}
        </main>
      </div>

      {dialog ? dialogContent : null}
    </div>
  );
}
