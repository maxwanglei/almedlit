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
  dialogContent?: ReactNode;
}

export default function ProjectPlatform({
  pathname,
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
  dialogContent,
}: ProjectPlatformProps): React.ReactElement {
  const route = parseProjectPlatformRoute(pathname);
  const requestedTab = route?.tab ?? "overview";
  const effectiveModules = new Set(data.projectModules.effective);
  const moduleConfigReady = data.projectModules.project_id === project.id;
  const visibleTabs = PROJECT_ROUTE_REGISTRY.filter(
    (item) =>
      canAccessProjectSection(access, item.id) &&
      (item.backendModule === null || effectiveModules.has(item.backendModule)),
  );
  const canManageAnnotation = canAccessProjectSection(access, "tasks");
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
        <DataScreen
          data={data}
          legacyDocumentCount={documents.length}
          onCreate={() => onDialogChange("dataset")}
          onPrepareTraining={
            effectiveModules.has("train")
              ? (dataset: Dataset) =>
                  onNavigate(
                    `/training/new?projectId=${project.id}&datasetId=${dataset.id}`,
                  )
              : undefined
          }
        />
      );
      break;
    case "tasks":
      content = (
        <AnnotateScreen
          view="tasks"
          data={data}
          onCreateRound={() => onDialogChange("round")}
          onCreateTask={() => onDialogChange("task")}
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
          documents={documents}
          assignments={assignments}
          progress={progress}
          onOpenData={() => navigateTab("data")}
          onOpenTraining={() =>
            onNavigate(`/training?projectId=${project.id}`)
          }
          onOpenModels={() =>
            onNavigate(`/models?projectId=${project.id}`)
          }
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
          {loading ? (
            <div className="platform-loading" role="status" aria-live="polite">
              <span aria-hidden="true" />
              Loading project resources…
            </div>
          ) : content}
        </main>
      </div>

      {dialog ? dialogContent : null}
    </div>
  );
}
