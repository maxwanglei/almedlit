import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import {
  BrowserRouter,
  Outlet,
  Route,
  Routes,
  useLocation,
  useMatch,
  useNavigate,
  useOutletContext,
  useParams,
} from "react-router-dom";

import {
  getAnnotationWorkbench,
  getMe,
  getWorkspaceCapabilities,
  updateProject,
} from "@/api/client";
import { clearToken, getToken, subscribeTokenChanges } from "@/auth/session";
import AppShell from "@/components/AppShell";
import WorkspaceSwitcher from "@/components/WorkspaceSwitcher";
import { useEvidenceBlockStore } from "@/features/evidence-block/evidenceBlockStore";
import {
  preferredMembership,
  readPreferredWorkspaceId,
  storePreferredWorkspaceId,
} from "@/lib/workspaceSelection";
import {
  AccessProvider,
  canPerform,
  createAccessSnapshot,
} from "@/navigation/AccessContext";
import {
  authorizedRedirectPath,
  availableModules,
  canAccessProjectSection,
  compatibilityRedirectNotice,
  currentModuleId,
  defaultModulePath,
  legacyModelsProjectId,
  normalizePathname,
  preserveLocation,
  resolveWorkspaceRoute,
  workspaceSettingsDestination,
  type ModuleNavigationContext,
} from "@/navigation/moduleNavigation";
import AcceptInvitePage from "@/pages/AcceptInvitePage";
import LoginPage from "@/pages/LoginPage";
import OnboardingPresetPicker from "@/pages/OnboardingPresetPicker";
import AddWorkspaceDialog from "@/components/AddWorkspaceDialog";
import {
  getRoundWorkContext,
  listWorkspaceRoundWorkContexts,
  type PlatformLoadScope,
} from "@/platform/api";
import type { PlatformDialogKind } from "@/platform/ProjectPlatform";
import type { RoundWorkContext } from "@/platform/types";
import {
  parseProjectPlatformRoute,
  projectPlatformPath,
  projectRouteDefinition,
  projectSupportsSection,
  withSearchAndHash,
} from "@/platform/navigation";
import { usePlatformProject } from "@/platform/usePlatformProject";
import { useProjectWorkspaceStore } from "@/store/projectWorkspaceStore";
import type { MeMembership } from "@/api/client";
import type {
  AnnotationWorkbench,
  ProjectUpdate,
} from "@/types/api";

const AnnotatorWorkspace = lazy(() => import("@/pages/AnnotatorWorkspace"));
const AccountActionPage = lazy(() => import("@/pages/AccountActionPage"));
const ModelsWorkspace = lazy(() => import("@/platform/ModelsWorkspace"));
const PlatformDialog = lazy(() => import("@/platform/PlatformDialog"));
const ProjectPlatform = lazy(() => import("@/platform/ProjectPlatform"));
const ProjectsWorkspace = lazy(() => import("@/platform/ProjectsWorkspace"));
const RoundWorkbench = lazy(() => import("@/platform/RoundWorkbench"));
const SystemAdministration = lazy(
  () => import("@/platform/SystemAdministration"),
);
const TrainingWorkspace = lazy(() => import("@/platform/TrainingWorkspace"));
const WorkspaceSettings = lazy(() => import("@/platform/WorkspaceSettings"));

type TrainingRouteView =
  | "runs"
  | "new"
  | "data"
  | "runtimes"
  | "run-detail";
type ModelsRouteView = "registry" | "detail" | "version";

interface ApplicationRouteContext {
  renderMyWork: () => React.ReactElement;
  renderRound: (roundId: number | null) => React.ReactElement;
  renderProjects: () => React.ReactElement;
  renderProject: (projectId: number | null) => React.ReactElement;
  renderTraining: (
    view: TrainingRouteView,
    runId: number | null,
  ) => React.ReactElement;
  renderModels: (
    view: ModelsRouteView,
    modelId: number | null,
    versionId: number | null,
  ) => React.ReactElement;
  renderWorkspaceSettings: () => React.ReactElement;
  renderSystemAdministration: () => React.ReactElement;
  renderNoAccess: () => React.ReactElement;
  renderNotFound: () => React.ReactElement;
}

interface AuthenticatedLayoutProps {
  access: ReturnType<typeof createAccessSnapshot>;
  modules: ReturnType<typeof availableModules>;
  currentModuleId: ReturnType<typeof currentModuleId>;
  workspaceSettings: ReturnType<typeof workspaceSettingsDestination>;
  workspaceSettingsCurrent: boolean;
  homePath: string;
  workspaceSwitcher: React.ReactNode;
  mobileWorkspaceSwitcher: React.ReactNode;
  announcement: string;
  onNavigate: (path: string) => void;
  onLogout: () => void;
  redirecting: boolean;
  routeContext: ApplicationRouteContext;
}

function AuthenticatedLayout({
  access,
  modules,
  currentModuleId: activeModuleId,
  workspaceSettings,
  workspaceSettingsCurrent,
  homePath,
  workspaceSwitcher,
  mobileWorkspaceSwitcher,
  announcement,
  onNavigate,
  onLogout,
  redirecting,
  routeContext,
}: AuthenticatedLayoutProps): React.ReactElement {
  return (
    <AccessProvider value={access}>
      <AppShell
        modules={modules}
        currentModuleId={activeModuleId}
        workspaceSettings={workspaceSettings}
        workspaceSettingsCurrent={workspaceSettingsCurrent}
        homePath={homePath}
        workspaceSwitcher={workspaceSwitcher}
        mobileWorkspaceSwitcher={mobileWorkspaceSwitcher}
        announcement={announcement}
        onNavigate={onNavigate}
        onLogout={onLogout}
      >
        <Suspense
          fallback={(
            <main id="main-content" tabIndex={-1}>
              <div className="status" role="status" aria-live="polite">
                Loading workspace…
              </div>
            </main>
          )}
        >
          {redirecting ? (
            <main id="main-content" tabIndex={-1}>
              <div className="status" role="status" aria-live="polite">
                Opening workspace…
              </div>
            </main>
          ) : (
            <Outlet context={routeContext} />
          )}
        </Suspense>
      </AppShell>
    </AccessProvider>
  );
}

function useApplicationRouteContext(): ApplicationRouteContext {
  return useOutletContext<ApplicationRouteContext>();
}

function positiveInteger(value: string | undefined): number | null {
  if (value === undefined) {
    return null;
  }
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : null;
}

function MyWorkRoute(): React.ReactElement {
  return useApplicationRouteContext().renderMyWork();
}

function RoundRoute(): React.ReactElement {
  const { roundId } = useParams();
  return useApplicationRouteContext().renderRound(positiveInteger(roundId));
}

function ProjectsRoute(): React.ReactElement {
  return useApplicationRouteContext().renderProjects();
}

function ProjectRoute(): React.ReactElement {
  const { projectId } = useParams();
  return useApplicationRouteContext().renderProject(
    positiveInteger(projectId),
  );
}

function TrainingRoute({
  view,
}: {
  view: TrainingRouteView;
}): React.ReactElement {
  const { runId } = useParams();
  return useApplicationRouteContext().renderTraining(
    view,
    view === "run-detail" ? positiveInteger(runId) : null,
  );
}

function ModelsRoute({
  view,
}: {
  view: ModelsRouteView;
}): React.ReactElement {
  const { modelId, versionId } = useParams();
  return useApplicationRouteContext().renderModels(
    view,
    view === "registry" ? null : positiveInteger(modelId),
    view === "version" ? positiveInteger(versionId) : null,
  );
}

function WorkspaceSettingsRoute(): React.ReactElement {
  return useApplicationRouteContext().renderWorkspaceSettings();
}

function SystemAdministrationRoute(): React.ReactElement {
  return useApplicationRouteContext().renderSystemAdministration();
}

function NoAccessRoute(): React.ReactElement {
  return useApplicationRouteContext().renderNoAccess();
}

function NotFoundRoute(): React.ReactElement {
  return useApplicationRouteContext().renderNotFound();
}

function isEntryPath(pathname: string): boolean {
  const normalizedPathname = normalizePathname(pathname);
  return normalizedPathname === "/" || normalizedPathname === "/my-work";
}

/** Extract the token from `/invites/:token`, or null when not that route. */
function inviteTokenFromPath(pathname: string): string | null {
  const segments = normalizePathname(pathname).split("/").filter(Boolean);
  if (segments.length !== 2 || segments[0] !== "invites") {
    return null;
  }
  try {
    return decodeURIComponent(segments[1]) || null;
  } catch {
    return segments[1] || null;
  }
}

/** Extract the token from `/account-actions/:token`, or null otherwise. */
function accountActionTokenFromPath(pathname: string): string | null {
  const segments = normalizePathname(pathname).split("/").filter(Boolean);
  if (segments.length !== 2 || segments[0] !== "account-actions") {
    return null;
  }
  try {
    return decodeURIComponent(segments[1]) || null;
  } catch {
    return segments[1] || null;
  }
}

function isIndividualWorkspace(membership: MeMembership | null | undefined): boolean {
  return membership?.workspace_kind === "individual";
}

function describeRouteState(search: string, hash: string): string {
  const details = Array.from(new URLSearchParams(search).entries())
    .slice(0, 3)
    .map(([key, value]) => {
      const label = key
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .replace(/[-_]+/g, " ")
        .toLowerCase();
      const conciseValue =
        value.length > 60 ? `${value.slice(0, 57)}...` : value;
      return `${label} ${conciseValue || "empty"}`;
    });
  if (hash.length > 1) {
    let section = hash.slice(1);
    try {
      section = decodeURIComponent(section);
    } catch {
      // Keep the literal fragment when it is not valid URI text.
    }
    details.push(`section ${section}`);
  }
  if (!details.length) return "";
  const description = details
    .map((detail) => `${detail[0].toUpperCase()}${detail.slice(1)}`)
    .join(". ");
  return ` ${description}.`;
}

export default function App(): React.ReactElement {
  return (
    <BrowserRouter>
      <Application />
    </BrowserRouter>
  );
}

function Application(): React.ReactElement {
  const location = useLocation();
  const routerNavigate = useNavigate();
  const [authed, setAuthed] = useState<boolean>(() => getToken() !== null);
  const [sessionReady, setSessionReady] = useState<boolean>(() => getToken() === null);
  const [justRegistered, setJustRegistered] = useState(false);
  const [onboarded, setOnboarded] = useState(false);
  const [activeWorkspaceId, setActiveWorkspaceId] = useState<number | null>(null);
  const [activeMembership, setActiveMembership] = useState<MeMembership | null>(null);
  const [memberships, setMemberships] = useState<MeMembership[]>([]);
  const [currentUserId, setCurrentUserId] = useState<number | null>(null);
  const [currentUsername, setCurrentUsername] = useState<string | null>(null);
  const [caps, setCaps] = useState<string[]>([]);
  const [blockedCaps, setBlockedCaps] = useState<Record<string, string>>({});
  const [projectsReady, setProjectsReady] = useState(false);
  const [isSuperuser, setIsSuperuser] = useState(false);
  const [workspaceDialogOpen, setWorkspaceDialogOpen] = useState(false);
  const [sessionGeneration, setSessionGeneration] = useState(0);
  const sessionGenerationRef = useRef(0);
  const pendingWorkspaceSwitchRef = useRef(false);
  const observedTokenRef = useRef<string | null>(getToken());
  const pathname = normalizePathname(location.pathname);
  const inviteToken = inviteTokenFromPath(location.pathname);
  const accountActionToken = accountActionTokenFromPath(location.pathname);
  const workspaceRoute = resolveWorkspaceRoute(pathname);
  const routeSearch = new URLSearchParams(location.search);
  const queryProjectValue = routeSearch.get("projectId");
  const queryProjectId =
    queryProjectValue !== null &&
    Number.isSafeInteger(Number(queryProjectValue)) &&
    Number(queryProjectValue) > 0
      ? Number(queryProjectValue)
      : null;
  const queryDatasetValue = routeSearch.get("datasetId");
  const queryDatasetId =
    queryDatasetValue !== null &&
    Number.isSafeInteger(Number(queryDatasetValue)) &&
    Number(queryDatasetValue) > 0
      ? Number(queryDatasetValue)
      : null;
  const roundRouteMatch = useMatch("/my-work/rounds/:roundId");
  const roundRouteId = positiveInteger(roundRouteMatch?.params.roundId);
  const parsedProjectRoute = useMemo(
    () => parseProjectPlatformRoute(pathname),
    [pathname],
  );
  const [workbench, setWorkbench] = useState<AnnotationWorkbench | null>(null);
  const [resolvedRoundContext, setResolvedRoundContext] =
    useState<RoundWorkContext | null>(null);
  const [roundContexts, setRoundContexts] = useState<RoundWorkContext[]>([]);
  const [roundContextsLoading, setRoundContextsLoading] = useState(false);
  const [roundContextsError, setRoundContextsError] = useState<string | null>(
    null,
  );
  const roundContextRequestIdRef = useRef(0);
  const roundQueueRequestIdRef = useRef(0);
  const [roundResolutionLoading, setRoundResolutionLoading] = useState(false);
  const [roundResolutionError, setRoundResolutionError] = useState<
    string | null
  >(null);
  const [routeAnnouncement, setRouteAnnouncement] = useState("");
  const pendingRouteNoticeRef = useRef<string | null>(null);
  const [platformDialog, setPlatformDialog] = useState<PlatformDialogKind>(null);
  const {
    projects,
    documents,
    assignments,
    projectProgress,
    selectedProjectId,
    selectedDocumentId,
    loading,
    busy,
    error,
    setSelectedProjectId,
    setSelectedDocumentId,
    setBusy,
    setError,
    clearProjectData,
    replaceProject,
    loadProjects,
    loadProjectData,
  } = useProjectWorkspaceStore();

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );
  const activeWorkspaceIsIndividual = isIndividualWorkspace(activeMembership);
  const access = useMemo(
    () =>
      createAccessSnapshot({
        workspaceId: activeWorkspaceId,
        workspaceKind: activeMembership?.workspace_kind,
        membershipRole: activeMembership?.role,
        effectiveCapabilities: caps,
        blockedCapabilities: blockedCaps,
        isSuperuser,
        ready: sessionReady,
      }),
    [
      activeMembership?.role,
      activeMembership?.workspace_kind,
      activeWorkspaceId,
      blockedCaps,
      caps,
      isSuperuser,
      sessionReady,
    ],
  );
  const navigationContext = useMemo<ModuleNavigationContext>(
    () => ({
      ...access,
      selectedProjectId,
    }),
    [access, selectedProjectId],
  );
  const moduleDestinations = useMemo(
    () => availableModules(navigationContext),
    [navigationContext],
  );
  const activeModuleId = currentModuleId(
    pathname,
    navigationContext,
  );
  const workspaceSettings = workspaceSettingsDestination(navigationContext);
  const policyProject =
    parsedProjectRoute === null
      ? null
      : projects.find(
          (project) => project.id === parsedProjectRoute.projectId,
        ) ?? null;
  const canOpenProjectRoute =
    parsedProjectRoute === null ||
    (
      canAccessProjectSection(navigationContext, parsedProjectRoute.tab) &&
      (
        !projectsReady ||
        policyProject === null ||
        projectSupportsSection(policyProject, parsedProjectRoute.tab)
      )
    );
  const routeRedirect = useMemo(
    () => authorizedRedirectPath(pathname, navigationContext),
    [navigationContext, pathname],
  );
  const platformLoadScope = useMemo<PlatformLoadScope>(() => {
    return parsedProjectRoute && canOpenProjectRoute
      ? parsedProjectRoute.tab
      : "overview";
  }, [canOpenProjectRoute, parsedProjectRoute]);
  const activeProjectContextId =
    parsedProjectRoute?.projectId ?? selectedProjectId;
  const platform = usePlatformProject(
    parsedProjectRoute?.projectId ?? null,
    activeWorkspaceId,
    authed &&
      sessionReady &&
      routeRedirect === null &&
      (
        (
          parsedProjectRoute !== null &&
          projectsReady &&
          policyProject !== null &&
          canOpenProjectRoute
        )
      ),
    platformLoadScope,
  );

  const navigatePath = useCallback(
    (nextPath: string, mode: "push" | "replace" = "push"): void => {
      routerNavigate(nextPath, { replace: mode === "replace" });
    },
    [routerNavigate],
  );
  const navigatePathRef = useRef(navigatePath);
  navigatePathRef.current = navigatePath;

  const resetForTokenChange = useCallback((token: string | null): void => {
    observedTokenRef.current = token;
    sessionGenerationRef.current += 1;
    setSessionGeneration(sessionGenerationRef.current);
    useProjectWorkspaceStore.getState().reset();
    useEvidenceBlockStore.getState().reset();
    setAuthed(token !== null);
    setSessionReady(token === null);
    setJustRegistered(false);
    setOnboarded(false);
    setActiveWorkspaceId(null);
    setActiveMembership(null);
    setMemberships([]);
    setCurrentUserId(null);
    setCurrentUsername(null);
    setCaps([]);
    setBlockedCaps({});
    setProjectsReady(false);
    setIsSuperuser(false);
    setWorkspaceDialogOpen(false);
    setWorkbench(null);
    setResolvedRoundContext(null);
    setRoundContexts([]);
    setRoundContextsLoading(false);
    setRoundContextsError(null);
    roundContextRequestIdRef.current += 1;
    roundQueueRequestIdRef.current += 1;
    setRoundResolutionLoading(false);
    setRoundResolutionError(null);
    setPlatformDialog(null);
    pendingWorkspaceSwitchRef.current = false;
    const publicTokenPath =
      inviteTokenFromPath(window.location.pathname) !== null ||
      accountActionTokenFromPath(window.location.pathname) !== null;
    if (token !== null || !publicTokenPath) {
      navigatePathRef.current("/", "replace");
    }
  }, []);

  async function refreshWorkspaceConfiguration(
    workspaceId: number,
  ): Promise<{
    effectiveCapabilities: string[];
    blockedCapabilities: Record<string, string>;
  }> {
    const generation = sessionGenerationRef.current;
    const nextCaps = await getWorkspaceCapabilities(workspaceId);
    if (
      generation !== sessionGenerationRef.current ||
      workspaceId !== activeWorkspaceId
    ) {
      throw new Error(
        "The active workspace changed before setup could be confirmed.",
      );
    }
    setCaps(nextCaps.effective);
    setBlockedCaps(nextCaps.blocked);
    return {
      effectiveCapabilities: nextCaps.effective,
      blockedCapabilities: nextCaps.blocked,
    };
  }

  function handleLogout(): void {
    const observedToken = observedTokenRef.current;
    clearToken();
    window.localStorage.removeItem("al-medlit.currentAnnotatorId");
    if (observedTokenRef.current === observedToken) {
      resetForTokenChange(null);
    }
  }

  function handleWorkspaceChange(workspaceId: number): void {
    const nextMembership = memberships.find(
      (membership) => membership.workspace_id === workspaceId,
    );
    if (
      !nextMembership ||
      currentUserId === null ||
      nextMembership.workspace_id === activeWorkspaceId
    ) {
      return;
    }

    storePreferredWorkspaceId(currentUserId, nextMembership.workspace_id);
    sessionGenerationRef.current += 1;
    setSessionGeneration(sessionGenerationRef.current);
    useProjectWorkspaceStore.getState().reset();
    useEvidenceBlockStore.getState().reset();
    setSessionReady(false);
    setActiveWorkspaceId(null);
    setActiveMembership(null);
    setCaps([]);
    setBlockedCaps({});
    setProjectsReady(false);
    setWorkbench(null);
    setResolvedRoundContext(null);
    setRoundContexts([]);
    setRoundContextsLoading(false);
    setRoundContextsError(null);
    roundContextRequestIdRef.current += 1;
    roundQueueRequestIdRef.current += 1;
    setRoundResolutionLoading(false);
    setRoundResolutionError(null);
    setPlatformDialog(null);
    pendingWorkspaceSwitchRef.current = true;
  }

  function handleWorkspaceCreated(workspaceId: number): void {
    if (currentUserId !== null) {
      storePreferredWorkspaceId(currentUserId, workspaceId);
    }
    pendingWorkspaceSwitchRef.current = true;
    sessionGenerationRef.current += 1;
    setSessionGeneration(sessionGenerationRef.current);
    useProjectWorkspaceStore.getState().reset();
    useEvidenceBlockStore.getState().reset();
    setSessionReady(false);
    setActiveWorkspaceId(null);
    setActiveMembership(null);
    setCaps([]);
    setBlockedCaps({});
    setProjectsReady(false);
    setWorkbench(null);
    setWorkspaceDialogOpen(false);
  }

  async function refreshMemberships(): Promise<void> {
    const generation = sessionGenerationRef.current;
    const me = await getMe();
    if (generation !== sessionGenerationRef.current) {
      throw new Error("The active account changed before memberships refreshed.");
    }
    setMemberships(me.memberships);
    setCurrentUserId(me.user.id);
    setCurrentUsername(me.user.username);
    setIsSuperuser(me.user.is_superuser);
    if (activeWorkspaceId !== null) {
      const refreshedActiveMembership =
        me.memberships.find(
          (membership) => membership.workspace_id === activeWorkspaceId,
        ) ?? null;
      setActiveMembership(refreshedActiveMembership);
    }
  }

  async function refreshProjectData(projectId: number): Promise<number | null> {
    return loadProjectData(projectId, true, "mine");
  }

  async function refreshWorkbench(documentId: number): Promise<void> {
    const generation = sessionGenerationRef.current;
    const nextWorkbench = await getAnnotationWorkbench(documentId);
    const current = useProjectWorkspaceStore.getState();
    if (
      generation === sessionGenerationRef.current &&
      current.selectedDocumentId === documentId &&
      current.selectedProjectId === nextWorkbench.project.id &&
      nextWorkbench.document.id === documentId
    ) {
      setWorkbench(nextWorkbench);
    }
  }

  useEffect(() => {
    const unsubscribe = subscribeTokenChanges(resetForTokenChange);
    const currentToken = getToken();
    if (currentToken !== observedTokenRef.current) {
      resetForTokenChange(currentToken);
    }
    return unsubscribe;
  }, [resetForTokenChange]);

  useEffect(() => {
    if (!authed) {
      return;
    }
    let cancelled = false;
    const generation = sessionGenerationRef.current;
    setSessionReady(false);

    async function loadCurrentWorkspace(): Promise<void> {
      try {
        const me = await getMe();
        if (cancelled || generation !== sessionGenerationRef.current) {
          return;
        }
        const selectedMembership = preferredMembership(
          me.memberships,
          readPreferredWorkspaceId(me.user.id),
        );
        const workspaceId = selectedMembership?.workspace_id ?? null;
        let effectiveCapabilities: string[] = [];
        let blockedCapabilities: Record<string, string> = {};
        if (workspaceId !== null) {
          const nextCaps = await getWorkspaceCapabilities(workspaceId);
          if (cancelled || generation !== sessionGenerationRef.current) {
            return;
          }
          effectiveCapabilities = nextCaps.effective;
          blockedCapabilities = nextCaps.blocked;
        }
        if (selectedMembership !== null) {
          storePreferredWorkspaceId(me.user.id, selectedMembership.workspace_id);
        }
        setMemberships(me.memberships);
        setCurrentUserId(me.user.id);
        setCurrentUsername(me.user.username);
        setIsSuperuser(me.user.is_superuser);
        setActiveMembership(selectedMembership);
        setActiveWorkspaceId(workspaceId);
        setCaps(effectiveCapabilities);
        setBlockedCaps(blockedCapabilities);
        if (
          pendingWorkspaceSwitchRef.current ||
          isEntryPath(window.location.pathname)
        ) {
          pendingWorkspaceSwitchRef.current = false;
          navigatePathRef.current(
            defaultModulePath({
              ...createAccessSnapshot({
                workspaceId,
                workspaceKind: selectedMembership?.workspace_kind,
                membershipRole: selectedMembership?.role,
                effectiveCapabilities,
                blockedCapabilities,
                isSuperuser: me.user.is_superuser,
              }),
              selectedProjectId: null,
            }),
            "replace",
          );
        }
      } catch {
        if (!cancelled && generation === sessionGenerationRef.current) {
          clearToken();
        }
      } finally {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setSessionReady(true);
        }
      }
    }

    void loadCurrentWorkspace();
    return () => {
      cancelled = true;
    };
  }, [authed, sessionGeneration]);

  useEffect(() => {
    if (!authed || !sessionReady) {
      return;
    }
    // Invite acceptance renders ahead of the workspace shell and is not a
    // workspace route, so the redirect below must not pull a signed-in user
    // off the invite before they can accept it.
    if (inviteToken !== null) {
      return;
    }

    const legacyProjectId = legacyModelsProjectId(pathname);
    if (
      legacyProjectId !== null &&
      selectedProjectId !== legacyProjectId
    ) {
      setSelectedProjectId(legacyProjectId);
    }

    if (routeRedirect !== null && routeRedirect !== pathname) {
      pendingRouteNoticeRef.current =
        compatibilityRedirectNotice(pathname) ??
        (pathname === "/"
          ? null
          : "That destination is unavailable in this workspace. Your default workspace loaded.");
      navigatePath(
        preserveLocation(routeRedirect, location.search, location.hash),
        "replace",
      );
    }
  }, [
    activeMembership,
    authed,
    inviteToken,
    location.hash,
    location.search,
    navigatePath,
    routeRedirect,
    selectedProjectId,
    sessionReady,
    setSelectedProjectId,
    pathname,
  ]);

  useEffect(() => {
    if (!authed || !sessionReady || routeRedirect !== null) {
      return;
    }
    const route = parseProjectPlatformRoute(pathname);
    if (route && !canOpenProjectRoute) {
      pendingRouteNoticeRef.current =
        "That project section is unavailable for your role. Project overview loaded.";
      navigatePath(
        preserveLocation(
          projectPlatformPath(route.projectId, "overview"),
          location.search,
          location.hash,
        ),
        "replace",
      );
      return;
    }
    if (route?.redirect) {
      pendingRouteNoticeRef.current =
        route.redirect.notice ??
        (route.redirect.kind === "legacy"
          ? "That project address moved. The corresponding project section loaded."
          : "The canonical project address loaded.");
      navigatePath(
        withSearchAndHash(route.canonicalPath, location.search, location.hash),
        "replace",
      );
      return;
    }

    const canonicalPathname = normalizePathname(location.pathname);
    if (
      canonicalPathname !== location.pathname &&
      canonicalPathname !== "/"
    ) {
      navigatePath(
        withSearchAndHash(canonicalPathname, location.search, location.hash),
        "replace",
      );
    }
  }, [
    authed,
    canOpenProjectRoute,
    location.hash,
    location.pathname,
    location.search,
    navigatePath,
    pathname,
    routeRedirect,
    sessionReady,
  ]);

  useEffect(() => {
    if (!authed || !sessionReady || routeRedirect !== null) {
      return;
    }
    const projectRoute = parseProjectPlatformRoute(pathname);
    if (
      projectRoute?.redirect ||
      (projectRoute !== null && !canOpenProjectRoute)
    ) {
      return;
    }
    let title = "Workspace";
    if (roundRouteId !== null) {
      title = "Round work";
    } else if (workspaceRoute === "/my-work") {
      title = "My Work";
    } else if (projectRoute) {
      title = projectRouteDefinition(projectRoute.tab).title;
    } else if (workspaceRoute === "/projects") {
      title = "Projects";
    } else if (workspaceRoute === "/training") {
      title = pathname === "/training/new"
        ? "New training"
        : pathname.startsWith("/training/runs/")
          ? "Training run"
          : pathname === "/training/data"
            ? "Training data"
            : pathname === "/training/runtimes"
              ? "Training runtimes"
              : "Training";
    } else if (workspaceRoute === "/models") {
      title = pathname === "/models" ? "Models" : "Model details";
    } else if (workspaceRoute === "/workspace-settings") {
      title = "Workspace settings";
    } else if (workspaceRoute === "/admin") {
      title = "System administration";
    } else if (workspaceRoute === "/no-access") {
      title = "No modules available";
    } else if (workspaceRoute === "/not-found") {
      title = "Page not found";
    }

    document.title = `${title} | AL-MedLit`;
    const notice = pendingRouteNoticeRef.current;
    pendingRouteNoticeRef.current = null;
    setRouteAnnouncement(
      notice ??
        `${title} loaded.${describeRouteState(
          location.search,
          location.hash,
        )}`,
    );
  }, [
    authed,
    canOpenProjectRoute,
    location.hash,
    location.search,
    pathname,
    roundRouteId,
    routeRedirect,
    sessionReady,
    workspaceRoute,
  ]);

  useEffect(() => {
    if (!authed || !sessionReady || routeRedirect !== null) {
      return;
    }
    const focusProjectRoute = parseProjectPlatformRoute(pathname);
    if (
      focusProjectRoute?.redirect ||
      (focusProjectRoute !== null && !canOpenProjectRoute)
    ) {
      return;
    }
    const routeDataLoading =
      (workspaceRoute === "/projects" && (loading || platform.loading)) ||
      (workspaceRoute === "/my-work" && (loading || platform.loading));
    if (routeDataLoading) {
      return;
    }

    let interactionCancelled = false;
    let observer: MutationObserver | null = null;
    let fallbackTimer = 0;
    let stopTimer = 0;

    const stopWatching = (): void => {
      observer?.disconnect();
      observer = null;
      window.clearTimeout(fallbackTimer);
      window.clearTimeout(stopTimer);
      document.removeEventListener("pointerdown", cancelForInteraction, true);
      document.removeEventListener("keydown", cancelForInteraction, true);
    };
    const cancelForInteraction = (): void => {
      interactionCancelled = true;
      stopWatching();
    };
    const focusElement = (target: HTMLElement): void => {
      const hadTabIndex = target.hasAttribute("tabindex");
      if (!hadTabIndex) target.setAttribute("tabindex", "-1");
      target.focus();
      if (!hadTabIndex) {
        target.addEventListener(
          "blur",
          () => target.removeAttribute("tabindex"),
          { once: true },
        );
      }
    };
    const focusHeading = (): boolean => {
      if (interactionCancelled) return false;
      const heading = document.querySelector<HTMLElement>("#main-content h1");
      if (!heading) return false;
      focusElement(heading);
      stopWatching();
      return true;
    };

    const firstFrame = window.requestAnimationFrame(() => {
      if (focusHeading()) return;
      observer = new MutationObserver(() => {
        focusHeading();
      });
      observer.observe(document.getElementById("root") ?? document.body, {
        childList: true,
        subtree: true,
      });
      fallbackTimer = window.setTimeout(() => {
        if (interactionCancelled || focusHeading()) return;
        const main = document.querySelector<HTMLElement>("#main-content");
        if (main) focusElement(main);
      }, 600);
      stopTimer = window.setTimeout(stopWatching, 5_000);
      document.addEventListener("pointerdown", cancelForInteraction, true);
      document.addEventListener("keydown", cancelForInteraction, true);
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      stopWatching();
    };
  }, [
    authed,
    canOpenProjectRoute,
    loading,
    location.hash,
    location.search,
    pathname,
    platform.loading,
    routeRedirect,
    sessionReady,
    workspaceRoute,
  ]);

  useEffect(() => {
    const shouldLoadSharedData =
      routeRedirect === null &&
      (
        workspaceRoute === "/my-work" ||
        workspaceRoute === "/projects" ||
        workspaceRoute === "/training" ||
        workspaceRoute === "/models"
      );
    if (
      !authed ||
      !sessionReady ||
      !shouldLoadSharedData ||
      projectsReady
    ) {
      return;
    }

    let cancelled = false;
    const generation = sessionGenerationRef.current;

    async function loadInitialProjects(): Promise<void> {
      setError(null);
      try {
        await loadProjects(
          undefined,
          true,
          activeWorkspaceId,
          !canPerform(access, "projects:read"),
        );
      } catch (caught) {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setError(caught instanceof Error ? caught.message : "Unable to load projects");
        }
      } finally {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setProjectsReady(true);
        }
      }
    }

    void loadInitialProjects();
    return () => {
      cancelled = true;
    };
  }, [
    activeWorkspaceId,
    access,
    authed,
    loadProjects,
    projectsReady,
    routeRedirect,
    sessionReady,
    setError,
    workspaceRoute,
  ]);

  useEffect(() => {
    if (
      !authed ||
      !sessionReady ||
      activeWorkspaceId === null ||
      roundRouteId === null
    ) {
      roundContextRequestIdRef.current += 1;
      setResolvedRoundContext(null);
      setRoundResolutionLoading(false);
      setRoundResolutionError(null);
      return;
    }

    const requestId = ++roundContextRequestIdRef.current;
    setResolvedRoundContext(null);
    setRoundResolutionLoading(true);
    setRoundResolutionError(null);
    void getRoundWorkContext(roundRouteId, activeWorkspaceId)
      .then((context) => {
        if (requestId !== roundContextRequestIdRef.current) return;
        setResolvedRoundContext(context);
        setSelectedProjectId(context.round.project_id);
      })
      .catch((caught: unknown) => {
        if (requestId !== roundContextRequestIdRef.current) return;
        setRoundResolutionError(
          caught instanceof Error
            ? caught.message
            : "This annotation round could not be opened.",
        );
      })
      .finally(() => {
        if (requestId === roundContextRequestIdRef.current) {
          setRoundResolutionLoading(false);
        }
      });

    return () => {
      if (requestId === roundContextRequestIdRef.current) {
        roundContextRequestIdRef.current += 1;
      }
    };
  }, [
    activeWorkspaceId,
    authed,
    roundRouteId,
    sessionReady,
    setSelectedProjectId,
  ]);

  useEffect(() => {
    if (
      !authed ||
      !sessionReady ||
      activeWorkspaceId === null ||
      workspaceRoute !== "/my-work" ||
      roundRouteId !== null ||
      routeRedirect !== null ||
      !canPerform(access, "annotation:work")
    ) {
      roundQueueRequestIdRef.current += 1;
      setRoundContexts([]);
      setRoundContextsLoading(false);
      setRoundContextsError(null);
      return;
    }

    const requestId = ++roundQueueRequestIdRef.current;
    setRoundContexts([]);
    setRoundContextsLoading(true);
    setRoundContextsError(null);
    void listWorkspaceRoundWorkContexts(activeWorkspaceId)
      .then((contexts) => {
        if (requestId === roundQueueRequestIdRef.current) {
          setRoundContexts(contexts);
        }
      })
      .catch((caught: unknown) => {
        if (requestId !== roundQueueRequestIdRef.current) return;
        setRoundContextsError(
          caught instanceof Error
            ? caught.message
            : "Assigned annotation rounds could not be loaded.",
        );
      })
      .finally(() => {
        if (requestId === roundQueueRequestIdRef.current) {
          setRoundContextsLoading(false);
        }
      });

    return () => {
      if (requestId === roundQueueRequestIdRef.current) {
        roundQueueRequestIdRef.current += 1;
      }
    };
  }, [
    access,
    activeWorkspaceId,
    authed,
    roundRouteId,
    routeRedirect,
    sessionReady,
    workspaceRoute,
  ]);

  useEffect(() => {
    const shouldLoadSelectedProject =
      routeRedirect === null &&
      projectsReady &&
      (
        (
          workspaceRoute === "/my-work" &&
          roundRouteId === null
        ) ||
        (
          canPerform(access, "tasks:manage") &&
          parsedProjectRoute !== null &&
          canOpenProjectRoute
        )
      );
    if (!authed || !sessionReady || !shouldLoadSelectedProject) {
      return;
    }

    if (activeProjectContextId === null) {
      clearProjectData();
      setWorkbench(null);
      return;
    }

    const projectId = activeProjectContextId;
    const scope = workspaceRoute === "/my-work" ? "mine" : "all";
    let cancelled = false;
    const generation = sessionGenerationRef.current;
    async function loadSelectedProjectData(): Promise<void> {
      setBusy(true);
      setError(null);
      try {
        await loadProjectData(projectId, false, scope);
      } catch (caught) {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setError(caught instanceof Error ? caught.message : "Unable to load project data");
        }
      } finally {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setBusy(false);
        }
      }
    }

    void loadSelectedProjectData();
    return () => {
      cancelled = true;
    };
  }, [
    access,
    authed,
    canOpenProjectRoute,
    clearProjectData,
    loadProjectData,
    activeProjectContextId,
    parsedProjectRoute,
    projectsReady,
    roundRouteId,
    routeRedirect,
    selectedProjectId,
    sessionReady,
    setBusy,
    setError,
    workspaceRoute,
  ]);

  useEffect(() => {
    if (!authed || !sessionReady || workspaceRoute !== "/my-work") {
      return;
    }

    if (selectedDocumentId === null) {
      setWorkbench(null);
      return;
    }

    const documentId = selectedDocumentId;
    let cancelled = false;
    const generation = sessionGenerationRef.current;
    setWorkbench((current) => (current?.document.id === documentId ? current : null));
    async function loadWorkbench(): Promise<void> {
      setBusy(true);
      setError(null);
      try {
        const nextWorkbench = await getAnnotationWorkbench(documentId);
        const current = useProjectWorkspaceStore.getState();
        if (
          !cancelled &&
          generation === sessionGenerationRef.current &&
          current.selectedDocumentId === documentId &&
          current.selectedProjectId === nextWorkbench.project.id &&
          nextWorkbench.document.id === documentId
        ) {
          setWorkbench(nextWorkbench);
        }
      } catch (caught) {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setError(caught instanceof Error ? caught.message : "Unable to load workbench");
        }
      } finally {
        if (!cancelled && generation === sessionGenerationRef.current) {
          setBusy(false);
        }
      }
    }

    void loadWorkbench();
    return () => {
      cancelled = true;
    };
  }, [authed, selectedDocumentId, sessionReady, setBusy, setError, workspaceRoute]);

  // Password activation/reset is public because the user may not have a usable
  // account yet. Completion adopts the returned session and the token-change
  // subscription reloads the authenticated shell.
  if (accountActionToken !== null) {
    return (
      <Suspense
        fallback={(
          <main id="main-content" tabIndex={-1}>
            <div className="status" role="status">Loading secure link…</div>
          </main>
        )}
      >
        <AccountActionPage
          token={accountActionToken}
          onCompleted={() => navigatePath("/", "replace")}
        />
      </Suspense>
    );
  }

  // Invite acceptance is resolved before the auth gate below, which otherwise
  // renders the login page for every unauthenticated path and discards the
  // URL. An invitee typically has no account yet, so this route has to survive
  // that gate.
  if (inviteToken !== null) {
    if (authed && !sessionReady) {
      return (
        <main id="main-content" tabIndex={-1}>
          <div className="status" role="status" aria-live="polite">
            Checking your session…
          </div>
        </main>
      );
    }
    return (
      <AcceptInvitePage
        token={inviteToken}
        signedIn={authed && sessionReady}
        onAccepted={() => {
          // A new or switched account already reset state through the token
          // subscription. An existing signed-in user keeps their token, so bump
          // the session generation to re-run the workspace load and pick up the
          // new membership.
          setSessionReady(false);
          setJustRegistered(false);
          setOnboarded(true);
          sessionGenerationRef.current += 1;
          setSessionGeneration(sessionGenerationRef.current);
          setAuthed(true);
          navigatePath("/", "replace");
        }}
      />
    );
  }

  if (!authed) {
    return (
      <LoginPage
        onAuthed={(registered) => {
          setSessionReady(false);
          setJustRegistered(registered);
          setOnboarded(!registered);
          setAuthed(true);
        }}
      />
    );
  }

  if (!sessionReady) {
    return (
      <div className="app-shell">
        <main id="main-content" tabIndex={-1}>
          <div className="status" role="status" aria-live="polite">
            Loading workspace…
          </div>
        </main>
      </div>
    );
  }

  if (justRegistered && !onboarded) {
    if (activeWorkspaceId === null) {
      return (
        <div className="app-shell">
          <main id="main-content" tabIndex={-1}>
            <div className="status" role="status" aria-live="polite">
              Loading workspace…
            </div>
          </main>
        </div>
      );
    }
    return (
      <OnboardingPresetPicker
        isPersonalWorkspace={activeWorkspaceIsIndividual}
        workspaceId={activeWorkspaceId}
        onDone={async () => {
          const refreshed = await refreshWorkspaceConfiguration(
            activeWorkspaceId,
          );
          const refreshedNavigationContext: ModuleNavigationContext = {
            ...createAccessSnapshot({
              workspaceId: activeWorkspaceId,
              workspaceKind: activeMembership?.workspace_kind,
              membershipRole: activeMembership?.role,
              effectiveCapabilities: refreshed.effectiveCapabilities,
              blockedCapabilities: refreshed.blockedCapabilities,
              isSuperuser,
            }),
            selectedProjectId: null,
          };
          setOnboarded(true);
          setJustRegistered(false);
          navigatePath(
            defaultModulePath(refreshedNavigationContext),
            "replace",
          );
        }}
      />
    );
  }

  const workspaceSwitcher = (
    <WorkspaceSwitcher
      memberships={memberships}
      activeWorkspaceId={activeWorkspaceId}
      onWorkspaceChange={handleWorkspaceChange}
      onManageWorkspaces={() => setWorkspaceDialogOpen(true)}
    />
  );
  const mobileWorkspaceSwitcher = (
    <WorkspaceSwitcher
      memberships={memberships}
      activeWorkspaceId={activeWorkspaceId}
      ariaLabel="Switch workspace"
      onWorkspaceChange={handleWorkspaceChange}
      onManageWorkspaces={() => setWorkspaceDialogOpen(true)}
    />
  );
  const renderNotFound = (): React.ReactElement => (
    <main
      id="main-content"
      className="not-found-shell"
      tabIndex={-1}
      aria-labelledby="not-found-title"
    >
      <div className="not-found-content">
        <p className="not-found-code">404</p>
        <h1 id="not-found-title">Page not found</h1>
        <p>
          The URL <code>{pathname}</code> does not match a workspace in this
          app.
        </p>
        <div className="not-found-actions">
          {moduleDestinations.map((module, index) => (
            <Button
              key={module.id}
              label={`Open ${module.label}`}
              variant={index === 0 ? "primary" : undefined}
              href={module.path}
              onClick={(event) => {
                if (
                  event.button !== 0 ||
                  event.metaKey ||
                  event.ctrlKey ||
                  event.shiftKey ||
                  event.altKey
                ) {
                  return;
                }
                event.preventDefault();
                navigatePath(module.path);
              }}
            />
          ))}
        </div>
      </div>
    </main>
  );

  const renderNoAccess = (): React.ReactElement => (
    <main
      id="main-content"
      className="not-found-shell"
      tabIndex={-1}
      aria-labelledby="no-access-title"
    >
      <div className="not-found-content">
        <h1 id="no-access-title">No modules available</h1>
        <p>
          This membership does not currently include access to a released
          module. Switch workspaces or ask a workspace administrator to review
          your role and enabled capabilities.
        </p>
      </div>
    </main>
  );

  const renderWorkspaceSettings = (): React.ReactElement =>
    activeWorkspaceId === null || activeMembership === null ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status">Loading workspace settings…</div>
      </main>
    ) : (
      <WorkspaceSettings
        workspaceId={activeWorkspaceId}
        workspaceName={activeMembership.workspace_name}
        workspaceKind={activeMembership.workspace_kind}
        currentUserId={currentUserId}
        onManageWorkspaces={() => setWorkspaceDialogOpen(true)}
        onCapabilitiesChanged={(nextCapabilities) => {
          setCaps(nextCapabilities.effective);
          setBlockedCaps(nextCapabilities.blocked);
        }}
      />
    );
  const renderSystemAdministration = (): React.ReactElement => (
    <SystemAdministration pathname={pathname} onNavigate={navigatePath} />
  );

  const renderProjects = (): React.ReactElement => (
    <>
      {error ? (
        <Banner
          className="shell-banner"
          status="error"
          title="Projects could not be loaded"
          description={error}
          container="section"
        />
      ) : null}
      <ProjectsWorkspace
        workspaceId={activeWorkspaceId}
        projects={projects}
        canCreate={canPerform(access, "projects:create")}
        capabilities={caps}
        loading={!projectsReady || loading}
        onOpenProject={(projectId) => {
          setSelectedProjectId(projectId);
          navigatePath(projectPlatformPath(projectId, "overview"));
        }}
        onProjectCreated={(project) => {
          replaceProject(project);
          setSelectedProjectId(project.id);
          navigatePath(projectPlatformPath(project.id, "overview"));
        }}
      />
    </>
  );

  const renderProject = (projectId: number | null): React.ReactElement => {
    const routedProject =
      projectId === null
        ? null
        : projects.find((project) => project.id === projectId) ?? null;
    const platformProjectReady =
      projectId !== null &&
      platform.data.projectModules.project_id === projectId;
    return projectId === null || parsedProjectRoute === null ? (
      renderNotFound()
    ) : !canOpenProjectRoute ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status" aria-live="polite">
          Opening project overview…
        </div>
      </main>
    ) : routedProject === null ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status" aria-live="polite">
          {!projectsReady || loading ? "Loading project…" : "Project not found."}
        </div>
      </main>
    ) : !platformProjectReady ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status" aria-live="polite">
          Loading project…
        </div>
      </main>
    ) : (
      <ProjectPlatform
        pathname={pathname}
        project={routedProject}
        projects={projects}
        documents={documents}
        assignments={assignments}
        progress={projectProgress}
        data={platform.data}
        loading={loading || platform.loading}
        busy={busy || platform.busy}
        error={error ?? platform.error}
        access={access}
        currentUserId={currentUserId}
        onUpdateModules={async (selected) => {
          const configuration = await platform.setModules(selected);
          replaceProject({
            ...routedProject,
            settings: {
              ...routedProject.settings,
              modules: configuration.selected,
            },
          });
        }}
        onRefresh={platform.reload}
        dialog={platformDialog}
        onDialogChange={setPlatformDialog}
        onProjectSelect={(projectId, nextTab) => {
          setSelectedProjectId(projectId);
          navigatePath(projectPlatformPath(projectId, nextTab));
        }}
        onNavigate={navigatePath}
        onOpenRound={(roundId) => navigatePath(`/my-work/rounds/${roundId}`)}
        onUpdateProject={async (payload: ProjectUpdate) => {
          const updated = await updateProject(routedProject.id, payload);
          replaceProject(updated);
        }}
        dialogContent={
          platformDialog ? (
            <Suspense fallback={null}>
              <PlatformDialog
                kind={platformDialog}
                data={platform.data}
                busy={platform.busy}
                onClose={() => setPlatformDialog(null)}
                onCreateDataset={platform.addDataset}
                onCreateCycle={platform.addCycle}
                onCreateRound={platform.addRound}
                onScoreFeedback={platform.scoreFeedback}
                onCreateGuideline={platform.addGuideline}
                onCreateTask={platform.addTask}
                onPrepareTrainingData={platform.prepareTrainingData}
              />
            </Suspense>
          ) : undefined
        }
      />
    );
  };

  const renderTraining = (
    view: TrainingRouteView,
    runId: number | null,
  ): React.ReactElement =>
    activeWorkspaceId === null ||
    currentUserId === null ||
    currentUsername === null ||
    !projectsReady ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status">Loading training workspace…</div>
      </main>
    ) : (
      <TrainingWorkspace
        workspaceId={activeWorkspaceId}
        projects={projects}
        view={view}
        runId={runId}
        initialProjectId={queryProjectId}
        initialDatasetId={queryDatasetId}
        currentUserId={currentUserId}
        currentUserName={currentUsername}
        canCreateTrainingProject={canPerform(access, "projects:create")}
        canCreateTask={canPerform(access, "tasks:manage")}
        canProvisionRuntimes={canPerform(access, "runtime:provision")}
        onNavigate={navigatePath}
        onProjectCreated={(project) => {
          replaceProject(project);
        }}
      />
    );

  const renderModels = (
    view: ModelsRouteView,
    modelId: number | null,
    versionId: number | null,
  ): React.ReactElement =>
    activeWorkspaceId === null || !projectsReady ? (
      <main id="main-content" tabIndex={-1}>
        <div className="status" role="status">Loading model registry…</div>
      </main>
    ) : (
      <ModelsWorkspace
        workspaceId={activeWorkspaceId}
        projects={projects}
        view={view}
        modelId={modelId}
        versionId={versionId}
        initialProjectId={queryProjectId}
        onNavigate={navigatePath}
      />
    );

  const renderRound = (roundId: number | null): React.ReactElement => {
    const context =
      roundId !== null && resolvedRoundContext?.round.id === roundId
        ? resolvedRoundContext
        : null;
    return (
      <main id="main-content" className="round-work-main" tabIndex={-1}>
      {roundResolutionLoading ? (
        <div className="status" role="status" aria-live="polite">
          Loading round…
        </div>
      ) : context ? (
        <RoundWorkbench
          round={context.round}
          task={context.task_version}
          currentUserId={currentUserId}
          canManage={
            canPerform(access, "rounds:manage")
          }
          onClose={() => navigatePath("/my-work")}
          onRefresh={async () => {
            const requestId = ++roundContextRequestIdRef.current;
            if (activeWorkspaceId === null) return;
            const refreshed = await getRoundWorkContext(
              context.round.id,
              activeWorkspaceId,
            );
            if (
              requestId === roundContextRequestIdRef.current &&
              roundRouteId === context.round.id
            ) {
              setResolvedRoundContext(refreshed);
            }
          }}
        />
      ) : (
        <div className="not-found-content">
          <h1>Round not found</h1>
          <p>
            {roundResolutionError ??
              "This round is unavailable in the selected project or workspace."}
          </p>
          <Button
            label="Back to My Work"
            variant="primary"
            onClick={() => navigatePath("/my-work")}
          />
        </div>
      )}
      </main>
    );
  };

  const renderMyWork = (): React.ReactElement => (
    <>
      {error ? (
        <Banner
          className="shell-banner"
          status="error"
          title="Workspace error"
          description={error}
          container="section"
        />
      ) : null}
      {roundContextsError ? (
        <Banner
          className="shell-banner"
          status="error"
          title="Assigned rounds unavailable"
          description={roundContextsError}
          container="section"
        />
      ) : null}
      {loading ? (
        <div className="status" role="status" aria-live="polite">
          Loading…
        </div>
      ) : null}
      {roundContextsLoading ? (
        <div className="status" role="status" aria-live="polite">
          Loading assigned rounds…
        </div>
      ) : null}
      <AnnotatorWorkspace
        projects={projects}
        selectedProject={selectedProject}
        selectedProjectId={selectedProjectId}
        setSelectedProjectId={setSelectedProjectId}
        documents={documents}
        assignments={assignments}
        annotatorId={currentUsername ?? ""}
        roundContexts={roundContexts}
        projectProgress={projectProgress}
        selectedDocumentId={selectedDocumentId}
        setSelectedDocumentId={setSelectedDocumentId}
        workbench={workbench}
        setWorkbench={setWorkbench}
        busy={busy}
        setBusy={setBusy}
        setError={setError}
        refreshProjectData={async (projectId) => {
          await refreshProjectData(projectId);
        }}
        refreshWorkbench={refreshWorkbench}
        allowAssignmentlessSubmit={activeWorkspaceIsIndividual}
        onOpenProjectTab={(tab) => {
          if (selectedProjectId === null) {
            navigatePath("/projects");
            return;
          }
          navigatePath(`/projects/${selectedProjectId}/${tab}`);
        }}
        onOpenRound={(nextRoundId) =>
          navigatePath(`/my-work/rounds/${nextRoundId}`)
        }
      />
    </>
  );

  const routeContext: ApplicationRouteContext = {
    renderMyWork,
    renderRound,
    renderProjects,
    renderProject,
    renderTraining,
    renderModels,
    renderWorkspaceSettings,
    renderSystemAdministration,
    renderNoAccess,
    renderNotFound,
  };

  return (
    <>
      <Routes>
        <Route
          element={(
            <AuthenticatedLayout
            access={access}
            modules={moduleDestinations}
            currentModuleId={activeModuleId}
            workspaceSettings={workspaceSettings}
            workspaceSettingsCurrent={
              workspaceRoute === "/workspace-settings"
            }
            homePath={defaultModulePath(navigationContext)}
            workspaceSwitcher={workspaceSwitcher}
            mobileWorkspaceSwitcher={mobileWorkspaceSwitcher}
            announcement={routeAnnouncement}
            onNavigate={navigatePath}
            onLogout={handleLogout}
            redirecting={routeRedirect !== null}
            routeContext={routeContext}
          />
          )}
        >
        <Route path="/" element={<MyWorkRoute />} />
        <Route path="/annotator/workbench" element={<MyWorkRoute />} />
        <Route path="/my-work">
          <Route index element={<MyWorkRoute />} />
          <Route path="rounds/:roundId" element={<RoundRoute />} />
        </Route>
        <Route path="/manager/projects" element={<ProjectsRoute />} />
        <Route path="/projects">
          <Route index element={<ProjectsRoute />} />
          <Route
            path=":projectId/train"
            element={<TrainingRoute view="runs" />}
          />
          <Route
            path=":projectId/models"
            element={<ModelsRoute view="registry" />}
          />
          <Route path=":projectId/*" element={<ProjectRoute />} />
        </Route>
        <Route
          path="/trainer/training"
          element={<TrainingRoute view="runs" />}
        />
        <Route path="/training">
          <Route index element={<TrainingRoute view="runs" />} />
          <Route path="new" element={<TrainingRoute view="new" />} />
          <Route path="data" element={<TrainingRoute view="data" />} />
          <Route
            path="runtimes"
            element={<TrainingRoute view="runtimes" />}
          />
          <Route
            path="runs/:runId"
            element={<TrainingRoute view="run-detail" />}
          />
        </Route>
        <Route path="/models">
          <Route index element={<ModelsRoute view="registry" />} />
          <Route path=":modelId" element={<ModelsRoute view="detail" />} />
          <Route
            path=":modelId/versions/:versionId"
            element={<ModelsRoute view="version" />}
          />
        </Route>
        <Route
          path="/workspace-settings"
          element={<WorkspaceSettingsRoute />}
        />
        <Route path="/admin/*" element={<SystemAdministrationRoute />} />
        <Route path="/no-access" element={<NoAccessRoute />} />
        <Route path="*" element={<NotFoundRoute />} />
        </Route>
      </Routes>
      <AddWorkspaceDialog
        open={workspaceDialogOpen}
        requestGeneration={sessionGeneration}
        onDismiss={() => setWorkspaceDialogOpen(false)}
        onWorkspaceCreated={handleWorkspaceCreated}
        onJoinRequested={refreshMemberships}
      />
    </>
  );
}
