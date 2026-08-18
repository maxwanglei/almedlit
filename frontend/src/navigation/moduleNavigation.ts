import type { CapabilityKey } from "@/auth/capabilities";
import {
  canPerform,
  type AccessCommand,
  type AccessSnapshot,
} from "@/navigation/AccessContext";
import type { WorkspaceRole } from "@/types/api";

export type WorkspaceRoute =
  | "/my-work"
  | "/projects"
  | "/training"
  | "/models"
  | "/workspace-settings"
  | "/admin"
  | "/no-access"
  | "/not-found";

export type CanonicalModuleId =
  | "my-work"
  | "projects"
  | "training"
  | "models"
  | "administration";
export type ModuleId = CanonicalModuleId;
export type ModuleRole = WorkspaceRole;
export type RouteReleaseState = "released" | "legacy";
export type ProjectSectionId =
  | "overview"
  | "data"
  | "tasks"
  | "rounds"
  | "quality"
  | "activity"
  | "settings";

export interface ModuleNavigationContext extends AccessSnapshot {
  /**
   * Retained as optional compatibility state for callers that also manage a
   * project selector. Global module destinations never depend on it.
   */
  selectedProjectId?: number | null;
}

export interface ModuleDestination {
  id: ModuleId;
  label: string;
  path: string;
}

export interface UtilityDestination {
  id: "workspace-settings";
  label: string;
  path: string;
}

export interface ModuleRouteDefinition extends ModuleDestination {
  id: CanonicalModuleId;
  title: string;
  level: "global";
  requiredRoles: readonly ModuleRole[] | null;
  requiredCapability: CapabilityKey | null;
  requiredCommand: AccessCommand;
  backendModule: string | null;
  releaseState: RouteReleaseState;
}

export const GLOBAL_ROUTE_REGISTRY = [
  {
    id: "my-work",
    label: "My Work",
    title: "My Work",
    path: "/my-work",
    level: "global",
    requiredRoles: ["annotator"],
    requiredCapability: "annotation",
    requiredCommand: "annotation:work",
    backendModule: "annotate",
    releaseState: "released",
  },
  {
    id: "projects",
    label: "Projects",
    title: "Projects",
    path: "/projects",
    level: "global",
    requiredRoles: ["trainer", "manager", "admin"],
    requiredCapability: null,
    requiredCommand: "projects:read",
    backendModule: null,
    releaseState: "released",
  },
  {
    id: "training",
    label: "Training",
    title: "Training",
    path: "/training",
    level: "global",
    requiredRoles: ["trainer", "manager", "admin"],
    requiredCapability: "training",
    requiredCommand: "training:read",
    backendModule: "train",
    releaseState: "released",
  },
  {
    id: "models",
    label: "Models",
    title: "Models",
    path: "/models",
    level: "global",
    requiredRoles: ["trainer", "manager", "admin"],
    requiredCapability: "training",
    requiredCommand: "models:read",
    backendModule: "models",
    releaseState: "released",
  },
  {
    id: "administration",
    label: "Administration",
    title: "System administration",
    path: "/admin/users",
    level: "global",
    requiredRoles: null,
    requiredCapability: null,
    requiredCommand: "system:admin",
    backendModule: null,
    releaseState: "released",
  },
] as const satisfies readonly ModuleRouteDefinition[];

export const WORKSPACE_SETTINGS_DESTINATION = {
  id: "workspace-settings",
  label: "Workspace Settings",
  path: "/workspace-settings",
} as const satisfies UtilityDestination;

const PROJECT_SECTIONS = new Set([
  "overview",
  "data",
  "tasks",
  "rounds",
  "quality",
  "activity",
  "settings",
  // Compatibility sections retained until legacy usage is retired.
  "annotate",
  "documents",
  "export",
  "progress",
  "learning",
  "loop",
  "guidelines",
]);
const TRAINING_SECTIONS = new Set(["new", "data", "runtimes"]);
const ADMIN_SECTIONS = new Set(["users", "plugins", "health", "audit", "settings"]);

export function normalizePathname(pathname: string): string {
  if (pathname === "/") {
    return pathname;
  }
  return pathname.replace(/\/+$/, "");
}

export function resolveWorkspaceRoute(pathname: string): WorkspaceRoute {
  const normalizedPathname = normalizePathname(pathname);

  if (normalizedPathname === "/no-access") {
    return "/no-access";
  }

  if (
    normalizedPathname === "/" ||
    normalizedPathname === "/my-work" ||
    normalizedPathname === "/annotator/workbench" ||
    /^\/my-work\/rounds\/\d+$/.test(normalizedPathname)
  ) {
    return "/my-work";
  }

  if (normalizedPathname === "/trainer/training") {
    return "/training";
  }
  if (
    normalizedPathname === "/training" ||
    TRAINING_SECTIONS.has(normalizedPathname.slice("/training/".length)) ||
    /^\/training\/runs\/\d+$/.test(normalizedPathname)
  ) {
    return "/training";
  }

  if (
    normalizedPathname === "/models" ||
    /^\/models\/\d+$/.test(normalizedPathname) ||
    /^\/models\/\d+\/versions\/\d+$/.test(normalizedPathname)
  ) {
    return "/models";
  }

  if (
    normalizedPathname === "/projects" ||
    normalizedPathname === "/manager/projects"
  ) {
    return "/projects";
  }

  const projectMatch = normalizedPathname.match(
    /^\/projects\/(\d+)(?:\/([^/]+))?$/,
  );
  if (projectMatch) {
    if (projectMatch[2] === "train") {
      return "/training";
    }
    if (projectMatch[2] === "models") {
      return "/models";
    }
    if (!projectMatch[2] || PROJECT_SECTIONS.has(projectMatch[2])) {
      return "/projects";
    }
  }

  if (normalizedPathname === "/workspace-settings") {
    return "/workspace-settings";
  }

  const adminMatch = normalizedPathname.match(/^\/admin(?:\/([^/]+))?$/);
  if (adminMatch && (!adminMatch[1] || ADMIN_SECTIONS.has(adminMatch[1]))) {
    return "/admin";
  }

  return "/not-found";
}

export function legacyModelsProjectId(pathname: string): number | null {
  const match = normalizePathname(pathname).match(
    /^\/projects\/(\d+)(?:\/[^/]+)?$/,
  );
  if (!match) {
    return null;
  }
  const projectId = Number(match[1]);
  return Number.isSafeInteger(projectId) ? projectId : null;
}

function routeIsAvailable(
  route: ModuleRouteDefinition,
  context: ModuleNavigationContext,
): boolean {
  return canPerform(context, route.requiredCommand);
}

export function availableModules(
  context: ModuleNavigationContext,
): ModuleDestination[] {
  return GLOBAL_ROUTE_REGISTRY.filter((route) =>
    routeIsAvailable(route, context),
  ).map(({ id, label, path }) => ({ id, label, path }));
}

export function workspaceSettingsDestination(
  context: ModuleNavigationContext,
): UtilityDestination | null {
  return canPerform(context, "workspace:manage")
    ? WORKSPACE_SETTINGS_DESTINATION
    : null;
}

export function canAccessProjectSection(
  context: ModuleNavigationContext,
  section: ProjectSectionId,
): boolean {
  if (!canPerform(context, "projects:read")) {
    return false;
  }
  if (
    section === "tasks" ||
    section === "rounds" ||
    section === "quality"
  ) {
    return (
      canPerform(context, "tasks:manage") &&
      context.effectiveCapabilities.includes("annotation") &&
      context.blockedCapabilities.annotation === undefined
    );
  }
  if (section === "settings") {
    return canPerform(context, "projects:create");
  }
  return true;
}

export function currentModuleId(
  pathname: string,
  context: ModuleNavigationContext,
): ModuleId | null {
  void context;
  switch (resolveWorkspaceRoute(pathname)) {
    case "/my-work":
      return "my-work";
    case "/projects":
      return "projects";
    case "/training":
      return "training";
    case "/models":
      return "models";
    case "/admin":
      return "administration";
    default:
      return null;
  }
}

export function defaultModulePath(context: ModuleNavigationContext): string {
  return availableModules(context)[0]?.path ?? "/no-access";
}

function projectAliasTarget(pathname: string): string | null {
  const match = pathname.match(/^\/projects\/(\d+)\/(train|models)$/);
  if (!match) {
    return null;
  }
  return `/${match[2] === "train" ? "training" : "models"}?projectId=${match[1]}`;
}

export function compatibilityRedirectNotice(pathname: string): string | null {
  const normalizedPathname = normalizePathname(pathname);
  if (normalizedPathname === "/annotator/workbench") {
    return "The annotation workbench moved. My Work loaded.";
  }
  if (normalizedPathname === "/manager/projects") {
    return "The project console moved. Projects loaded.";
  }
  if (
    normalizedPathname === "/trainer/training" ||
    /^\/projects\/\d+\/train$/.test(normalizedPathname)
  ) {
    return "Training is now an independent workspace. Training loaded.";
  }
  if (/^\/projects\/\d+\/models$/.test(normalizedPathname)) {
    return "Models is now an independent workspace. Models loaded.";
  }
  return null;
}

/**
 * Returns a canonical replacement for a known legacy or unauthorized route.
 * Query strings and hashes are added separately with preserveLocation().
 */
export function authorizedRedirectPath(
  pathname: string,
  context: ModuleNavigationContext,
): string | null {
  const normalizedPathname = normalizePathname(pathname);
  const route = resolveWorkspaceRoute(normalizedPathname);
  if (route === "/not-found") {
    return null;
  }

  if (route === "/no-access") {
    return availableModules(context).length
      ? defaultModulePath(context)
      : null;
  }

  if (normalizedPathname === "/") {
    return defaultModulePath(context);
  }

  const legacyTarget =
    normalizedPathname === "/annotator/workbench"
      ? "/my-work"
      : normalizedPathname === "/manager/projects"
        ? "/projects"
        : normalizedPathname === "/trainer/training"
          ? "/training"
          : projectAliasTarget(normalizedPathname);

  if (legacyTarget !== null) {
    const targetModule = currentModuleId(legacyTarget, context);
    if (
      targetModule !== null &&
      !availableModules(context).some((module) => module.id === targetModule)
    ) {
      return defaultModulePath(context);
    }
    return legacyTarget;
  }

  if (route === "/workspace-settings") {
    return canPerform(context, "workspace:manage")
      ? null
      : defaultModulePath(context);
  }

  const moduleId = currentModuleId(normalizedPathname, context);
  const isAllowed =
    moduleId !== null &&
    availableModules(context).some((module) => module.id === moduleId);
  if (!isAllowed) {
    return defaultModulePath(context);
  }

  if (normalizedPathname === "/admin") {
    return "/admin/users";
  }
  return null;
}

export function preserveLocation(
  target: string,
  currentSearch = "",
  currentHash = "",
): string {
  const [targetWithoutHash, targetHash = ""] = target.split("#", 2);
  const [targetPath, targetSearch = ""] = targetWithoutHash.split("?", 2);
  const params = new URLSearchParams(
    currentSearch.startsWith("?") ? currentSearch.slice(1) : currentSearch,
  );
  const targetParams = new URLSearchParams(targetSearch);
  targetParams.forEach((value, key) => params.set(key, value));
  const nextSearch = params.toString();
  const nextHash = targetHash
    ? `#${targetHash}`
    : currentHash === "" || currentHash.startsWith("#")
      ? currentHash
      : `#${currentHash}`;
  return `${targetPath}${nextSearch ? `?${nextSearch}` : ""}${nextHash}`;
}
