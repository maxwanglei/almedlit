import type { CapabilityKey } from "@/auth/capabilities";
import type { Project } from "@/types/api";

import type { ProjectModule } from "./types";

export type ProjectPlatformTab =
  | "overview"
  | "data"
  | "inference"
  | "tasks"
  | "rounds"
  | "quality"
  | "activity"
  | "settings";

export type ProjectRouteRedirectKind = "canonical" | "legacy" | "planned";

export interface ProjectRouteRedirect {
  kind: ProjectRouteRedirectKind;
  notice: string | null;
}

export interface ProjectPlatformRoute {
  projectId: number;
  tab: ProjectPlatformTab;
  requestedTab: string | null;
  canonicalPath: string;
  redirect: ProjectRouteRedirect | null;
}

export interface ProjectRouteDefinition {
  id: ProjectPlatformTab;
  label: string;
  title: string;
  level: "project";
  requiredRoles: readonly ("trainer" | "manager" | "admin")[];
  requiredCapability: CapabilityKey | null;
  backendModule: ProjectModule | null;
  releaseState: "released";
}

const PROJECT_ROLES = ["trainer", "manager", "admin"] as const;
const PROJECT_MANAGERS = ["manager", "admin"] as const;

export const PROJECT_ROUTE_REGISTRY = [
  {
    id: "overview",
    label: "Overview",
    title: "Project overview",
    level: "project",
    requiredRoles: PROJECT_ROLES,
    requiredCapability: null,
    backendModule: null,
    releaseState: "released",
  },
  {
    id: "data",
    label: "Data",
    title: "Data",
    level: "project",
    requiredRoles: PROJECT_ROLES,
    requiredCapability: null,
    backendModule: "data",
    releaseState: "released",
  },
  {
    id: "inference", label: "Predictions", title: "Predictions", level: "project",
    requiredRoles: PROJECT_ROLES, requiredCapability: "inference", backendModule: "models", releaseState: "released",
  },
  {
    id: "tasks",
    label: "Tasks",
    title: "Tasks",
    level: "project",
    requiredRoles: PROJECT_MANAGERS,
    requiredCapability: "annotation",
    backendModule: "annotate",
    releaseState: "released",
  },
  {
    id: "rounds",
    label: "Team & rounds",
    title: "Team & rounds",
    level: "project",
    requiredRoles: PROJECT_ROLES,
    requiredCapability: "annotation",
    backendModule: "annotate",
    releaseState: "released",
  },
  {
    id: "quality",
    label: "Quality & review",
    title: "Quality & review",
    level: "project",
    requiredRoles: PROJECT_MANAGERS,
    requiredCapability: "annotation",
    backendModule: "annotate",
    releaseState: "released",
  },
  {
    id: "activity",
    label: "Activity",
    title: "Activity",
    level: "project",
    requiredRoles: PROJECT_ROLES,
    requiredCapability: null,
    backendModule: "activity",
    releaseState: "released",
  },
  {
    id: "settings",
    label: "Settings",
    title: "Project settings",
    level: "project",
    requiredRoles: PROJECT_MANAGERS,
    requiredCapability: null,
    backendModule: null,
    releaseState: "released",
  },
] as const satisfies readonly ProjectRouteDefinition[];

interface ProjectRouteAlias {
  tab: ProjectPlatformTab;
  kind: "legacy" | "planned";
  notice: string | null;
}

export const PROJECT_ROUTE_ALIASES = {
  annotate: { tab: "rounds", kind: "legacy", notice: null },
  documents: { tab: "data", kind: "legacy", notice: null },
  export: { tab: "data", kind: "legacy", notice: null },
  progress: { tab: "activity", kind: "legacy", notice: null },
  learning: {
    tab: "overview",
    kind: "planned",
    notice: "Active learning is planned. See the research loop roadmap on Project overview.",
  },
  loop: {
    tab: "overview",
    kind: "planned",
    notice: "Active learning is planned. See the research loop roadmap on Project overview.",
  },
  guidelines: {
    tab: "overview",
    kind: "planned",
    notice: "Guideline Learning is planned and is now summarized on Project overview.",
  },
} as const satisfies Record<string, ProjectRouteAlias>;

const PLATFORM_TABS = new Set<ProjectPlatformTab>(
  PROJECT_ROUTE_REGISTRY.map((route) => route.id),
);

const PROJECT_MODULE_DEPENDENCIES: Partial<
  Record<ProjectModule, readonly ProjectModule[]>
> = {
  annotate: ["data"],
  learning: ["data", "annotate"],
  train: ["data", "models"],
  guidelines: ["data", "annotate"],
};

const PROJECT_MODULE_REQUIREMENTS = {
  data: ["annotation", "training", "inference", "active_learning", "co_learning", "lineage"],
  annotate: ["annotation"],
  learning: ["active_learning", "co_learning"],
  train: ["training"],
  models: ["training", "inference", "lineage"],
  guidelines: ["co_learning"],
  activity: ["lineage"],
} satisfies Record<ProjectModule, readonly string[]>;

const PROJECT_MODULES = Object.keys(PROJECT_MODULE_REQUIREMENTS).filter(
  (module): module is ProjectModule => module in PROJECT_MODULE_REQUIREMENTS,
);

export function configuredProjectModules(project: Project): Set<ProjectModule> {
  const configured = project.settings.modules;
  if (configured === undefined || configured === null) return new Set(PROJECT_MODULES);
  if (!Array.isArray(configured)) return new Set();
  const selected = new Set<ProjectModule>(
    PROJECT_MODULES.filter((module) => configured.includes(module)),
  );
  let changed = true;
  while (changed) {
    changed = false;
    for (const module of [...selected]) {
      for (const dependency of PROJECT_MODULE_DEPENDENCIES[module] ?? []) {
        if (!selected.has(dependency)) {
          selected.add(dependency);
          changed = true;
        }
      }
    }
  }
  return selected;
}

export function projectSupportsSection(
  project: Project,
  tab: ProjectPlatformTab,
  effectiveCapabilities: readonly string[],
): boolean {
  const selected = configuredProjectModules(project);
  const available = new Set([...selected].filter((module) =>
    PROJECT_MODULE_REQUIREMENTS[module].some((capability) => effectiveCapabilities.includes(capability)),
  ));
  const effective = new Set([...available].filter((module) =>
    (PROJECT_MODULE_DEPENDENCIES[module] ?? []).every((dependency) => available.has(dependency)),
  ));
  if (tab === "inference") return effective.has("data") && effective.has("models");
  const module = projectRouteDefinition(tab).backendModule;
  return module === null || effective.has(module);
}

export function projectRouteDefinition(
  tab: ProjectPlatformTab,
): ProjectRouteDefinition {
  const definition = PROJECT_ROUTE_REGISTRY.find((route) => route.id === tab);
  if (!definition) {
    throw new Error(`Unknown project tab: ${tab}`);
  }
  return definition;
}

export function projectBackendModule(
  tab: ProjectPlatformTab,
): ProjectModule | null {
  return projectRouteDefinition(tab).backendModule;
}

export function parseProjectPlatformRoute(
  pathname: string,
): ProjectPlatformRoute | null {
  const normalizedPathname =
    pathname === "/" ? pathname : pathname.replace(/\/+$/, "");
  const match = normalizedPathname.match(/^\/projects\/(\d+)(?:\/([^/]+))?$/);
  if (!match) {
    return null;
  }

  const projectId = Number(match[1]);
  if (!Number.isSafeInteger(projectId)) {
    return null;
  }

  const requestedTab = match[2] ?? null;
  const alias =
    requestedTab === null
      ? null
      : PROJECT_ROUTE_ALIASES[
          requestedTab as keyof typeof PROJECT_ROUTE_ALIASES
        ] ?? null;
  const tab = alias?.tab ?? requestedTab ?? "overview";
  if (!PLATFORM_TABS.has(tab as ProjectPlatformTab)) {
    return null;
  }

  const canonicalTab = tab as ProjectPlatformTab;
  const canonicalPath = projectPlatformPath(projectId, canonicalTab);
  let redirect: ProjectRouteRedirect | null = null;
  if (alias) {
    redirect = { kind: alias.kind, notice: alias.notice };
  } else if (normalizedPathname !== canonicalPath) {
    redirect = { kind: "canonical", notice: null };
  }

  return {
    projectId,
    tab: canonicalTab,
    requestedTab,
    canonicalPath,
    redirect,
  };
}

export function projectPlatformPath(
  projectId: number,
  tab: ProjectPlatformTab,
): string {
  return `/projects/${projectId}/${tab}`;
}

export function projectSwitchPath(
  projectId: number,
  currentTab: ProjectPlatformTab,
  availableTabs: readonly ProjectPlatformTab[],
): string {
  const nextTab = availableTabs.includes(currentTab) ? currentTab : "overview";
  return projectPlatformPath(projectId, nextTab);
}

export function withSearchAndHash(
  pathname: string,
  search = "",
  hash = "",
): string {
  const normalizedSearch =
    search === "" || search.startsWith("?") ? search : `?${search}`;
  const normalizedHash = hash === "" || hash.startsWith("#") ? hash : `#${hash}`;
  return `${pathname}${normalizedSearch}${normalizedHash}`;
}
