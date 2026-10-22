import type { Route } from "@playwright/test";
import contract from "./backendContract.json";

const policy = contract.policy;
export type Capability = keyof typeof policy.capabilities;
export type CapabilityPreset = keyof typeof policy.presets | "custom";
export type ProjectModule = keyof typeof policy.module_requirements;
export type WorkspaceRole = keyof typeof policy.role_rank;
export type MockWorkspaceRole = WorkspaceRole | "personal";
export interface MockAccessOptions {
  capabilityPreset?: CapabilityPreset;
  capabilityOverrides?: Capability[];
  projectModules?: ProjectModule[] | null;
  actorUserId?: number;
  profile?: "laptop" | "lab";
}
export interface MockWorkspaceAccess {
  role: WorkspaceRole;
  kind: "individual" | "team";
  preset: CapabilityPreset;
  overrides: Capability[];
}
export interface MockAccessState {
  userId: number;
  username: string;
  profile: "laptop" | "lab";
  workspaces: Map<number, MockWorkspaceAccess>;
  projects: Map<number, { workspaceId: number; modules: ProjectModule[] }>;
}
export function createMockAccess(role: MockWorkspaceRole, options: MockAccessOptions = {}): MockAccessState {
  const userId = options.actorUserId ?? (role === "manager" ? 1 : 2);
  return {
    userId,
    username: userId === 1 ? "manager" : userId === 2 ? "alice" : userId === 3 ? "bob" : `user-${userId}`,
    profile: options.profile ?? "laptop",
    workspaces: new Map([[1, {
      role: role === "personal" ? "admin" : role,
      kind: role === "personal" ? "individual" : "team",
      preset: options.capabilityPreset ?? "full",
      overrides: options.capabilityOverrides ?? [],
    }]]),
    projects: new Map([[1, { workspaceId: 1, modules: selectedModules(options.projectModules) }]]),
  };
}
export function selectedModules(value: unknown): ProjectModule[] {
  if (value == null) return Object.keys(policy.module_requirements).filter(isModule);
  return Array.isArray(value)
    ? value.filter((item): item is ProjectModule => typeof item === "string" && isModule(item))
    : [];
}
function isModule(value: string): value is ProjectModule { return Object.hasOwn(policy.module_requirements, value); }
function isCapability(value: string): value is Capability { return Object.hasOwn(policy.capabilities, value); }
function closure(keys: readonly string[], dependencies: (key: string) => readonly string[]): Set<string> {
  const result = new Set(keys);
  for (const key of result) for (const dependency of dependencies(key)) result.add(dependency);
  return result;
}
export function capabilitiesFor(state: MockAccessState, workspaceId = 1) {
  const workspace = state.workspaces.get(workspaceId);
  if (!workspace) {
    const effective: Capability[] = [];
    return { preset: "custom", overrides: [], effective, blocked: {} };
  }
  const selected = workspace.preset === "custom" ? workspace.overrides : policy.presets[workspace.preset];
  const closed = closure(selected, (key) => isCapability(key) ? policy.capabilities[key].depends_on : []);
  const blocked: Record<string, string> = {};
  const effective = [...closed].filter(isCapability).filter((key) => {
    const item = policy.capabilities[key];
    if (item.requires_profile !== "any" && item.requires_profile !== state.profile) {
      blocked[key] = "requires the lab deployment profile";
      return false;
    }
    if (item.requires_team && workspace.kind !== "team") {
      blocked[key] = "available only in team workspaces";
      return false;
    }
    return true;
  }).sort();
  return { preset: workspace.preset, overrides: workspace.overrides, effective, blocked };
}
export function modulesFor(state: MockAccessState, projectId: number) {
  const selected = closure(state.projects.get(projectId)?.modules ?? [], (key) => isModule(key) ? policy.module_dependencies[key] : []);
  const capabilities = capabilitiesFor(state, state.projects.get(projectId)?.workspaceId ?? 1).effective;
  const available = [...selected].filter(isModule).filter((key) => policy.module_requirements[key].some((capability) => capabilities.some((item) => item === capability)));
  return {
    project_id: projectId,
    selected: policy.module_order.filter((key) => selected.has(key)),
    effective: policy.module_order.filter((key) => isModule(key) && available.includes(key) && policy.module_dependencies[key].every((dependency) => available.some((item) => item === dependency))),
    workspace_capabilities: capabilities,
  };
}
interface AccessRule { methods: readonly string[]; path: RegExp; role?: WorkspaceRole; capability?: Capability; modules?: readonly ProjectModule[]; allModules?: readonly ProjectModule[]; }
const rules: AccessRule[] = [
  { methods: ["GET"], path: /^\/api\/(?:auth\/me|training\/recipes)$/ },
  { methods: ["POST"], path: /^\/api\/training\/recipes\/[^/]+\/validate-configuration$/ },
  { methods: ["GET"], path: /^\/api\/(?:projects\/my-work|workspaces\/\d+\/(?:capabilities|members)|documents|guidelines|export-formats|annotation-workbench\/documents\/\d+|documents\/\d+\/structure|annotations(?:\/\d+|\/evidence-blocks\/commands)|projects\/\d+\/(?:assignments|progress|evidence-targets)|projects\/\d+\/documents\/\d+\/evidence-review-coverage)$/ },
  { methods: ["POST"], path: /^\/api\/(?:annotations|projects\/\d+\/documents\/\d+\/(?:submissions|evidence-review-coverage\/mark-reviewed))$/ },
  { methods: ["POST"], path: /^\/api\/projects\/\d+\/import\/pubmed\/preview$/, role: "manager" },
  { methods: ["POST"], path: /^\/api\/projects\/\d+\/(?:datasets\/\d+\/versions\/public-registry-snapshot|training-datasets\/(?:preview|prepare))$/, role: "trainer", modules: ["data"] },
  { methods: ["POST"], path: /^\/api\/training-recipes\/trusted\/[^/]+$/, role: "trainer", modules: ["train"] },
  { methods: ["GET"], path: /^\/api\/rounds\/\d+\/work-context$/ },
  { methods: ["GET"], path: /^\/api\/projects(?:\/\d+(?:\/tasks)?)?$/, role: "trainer" },
  { methods: ["POST", "PATCH"], path: /^\/api\/projects(?:\/\d+(?:\/tasks)?)?$/, role: "manager" },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/modules$/, role: "trainer" },
  { methods: ["GET"], path: /^\/api\/workspaces\/\d+\/join-requests$/, role: "admin" },
  { methods: ["GET"], path: /^\/api\/workspaces\/\d+\/(?:training-contexts|training-runs|models)$/, role: "trainer", capability: "training" },
  { methods: ["GET"], path: /^\/api\/workspaces\/\d+\/my-work\/rounds$/, capability: "annotation" },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/(?:corpus-snapshots|annotation-sets)$/, role: "trainer", capability: "lineage" },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/exports$/, role: "trainer", capability: "export" },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/(?:artifact-packages|base-models)$/, role: "trainer", capability: "training" },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/inference\/runs$/, capability: "inference" },
  { methods: ["GET"], path: /^\/api\/inference\/runs\/\d+$/, role: "trainer", capability: "inference" },
  { methods: ["GET"], path: /^\/api\/inference\/runs\/\d+\/predictions$/, capability: "inference" },
  { methods: ["POST"], path: /^\/api\/inference\/predictions\/\d+\/review$/, capability: "inference" },
  { methods: ["GET", "POST"], path: /^\/api\/projects\/\d+\/documents\/\d+\/evidence-adjudication$/, role: "manager" },
  { methods: ["GET"], path: /^\/api\/(?:tasks(?:\/versions)?|datasets(?:\/(?:versions|items|label-sets|split-maps|training-versions))?)$/, role: "trainer", modules: ["data"] },
  { methods: ["POST"], path: /^\/api\/tasks(?:\/versions)?$/, role: "manager", modules: ["data"] },
  { methods: ["POST"], path: /^\/api\/datasets(?:\/(?:versions|label-sets|split-maps|training-versions(?:\/compose)?))?$/, role: "trainer", modules: ["data"] },
  { methods: ["GET"], path: /^\/api\/(?:models(?:\/versions|\/\d+\/versions\/\d+\/evaluations)?|training-runs\/\d+\/evaluations)$/, role: "trainer", modules: ["models", "train", "learning"] },
  { methods: ["GET"], path: /^\/api\/(?:training-runs|training-recipes(?:\/versions)?)$/, role: "trainer", modules: ["train", "activity"] },
  { methods: ["GET"], path: /^\/api\/(?:environments|storage\/policies)$/, role: "trainer", modules: ["train"] },
  { methods: ["POST"], path: /^\/api\/training-runs(?:\/\d+\/(?:launch|cancel))?$/, role: "trainer", modules: ["train"] },
  { methods: ["POST"], path: /^\/api\/training-recipes(?:\/versions)?$/, role: "manager", modules: ["train"] },
  { methods: ["POST"], path: /^\/api\/(?:environments|storage\/policies)$/, role: "admin", modules: ["train"] },
  { methods: ["POST"], path: /^\/api\/models$/, role: "trainer", modules: ["models", "train"] },
  { methods: ["GET"], path: /^\/api\/cycles$/, role: "trainer", modules: ["learning", "activity"] },
  { methods: ["POST"], path: /^\/api\/cycles$/, role: "trainer", modules: ["learning"] },
  { methods: ["GET"], path: /^\/api\/rounds$/, role: "trainer", modules: ["annotate", "activity"] },
  { methods: ["POST"], path: /^\/api\/rounds(?:\/\d+\/transition)?$/, role: "manager", modules: ["annotate"] },
  { methods: ["GET"], path: /^\/api\/rounds\/\d+(?:\/items)?$/, role: "manager", modules: ["annotate"] },
  { methods: ["GET"], path: /^\/api\/rounds\/\d+\/(?:work-items|decisions|submissions)$/, modules: ["annotate"] },
  { methods: ["GET"], path: /^\/api\/feedback-runs(?:\/sets)?$/, role: "trainer", modules: ["learning", "annotate"] },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/feedback-runs\/\d+$/, role: "trainer", modules: ["learning"] },
  { methods: ["GET"], path: /^\/api\/feedback-runs\/(?:events|review-cases)$/, role: "manager", modules: ["learning", "activity"] },
  { methods: ["POST"], path: /^\/api\/(?:feedback-runs|projects\/\d+\/(?:feedback-runs|selection-runs)\/\d+\/materialize|selection-runs)$/, role: "trainer", modules: ["learning"] },
  { methods: ["GET"], path: /^\/api\/workflow-guidelines(?:\/(?:revisions|proposals|impact-evaluations))?$/, role: "manager", modules: ["guidelines", "activity"] },
  { methods: ["POST"], path: /^\/api\/workflow-guidelines(?:\/.*)?$/, role: "manager", modules: ["guidelines"] },
  { methods: ["GET"], path: /^\/api\/projects\/\d+\/datasets\/annotation-progress$/, role: "trainer", modules: ["data"] },
  { methods: ["POST"], path: /^\/api\/projects\/\d+\/prediction-runs\/\d+\/review-round$/, role: "manager", capability: "inference", allModules: ["data", "models", "annotate"] },
  { methods: ["GET", "POST"], path: /^\/api\/projects\/\d+\/prediction-runs(?:\/\d+(?:\/(?:retry|results|download))?)?$/, role: "trainer", capability: "inference", allModules: ["data", "models"] },
];
export async function enforceMockAccess(route: Route, state: MockAccessState): Promise<boolean> {
  const url = new URL(route.request().url());
  const rule = rules.find((candidate) => candidate.methods.includes(route.request().method()) && candidate.path.test(url.pathname));
  if (!rule) return false;
  const isJson = route.request().headers()["content-type"]?.includes("application/json");
  const body: unknown = isJson && route.request().postData() ? route.request().postDataJSON() : null;
  const payload = body && typeof body === "object" && !Array.isArray(body) ? body : {};
  const projectMatch = url.pathname.match(/^\/api\/projects\/(\d+)/);
  const payloadProjectId = "project_id" in payload ? Number(payload.project_id) : undefined;
  const projectId = projectMatch ? Number(projectMatch[1]) : Number(url.searchParams.get("project_id") ?? payloadProjectId ?? 1);
  const workspaceMatch = url.pathname.match(/^\/api\/workspaces\/(\d+)/);
  const workspaceId = workspaceMatch ? Number(workspaceMatch[1])
    : url.pathname === "/api/projects" || url.pathname === "/api/projects/my-work"
      ? Number(url.searchParams.get("workspace_id") ?? ("workspace_id" in payload ? payload.workspace_id : 1))
      : state.projects.get(projectId)?.workspaceId ?? 1;
  const workspace = state.workspaces.get(workspaceId);
  const minimumRole = url.pathname === "/api/feedback-runs" && route.request().method() === "POST" && "producer_type" in payload && payload.producer_type === "external_llm"
    ? "manager" : rule.role;
  const globalRoute = /^\/api\/(?:auth\/me|training\/recipes(?:\/[^/]+\/validate-configuration)?)$/.test(url.pathname);
  let detail: string | undefined;
  let status = 403;
  if ((projectMatch || payloadProjectId !== undefined || url.searchParams.has("project_id")) && !state.projects.has(projectId)) { detail = "Project not found"; status = 404; }
  else if (!globalRoute && !workspace) { detail = "Workspace not found"; status = 404; }
  else if (minimumRole && workspace && policy.role_rank[workspace.role] < policy.role_rank[minimumRole]) detail = "Insufficient workspace role";
  else if (rule.modules && !rule.modules.some((module) => modulesFor(state, projectId).effective.includes(module))) detail = `None of the required project modules are enabled or available: ${rule.modules.join(", ")}`;
  else if (rule.allModules?.some((module) => !modulesFor(state, projectId).effective.includes(module))) {
    detail = `None of the required project modules are enabled or available: ${rule.allModules.find((module) => !modulesFor(state, projectId).effective.includes(module))}`;
  } else if (rule.capability && !capabilitiesFor(state, workspaceId).effective.includes(rule.capability)) detail = `Capability '${rule.capability}' is not enabled for this workspace`;
  if (!detail) return false;
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify({ detail }) });
  return true;
}
export function pageByDescendingId<T>(items: readonly T[], url: URL, id: (item: T) => number, project: (item: T) => number) {
  const cursor = Number(url.searchParams.get("cursor") ?? Infinity);
  const limit = Number(url.searchParams.get("limit") ?? 50);
  const projectId = url.searchParams.get("project_id");
  const eligible = items.filter((item) => id(item) < cursor && (projectId === null || project(item) === Number(projectId))).sort((a, b) => id(b) - id(a));
  const page = eligible.slice(0, limit);
  const last = page.at(-1);
  return { items: page, next_cursor: eligible.length > limit && last ? id(last) : null };
}

export function hasMockAccessRule(route: Route): boolean {
  const pathname = new URL(route.request().url()).pathname;
  return rules.some((rule) => rule.methods.includes(route.request().method()) && rule.path.test(pathname));
}
