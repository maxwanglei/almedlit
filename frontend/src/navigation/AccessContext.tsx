import {
  createContext,
  useContext,
  type ReactNode,
} from "react";

import type { WorkspaceRole } from "@/types/api";

const ROLE_ORDER = [
  "annotator",
  "trainer",
  "manager",
  "admin",
] as const satisfies readonly WorkspaceRole[];

export type AccessCommand =
  | "annotation:work"
  | "projects:read"
  | "projects:create"
  | "tasks:manage"
  | "rounds:manage"
  | "quality:review"
  | "training:read"
  | "training:launch"
  | "training:cancel-own"
  | "recipes:author"
  | "models:read"
  | "runtime:provision"
  | "storage:provision"
  | "workspace:manage"
  | "system:admin";

export interface AccessSnapshot {
  workspaceId: number | null;
  workspaceKind: string | null;
  membershipRole: WorkspaceRole | null;
  effectiveRoles: readonly WorkspaceRole[];
  effectiveCapabilities: readonly string[];
  blockedCapabilities: Readonly<Record<string, string>>;
  isWorkspaceOwner: boolean;
  isSuperuser: boolean;
  ready: boolean;
}

export interface AccessSnapshotInput {
  workspaceId: number | null;
  workspaceKind: string | null | undefined;
  membershipRole: string | null | undefined;
  effectiveCapabilities: readonly string[];
  blockedCapabilities?: Readonly<Record<string, string>>;
  isSuperuser?: boolean;
  ready?: boolean;
}

function isWorkspaceRole(value: string | null | undefined): value is WorkspaceRole {
  return ROLE_ORDER.includes(value as WorkspaceRole);
}

export function expandEffectiveRoles(
  role: string | null | undefined,
  workspaceKind: string | null | undefined,
): readonly WorkspaceRole[] {
  if (workspaceKind === "individual") {
    return ROLE_ORDER;
  }
  if (!isWorkspaceRole(role)) {
    return [];
  }
  return ROLE_ORDER.slice(0, ROLE_ORDER.indexOf(role) + 1);
}

export function createAccessSnapshot(
  input: AccessSnapshotInput,
): AccessSnapshot {
  const membershipRole = isWorkspaceRole(input.membershipRole)
    ? input.membershipRole
    : null;
  const workspaceKind = input.workspaceKind ?? null;
  return {
    workspaceId: input.workspaceId,
    workspaceKind,
    membershipRole,
    effectiveRoles: expandEffectiveRoles(membershipRole, workspaceKind),
    effectiveCapabilities: input.effectiveCapabilities,
    blockedCapabilities: input.blockedCapabilities ?? {},
    isWorkspaceOwner: workspaceKind === "individual",
    isSuperuser: input.isSuperuser ?? false,
    ready: input.ready ?? true,
  };
}

export function hasEffectiveRole(
  access: AccessSnapshot,
  role: WorkspaceRole,
): boolean {
  return access.effectiveRoles.includes(role);
}

export function hasEffectiveCapability(
  access: AccessSnapshot,
  capability: string,
): boolean {
  return (
    access.effectiveCapabilities.includes(capability) &&
    access.blockedCapabilities[capability] === undefined
  );
}

export function canPerform(
  access: AccessSnapshot,
  command: AccessCommand,
): boolean {
  switch (command) {
    case "annotation:work":
      return (
        hasEffectiveRole(access, "annotator") &&
        hasEffectiveCapability(access, "annotation")
      );
    case "projects:read":
      return hasEffectiveRole(access, "trainer");
    case "projects:create":
    case "tasks:manage":
    case "rounds:manage":
    case "quality:review":
    case "recipes:author":
      return hasEffectiveRole(access, "manager");
    case "training:read":
    case "training:launch":
    case "training:cancel-own":
    case "models:read":
      return (
        hasEffectiveRole(access, "trainer") &&
        hasEffectiveCapability(access, "training")
      );
    case "runtime:provision":
    case "storage:provision":
    case "workspace:manage":
      return hasEffectiveRole(access, "admin");
    case "system:admin":
      return access.isSuperuser;
  }
}

export function effectiveRoleLabel(access: AccessSnapshot): string {
  if (access.isWorkspaceOwner) {
    return "Owner";
  }
  switch (access.membershipRole) {
    case "trainer":
      return "Model Trainer";
    case "manager":
      return "Manager";
    case "admin":
      return "Administrator";
    case "annotator":
      return "Annotator";
    default:
      return "Member";
  }
}

const AccessContext = createContext<AccessSnapshot | null>(null);

interface AccessProviderProps {
  value: AccessSnapshot;
  children: ReactNode;
}

export function AccessProvider({
  value,
  children,
}: AccessProviderProps): React.ReactElement {
  return (
    <AccessContext.Provider value={value}>
      {children}
    </AccessContext.Provider>
  );
}

export function useAccess(): AccessSnapshot {
  const access = useContext(AccessContext);
  if (access === null) {
    throw new Error("useAccess must be used within AccessProvider");
  }
  return access;
}
