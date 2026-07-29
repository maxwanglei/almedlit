import type { MeMembership } from "@/api/client";

const ACTIVE_WORKSPACE_STORAGE_PREFIX = "al_medlit_active_workspace:";

export function preferredMembership(
  memberships: MeMembership[],
  preferredWorkspaceId: number | null,
): MeMembership | null {
  if (preferredWorkspaceId !== null) {
    const preferred = memberships.find(
      (membership) => membership.workspace_id === preferredWorkspaceId,
    );
    if (preferred) {
      return preferred;
    }
  }
  return memberships[0] ?? null;
}

export function readPreferredWorkspaceId(userId: number): number | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const value = window.localStorage.getItem(`${ACTIVE_WORKSPACE_STORAGE_PREFIX}${userId}`);
    if (value === null) {
      return null;
    }
    const workspaceId = Number(value);
    return Number.isInteger(workspaceId) && workspaceId > 0 ? workspaceId : null;
  } catch {
    return null;
  }
}

export function storePreferredWorkspaceId(userId: number, workspaceId: number): void {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(
      `${ACTIVE_WORKSPACE_STORAGE_PREFIX}${userId}`,
      String(workspaceId),
    );
  } catch {
    // Workspace switching still works for this session when storage is unavailable.
  }
}
