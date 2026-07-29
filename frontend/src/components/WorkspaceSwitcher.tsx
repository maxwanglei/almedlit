import type { MeMembership } from "@/api/client";

interface WorkspaceSwitcherProps {
  memberships: MeMembership[];
  activeWorkspaceId: number | null;
  onWorkspaceChange: (workspaceId: number) => void;
  ariaLabel?: string;
}

export function workspaceMembershipLabel(membership: MeMembership): string {
  const role =
    membership.workspace_kind === "individual"
      ? "Owner"
      : membership.role === "trainer"
        ? "Model Trainer"
        : membership.role === "admin"
          ? "Administrator"
          : membership.role === "manager"
            ? "Manager"
            : "Annotator";
  return `${membership.workspace_name} · ${role}`;
}

export default function WorkspaceSwitcher({
  memberships,
  activeWorkspaceId,
  onWorkspaceChange,
  ariaLabel = "Workspace",
}: WorkspaceSwitcherProps): React.ReactElement | null {
  if (memberships.length === 0 || activeWorkspaceId === null) {
    return null;
  }

  return (
    <label className="workspace-switcher">
      <span>Workspace</span>
      <select
        aria-label={ariaLabel}
        value={activeWorkspaceId}
        disabled={memberships.length === 1}
        onChange={(event) => onWorkspaceChange(Number(event.target.value))}
      >
        {memberships.map((membership) => (
          <option key={membership.workspace_id} value={membership.workspace_id}>
            {workspaceMembershipLabel(membership)}
          </option>
        ))}
      </select>
    </label>
  );
}
