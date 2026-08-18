import type { MeMembership } from "@/api/client";
import { Button } from "@astryxdesign/core/Button";
import { Plus } from "lucide-react";

interface WorkspaceSwitcherProps {
  memberships: MeMembership[];
  activeWorkspaceId: number | null;
  onWorkspaceChange: (workspaceId: number) => void;
  onManageWorkspaces?: () => void;
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
  onManageWorkspaces,
  ariaLabel = "Workspace",
}: WorkspaceSwitcherProps): React.ReactElement | null {
  if (memberships.length === 0 || activeWorkspaceId === null) {
    return onManageWorkspaces ? (
      <Button
        label="Add workspace"
        icon={<Plus size={16} aria-hidden="true" />}
        variant="ghost"
        size="sm"
        onClick={onManageWorkspaces}
      />
    ) : null;
  }

  return (
    <div className="workspace-switcher-group">
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
      {onManageWorkspaces ? (
        <Button
          label="Add workspace"
          icon={<Plus size={16} aria-hidden="true" />}
          isIconOnly
          variant="ghost"
          size="sm"
          onClick={onManageWorkspaces}
        />
      ) : null}
    </div>
  );
}
