import { NativeSelect } from "@/components/NativeField";
import { useEffect, useState } from "react";

import type { MeMembership } from "@/api/client";
import { workspaceRoleLabel } from "@/lib/workspacePresentation";
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
  const role = workspaceRoleLabel(membership.role, membership.workspace_kind);
  return `${membership.workspace_name} · ${role}`;
}

export default function WorkspaceSwitcher({
  memberships,
  activeWorkspaceId,
  onWorkspaceChange,
  onManageWorkspaces,
  ariaLabel = "Workspace",
}: WorkspaceSwitcherProps): React.ReactElement | null {
  const [selection, setSelection] = useState<{ activeId: number | null; selectedId: number } | null>(null);
  const selectedId = selection?.activeId === activeWorkspaceId && memberships.some((membership) => membership.workspace_id === selection.selectedId)
    ? selection.selectedId
    : activeWorkspaceId;
  useEffect(() => { setSelection(null); }, [activeWorkspaceId]);
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
        <NativeSelect
          aria-label={ariaLabel}
          value={selectedId ?? activeWorkspaceId}
          disabled={memberships.length === 1}
          onChange={(event) => setSelection({ activeId: activeWorkspaceId, selectedId: Number(event.target.value) })}
        >
          {memberships.map((membership) => (
            <option key={membership.workspace_id} value={membership.workspace_id}>
              {workspaceMembershipLabel(membership)}
            </option>
          ))}
        </NativeSelect>
      </label>
      {selectedId !== null && selectedId !== activeWorkspaceId ? (
        <Button label="Switch workspace" variant="primary" size="sm" onClick={() => onWorkspaceChange(selectedId)} />
      ) : null}
      {onManageWorkspaces ? (
        <Button
          label="Add workspace"
          icon={<Plus size={16} aria-hidden="true" />}
          tooltip="Manage workspaces"
          isIconOnly
          variant="ghost"
          size="sm"
          onClick={onManageWorkspaces}
        />
      ) : null}
    </div>
  );
}
