import { NativeSelect } from "@/components/NativeField";
import ValidatedForm from "@/components/ValidatedForm";
import LoadingState from "@/components/LoadingState";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Check, Copy, RefreshCw, RotateCw, Trash2, UserRoundPlus, UserRoundX } from "lucide-react";

import {
  approveJoinRequest,
  createWorkspaceInvite,
  deleteWorkspaceMember,
  getWorkspaceCapabilities,
  getWorkspaceGovernance,
  listWorkspaceInvites,
  listWorkspaceJoinRequests,
  listWorkspaceMembers,
  rejectJoinRequest,
  revokeWorkspaceInvite,
  rotateWorkspaceJoinCode,
  setWorkspaceCapability,
  updateWorkspaceMemberRole,
  type WorkspaceCapabilities,
} from "@/api/client";
import ConfirmDialog from "@/components/ConfirmDialog";
import {
  WORKSPACE_PRESET_LABELS,
  WORKSPACE_ROLE_OPTIONS as ROLE_OPTIONS,
  workspaceRoleLabel,
} from "@/lib/workspacePresentation";
import type {
  WorkspaceJoinRequest,
  WorkspaceGovernance,
  WorkspaceInviteSummary,
  WorkspaceMember,
  WorkspaceRole,
} from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
  formatCount,
  formatDateTime,
  formatStatus,
} from "./components";

const CAPABILITY_PRESETS = [
  {
    value: "annotate",
    description: "Data and human annotation.",
  },
  {
    value: "train",
    description: "Model training and the model registry without annotation.",
  },
  {
    value: "annotate_train",
    description: "Annotation, model training, predictions, and export.",
  },
  {
    value: "annotate_train_al",
    description: "Includes the planned active-learning capability.",
  },
  {
    value: "full",
    description: "Every capability supported by this deployment.",
  },
] as const;

function memberName(member: WorkspaceMember): string {
  return member.display_name.trim() || member.username;
}

function formatExpiryMinutes(minutes: number): string {
  if (minutes % 1_440 === 0) {
    const days = minutes / 1_440;
    return formatCount(days, "day");
  }
  if (minutes % 60 === 0) {
    const hours = minutes / 60;
    return formatCount(hours, "hour");
  }
  return formatCount(minutes, "minute");
}

export default function WorkspaceSettings({
  workspaceId,
  workspaceName,
  workspaceKind,
  currentUserId,
  onCapabilitiesChanged,
  onManageWorkspaces,
}: {
  workspaceId: number;
  workspaceName: string;
  workspaceKind: string;
  currentUserId: number | null;
  onCapabilitiesChanged: (capabilities: WorkspaceCapabilities) => void;
  onManageWorkspaces: () => void;
}): React.ReactElement {
  const inviteTokenId = useId();
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [joinRequests, setJoinRequests] = useState<WorkspaceJoinRequest[]>([]);
  const [governance, setGovernance] = useState<WorkspaceGovernance | null>(null);
  const [invites, setInvites] = useState<WorkspaceInviteSummary[]>([]);
  const [capabilities, setCapabilities] =
    useState<WorkspaceCapabilities | null>(null);
  const [preset, setPreset] = useState<string | null>(null);
  const [inviteRole, setInviteRole] = useState<WorkspaceRole>("annotator");
  const [inviteExpiry, setInviteExpiry] = useState("default");
  const [createdInvite, setCreatedInvite] = useState<{
    id: number;
    token: string;
  } | null>(null);
  const [inviteToRevoke, setInviteToRevoke] =
    useState<WorkspaceInviteSummary | null>(null);
  const [rotateJoinCode, setRotateJoinCode] = useState(false);
  const [roleToChange, setRoleToChange] = useState<{ member: WorkspaceMember; role: WorkspaceRole } | null>(null);
  const [requestToReject, setRequestToReject] = useState<WorkspaceJoinRequest | null>(null);
  const [memberToRemove, setMemberToRemove] =
    useState<WorkspaceMember | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);
  const mutationRequestIdRef = useRef(0);
  const contextKey = `${workspaceId}:${workspaceKind}`;
  const contextKeyRef = useRef(contextKey);
  contextKeyRef.current = contextKey;

  const load = useCallback(async (): Promise<void> => {
    if (contextKeyRef.current !== contextKey) return;
    const requestId = ++loadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === loadRequestIdRef.current &&
      contextKey === contextKeyRef.current;
    setLoading(true);
    setCapabilities(null);
    setPreset(null);
    setError(null);
    try {
      const [nextMembers, nextRequests, nextCapabilities, nextGovernance, nextInvites] = await Promise.all([
        listWorkspaceMembers(workspaceId),
        workspaceKind === "team"
          ? listWorkspaceJoinRequests(workspaceId)
          : Promise.resolve([]),
        getWorkspaceCapabilities(workspaceId),
        workspaceKind === "team"
          ? getWorkspaceGovernance(workspaceId)
          : Promise.resolve(null),
        workspaceKind === "team"
          ? listWorkspaceInvites(workspaceId)
          : Promise.resolve([]),
      ]);
      if (!isCurrentRequest()) return;
      setMembers(nextMembers);
      setJoinRequests(nextRequests);
      setCapabilities(nextCapabilities);
      setPreset(nextCapabilities.preset);
      setGovernance(nextGovernance);
      setInvites(nextInvites);
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Workspace settings could not be loaded.",
        );
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false);
      }
    }
  }, [contextKey, workspaceId, workspaceKind]);

  useEffect(() => {
    loadRequestIdRef.current += 1;
    mutationRequestIdRef.current += 1;
    setMembers([]);
    setJoinRequests([]);
    setCapabilities(null);
    setGovernance(null);
    setInvites([]);
    setCreatedInvite(null);
    setInviteToRevoke(null);
    setRotateJoinCode(false);
    setMemberToRemove(null);
    setRoleToChange(null);
    setRequestToReject(null);
    setStatus(null);
    setBusy(false);
    void load();
    return () => {
      loadRequestIdRef.current += 1;
      mutationRequestIdRef.current += 1;
    };
  }, [load]);

  async function run(
    action: (isCurrentRequest: () => boolean) => Promise<void>,
    successMessage: string,
  ): Promise<void> {
    if (contextKeyRef.current !== contextKey) return;
    const requestId = ++mutationRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === mutationRequestIdRef.current &&
      contextKey === contextKeyRef.current;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      await action(isCurrentRequest);
      if (isCurrentRequest()) {
        setStatus(successMessage);
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Workspace settings could not be updated.",
        );
      }
    } finally {
      if (isCurrentRequest()) {
        setBusy(false);
      }
    }
  }

  async function saveCapabilities(): Promise<void> {
    if (loading || busy || capabilities === null || preset === null || preset === "custom") return;
    await run(async (isCurrentRequest) => {
      const updated = await setWorkspaceCapability(workspaceId, preset);
      if (!isCurrentRequest()) return;
      setCapabilities(updated);
      onCapabilitiesChanged(updated);
    }, "Workspace capabilities saved.");
  }

  async function changeRole(
    member: WorkspaceMember,
    role: WorkspaceRole,
  ): Promise<void> {
    await run(async (isCurrentRequest) => {
      const updated = await updateWorkspaceMemberRole(
        workspaceId,
        member.user_id,
        role,
      );
      if (!isCurrentRequest()) return;
      setRoleToChange(null);
      setMembers((current) =>
        current.map((item) =>
          item.user_id === updated.user_id ? updated : item,
        ),
      );
    }, `${memberName(member)} is now ${workspaceRoleLabel(role, workspaceKind)}.`);
  }

  async function removeMember(): Promise<void> {
    if (!memberToRemove) {
      return;
    }
    const member = memberToRemove;
    await run(async (isCurrentRequest) => {
      await deleteWorkspaceMember(workspaceId, member.user_id);
      if (!isCurrentRequest()) return;
      setMembers((current) =>
        current.filter((item) => item.user_id !== member.user_id),
      );
      setMemberToRemove(null);
    }, `${memberName(member)} was removed from the workspace.`);
  }

  async function createInvite(): Promise<void> {
    await run(async (isCurrentRequest) => {
      const expiresMinutes =
        inviteExpiry === "default" ? undefined : Number(inviteExpiry);
      const invite = await createWorkspaceInvite(
        workspaceId,
        inviteRole,
        expiresMinutes,
      );
      if (!isCurrentRequest()) return;
      setCreatedInvite({ id: invite.id, token: invite.token });
      const nextInvites = await listWorkspaceInvites(workspaceId);
      if (!isCurrentRequest()) return;
      setInvites(nextInvites);
    }, "Invitation created.");
  }

  async function revokeInvite(): Promise<void> {
    if (!inviteToRevoke) return;
    const invite = inviteToRevoke;
    await run(async (isCurrentRequest) => {
      await revokeWorkspaceInvite(workspaceId, invite.id);
      if (!isCurrentRequest()) return;
      setInvites((current) => current.filter((item) => item.id !== invite.id));
      setCreatedInvite((current) =>
        current?.id === invite.id ? null : current,
      );
      setInviteToRevoke(null);
    }, "Invitation revoked.");
  }

  async function rotateCode(): Promise<void> {
    await run(async (isCurrentRequest) => {
      const nextGovernance = await rotateWorkspaceJoinCode(workspaceId);
      if (!isCurrentRequest()) return;
      setGovernance(nextGovernance);
      setRotateJoinCode(false);
    }, "Join code rotated. The previous code no longer works.");
  }

  async function copyText(value: string, successMessage: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(value);
      setStatus(successMessage);
      setError(null);
    } catch {
      setError("Copy failed. Select the value and copy it manually.");
    }
  }

  async function decideRequest(
    request: WorkspaceJoinRequest,
    approve: boolean,
  ): Promise<void> {
    await run(async (isCurrentRequest) => {
      if (approve) {
        await approveJoinRequest(request.id);
      } else {
        await rejectJoinRequest(request.id);
      }
      const [nextMembers, nextRequests] = await Promise.all([
        listWorkspaceMembers(workspaceId),
        listWorkspaceJoinRequests(workspaceId),
      ]);
      if (!isCurrentRequest()) return;
      setMembers(nextMembers);
      setJoinRequests(nextRequests);
      setRequestToReject(null);
    }, approve ? "Join request approved." : "Join request rejected.");
  }

  return (
    <main id="main-content" className="module-workspace-main" tabIndex={-1}>
      <div className="platform-page workspace-settings-page">
        <PlatformPageHeader
          title="Workspace settings"
          description={`${workspaceName} · ${
            workspaceKind === "individual" ? "Individual workspace" : "Team workspace"
          }`}
          secondary={
            <div className="workspace-settings-header-actions">
              {workspaceKind === "individual" ? (
                <Button
                  label="Create or join a team"
                  icon={<UserRoundPlus size={17} aria-hidden="true" />}
                  variant="ghost"
                  isDisabled={loading || busy}
                  onClick={onManageWorkspaces}
                />
              ) : null}
              <Button
                label="Refresh" tooltip="Refresh"
                icon={<RefreshCw size={17} />}
                isIconOnly
                variant="ghost"
                isDisabled={loading || busy}
                onClick={() => void load()}
              />
            </div>
          }
        />

        {error ? (
          <Banner
            status="error"
            title="Workspace update failed"
            description={error}
            container="section"
          />
        ) : null}
        {status ? (
          <p className="platform-form-success" role="status">
            {status}
          </p>
        ) : null}

        {loading ? (
          <LoadingState label="Loading workspace settings…" />
        ) : (
          <>
            <PlatformSection
              title="Capabilities"
              description="Capabilities determine which released modules and infrastructure are available. Individual ownership does not bypass this policy."
            >
              <div className="workspace-capability-settings">
                <label>
                  <span>Capability preset</span>
                  <NativeSelect
                    value={preset ?? ""}
                    disabled={loading || busy || capabilities === null}
                    onChange={(event) => setPreset(event.target.value)}
                  >
                    {preset === null ? <option value="">Capabilities unavailable</option> : null}
                    {capabilities?.preset === "custom" ? <option value="custom" disabled>Custom</option> : null}
                    {CAPABILITY_PRESETS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {WORKSPACE_PRESET_LABELS[option.value]}
                      </option>
                    ))}
                  </NativeSelect>
                  <small>
                    {CAPABILITY_PRESETS.find((item) => item.value === preset)
                      ?.description ?? "Custom workspace capability policy."}
                  </small>
                </label>
                <Button
                  label={busy ? "Saving…" : "Save capabilities"}
                  variant="primary"
                  isDisabled={loading || busy || capabilities === null || preset === null || preset === capabilities.preset}
                  onClick={() => void saveCapabilities()}
                />
              </div>
              {capabilities ? (
                <div
                  className="workspace-capability-list"
                  aria-label="Effective capabilities"
                >
                  {capabilities.effective.map((capability) => (
                    <span key={capability}>
                      {formatStatus(capability)}
                    </span>
                  ))}
                  {Object.entries(capabilities.blocked).map(
                    ([capability, reason]) => (
                      <span key={capability}>
                        <span>{formatStatus(capability)} unavailable</span>
                        <span className="platform-footnote">{reason}</span>
                      </span>
                    ),
                  )}
                </div>
              ) : null}
            </PlatformSection>

            <PlatformSection
              title={workspaceKind === "individual" ? "Owner" : "Members"}
              description={
                workspaceKind === "individual"
                  ? "An individual workspace is permanently single-owner. Create or join a team to collaborate."
                  : "Roles are hierarchical; each higher role inherits the permissions below it."
              }
            >
              {members.length ? (
                <div
                  className="platform-table-scroll platform-table-scroll--summary"
                  role="region"
                  aria-label="Workspace members table"
                  tabIndex={0}
                >
                  <table className="platform-table platform-table--summary">
                    <thead>
                      <tr>
                        <th scope="col">Member</th>
                        <th scope="col">Email</th>
                        <th scope="col">Status</th>
                        <th scope="col">Effective role</th>
                        {workspaceKind === "team" ? (
                          <th scope="col">
                            <span className="sr-only">Actions</span>
                          </th>
                        ) : null}
                      </tr>
                    </thead>
                    <tbody>
                      {members.map((member) => {
                        const isCurrentUser = member.user_id === currentUserId;
                        return (
                          <tr key={member.user_id}>
                            <td data-label="Member" data-priority="identity">
                              <strong>{memberName(member)}</strong>
                              <span>{member.username}</span>
                            </td>
                            <td data-label="Email">{member.email ?? "Not set"}</td>
                            <td data-label="Status" data-priority="status">
                              <PlatformStatus
                                value={member.is_active ? "active" : "inactive"}
                              />
                            </td>
                            <td data-label="Effective role">
                              {workspaceKind === "individual" ? (
                                <strong>{workspaceRoleLabel(member.role, workspaceKind)}</strong>
                              ) : (
                                <NativeSelect
                                  aria-label={`Role for ${memberName(member)}`}
                                  value={member.role}
                                  disabled={busy || isCurrentUser}
                                  onChange={(event) => {
                                    const option = ROLE_OPTIONS.find((item) => item.value === event.target.value);
                                    if (option) { setError(null); setRoleToChange({ member, role: option.value }); }
                                  }}
                                >
                                  {ROLE_OPTIONS.map((option) => (
                                    <option
                                      key={option.value}
                                      value={option.value}
                                    >
                                      {option.label}
                                    </option>
                                  ))}
                                </NativeSelect>
                              )}
                            </td>
                            {workspaceKind === "team" ? (
                              <td data-label="Action" data-priority="action">
                                <Button
                                  label={`Remove ${memberName(member)}`} tooltip={`Remove ${memberName(member)}`}
                                  icon={<Trash2 size={16} />}
                                  isIconOnly
                                  variant="ghost"
                                  isDisabled={busy || isCurrentUser}
                                  onClick={() => setMemberToRemove(member)}
                                />
                              </td>
                            ) : null}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <PlatformEmpty
                  title="No workspace members"
                  detail="Members appear after an invitation is accepted or a join request is approved."
                />
              )}
            </PlatformSection>

            {workspaceKind === "team" ? (
              <div className="workspace-team-settings">
                <PlatformSection
                  title="Team access"
                  description="Share the join code for an approval-based request. Rotating it immediately invalidates the previous code."
                  action={
                    <Button
                      label={governance?.join_code ? "Rotate join code" : "Generate join code"}
                      icon={<RotateCw size={16} aria-hidden="true" />}
                      variant="ghost"
                      isDisabled={busy || governance === null}
                      onClick={() => setRotateJoinCode(true)}
                    />
                  }
                >
                  {governance?.join_code ? (
                    <div className="workspace-share-value">
                      <code>{governance.join_code}</code>
                      <Button
                        label="Copy join code" tooltip="Copy join code"
                        icon={<Copy size={16} aria-hidden="true" />}
                        isIconOnly
                        variant="ghost"
                        onClick={() =>
                          void copyText(governance.join_code ?? "", "Join code copied.")
                        }
                      />
                    </div>
                  ) : (
                    <p className="platform-inline-empty">No join code is available.</p>
                  )}
                </PlatformSection>

                <PlatformSection
                  title="Invitations"
                  description="Create a single-use, role-bounded link or revoke an unused link."
                >
                  <ValidatedForm
                    className="workspace-invite-form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void createInvite();
                    }}
                  >
                    <label>
                      <span>Initial role</span>
                      <NativeSelect
                        value={inviteRole}
                        disabled={busy}
                        onChange={(event) => {
                          const option = ROLE_OPTIONS.find((item) => item.value === event.target.value);
                          if (option) setInviteRole(option.value);
                        }}
                      >
                        {ROLE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </NativeSelect>
                    </label>
                    <label>
                      <span>Expires</span>
                      <NativeSelect
                        value={inviteExpiry}
                        disabled={busy}
                        onChange={(event) => setInviteExpiry(event.target.value)}
                      >
                        <option value="default">
                          Instance default
                          {governance
                            ? ` (${formatExpiryMinutes(governance.default_invite_expiry_minutes)})`
                            : ""}
                        </option>
                        <option value="1440">24 hours</option>
                        <option value="10080">7 days</option>
                        <option value="43200">30 days</option>
                      </NativeSelect>
                    </label>
                    <Button
                      label="Create invitation"
                      variant="primary"
                      type="submit"
                      isDisabled={busy}
                    />
                  </ValidatedForm>
                  {createdInvite ? (
                    <div
                      id={inviteTokenId}
                      className="workspace-invite-token"
                      role="status"
                    >
                      <span>Invitation link</span>
                      <div className="workspace-share-value">
                        <code>{`${window.location.origin}/invites#token=${encodeURIComponent(createdInvite.token)}`}</code>
                        <Button
                          label="Copy invitation link" tooltip="Copy invitation link"
                          icon={<Copy size={16} aria-hidden="true" />}
                          isIconOnly
                          variant="ghost"
                          onClick={() =>
                            void copyText(
                              `${window.location.origin}/invites#token=${encodeURIComponent(createdInvite.token)}`,
                              "Invitation link copied.",
                            )
                          }
                        />
                      </div>
                    </div>
                  ) : null}

                  {invites.length ? (
                    <div
                      className="platform-table-scroll platform-table-scroll--summary"
                      role="region"
                      aria-label="Open invitations table"
                      tabIndex={0}
                    >
                      <table className="platform-table platform-table--summary">
                        <thead>
                          <tr>
                            <th scope="col">Role</th>
                            <th scope="col">Created by</th>
                            <th scope="col">Expires</th>
                            <th scope="col"><span className="sr-only">Actions</span></th>
                          </tr>
                        </thead>
                        <tbody>
                          {invites.map((invite) => (
                            <tr key={invite.id}>
                              <td data-label="Role">{workspaceRoleLabel(invite.role)}</td>
                              <td data-label="Created by">{invite.created_by_username}</td>
                              <td data-label="Expires">
                                {formatDateTime(invite.expires_at, "No expiry")}
                              </td>
                              <td data-label="Action" data-priority="action">
                                <Button
                                  label={`Revoke ${workspaceRoleLabel(invite.role).toLowerCase()} invitation`} tooltip={`Revoke ${workspaceRoleLabel(invite.role).toLowerCase()} invitation`}
                                  icon={<Trash2 size={16} aria-hidden="true" />}
                                  isIconOnly
                                  variant="ghost"
                                  isDisabled={busy}
                                  onClick={() => setInviteToRevoke(invite)}
                                />
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <p className="platform-inline-empty">No open invitations.</p>
                  )}
                </PlatformSection>

                <PlatformSection
                  title="Join requests"
                  description="Approve only users who should enter this workspace."
                >
                  {joinRequests.length ? (
                    <ul className="workspace-request-list">
                      {joinRequests.map((request) => (
                        <li key={request.id}>
                          <div>
                            <strong>{request.display_name.trim() || request.username}</strong>
                            <span>
                              {request.username}
                              {request.email ? ` · ${request.email}` : ""}
                            </span>
                            <span>{request.message ?? "No message provided"}</span>
                          </div>
                          <div>
                            <Button
                              label={`Approve request from ${request.username}`} tooltip={`Approve request from ${request.username}`}
                              icon={<Check size={16} />}
                              isIconOnly
                              variant="ghost"
                              isDisabled={busy}
                              onClick={() => void decideRequest(request, true)}
                            />
                            <Button
                              label={`Reject request from ${request.username}`} tooltip={`Reject request from ${request.username}`}
                              icon={<UserRoundX size={16} aria-hidden="true" />}
                              isIconOnly
                              variant="ghost"
                              isDisabled={busy}
                              onClick={() => { setError(null); setRequestToReject(request); }}
                            />
                          </div>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="platform-inline-empty">
                      No pending join requests.
                    </p>
                  )}
                </PlatformSection>
              </div>
            ) : null}
          </>
        )}
      </div>

      <ConfirmDialog open={roleToChange !== null} title="Change member role?" description={roleToChange ? `${memberName(roleToChange.member)} will become ${workspaceRoleLabel(roleToChange.role, workspaceKind).toLowerCase()}. Their workspace permissions change immediately.` : ""} confirmLabel="Change role" busy={busy} error={error} onCancel={() => { setRoleToChange(null); setError(null); }} onConfirm={async () => { if (roleToChange) await changeRole(roleToChange.member, roleToChange.role); }} />
      <ConfirmDialog open={requestToReject !== null} title="Reject join request?" description={requestToReject ? `${requestToReject.username} will not join this workspace. They must send another request to be considered again.` : ""} confirmLabel="Reject request" busy={busy} error={error} onCancel={() => { setRequestToReject(null); setError(null); }} onConfirm={async () => { if (requestToReject) await decideRequest(requestToReject, false); }} />
      <ConfirmDialog
        open={memberToRemove !== null}
        title="Remove workspace member?"
        description={
          memberToRemove
            ? `${memberName(memberToRemove)} will lose workspace access. Mutable assignments are withdrawn while completed work remains attributed.`
            : ""
        }
        confirmLabel="Remove member"
        busy={busy}
        error={error}
        onCancel={() => setMemberToRemove(null)}
        onConfirm={removeMember}
      />
      <ConfirmDialog
        open={inviteToRevoke !== null}
        title="Revoke invitation?"
        description="This invitation link will stop working immediately. Existing members are not affected."
        confirmLabel="Revoke invitation"
        busy={busy}
        error={error}
        onCancel={() => setInviteToRevoke(null)}
        onConfirm={revokeInvite}
      />
      <ConfirmDialog
        open={rotateJoinCode}
        title="Rotate the join code?"
        description="The current code will stop accepting new requests. Pending requests are not affected."
        confirmLabel="Rotate code"
        busy={busy}
        error={error}
        onCancel={() => setRotateJoinCode(false)}
        onConfirm={rotateCode}
      />
    </main>
  );
}
