import { useCallback, useEffect, useId, useRef, useState } from "react";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Check, RefreshCw, Trash2, X } from "lucide-react";

import {
  approveJoinRequest,
  createWorkspaceInvite,
  deleteWorkspaceMember,
  getWorkspaceCapabilities,
  listWorkspaceJoinRequests,
  listWorkspaceMembers,
  rejectJoinRequest,
  setWorkspaceCapability,
  updateWorkspaceMemberRole,
  type WorkspaceCapabilities,
} from "@/api/client";
import ConfirmDialog from "@/components/ConfirmDialog";
import type {
  WorkspaceJoinRequest,
  WorkspaceMember,
  WorkspaceRole,
} from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";

const ROLE_OPTIONS: Array<{ value: WorkspaceRole; label: string }> = [
  { value: "annotator", label: "Annotator" },
  { value: "trainer", label: "Model Trainer" },
  { value: "manager", label: "Manager" },
  { value: "admin", label: "Administrator" },
];

const CAPABILITY_PRESETS = [
  {
    value: "annotate",
    label: "Annotate",
    description: "Data and human annotation.",
  },
  {
    value: "train",
    label: "Train Models",
    description: "Training and Models without Annotation.",
  },
  {
    value: "annotate_train",
    label: "Annotate + Train",
    description: "Annotation, model training, inference, and export.",
  },
  {
    value: "annotate_train_al",
    label: "Annotate + Train + Active Learning",
    description: "Includes the planned active-learning capability.",
  },
  {
    value: "full",
    label: "Full",
    description: "Every capability supported by this deployment.",
  },
] as const;

function memberName(member: WorkspaceMember): string {
  return member.display_name.trim() || member.username;
}

export default function WorkspaceSettings({
  workspaceId,
  workspaceName,
  workspaceKind,
  currentUserId,
  onCapabilitiesChanged,
}: {
  workspaceId: number;
  workspaceName: string;
  workspaceKind: string;
  currentUserId: number | null;
  onCapabilitiesChanged: (capabilities: WorkspaceCapabilities) => void;
}): React.ReactElement {
  const inviteTokenId = useId();
  const [members, setMembers] = useState<WorkspaceMember[]>([]);
  const [joinRequests, setJoinRequests] = useState<WorkspaceJoinRequest[]>([]);
  const [capabilities, setCapabilities] =
    useState<WorkspaceCapabilities | null>(null);
  const [preset, setPreset] = useState("annotate");
  const [inviteRole, setInviteRole] = useState<WorkspaceRole>("annotator");
  const [inviteToken, setInviteToken] = useState<string | null>(null);
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
    setError(null);
    try {
      const [nextMembers, nextRequests, nextCapabilities] = await Promise.all([
        listWorkspaceMembers(workspaceId),
        workspaceKind === "team"
          ? listWorkspaceJoinRequests(workspaceId)
          : Promise.resolve([]),
        getWorkspaceCapabilities(workspaceId),
      ]);
      if (!isCurrentRequest()) return;
      setMembers(nextMembers);
      setJoinRequests(nextRequests);
      setCapabilities(nextCapabilities);
      setPreset(nextCapabilities.preset);
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
    setInviteToken(null);
    setMemberToRemove(null);
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
      setMembers((current) =>
        current.map((item) =>
          item.user_id === updated.user_id ? updated : item,
        ),
      );
    }, `${memberName(member)} is now ${ROLE_OPTIONS.find((item) => item.value === role)?.label}.`);
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
      const invite = await createWorkspaceInvite(workspaceId, inviteRole);
      if (!isCurrentRequest()) return;
      setInviteToken(invite.token);
    }, "Invitation created.");
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
            <Button
              label="Refresh"
              icon={<RefreshCw size={17} />}
              isIconOnly
              variant="ghost"
              isDisabled={loading || busy}
              onClick={() => void load()}
            />
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
          <div className="platform-loading" role="status" aria-live="polite">
            <span aria-hidden="true" />
            Loading workspace settings…
          </div>
        ) : (
          <>
            <PlatformSection
              title="Capabilities"
              description="Capabilities determine which released modules and infrastructure are available. Individual ownership does not bypass this policy."
            >
              <div className="workspace-capability-settings">
                <label>
                  <span>Capability preset</span>
                  <select
                    value={preset}
                    disabled={busy}
                    onChange={(event) => setPreset(event.target.value)}
                  >
                    {CAPABILITY_PRESETS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <small>
                    {CAPABILITY_PRESETS.find((item) => item.value === preset)
                      ?.description ?? "Custom workspace capability policy."}
                  </small>
                </label>
                <Button
                  label={busy ? "Saving…" : "Save capabilities"}
                  variant="primary"
                  isDisabled={busy || preset === capabilities?.preset}
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
                      <PlatformStatus value={capability} />
                    </span>
                  ))}
                  {Object.entries(capabilities.blocked).map(
                    ([capability, reason]) => (
                      <span key={capability} title={reason}>
                        <PlatformStatus value={`${capability} unavailable`} />
                      </span>
                    ),
                  )}
                </div>
              ) : null}
            </PlatformSection>

            <PlatformSection
              title="Members"
              description="Roles are hierarchical; each higher role inherits the permissions below it."
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
                        <th scope="col">
                          <span className="sr-only">Actions</span>
                        </th>
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
                              <select
                                aria-label={`Role for ${memberName(member)}`}
                                value={member.role}
                                disabled={busy || isCurrentUser}
                                onChange={(event) =>
                                  void changeRole(
                                    member,
                                    event.target.value as WorkspaceRole,
                                  )
                                }
                              >
                                {ROLE_OPTIONS.map((option) => (
                                  <option
                                    key={option.value}
                                    value={option.value}
                                  >
                                    {option.label}
                                  </option>
                                ))}
                              </select>
                            </td>
                            <td data-label="Action" data-priority="action">
                              <Button
                                label={`Remove ${memberName(member)}`}
                                icon={<Trash2 size={16} />}
                                isIconOnly
                                variant="ghost"
                                isDisabled={busy || isCurrentUser}
                                onClick={() => setMemberToRemove(member)}
                              />
                            </td>
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
              <div className="workspace-collaboration-grid">
                <PlatformSection
                  title="Invite"
                  description="Create a role-bounded invitation for this workspace."
                >
                  <form
                    className="workspace-invite-form"
                    onSubmit={(event) => {
                      event.preventDefault();
                      void createInvite();
                    }}
                  >
                    <label>
                      <span>Initial role</span>
                      <select
                        value={inviteRole}
                        disabled={busy}
                        onChange={(event) =>
                          setInviteRole(event.target.value as WorkspaceRole)
                        }
                      >
                        {ROLE_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <Button
                      label="Create invitation"
                      variant="primary"
                      type="submit"
                      isDisabled={busy}
                    />
                  </form>
                  {inviteToken ? (
                    <div
                      id={inviteTokenId}
                      className="workspace-invite-token"
                      role="status"
                    >
                      <span>Invitation token</span>
                      <code>{inviteToken}</code>
                    </div>
                  ) : null}
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
                            <strong>User {request.user_id}</strong>
                            <span>{request.message ?? "No message provided"}</span>
                          </div>
                          <div>
                            <Button
                              label={`Approve request from user ${request.user_id}`}
                              icon={<Check size={16} />}
                              isIconOnly
                              variant="ghost"
                              isDisabled={busy}
                              onClick={() => void decideRequest(request, true)}
                            />
                            <Button
                              label={`Reject request from user ${request.user_id}`}
                              icon={<X size={16} />}
                              isIconOnly
                              variant="ghost"
                              isDisabled={busy}
                              onClick={() => void decideRequest(request, false)}
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
        onCancel={() => setMemberToRemove(null)}
        onConfirm={removeMember}
      />
    </main>
  );
}
