// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  createWorkspaceInvite,
  getWorkspaceCapabilities,
  getWorkspaceGovernance,
  listWorkspaceInvites,
  listWorkspaceJoinRequests,
  listWorkspaceMembers,
  revokeWorkspaceInvite,
  setWorkspaceCapability,
} from "@/api/client";

import WorkspaceSettings from "./WorkspaceSettings";

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

vi.mock("@/api/client", () => ({
  approveJoinRequest: vi.fn(),
  createWorkspaceInvite: vi.fn(),
  deleteWorkspaceMember: vi.fn(),
  getWorkspaceCapabilities: vi.fn(),
  getWorkspaceGovernance: vi.fn(),
  listWorkspaceInvites: vi.fn(),
  listWorkspaceJoinRequests: vi.fn(),
  listWorkspaceMembers: vi.fn(),
  rejectJoinRequest: vi.fn(),
  revokeWorkspaceInvite: vi.fn(),
  rotateWorkspaceJoinCode: vi.fn(),
  setWorkspaceCapability: vi.fn(),
  updateWorkspaceMemberRole: vi.fn(),
}));

beforeEach(() => {
  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    value: (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    },
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    value: vi.fn(),
  });
  vi.mocked(listWorkspaceMembers).mockResolvedValue([
    {
      id: 1,
      workspace_id: 4,
      user_id: 7,
      username: "owner",
      display_name: "Workspace Owner",
      email: "owner@example.test",
      is_active: true,
      role: "admin",
    },
  ]);
  vi.mocked(listWorkspaceJoinRequests).mockResolvedValue([]);
  vi.mocked(getWorkspaceCapabilities).mockResolvedValue({
    preset: "annotate",
    overrides: [],
    effective: ["annotation"],
    blocked: {},
  });
  vi.mocked(getWorkspaceGovernance).mockResolvedValue({
    workspace_id: 4,
    workspace_kind: "team",
    join_code: "join-abc",
    default_invite_expiry_minutes: 10080,
  });
  vi.mocked(listWorkspaceInvites).mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("WorkspaceSettings", () => {
  it("uses sentence-case preset and role labels without status styling", async () => {
    vi.mocked(getWorkspaceCapabilities).mockResolvedValueOnce({
      preset: "annotate", overrides: [], effective: ["annotation"],
      blocked: { active_learning: "This capability is planned." },
    });
    vi.mocked(listWorkspaceInvites).mockResolvedValueOnce([{
      id: 31, workspace_id: 4, role: "trainer", created_by: 7,
      created_by_username: "owner", expires_at: "2026-08-07T12:00:00Z",
      created_at: "2026-07-31T12:00:00Z",
    }]);
    render(<WorkspaceSettings workspaceId={4} workspaceName="Research team"
      workspaceKind="team" currentUserId={7} onManageWorkspaces={vi.fn()}
      onCapabilitiesChanged={vi.fn()} />);

    const preset = await screen.findByRole("combobox", { name: /Capability preset/ });
    expect(within(preset).getAllByRole("option").map((option) => option.textContent)).toEqual([
      "Annotate", "Train models", "Annotate and train",
      "Annotate and train with active learning", "Full",
    ]);
    const capabilities = screen.getByLabelText("Effective capabilities");
    expect(capabilities.textContent).toContain("Annotation");
    expect(capabilities.textContent).toContain("Active learning unavailable");
    expect(capabilities.textContent).toContain("This capability is planned.");
    expect(capabilities.querySelector(".platform-status")).toBeNull();
    expect(screen.getByRole("cell", { name: "Model trainer" }).querySelector(".platform-status")).toBeNull();
    expect(screen.getByRole("cell", { name: "Active" }).querySelector(".platform-status")).not.toBeNull();
  });

  it("shows each member's actual role in an individual workspace", async () => {
    const [owner] = await listWorkspaceMembers(4);
    vi.mocked(listWorkspaceMembers).mockResolvedValueOnce([
      owner, { ...owner, id: 2, user_id: 8, username: "trainer", display_name: "Researcher", role: "trainer" },
    ]);
    render(<WorkspaceSettings workspaceId={4} workspaceName="Personal"
      workspaceKind="individual" currentUserId={7} onManageWorkspaces={vi.fn()}
      onCapabilitiesChanged={vi.fn()} />);

    expect(await screen.findByRole("cell", { name: "Owner" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "Model trainer" })).toBeTruthy();
  });

  it.each([
    [1, "1 minute"], [2, "2 minutes"], [60, "1 hour"], [120, "2 hours"],
    [1440, "1 day"], [2880, "2 days"],
  ])("formats a default invitation expiry of %s minutes as %s", async (minutes, label) => {
    vi.mocked(getWorkspaceGovernance).mockResolvedValueOnce({
      workspace_id: 4, workspace_kind: "team", join_code: "join-abc",
      default_invite_expiry_minutes: minutes,
    });
    render(<WorkspaceSettings workspaceId={4} workspaceName="Research team"
      workspaceKind="team" currentUserId={7} onManageWorkspaces={vi.fn()}
      onCapabilitiesChanged={vi.fn()} />);

    expect(await screen.findByRole("option", { name: `Instance default (${label})` })).toBeTruthy();
  });

  it("cannot overwrite capabilities after their initial load fails", async () => {
    vi.mocked(getWorkspaceCapabilities).mockRejectedValueOnce(new Error("Connection lost"));
    render(<WorkspaceSettings workspaceId={4} workspaceName="Research team"
      workspaceKind="team" currentUserId={7} onManageWorkspaces={vi.fn()}
      onCapabilitiesChanged={vi.fn()} />);
    await screen.findByText("Connection lost");
    const save = screen.getByRole("button", { name: "Save capabilities" });
    expect(save.hasAttribute("disabled")).toBe(true);
    expect(screen.getByRole("combobox", { name: /^Capability preset/ }).hasAttribute("disabled")).toBe(true);
    await userEvent.setup().click(save);
    expect(setWorkspaceCapability).not.toHaveBeenCalled();
  });

  it("shows a custom capability policy as Custom", async () => {
    vi.mocked(getWorkspaceCapabilities).mockResolvedValueOnce({
      preset: "custom", overrides: [], effective: ["annotation"], blocked: {},
    });
    render(<WorkspaceSettings workspaceId={4} workspaceName="Research team"
      workspaceKind="team" currentUserId={7} onManageWorkspaces={vi.fn()}
      onCapabilitiesChanged={vi.fn()} />);
    await screen.findByRole("heading", { name: "Members" });
    const preset = screen.getByRole<HTMLSelectElement>("combobox", { name: /^Capability preset/ });
    expect(preset.value).toBe("custom");
    expect(preset.selectedOptions[0]?.textContent).toBe("Custom");
    expect(screen.getByRole("button", { name: "Save capabilities" }).hasAttribute("disabled")).toBe(true);
  });

  it("exposes and saves the independent Train Models capability preset", async () => {
    const user = userEvent.setup();
    const onCapabilitiesChanged = vi.fn();
    const updated = {
      preset: "train",
      overrides: [],
      effective: ["lineage", "training"],
      blocked: {},
    };
    vi.mocked(setWorkspaceCapability).mockResolvedValue(updated);

    render(
      <WorkspaceSettings
        workspaceId={4}
        workspaceName="Research team"
        workspaceKind="team"
        currentUserId={7}
        onManageWorkspaces={vi.fn()}
        onCapabilitiesChanged={onCapabilitiesChanged}
      />,
    );

    const preset = await screen.findByRole("combobox", {
      name: /Capability preset/,
    });
    await user.selectOptions(preset, "train");
    await user.click(
      screen.getByRole("button", { name: "Save capabilities" }),
    );

    await waitFor(() =>
      expect(setWorkspaceCapability).toHaveBeenCalledWith(4, "train"),
    );
    expect(onCapabilitiesChanged).toHaveBeenCalledWith(updated);
    expect(screen.getByText("Workspace capabilities saved.")).toBeTruthy();
  });

  it("keeps collaboration controls out of an individual workspace", async () => {
    render(
      <WorkspaceSettings
        workspaceId={4}
        workspaceName="Personal"
        workspaceKind="individual"
        currentUserId={7}
        onManageWorkspaces={vi.fn()}
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    await screen.findByRole("heading", { name: "Owner" });
    expect(screen.queryByRole("heading", { name: "Invite" })).toBeNull();
    expect(
      screen.queryByRole("heading", { name: "Join requests" }),
    ).toBeNull();
    expect(screen.getByRole("cell", { name: "Owner" })).toBeTruthy();
    expect(screen.queryByLabelText("Role for Workspace Owner")).toBeNull();
    expect(screen.getByRole("button", { name: "Create or join a team" })).toBeTruthy();
  });

  it("removes a freshly generated invitation link when that invite is revoked", async () => {
    const user = userEvent.setup();
    const summary = {
      id: 31,
      workspace_id: 4,
      role: "annotator" as const,
      created_by: 7,
      created_by_username: "owner",
      expires_at: "2026-08-07T12:00:00Z",
      created_at: "2026-07-31T12:00:00Z",
    };
    vi.mocked(createWorkspaceInvite).mockResolvedValue({
      id: 31,
      token: "fresh-invite-token",
      workspace_id: 4,
      role: "annotator",
      expires_at: summary.expires_at,
    });
    vi.mocked(listWorkspaceInvites)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([summary]);
    vi.mocked(revokeWorkspaceInvite).mockResolvedValue(undefined);

    render(
      <WorkspaceSettings
        workspaceId={4}
        workspaceName="Research team"
        workspaceKind="team"
        currentUserId={7}
        onManageWorkspaces={vi.fn()}
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    await screen.findByRole("heading", { name: "Invitations" });
    await user.click(screen.getByRole("button", { name: "Create invitation" }));
    expect(await screen.findByText(/fresh-invite-token/)).toBeTruthy();
    await user.click(
      await screen.findByRole("button", {
        name: "Revoke annotator invitation",
      }),
    );
    await user.click(screen.getByRole("button", { name: "Revoke invitation" }));

    await waitFor(() =>
      expect(revokeWorkspaceInvite).toHaveBeenCalledWith(4, 31),
    );
    expect(screen.queryByText(/fresh-invite-token/)).toBeNull();
  });

  it("ignores a late response from the previously selected workspace", async () => {
    const oldMembers = deferred<Awaited<ReturnType<typeof listWorkspaceMembers>>>();
    vi.mocked(listWorkspaceMembers).mockImplementation(async (workspaceId) =>
      workspaceId === 4
        ? oldMembers.promise
        : [
            {
              id: 2,
              workspace_id: 5,
              user_id: 8,
              username: "new-owner",
              display_name: "New Workspace Owner",
              email: null,
              is_active: true,
              role: "admin",
            },
          ],
    );

    const { rerender } = render(
      <WorkspaceSettings
        workspaceId={4}
        workspaceName="Old workspace"
        workspaceKind="team"
        currentUserId={7}
        onManageWorkspaces={vi.fn()}
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    rerender(
      <WorkspaceSettings
        workspaceId={5}
        workspaceName="New workspace"
        workspaceKind="individual"
        currentUserId={8}
        onManageWorkspaces={vi.fn()}
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    expect(await screen.findByText("New Workspace Owner")).toBeTruthy();

    await act(async () => {
      oldMembers.resolve([
        {
          id: 1,
          workspace_id: 4,
          user_id: 7,
          username: "old-owner",
          display_name: "Old Workspace Owner",
          email: null,
          is_active: true,
          role: "admin",
        },
      ]);
      await oldMembers.promise;
    });

    expect(screen.getByText("New Workspace Owner")).toBeTruthy();
    expect(screen.queryByText("Old Workspace Owner")).toBeNull();
  });
});

it("does not label an individual workspace member as owner when their role is trainer", async () => {
  vi.mocked(listWorkspaceMembers).mockResolvedValue([{ id: 1, workspace_id: 4, user_id: 7, username: "trainer", display_name: "Trainer", email: null, is_active: true, role: "trainer" }]);
  render(<WorkspaceSettings workspaceId={4} workspaceName="Research" workspaceKind="individual" currentUserId={7} onCapabilitiesChanged={vi.fn()} onManageWorkspaces={vi.fn()} />);
  expect(await screen.findByRole("cell", { name: "Model trainer" })).toBeTruthy();
  expect(screen.queryByRole("cell", { name: "Owner" })).toBeNull();
});
