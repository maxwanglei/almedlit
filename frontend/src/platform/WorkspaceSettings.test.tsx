// @vitest-environment jsdom

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  getWorkspaceCapabilities,
  listWorkspaceJoinRequests,
  listWorkspaceMembers,
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
  listWorkspaceJoinRequests: vi.fn(),
  listWorkspaceMembers: vi.fn(),
  rejectJoinRequest: vi.fn(),
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
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("WorkspaceSettings", () => {
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
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    await screen.findByRole("heading", { name: "Members" });
    expect(screen.queryByRole("heading", { name: "Invite" })).toBeNull();
    expect(
      screen.queryByRole("heading", { name: "Join requests" }),
    ).toBeNull();
    expect(
      (screen.getByLabelText("Role for Workspace Owner") as HTMLSelectElement)
        .disabled,
    ).toBe(true);
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
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    rerender(
      <WorkspaceSettings
        workspaceId={5}
        workspaceName="New workspace"
        workspaceKind="individual"
        currentUserId={8}
        onCapabilitiesChanged={vi.fn()}
      />,
    );

    expect(
      await screen.findByLabelText("Role for New Workspace Owner"),
    ).toBeTruthy();

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

    expect(screen.getByLabelText("Role for New Workspace Owner")).toBeTruthy();
    expect(screen.queryByText("Old Workspace Owner")).toBeNull();
  });
});
