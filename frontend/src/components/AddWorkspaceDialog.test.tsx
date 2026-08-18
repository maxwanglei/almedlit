// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { applyToWorkspace, createWorkspace } from "@/api/client";

import AddWorkspaceDialog from "./AddWorkspaceDialog";

vi.mock("@/api/client", () => ({
  applyToWorkspace: vi.fn(),
  createWorkspace: vi.fn(),
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
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("AddWorkspaceDialog", () => {
  it("creates a separate team workspace", async () => {
    const user = userEvent.setup();
    const onWorkspaceCreated = vi.fn();
    vi.mocked(createWorkspace).mockResolvedValue({
      id: 22,
      name: "Evidence team",
      kind: "team",
      capability_preset: "annotate",
    });

    render(
      <AddWorkspaceDialog
        open
        requestGeneration={1}
        onDismiss={vi.fn()}
        onWorkspaceCreated={onWorkspaceCreated}
        onJoinRequested={vi.fn()}
      />,
    );

    await user.type(screen.getByRole("textbox", { name: /Team name/ }), "Evidence team");
    await user.click(screen.getByRole("button", { name: "Create team" }));

    await waitFor(() => expect(createWorkspace).toHaveBeenCalledWith("Evidence team"));
    expect(onWorkspaceCreated).toHaveBeenCalledWith(22);
  });

  it("submits an approval-based join request", async () => {
    const user = userEvent.setup();
    const onJoinRequested = vi.fn();
    vi.mocked(applyToWorkspace).mockResolvedValue({
      id: 9,
      workspace_id: 5,
      user_id: 3,
      username: "ana",
      display_name: "Ana",
      email: null,
      status: "pending",
      message: "Research collaborator",
      created_at: "2026-07-31T12:00:00Z",
    });

    render(
      <AddWorkspaceDialog
        open
        requestGeneration={1}
        onDismiss={vi.fn()}
        onWorkspaceCreated={vi.fn()}
        onJoinRequested={onJoinRequested}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Join a team" }));
    await user.type(screen.getByRole("textbox", { name: "Join code" }), "join-code");
    await user.type(screen.getByRole("textbox", { name: /Message/ }), "Research collaborator");
    await user.click(screen.getByRole("button", { name: "Send request" }));

    await waitFor(() =>
      expect(applyToWorkspace).toHaveBeenCalledWith(
        "join-code",
        "Research collaborator",
      ),
    );
    expect(onJoinRequested).not.toHaveBeenCalled();
    expect(screen.getByText(/request was sent/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Refresh workspaces" }));
    expect(onJoinRequested).toHaveBeenCalledTimes(1);
    expect(screen.getByText(/memberships refreshed/)).toBeTruthy();
  });

  it("ignores a create response from an earlier account generation", async () => {
    const user = userEvent.setup();
    const onWorkspaceCreated = vi.fn();
    let resolveCreate!: (workspace: {
      id: number;
      name: string;
      kind: string;
      capability_preset: string;
    }) => void;
    vi.mocked(createWorkspace).mockReturnValue(
      new Promise((resolve) => {
        resolveCreate = resolve;
      }),
    );

    const { rerender } = render(
      <AddWorkspaceDialog
        open
        requestGeneration={1}
        onDismiss={vi.fn()}
        onWorkspaceCreated={onWorkspaceCreated}
        onJoinRequested={vi.fn()}
      />,
    );
    await user.type(screen.getByRole("textbox", { name: /Team name/ }), "Old team");
    await user.click(screen.getByRole("button", { name: "Create team" }));
    await waitFor(() => expect(createWorkspace).toHaveBeenCalledWith("Old team"));

    rerender(
      <AddWorkspaceDialog
        open={false}
        requestGeneration={2}
        onDismiss={vi.fn()}
        onWorkspaceCreated={onWorkspaceCreated}
        onJoinRequested={vi.fn()}
      />,
    );
    resolveCreate({
      id: 23,
      name: "Old team",
      kind: "team",
      capability_preset: "annotate",
    });

    await waitFor(() => expect(onWorkspaceCreated).not.toHaveBeenCalled());
  });
});
