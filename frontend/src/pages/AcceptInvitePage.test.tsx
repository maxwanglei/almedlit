// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AcceptInvitePage from "@/pages/AcceptInvitePage";

const mocks = vi.hoisted(() => ({
  acceptInvite: vi.fn(),
  getInvitePreview: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  acceptInvite: mocks.acceptInvite,
  getInvitePreview: mocks.getInvitePreview,
}));

function input(label: string): HTMLInputElement {
  return screen.getByLabelText(new RegExp(`^${label}`)) as HTMLInputElement;
}

function acceptButton(): HTMLElement {
  return screen.getByRole("button", { name: /accept invite/i });
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.getInvitePreview.mockResolvedValue({
    workspace_name: "Lab Team",
    workspace_kind: "team",
    role: "annotator",
    expires_at: null,
  });
  mocks.acceptInvite.mockResolvedValue(undefined);
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })),
  });
});

afterEach(cleanup);

describe("AcceptInvitePage", () => {
  it("names the workspace and role from the preview", async () => {
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={vi.fn()} />);

    expect(await screen.findByText(/Lab Team/)).toBeTruthy();
    expect(screen.getByText(/an annotator/)).toBeTruthy();
    expect(mocks.getInvitePreview).toHaveBeenCalledWith("tok-1");
  });

  it("creates a new account and accepts in one request", async () => {
    const onAccepted = vi.fn();
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={onAccepted} />);
    await screen.findByText(/Lab Team/);

    fireEvent.change(input("Username"), { target: { value: "newcomer" } });
    fireEvent.change(input("Password"), { target: { value: "strong-password" } });
    fireEvent.click(acceptButton());

    await waitFor(() => {
      expect(mocks.acceptInvite).toHaveBeenCalledWith("tok-1", {
        username: "newcomer",
        password: "strong-password",
        display_name: undefined,
        create_account: true,
      });
    });
    await waitFor(() => expect(onAccepted).toHaveBeenCalled());
  });

  it("authenticates an existing user while accepting", async () => {
    const onAccepted = vi.fn();
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={onAccepted} />);
    await screen.findByText(/Lab Team/);

    fireEvent.click(screen.getByRole("button", { name: "Use an existing account" }));
    fireEvent.change(input("Username"), { target: { value: "existing" } });
    fireEvent.change(input("Password"), { target: { value: "pw" } });
    fireEvent.click(acceptButton());

    await waitFor(() => {
      expect(mocks.acceptInvite).toHaveBeenCalledWith("tok-1", {
        username: "existing",
        password: "pw",
        create_account: false,
      });
    });
    await waitFor(() => expect(onAccepted).toHaveBeenCalled());
  });

  it("confirms directly when a session already exists", async () => {
    const onAccepted = vi.fn();
    render(<AcceptInvitePage token="tok-1" signedIn onAccepted={onAccepted} />);
    await screen.findByText(/Lab Team/);

    expect(screen.queryByLabelText(/^Username/)).toBeNull();
    fireEvent.click(acceptButton());

    await waitFor(() => expect(mocks.acceptInvite).toHaveBeenCalledWith("tok-1"));
    await waitFor(() => expect(onAccepted).toHaveBeenCalled());
  });

  it("explains an expired or consumed invite instead of showing a form", async () => {
    mocks.getInvitePreview.mockRejectedValue(new Error("Invite has expired"));
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={vi.fn()} />);

    expect(await screen.findByText(/Invite has expired/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: /accept invite/i })).toBeNull();
  });

  it("rejects a short password before calling the API", async () => {
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={vi.fn()} />);
    await screen.findByText(/Lab Team/);

    fireEvent.change(input("Username"), { target: { value: "newcomer" } });
    fireEvent.change(input("Password"), { target: { value: "short" } });
    fireEvent.click(acceptButton());

    expect(await screen.findByText(/between 12 and 72 UTF-8 bytes/)).toBeTruthy();
    expect(mocks.acceptInvite).not.toHaveBeenCalled();
  });

  it("surfaces a failed acceptance without claiming success", async () => {
    mocks.acceptInvite.mockRejectedValue(new Error("Invite is invalid or already used"));
    const onAccepted = vi.fn();
    render(<AcceptInvitePage token="tok-1" signedIn={false} onAccepted={onAccepted} />);
    await screen.findByText(/Lab Team/);

    fireEvent.change(input("Username"), { target: { value: "newcomer" } });
    fireEvent.change(input("Password"), { target: { value: "strong-password" } });
    fireEvent.click(acceptButton());

    expect(await screen.findByText(/Invite is invalid or already used/)).toBeTruthy();
    expect(onAccepted).not.toHaveBeenCalled();
  });
});
