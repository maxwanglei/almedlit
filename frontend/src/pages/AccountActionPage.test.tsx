// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AccountActionPage from "./AccountActionPage";

const mocks = vi.hoisted(() => ({
  completeAccountAction: vi.fn(),
  getAccountActionPreview: vi.fn(),
}));

vi.mock("@/api/client", () => mocks);

function passwordField(label: string): HTMLInputElement {
  return screen.getByLabelText(new RegExp(`^${label}`)) as HTMLInputElement;
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.getAccountActionPreview.mockResolvedValue({
    purpose: "activation",
    username: "ada",
    display_name: "Ada Lovelace",
    expires_at: "2026-08-01T15:00:00Z",
  });
  mocks.completeAccountAction.mockResolvedValue(undefined);
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

describe("AccountActionPage", () => {
  it("previews an activation action without exposing the token", async () => {
    render(<AccountActionPage token="secret-token" onCompleted={vi.fn()} />);

    expect(await screen.findByRole("heading", { name: "Activate your account" })).toBeTruthy();
    expect(screen.getByText(/Ada Lovelace/)).toBeTruthy();
    expect(screen.getByText(/This link expires/)).toBeTruthy();
    expect(screen.queryByText("secret-token")).toBeNull();
    expect(mocks.getAccountActionPreview).toHaveBeenCalledWith("secret-token");
  });

  it("validates password length and confirmation before completing", async () => {
    render(<AccountActionPage token="token" onCompleted={vi.fn()} />);
    await screen.findByRole("heading", { name: "Activate your account" });

    fireEvent.change(passwordField("New password"), { target: { value: "short" } });
    fireEvent.change(passwordField("Confirm new password"), { target: { value: "short" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));
    expect(await screen.findByText(/between 12 and 72 UTF-8 bytes/)).toBeTruthy();
    expect(mocks.completeAccountAction).not.toHaveBeenCalled();

    fireEvent.change(passwordField("New password"), { target: { value: "long-enough-password" } });
    fireEvent.change(passwordField("Confirm new password"), { target: { value: "different-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));
    expect(await screen.findByText("Passwords do not match.")).toBeTruthy();
    expect(mocks.completeAccountAction).not.toHaveBeenCalled();
  });

  it("completes a password reset and hands the adopted session back to the shell", async () => {
    mocks.getAccountActionPreview.mockResolvedValue({
      purpose: "password_reset",
      username: "ada",
      display_name: "Ada Lovelace",
      expires_at: "2026-08-01T15:00:00Z",
    });
    const onCompleted = vi.fn();
    render(<AccountActionPage token="reset-token" onCompleted={onCompleted} />);
    await screen.findByRole("heading", { name: "Reset your password" });

    fireEvent.change(passwordField("New password"), { target: { value: "replacement-password" } });
    fireEvent.change(passwordField("Confirm new password"), { target: { value: "replacement-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Reset password" }));

    await waitFor(() =>
      expect(mocks.completeAccountAction).toHaveBeenCalledWith(
        "reset-token",
        "replacement-password",
      ),
    );
    expect(onCompleted).toHaveBeenCalledTimes(1);
  });

  it("explains invalid, consumed, and expired links without rendering a password form", async () => {
    mocks.getAccountActionPreview.mockRejectedValue(new Error("Account action has expired"));
    render(<AccountActionPage token="expired-token" onCompleted={vi.fn()} />);

    expect(await screen.findByText("Account action has expired")).toBeTruthy();
    expect(screen.queryByLabelText("New password")).toBeNull();
  });

  it("surfaces completion failure and keeps the form available", async () => {
    mocks.completeAccountAction.mockRejectedValue(new Error("Account action was already used"));
    const onCompleted = vi.fn();
    render(<AccountActionPage token="used-token" onCompleted={onCompleted} />);
    await screen.findByRole("heading", { name: "Activate your account" });

    fireEvent.change(passwordField("New password"), { target: { value: "replacement-password" } });
    fireEvent.change(passwordField("Confirm new password"), { target: { value: "replacement-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Activate account" }));

    expect(await screen.findByText("Account action was already used")).toBeTruthy();
    expect(onCompleted).not.toHaveBeenCalled();
    expect(passwordField("New password")).toBeTruthy();
  });
});
