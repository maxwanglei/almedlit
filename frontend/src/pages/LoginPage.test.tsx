// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import LoginPage from "@/pages/LoginPage";

const mocks = vi.hoisted(() => ({
  login: vi.fn(),
  register: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  login: mocks.login,
  register: mocks.register,
}));

function input(label: string): HTMLInputElement {
  return screen.getByLabelText(new RegExp(`^${label}`)) as HTMLInputElement;
}

beforeEach(() => {
  vi.clearAllMocks();
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
  mocks.login.mockResolvedValue(undefined);
  mocks.register.mockResolvedValue(undefined);
});

afterEach(cleanup);

describe("LoginPage credential fields", () => {
  it("does not carry saved login credentials into registration", () => {
    render(<LoginPage onAuthed={vi.fn()} />);

    expect(input("Username").value).toBe("");
    expect(input("Username").getAttribute("autocomplete")).toBe(
      "section-login username",
    );
    expect(input("Password").getAttribute("autocomplete")).toBe(
      "section-login current-password",
    );

    fireEvent.change(input("Username"), { target: { value: "test3" } });
    fireEvent.change(input("Password"), { target: { value: "old-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Create an account" }));

    expect(input("Username").value).toBe("");
    expect(input("Password").value).toBe("");
    expect(input("Username").getAttribute("autocomplete")).toBe(
      "section-register username",
    );
    expect(input("Password").getAttribute("autocomplete")).toBe(
      "section-register new-password",
    );
    expect(input("Display name").getAttribute("autocomplete")).toBe(
      "section-register name",
    );
  });

  it("submits only credentials entered in the fresh registration form", async () => {
    const onAuthed = vi.fn();
    render(<LoginPage onAuthed={onAuthed} />);

    fireEvent.change(input("Username"), { target: { value: "test3" } });
    fireEvent.change(input("Password"), { target: { value: "old-password" } });
    fireEvent.click(screen.getByRole("button", { name: "Create an account" }));

    fireEvent.change(input("Username"), { target: { value: "fresh-user" } });
    fireEvent.change(input("Password"), {
      target: { value: "correct-horse-battery" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Register" }));

    await waitFor(() =>
      expect(mocks.register).toHaveBeenCalledWith(
        "fresh-user",
        "correct-horse-battery",
        {
          displayName: undefined,
          workspaceKind: "individual",
          workspaceName: undefined,
        },
      ),
    );
    expect(onAuthed).toHaveBeenCalledWith(true);
  });
});
