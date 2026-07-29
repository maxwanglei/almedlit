// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ModuleSwitcher from "./ModuleSwitcher";

afterEach(cleanup);

describe("ModuleSwitcher", () => {
  const modules = [
    { id: "my-work" as const, label: "My Work", path: "/my-work" },
    {
      id: "projects" as const,
      label: "Project Setup",
      path: "/projects/9/overview",
    },
  ];

  it("renders real links and intercepts only an unmodified primary click", () => {
    const onNavigate = vi.fn();
    render(
      <ModuleSwitcher
        modules={modules}
        currentModuleId="my-work"
        onNavigate={onNavigate}
      />,
    );

    const current = screen.getByRole("link", { name: "My Work" });
    const projectSetup = screen.getByRole("link", { name: "Project Setup" });
    expect(current.getAttribute("aria-current")).toBe("page");
    expect(projectSetup.getAttribute("href")).toBe("/projects/9/overview");

    fireEvent.click(projectSetup);
    expect(onNavigate).toHaveBeenCalledWith("/projects/9/overview");

    onNavigate.mockClear();
    fireEvent.click(projectSetup, { ctrlKey: true });
    expect(onNavigate).not.toHaveBeenCalled();
  });

  it("is hidden when no module is available", () => {
    const { container } = render(
      <ModuleSwitcher
        modules={[]}
        currentModuleId={null}
        onNavigate={vi.fn()}
      />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders an accessible mobile drawer with distinct destination and workspace groups", async () => {
    const onNavigate = vi.fn();
    render(
      <ModuleSwitcher
        presentation="mobile"
        modules={modules}
        currentModuleId="my-work"
        workspaceSwitcher={(
          <label>
            Switch workspace
            <select defaultValue="10">
              <option value="10">Personal · Owner</option>
            </select>
          </label>
        )}
        workspaceSettings={{
          id: "workspace-settings",
          label: "Workspace Settings",
          path: "/workspace-settings",
        }}
        onNavigate={onNavigate}
        onLogout={vi.fn()}
      />,
    );

    const trigger = screen.getByRole("button", {
      name: "Open navigation menu",
    });
    fireEvent.click(trigger);

    expect(screen.getByRole("dialog", { name: "Navigation" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "Go to" })).toBeTruthy();
    expect(
      screen.getByRole("heading", { name: "Switch workspace" }),
    ).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Switch workspace" })).toBeTruthy();
    expect(document.body.style.overflow).toBe("hidden");

    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() =>
      expect(screen.queryByRole("dialog", { name: "Navigation" })).toBeNull(),
    );
    expect(document.body.style.overflow).toBe("");
    expect(document.activeElement).toBe(trigger);
  });
});
