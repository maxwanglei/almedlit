// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { MeMembership } from "@/api/client";

import WorkspaceSwitcher from "./WorkspaceSwitcher";

const memberships: MeMembership[] = [
  {
    workspace_id: 10,
    workspace_name: "Personal",
    workspace_kind: "individual",
    role: "admin",
  },
  {
    workspace_id: 20,
    workspace_name: "Research team",
    workspace_kind: "team",
    role: "annotator",
  },
];

afterEach(cleanup);

describe("WorkspaceSwitcher", () => {
  it("selects another membership", () => {
    const onWorkspaceChange = vi.fn();
    render(
      <WorkspaceSwitcher
        memberships={memberships}
        activeWorkspaceId={10}
        onWorkspaceChange={onWorkspaceChange}
      />,
    );

    fireEvent.change(screen.getByLabelText("Workspace"), { target: { value: "20" } });

    expect(onWorkspaceChange).toHaveBeenCalledWith(20);
  });

  it("keeps the active workspace and effective role visible for one workspace", () => {
    render(
      <WorkspaceSwitcher
        memberships={[memberships[0]]}
        activeWorkspaceId={10}
        onWorkspaceChange={vi.fn()}
      />,
    );

    const selector = screen.getByLabelText("Workspace") as HTMLSelectElement;
    expect(selector.disabled).toBe(true);
    expect(selector.options[0]?.text).toBe("Personal · Owner");
  });

  it("uses the user-facing hierarchical role label", () => {
    render(
      <WorkspaceSwitcher
        memberships={[
          {
            workspace_id: 30,
            workspace_name: "Model lab",
            workspace_kind: "team",
            role: "trainer",
          },
        ]}
        activeWorkspaceId={30}
        onWorkspaceChange={vi.fn()}
      />,
    );

    expect(
      (screen.getByLabelText("Workspace") as HTMLSelectElement).options[0]?.text,
    ).toBe("Model lab · Model Trainer");
  });
});
