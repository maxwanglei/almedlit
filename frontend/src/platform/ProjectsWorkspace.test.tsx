// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createProject } from "@/api/client";
import type { Project } from "@/types/api";

import ProjectsWorkspace from "./ProjectsWorkspace";

vi.mock("@/api/client", () => ({
  createProject: vi.fn(),
}));

const trainingProject: Project = {
  id: 12,
  workspace_id: 4,
  name: "Public benchmark",
  description: "Standalone model development",
  annotation_schema: { labels: {} },
  annotation_validation_mode: "relaxed",
  tasks: [],
  settings: {
    modules: ["data", "models", "train", "activity"],
  },
};

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

describe("ProjectsWorkspace", () => {
  it("lets managers create a training-only project without Annotation", async () => {
    const user = userEvent.setup();
    const onProjectCreated = vi.fn();
    vi.mocked(createProject).mockResolvedValue(trainingProject);

    render(
      <ProjectsWorkspace
        workspaceId={4}
        projects={[]}
        canCreate
        capabilities={["training", "lineage"]}
        loading={false}
        onOpenProject={vi.fn()}
        onProjectCreated={onProjectCreated}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Create project" }));
    expect(
      (screen.getByRole("radio", { name: /Training only/ }) as HTMLInputElement)
        .checked,
    ).toBe(true);
    expect(
      (
        screen
          .getAllByRole("radio")
          .find((input) => input.getAttribute("value") === "annotation") as
          | HTMLInputElement
          | undefined
      )?.disabled,
    ).toBe(true);

    await user.type(screen.getByRole("textbox", { name: "Project name" }), "Public benchmark");
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create project",
      }),
    );

    await waitFor(() =>
      expect(createProject).toHaveBeenCalledWith(
        expect.objectContaining({
          workspace_id: 4,
          settings: {
            modules: ["data", "models", "train", "activity"],
          },
        }),
      ),
    );
    expect(onProjectCreated).toHaveBeenCalledWith(trainingProject);
  });

  it("gives Trainers read access without project-creation commands", async () => {
    const user = userEvent.setup();
    const onOpenProject = vi.fn();

    render(
      <ProjectsWorkspace
        workspaceId={4}
        projects={[trainingProject]}
        canCreate={false}
        capabilities={["training", "lineage"]}
        loading={false}
        onOpenProject={onOpenProject}
        onProjectCreated={vi.fn()}
      />,
    );

    expect(screen.queryByRole("button", { name: "New project" })).toBeNull();
    expect(screen.getByText("Training only")).toBeTruthy();
    const openLink = screen.getByRole("link", { name: "Open" });
    expect(openLink.getAttribute("href")).toBe("/projects/12/overview");
    openLink.addEventListener(
      "click",
      (event) => {
        if (event.ctrlKey) event.preventDefault();
      },
      { once: true },
    );
    fireEvent.click(openLink, { ctrlKey: true });
    expect(onOpenProject).not.toHaveBeenCalled();
    await user.click(openLink);
    expect(onOpenProject).toHaveBeenCalledWith(12);
  });
});

it("shows a load failure without declaring the project list empty", async () => {
  const onRetry = vi.fn();
  const user = userEvent.setup();
  render(<ProjectsWorkspace workspaceId={4} workspaceName="Research team" projects={[]} canCreate capabilities={["annotation"]} loading={false} loadError="Project service offline" onRetry={onRetry} onOpenProject={vi.fn()} onProjectCreated={vi.fn()} />);
  expect(screen.getByRole("alert").textContent).toContain("Project service offline");
  expect(screen.queryByText("No projects yet")).toBeNull();
  await user.click(screen.getByRole("button", { name: "Retry loading projects" }));
  expect(onRetry).toHaveBeenCalledOnce();
});

it("shows the workspace name in the project directory", () => {
  render(<ProjectsWorkspace workspaceId={4} workspaceName="Research team" projects={[trainingProject]} canCreate={false} capabilities={["training"]} loading={false} onOpenProject={vi.fn()} onProjectCreated={vi.fn()} />);
  expect(screen.getByRole("cell", { name: "Research team" })).toBeTruthy();
  expect(screen.queryByText("#4")).toBeNull();
});
