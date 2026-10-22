// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";

import type { Project } from "@/types/api";
import SettingsScreen from "./SettingsScreen";
import { EMPTY_PLATFORM_PROJECT_DATA, type PlatformProjectData } from "./types";

const project: Project = {
  id: 7, workspace_id: 4, name: "Evidence", description: null,
  annotation_schema: { labels: {} }, annotation_validation_mode: "strict",
  tasks: [], settings: { default_export_format: "parquet" },
};
const data: PlatformProjectData = {
  ...EMPTY_PLATFORM_PROJECT_DATA,
  projectModules: {
    project_id: 7,
    selected: ["data", "annotate", "models", "train", "activity", "learning", "guidelines"],
    effective: ["data", "annotate", "models", "train", "activity"],
    workspace_capabilities: ["annotation", "training", "inference", "lineage"],
  },
};

afterEach(cleanup);

it("counts visible modules and retains saved hidden choices after confirmation", async () => {
  const user = userEvent.setup();
  const onUpdateModules = vi.fn().mockResolvedValue(undefined);
  render(<SettingsScreen project={project} data={data} busy={false} onUpdateProject={vi.fn()} onUpdateModules={onUpdateModules} />);
  expect(screen.getByText("5 selected for this project")).toBeTruthy();
  expect(screen.getAllByRole("checkbox")).toHaveLength(5);
  await user.click(screen.getByRole("checkbox", { name: /^Data/ }));
  expect(onUpdateModules).not.toHaveBeenCalled();
  const dialog = screen.getByRole("alertdialog");
  await user.click(within(dialog).getByRole("button", { name: "Disable module" }));
  await waitFor(() => expect(onUpdateModules).toHaveBeenCalledWith(["models", "activity", "learning", "guidelines"]));
});

it("associates an empty project name with its inline error", () => {
  render(<SettingsScreen project={project} data={data} busy={false} onUpdateProject={vi.fn()} onUpdateModules={vi.fn()} />);
  const name = screen.getByRole("textbox", { name: "Project name" });
  fireEvent.change(name, { target: { value: " " } });
  fireEvent.blur(name);
  expect(name.getAttribute("aria-invalid")).toBe("true");
  expect(document.getElementById(name.getAttribute("aria-describedby") ?? "")?.textContent).toBe("Enter a project name.");
  expect(screen.queryByLabelText("Default export format")).toBeNull();
});
