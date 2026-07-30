// @vitest-environment jsdom

import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import OverviewScreen from "./OverviewScreen";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
} from "./types";

afterEach(cleanup);

describe("OverviewScreen", () => {
  it("shows the staged research loop and deep-links model development", async () => {
    const onOpenTraining = vi.fn();
    const onOpenModels = vi.fn();
    const data: PlatformProjectData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        project_id: 1,
        selected: [
          "data",
          "annotate",
          "learning",
          "train",
          "models",
          "guidelines",
          "activity",
        ],
        effective: [
          "data",
          "annotate",
          "learning",
          "train",
          "models",
          "guidelines",
          "activity",
        ],
        workspace_capabilities: [
          "annotation",
          "training",
          "inference",
          "active_learning",
          "co_learning",
          "lineage",
        ],
      },
    };

    render(
      <OverviewScreen
        data={data}
        documents={[]}
        assignments={[]}
        progress={null}
        onOpenData={vi.fn()}
        onOpenTraining={onOpenTraining}
        onOpenModels={onOpenModels}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Research loop roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Inference roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Active Learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Co-learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Lineage & Export roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "Guideline Learning roadmap" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("article", { name: "HPC & LLM Serving roadmap" }),
    ).toBeTruthy();
    expect(screen.getByText("Inference capability available")).toBeTruthy();
    expect(
      screen.getByText("Active learning capability available"),
    ).toBeTruthy();
    expect(screen.getAllByText("Not available in this workspace")).toHaveLength(
      1,
    );
    const roadmap = screen.getByLabelText(
      "Research loop capability roadmap",
    );
    expect(within(roadmap).queryByRole("link")).toBeNull();
    expect(within(roadmap).queryByRole("button")).toBeNull();
    screen.getByRole("link", { name: /Training/ }).click();
    screen.getByRole("link", { name: /Models/ }).click();
    expect(onOpenTraining).toHaveBeenCalledTimes(1);
    expect(onOpenModels).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("checkbox")).toBeNull();
  });
});
