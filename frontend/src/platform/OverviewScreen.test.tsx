// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import OverviewScreen from "./OverviewScreen";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
} from "./types";

afterEach(cleanup);

describe("OverviewScreen", () => {
  it("keeps future modules compact and deep-links model development", async () => {
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

    expect(screen.getByRole("heading", { name: "Future modules" })).toBeTruthy();
    expect(
      screen.getByText("Learning Loop and Guideline Learning"),
    ).toBeTruthy();
    screen.getByRole("link", { name: /Training/ }).click();
    screen.getByRole("link", { name: /Models/ }).click();
    expect(onOpenTraining).toHaveBeenCalledTimes(1);
    expect(onOpenModels).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("checkbox")).toBeNull();
  });
});
