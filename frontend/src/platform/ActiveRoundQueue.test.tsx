// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ActiveRoundQueue from "./ActiveRoundQueue";
import type { RoundWorkContext } from "./types";

afterEach(cleanup);

function context(
  id: number,
  sequence: number,
  name: string,
  guideline: RoundWorkContext["guideline"],
): RoundWorkContext {
  return {
    project: { id: 1, name: "Review corpus" },
    round: {
      id,
      project_id: 1,
      name,
      sequence,
      dataset_version_id: 22,
      task_version_id: 12,
      assistance_policy: "reveal_after_first_pass",
      feedback_available: true,
      status: "open",
      opened_at: "2026-07-27T14:00:00Z",
      closed_at: null,
    },
    task: {
      id: 11,
      key: "screening",
      name: "Abstract screening",
    },
    task_version: {
      id: 12,
      project_id: 1,
      task_definition_id: 11,
      version_number: 3,
      task_kind: "classification",
      input_schema: {},
      output_schema: {},
      label_rules: {},
      annotation_ui: {},
      metrics: ["f1"],
      trainer_compatibility: [],
      content_hash: "task-hash",
    },
    cycle: null,
    guideline,
  };
}

describe("ActiveRoundQueue", () => {
  it("renders only the assignment-scoped contexts supplied by the API", () => {
    const onOpenRound = vi.fn();
    const contexts = [
      context(71, 4, "False-negative review", {
        guideline_id: 91,
        guideline_revision_id: 92,
        name: "Screening guidance",
        version_number: 5,
        status: "pilot",
      }),
      context(74, 7, "Open project round", null),
    ];

    render(
      <ActiveRoundQueue
        contexts={contexts}
        onOpenRound={onOpenRound}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Annotation rounds 2" }),
    ).toBeTruthy();
    expect(screen.getByText("False-negative review")).toBeTruthy();
    expect(screen.getByText("Open project round")).toBeTruthy();
    expect(screen.getByText("Review corpus · Round 4")).toBeTruthy();
    expect(screen.getByText("Review corpus · Round 7")).toBeTruthy();
    expect(screen.getAllByText("Abstract screening · v3")).toHaveLength(2);
    expect(screen.getByText("Screening guidance · v5")).toBeTruthy();
    expect(screen.getByText("Not pinned")).toBeTruthy();

    const openLinks = screen.getAllByRole("link", { name: "Open round" });
    expect(openLinks[0]?.getAttribute("href")).toBe("/my-work/rounds/74");
    fireEvent.click(openLinks[0]);
    expect(onOpenRound).toHaveBeenCalledWith(74);
  });
});
