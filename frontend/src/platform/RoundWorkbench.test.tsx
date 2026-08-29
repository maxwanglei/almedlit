// @vitest-environment jsdom

import {
  act,
  cleanup,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RoundWorkbench from "./RoundWorkbench";
import type { RoundWorkData } from "./api";
import type { RoundWorkRound, TaskVersion } from "./types";

const mocks = vi.hoisted(() => ({
  addRoundDecision: vi.fn(),
  loadRoundWork: vi.fn(),
  publishRoundLabelSet: vi.fn(),
  recordFeedbackDecision: vi.fn(),
  revealRoundFeedback: vi.fn(),
  submitRoundDecisions: vi.fn(),
  transitionRound: vi.fn(),
}));

vi.mock("./api", () => mocks);

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
} {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function round(id: number, name: string): RoundWorkRound {
  return {
    id,
    project_id: 7,
    name,
    sequence: id,
    dataset_version_id: 21,
    task_version_id: 31,
    assistance_policy: "blind",
    feedback_available: false,
    status: "open",
    opened_at: null,
    closed_at: null,
  };
}

function roundWork(roundId: number, text: string): RoundWorkData {
  return {
    roundItems: [
      {
        id: roundId * 10,
        project_id: 7,
        annotation_round_id: roundId,
        dataset_item_id: roundId * 100,
        selection_rank: 1,
        selection_score: 0.42,
        selection_reason: { strategy: "uncertainty" },
      },
    ],
    datasetItems: [
      {
        id: roundId * 100,
        project_id: 7,
        dataset_version_id: 21,
        stable_key: `item-${roundId}`,
        group_key: null,
        payload: { text },
        content_hash: `hash-${roundId}`,
      },
    ],
    decisions: [],
    submissions: [],
  };
}

const task: TaskVersion = {
  id: 31,
  project_id: 7,
  task_definition_id: 30,
  version_number: 1,
  task_kind: "classification",
  input_schema: {},
  output_schema: { enum: ["Relevant", "Not relevant"] },
  label_rules: {},
  annotation_ui: {},
  metrics: ["accuracy"],
  trainer_compatibility: [],
  content_hash: "task-hash",
};

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(cleanup);

describe("RoundWorkbench", () => {
  it("ignores an older round response and its loading completion", async () => {
    const firstRound = round(41, "First round");
    const currentRound = round(42, "Current round");
    const oldWork = deferred<RoundWorkData>();
    const currentWork = deferred<RoundWorkData>();
    mocks.loadRoundWork.mockImplementation(
      async (_projectId: number, selectedRound: RoundWorkRound) =>
        selectedRound.id === firstRound.id
          ? oldWork.promise
          : currentWork.promise,
    );

    const { rerender } = render(
      <RoundWorkbench
        round={firstRound}
        task={task}
        currentUserId={12}
        canManage={false}
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await waitFor(() =>
      expect(mocks.loadRoundWork).toHaveBeenCalledWith(7, firstRound),
    );

    rerender(
      <RoundWorkbench
        round={currentRound}
        task={task}
        currentUserId={12}
        canManage={false}
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await waitFor(() =>
      expect(mocks.loadRoundWork).toHaveBeenCalledWith(7, currentRound),
    );

    await act(async () => {
      oldWork.resolve(roundWork(firstRound.id, "Stale source record"));
      await oldWork.promise;
    });
    expect(screen.getByText("Loading round…")).toBeTruthy();
    expect(screen.queryByText("Stale source record")).toBeNull();

    await act(async () => {
      currentWork.resolve(
        roundWork(currentRound.id, "Current source record"),
      );
      await currentWork.promise;
    });

    expect(await screen.findByText("Current source record")).toBeTruthy();
    expect(screen.getByText(/Priority #1/)).toBeTruthy();
    expect(screen.getByText(/Uncertainty 0.420/)).toBeTruthy();
    expect(screen.getByText("Uncertainty")).toBeTruthy();
    expect(screen.queryByText("Stale source record")).toBeNull();
  });
});
