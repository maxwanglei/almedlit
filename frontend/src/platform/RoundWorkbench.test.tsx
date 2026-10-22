// @vitest-environment jsdom

import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import RoundWorkbench from "./RoundWorkbench";
import type { RoundWorkData } from "./api";
import type { AnnotationDecision, RoundWorkRound, TaskVersion } from "./types";

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

function decision(
  roundItemId: number,
  id: number,
  kind: AnnotationDecision["decision_kind"],
): AnnotationDecision {
  return {
    id,
    project_id: 7,
    round_item_id: roundItemId,
    supersedes_decision_id: null,
    output: "Relevant",
    decision_kind: kind,
    is_initial_checkpoint: false,
    rationale: null,
    annotator_user_id: 12,
    content_hash: `decision-${id}`,
  };
}

beforeEach(() => {
  vi.resetAllMocks();
});

afterEach(cleanup);

describe("RoundWorkbench", () => {
  it("rejects blank regression but accepts explicit zero", async () => {
    mocks.loadRoundWork.mockResolvedValue(roundWork(41, "Numeric source"));
    mocks.addRoundDecision.mockResolvedValue({ ...decision(410, 501, "annotation"), output: 0 });
    render(<RoundWorkbench round={round(41, "Regression")} task={{ ...task, task_kind: "regression", output_schema: { type: "number" } }} currentUserId={12} canManage onClose={vi.fn()} onRefresh={vi.fn()} />);
    await screen.findByRole("spinbutton", { name: "Value" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Save decision" })).toHaveProperty("disabled", false));
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "Enter a valid numeric value.");
    expect(mocks.addRoundDecision).not.toHaveBeenCalled();
    expect(screen.getByRole("spinbutton", { name: "Value" }).getAttribute("aria-invalid")).toBe("true");
    expect(document.activeElement).toBe(screen.getByRole("spinbutton", { name: "Value" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Value" }), { target: { value: "0" } });
    mocks.addRoundDecision.mockResolvedValue({ ...decision(410, 501, "annotation"), output: 0 });
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    await waitFor(() => expect(mocks.addRoundDecision).toHaveBeenCalledWith(7, expect.objectContaining({ output: 0 })));
  });

  it("confirms round closure and keeps a failed close error inside the dialog", async () => {
    mocks.loadRoundWork.mockResolvedValue(roundWork(41, "Source"));
    mocks.transitionRound.mockRejectedValue(new Error("Close failed"));
    render(<RoundWorkbench round={round(41, "Round")} task={task} currentUserId={12} canManage onClose={vi.fn()} onRefresh={vi.fn()} />);
    fireEvent.click(await screen.findByRole("button", { name: "Close round" }));
    expect(mocks.transitionRound).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Cancel" }));
    expect(mocks.transitionRound).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Close round" }));
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Close round" }));
    expect(await within(screen.getByRole("alertdialog")).findByRole("alert")).toHaveProperty("textContent", "Close failed");
  });

  it("round-trips plain classification text and retains save confirmation", async () => {
    const saved = decision(410, 501, "annotation");
    mocks.loadRoundWork.mockResolvedValue({ ...roundWork(41, "Text source"), decisions: [saved] });
    mocks.addRoundDecision.mockResolvedValue({ ...saved, id: 502, supersedes_decision_id: 501 });
    render(<RoundWorkbench round={round(41, "Classification")} task={{ ...task, output_schema: { type: "string" } }} currentUserId={12} canManage onClose={vi.fn()} onRefresh={vi.fn()} />);
    await screen.findByDisplayValue("Relevant");
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    await waitFor(() => expect(mocks.addRoundDecision).toHaveBeenCalledWith(7, expect.objectContaining({ output: "Relevant", supersedesDecisionId: 501 })));
    expect(await screen.findByText("Decision saved.")).toBeTruthy();
  });

  it("keeps publication confirmation through refresh and prevents duplicates", async () => {
    const work = { ...roundWork(41, "Published source"), submissions: [{ id: 601, project_id: 7, annotation_round_id: 41, annotator_user_id: 12, sequence: 1, decision_ids: [501], content_hash: "submission" }] };
    mocks.loadRoundWork.mockImplementation(async () => structuredClone(work));
    mocks.publishRoundLabelSet.mockResolvedValue({ name: "Published labels", version_number: 1 });
    const closed = { ...round(41, "Closed"), status: "closed" };
    const onRefresh = vi.fn(async () => { rendered.rerender(<RoundWorkbench round={{ ...closed }} task={{ ...task }} currentUserId={12} canManage onClose={vi.fn()} onRefresh={onRefresh} />); });
    const rendered = render(<RoundWorkbench round={closed} task={task} currentUserId={12} canManage onClose={vi.fn()} onRefresh={onRefresh} />);
    fireEvent.click(await screen.findByRole("button", { name: "Publish label set" }));
    expect(mocks.publishRoundLabelSet).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Publish label set" }));
    await waitFor(() => expect(onRefresh).toHaveBeenCalledOnce());
    await waitFor(() => expect(mocks.loadRoundWork).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("Published labels v1 published.")).toBeTruthy();
    expect(screen.getByRole<HTMLButtonElement>("button", { name: "Publish label set" }).disabled).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Publish label set" }));
    expect(mocks.publishRoundLabelSet).toHaveBeenCalledOnce();
  });

  it("lets a manager adjudicate, finalize, and publish a closed round", async () => {
    const closedRound = {
      ...round(41, "Closed disagreement"),
      status: "closed",
      assistance_policy: "immediate_suggestions" as const,
      feedback_available: true,
    };
    const saved = decision(410, 501, "adjudication");
    mocks.loadRoundWork.mockResolvedValue(roundWork(41, "Disputed source record"));
    mocks.addRoundDecision.mockResolvedValue(saved);
    mocks.submitRoundDecisions.mockResolvedValue({
      id: 601,
      project_id: 7,
      annotation_round_id: 41,
      decision_ids: [501],
      annotator_user_id: 12,
      sequence: 1,
    });
    mocks.publishRoundLabelSet.mockResolvedValue({
      name: "Closed disagreement finalized labels",
      version_number: 1,
    });
    const onRefresh = vi.fn().mockResolvedValue(undefined);
    render(
      <RoundWorkbench
        round={closedRound}
        task={task}
        currentUserId={12}
        canManage
        onClose={vi.fn()}
        onRefresh={onRefresh}
      />,
    );

    const decisionType = await screen.findByRole("combobox", { name: "Decision type" });
    expect((decisionType as HTMLSelectElement).disabled).toBe(false);
    expect((screen.getByRole("option", { name: "Annotation" }) as HTMLOptionElement).disabled).toBe(true);
    expect((screen.getByRole("option", { name: "Correction" }) as HTMLOptionElement).disabled).toBe(true);
    expect((screen.getByRole("combobox", { name: "Label" }) as HTMLSelectElement).disabled).toBe(true);
    fireEvent.change(decisionType, { target: { value: "adjudication" } });
    expect((screen.getByRole("combobox", { name: "Label" }) as HTMLSelectElement).disabled).toBe(false);
    fireEvent.change(screen.getByRole("combobox", { name: "Label" }), {
      target: { value: "Relevant" },
    });
    fireEvent.change(screen.getByRole("textbox", { name: "Rationale" }), {
      target: { value: "Resolved against the source record." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    await waitFor(() => expect(mocks.addRoundDecision).toHaveBeenCalledWith(7, {
      roundItemId: 410,
      output: "Relevant",
      supersedesDecisionId: null,
      decisionKind: "adjudication",
      isInitialCheckpoint: false,
      rationale: "Resolved against the source record.",
    }));
    await waitFor(() => expect(
      (screen.getByRole("button", { name: "Finalize submission" }) as HTMLButtonElement).disabled,
    ).toBe(false));
    fireEvent.click(screen.getByRole("button", { name: "Finalize submission" }));
    expect(mocks.submitRoundDecisions).not.toHaveBeenCalled();
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Finalize submission" }));
    await waitFor(() => expect(mocks.submitRoundDecisions).toHaveBeenCalledWith(7, 41, [501]));
    await screen.findByText("Submission 1 finalized.");
    fireEvent.change(screen.getByRole("combobox", { name: "Label set source" }), {
      target: { value: "adjudicated" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Publish label set" }));
    fireEvent.click(within(screen.getByRole("alertdialog")).getByRole("button", { name: "Publish label set" }));
    await waitFor(() => expect(mocks.publishRoundLabelSet).toHaveBeenCalledWith(7, 41, {
      name: "Closed disagreement finalized labels",
      sourceKind: "adjudicated",
      submissionIds: [601],
    }));
    expect(mocks.transitionRound).not.toHaveBeenCalled();
    expect(mocks.revealRoundFeedback).not.toHaveBeenCalled();
    expect(mocks.recordFeedbackDecision).not.toHaveBeenCalled();
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("keeps closed ordinary annotation controls disabled for non-managers", async () => {
    const work = roundWork(41, "Completed ordinary source");
    work.decisions = [decision(410, 501, "annotation")];
    mocks.loadRoundWork.mockResolvedValue(work);
    render(
      <RoundWorkbench
        round={{ ...round(41, "Closed round"), status: "closed" }}
        task={task}
        currentUserId={12}
        canManage={false}
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await screen.findByText("Completed ordinary source");
    expect(screen.queryByRole("combobox", { name: "Decision type" })).toBeNull();
    expect((screen.getByRole("combobox", { name: "Label" }) as HTMLSelectElement).disabled).toBe(true);
    expect((screen.getByRole("textbox", { name: "Rationale" }) as HTMLTextAreaElement).disabled).toBe(true);
    for (const name of ["Save decision", "Finalize submission"]) {
      const button = screen.getByRole("button", { name });
      expect((button as HTMLButtonElement).disabled).toBe(true);
      fireEvent.click(button);
    }
    expect(mocks.addRoundDecision).not.toHaveBeenCalled();
    expect(mocks.submitRoundDecisions).not.toHaveBeenCalled();
  });

  it("does not finalize mixed ordinary and adjudication decisions on a closed round", async () => {
    const work = roundWork(41, "Ordinary decision needing adjudication");
    work.roundItems.push({ ...work.roundItems[0], id: 411, dataset_item_id: 4101 });
    work.datasetItems.push({ ...work.datasetItems[0], id: 4101, stable_key: "second-item" });
    work.decisions = [decision(410, 501, "annotation"), decision(411, 502, "adjudication")];
    mocks.loadRoundWork.mockResolvedValue(work);
    render(
      <RoundWorkbench
        round={{ ...round(41, "Mixed closed round"), status: "closed" }}
        task={task}
        currentUserId={12}
        canManage
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await screen.findByText("Ordinary decision needing adjudication");
    fireEvent.change(screen.getByRole("combobox", { name: "Decision type" }), {
      target: { value: "adjudication" },
    });
    expect((screen.getByRole("button", { name: "Save decision" }) as HTMLButtonElement).disabled).toBe(false);
    const finalize = screen.getByRole("button", { name: "Finalize submission" });
    expect((finalize as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(finalize);
    expect(mocks.submitRoundDecisions).not.toHaveBeenCalled();
  });

  it("preserves ordinary decision authoring on open rounds", async () => {
    mocks.loadRoundWork.mockResolvedValue(roundWork(41, "Open source record"));
    mocks.addRoundDecision.mockResolvedValue(decision(410, 501, "annotation"));
    render(
      <RoundWorkbench
        round={round(41, "Open annotation")}
        task={task}
        currentUserId={12}
        canManage={false}
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await screen.findByText("Open source record");
    const label = screen.getByRole("combobox", { name: "Label" });
    expect((label as HTMLSelectElement).disabled).toBe(false);
    fireEvent.change(label, { target: { value: "Relevant" } });
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    await waitFor(() => expect(mocks.addRoundDecision).toHaveBeenCalledWith(7, expect.objectContaining({
      decisionKind: "annotation",
      output: "Relevant",
    })));
    await waitFor(() => expect(
      (screen.getByRole("button", { name: "Finalize submission" }) as HTMLButtonElement).disabled,
    ).toBe(false));
  });

  it("retains feedback disposition checks for open assisted annotation", async () => {
    mocks.loadRoundWork.mockResolvedValue(roundWork(41, "Assisted open source"));
    mocks.addRoundDecision.mockResolvedValue(decision(410, 501, "annotation"));
    mocks.revealRoundFeedback.mockResolvedValue({
      exposure: { id: 701 },
      candidate: { output: "Relevant", score: 0.8 },
    });
    render(
      <RoundWorkbench
        round={{
          ...round(41, "Open assisted annotation"),
          assistance_policy: "immediate_suggestions",
          feedback_available: true,
        }}
        task={task}
        currentUserId={12}
        canManage={false}
        onClose={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );
    await screen.findByRole("region", { name: "Revealed feedback" });
    expect(mocks.revealRoundFeedback).toHaveBeenCalledWith(7, 41, 410);
    fireEvent.change(screen.getByRole("combobox", { name: "Label" }), {
      target: { value: "Relevant" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    expect(screen.getByRole("alert").textContent).toContain(
      "Record how the feedback affected this decision.",
    );
    expect(mocks.addRoundDecision).not.toHaveBeenCalled();
    fireEvent.change(screen.getByRole("combobox", { name: "Feedback decision" }), {
      target: { value: "accepted" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save decision" }));
    await waitFor(() => expect(mocks.addRoundDecision).toHaveBeenCalledWith(7, expect.objectContaining({ feedbackDecision: { exposureId: 701, decision: "accepted" } })));
    expect(mocks.recordFeedbackDecision).not.toHaveBeenCalled();
    await screen.findByText("Decision saved.");
  });

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
