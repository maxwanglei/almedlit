// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import ModelScoringPanel from "./ModelScoringPanel";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type FeedbackRun,
  type PlatformProjectData,
} from "./types";

const mocks = vi.hoisted(() => ({
  getFeedbackRun: vi.fn(),
}));

vi.mock("./api", () => ({
  getFeedbackRun: mocks.getFeedbackRun,
}));

function feedbackRun(
  status: FeedbackRun["status"] = "queued",
): FeedbackRun {
  return {
    id: 91,
    project_id: 7,
    dataset_version_id: 22,
    task_version_id: 12,
    producer_type: "registered_model",
    cycle_id: null,
    model_version_id: 81,
    provider: null,
    external_model_id: null,
    exact_revision: null,
    prompt_template_hash: null,
    configuration: {},
    data_egress_policy: {},
    status,
    output_feedback_set_version_id: null,
    failure_code: null,
    failure_reason: null,
    started_at: null,
    heartbeat_at: null,
    completed_at: null,
  };
}

function projectData(run: FeedbackRun): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    feedbackRuns: [run],
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("ModelScoringPanel", () => {
  it("polls the single-run endpoint and cleans up when unmounted", async () => {
    mocks.getFeedbackRun.mockResolvedValue(feedbackRun("running"));
    const onRefresh = vi.fn().mockResolvedValue(undefined);
    const rendered = render(
      <ModelScoringPanel
        projectId={7}
        data={projectData(feedbackRun())}
        onCreate={vi.fn()}
        onRefresh={onRefresh}
      />,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3_000);
    });
    expect(mocks.getFeedbackRun).toHaveBeenCalledWith(7, 91);
    const callsBeforeUnmount = mocks.getFeedbackRun.mock.calls.length;

    rendered.unmount();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(9_000);
    });
    expect(mocks.getFeedbackRun).toHaveBeenCalledTimes(callsBeforeUnmount);
  });

  it("pauses automatic polling after two minutes and offers manual refresh", async () => {
    mocks.getFeedbackRun.mockResolvedValue(feedbackRun("running"));
    render(
      <ModelScoringPanel
        projectId={7}
        data={projectData(feedbackRun())}
        onCreate={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(120_000);
    });
    expect(screen.getByText(/Automatic updates paused/)).toBeTruthy();
    const callsBeforeRefresh = mocks.getFeedbackRun.mock.calls.length;

    fireEvent.click(screen.getByRole("button", { name: "Refresh status" }));
    await act(async () => undefined);
    expect(mocks.getFeedbackRun.mock.calls.length).toBeGreaterThan(callsBeforeRefresh);
  });

  it("shows terminal failure details without polling", () => {
    render(
      <ModelScoringPanel
        projectId={7}
        data={projectData({
          ...feedbackRun("failed"),
          failure_code: "ValidationError",
          failure_reason: "Dataset item has no input text",
        })}
        onCreate={vi.fn()}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    expect(screen.getByText("Dataset item has no input text")).toBeTruthy();
    expect(mocks.getFeedbackRun).not.toHaveBeenCalled();
  });
});
