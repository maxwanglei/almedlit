// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import OnboardingPresetPicker from "./OnboardingPresetPicker";

const mocks = vi.hoisted(() => ({
  setWorkspaceCapability: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  setWorkspaceCapability: mocks.setWorkspaceCapability,
}));

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
  reject: (reason: unknown) => void;
} {
  let resolve!: () => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<void>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.setWorkspaceCapability.mockResolvedValue({
    workspace_id: 5,
    preset: "train",
    overrides: [],
    effective: ["training", "lineage"],
    blocked: {},
  });
});

afterEach(cleanup);

describe("OnboardingPresetPicker", () => {
  it("waits for refreshed workspace access before completing setup", async () => {
    const refresh = deferred();
    const onDone = vi.fn(() => refresh.promise);

    render(
      <OnboardingPresetPicker workspaceId={5} onDone={onDone} />,
    );

    const train = screen.getByRole("button", { name: "Train Models" });
    await userEvent.click(train);

    await waitFor(() => {
      expect(mocks.setWorkspaceCapability).toHaveBeenCalledWith(5, "train");
      expect(onDone).toHaveBeenCalledTimes(1);
      expect((train as HTMLButtonElement).disabled).toBe(true);
    });

    refresh.resolve();
    await waitFor(() =>
      expect((train as HTMLButtonElement).disabled).toBe(false),
    );
  });

  it("stays on setup and reports a capability refresh failure", async () => {
    const onDone = vi.fn().mockRejectedValue(
      new Error("Capabilities could not be refreshed."),
    );

    render(
      <OnboardingPresetPicker workspaceId={5} onDone={onDone} />,
    );

    const train = screen.getByRole("button", { name: "Train Models" });
    await userEvent.click(train);

    expect(
      await screen.findByText("Capabilities could not be refreshed."),
    ).toBeTruthy();
    expect(screen.getByRole("heading", {
      name: "Choose your workspace",
    })).toBeTruthy();
    expect((train as HTMLButtonElement).disabled).toBe(false);
  });
});
