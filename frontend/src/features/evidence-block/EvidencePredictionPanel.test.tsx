// @vitest-environment jsdom

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import EvidencePredictionPanel from "./EvidencePredictionPanel";
import type { EvidenceCandidatePrediction } from "@/types/api";

const PREDICTION: EvidenceCandidatePrediction = {
  id: 41,
  project_id: 1,
  run_id: 7,
  checkpoint_id: 5,
  document_id: 2,
  structure_version_id: 3,
  target_version_id: 11,
  start_sentence_id: 101,
  end_sentence_id: 103,
  start_sentence_ordinal: 0,
  end_sentence_ordinal: 2,
  start_char: 0,
  end_char: 44,
  block_confidence: 0.87,
  boundary_confidence: { start: 0.91, end: 0.82 },
  uncertainty: 0.13,
  decoder_version: "evidence-block-decoder-v1",
  source_window_ids: [21, 22],
  status: "pending",
  review_status: "pending",
  diagnostics_artifact_id: 99,
  metadata_: {},
  reviews: [],
};

afterEach(cleanup);

describe("EvidencePredictionPanel", () => {
  it("renders provenance and sends accept/reject review actions", async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    render(
      <EvidencePredictionPanel
        predictions={[PREDICTION]}
        loading={false}
        busy={false}
        selectionAvailable={false}
        selectedPredictionId={null}
        onSelect={vi.fn()}
        onReview={onReview}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText(/run #7 · checkpoint #5/i)).toBeTruthy();
    expect(screen.getByText(/87% confidence/i)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "Accept prediction 41" }));
    await user.click(screen.getByRole("button", { name: "Reject prediction 41" }));

    expect(onReview).toHaveBeenNthCalledWith(1, PREDICTION, "accept");
    expect(onReview).toHaveBeenNthCalledWith(2, PREDICTION, "reject");
  });

  it("only enables boundary modification when a sentence selection exists", async () => {
    const user = userEvent.setup();
    const onReview = vi.fn();
    const { rerender } = render(
      <EvidencePredictionPanel
        predictions={[PREDICTION]}
        loading={false}
        busy={false}
        selectionAvailable={false}
        selectedPredictionId={41}
        onSelect={vi.fn()}
        onReview={onReview}
        onRefresh={vi.fn()}
      />,
    );
    const modify = screen.getByRole("button", {
      name: "Modify prediction 41 using selected sentences",
    });
    expect((modify as HTMLButtonElement).disabled).toBe(true);

    rerender(
      <EvidencePredictionPanel
        predictions={[PREDICTION]}
        loading={false}
        busy={false}
        selectionAvailable
        selectedPredictionId={41}
        onSelect={vi.fn()}
        onReview={onReview}
        onRefresh={vi.fn()}
      />,
    );
    await user.click(
      screen.getByRole("button", {
        name: "Modify prediction 41 using selected sentences",
      }),
    );
    expect(onReview).toHaveBeenCalledWith(PREDICTION, "modify");
  });
});
