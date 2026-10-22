// @vitest-environment jsdom
import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import EvidenceAdjudicationPanel from "./EvidenceAdjudicationPanel";
import type { TaskAssignment } from "@/types/api";

const api = vi.hoisted(() => ({ adjudicateEvidence: vi.fn(), getEvidenceAdjudication: vi.fn(), getDocumentStructure: vi.fn() }));
vi.mock("@/api/client", () => api);
const assignment: TaskAssignment = {
  id: 1, project_id: 1, document_id: 41, task_id: 1, assignee_user_id: 1, annotator_id: "alice",
  status: "assigned", assigned_by_user_id: 1, assigned_by: "alice", notes: null, metadata_: {},
  target_version_id: 101, structure_version_id: 201, guideline_version_id: 301, assignment_scope_key: "evidence",
};
beforeEach(() => {
  vi.clearAllMocks();
  api.getEvidenceAdjudication.mockResolvedValue({ project_id: 1, document_id: 41, target_version_id: 101,
    structure_version_id: 201, guideline_version_id: 301, blocks: [{ annotation_id: 4, annotator_user_id: 1,
      annotator_id: "alice", status: "draft", start_sentence_id: 701, end_sentence_id: 703,
      start_sentence_ordinal: 0, end_sentence_ordinal: 2, labels: [], note: null }] });
  api.getDocumentStructure.mockImplementation(async (_id: number, query: { sentenceStart: number }) => ({
    sentences: [{ id: 701 + query.sentenceStart, ordinal: query.sentenceStart }],
  }));
  api.adjudicateEvidence.mockResolvedValue({ id: 5 });
});
afterEach(cleanup);

async function openPanel() {
  const user = userEvent.setup();
  render(<EvidenceAdjudicationPanel projectId={1} documents={[]} assignments={[assignment]} allowSoloGold />);
  await user.click(screen.getByRole("button", { name: "Load comparison" }));
  await screen.findByText("Sentences 0-2");
  return user;
}

it("identifies opaque pinned-version IDs without presenting them as version numbers", () => {
  render(<EvidenceAdjudicationPanel projectId={1} documents={[]} assignments={[assignment]} allowSoloGold />);
  expect(screen.getByRole("option").textContent).toMatch(/target ID\s*101.*guideline ID\s*301/);
  expect(screen.getByRole("option").textContent).not.toContain("v101");
});

it("rejects blank boundaries instead of treating them as sentence zero", async () => {
  const user = await openPanel();
  await user.click(screen.getByRole("button", { name: "Create gold block" }));
  await screen.findByRole("alert");
  expect(api.adjudicateEvidence).not.toHaveBeenCalled();
  expect(api.getDocumentStructure).not.toHaveBeenCalled();
  expect(screen.getByLabelText("Start sentence ordinal").getAttribute("aria-invalid")).toBe("true");
  expect(document.activeElement).toBe(screen.getByLabelText("Start sentence ordinal"));
});

it("resolves displayed sentence ordinals in the pinned structure before saving gold", async () => {
  const user = await openPanel();
  await user.type(screen.getByLabelText("Start sentence ordinal"), "1");
  await user.type(screen.getByLabelText("End sentence ordinal"), "2");
  await user.click(screen.getByRole("button", { name: "Create gold block" }));
  await waitFor(() => expect(api.adjudicateEvidence).toHaveBeenCalledWith(1, 41, expect.objectContaining({
    start_sentence_id: 702, end_sentence_id: 703, solo_gold: true,
  })));
  expect(api.getDocumentStructure).toHaveBeenCalledWith(41, { versionId: 201, sentenceStart: 1, sentenceLimit: 1 });
  expect(await screen.findByText("Gold evidence block #5 created")).toBeTruthy();
});

it.each([["2", "1"], ["-1", "2"], ["0.5", "2"]])("rejects invalid custom range %s to %s", async (start, end) => {
  const user = await openPanel();
  await user.type(screen.getByLabelText("Start sentence ordinal"), start);
  await user.type(screen.getByLabelText("End sentence ordinal"), end);
  await user.click(screen.getByRole("button", { name: "Create gold block" }));
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(api.adjudicateEvidence).not.toHaveBeenCalled();
});

it("rejects an ordinal outside the pinned structure", async () => {
  api.getDocumentStructure.mockResolvedValue({ sentences: [] });
  const user = await openPanel();
  await user.type(screen.getByLabelText("Start sentence ordinal"), "99");
  await user.type(screen.getByLabelText("End sentence ordinal"), "99");
  await user.click(screen.getByRole("button", { name: "Create gold block" }));
  expect(await screen.findByRole("alert")).toBeTruthy();
  expect(api.adjudicateEvidence).not.toHaveBeenCalled();
});

it("discards a comparison from the previous scope and releases its busy state", async () => {
  let resolveOld!: (value: { blocks: { annotation_id: number; status: string; start_sentence_ordinal: number; end_sentence_ordinal: number }[] }) => void;
  api.getEvidenceAdjudication.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
  const user = userEvent.setup();
  const { rerender } = render(<EvidenceAdjudicationPanel projectId={1} documents={[]} assignments={[assignment]} allowSoloGold />);
  await user.click(screen.getByRole("button", { name: "Load comparison" }));
  rerender(<EvidenceAdjudicationPanel projectId={2} documents={[]} assignments={[{ ...assignment, project_id: 2, document_id: 42 }]} allowSoloGold />);
  await act(async () => resolveOld({ blocks: [{ annotation_id: 4, status: "draft", start_sentence_ordinal: 0, end_sentence_ordinal: 2 }] }));
  expect(screen.queryByText("Sentences 0-2")).toBeNull();
  expect(screen.getByRole<HTMLButtonElement>("button", { name: "Load comparison" }).disabled).toBe(false);
});
