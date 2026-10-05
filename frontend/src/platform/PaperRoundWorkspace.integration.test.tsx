// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { addRoundDecision, loadRoundWork, submitRoundDecisions, type RoundWorkData } from "./api";
import { createPaperEntityTaskVersionPayload } from "./paperTaskContracts";
import PaperRoundWorkspace from "./PaperRoundWorkspace";
import type { RoundWorkContext } from "./types";

vi.mock("./api", () => ({ addRoundDecision: vi.fn(), loadRoundWork: vi.fn(), submitRoundDecisions: vi.fn() }));
vi.mock("@/features/evidence-block/EvidenceBlockCanvas", () => ({ default: () => <div>Evidence canvas</div> }));
const legacy = vi.hoisted(() => ({ createAnnotation: vi.fn(), createSubmission: vi.fn(), deleteAnnotation: vi.fn(), reopenPersonalTaskAssignment: vi.fn(), updateAnnotation: vi.fn() }));
vi.mock("@/api/client", () => legacy);

function fixture() {
  const classification: RoundWorkContext = {
    project: { id: 9, name: "Tucatinib" }, task: { id: 1, key: "relevance", name: "Paper relevance" }, cycle: null, guideline: null,
    round: { id: 101, project_id: 9, name: "Classification round", sequence: 1, dataset_version_id: 7, task_version_id: 11, assistance_policy: "blind", feedback_available: false, status: "open", opened_at: null, closed_at: null },
    task_version: { id: 11, project_id: 9, task_definition_id: 1, version_number: 1, task_kind: "classification", content_hash: "classification-v1", input_schema: { type: "object", properties: { text: { type: "string" } } }, output_schema: { type: "string", enum: ["Relevant", "Not relevant"] }, label_rules: {}, annotation_ui: { preset: "classification" }, metrics: [], trainer_compatibility: [] },
  };
  const ner: RoundWorkContext = { ...classification, task: { id: 2, key: "ner", name: "Drug names" },
    round: { ...classification.round, id: 202, name: "NER round", task_version_id: 22 },
    task_version: { ...createPaperEntityTaskVersionPayload(9, 2, ["DRUG"]), id: 22, version_number: 1, content_hash: "ner-v1" } };
  const work = new Map<number, RoundWorkData>([classification, ner].map((context) => [context.round.id, {
    decisions: [], submissions: [],
    datasetItems: [{ id: 701, project_id: 9, dataset_version_id: 7, stable_key: "paper:31", group_key: "paper:31", content_hash: "paper-31", payload: { document_id: 31, title: "Pinned trial paper", text: "Tucatinib treats cancer.", external_id: "123456" } }],
    roundItems: [{ id: context.round.id * 10, project_id: 9, annotation_round_id: context.round.id, dataset_item_id: 701, selection_rank: null, selection_score: null, selection_reason: {} }],
  }]));
  vi.mocked(loadRoundWork).mockImplementation(async (_, round) => structuredClone(work.get(round.id)!));
  let decisionId = 100;
  let submissionId = 500;
  vi.mocked(addRoundDecision).mockImplementation(async (projectId, payload) => {
    const decision = { id: ++decisionId, project_id: projectId, round_item_id: payload.roundItemId, supersedes_decision_id: payload.supersedesDecisionId, output: payload.output, decision_kind: payload.decisionKind, is_initial_checkpoint: payload.isInitialCheckpoint, rationale: payload.rationale, annotator_user_id: 5, content_hash: `decision-${decisionId}` };
    work.get(payload.roundItemId / 10)!.decisions.push(decision);
    return decision;
  });
  vi.mocked(submitRoundDecisions).mockImplementation(async (projectId, roundId, decisionIds) => {
    const submission = { id: ++submissionId, project_id: projectId, annotation_round_id: roundId, decision_ids: decisionIds, annotator_user_id: 5, sequence: 1, content_hash: "submission", submitted_at: "2026-09-18T00:00:00Z" };
    work.get(roundId)!.submissions.push(submission);
    return submission;
  });
  return { classification, ner, work };
}

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/my-work/rounds/101?view=annotate&document=31&assignment=1010");
  Object.defineProperty(globalThis, "ResizeObserver", { configurable: true, value: class { observe(): void {} disconnect(): void {} } });
});
afterEach(cleanup);

it("uses the existing editor for both task panels, submits only classification, and leaves NER editable", async () => {
  const { classification, ner, work } = fixture();
  render(<BrowserRouter><PaperRoundWorkspace context={classification} contexts={[classification, ner]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} /></BrowserRouter>);
  expect(await screen.findByRole("heading", { name: "Pinned trial paper" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Paper relevance" })).toBeTruthy();
  expect(screen.getByRole("heading", { name: "Drug names" })).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: /^Relevant\s*Alt/ }));
  await waitFor(() => expect(screen.getByRole("button", { name: /^Relevant\s*OK/ }).getAttribute("aria-pressed")).toBe("true"));
  fireEvent.click(screen.getByRole("button", { name: /^Relevant\s*OK/ }));
  await waitFor(() => expect(addRoundDecision).toHaveBeenCalledTimes(1));
  fireEvent.click(screen.getByRole("button", { name: "Submit task for this paper" }));
  const dialog = screen.getByRole("dialog", { name: "Submit this paper task?" });
  expect(within(dialog).getByText("Paper relevance")).toBeTruthy();
  fireEvent.click(within(dialog).getByRole("button", { name: "Submit this paper task" }));
  await screen.findByText("Paper task submitted");
  expect(submitRoundDecisions).toHaveBeenCalledWith(9, 101, [101]);
  expect(work.get(202)!.submissions).toHaveLength(0);
  expect((screen.getByRole("button", { name: /^Relevant\s*OK/ }) as HTMLButtonElement).disabled).toBe(true);
  expect((screen.getByRole("button", { name: /^DRUG/ }) as HTMLButtonElement).disabled).toBe(false);
  for (const mutation of Object.values(legacy)) expect(mutation).not.toHaveBeenCalled();
});

it("reopens classification through a superseding draft while keeping the original submitted decision", async () => {
  const { classification, ner, work } = fixture();
  work.get(101)!.decisions.push({ id: 99, project_id: 9, round_item_id: 1010, output: "Relevant", supersedes_decision_id: null, decision_kind: "annotation", is_initial_checkpoint: false, rationale: null, annotator_user_id: 5, content_hash: "old" });
  work.get(101)!.submissions.push({ id: 499, project_id: 9, annotation_round_id: 101, decision_ids: [99], annotator_user_id: 5, sequence: 1, content_hash: "old-submit", submitted_at: "2026-09-17T00:00:00Z" });
  render(<BrowserRouter><PaperRoundWorkspace context={classification} contexts={[classification, ner]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} /></BrowserRouter>);
  fireEvent.click(await screen.findByRole("button", { name: "Edit this paper task" }));
  fireEvent.click(screen.getByRole("button", { name: "Reopen and edit" }));
  await waitFor(() => expect((screen.getByRole("button", { name: /^Relevant\s*OK/ }) as HTMLButtonElement).disabled).toBe(false));
  expect(addRoundDecision).toHaveBeenCalledWith(9, expect.objectContaining({ roundItemId: 1010, output: "Relevant", supersedesDecisionId: 99 }));
  expect(work.get(101)!.submissions[0].decision_ids).toEqual([99]);
  expect(submitRoundDecisions).not.toHaveBeenCalled();
  expect(legacy.reopenPersonalTaskAssignment).not.toHaveBeenCalled();
});
