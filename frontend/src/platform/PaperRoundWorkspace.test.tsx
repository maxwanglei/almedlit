// @vitest-environment jsdom
import { cleanup, fireEvent, render as renderComponent, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrowserRouter } from "react-router-dom";
import type { ReactElement } from "react";

import { addRoundDecision, loadRoundWork, submitRoundDecisions } from "./api";
import { createPaperEntityTaskVersionPayload } from "./paperTaskContracts";
import PaperRoundWorkspace, { createPaperAnnotationIds, createPaperRoundAnnotationApi, paperRoundAnnotationType, projectPaperRoundWork, selectPaperRoundContexts, type LoadedPaperRound } from "./PaperRoundWorkspace";
import type { AnnotationDecision, DatasetItem, RoundWorkContext } from "./types";
import type { AnnotationWorkbench, Document } from "@/types/api";

vi.mock("./api", () => ({ addRoundDecision: vi.fn(), loadRoundWork: vi.fn(), submitRoundDecisions: vi.fn() }));
vi.mock("./RoundWorkbench", () => ({ default: () => <div>Generic annotation editor</div> }));
vi.mock("@/pages/AnnotatorWorkspace", () => ({ default: ({ workbench, documents }: { workbench: AnnotationWorkbench | null; documents: Document[] }) => <div>
  <p>Shared paper editor</p>{workbench?.tasks.map((task) => <span key={task.id}>{task.display_name}</span>)}
  {documents.map((document) => <p key={document.id}>{document.text}</p>)}
</div> }));

function render(element: ReactElement) { return renderComponent(<BrowserRouter>{element}</BrowserRouter>); }

const text = "Tucatinib treats cancer.";
function fixture(entity = false): LoadedPaperRound {
  const roundId = entity ? 202 : 101;
  const taskId = entity ? 2 : 1;
  const taskVersionId = entity ? 22 : 11;
  const context: RoundWorkContext = {
    project: { id: 9, name: "Tucatinib" }, task: { id: taskId, key: entity ? "ner" : "relevance", name: entity ? "Drug names" : "Paper relevance" }, cycle: null, guideline: null,
    round: { id: roundId, project_id: 9, name: entity ? "NER round" : "Classification round", sequence: 1, dataset_version_id: 7, task_version_id: taskVersionId, assistance_policy: "blind", feedback_available: false, status: "open", opened_at: null, closed_at: null },
    task_version: entity ? {
      ...createPaperEntityTaskVersionPayload(9, taskId, ["DRUG", "CHEMICAL"]),
      id: taskVersionId, version_number: 1, content_hash: "ner-v1",
    } : {
      id: taskVersionId, project_id: 9, task_definition_id: taskId, version_number: 1, task_kind: "classification", content_hash: "classification-v1",
      input_schema: { type: "object", required: ["text"], properties: { text: { type: "string" } } },
      output_schema: { type: "string", enum: ["Relevant", "Not relevant"] }, label_rules: { values: ["Relevant", "Not relevant"] }, annotation_ui: { preset: "classification" }, metrics: [], trainer_compatibility: [],
    },
  };
  const datasetItems: DatasetItem[] = [1, 2].map((id) => ({ id: 700 + id, project_id: 9, dataset_version_id: 7, stable_key: `paper:${id}`, group_key: `paper:${id}`, content_hash: `paper-${id}`, payload: { document_id: 30 + id, title: `Pinned paper ${id}`, text: id === 1 ? text : "A second saved paper.", external_id: `12345${id}` } }));
  return { context, work: { decisions: [], submissions: [], datasetItems,
    roundItems: datasetItems.map((item, index) => ({ id: roundId * 10 + index, project_id: 9, annotation_round_id: roundId, dataset_item_id: item.id, selection_rank: null, selection_score: null, selection_reason: {} })) } };
}

function harness(initial = [fixture(), fixture(true)]) {
  let rounds = initial;
  const annotationId = createPaperAnnotationIds();
  const api = createPaperRoundAnnotationApi({ getRounds: () => rounds, getVisibleRounds: () => rounds,
    setRounds: (next) => { rounds = next; }, userId: 5, annotatorId: "researcher", annotationId });
  return { api, rounds: () => rounds, projection: () => projectPaperRoundWork(rounds, 5, "researcher", annotationId) };
}

beforeEach(() => {
  vi.clearAllMocks();
  window.history.replaceState(null, "", "/my-work/rounds/101");
  let decisionId = 100;
  vi.mocked(addRoundDecision).mockImplementation(async (projectId, payload) => ({ id: ++decisionId, project_id: projectId, round_item_id: payload.roundItemId,
    supersedes_decision_id: payload.supersedesDecisionId, output: payload.output, decision_kind: payload.decisionKind,
    is_initial_checkpoint: payload.isInitialCheckpoint, rationale: payload.rationale, annotator_user_id: 5, content_hash: `decision-${decisionId}` }));
  vi.mocked(submitRoundDecisions).mockImplementation(async (projectId, roundId, decisionIds) => ({ id: 501, project_id: projectId, annotation_round_id: roundId, decision_ids: decisionIds, annotator_user_id: 5, sequence: 1, content_hash: "submission", submitted_at: "2026-09-18T00:00:00Z" }));
});
afterEach(cleanup);

describe("paper round selection and projection", () => {
  it("pins the active round, selects newest siblings per task, and excludes other sources and assisted rounds", () => {
    const active = fixture().context;
    const ner = fixture(true).context;
    const newerClassification = { ...active, round: { ...active.round, id: 105, sequence: 5 } };
    const newerNer = { ...ner, round: { ...ner.round, id: 205, sequence: 5 } };
    const otherSource = { ...ner, round: { ...ner.round, id: 209, dataset_version_id: 8, sequence: 9 } };
    const assisted = { ...ner, round: { ...ner.round, id: 210, assistance_policy: "immediate_suggestions" as const, sequence: 10 } };
    expect(selectPaperRoundContexts(active, [active, ner, newerClassification, newerNer, otherSource, assisted]).map((item) => item.round.id)).toEqual([101, 205]);
    expect(paperRoundAnnotationType({ ...ner, task_version: { ...ner.task_version, task_kind: "token_labeling" } })).toBeNull();
  });

  it("shows both tasks over identical pinned paper text without changing source document identities", () => {
    const projected = harness().projection();
    expect(projected.documents.map((document) => document.id)).toEqual([31, 32]);
    expect(projected.documents[0]).toMatchObject({ title: "Pinned paper 1", text, sentences: [[0, text.length]], metadata_: { dataset_item_id: 701, dataset_version_id: 7 } });
    expect(projected.tasks.map((task) => [task.id, task.annotation_type])).toEqual([[11, "doc_label"], [22, "entity"]]);
    expect(projected.assignments.map((assignment) => assignment.id)).toEqual([1010, 1011, 2020, 2021]);
  });

  it("rejects cross-version and conflicting snapshots instead of substituting live documents", () => {
    const source = fixture();
    source.work.datasetItems[0].dataset_version_id = 8;
    expect(() => harness([source]).projection()).toThrow(/saved paper collection/);
    const classification = fixture();
    const ner = fixture(true);
    ner.work.datasetItems[0].payload.text = "Changed after snapshot";
    expect(() => harness([classification, ner]).projection()).toThrow(/different saved versions/);
  });
});

describe("canonical mutations through the paper editor adapter", () => {
  it("replaces one classification decision atomically and keeps a selected label selected", async () => {
    const subject = harness();
    await subject.api.setDocumentLabel(31, "Relevant");
    await subject.api.setDocumentLabel(31, "Relevant");
    expect(addRoundDecision).toHaveBeenCalledTimes(1);
    await subject.api.setDocumentLabel(31, "Not relevant");
    expect(addRoundDecision).toHaveBeenLastCalledWith(9, expect.objectContaining({ roundItemId: 1010, output: "Not relevant", supersedesDecisionId: 101 }));
    expect(subject.projection().annotations).toHaveLength(1);
    expect(subject.projection().annotations[0]).toMatchObject({ document_id: 31, label: "Not relevant", attributes: { round_id: 101, dataset_item_id: 701, decision_id: 102 } });
  });

  it("serializes concurrent NER edits and saves the whole entity output with its predecessor", async () => {
    const subject = harness();
    const [first, second] = await Promise.all([
      subject.api.createAnnotation({ project_id: 9, document_id: 31, annotation_type: "entity", label: "DRUG", start_offset: 0, end_offset: 9 }),
      subject.api.createAnnotation({ project_id: 9, document_id: 31, annotation_type: "entity", label: "DRUG", start_offset: 10, end_offset: 16 }),
    ]);
    expect(subject.projection().annotations.filter((annotation) => annotation.annotation_type === "entity")).toHaveLength(2);
    expect(addRoundDecision).toHaveBeenLastCalledWith(9, expect.objectContaining({ roundItemId: 2020, supersedesDecisionId: 101, output: { entities: [{ start: 0, end: 9, label: "DRUG" }, { start: 10, end: 16, label: "DRUG" }] } }));
    await subject.api.updateAnnotation(first.id, { label: "CHEMICAL" });
    await subject.api.deleteAnnotation(second.id);
    expect(addRoundDecision).toHaveBeenLastCalledWith(9, expect.objectContaining({ supersedesDecisionId: 103, output: { entities: [{ start: 0, end: 9, label: "CHEMICAL" }] } }));
  });

  it("finishes exactly one paper/task and keeps other tasks and papers unfinished", async () => {
    const subject = harness();
    await subject.api.setDocumentLabel(31, "Relevant");
    await subject.api.setDocumentLabel(32, "Not relevant");
    await subject.api.createSubmission(9, 31, { assignment_id: 1010 });
    expect(submitRoundDecisions).toHaveBeenCalledWith(9, 101, [101]);
    expect(subject.projection().assignments.map((assignment) => [assignment.id, assignment.status])).toEqual([[1010, "submitted"], [1011, "in_progress"], [2020, "assigned"], [2021, "assigned"]]);
    await expect(subject.api.setDocumentLabel(31, "Not relevant")).rejects.toThrow(/Reopen/);
    await subject.api.createAnnotation({ project_id: 9, document_id: 31, annotation_type: "entity", label: "DRUG", start_offset: 0, end_offset: 9 });
    expect(subject.projection().assignments.find((assignment) => assignment.id === 2020)?.status).toBe("in_progress");
  });

  it("reopens by appending the same submitted output as a new draft without rewriting history", async () => {
    const subject = harness();
    await subject.api.setDocumentLabel(31, "Relevant");
    await subject.api.createSubmission(9, 31, { assignment_id: 1010 });
    const originalSubmission = structuredClone(subject.rounds()[0].work.submissions[0]);
    const reopened = await subject.api.reopenPersonalTaskAssignment(9, 1010);
    expect(reopened.status).toBe("in_progress");
    expect(addRoundDecision).toHaveBeenLastCalledWith(9, expect.objectContaining({ output: "Relevant", supersedesDecisionId: 101, roundItemId: 1010 }));
    expect(subject.rounds()[0].work.submissions[0]).toEqual(originalSubmission);
    expect(subject.projection().annotations[0].status).toBe("draft");
    expect(submitRoundDecisions).toHaveBeenCalledTimes(1);
  });

  it("records a deliberately empty entity decision but requires a classification label", async () => {
    const subject = harness();
    await expect(subject.api.createSubmission(9, 31, { assignment_id: 1010 })).rejects.toThrow(/classification label/);
    const result = await subject.api.createSubmission(9, 31, { assignment_id: 2020 });
    expect(addRoundDecision).toHaveBeenCalledWith(9, expect.objectContaining({ roundItemId: 2020, output: { entities: [] } }));
    expect(submitRoundDecisions).toHaveBeenCalledWith(9, 202, [101]);
    expect(result.annotation_count).toBe(0);
  });

  it("rejects wrong project, wrong paper, and offsets outside the original text", async () => {
    const subject = harness();
    await expect(subject.api.createAnnotation({ project_id: 99, document_id: 31, annotation_type: "entity", label: "DRUG", start_offset: 0, end_offset: 9 })).rejects.toThrow(/another project/);
    await expect(subject.api.createAnnotation({ project_id: 9, document_id: 31, annotation_type: "entity", label: "DRUG", start_offset: 0, end_offset: 999 })).rejects.toThrow(/valid span/);
    await expect(subject.api.createSubmission(9, 32, { assignment_id: 1010 })).rejects.toThrow(/does not match/);
    expect(addRoundDecision).not.toHaveBeenCalled();
    expect(submitRoundDecisions).not.toHaveBeenCalled();
  });

  it("projects only this annotator's decisions, even when another annotator submitted later", () => {
    const source = fixture();
    source.work.decisions = [
      { id: 1, round_item_id: 1010, annotator_user_id: 5, output: "Relevant", supersedes_decision_id: null },
      { id: 2, round_item_id: 1010, annotator_user_id: 6, output: "Not relevant", supersedes_decision_id: null },
    ] as AnnotationDecision[];
    expect(harness([source]).projection().annotations.map((annotation) => annotation.label)).toEqual(["Relevant"]);
  });
});

describe("paper workspace source loading", () => {
  it("persists the selected classification task in the URL and restores it when navigating back", async () => {
    const classification = fixture();
    const alternative = fixture();
    alternative.context = { ...alternative.context, task: { id: 3, key: "safety", name: "Safety relevance" },
      task_version: { ...alternative.context.task_version, id: 33, task_definition_id: 3 },
      round: { ...alternative.context.round, id: 303, task_version_id: 33 } };
    alternative.work.roundItems = alternative.work.roundItems.map((item, index) => ({ ...item, id: 3030 + index, annotation_round_id: 303 }));
    vi.mocked(loadRoundWork).mockImplementation(async (_, round) => round.id === 101 ? classification.work : alternative.work);
    render(<PaperRoundWorkspace context={classification.context} contexts={[classification.context, alternative.context]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} />);
    const selector = await screen.findByRole("combobox", { name: "Classification task" });
    expect((selector as HTMLSelectElement).value).toBe("11");
    fireEvent.change(selector, { target: { value: "33" } });
    await waitFor(() => expect(new URLSearchParams(window.location.search).get("classificationTaskVersionId")).toBe("33"));
    expect(screen.getByText("Safety relevance")).toBeTruthy();
    window.history.back();
    await waitFor(() => expect((screen.getByRole("combobox", { name: "Classification task" }) as HTMLSelectElement).value).toBe("11"));
    expect(screen.queryByRole("button", { name: "Manage tasks" })).toBeNull();
    expect(loadRoundWork).toHaveBeenCalledTimes(2);
  });

  it("loads both compatible task rounds into the same existing paper editor", async () => {
    const classification = fixture();
    const ner = fixture(true);
    vi.mocked(loadRoundWork).mockImplementation(async (_, round) => round.id === 101 ? classification.work : ner.work);
    render(<PaperRoundWorkspace context={classification.context} contexts={[classification.context, ner.context]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} />);
    await screen.findByText("Shared paper editor");
    expect(screen.getByText("Paper relevance")).toBeTruthy();
    expect(screen.getByText("Drug names")).toBeTruthy();
    expect(screen.getByText(text)).toBeTruthy();
    expect(loadRoundWork).toHaveBeenCalledTimes(2);
  });

  it("keeps external classification sources in the original generic editor", async () => {
    const source = fixture();
    delete source.work.datasetItems[0].payload.document_id;
    vi.mocked(loadRoundWork).mockResolvedValue(source.work);
    render(<PaperRoundWorkspace context={source.context} contexts={[source.context]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} />);
    await screen.findByText("Generic annotation editor");
    expect(screen.queryByText("Shared paper editor")).toBeNull();
  });

  it("does not hide cross-project records behind a fallback editor", async () => {
    const source = fixture();
    delete source.work.datasetItems[0].payload.document_id;
    source.work.datasetItems[1].project_id = 99;
    vi.mocked(loadRoundWork).mockResolvedValue(source.work);
    render(<PaperRoundWorkspace context={source.context} contexts={[source.context]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("different project"));
    expect(screen.queryByText("Generic annotation editor")).toBeNull();
    expect(screen.queryByText("Shared paper editor")).toBeNull();
  });

  it("reports network failures without falling back to another annotation system", async () => {
    const source = fixture();
    vi.mocked(loadRoundWork).mockRejectedValue(new Error("Could not reach the annotation API"));
    render(<PaperRoundWorkspace context={source.context} contexts={[source.context]} currentUserId={5} annotatorId="researcher" onClose={vi.fn()} onOpenProjectTasks={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("Could not reach the annotation API"));
    expect(screen.queryByText("Generic annotation editor")).toBeNull();
    expect(screen.queryByText("Shared paper editor")).toBeNull();
  });
});
