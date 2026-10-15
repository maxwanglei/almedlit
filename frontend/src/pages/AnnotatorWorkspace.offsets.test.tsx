// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { BrowserRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AnnotatorWorkspace, { type AnnotationMutationAdapter } from "@/pages/AnnotatorWorkspace";
import type {
  Annotation,
  AnnotationCreate,
  AnnotationWorkbench,
  AnnotationWorkbenchTask,
  Document,
  Project,
  TaskAssignment,
} from "@/types/api";

const mocks = vi.hoisted(() => ({
  createAnnotation: vi.fn(),
  createSubmission: vi.fn(),
  deleteAnnotation: vi.fn(),
  reopenPersonalTaskAssignment: vi.fn(),
  updateAnnotation: vi.fn(),
}));

vi.mock("@/api/client", () => mocks);
vi.mock("@/features/evidence-block/EvidenceBlockCanvas", () => ({
  default: () => <div>Evidence canvas</div>,
}));

const entityTask: AnnotationWorkbenchTask = {
  id: 1,
  project_id: 1,
  annotation_type: "entity",
  display_name: "Entity labels",
  description: null,
  enabled: true,
  sort_order: 0,
  labels: [{ name: "Disease", color: "#4d6e5b", description: null }],
  settings: {},
  annotation_type_spec: {
    name: "entity",
    requires_span: true,
    requires_head_tail: false,
    description: "Entity spans",
    selection_mode: "character_span",
    renderer_key: "legacy",
    relation_endpoint_allowed: true,
    handler_key: "generic",
  },
};

const sentenceTask: AnnotationWorkbenchTask = {
  ...entityTask,
  id: 2,
  annotation_type: "sentence_label",
  display_name: "Sentence labels",
  labels: [{ name: "Finding", color: "#4d6e5b", description: null }],
  annotation_type_spec: {
    ...entityTask.annotation_type_spec,
    name: "sentence_label",
    selection_mode: "sentence",
  },
};

function annotation(overrides: Partial<Annotation> = {}): Annotation {
  return {
    id: 1,
    project_id: 1,
    document_id: 1,
    annotation_type: "entity",
    label: "Disease",
    start_offset: 4,
    end_offset: 11,
    text_span: "disease",
    source: "human",
    status: "draft",
    confidence: null,
    annotator_user_id: null,
    annotator_id: "max",
    model_checkpoint_id: null,
    guideline_version_id: null,
    structure_version_id: 1,
    head_annotation_id: null,
    tail_annotation_id: null,
    evidence: {},
    attributes: {},
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function taskAssignment(task: AnnotationWorkbenchTask): TaskAssignment {
  return {
    id: task.id,
    project_id: 1,
    task_id: task.id,
    document_id: 1,
    assignee_user_id: 1,
    annotator_id: "max",
    status: "assigned",
    assigned_by_user_id: 1,
    assigned_by: "max",
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: 1,
    guideline_version_id: null,
    assignment_scope_key: `offset-test-${task.id}`,
  };
}

interface HarnessProps {
  text: string;
  sentences: number[][];
  annotations?: Annotation[];
  task?: AnnotationWorkbenchTask;
  canonical?: boolean;
}

function Harness({ text, sentences, annotations = [], task = entityTask, canonical = false }: HarnessProps): React.ReactElement {
  const documentItem: Document = {
    id: 1,
    project_id: 1,
    external_id: "PMID-1",
    title: "Unicode offset regression",
    text,
    source: "test",
    metadata_: {},
    sentences,
    active_structure_version_id: 1,
  };
  const project: Project = {
    id: 1,
    name: "Unicode annotations",
    description: null,
    annotation_schema: { labels: { [task.annotation_type]: task.labels } },
    annotation_validation_mode: "strict",
    tasks: [task],
    settings: {},
    workspace_id: 1,
  };
  const assignments = canonical ? [taskAssignment(task)] : [];
  const [workbench, setWorkbench] = useState<AnnotationWorkbench | null>({
    project,
    document: documentItem,
    active_guideline: null,
    guideline_versions_by_id: {},
    tasks: [task],
    annotation_type_specs: [task.annotation_type_spec],
    annotations,
    assignments,
    correction_locked_annotation_ids: [],
  });
  const [busy, setBusy] = useState(false);
  const annotationApi: AnnotationMutationAdapter = {
    ...mocks,
    setDocumentLabel: vi.fn(),
  };

  return <BrowserRouter><AnnotatorWorkspace
    projects={[project]} selectedProject={project} selectedProjectId={1}
    setSelectedProjectId={vi.fn()} documents={[documentItem]} assignments={assignments}
    annotatorId="max" projectProgress={null} selectedDocumentId={1} setSelectedDocumentId={vi.fn()}
    workbench={workbench} setWorkbench={setWorkbench} busy={busy} setBusy={setBusy} setError={vi.fn()}
    refreshProjectData={async () => undefined} refreshWorkbench={async () => undefined}
    allowAssignmentlessSubmit annotationApi={canonical ? annotationApi : undefined} defaultView="annotate"
  /></BrowserRouter>;
}

function sentenceBodies(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(".aw-sentence-body"));
}

function documentText(element: HTMLElement): string | null {
  const copy = element.cloneNode(true) as HTMLElement;
  copy.querySelectorAll(".aw-span-badge").forEach((badge) => badge.remove());
  return copy.textContent;
}

function selectText(body: HTMLElement, target: string): void {
  const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
  let node = walker.nextNode();
  while (node && !node.textContent?.includes(target)) node = walker.nextNode();
  if (!node) throw new Error(`Unable to select ${target} in ${body.textContent}`);
  const start = node.textContent!.indexOf(target);
  const range = document.createRange();
  range.setStart(node, start);
  range.setEnd(node, start + target.length);
  Object.defineProperty(range, "getBoundingClientRect", {
    value: () => ({ left: 10, top: 10, width: 100, height: 20 }),
  });
  window.getSelection()!.removeAllRanges();
  window.getSelection()!.addRange(range);
  fireEvent.mouseUp(body);
}

beforeEach(() => {
  window.history.replaceState(null, "", "/annotator/workbench?view=annotate&document=1");
  vi.clearAllMocks();
  mocks.createAnnotation.mockImplementation(async (payload: AnnotationCreate) => annotation({
    ...payload,
    id: 10,
    start_offset: payload.start_offset ?? null,
    end_offset: payload.end_offset ?? null,
    text_span: payload.text_span ?? null,
    evidence_block: null,
  }));
  Object.defineProperty(globalThis, "ResizeObserver", {
    configurable: true,
    value: class {
      observe(): void {}
      disconnect(): void {}
    },
  });
});

afterEach(() => {
  cleanup();
  window.getSelection()?.removeAllRanges();
});

describe("AnnotatorWorkspace legacy API Unicode offsets", () => {
  it("renders code-point sentence boundaries and existing highlights after math italic and emoji", () => {
    const first = "𝑃 😀 improved.";
    const second = "Next 𝑛 😀 disease.";
    const text = `${first} ${second}`;
    const start = Array.from(`${first} Next 𝑛 😀 `).length;
    const { container } = render(<Harness text={text}
      sentences={[[0, Array.from(first).length], [Array.from(first).length + 1, Array.from(text).length]]}
      annotations={[annotation({ start_offset: start, end_offset: start + 7 })]} />);

    expect(sentenceBodies(container).map(documentText)).toEqual([first, second]);
    expect(within(sentenceBodies(container)[1]).getByRole("button", { name: "Disease: disease" })).toBeTruthy();
  });

  it.each([
    { selectedText: "disease", start: 4, end: 11 },
    { selectedText: "𝑃 😀", start: 0, end: 3 },
  ])("saves DOM selection '$selectedText' as code points and renders the returned span", async ({ selectedText, start, end }) => {
    const text = "𝑃 😀 disease responded.";
    const { container } = render(<Harness text={text} sentences={[[0, Array.from(text).length]]} />);
    selectText(sentenceBodies(container)[0], selectedText);
    fireEvent.click(screen.getByRole("button", { name: /^Disease\s*D$/ }));

    await waitFor(() => expect(mocks.createAnnotation).toHaveBeenCalledWith(expect.objectContaining({
      start_offset: start,
      end_offset: end,
      text_span: selectedText,
    })));
    expect(within(sentenceBodies(container)[0]).getByRole("button", { name: `Disease: ${selectedText}` })).toBeTruthy();
  });

  it("saves a second-sentence selection after a highlighted Unicode prefix without counting its badge", async () => {
    const first = "𝑃 😀 improved.";
    const second = "Next 𝑛 😀 disease.";
    const text = `${first} ${second}`;
    const highlightStart = Array.from(`${first} Next `).length;
    const start = Array.from(`${first} Next 𝑛 😀 `).length;
    const { container } = render(<Harness text={text}
      sentences={[[0, Array.from(first).length], [Array.from(first).length + 1, Array.from(text).length]]}
      annotations={[annotation({ start_offset: highlightStart, end_offset: highlightStart + 3, text_span: "𝑛 😀" })]} />);
    const body = sentenceBodies(container)[1];
    selectText(body, "iseas");
    fireEvent.click(screen.getByRole("button", { name: /^Disease\s*D$/ }));

    await waitFor(() => expect(mocks.createAnnotation).toHaveBeenCalledWith(expect.objectContaining({
      start_offset: start,
      end_offset: start + 7,
      text_span: "disease",
    })));
    expect(documentText(body)).toBe(second);
    expect(within(body).getByRole("button", { name: "Disease: disease" })).toBeTruthy();
  });

  it("uses code-point offsets when falling back to a loaded annotation's document text", () => {
    const text = "𝑃 😀 disease responded.";
    render(<Harness text={text} sentences={[[0, Array.from(text).length]]}
      annotations={[annotation({ text_span: null })]} />);

    const inspector = screen.getByRole("tabpanel", { name: /^annotations\s*1$/ });
    expect(within(inspector).getByRole("button", { name: "Disease: disease" })).toBeTruthy();
  });

  it.each([true, false])("saves the entire Unicode sentence with matching code-point bounds (segmented: %s)", async (segmented) => {
    const prefix = segmented ? "Lead 𝑛 😀. " : "";
    const sentence = "𝑃 😀 response.";
    const text = prefix + sentence;
    const start = Array.from(prefix).length;
    const { container } = render(<Harness text={text} task={sentenceTask}
      sentences={segmented ? [[0, start - 1], [start, Array.from(text).length]] : []} />);
    const bodies = sentenceBodies(container);
    const sentenceSection = bodies[bodies.length - 1].closest<HTMLElement>(".aw-sentence")!;
    fireEvent.click(within(sentenceSection).getByRole("button", { name: "+ label" }));
    fireEvent.click(screen.getByRole("button", { name: /^Finding\s*1$/ }));

    await waitFor(() => expect(mocks.createAnnotation).toHaveBeenCalledWith(expect.objectContaining({
      annotation_type: "sentence_label",
      start_offset: start,
      end_offset: Array.from(text).length,
      text_span: sentence,
    })));
    expect(within(sentenceSection).getByRole("button", { name: "Finding" })).toBeTruthy();
  });

  it("preserves ASCII selections and the whole-document fallback", async () => {
    const text = "A disease responded.";
    const { container } = render(<Harness text={text} sentences={[]} />);
    expect(sentenceBodies(container).map(documentText)).toEqual([text]);
    selectText(sentenceBodies(container)[0], "disease");
    fireEvent.click(screen.getByRole("button", { name: /^Disease\s*D$/ }));

    await waitFor(() => expect(mocks.createAnnotation).toHaveBeenCalledWith(expect.objectContaining({
      start_offset: 2,
      end_offset: 9,
      text_span: "disease",
    })));
  });

  it("preserves the canonical annotation adapter's UTF-16 sentence and selection offsets", async () => {
    const text = "𝑃 😀 disease responded.";
    const { container } = render(<Harness canonical text={text} sentences={[[0, text.length]]} />);
    expect(sentenceBodies(container).map(documentText)).toEqual([text]);
    selectText(sentenceBodies(container)[0], "disease");
    fireEvent.click(screen.getByRole("button", { name: /^Disease\s*D$/ }));

    await waitFor(() => expect(mocks.createAnnotation).toHaveBeenCalledWith(expect.objectContaining({
      start_offset: 6,
      end_offset: 13,
      text_span: "disease",
    })));
    expect(within(sentenceBodies(container)[0]).getByRole("button", { name: "Disease: disease" })).toBeTruthy();
  });
});
