// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  Annotation,
  Document,
  DocumentStructureRead,
  EvidenceReviewCoverage,
  EvidenceTarget,
  ProjectTask,
  TaskAssignment,
} from "@/types/api";

import EvidenceBlockCanvas from "./EvidenceBlockCanvas";

const api = vi.hoisted(() => ({
  createAnnotation: vi.fn(),
  deleteAnnotation: vi.fn(),
  getAnnotation: vi.fn(),
  getDocumentStructure: vi.fn(),
  getEvidenceReviewCoverage: vi.fn(),
  listEvidenceTargets: vi.fn(),
  listEvidenceCommands: vi.fn(),
  listInferencePredictions: vi.fn(),
  listInferenceRuns: vi.fn(),
  markEvidenceReviewed: vi.fn(),
  mergeEvidenceBlocks: vi.fn(),
  reopenEvidenceReview: vi.fn(),
  redoEvidenceCommand: vi.fn(),
  reviewInferencePrediction: vi.fn(),
  splitEvidenceBlock: vi.fn(),
  updateAnnotation: vi.fn(),
  undoEvidenceCommand: vi.fn(),
}));

vi.mock("@/api/client", () => api);

const DOCUMENT: Document = {
  id: 41,
  project_id: 1,
  external_id: null,
  title: "Evidence document",
  text: "Alpha evidence.\n\nBeta context. Gamma result.",
  source: "test",
  metadata_: {},
  sentences: [],
  active_structure_version_id: 201,
};

const STRUCTURE: DocumentStructureRead = {
  document_id: 41,
  active_structure_version_id: 201,
  structure_version: {
    id: 201,
    document_id: 41,
    version: 1,
    segmenter_name: "builtin",
    segmenter_version: "1",
    source_hash: "hash",
    text_length: DOCUMENT.text.length,
    status: "ready",
    created_at: "2026-07-15T12:00:00Z",
  },
  range: {
    start_ordinal: 0,
    end_ordinal: 3,
    total_sentences: 3,
    has_more: false,
  },
  sections: [
    {
      id: 501,
      ordinal: 0,
      title: "Results",
      path: ["Results"],
      kind: "jats",
      start_offset: 0,
      end_offset: DOCUMENT.text.length,
      locator: null,
    },
  ],
  paragraphs: [
    {
      id: 601,
      section_id: 501,
      ordinal: 0,
      section_ordinal: 0,
      start_offset: 0,
      end_offset: 15,
      locator: null,
    },
    {
      id: 602,
      section_id: 501,
      ordinal: 1,
      section_ordinal: 1,
      start_offset: 17,
      end_offset: DOCUMENT.text.length,
      locator: null,
    },
  ],
  sentences: [
    {
      id: 701,
      section_id: 501,
      paragraph_id: 601,
      ordinal: 0,
      paragraph_ordinal: 0,
      start_offset: 0,
      end_offset: 15,
      text: "Alpha evidence.",
    },
    {
      id: 702,
      section_id: 501,
      paragraph_id: 602,
      ordinal: 1,
      paragraph_ordinal: 0,
      start_offset: 17,
      end_offset: 30,
      text: "Beta context.",
    },
    {
      id: 703,
      section_id: 501,
      paragraph_id: 602,
      ordinal: 2,
      paragraph_ordinal: 1,
      start_offset: 31,
      end_offset: DOCUMENT.text.length,
      text: "Gamma result.",
    },
  ],
};

const TARGETS: EvidenceTarget[] = [
  {
    id: 21,
    project_id: 1,
    task_id: 11,
    key: "benefit",
    name: "Benefit",
    description: null,
    is_active: true,
    active_version_id: 101,
    versions: [
      {
        id: 101,
        target_id: 21,
        version_number: 1,
        text: "Does treatment improve outcomes?",
        guidance: "Use complete supporting sentences.",
        inclusion_guidance: "Include outcome statements.",
        exclusion_guidance: "Exclude background only.",
        metadata_: {},
        created_by_user_id: 1,
        created_at: "2026-07-15T12:00:00Z",
        updated_at: "2026-07-15T12:00:00Z",
      },
    ],
    created_by_user_id: 1,
    created_at: "2026-07-15T12:00:00Z",
    updated_at: "2026-07-15T12:00:00Z",
  },
];

const TASK: ProjectTask = {
  id: 11,
  project_id: 1,
  annotation_type: "evidence_block",
  display_name: "Evidence blocks",
  description: null,
  enabled: true,
  sort_order: 0,
  labels: [{ name: "support", color: "#4d6e5b", description: "Supporting evidence" }],
  settings: {},
};

const ASSIGNMENT: TaskAssignment = {
  id: 51,
  project_id: 1,
  task_id: 11,
  document_id: 41,
  assignee_user_id: 2,
  annotator_id: "alice",
  status: "in_progress",
  assigned_by_user_id: 1,
  assigned_by: "manager",
  notes: null,
  metadata_: {},
  target_version_id: 101,
  structure_version_id: 201,
  guideline_version_id: 301,
  assignment_scope_key: "target:101",
};

const COVERAGE: EvidenceReviewCoverage = {
  project_id: 1,
  document_id: 41,
  target_version_id: 101,
  structure_version_id: 201,
  guideline_version_id: 301,
  reviewer_user_id: 2,
  intervals: [],
  events: [],
  fully_reviewed: false,
};

function annotation(id = 901): Annotation {
  return {
    id,
    project_id: 1,
    document_id: 41,
    annotation_type: "evidence_block",
    label: "evidence_block",
    start_offset: 0,
    end_offset: 30,
    text_span: DOCUMENT.text.slice(0, 30),
    source: "human",
    status: "draft",
    confidence: null,
    annotator_user_id: 2,
    annotator_id: "alice",
    model_checkpoint_id: null,
    guideline_version_id: 301,
    structure_version_id: 201,
    head_annotation_id: null,
    tail_annotation_id: null,
    evidence: {},
    attributes: {},
    evidence_block: {
      annotation_id: id,
      structure_version_id: 201,
      target_version_id: 101,
      start_sentence_id: 701,
      end_sentence_id: 702,
      start_sentence_ordinal: 0,
      end_sentence_ordinal: 1,
      start_offset: 0,
      end_offset: 30,
      labels: ["support"],
      note: "Useful",
      boundary_policy: "sentence",
      revision: 1,
      locked: false,
    },
    created_at: "2026-07-15T12:00:00Z",
    updated_at: "2026-07-15T12:00:00Z",
  };
}

function renderCanvas(
  assignment: TaskAssignment | TaskAssignment[] = ASSIGNMENT,
  annotations: Annotation[] = [],
  onActiveAssignmentChange = vi.fn(),
  allowAssignmentlessEditing = false,
) {
  return render(
    <EvidenceBlockCanvas
      projectId={1}
      document={DOCUMENT}
      task={TASK}
      assignments={Array.isArray(assignment) ? assignment : [assignment]}
      annotations={annotations}
      annotatorId="alice"
      allowAssignmentlessEditing={allowAssignmentlessEditing}
      busy={false}
      setBusy={vi.fn()}
      setError={vi.fn()}
      onAnnotationsChanged={vi.fn()}
      onRefreshAnnotations={vi.fn().mockResolvedValue(undefined)}
      onActiveAssignmentChange={onActiveAssignmentChange}
    />,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  api.listEvidenceTargets.mockResolvedValue(TARGETS);
  api.listInferenceRuns.mockResolvedValue([]);
  api.listEvidenceCommands.mockResolvedValue([]);
  api.getDocumentStructure.mockResolvedValue(STRUCTURE);
  api.getEvidenceReviewCoverage.mockResolvedValue(COVERAGE);
  api.listInferencePredictions.mockResolvedValue([]);
  api.createAnnotation.mockResolvedValue(annotation());
  api.markEvidenceReviewed.mockResolvedValue({ ...COVERAGE, fully_reviewed: true });
  vi.spyOn(window, "confirm").mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("EvidenceBlockCanvas", () => {
  it("creates from sentence IDs after keyboard-safe boundary stepping", async () => {
    const user = userEvent.setup();
    renderCanvas();

    const second = await screen.findByRole(
      "button",
      { name: /Sentence 2, unreviewed/i },
      { timeout: 5000 },
    );
    await user.click(second);
    await user.click(screen.getByRole("button", { name: "Expand start" }));
    expect(
      screen
        .getByRole("button", { name: /Sentence 1, unreviewed, selected/i })
        .getAttribute("aria-pressed"),
    ).toBe("true");
    await user.click(screen.getByRole("button", { name: "support" }));
    await user.type(screen.getByLabelText("Note"), "Useful");
    await user.click(screen.getByRole("button", { name: "Create block" }));

    await waitFor(() => expect(api.createAnnotation).toHaveBeenCalledTimes(1));
    expect(api.createAnnotation).toHaveBeenCalledWith(
      expect.objectContaining({
        guideline_version_id: 301,
        evidence_block: expect.objectContaining({
          structure_version_id: 201,
          target_version_id: 101,
          start_sentence_id: 701,
          end_sentence_id: 702,
          labels: ["support"],
          note: "Useful",
        }),
      }),
    );
    expect(api.createAnnotation.mock.calls[0][0]).not.toHaveProperty("start_offset");
  });

  it("separates reviewed-region selection and preserves text-input undo", async () => {
    const user = userEvent.setup();
    api.listEvidenceCommands.mockResolvedValue([
      {
        command_group_key: "command-1",
        operation: "create",
        status: "applied",
        project_id: 1,
        document_id: 41,
        target_version_id: 101,
        structure_version_id: 201,
        guideline_version_id: 301,
        actor_user_id: 2,
        created_at: "2026-07-15T12:00:00Z",
      },
    ]);
    renderCanvas();

    const note = await screen.findByLabelText("Note");
    await user.click(note);
    await user.type(note, "draft");
    fireEvent.keyDown(note, { key: "z", ctrlKey: true });
    expect(api.undoEvidenceCommand).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Review region" }));
    await user.click(screen.getByRole("button", { name: /Sentence 1, unreviewed/i }));
    fireEvent.pointerDown(screen.getByRole("button", { name: /Sentence 3, unreviewed/i }), {
      shiftKey: true,
    });
    await user.click(screen.getByRole("button", { name: "Mark reviewed" }));

    await waitFor(() => expect(api.markEvidenceReviewed).toHaveBeenCalledTimes(1));
    expect(api.markEvidenceReviewed).toHaveBeenCalledWith(
      1,
      41,
      expect.objectContaining({
        guideline_version_id: 301,
        start_sentence_id: 701,
        end_sentence_id: 703,
      }),
    );
  });

  it("renders a submitted assignment as read-only", async () => {
    renderCanvas({ ...ASSIGNMENT, status: "submitted" });
    await screen.findByRole("button", { name: /Sentence 1, unreviewed/i });
    expect(screen.getByText("submitted assignment · read-only")).toBeTruthy();
    expect((screen.getByRole("button", { name: "Create block" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText("Note") as HTMLInputElement).disabled).toBe(true);
  });

  it("uses personal task and setup language for evidence review", async () => {
    api.listEvidenceTargets.mockResolvedValue([]);
    renderCanvas({ ...ASSIGNMENT, status: "submitted" }, [], vi.fn(), true);

    expect(
      await screen.findByText("Finished task · read-only"),
    ).toBeTruthy();
    expect(
      screen.getByRole("option", { name: "No targets configured" }),
    ).toBeTruthy();
    expect(
      screen.getByText(
        "Configure and activate an evidence target in Project Setup before annotation.",
      ),
    ).toBeTruthy();
    expect(screen.queryByText(/manager must assign/i)).toBeNull();
  });

  it("prefers a newly provisioned mutable round over the submitted round", async () => {
    const onActiveAssignmentChange = vi.fn();
    const submitted = { ...ASSIGNMENT, status: "submitted" as const };
    const replacement = { ...ASSIGNMENT, id: 52, status: "assigned" as const };

    renderCanvas([submitted, replacement], [], onActiveAssignmentChange);

    await screen.findByRole("button", { name: /Sentence 1, unreviewed/i });
    await waitFor(() =>
      expect(onActiveAssignmentChange).toHaveBeenCalledWith(
        expect.objectContaining({ id: replacement.id, status: "assigned" }),
      ),
    );
    expect(screen.queryByText(/assignment · read-only/i)).toBeNull();
  });

  it("prefers a mutable assignment on a newer target version", async () => {
    const onActiveAssignmentChange = vi.fn();
    const submitted = { ...ASSIGNMENT, status: "submitted" as const };
    const replacement = {
      ...ASSIGNMENT,
      id: 52,
      status: "assigned" as const,
      target_version_id: 102,
      assignment_scope_key: "target:102",
    };
    api.listEvidenceTargets.mockResolvedValue([
      {
        ...TARGETS[0],
        active_version_id: 102,
        versions: [
          ...TARGETS[0].versions,
          {
            ...TARGETS[0].versions[0],
            id: 102,
            version_number: 2,
            text: "Does the updated treatment improve outcomes?",
          },
        ],
      },
    ]);

    renderCanvas([submitted, replacement], [], onActiveAssignmentChange);

    const targetSelect = await screen.findByLabelText("Evidence target");
    await waitFor(() => expect((targetSelect as HTMLSelectElement).value).toBe("102"));
    await waitFor(() =>
      expect(onActiveAssignmentChange).toHaveBeenCalledWith(
        expect.objectContaining({ id: replacement.id, target_version_id: 102 }),
      ),
    );
    expect(screen.queryByText(/assignment · read-only/i)).toBeNull();
  });

  it("keeps evidence editing read-only without an evidence assignment in a team workspace", async () => {
    const unrelatedAssignment = { ...ASSIGNMENT, id: 60, task_id: 99 };

    renderCanvas(unrelatedAssignment);

    await screen.findByRole("button", { name: /Sentence 1, unreviewed/i });
    expect(screen.getByText("closed assignment · read-only")).toBeTruthy();
    expect((screen.getByLabelText("Note") as HTMLInputElement).disabled).toBe(true);
  });

  it("allows explicit assignmentless evidence editing in a personal workspace", async () => {
    const unrelatedAssignment = { ...ASSIGNMENT, id: 60, task_id: 99 };

    renderCanvas(unrelatedAssignment, [], vi.fn(), true);

    await screen.findByRole("button", { name: /Sentence 1, unreviewed/i });
    expect(screen.queryByText(/assignment · read-only/i)).toBeNull();
    expect((screen.getByLabelText("Note") as HTMLInputElement).disabled).toBe(false);
  });

  it("does not render another annotator's human evidence blocks", async () => {
    const aliceBlock = annotation(901);
    const bobBlock: Annotation = {
      ...annotation(902),
      annotator_user_id: 8,
      annotator_id: "bob",
      evidence_block: {
        ...annotation(902).evidence_block!,
        annotation_id: 902,
        note: "Bob only",
      },
    };

    renderCanvas(ASSIGNMENT, [aliceBlock, bobBlock]);

    expect(await screen.findByText("1 blocks")).toBeTruthy();
    expect(screen.getByText("Useful")).toBeTruthy();
    expect(screen.queryByText("Bob only")).toBeNull();
  });
});
