// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  Annotation,
  Document,
  DocumentStructureRead,
  EvidenceReviewCoverage,
  EvidenceCandidatePrediction,
  EvidenceTarget,
  InferenceRun,
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

const INFERENCE_RUN: InferenceRun = {
  id: 7,
  project_id: 1,
  corpus_snapshot_id: 1,
  checkpoint_id: 5,
  compute_profile_id: 1,
  name: "Evidence model",
  target_version_ids: [101],
  window_config: {},
  decoder_config: {},
  status: "succeeded",
  idempotency_key: "evidence-run",
  external_job_id: null,
  diagnostics_artifact_id: null,
  started_at: null,
  completed_at: null,
  failure_reason: null,
  metrics: {},
};

function prediction(id: number): EvidenceCandidatePrediction {
  return {
    id,
    project_id: 1,
    run_id: 7,
    checkpoint_id: 5,
    document_id: DOCUMENT.id,
    structure_version_id: 201,
    target_version_id: 101,
    start_sentence_id: 701,
    end_sentence_id: 702,
    start_sentence_ordinal: 0,
    end_sentence_ordinal: 1,
    start_char: 0,
    end_char: 30,
    block_confidence: 0.87,
    boundary_confidence: {},
    uncertainty: 0.13,
    decoder_version: "evidence-block-decoder-v1",
    source_window_ids: [21],
    status: "pending",
    review_status: "pending",
    diagnostics_artifact_id: null,
    metadata_: {},
    reviews: [],
  };
}

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
  it("preserves run discovery errors after prediction refresh effects settle", async () => {
    let rejectRuns!: (error: Error) => void;
    api.listInferenceRuns.mockImplementationOnce(
      () => new Promise<InferenceRun[]>((_resolve, reject) => { rejectRuns = reject; }),
    );
    renderCanvas();
    await screen.findByLabelText("Note");

    await act(async () => rejectRuns(new Error("Unable to discover inference runs.")));

    expect(screen.getByText("Unable to discover inference runs.")).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(screen.getByText("Unable to discover inference runs.")).toBeTruthy();
    expect(api.listInferencePredictions).not.toHaveBeenCalled();
  });

  it("loads every prediction page before displaying candidates, preserving scope filters", async () => {
    const predictions = Array.from({ length: 102 }, (_, index) => prediction(1000 + index));
    predictions[0].structure_version_id = 200;
    let resolveSecondPage!: (items: EvidenceCandidatePrediction[]) => void;
    api.listInferenceRuns.mockResolvedValue([INFERENCE_RUN]);
    api.listInferencePredictions
      .mockResolvedValueOnce(predictions.slice(0, 100))
      .mockImplementationOnce(() => new Promise<EvidenceCandidatePrediction[]>((resolve) => {
        resolveSecondPage = resolve;
      }));
    renderCanvas();

    await waitFor(() => expect(api.listInferencePredictions).toHaveBeenCalledTimes(2));
    expect(screen.getByText("Loading prediction candidates…")).toBeTruthy();
    expect(screen.queryByText("No candidates for this run, document, and target.")).toBeNull();
    expect(screen.queryByLabelText("Accept prediction 1001")).toBeNull();
    await act(async () => resolveSecondPage(predictions.slice(100)));

    expect(screen.getByText("101 model candidates")).toBeTruthy();
    expect(screen.getByLabelText("Accept prediction 1101")).toBeTruthy();
    expect(screen.queryByLabelText("Accept prediction 1000")).toBeNull();
    expect(api.listInferencePredictions).toHaveBeenNthCalledWith(1, 7, {
      documentId: 41, targetVersionId: 101, limit: 100, offset: 0,
    });
    expect(api.listInferencePredictions).toHaveBeenNthCalledWith(2, 7, {
      documentId: 41, targetVersionId: 101, limit: 100, offset: 100,
    });
  });

  it("reports a later prediction page failure and retries the complete list", async () => {
    const firstPage = Array.from({ length: 100 }, (_, index) => prediction(1000 + index));
    api.listInferenceRuns.mockResolvedValue([INFERENCE_RUN]);
    api.listInferencePredictions
      .mockResolvedValueOnce(firstPage)
      .mockRejectedValueOnce(new Error("Prediction service unavailable"));
    renderCanvas();

    await screen.findByText("Prediction service unavailable");
    expect(screen.queryByLabelText("Accept prediction 1000")).toBeNull();
    api.listInferencePredictions
      .mockResolvedValueOnce(firstPage)
      .mockResolvedValueOnce([prediction(1100)]);
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));

    await screen.findByText("101 model candidates");
    expect(screen.queryByText("Prediction service unavailable")).toBeNull();
    expect(screen.getByLabelText("Accept prediction 1100")).toBeTruthy();
  });

  it("discards a pending page and stops pagination when the document scope changes", async () => {
    const firstPage = Array.from({ length: 100 }, (_, index) => prediction(1000 + index));
    let resolveOldPage!: (items: EvidenceCandidatePrediction[]) => void;
    api.listInferenceRuns.mockResolvedValue([INFERENCE_RUN]);
    api.listInferencePredictions
      .mockResolvedValueOnce(firstPage)
      .mockImplementationOnce(() => new Promise<EvidenceCandidatePrediction[]>((resolve) => {
        resolveOldPage = resolve;
      }))
      .mockResolvedValueOnce([{ ...prediction(2000), document_id: 42, structure_version_id: 202 }]);
    const props = {
      projectId: 1,
      task: TASK,
      annotations: [],
      annotatorId: "alice",
      busy: false,
      setBusy: vi.fn(),
      setError: vi.fn(),
      onAnnotationsChanged: vi.fn(),
      onRefreshAnnotations: vi.fn().mockResolvedValue(undefined),
    };
    const { rerender } = render(
      <EvidenceBlockCanvas {...props} document={DOCUMENT} assignments={[ASSIGNMENT]} />,
    );
    await waitFor(() => expect(api.listInferencePredictions).toHaveBeenCalledTimes(2));

    rerender(
      <EvidenceBlockCanvas
        {...props}
        document={{ ...DOCUMENT, id: 42, active_structure_version_id: 202 }}
        assignments={[{ ...ASSIGNMENT, document_id: 42, structure_version_id: 202 }]}
      />,
    );
    await screen.findByRole("button", { name: "Accept prediction 2000" });
    await act(async () => resolveOldPage(
      Array.from({ length: 100 }, (_, index) => prediction(1100 + index)),
    ));

    expect(screen.getByText("1 model candidates")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Accept prediction 2000" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Accept prediction 1100" })).toBeNull();
    expect(api.listInferencePredictions).toHaveBeenCalledTimes(3);
    expect(api.listInferencePredictions).toHaveBeenLastCalledWith(7, {
      documentId: 42, targetVersionId: 101, limit: 100, offset: 0,
    });
  });

  it("clears loading and ignores pending pages when the target has no compatible run", async () => {
    let resolvePage!: (items: EvidenceCandidatePrediction[]) => void;
    api.listEvidenceTargets.mockResolvedValue([
      {
        ...TARGETS[0],
        versions: [
          ...TARGETS[0].versions,
          { ...TARGETS[0].versions[0], id: 102, version_number: 2 },
        ],
      },
    ]);
    api.listInferenceRuns.mockResolvedValue([INFERENCE_RUN]);
    api.listInferencePredictions.mockImplementationOnce(
      () => new Promise<EvidenceCandidatePrediction[]>((resolve) => { resolvePage = resolve; }),
    );
    renderCanvas([
      ASSIGNMENT,
      { ...ASSIGNMENT, id: 52, target_version_id: 102, assignment_scope_key: "target:102" },
    ]);
    await screen.findByText("Loading prediction candidates…");

    fireEvent.change(screen.getByLabelText("Evidence target"), { target: { value: "102" } });
    await screen.findByRole("option", { name: "No compatible runs" });
    expect(screen.queryByText("Loading prediction candidates…")).toBeNull();
    await act(async () => resolvePage([prediction(1000)]));

    expect(screen.queryByRole("button", { name: "Accept prediction 1000" })).toBeNull();
    expect(api.listInferencePredictions).toHaveBeenCalledTimes(1);
  });

  it("reports repeated prediction pages instead of looping or displaying an incomplete list", async () => {
    api.listInferenceRuns.mockResolvedValue([INFERENCE_RUN]);
    api.listInferencePredictions.mockResolvedValue(
      Array.from({ length: 100 }, (_, index) => prediction(1000 + index)),
    );
    renderCanvas();

    await screen.findByText("Prediction loading did not advance. Please refresh to try again.");
    expect(screen.queryByRole("button", { name: "Accept prediction 1000" })).toBeNull();
    expect(api.listInferencePredictions).toHaveBeenCalledTimes(2);
  });

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
