import type { Page, Route } from "@playwright/test";

export type MockWorkspaceRole = "annotator" | "manager" | "personal";

export interface CapturedApiRequest {
  method: string;
  pathname: string;
  search: string;
  body: unknown;
}

export interface EvidenceApiMockState {
  requests: CapturedApiRequest[];
  unhandledRequests: string[];
  annotations: Array<Record<string, unknown>>;
  predictions: Array<Record<string, unknown>>;
  comparisonBlocks: Array<Record<string, unknown>>;
  assignments: Array<Record<string, unknown>>;
  coverage: Record<string, unknown>;
}

const timestamp = "2026-07-15T12:00:00Z";
const paragraphTexts = [
  ["Trial enrolled adults at three centers."],
  ["Participants received the study drug.", "Pain scores improved by week four."],
  ["No serious adverse events occurred.", "Follow-up lasted twelve weeks."],
];
const canonicalText = paragraphTexts.map((sentences) => sentences.join(" ")).join("\n\n");

interface SentenceFixture {
  id: number;
  section_id: number;
  paragraph_id: number;
  ordinal: number;
  paragraph_ordinal: number;
  start_offset: number;
  end_offset: number;
  text: string;
}

const paragraphs: Array<Record<string, unknown>> = [];
const sentences: SentenceFixture[] = [];
let sentenceOrdinal = 0;
let cursor = 0;
paragraphTexts.forEach((paragraphSentences, paragraphOrdinal) => {
  const paragraphId = 801 + paragraphOrdinal;
  const paragraphStart = cursor;
  paragraphSentences.forEach((text, paragraphSentenceOrdinal) => {
    if (paragraphSentenceOrdinal > 0) {
      cursor += 1;
    }
    sentences.push({
      id: 1001 + sentenceOrdinal,
      section_id: 601,
      paragraph_id: paragraphId,
      ordinal: sentenceOrdinal,
      paragraph_ordinal: paragraphSentenceOrdinal,
      start_offset: cursor,
      end_offset: cursor + text.length,
      text,
    });
    cursor += text.length;
    sentenceOrdinal += 1;
  });
  paragraphs.push({
    id: paragraphId,
    section_id: 601,
    ordinal: paragraphOrdinal,
    section_ordinal: paragraphOrdinal,
    start_offset: paragraphStart,
    end_offset: cursor,
    locator: { paragraph: paragraphOrdinal + 1 },
  });
  if (paragraphOrdinal < paragraphTexts.length - 1) {
    cursor += 2;
  }
});

const evidenceTask = {
  id: 11,
  project_id: 1,
  annotation_type: "evidence_block",
  display_name: "Evidence blocks",
  description: "Find complete-sentence evidence blocks.",
  enabled: true,
  sort_order: 0,
  labels: [],
  settings: {
    schema_version: "1",
    active_target_ids: [21],
    sentence_boundaries: true,
    multi_paragraph_allowed: true,
    cross_section_allowed: false,
    same_target_overlap_allowed: false,
    adjacency_allowed: true,
    soft_token_warning: 3072,
    model_context_tokens: 4096,
    window_overlap_tokens: 512,
    review_scope: "document",
  },
};

const project = {
  id: 1,
  name: "Evidence Review Demo",
  description: "Deterministic Playwright fixture",
  annotation_schema: { labels: { evidence_block: [] } },
  annotation_validation_mode: "strict",
  tasks: [evidenceTask],
  settings: {},
  workspace_id: 1,
};

const documentFixture = {
  id: 41,
  project_id: 1,
  external_id: "PMID-E2E-41",
  title: "Three-paragraph clinical trial",
  text: canonicalText,
  source: "test",
  metadata_: {},
  sentences: sentences.map((sentence) => [sentence.start_offset, sentence.end_offset]),
  active_structure_version_id: 201,
};

const target = {
  id: 21,
  project_id: 1,
  task_id: 11,
  key: "pain-benefit",
  name: "Pain benefit",
  description: null,
  is_active: true,
  active_version_id: 101,
  versions: [
    {
      id: 101,
      target_id: 21,
      version_number: 1,
      text: "Evidence that the intervention improves pain outcomes.",
      guidance: "Include benefit statements with complete sentence context.",
      inclusion_guidance: null,
      exclusion_guidance: null,
      metadata_: {},
      created_by_user_id: 1,
      created_at: timestamp,
      updated_at: timestamp,
    },
  ],
  created_by_user_id: 1,
  created_at: timestamp,
  updated_at: timestamp,
};

function assignment(id: number, userId: number, annotatorId: string) {
  return {
    id,
    project_id: 1,
    task_id: 11,
    document_id: 41,
    assignee_user_id: userId,
    annotator_id: annotatorId,
    status: "in_progress",
    assigned_by_user_id: 1,
    assigned_by: "manager",
    notes: null,
    metadata_: {},
    target_version_id: 101,
    structure_version_id: 201,
    guideline_version_id: 301,
    assignment_scope_key: `target:101:user:${userId}`,
  };
}

function sentenceById(sentenceId: number): SentenceFixture {
  const sentence = sentences.find((item) => item.id === sentenceId);
  if (!sentence) {
    throw new Error(`Unknown fixture sentence ${sentenceId}`);
  }
  return sentence;
}

function evidenceAnnotation(
  id: number,
  startSentenceId: number,
  endSentenceId: number,
  options: {
    status?: string;
    annotatorId?: string;
    annotatorUserId?: number;
    note?: string | null;
  } = {},
): Record<string, unknown> {
  const start = sentenceById(startSentenceId);
  const end = sentenceById(endSentenceId);
  return {
    id,
    project_id: 1,
    document_id: 41,
    annotation_type: "evidence_block",
    label: "evidence_block",
    start_offset: start.start_offset,
    end_offset: end.end_offset,
    text_span: canonicalText.slice(start.start_offset, end.end_offset),
    source: "human",
    status: options.status ?? "draft",
    confidence: null,
    annotator_user_id: options.annotatorUserId ?? 2,
    annotator_id: options.annotatorId ?? "alice",
    model_checkpoint_id: null,
    guideline_version_id: 301,
    structure_version_id: 201,
    head_annotation_id: null,
    tail_annotation_id: null,
    evidence: {},
    attributes: {},
    revision: 1,
    evidence_block: {
      annotation_id: id,
      structure_version_id: 201,
      target_version_id: 101,
      start_sentence_id: start.id,
      end_sentence_id: end.id,
      start_sentence_ordinal: start.ordinal,
      end_sentence_ordinal: end.ordinal,
      start_offset: start.start_offset,
      end_offset: end.end_offset,
      labels: [],
      note: options.note ?? null,
      boundary_policy: "sentence",
      revision: 1,
      locked: options.status === "gold",
      last_command_group_key: options.status === "gold" ? null : `command-${id}`,
      created_at: timestamp,
      updated_at: timestamp,
    },
    created_at: timestamp,
    updated_at: timestamp,
  };
}

function prediction(
  id: number,
  startSentenceId: number,
  endSentenceId: number,
): Record<string, unknown> {
  const start = sentenceById(startSentenceId);
  const end = sentenceById(endSentenceId);
  return {
    id,
    project_id: 1,
    run_id: 401,
    checkpoint_id: 301,
    document_id: 41,
    structure_version_id: 201,
    target_version_id: 101,
    start_sentence_id: start.id,
    end_sentence_id: end.id,
    start_sentence_ordinal: start.ordinal,
    end_sentence_ordinal: end.ordinal,
    start_char: start.start_offset,
    end_char: end.end_offset,
    block_confidence: id === 501 ? 0.91 : 0.62,
    boundary_confidence: { start: 0.9, end: 0.88 },
    uncertainty: id === 501 ? 0.08 : 0.31,
    decoder_version: "evidence-block-decoder-v1",
    source_window_ids: [901, 902],
    status: "pending",
    review_status: "pending",
    diagnostics_artifact_id: 701,
    metadata_: {},
    reviews: [],
  };
}

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function requestBody(route: Route): unknown {
  const raw = route.request().postData();
  if (!raw) {
    return null;
  }
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return raw;
  }
}

export async function installEvidenceApiMock(
  page: Page,
  role: MockWorkspaceRole,
): Promise<EvidenceApiMockState> {
  const assignments =
    role === "manager" ? [assignment(51, 2, "alice"), assignment(52, 3, "bob")] : [assignment(51, 2, "alice")];
  const state: EvidenceApiMockState = {
    requests: [],
    unhandledRequests: [],
    annotations: [],
    predictions: [prediction(501, 1005, 1005), prediction(502, 1003, 1003)],
    comparisonBlocks: [
      {
        annotation_id: 701,
        annotator_user_id: 2,
        annotator_id: "alice",
        status: "accepted",
        start_sentence_id: 1001,
        end_sentence_id: 1003,
        start_sentence_ordinal: 0,
        end_sentence_ordinal: 2,
        labels: [],
        note: "Alice boundary",
      },
      {
        annotation_id: 702,
        annotator_user_id: 3,
        annotator_id: "bob",
        status: "accepted",
        start_sentence_id: 1002,
        end_sentence_id: 1004,
        start_sentence_ordinal: 1,
        end_sentence_ordinal: 3,
        labels: [],
        note: "Bob boundary",
      },
    ],
    assignments,
    coverage: {
      project_id: 1,
      document_id: 41,
      target_version_id: 101,
      structure_version_id: 201,
      guideline_version_id: 301,
      reviewer_user_id: 2,
      intervals: [],
      events: [],
      fully_reviewed: false,
    },
  };
  let nextAnnotationId = 800;
  let nextReviewId = 900;

  await page.route(/^https?:\/\/[^/]+\/api\//, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method();
    const body = requestBody(route);
    state.requests.push({ method, pathname: url.pathname, search: url.search, body });

    if (method === "GET" && url.pathname === "/api/auth/me") {
      const isManager = role === "manager";
      const isPersonal = role === "personal";
      return json(route, {
        user: {
          id: isManager ? 1 : 2,
          username: isManager ? "manager" : "alice",
          display_name: isManager ? "Project Manager" : "Alice Annotator",
          is_active: true,
          is_superuser: false,
        },
        memberships: [
          {
            workspace_id: 1,
            workspace_name: isPersonal ? "Alice's Workspace" : "Evidence Team",
            workspace_kind: isPersonal ? "individual" : "team",
            role: isPersonal ? "admin" : role,
          },
        ],
      });
    }
    if (method === "GET" && url.pathname === "/api/workspaces/1/capabilities") {
      return json(route, {
        preset: "full_loop",
        overrides: [],
        effective: ["annotation", "lineage", "export", "training", "inference"],
        blocked: {},
      });
    }
    if (method === "GET" && url.pathname === "/api/projects") {
      return json(route, [project]);
    }
    if (method === "GET" && url.pathname === "/api/projects/my-work") {
      return json(route, [project]);
    }
    if (
      method === "GET" &&
      url.pathname === "/api/workspaces/1/my-work/rounds"
    ) {
      return json(route, []);
    }
    if (method === "GET" && url.pathname === "/api/projects/1") {
      return json(route, project);
    }
    if (method === "GET" && url.pathname === "/api/documents") {
      return json(route, [documentFixture]);
    }
    if (method === "GET" && url.pathname === "/api/projects/1/assignments") {
      return json(route, state.assignments);
    }
    if (method === "GET" && url.pathname === "/api/projects/1/progress") {
      return json(route, {
        project_id: 1,
        total: state.assignments.length,
        by_status: { in_progress: state.assignments.length },
        by_task: [
          {
            task_id: 11,
            annotation_type: "evidence_block",
            display_name: "Evidence blocks",
            total: state.assignments.length,
            by_status: { in_progress: state.assignments.length },
          },
        ],
        by_document: [
          {
            document_id: 41,
            total: state.assignments.length,
            by_status: { in_progress: state.assignments.length },
          },
        ],
        by_annotator: state.assignments.map((item) => ({
          assignee_user_id: item.assignee_user_id,
          annotator_id: item.annotator_id,
          total: 1,
          by_status: { in_progress: 1 },
        })),
      });
    }
    if (method === "GET" && url.pathname === "/api/guidelines") {
      return json(route, [
        {
          id: 301,
          project_id: 1,
          version_label: "v1",
          markdown: "# Evidence guidance",
          author_id: "manager",
          status: "active",
        },
      ]);
    }
    if (method === "GET" && url.pathname === "/api/projects/1/evidence-targets") {
      return json(route, [target]);
    }
    if (
      method === "GET" &&
      [
        "/api/projects/1/corpus-snapshots",
        "/api/projects/1/annotation-sets",
        "/api/projects/1/exports",
        "/api/projects/1/artifact-packages",
        "/api/projects/1/base-models",
      ].includes(url.pathname)
    ) {
      return json(route, []);
    }
    if (method === "GET" && url.pathname === "/api/export-formats") {
      return json(route, []);
    }
    if (
      method === "POST" &&
      url.pathname === "/api/projects/1/import/pubmed/preview"
    ) {
      return json(route, {
        items: [
          {
            pmid: "29860986",
            title:
              "Immunotherapy and checkpoint inhibition in a long clinical study title",
            journal: "Clinical Research",
            year: "2024",
            pmcid: "PMC100",
            status: "full_text",
            has_full_text: true,
            has_abstract: true,
          },
          {
            pmid: "29717446",
            title:
              "Detection of cancer outcomes in another deliberately long article title",
            journal: "Oncology",
            year: "2023",
            pmcid: null,
            status: "abstract_only",
            has_full_text: false,
            has_abstract: true,
          },
        ],
      });
    }
    if (method === "GET" && url.pathname === "/api/workspaces/1/members") {
      return json(route, [
        {
          id: 1,
          workspace_id: 1,
          user_id: 1,
          username: "manager",
          display_name: "Project Manager",
          email: null,
          is_active: true,
          role: "manager",
        },
        {
          id: 2,
          workspace_id: 1,
          user_id: 2,
          username: "alice",
          display_name: "Alice Annotator",
          email: null,
          is_active: true,
          role: "annotator",
        },
        {
          id: 3,
          workspace_id: 1,
          user_id: 3,
          username: "bob",
          display_name: "Bob Annotator",
          email: null,
          is_active: true,
          role: "annotator",
        },
      ]);
    }
    if (method === "GET" && url.pathname === "/api/workspaces/1/join-requests") {
      return json(route, []);
    }
    if (method === "GET" && url.pathname === "/api/annotation-workbench/documents/41") {
      return json(route, {
        project,
        document: documentFixture,
        active_guideline: {
          id: 301,
          project_id: 1,
          version_label: "v1",
          markdown: "# Evidence guidance",
          author_id: "manager",
          status: "active",
        },
        guideline_versions_by_id: {
          301: {
            id: 301,
            project_id: 1,
            version_label: "v1",
            markdown: "# Evidence guidance",
            author_id: "manager",
            status: "active",
          },
        },
        tasks: [
          {
            ...evidenceTask,
            annotation_type_spec: {
              name: "evidence_block",
              requires_span: false,
              requires_head_tail: false,
              description: "Complete-sentence evidence ranges.",
              selection_mode: "sentence_range",
              renderer_key: "evidence_block_v1",
              relation_endpoint_allowed: false,
              handler_key: "evidence_block_v1",
            },
          },
        ],
        annotation_type_specs: [],
        annotations: state.annotations,
        assignments: state.assignments,
        correction_locked_annotation_ids: [],
      });
    }
    if (method === "GET" && url.pathname === "/api/documents/41/structure") {
      return json(route, {
        document_id: 41,
        active_structure_version_id: 201,
        structure_version: {
          id: 201,
          document_id: 41,
          version: 1,
          segmenter_name: "builtin",
          segmenter_version: "evidence-v1",
          source_hash: "playwright-source-hash",
          text_length: canonicalText.length,
          status: "ready",
          created_at: timestamp,
        },
        range: {
          start_ordinal: 0,
          end_ordinal: sentences.length,
          total_sentences: sentences.length,
          has_more: false,
        },
        sections: [
          {
            id: 601,
            ordinal: 0,
            title: "Results",
            path: ["Results"],
            kind: "body",
            start_offset: 0,
            end_offset: canonicalText.length,
            locator: { section: "results" },
          },
        ],
        paragraphs,
        sentences,
      });
    }
    if (
      method === "GET" &&
      url.pathname === "/api/projects/1/documents/41/evidence-review-coverage"
    ) {
      return json(route, state.coverage);
    }
    if (
      method === "POST" &&
      url.pathname === "/api/projects/1/documents/41/evidence-review-coverage/mark-reviewed"
    ) {
      const payload = body as { start_sentence_id: number; end_sentence_id: number };
      const start = sentenceById(payload.start_sentence_id);
      const end = sentenceById(payload.end_sentence_id);
      state.coverage = {
        ...state.coverage,
        intervals: [
          {
            id: 1,
            start_sentence_ordinal: start.ordinal,
            end_sentence_ordinal: end.ordinal,
            start_sentence_id: start.id,
            end_sentence_id: end.id,
            created_at: timestamp,
            updated_at: timestamp,
          },
        ],
        events: [
          {
            id: 1,
            action: "mark_reviewed",
            start_sentence_ordinal: start.ordinal,
            end_sentence_ordinal: end.ordinal,
            actor_user_id: 2,
            start_sentence_id: start.id,
            end_sentence_id: end.id,
            reason: null,
            metadata_: {},
            created_at: timestamp,
          },
        ],
        fully_reviewed: start.ordinal === 0 && end.ordinal === sentences.length - 1,
      };
      return json(route, state.coverage);
    }
    if (method === "GET" && url.pathname === "/api/annotations/evidence-blocks/commands") {
      return json(route, []);
    }
    if (method === "GET" && url.pathname === "/api/projects/1/inference/runs") {
      return json(route, []);
    }
    if (method === "GET" && url.pathname === "/api/inference/runs/401/predictions") {
      return json(route, state.predictions);
    }
    if (method === "POST" && url.pathname === "/api/annotations") {
      const payload = body as {
        annotator_id: string;
        evidence_block: {
          start_sentence_id: number;
          end_sentence_id: number;
          note?: string | null;
        };
      };
      const created = evidenceAnnotation(
        nextAnnotationId++,
        payload.evidence_block.start_sentence_id,
        payload.evidence_block.end_sentence_id,
        { annotatorId: payload.annotator_id, note: payload.evidence_block.note },
      );
      state.annotations.unshift(created);
      return json(route, created, 201);
    }
    const annotationMatch = url.pathname.match(/^\/api\/annotations\/(\d+)$/);
    if (method === "GET" && annotationMatch) {
      const found = state.annotations.find(
        (annotation) => annotation.id === Number(annotationMatch[1]),
      );
      return found ? json(route, found) : json(route, { detail: "Annotation not found" }, 404);
    }
    const reviewMatch = url.pathname.match(/^\/api\/inference\/predictions\/(\d+)\/review$/);
    if (method === "POST" && reviewMatch) {
      const predictionId = Number(reviewMatch[1]);
      const selectedPrediction = state.predictions.find(
        (candidate) => candidate.id === predictionId,
      );
      if (!selectedPrediction) {
        return json(route, { detail: "Prediction not found" }, 404);
      }
      const payload = body as {
        action: "accept" | "modify" | "reject";
        start_sentence_id?: number;
        end_sentence_id?: number;
        note?: string | null;
      };
      let resultingAnnotationId: number | null = null;
      if (payload.action !== "reject") {
        resultingAnnotationId = nextAnnotationId++;
        state.annotations.unshift(
          evidenceAnnotation(
            resultingAnnotationId,
            payload.start_sentence_id ?? (selectedPrediction.start_sentence_id as number),
            payload.end_sentence_id ?? (selectedPrediction.end_sentence_id as number),
            { note: payload.note },
          ),
        );
      }
      const review = {
        id: nextReviewId++,
        prediction_id: predictionId,
        reviewer_user_id: 2,
        action: payload.action,
        revision: 1,
        resulting_annotation_id: resultingAnnotationId,
        selected_boundaries:
          payload.action === "modify"
            ? {
                start_sentence_id: payload.start_sentence_id,
                end_sentence_id: payload.end_sentence_id,
              }
            : null,
        note: payload.note ?? null,
        metadata_: {},
        created_at: timestamp,
      };
      selectedPrediction.status =
        payload.action === "accept"
          ? "accepted"
          : payload.action === "modify"
            ? "modified"
            : "rejected";
      selectedPrediction.review_status = selectedPrediction.status;
      (selectedPrediction.reviews as Array<Record<string, unknown>>).push(review);
      return json(route, review, 201);
    }
    if (
      method === "POST" &&
      url.pathname === "/api/projects/1/documents/41/submissions"
    ) {
      state.assignments = state.assignments.map((item) => ({ ...item, status: "submitted" }));
      return json(
        route,
        {
          id: 1,
          project_id: 1,
          document_id: 41,
          annotator_user_id: 2,
          annotator_id: "alice",
          assignment_id: (body as { assignment_id?: number }).assignment_id ?? null,
          kind: "submit",
          file_name: "submission.json",
          created_at: timestamp,
        },
        201,
      );
    }
    if (
      method === "GET" &&
      url.pathname === "/api/projects/1/documents/41/evidence-adjudication"
    ) {
      return json(route, {
        project_id: 1,
        document_id: 41,
        target_version_id: 101,
        structure_version_id: 201,
        guideline_version_id: 301,
        blocks: state.comparisonBlocks,
      });
    }
    if (
      method === "POST" &&
      url.pathname === "/api/projects/1/documents/41/evidence-adjudication"
    ) {
      const payload = body as { note?: string | null };
      const gold = evidenceAnnotation(999, 1001, 1004, {
        status: "gold",
        annotatorId: "manager",
        annotatorUserId: 1,
        note: payload.note,
      });
      state.annotations.unshift(gold);
      state.comparisonBlocks.push({
        annotation_id: 999,
        annotator_user_id: 1,
        annotator_id: "manager",
        status: "gold",
        start_sentence_id: 1001,
        end_sentence_id: 1004,
        start_sentence_ordinal: 0,
        end_sentence_ordinal: 3,
        labels: [],
        note: payload.note ?? null,
      });
      return json(route, gold, 201);
    }

    const requestKey = `${method} ${url.pathname}${url.search}`;
    state.unhandledRequests.push(requestKey);
    return json(route, { detail: `Unhandled Playwright API fixture: ${requestKey}` }, 501);
  });

  return state;
}
