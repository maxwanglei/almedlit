import { describe, expect, it } from "vitest";

import type { Annotation, TaskAssignment, TaskAssignmentStatus } from "@/types/api";

import {
  isAnnotationVisibleToAnnotator,
  isHumanAnnotationOwnedBy,
  isMutableAssignment,
  isTaskEditableForAssignment,
  nextMutableAssignment,
  personalAssignmentStatusLabel,
  personalDocumentAction,
  personalDocumentStatusLabel,
  preferredAssignment,
  preferredAssignmentId,
} from "./annotationWorkspacePolicy";

function assignment(id: number, status: TaskAssignmentStatus): TaskAssignment {
  return {
    id,
    project_id: 1,
    task_id: 2,
    document_id: 3,
    assignee_user_id: 4,
    annotator_id: "alice",
    status,
    assigned_by_user_id: 5,
    assigned_by: "manager",
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: 6,
    guideline_version_id: 7,
    assignment_scope_key: "document",
  };
}

function annotation(source: Annotation["source"], annotatorId: string | null): Annotation {
  return {
    id: 1,
    project_id: 1,
    document_id: 3,
    annotation_type: "entity",
    label: "finding",
    start_offset: 0,
    end_offset: 4,
    text_span: "test",
    source,
    status: "draft",
    confidence: null,
    annotator_user_id: 4,
    annotator_id: annotatorId,
    model_checkpoint_id: null,
    guideline_version_id: 7,
    structure_version_id: 6,
    head_annotation_id: null,
    tail_annotation_id: null,
    evidence: {},
    attributes: {},
    created_at: "2026-07-17T00:00:00Z",
    updated_at: "2026-07-17T00:00:00Z",
  };
}

describe("annotation workspace assignment policy", () => {
  it("moves a finalized selection to a newly provisioned mutable round", () => {
    const submitted = assignment(10, "submitted");
    const replacement = assignment(11, "assigned");

    expect(preferredAssignmentId([submitted, replacement], submitted.id)).toBe(replacement.id);
    expect(preferredAssignment([submitted, replacement])).toBe(replacement);
  });

  it("preserves the current mutable round and keeps finalized history selectable", () => {
    const first = assignment(10, "in_progress");
    const second = assignment(11, "assigned");
    const completed = assignment(12, "completed");
    const blocked = assignment(13, "blocked");
    const withdrawn = assignment(14, "withdrawn");

    expect(preferredAssignmentId([first, second], first.id)).toBe(first.id);
    expect(preferredAssignmentId([completed], completed.id)).toBe(completed.id);
    expect(isMutableAssignment(completed)).toBe(false);
    expect(isMutableAssignment(blocked)).toBe(true);
    expect(isMutableAssignment(withdrawn)).toBe(false);
  });

  it("binds editing to the selected assignment task", () => {
    const openEntity = assignment(10, "in_progress");
    openEntity.task_id = 2;
    const finalizedRelation = assignment(11, "submitted");
    finalizedRelation.task_id = 3;

    expect(isTaskEditableForAssignment(openEntity, openEntity.task_id, false, 2)).toBe(true);
    expect(isTaskEditableForAssignment(openEntity, finalizedRelation.task_id, false, 2)).toBe(false);
    expect(isTaskEditableForAssignment(finalizedRelation, finalizedRelation.task_id, false, 2)).toBe(false);
  });

  it("only permits assignmentless task editing when explicitly enabled", () => {
    expect(isTaskEditableForAssignment(null, 2, false, 0)).toBe(false);
    expect(isTaskEditableForAssignment(null, 2, true, 1)).toBe(false);
    expect(isTaskEditableForAssignment(null, 2, true, 0)).toBe(true);
  });

  it("finds the next open task without returning the task under review", () => {
    const finished = assignment(10, "submitted");
    const next = assignment(11, "assigned");
    const later = assignment(12, "in_progress");

    expect(nextMutableAssignment([finished, next, later], finished.id)).toBe(next);
    expect(nextMutableAssignment([finished], finished.id)).toBeNull();
  });

  it("uses personal language for task and document states", () => {
    expect(personalAssignmentStatusLabel("assigned")).toBe("To do");
    expect(personalAssignmentStatusLabel("submitted")).toBe("Finished");
    expect(personalAssignmentStatusLabel("blocked")).toBe("Needs attention");

    expect(personalDocumentStatusLabel("todo")).toBe("To do");
    expect(personalDocumentStatusLabel("partial")).toBe("In progress");
    expect(personalDocumentStatusLabel("done")).toBe("Finished");

    expect(personalDocumentAction("todo")).toBe("Start");
    expect(personalDocumentAction("partial")).toBe("Continue");
    expect(personalDocumentAction("done")).toBe("Review");
  });
});

describe("annotation workspace ownership policy", () => {
  it("only treats the current annotator's human rows as editable and visible", () => {
    const alice = annotation("human", "alice");
    const bob = annotation("human", "bob");

    expect(isHumanAnnotationOwnedBy(alice, "alice")).toBe(true);
    expect(isHumanAnnotationOwnedBy(bob, "alice")).toBe(false);
    expect(isAnnotationVisibleToAnnotator(alice, "alice")).toBe(true);
    expect(isAnnotationVisibleToAnnotator(bob, "alice")).toBe(false);
  });

  it("keeps non-human reference overlays visible but never owner-editable", () => {
    const model = annotation("model", null);
    const gold = { ...annotation("human", "manager"), status: "gold" as const, source: "llm" as const };

    expect(isAnnotationVisibleToAnnotator(model, "alice")).toBe(true);
    expect(isAnnotationVisibleToAnnotator(gold, "alice")).toBe(true);
    expect(isHumanAnnotationOwnedBy(model, "alice")).toBe(false);
  });
});
