import type {
  Annotation,
  TaskAssignment,
  TaskAssignmentStatus,
} from "@/types/api";

export function isMutableAssignment(assignment: TaskAssignment | null): boolean {
  return (
    assignment !== null &&
    (assignment.status === "assigned" ||
      assignment.status === "in_progress" ||
      assignment.status === "blocked")
  );
}

export function preferredAssignmentId(
  assignments: TaskAssignment[],
  currentAssignmentId: number | null,
): number | null {
  const current =
    assignments.find((assignment) => assignment.id === currentAssignmentId) ?? null;
  if (current && isMutableAssignment(current)) {
    return current.id;
  }

  const mutable = assignments.find((assignment) => isMutableAssignment(assignment));
  return mutable?.id ?? current?.id ?? assignments[0]?.id ?? null;
}

export function preferredAssignment(
  assignments: TaskAssignment[],
): TaskAssignment | null {
  return (
    assignments.find((assignment) => isMutableAssignment(assignment)) ??
    assignments[0] ??
    null
  );
}

export function nextMutableAssignment(
  assignments: TaskAssignment[],
  currentAssignmentId: number | null,
): TaskAssignment | null {
  return (
    assignments.find(
      (assignment) =>
        assignment.id !== currentAssignmentId && isMutableAssignment(assignment),
    ) ?? null
  );
}

export function personalAssignmentStatusLabel(
  status: TaskAssignmentStatus,
): string {
  switch (status) {
    case "assigned":
      return "To do";
    case "in_progress":
      return "In progress";
    case "submitted":
    case "adjudicated":
    case "completed":
      return "Finished";
    case "adjudication_ready":
      return "Ready to review";
    case "blocked":
      return "Needs attention";
    case "withdrawn":
      return "Unavailable";
  }
}

export function personalDocumentAction(
  status: "todo" | "partial" | "done" | "review" | "blocked",
): "Start" | "Continue" | "Review" | "Resolve" {
  switch (status) {
    case "partial":
      return "Continue";
    case "done":
    case "review":
      return "Review";
    case "blocked":
      return "Resolve";
    case "todo":
      return "Start";
  }
}

export function personalDocumentStatusLabel(
  status: "todo" | "partial" | "done" | "review" | "blocked",
): string {
  switch (status) {
    case "todo":
      return "To do";
    case "partial":
      return "In progress";
    case "done":
      return "Finished";
    case "review":
      return "Ready to review";
    case "blocked":
      return "Needs attention";
  }
}

export function isTaskEditableForAssignment(
  assignment: TaskAssignment | null,
  taskId: number | null | undefined,
  allowAssignmentlessEditing: boolean,
  documentAssignmentCount: number,
): boolean {
  if (taskId === null || taskId === undefined) {
    return false;
  }
  if (assignment !== null) {
    return isMutableAssignment(assignment) && assignment.task_id === taskId;
  }
  return allowAssignmentlessEditing && documentAssignmentCount === 0;
}

export function isHumanAnnotationOwnedBy(
  annotation: Annotation,
  annotatorId: string,
): boolean {
  return annotation.source === "human" && annotation.annotator_id === annotatorId;
}

export function isAnnotationVisibleToAnnotator(
  annotation: Annotation,
  annotatorId: string,
): boolean {
  return annotation.source !== "human" || isHumanAnnotationOwnedBy(annotation, annotatorId);
}
