import type { Document, TaskAssignment } from "@/types/api";

/**
 * Decide which documents an annotator can see.
 *
 * The annotator workspace is normally assignment-driven: an annotator only sees
 * the documents assigned to them. But a project with documents and *no*
 * assignments at all (a fresh project) would otherwise show nothing — and,
 * combined with the annotator selector being empty, that blocks annotation
 * entirely. In that case we fall back to showing every document so the annotator
 * can work without waiting for a manager to create assignments.
 */
export function visibleDocumentsFor(
  documents: Document[],
  assignments: TaskAssignment[],
  selectedAnnotatorId: string,
): Document[] {
  if (!selectedAnnotatorId || assignments.length === 0) {
    return documents;
  }
  const assignedDocumentIds = new Set(
    assignments
      .filter((assignment) => assignment.annotator_id === selectedAnnotatorId)
      .map((assignment) => assignment.document_id),
  );
  return documents.filter((documentItem) => assignedDocumentIds.has(documentItem.id));
}
