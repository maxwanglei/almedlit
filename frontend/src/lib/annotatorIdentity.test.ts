import { describe, expect, it } from "vitest";

import type { Document, TaskAssignment } from "@/types/api";

import { visibleDocumentsFor } from "./annotatorIdentity";

const doc = (id: number): Document => ({ id }) as Document;
const assignment = (annotator_id: string, document_id: number): TaskAssignment =>
  ({ annotator_id, document_id }) as TaskAssignment;

describe("visibleDocumentsFor", () => {
  it("shows all documents when the project has no assignments at all", () => {
    const docs = [doc(1), doc(2)];
    // This is the bug case: a fresh project with documents but no assignments.
    // The annotator must still be able to see (and annotate) the documents.
    expect(visibleDocumentsFor(docs, [], "bob")).toEqual(docs);
  });

  it("shows all documents when no annotator is selected", () => {
    const docs = [doc(1), doc(2)];
    expect(visibleDocumentsFor(docs, [assignment("bob", 1)], "")).toEqual(docs);
  });

  it("filters to the selected annotator's assigned documents in assignment mode", () => {
    const docs = [doc(1), doc(2), doc(3)];
    const assignments = [assignment("bob", 1), assignment("amy", 2)];
    expect(visibleDocumentsFor(docs, assignments, "bob")).toEqual([doc(1)]);
  });

  it("shows no documents when the annotator has no assignments but others do", () => {
    const docs = [doc(1), doc(2)];
    const assignments = [assignment("amy", 1)];
    expect(visibleDocumentsFor(docs, assignments, "bob")).toEqual([]);
  });
});
