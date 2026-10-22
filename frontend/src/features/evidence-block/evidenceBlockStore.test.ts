import { beforeEach, describe, expect, it } from "vitest";

import { useEvidenceBlockStore } from "./evidenceBlockStore";

describe("evidence block selection", () => {
  beforeEach(() => {
    useEvidenceBlockStore.getState().reset();
  });

  it("clears the active scope and selected evidence when the session resets", () => {
    const state = useEvidenceBlockStore.getState();
    state.setScope({ documentId: 1, targetVersionId: 2, structureVersionId: 3 });
    state.startSelection(10, 4);
    state.selectAnnotation(99);

    useEvidenceBlockStore.getState().reset();

    expect(useEvidenceBlockStore.getState()).toMatchObject({
      scope: null,
      selection: null,
      selectedAnnotationId: null,
    });
  });

  it("keeps selection within the current document, target, and structure", () => {
    const state = useEvidenceBlockStore.getState();
    const scope = { documentId: 1, targetVersionId: 2, structureVersionId: 3 };
    state.setScope(scope);
    state.startSelection(10, 4);
    state.selectAnnotation(99);
    state.setScope({ ...scope });

    expect(useEvidenceBlockStore.getState().selection?.anchorSentenceId).toBe(10);
    expect(useEvidenceBlockStore.getState().selectedAnnotationId).toBe(99);

    state.setScope({ ...scope, structureVersionId: 4 });

    expect(useEvidenceBlockStore.getState()).toMatchObject({
      scope: { ...scope, structureVersionId: 4 },
      selection: null,
      selectedAnnotationId: null,
    });
  });
});
