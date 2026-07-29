import { beforeEach, describe, expect, it, vi } from "vitest";

import { useEvidenceBlockStore } from "./evidenceBlockStore";

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

describe("evidence block store session reset", () => {
  beforeEach(() => {
    useEvidenceBlockStore.getState().reset();
  });

  it("clears scope, selection, annotation selection, and command history", () => {
    const state = useEvidenceBlockStore.getState();
    state.setScope({ documentId: 1, targetVersionId: 2, structureVersionId: 3 });
    state.startSelection(10, 4);
    state.selectAnnotation(99);
    state.recordCommand({
      key: "command-1",
      operation: "create",
      annotationIds: [99],
      undo: async () => undefined,
      redo: async () => undefined,
    });
    const generation = useEvidenceBlockStore.getState().requestGeneration;

    useEvidenceBlockStore.getState().reset();

    expect(useEvidenceBlockStore.getState()).toMatchObject({
      scope: null,
      selection: null,
      selectedAnnotationId: null,
      past: [],
      future: [],
      commandPending: false,
      requestGeneration: generation + 1,
    });
  });

  it("does not restore command history when an old undo resolves after reset", async () => {
    const pendingUndo = deferred();
    const undo = vi.fn(() => pendingUndo.promise);
    useEvidenceBlockStore.getState().recordCommand({
      key: "old-session-command",
      operation: "delete",
      annotationIds: [42],
      undo,
      redo: async () => undefined,
    });

    const undoOperation = useEvidenceBlockStore.getState().undo();
    expect(useEvidenceBlockStore.getState().commandPending).toBe(true);
    useEvidenceBlockStore.getState().reset();
    pendingUndo.resolve();
    await undoOperation;

    expect(undo).toHaveBeenCalledOnce();
    expect(useEvidenceBlockStore.getState()).toMatchObject({
      scope: null,
      selection: null,
      selectedAnnotationId: null,
      past: [],
      future: [],
      commandPending: false,
    });
  });
});
