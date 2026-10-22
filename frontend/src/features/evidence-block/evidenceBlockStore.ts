import { create } from "zustand";
import { immer } from "zustand/middleware/immer";

import { normalizeSentenceRange } from "./evidenceBlockSelection";

export interface EvidenceScope {
  documentId: number;
  targetVersionId: number;
  structureVersionId: number;
}

export interface EvidenceSelection {
  anchorSentenceId: number;
  focusSentenceId: number;
  anchorOrdinal: number;
  focusOrdinal: number;
  startOrdinal: number;
  endOrdinal: number;
}

interface EvidenceBlockState {
  scope: EvidenceScope | null;
  selection: EvidenceSelection | null;
  selectedAnnotationId: number | null;
  reset: () => void;
  setScope: (scope: EvidenceScope | null) => void;
  startSelection: (sentenceId: number, ordinal: number) => void;
  extendSelection: (sentenceId: number, ordinal: number) => void;
  setSelection: (
    anchorSentenceId: number,
    focusSentenceId: number,
    anchorOrdinal: number,
    focusOrdinal: number,
  ) => void;
  clearSelection: () => void;
  selectAnnotation: (annotationId: number | null) => void;
}

function scopesMatch(left: EvidenceScope | null, right: EvidenceScope | null): boolean {
  return (
    left?.documentId === right?.documentId &&
    left?.targetVersionId === right?.targetVersionId &&
    left?.structureVersionId === right?.structureVersionId
  );
}

export const useEvidenceBlockStore = create<EvidenceBlockState>()(
  immer((set) => ({
    scope: null,
    selection: null,
    selectedAnnotationId: null,

    reset: () =>
      set((state) => {
        state.scope = null;
        state.selection = null;
        state.selectedAnnotationId = null;
      }),

    setScope: (scope) =>
      set((state) => {
        if (scopesMatch(state.scope, scope)) {
          return;
        }
        state.scope = scope;
        state.selection = null;
        state.selectedAnnotationId = null;
      }),

    startSelection: (sentenceId, ordinal) =>
      set((state) => {
        state.selection = {
          anchorSentenceId: sentenceId,
          focusSentenceId: sentenceId,
          anchorOrdinal: ordinal,
          focusOrdinal: ordinal,
          startOrdinal: ordinal,
          endOrdinal: ordinal,
        };
      }),

    extendSelection: (sentenceId, ordinal) =>
      set((state) => {
        if (!state.selection) {
          state.selection = {
            anchorSentenceId: sentenceId,
            focusSentenceId: sentenceId,
            anchorOrdinal: ordinal,
            focusOrdinal: ordinal,
            startOrdinal: ordinal,
            endOrdinal: ordinal,
          };
          return;
        }
        state.selection.focusSentenceId = sentenceId;
        state.selection.focusOrdinal = ordinal;
        const normalized = normalizeSentenceRange(state.selection.anchorOrdinal, ordinal);
        state.selection.startOrdinal = normalized.startOrdinal;
        state.selection.endOrdinal = normalized.endOrdinal;
      }),

    setSelection: (anchorSentenceId, focusSentenceId, anchorOrdinal, focusOrdinal) =>
      set((state) => {
        const normalized = normalizeSentenceRange(anchorOrdinal, focusOrdinal);
        state.selection = {
          anchorSentenceId,
          focusSentenceId,
          anchorOrdinal,
          focusOrdinal,
          startOrdinal: normalized.startOrdinal,
          endOrdinal: normalized.endOrdinal,
        };
      }),

    clearSelection: () =>
      set((state) => {
        state.selection = null;
      }),

    selectAnnotation: (annotationId) =>
      set((state) => {
        state.selectedAnnotationId = annotationId;
      }),
  })),
);
