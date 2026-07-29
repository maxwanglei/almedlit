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

export interface EvidenceServerCommand {
  key: string;
  operation: "create" | "update" | "delete" | "merge" | "split";
  annotationIds: number[];
  undo: () => Promise<void>;
  redo: () => Promise<void>;
}

interface EvidenceBlockState {
  scope: EvidenceScope | null;
  selection: EvidenceSelection | null;
  selectedAnnotationId: number | null;
  requestGeneration: number;
  past: EvidenceServerCommand[];
  future: EvidenceServerCommand[];
  commandPending: boolean;
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
  recordCommand: (command: EvidenceServerCommand) => void;
  undo: () => Promise<void>;
  redo: () => Promise<void>;
}

function scopesMatch(left: EvidenceScope | null, right: EvidenceScope | null): boolean {
  return (
    left?.documentId === right?.documentId &&
    left?.targetVersionId === right?.targetVersionId &&
    left?.structureVersionId === right?.structureVersionId
  );
}

export const useEvidenceBlockStore = create<EvidenceBlockState>()(
  immer((set, get) => ({
    scope: null,
    selection: null,
    selectedAnnotationId: null,
    requestGeneration: 0,
    past: [],
    future: [],
    commandPending: false,

    reset: () =>
      set((state) => {
        state.scope = null;
        state.selection = null;
        state.selectedAnnotationId = null;
        state.past = [];
        state.future = [];
        state.commandPending = false;
        state.requestGeneration += 1;
      }),

    setScope: (scope) =>
      set((state) => {
        if (scopesMatch(state.scope, scope)) {
          return;
        }
        state.scope = scope;
        state.selection = null;
        state.selectedAnnotationId = null;
        state.past = [];
        state.future = [];
        state.requestGeneration += 1;
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

    recordCommand: (command) =>
      set((state) => {
        state.past.push(command);
        state.future = [];
      }),

    undo: async () => {
      const state = get();
      const command = state.past[state.past.length - 1];
      const generation = state.requestGeneration;
      if (!command || state.commandPending) {
        return;
      }
      set((draft) => {
        draft.commandPending = true;
      });
      try {
        await command.undo();
        set((draft) => {
          if (draft.requestGeneration !== generation) {
            return;
          }
          draft.past.pop();
          draft.future.push(command);
        });
      } finally {
        set((draft) => {
          if (draft.requestGeneration === generation) {
            draft.commandPending = false;
          }
        });
      }
    },

    redo: async () => {
      const state = get();
      const command = state.future[state.future.length - 1];
      const generation = state.requestGeneration;
      if (!command || state.commandPending) {
        return;
      }
      set((draft) => {
        draft.commandPending = true;
      });
      try {
        await command.redo();
        set((draft) => {
          if (draft.requestGeneration !== generation) {
            return;
          }
          draft.future.pop();
          draft.past.push(command);
        });
      } finally {
        set((draft) => {
          if (draft.requestGeneration === generation) {
            draft.commandPending = false;
          }
        });
      }
    },
  })),
);
