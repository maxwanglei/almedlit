import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  createAnnotation,
  deleteAnnotation,
  getAnnotation,
  getDocumentStructure,
  getEvidenceReviewCoverage,
  listEvidenceTargets,
  listEvidenceCommands,
  listInferencePredictions,
  listInferenceRuns,
  markEvidenceReviewed,
  mergeEvidenceBlocks,
  reopenEvidenceReview,
  redoEvidenceCommand,
  reviewInferencePrediction,
  splitEvidenceBlock,
  updateAnnotation,
  undoEvidenceCommand,
} from "@/api/client";
import ConfirmDialog from "@/components/ConfirmDialog";
import {
  isAnnotationVisibleToAnnotator,
  isHumanAnnotationOwnedBy,
  isMutableAssignment,
  personalAssignmentStatusLabel,
  preferredAssignment,
} from "@/lib/annotationWorkspacePolicy";
import type {
  Annotation,
  Document,
  DocumentStructureParagraph,
  DocumentStructureRead,
  DocumentStructureSection,
  DocumentStructureSentence,
  EvidenceBlockTaskSettingsV1,
  EvidenceCandidatePrediction,
  EvidenceCommandSummary,
  InferenceRun,
  PredictionReviewAction,
  EvidenceReviewCoverage,
  EvidenceTarget,
  ProjectTask,
  TaskAssignment,
} from "@/types/api";

import {
  contractRangeEnd,
  contractRangeStart,
  expandRangeEnd,
  expandRangeStart,
  rangeContainsOrdinal,
  rangesAreAdjacent,
  rangesOverlap,
} from "./evidenceBlockSelection";
import { nextRedoCommand, nextUndoCommand } from "./evidenceCommandHistory";
import { useEvidenceBlockStore } from "./evidenceBlockStore";
import EvidencePredictionPanel from "./EvidencePredictionPanel";

interface EvidenceBlockCanvasProps {
  projectId: number;
  document: Document;
  task: ProjectTask;
  assignments: TaskAssignment[];
  annotations: Annotation[];
  annotatorId: string;
  allowAssignmentlessEditing?: boolean;
  busy: boolean;
  setBusy: (busy: boolean) => void;
  setError: (message: string | null) => void;
  onAnnotationsChanged: (upserted: Annotation[], removedIds?: number[]) => void;
  onRefreshAnnotations: () => Promise<void>;
  onActiveAssignmentChange?: (assignment: TaskAssignment | null) => void;
  onSaveStatusChange?: (status: "saving" | "saved" | "error") => void;
}

interface TargetOption {
  target: EvidenceTarget;
  versionId: number;
  versionNumber: number;
  text: string;
  guidance: string | null;
  inclusionGuidance: string | null;
  exclusionGuidance: string | null;
}

const DEFAULT_SETTINGS: EvidenceBlockTaskSettingsV1 = {
  schema_version: "1",
  active_target_ids: [],
  sentence_boundaries: true,
  multi_paragraph_allowed: true,
  cross_section_allowed: false,
  same_target_overlap_allowed: false,
  adjacency_allowed: true,
  soft_token_warning: 3072,
  model_context_tokens: 4096,
  window_overlap_tokens: 512,
  review_scope: "document",
  keyboard_shortcuts: {
    create: "e",
    expand_start: "shift+arrowup",
    expand_end: "shift+arrowdown",
    contract_start: "alt+arrowup",
    contract_end: "alt+arrowdown",
    merge: "m",
    split: "s",
    delete: "delete",
    mark_reviewed: "r",
    cancel: "escape",
  },
};

function asSettings(task: ProjectTask): EvidenceBlockTaskSettingsV1 {
  const configured = task.settings as Partial<EvidenceBlockTaskSettingsV1>;
  return {
    ...DEFAULT_SETTINGS,
    ...configured,
    keyboard_shortcuts: {
      ...DEFAULT_SETTINGS.keyboard_shortcuts,
      ...(configured.keyboard_shortcuts ?? {}),
    },
  };
}

function eventShortcut(event: {
  ctrlKey: boolean;
  metaKey: boolean;
  altKey: boolean;
  shiftKey: boolean;
  key: string;
}): string {
  const modifiers = [
    event.ctrlKey ? "ctrl" : null,
    event.metaKey ? "meta" : null,
    event.altKey ? "alt" : null,
    event.shiftKey ? "shift" : null,
  ].filter((item): item is string => item !== null);
  return [...modifiers, event.key.toLowerCase()].join("+");
}

function mergeStructurePages(pages: DocumentStructureRead[]): DocumentStructureRead {
  const first = pages[0];
  const byId = <T extends { id: number }>(items: T[][]): T[] =>
    Array.from(new Map(items.flat().map((item) => [item.id, item])).values());
  const sentences = byId(pages.map((page) => page.sentences)).sort(
    (left, right) => left.ordinal - right.ordinal,
  );
  return {
    ...first,
    range: {
      start_ordinal: 0,
      end_ordinal: sentences.length,
      total_sentences: first.range.total_sentences,
      has_more: false,
    },
    sections: byId(pages.map((page) => page.sections)).sort(
      (left, right) => left.ordinal - right.ordinal,
    ),
    paragraphs: byId(pages.map((page) => page.paragraphs)).sort(
      (left, right) => left.ordinal - right.ordinal,
    ),
    sentences,
  };
}

async function loadCompleteStructure(
  documentId: number,
  versionId: number,
): Promise<DocumentStructureRead> {
  const pages: DocumentStructureRead[] = [];
  let start = 0;
  let hasMore = true;
  while (hasMore) {
    const page = await getDocumentStructure(documentId, {
      versionId,
      sentenceStart: start,
      sentenceLimit: 500,
    });
    pages.push(page);
    hasMore = page.range.has_more;
    if (!hasMore) {
      break;
    }
    const next = page.range.end_ordinal;
    if (next <= start) {
      throw new Error("The structure API returned a non-advancing sentence range.");
    }
    start = next;
  }
  return mergeStructurePages(pages);
}

function targetOptions(
  targets: EvidenceTarget[],
  assignments: TaskAssignment[],
  settings: EvidenceBlockTaskSettingsV1,
): TargetOption[] {
  const assignedVersionIds = new Set(
    assignments
      .map((assignment) => assignment.target_version_id)
      .filter((value): value is number => value !== null && value !== undefined),
  );
  const activeTargetIds = new Set(settings.active_target_ids);
  const options: TargetOption[] = [];

  targets.forEach((target) => {
    target.versions.forEach((version) => {
      const assigned = assignedVersionIds.has(version.id);
      const active =
        assignedVersionIds.size === 0 &&
        target.is_active &&
        target.active_version_id === version.id &&
        (activeTargetIds.size === 0 || activeTargetIds.has(target.id));
      if (assigned || active) {
        options.push({
          target,
          versionId: version.id,
          versionNumber: version.version_number,
          text: version.text,
          guidance: version.guidance,
          inclusionGuidance: version.inclusion_guidance,
          exclusionGuidance: version.exclusion_guidance,
        });
      }
    });
  });
  return options.sort((left, right) =>
    left.target.name.localeCompare(right.target.name) || left.versionNumber - right.versionNumber,
  );
}

function preferredTargetVersionId(
  options: TargetOption[],
  assignments: TaskAssignment[],
  currentTargetVersionId: number | null,
): number | null {
  const availableVersionIds = new Set(options.map((option) => option.versionId));
  const currentIsMutable = assignments.some(
    (assignment) =>
      assignment.target_version_id === currentTargetVersionId && isMutableAssignment(assignment),
  );
  if (
    currentTargetVersionId !== null &&
    availableVersionIds.has(currentTargetVersionId) &&
    currentIsMutable
  ) {
    return currentTargetVersionId;
  }

  const mutableTargetVersionId = assignments.find(
    (assignment) =>
      assignment.target_version_id !== null &&
      assignment.target_version_id !== undefined &&
      availableVersionIds.has(assignment.target_version_id) &&
      isMutableAssignment(assignment),
  )?.target_version_id;
  if (mutableTargetVersionId !== null && mutableTargetVersionId !== undefined) {
    return mutableTargetVersionId;
  }

  if (currentTargetVersionId !== null && availableVersionIds.has(currentTargetVersionId)) {
    return currentTargetVersionId;
  }
  return options[0]?.versionId ?? null;
}

function annotationRange(annotation: Annotation): { startOrdinal: number; endOrdinal: number } | null {
  const block = annotation.evidence_block;
  return block?.start_sentence_ordinal === undefined || block.end_sentence_ordinal === undefined
    ? null
    : { startOrdinal: block.start_sentence_ordinal, endOrdinal: block.end_sentence_ordinal };
}

function sentenceStateClass(
  sentence: DocumentStructureSentence,
  selection: { startOrdinal: number; endOrdinal: number } | null,
  blocks: Annotation[],
  predictions: EvidenceCandidatePrediction[],
  coverage: EvidenceReviewCoverage | null,
): string {
  const classes = ["eb-sentence"];
  if (selection && rangeContainsOrdinal(selection, sentence.ordinal)) {
    classes.push("selected");
  }
  const containing = blocks.filter((annotation) => {
    const range = annotationRange(annotation);
    return range ? rangeContainsOrdinal(range, sentence.ordinal) : false;
  });
  if (containing.some((annotation) => annotation.status === "gold")) {
    classes.push("gold");
  }
  if (containing.some((annotation) => annotation.source === "model")) {
    classes.push("prediction");
  }
  if (containing.some((annotation) => annotation.source === "human")) {
    classes.push("human");
  }
  const containingPredictions = predictions.filter(
    (prediction) =>
      prediction.start_sentence_ordinal <= sentence.ordinal &&
      prediction.end_sentence_ordinal >= sentence.ordinal,
  );
  if (containingPredictions.length > 0) {
    classes.push("prediction");
  }
  if (containingPredictions.some((prediction) => prediction.review_status === "pending")) {
    classes.push("prediction-pending");
  } else if (containingPredictions.length > 0) {
    classes.push("prediction-reviewed");
  }
  const reviewed = coverage?.intervals.some((interval) =>
    rangeContainsOrdinal(
      { startOrdinal: interval.start_sentence_ordinal, endOrdinal: interval.end_sentence_ordinal },
      sentence.ordinal,
    ),
  );
  classes.push(reviewed ? "reviewed" : "unreviewed");
  return classes.join(" ");
}

export default function EvidenceBlockCanvas({
  projectId,
  document,
  task,
  assignments,
  annotations,
  annotatorId,
  allowAssignmentlessEditing = false,
  busy,
  setBusy,
  setError,
  onAnnotationsChanged,
  onRefreshAnnotations,
  onActiveAssignmentChange,
  onSaveStatusChange,
}: EvidenceBlockCanvasProps): React.ReactElement {
  const settings = useMemo(() => asSettings(task), [task]);
  const [targets, setTargets] = useState<EvidenceTarget[]>([]);
  const [selectedTargetVersionId, setSelectedTargetVersionId] = useState<number | null>(null);
  const [structure, setStructure] = useState<DocumentStructureRead | null>(null);
  const [coverage, setCoverage] = useState<EvidenceReviewCoverage | null>(null);
  const [note, setNote] = useState("");
  const [labels, setLabels] = useState<string[]>([]);
  const [selectionIntent, setSelectionIntent] = useState<"block" | "review">("block");
  const [checkedBlockIds, setCheckedBlockIds] = useState<number[]>([]);
  const [dragging, setDragging] = useState(false);
  const [inferenceRuns, setInferenceRuns] = useState<InferenceRun[]>([]);
  const [selectedInferenceRunId, setSelectedInferenceRunId] = useState<number | null>(null);
  const [predictions, setPredictions] = useState<EvidenceCandidatePrediction[]>([]);
  const [selectedPredictionId, setSelectedPredictionId] = useState<number | null>(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionMessage, setPredictionMessage] = useState<string | null>(null);
  const [commands, setCommands] = useState<EvidenceCommandSummary[]>([]);
  const [commandLoading, setCommandLoading] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const loadGeneration = useRef(0);
  const predictionLoadGeneration = useRef(0);
  const commandLoadGeneration = useRef(0);

  const selection = useEvidenceBlockStore((state) => state.selection);
  const selectedAnnotationId = useEvidenceBlockStore((state) => state.selectedAnnotationId);
  const setScope = useEvidenceBlockStore((state) => state.setScope);
  const startSelection = useEvidenceBlockStore((state) => state.startSelection);
  const extendSelection = useEvidenceBlockStore((state) => state.extendSelection);
  const setSelection = useEvidenceBlockStore((state) => state.setSelection);
  const clearSelection = useEvidenceBlockStore((state) => state.clearSelection);
  const selectAnnotation = useEvidenceBlockStore((state) => state.selectAnnotation);

  const documentAssignments = useMemo(
    () =>
      assignments.filter(
        (assignment) =>
          assignment.document_id === document.id &&
          assignment.task_id === task.id &&
          assignment.annotator_id === annotatorId,
      ),
    [annotatorId, assignments, document.id, task.id],
  );
  const options = useMemo(
    () => targetOptions(targets, documentAssignments, settings),
    [documentAssignments, settings, targets],
  );
  const selectedTargetOption = useMemo(
    () => options.find((option) => option.versionId === selectedTargetVersionId) ?? null,
    [options, selectedTargetVersionId],
  );

  useEffect(() => {
    let active = true;
    void listEvidenceTargets(projectId)
      .then((items) => {
        if (active) {
          setTargets(items);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setError(error instanceof Error ? error.message : "Unable to load evidence targets.");
        }
      });
    return () => {
      active = false;
    };
  }, [projectId, setError]);

  useEffect(() => {
    let active = true;
    void listInferenceRuns(projectId)
      .then((items) => {
        if (active) {
          setInferenceRuns(items);
          setPredictionMessage(null);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setInferenceRuns([]);
          setPredictionMessage(
            error instanceof Error ? error.message : "Unable to discover inference runs.",
          );
        }
      });
    return () => {
      active = false;
    };
  }, [projectId]);

  useEffect(() => {
    setSelectedTargetVersionId((current) =>
      preferredTargetVersionId(options, documentAssignments, current),
    );
  }, [documentAssignments, options]);

  const selectedAssignment = useMemo(
    () => {
      const targetAssignments = documentAssignments.filter(
        (assignment) => assignment.target_version_id === selectedTargetVersionId,
      );
      return (
        preferredAssignment(targetAssignments) ??
        preferredAssignment(
          documentAssignments.filter((assignment) => assignment.target_version_id == null),
        )
      );
    },
    [documentAssignments, selectedTargetVersionId],
  );
  useEffect(() => {
    onActiveAssignmentChange?.(selectedAssignment);
    return () => onActiveAssignmentChange?.(null);
  }, [onActiveAssignmentChange, selectedAssignment]);
  const structureVersionId =
    selectedAssignment?.structure_version_id ?? document.active_structure_version_id;
  const assignmentEditable = selectedAssignment
    ? isMutableAssignment(selectedAssignment)
    : allowAssignmentlessEditing && documentAssignments.length === 0;

  useEffect(() => {
    setNote("");
    setLabels([]);
    setCheckedBlockIds([]);
    setSelectedPredictionId(null);
    setSelectionIntent("block");
  }, [document.id, selectedTargetVersionId, structureVersionId]);

  const compatibleInferenceRuns = useMemo(
    () =>
      inferenceRuns.filter((run) =>
        selectedTargetVersionId === null
          ? false
          : run.target_version_ids.includes(selectedTargetVersionId),
      ),
    [inferenceRuns, selectedTargetVersionId],
  );

  useEffect(() => {
    if (compatibleInferenceRuns.length === 0) {
      setSelectedInferenceRunId(null);
      setPredictions([]);
      setSelectedPredictionId(null);
      return;
    }
    if (!compatibleInferenceRuns.some((run) => run.id === selectedInferenceRunId)) {
      const completed = compatibleInferenceRuns.find((run) => run.status === "succeeded");
      setSelectedInferenceRunId((completed ?? compatibleInferenceRuns[0]).id);
      setSelectedPredictionId(null);
    }
  }, [compatibleInferenceRuns, selectedInferenceRunId]);

  const refreshPredictions = useCallback(async (): Promise<void> => {
    if (
      selectedInferenceRunId === null ||
      selectedTargetVersionId === null ||
      structureVersionId === null
    ) {
      setPredictions([]);
      return;
    }
    const generation = ++predictionLoadGeneration.current;
    setPredictionLoading(true);
    setPredictionMessage(null);
    try {
      const items = await listInferencePredictions(selectedInferenceRunId, {
        documentId: document.id,
        targetVersionId: selectedTargetVersionId,
      });
      if (generation === predictionLoadGeneration.current) {
        setPredictions(
          items.filter((prediction) => prediction.structure_version_id === structureVersionId),
        );
      }
    } catch (error) {
      if (generation === predictionLoadGeneration.current) {
        setPredictions([]);
        setPredictionMessage(
          error instanceof Error ? error.message : "Unable to load prediction candidates.",
        );
      }
    } finally {
      if (generation === predictionLoadGeneration.current) {
        setPredictionLoading(false);
      }
    }
  }, [document.id, selectedInferenceRunId, selectedTargetVersionId, structureVersionId]);

  const refreshCommands = useCallback(async (): Promise<void> => {
    if (selectedTargetVersionId === null || structureVersionId === null) {
      setCommands([]);
      return;
    }
    const generation = ++commandLoadGeneration.current;
    setCommandLoading(true);
    try {
      const items = await listEvidenceCommands(projectId, {
        documentId: document.id,
        targetVersionId: selectedTargetVersionId,
        structureVersionId,
        guidelineVersionId: selectedAssignment?.guideline_version_id,
      });
      if (generation === commandLoadGeneration.current) {
        setCommands(items);
      }
    } catch (error) {
      if (generation === commandLoadGeneration.current) {
        setCommands([]);
        setError(error instanceof Error ? error.message : "Unable to load evidence command history.");
      }
    } finally {
      if (generation === commandLoadGeneration.current) {
        setCommandLoading(false);
      }
    }
  }, [
    document.id,
    projectId,
    selectedAssignment?.guideline_version_id,
    selectedTargetVersionId,
    setError,
    structureVersionId,
  ]);

  useEffect(() => {
    void refreshPredictions();
    return () => {
      predictionLoadGeneration.current += 1;
    };
  }, [refreshPredictions]);

  useEffect(() => {
    setCommands([]);
    void refreshCommands();
    return () => {
      commandLoadGeneration.current += 1;
    };
  }, [refreshCommands, structureVersionId]);

  useEffect(() => {
    if (selectedTargetVersionId === null || structureVersionId === null) {
      setStructure(null);
      setCoverage(null);
      setScope(null);
      return;
    }

    const generation = ++loadGeneration.current;
    setScope({
      documentId: document.id,
      targetVersionId: selectedTargetVersionId,
      structureVersionId,
    });
    setBusy(true);
    setError(null);
    Promise.all([
      loadCompleteStructure(document.id, structureVersionId),
      getEvidenceReviewCoverage(
        projectId,
        document.id,
        selectedTargetVersionId,
        structureVersionId,
        selectedAssignment?.guideline_version_id,
      ).catch(() => null),
    ])
      .then(([nextStructure, nextCoverage]) => {
        if (generation !== loadGeneration.current) {
          return;
        }
        setStructure(nextStructure);
        setCoverage(nextCoverage);
      })
      .catch((error: unknown) => {
        if (generation === loadGeneration.current) {
          setError(error instanceof Error ? error.message : "Unable to load document structure.");
        }
      })
      .finally(() => {
        if (generation === loadGeneration.current) {
          setBusy(false);
        }
      });
  }, [
    document.id,
    projectId,
    selectedAssignment?.guideline_version_id,
    selectedTargetVersionId,
    setBusy,
    setError,
    setScope,
    structureVersionId,
  ]);

  useEffect(() => {
    const stopDragging = (): void => setDragging(false);
    window.addEventListener("pointerup", stopDragging);
    return () => window.removeEventListener("pointerup", stopDragging);
  }, []);

  const targetBlocks = useMemo(
    () =>
      annotations.filter(
        (annotation) =>
          annotation.annotation_type === "evidence_block" &&
          isAnnotationVisibleToAnnotator(annotation, annotatorId) &&
          annotation.evidence_block?.target_version_id === selectedTargetVersionId &&
          annotation.evidence_block?.structure_version_id === structureVersionId &&
          (selectedAssignment === null ||
            annotation.guideline_version_id === selectedAssignment.guideline_version_id),
      ),
    [annotations, annotatorId, selectedAssignment, selectedTargetVersionId, structureVersionId],
  );

  const sentenceByOrdinal = useMemo(
    () => new Map((structure?.sentences ?? []).map((sentence) => [sentence.ordinal, sentence])),
    [structure],
  );
  const selectedStartSentence = selection ? sentenceByOrdinal.get(selection.startOrdinal) : undefined;
  const selectedEndSentence = selection ? sentenceByOrdinal.get(selection.endOrdinal) : undefined;
  const selectedBlock = targetBlocks.find((annotation) => annotation.id === selectedAnnotationId) ?? null;
  const selectedBlockEditable = Boolean(
    selectedBlock &&
      isHumanAnnotationOwnedBy(selectedBlock, annotatorId) &&
      selectedBlock.status !== "gold" &&
      !selectedBlock.evidence_block?.locked,
  );
  const editorReadOnly = !assignmentEditable || Boolean(selectedBlock && !selectedBlockEditable);
  const selectedSentences = useMemo(
    () =>
      selection
        ? (structure?.sentences ?? []).filter((sentence) =>
            rangeContainsOrdinal(selection, sentence.ordinal),
          )
        : [],
    [selection, structure],
  );
  const selectedParagraphCount = useMemo(
    () => new Set(selectedSentences.map((sentence) => sentence.paragraph_id)).size,
    [selectedSentences],
  );
  const selectedSectionCount = useMemo(
    () => new Set(selectedSentences.map((sentence) => sentence.section_id)).size,
    [selectedSentences],
  );
  const estimatedTokenCount = useMemo(() => {
    const selectedText = selectedSentences.map((sentence) => sentence.text).join(" ");
    return selectedText ? Math.max(1, Math.ceil(selectedText.length / 4)) : 0;
  }, [selectedSentences]);

  const mutate = useCallback(
    async (operation: () => Promise<void>): Promise<void> => {
      if (!assignmentEditable) {
        setError(
          allowAssignmentlessEditing
            ? "This task is read-only because it has already been finished."
            : "This assignment is read-only because it has already been submitted or closed.",
        );
        return;
      }
      setBusy(true);
      onSaveStatusChange?.("saving");
      setError(null);
      try {
        await operation();
        await refreshCommands();
        onSaveStatusChange?.("saved");
      } catch (error) {
        onSaveStatusChange?.("error");
        setError(error instanceof Error ? error.message : "Evidence block operation failed.");
      } finally {
        setBusy(false);
      }
    },
    [
      allowAssignmentlessEditing,
      assignmentEditable,
      refreshCommands,
      setBusy,
      setError,
      onSaveStatusChange,
    ],
  );

  const commandToUndo = useMemo(
    () => nextUndoCommand(commands),
    [commands],
  );
  const commandToRedo = useMemo(
    () => nextRedoCommand(commands),
    [commands],
  );

  const executeHistoryCommand = useCallback(
    async (action: "undo" | "redo"): Promise<void> => {
      const command = action === "undo" ? commandToUndo : commandToRedo;
      if (!command || selectedTargetVersionId === null || structureVersionId === null) {
        return;
      }
      await mutate(async () => {
        if (action === "undo") {
          await undoEvidenceCommand(command.command_group_key);
        } else {
          await redoEvidenceCommand(command.command_group_key);
        }
        await onRefreshAnnotations();
        const nextCoverage = await getEvidenceReviewCoverage(
          projectId,
          document.id,
          selectedTargetVersionId,
          structureVersionId,
          selectedAssignment?.guideline_version_id,
        ).catch(() => null);
        setCoverage(nextCoverage);
        selectAnnotation(null);
        setSelectedPredictionId(null);
        clearSelection();
      });
    },
    [
      clearSelection,
      commandToRedo,
      commandToUndo,
      document.id,
      mutate,
      onRefreshAnnotations,
      projectId,
      selectAnnotation,
      selectedAssignment?.guideline_version_id,
      selectedTargetVersionId,
      structureVersionId,
    ],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      if (eventShortcut(event) === settings.keyboard_shortcuts.cancel) {
        clearSelection();
        selectAnnotation(null);
        setSelectedPredictionId(null);
        return;
      }
      const target = event.target as HTMLElement | null;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target?.isContentEditable
      ) {
        return;
      }
      const undoShortcut =
        (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && !event.shiftKey;
      const redoShortcut =
        ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z" && event.shiftKey) ||
        ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y");
      if (undoShortcut || redoShortcut) {
        event.preventDefault();
        void executeHistoryCommand(redoShortcut ? "redo" : "undo");
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clearSelection, executeHistoryCommand, selectAnnotation, settings.keyboard_shortcuts.cancel]);

  function adjustSelection(
    action: "expand-start" | "expand-end" | "contract-start" | "contract-end",
  ): void {
    if (!selection || !structure || structure.sentences.length === 0) {
      return;
    }
    const maximumOrdinal = structure.sentences[structure.sentences.length - 1].ordinal;
    const current = {
      startOrdinal: selection.startOrdinal,
      endOrdinal: selection.endOrdinal,
    };
    const next =
      action === "expand-start"
        ? expandRangeStart(current, 0)
        : action === "expand-end"
          ? expandRangeEnd(current, maximumOrdinal)
          : action === "contract-start"
            ? contractRangeStart(current)
            : contractRangeEnd(current);
    const start = sentenceByOrdinal.get(next.startOrdinal);
    const end = sentenceByOrdinal.get(next.endOrdinal);
    if (start && end) {
      setSelection(start.id, end.id, start.ordinal, end.ordinal);
    }
  }

  function validateSelection(
    excludeSelectedAnnotation = false,
  ): [DocumentStructureSentence, DocumentStructureSentence] | null {
    if (selectionIntent !== "block") {
      setError("Switch to Block selection before editing evidence blocks.");
      return null;
    }
    if (!selectedStartSentence || !selectedEndSentence || selectedTargetVersionId === null) {
      setError("Select one or more complete sentences first.");
      return null;
    }
    if (
      !settings.multi_paragraph_allowed &&
      selectedStartSentence.paragraph_id !== selectedEndSentence.paragraph_id
    ) {
      setError("Evidence blocks cannot cross paragraph boundaries for this task.");
      return null;
    }
    if (
      !settings.cross_section_allowed &&
      selectedStartSentence.section_id !== selectedEndSentence.section_id
    ) {
      setError("Evidence blocks cannot cross section boundaries for this task.");
      return null;
    }
    const candidate = {
      startOrdinal: selectedStartSentence.ordinal,
      endOrdinal: selectedEndSentence.ordinal,
    };
    if (
      !settings.same_target_overlap_allowed &&
      targetBlocks.some((annotation) => {
        if (
          (excludeSelectedAnnotation && annotation.id === selectedAnnotationId) ||
          annotation.source !== "human"
        ) {
          return false;
        }
        const existing = annotationRange(annotation);
        return existing ? rangesOverlap(candidate, existing) : false;
      })
    ) {
      setError("This range overlaps another block for the same target.");
      return null;
    }
    if (
      !settings.adjacency_allowed &&
      targetBlocks.some((annotation) => {
        if (
          (excludeSelectedAnnotation && annotation.id === selectedAnnotationId) ||
          annotation.source !== "human" ||
          annotation.status === "gold"
        ) {
          return false;
        }
        const existing = annotationRange(annotation);
        return existing ? rangesAreAdjacent(candidate, existing) : false;
      })
    ) {
      setError("Adjacent evidence blocks are disabled for this task.");
      return null;
    }
    return [selectedStartSentence, selectedEndSentence];
  }

  function payloadFor(start: DocumentStructureSentence, end: DocumentStructureSentence) {
    return {
      structure_version_id: structureVersionId!,
      target_version_id: selectedTargetVersionId!,
      start_sentence_id: start.id,
      end_sentence_id: end.id,
      labels,
      note: note.trim() || null,
      boundary_policy: "sentence" as const,
    };
  }

  async function createBlock(): Promise<void> {
    if (selectedBlock) {
      setError("Cancel the current block edit before creating a new evidence block.");
      return;
    }
    const boundaries = validateSelection();
    if (!boundaries) {
      return;
    }
    await mutate(async () => {
      const created = await createAnnotation({
        project_id: projectId,
        document_id: document.id,
        annotation_type: "evidence_block",
        label: "evidence_block",
        annotator_id: annotatorId,
        guideline_version_id: selectedAssignment?.guideline_version_id ?? null,
        evidence_block: payloadFor(...boundaries),
      });
      onAnnotationsChanged([created]);
      selectAnnotation(created.id);
      clearSelection();
    });
  }

  async function updateBlockBounds(): Promise<void> {
    const boundaries = validateSelection(true);
    if (!selectedBlock || !boundaries) {
      return;
    }
    await mutate(async () => {
      const updated = await updateAnnotation(selectedBlock.id, {
        expected_revision: selectedBlock.evidence_block?.revision,
        evidence_block: payloadFor(...boundaries),
      });
      onAnnotationsChanged([updated]);
      clearSelection();
    });
  }

  async function removeBlock(): Promise<void> {
    if (!selectedBlock || !selectedBlockEditable) {
      return;
    }
    setDeleteDialogOpen(true);
  }

  async function confirmRemoveBlock(): Promise<void> {
    if (!selectedBlock || !selectedBlockEditable) {
      setDeleteDialogOpen(false);
      return;
    }
    await mutate(async () => {
      await deleteAnnotation(selectedBlock.id, selectedBlock.evidence_block?.revision);
      onAnnotationsChanged([], [selectedBlock.id]);
      selectAnnotation(null);
    });
    setDeleteDialogOpen(false);
  }

  async function mergeCheckedBlocks(): Promise<void> {
    const selected = targetBlocks.filter((block) => checkedBlockIds.includes(block.id));
    if (selected.length < 2) {
      setError("Choose at least two adjacent evidence blocks to merge.");
      return;
    }
    await mutate(async () => {
      const merged = await mergeEvidenceBlocks({
        annotation_ids: selected.map((block) => block.id),
        expected_revisions: Object.fromEntries(
          selected.map((block) => [block.id, block.evidence_block?.revision ?? 1]),
        ),
        labels,
        note: note.trim() || null,
      });
      onAnnotationsChanged([merged], selected.map((block) => block.id));
      setCheckedBlockIds([]);
      selectAnnotation(merged.id);
    });
  }

  async function splitSelectedBlock(): Promise<void> {
    if (!selectedBlock || !selectedStartSentence) {
      setError("Select a sentence inside a block as the start of its second half.");
      return;
    }
    await mutate(async () => {
      const split = await splitEvidenceBlock(selectedBlock.id, {
        expected_revision: selectedBlock.evidence_block?.revision ?? 1,
        split_before_sentence_id: selectedStartSentence.id,
      });
      onAnnotationsChanged(split, [selectedBlock.id]);
      selectAnnotation(split[0]?.id ?? null);
      clearSelection();
    });
  }

  async function reviewPrediction(
    prediction: EvidenceCandidatePrediction,
    action: PredictionReviewAction,
  ): Promise<void> {
    let replacement: [DocumentStructureSentence, DocumentStructureSentence] | null = null;
    if (action === "modify") {
      replacement = validateSelection();
      if (!replacement) {
        return;
      }
    }
    await mutate(async () => {
      const review = await reviewInferencePrediction(prediction.id, {
        action,
        start_sentence_id: replacement?.[0].id,
        end_sentence_id: replacement?.[1].id,
        labels: action === "reject" ? [] : labels,
        note: note.trim() || null,
        metadata_: { reviewed_in: "evidence_block_canvas" },
      });
      if (review.resulting_annotation_id !== null) {
        const annotation = await getAnnotation(review.resulting_annotation_id);
        onAnnotationsChanged([annotation]);
        selectAnnotation(annotation.id);
      }
      await refreshPredictions();
      setSelectedPredictionId(null);
      clearSelection();
    });
  }

  async function changeCoverage(action: "mark" | "reopen"): Promise<void> {
    if (!selectedStartSentence || !selectedEndSentence || selectedTargetVersionId === null || structureVersionId === null) {
      setError("Select a sentence range first.");
      return;
    }
    if (selectionIntent !== "review") {
      setError("Switch to Review region before changing reviewed coverage.");
      return;
    }
    await mutate(async () => {
      const payload = {
        target_version_id: selectedTargetVersionId,
        structure_version_id: structureVersionId,
        guideline_version_id: selectedAssignment?.guideline_version_id ?? null,
        start_sentence_id: selectedStartSentence.id,
        end_sentence_id: selectedEndSentence.id,
      };
      const next =
        action === "mark"
          ? await markEvidenceReviewed(projectId, document.id, payload)
          : await reopenEvidenceReview(projectId, document.id, payload);
      setCoverage(next);
      clearSelection();
    });
  }

  function chooseSentence(event: React.PointerEvent, sentence: DocumentStructureSentence): void {
    event.preventDefault();
    if (event.shiftKey && selection) {
      setSelection(
        selection.anchorSentenceId,
        sentence.id,
        selection.anchorOrdinal,
        sentence.ordinal,
      );
    } else {
      startSelection(sentence.id, sentence.ordinal);
    }
    setDragging(true);
  }

  function choosePrediction(prediction: EvidenceCandidatePrediction): void {
    setSelectionIntent("block");
    setSelectedPredictionId(prediction.id);
    selectAnnotation(null);
    setSelection(
      prediction.start_sentence_id,
      prediction.end_sentence_id,
      prediction.start_sentence_ordinal,
      prediction.end_sentence_ordinal,
    );
  }

  function changeSelectionIntent(intent: "block" | "review"): void {
    setSelectionIntent(intent);
    if (intent === "review") {
      selectAnnotation(null);
      setLabels([]);
      setNote("");
    }
  }

  function handleWorkbenchShortcut(event: React.KeyboardEvent<HTMLDivElement>): void {
    const shortcut = eventShortcut(event);
    const shortcuts = settings.keyboard_shortcuts;
    if (shortcut === shortcuts.cancel) {
      event.preventDefault();
      event.stopPropagation();
      clearSelection();
      selectAnnotation(null);
      setSelectedPredictionId(null);
      return;
    }
    const target = event.target as HTMLElement;
    if (
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target instanceof HTMLSelectElement ||
      target.isContentEditable ||
      busy
    ) {
      return;
    }
    const action = Object.entries(shortcuts).find(([, value]) => value === shortcut)?.[0];
    if (!action) {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    if (action === "create") {
      void createBlock();
    } else if (action === "expand_start") {
      adjustSelection("expand-start");
    } else if (action === "expand_end") {
      adjustSelection("expand-end");
    } else if (action === "contract_start") {
      adjustSelection("contract-start");
    } else if (action === "contract_end") {
      adjustSelection("contract-end");
    } else if (action === "merge") {
      void mergeCheckedBlocks();
    } else if (action === "split") {
      void splitSelectedBlock();
    } else if (action === "delete" && selectedBlockEditable) {
      void removeBlock();
    } else if (action === "mark_reviewed") {
      if (selectionIntent === "review") {
        void changeCoverage("mark");
      } else {
        setError("Switch to Review region before marking coverage.");
      }
    }
  }

  const paragraphsBySection = useMemo(() => {
    const grouped = new Map<number | null, DocumentStructureParagraph[]>();
    (structure?.paragraphs ?? []).forEach((paragraph) => {
      const items = grouped.get(paragraph.section_id) ?? [];
      items.push(paragraph);
      grouped.set(paragraph.section_id, items);
    });
    return grouped;
  }, [structure]);
  const sentencesByParagraph = useMemo(() => {
    const grouped = new Map<number, DocumentStructureSentence[]>();
    (structure?.sentences ?? []).forEach((sentence) => {
      const items = grouped.get(sentence.paragraph_id) ?? [];
      items.push(sentence);
      grouped.set(sentence.paragraph_id, items);
    });
    return grouped;
  }, [structure]);
  const renderedSections: Array<DocumentStructureSection | null> = structure
    ? structure.sections.length > 0
      ? structure.sections
      : [null]
    : [];

  return (
    <div className="eb-workbench" onKeyDown={handleWorkbenchShortcut}>
      <header className="eb-toolbar">
        <label>
          <span>Evidence target</span>
          <select
            value={selectedTargetVersionId ?? ""}
            onChange={(event) => setSelectedTargetVersionId(Number(event.target.value))}
            disabled={busy || options.length === 0}
          >
            {options.length === 0 ? (
              <option value="">
                {allowAssignmentlessEditing ? "No targets configured" : "No assigned targets"}
              </option>
            ) : null}
            {options.map((option) => (
              <option key={option.versionId} value={option.versionId}>
                {option.target.name} (v{option.versionNumber})
              </option>
            ))}
          </select>
        </label>
        <div className="eb-target-copy">
          <span className="eb-target-text">
            {selectedTargetOption?.text ??
              (allowAssignmentlessEditing
                ? "Configure and activate an evidence target in Project Setup before annotation."
                : "A manager must assign and activate a target before annotation.")}
          </span>
          {selectedTargetOption?.guidance ||
          selectedTargetOption?.inclusionGuidance ||
          selectedTargetOption?.exclusionGuidance ? (
            <details>
              <summary>Target guidance</summary>
              {selectedTargetOption.guidance ? <p>{selectedTargetOption.guidance}</p> : null}
              {selectedTargetOption.inclusionGuidance ? (
                <p><strong>Include:</strong> {selectedTargetOption.inclusionGuidance}</p>
              ) : null}
              {selectedTargetOption.exclusionGuidance ? (
                <p><strong>Exclude:</strong> {selectedTargetOption.exclusionGuidance}</p>
              ) : null}
            </details>
          ) : null}
        </div>
        <label className="eb-run-select">
          <span>Prediction run</span>
          <select
            value={selectedInferenceRunId ?? ""}
            onChange={(event) => {
              setSelectedInferenceRunId(event.target.value ? Number(event.target.value) : null);
              setSelectedPredictionId(null);
            }}
            disabled={predictionLoading || compatibleInferenceRuns.length === 0}
          >
            {compatibleInferenceRuns.length === 0 ? (
              <option value="">No compatible runs</option>
            ) : null}
            {compatibleInferenceRuns.map((run) => (
              <option key={run.id} value={run.id}>
                #{run.id} · {run.name} · {run.status}
              </option>
            ))}
          </select>
        </label>
        <div className="eb-history" aria-label="Evidence command history">
          <button
            type="button"
            onClick={() => void executeHistoryCommand("undo")}
            disabled={!assignmentEditable || busy || commandLoading || commandToUndo === null}
            title={commandToUndo ? `Undo ${commandToUndo.operation}` : "Nothing to undo"}
          >
            Undo
          </button>
          <button
            type="button"
            onClick={() => void executeHistoryCommand("redo")}
            disabled={!assignmentEditable || busy || commandLoading || commandToRedo === null}
            title={commandToRedo ? `Redo ${commandToRedo.operation}` : "Nothing to redo"}
          >
            Redo
          </button>
        </div>
      </header>

      <div className="eb-actions">
        {!assignmentEditable ? (
          <span className="eb-read-only" role="status">
            {allowAssignmentlessEditing
              ? `${selectedAssignment ? personalAssignmentStatusLabel(selectedAssignment.status) : "Finished"} task · read-only`
              : `${selectedAssignment?.status ?? "closed"} assignment · read-only`}
          </span>
        ) : null}
        {predictionMessage ? (
          <span className="eb-prediction-message" role="status">
            {predictionMessage}
          </span>
        ) : null}
        <div className="eb-selection-intent" role="group" aria-label="Selection purpose">
          <button
            type="button"
            aria-pressed={selectionIntent === "block"}
            onClick={() => changeSelectionIntent("block")}
          >
            Block selection
          </button>
          <button
            type="button"
            aria-pressed={selectionIntent === "review"}
            onClick={() => changeSelectionIntent("review")}
          >
            Review region
          </button>
        </div>
        <button
          className="primary"
          type="button"
          onClick={() => void createBlock()}
          disabled={
            !assignmentEditable ||
            busy ||
            !selection ||
            selectionIntent !== "block" ||
            selectedBlock !== null
          }
          title={settings.keyboard_shortcuts.create}
        >
          Create block
        </button>
        <button
          type="button"
          onClick={() => adjustSelection("expand-start")}
          disabled={busy || !selection || selection.startOrdinal === 0}
          title={settings.keyboard_shortcuts.expand_start}
        >
          Expand start
        </button>
        <button
          type="button"
          onClick={() => adjustSelection("expand-end")}
          disabled={
            busy ||
            !selection ||
            !structure ||
            selection.endOrdinal === structure.sentences[structure.sentences.length - 1]?.ordinal
          }
          title={settings.keyboard_shortcuts.expand_end}
        >
          Expand end
        </button>
        <button
          type="button"
          onClick={() => adjustSelection("contract-start")}
          disabled={busy || !selection || selection.startOrdinal === selection.endOrdinal}
          title={settings.keyboard_shortcuts.contract_start}
        >
          Contract start
        </button>
        <button
          type="button"
          onClick={() => adjustSelection("contract-end")}
          disabled={busy || !selection || selection.startOrdinal === selection.endOrdinal}
          title={settings.keyboard_shortcuts.contract_end}
        >
          Contract end
        </button>
        <button type="button" onClick={() => void updateBlockBounds()} disabled={!assignmentEditable || busy || !selection || selectionIntent !== "block" || !selectedBlockEditable}>
          Apply bounds
        </button>
        <button
          type="button"
          onClick={() => void splitSelectedBlock()}
          disabled={!assignmentEditable || busy || !selection || selectionIntent !== "block" || !selectedBlockEditable}
          title={settings.keyboard_shortcuts.split}
        >
          Split before selection
        </button>
        <button
          type="button"
          onClick={() => void mergeCheckedBlocks()}
          disabled={!assignmentEditable || busy || checkedBlockIds.length < 2}
          title={settings.keyboard_shortcuts.merge}
        >
          Merge checked
        </button>
        <button
          type="button"
          onClick={() => void changeCoverage("mark")}
          disabled={!assignmentEditable || busy || !selection || selectionIntent !== "review"}
          title={settings.keyboard_shortcuts.mark_reviewed}
        >
          Mark reviewed
        </button>
        <button
          type="button"
          onClick={() => void changeCoverage("reopen")}
          disabled={!assignmentEditable || busy || !selection || selectionIntent !== "review"}
        >
          Reopen
        </button>
        <button
          className="danger"
          type="button"
          onClick={() => void removeBlock()}
          disabled={!assignmentEditable || busy || !selectedBlockEditable}
          title={settings.keyboard_shortcuts.delete}
        >
          Delete
        </button>
        <button
          type="button"
          onClick={() => {
            clearSelection();
            selectAnnotation(null);
            setLabels([]);
            setNote("");
          }}
          disabled={!selection && !selectedBlock}
        >
          Cancel edit
        </button>
        <label className="eb-note">
          <span>Note</span>
          <input
            value={note}
            onChange={(event) => setNote(event.target.value)}
            disabled={editorReadOnly}
          />
        </label>
        {task.labels.length > 0 ? (
          <fieldset className="eb-labels" disabled={editorReadOnly || busy}>
            <legend>Evidence labels</legend>
            {task.labels.map((label) => {
              const active = labels.includes(label.name);
              return (
                <button
                  key={label.name}
                  type="button"
                  aria-pressed={active}
                  title={label.description ?? undefined}
                  onClick={() =>
                    setLabels((current) =>
                      active
                        ? current.filter((item) => item !== label.name)
                        : [...current, label.name],
                    )
                  }
                >
                  <i style={{ background: label.color }} />
                  {label.name}
                </button>
              );
            })}
          </fieldset>
        ) : null}
        {selection ? (
          <div className="eb-selection-summary" role="status">
            <strong>
              Sentences {selection.startOrdinal + 1}–{selection.endOrdinal + 1}
            </strong>
            <span>
              {selectedParagraphCount} paragraph{selectedParagraphCount === 1 ? "" : "s"} · approximately{" "}
              {estimatedTokenCount} tokens
            </span>
            {!settings.multi_paragraph_allowed && selectedParagraphCount > 1 ? (
              <small className="warning">This task does not allow cross-paragraph blocks.</small>
            ) : null}
            {selectedSectionCount > 1 ? (
              <small className="warning">
                {settings.cross_section_allowed
                  ? "This block crosses a section boundary; confirm the context before saving."
                  : "This task does not allow cross-section blocks."}
              </small>
            ) : null}
            {estimatedTokenCount > settings.soft_token_warning ? (
              <small className="warning">
                This selection exceeds the {settings.soft_token_warning.toLocaleString()}-token soft warning.
              </small>
            ) : null}
          </div>
        ) : null}
      </div>

      <div className="eb-layout">
        <section className="eb-document" aria-label="Structure-aware evidence annotation canvas">
          {!structure ? <p className="aw-empty">Loading sentence structure...</p> : null}
          {renderedSections.map((section, sectionIndex) => {
            const paragraphs = paragraphsBySection.get(section?.id ?? null) ??
              (section === null ? structure?.paragraphs ?? [] : []);
            return (
              <section className="eb-section" key={section?.id ?? `unknown-${sectionIndex}`}>
                <header>
                  <span>{section?.path.join(" / ") || "Unknown section"}</span>
                  {section?.title ? <h2>{section.title}</h2> : null}
                </header>
                {paragraphs.map((paragraph) => (
                  <div
                    className="eb-paragraph"
                    key={paragraph.id}
                    data-paragraph-id={paragraph.id}
                    onDoubleClick={() => {
                      const sentences = sentencesByParagraph.get(paragraph.id) ?? [];
                      const first = sentences[0];
                      const last = sentences[sentences.length - 1];
                      if (first && last) {
                        setSelection(first.id, last.id, first.ordinal, last.ordinal);
                      }
                    }}
                  >
                    <button
                      className="eb-gutter"
                      type="button"
                      aria-label={`Select paragraph ${paragraph.ordinal + 1}`}
                      onClick={() => {
                        const sentences = sentencesByParagraph.get(paragraph.id) ?? [];
                        const first = sentences[0];
                        const last = sentences[sentences.length - 1];
                        if (first && last) {
                          setSelection(first.id, last.id, first.ordinal, last.ordinal);
                        }
                      }}
                    >
                      P{paragraph.ordinal + 1}
                    </button>
                    <div>
                      {(sentencesByParagraph.get(paragraph.id) ?? []).map((sentence) => {
                        const stateClass = sentenceStateClass(
                          sentence,
                          selection,
                          targetBlocks,
                          predictions,
                          coverage,
                        );
                        const accessibleStates = [
                          stateClass.includes(" gold") ? "gold evidence" : null,
                          stateClass.includes(" human") ? "human evidence" : null,
                          stateClass.includes(" prediction") ? "model prediction" : null,
                          stateClass.includes(" reviewed") ? "reviewed" : "unreviewed",
                          stateClass.includes(" selected") ? "selected" : null,
                        ].filter((value): value is string => value !== null);
                        return (
                          <button
                            className={stateClass}
                            aria-label={`Sentence ${sentence.ordinal + 1}, ${accessibleStates.join(", ")}: ${sentence.text}`}
                            key={sentence.id}
                            type="button"
                            data-sentence-id={sentence.id}
                            data-sentence-ordinal={sentence.ordinal}
                            aria-pressed={Boolean(
                              selection && rangeContainsOrdinal(selection, sentence.ordinal)
                            )}
                            onPointerDown={(event) => chooseSentence(event, sentence)}
                            onClick={(event) => {
                              if (event.detail === 0) {
                                if (event.shiftKey && selection) {
                                  setSelection(
                                    selection.anchorSentenceId,
                                    sentence.id,
                                    selection.anchorOrdinal,
                                    sentence.ordinal,
                                  );
                                } else {
                                  startSelection(sentence.id, sentence.ordinal);
                                }
                              }
                            }}
                            onPointerEnter={() => {
                              if (dragging) {
                                extendSelection(sentence.id, sentence.ordinal);
                              }
                            }}
                          >
                            <span className="eb-sentence-number">{sentence.ordinal + 1}</span>
                            <span>{sentence.text}</span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </section>
            );
          })}
        </section>

        <aside className="eb-block-list" aria-label="Evidence blocks and model predictions">
          <header>
            <strong>{targetBlocks.length} blocks</strong>
            <span>{coverage?.fully_reviewed ? "Fully reviewed" : "Review incomplete"}</span>
          </header>
          {targetBlocks.length === 0 ? <p className="aw-empty">No blocks for this target.</p> : null}
          {targetBlocks.map((annotation) => {
            const block = annotation.evidence_block!;
            const active = annotation.id === selectedAnnotationId;
            return (
              <article className={`eb-block-card ${active ? "active" : ""} ${annotation.status}`} key={annotation.id}>
                <label>
                  <input
                    type="checkbox"
                    checked={checkedBlockIds.includes(annotation.id)}
                    disabled={
                      !assignmentEditable || Boolean(block.locked || annotation.status === "gold")
                    }
                    onChange={(event) =>
                      setCheckedBlockIds((current) =>
                        event.target.checked
                          ? [...current, annotation.id]
                          : current.filter((id) => id !== annotation.id),
                      )
                    }
                  />
                  merge
                </label>
                <button
                  type="button"
                  onClick={() => {
                    setSelectionIntent("block");
                    selectAnnotation(annotation.id);
                    setSelection(
                      block.start_sentence_id,
                      block.end_sentence_id,
                      block.start_sentence_ordinal ?? 0,
                      block.end_sentence_ordinal ?? 0,
                    );
                    setLabels(block.labels);
                    setNote(block.note ?? "");
                  }}
                >
                  <strong>
                    Sentences {(block.start_sentence_ordinal ?? 0) + 1}–{(block.end_sentence_ordinal ?? 0) + 1}
                  </strong>
                  <span>{annotation.status} · {annotation.source} · revision {block.revision ?? 1}</span>
                  {block.note ? <small>{block.note}</small> : null}
                </button>
              </article>
            );
          })}
          <EvidencePredictionPanel
            predictions={predictions}
            loading={predictionLoading}
            busy={busy}
            selectionAvailable={selection !== null}
            selectedPredictionId={selectedPredictionId}
            onSelect={choosePrediction}
            onReview={reviewPrediction}
            onRefresh={refreshPredictions}
          />
          <div className="eb-legend" aria-label="Evidence state legend">
            <span className="human">Human block</span>
            <span className="gold">Gold block</span>
            <span className="prediction">Prediction</span>
            <span className="reviewed">Reviewed</span>
            <span className="unreviewed">Unreviewed</span>
          </div>
        </aside>
      </div>
      <ConfirmDialog
        open={deleteDialogOpen}
        title="Delete evidence block?"
        description="Delete this evidence block? The change is recorded in command history and can be restored with Undo."
        confirmLabel="Delete block"
        busy={busy}
        onCancel={() => setDeleteDialogOpen(false)}
        onConfirm={confirmRemoveBlock}
      />
    </div>
  );
}
