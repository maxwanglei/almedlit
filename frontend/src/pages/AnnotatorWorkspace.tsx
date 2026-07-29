import {
  type Dispatch,
  type SetStateAction,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from "react";
import { Trash2 } from "lucide-react";

import {
  createAnnotation,
  createSubmission,
  deleteAnnotation,
  reopenPersonalTaskAssignment,
  updateAnnotation,
} from "@/api/client";
import EvidenceBlockCanvas from "@/features/evidence-block/EvidenceBlockCanvas";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import {
  allowedSearchValue,
  useSearchState,
} from "@/hooks/useSearchState";
import DialogFrame from "@/components/DialogFrame";
import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";
import {
  isHumanAnnotationOwnedBy,
  isMutableAssignment,
  isTaskEditableForAssignment,
  nextMutableAssignment,
  personalAssignmentStatusLabel,
  personalDocumentAction,
  personalDocumentStatusLabel,
  preferredAssignmentId,
} from "@/lib/annotationWorkspacePolicy";
import { visibleDocumentsFor } from "@/lib/annotatorIdentity";
import { filterRelationLabels, relationConstraintsOf } from "@/lib/relationConstraints";
import ActiveRoundQueue from "@/platform/ActiveRoundQueue";
import type { RoundWorkContext } from "@/platform/types";
import type {
  Annotation,
  AnnotationTypeSpec,
  AnnotationType,
  AnnotationWorkbench,
  AnnotationWorkbenchTask,
  Document,
  LabelDef,
  Project,
  ProjectProgress,
  ProjectTask,
  TaskAssignment,
  TaskAssignmentStatus,
} from "@/types/api";

type AnnotatorTab = "progress" | "annotate";
type AnnotationTool = "legacy" | "evidence";
export type SaveStatus = "saving" | "saved" | "error";
type WorkspacePane = "document" | "queue" | "tools" | "review";
type InspectorTab = "annotations" | "relations" | "sentence" | "guideline";
type ProjectSetupTab = "overview" | "documents" | "tasks" | "export" | "rounds";
type QueueStatus = "todo" | "partial" | "done" | "review" | "blocked";

const WORKSPACE_PANES = [
  "document",
  "queue",
  "tools",
  "review",
] as const satisfies readonly WorkspacePane[];
const INSPECTOR_TABS = [
  "annotations",
  "relations",
  "sentence",
  "guideline",
] as const satisfies readonly InspectorTab[];

function handleTabKey<T extends string>(
  event: ReactKeyboardEvent<HTMLButtonElement>,
  tabs: readonly T[],
  current: T,
  select: (tab: T) => void,
  idPrefix: string,
): void {
  let nextIndex: number | null = null;
  const currentIndex = tabs.indexOf(current);
  if (event.key === "ArrowRight") {
    nextIndex = (currentIndex + 1) % tabs.length;
  } else if (event.key === "ArrowLeft") {
    nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
  } else if (event.key === "Home") {
    nextIndex = 0;
  } else if (event.key === "End") {
    nextIndex = tabs.length - 1;
  }
  if (nextIndex === null) return;
  event.preventDefault();
  const next = tabs[nextIndex];
  select(next);
  window.requestAnimationFrame(() => {
    document.getElementById(`${idPrefix}-${next}`)?.focus();
  });
}

interface PendingSelection {
  annotationType: AnnotationType;
  start: number;
  end: number;
  text: string;
  left: number;
  top: number;
}

interface SpanMenuState {
  annotationId: number;
  x: number;
  y: number;
}

interface RelationChoiceState {
  fromId: number;
  toId: number;
  x: number;
  y: number;
}

interface RelationMenuState {
  relationId: number;
  x: number;
  y: number;
}

interface SentenceMenuState {
  sentenceIndex: number;
  x: number;
  y: number;
}

interface SentenceRange {
  index: number;
  start: number;
  end: number;
  text: string;
}

interface RelationArc {
  id: number;
  label: string;
  color: string;
  fromId: number;
  toId: number;
  ax: number;
  ay: number;
  bx: number;
  by: number;
}

interface AnnotatorWorkspaceProps {
  projects: Project[];
  selectedProject: Project | null;
  selectedProjectId: number | null;
  setSelectedProjectId: (projectId: number) => void;
  documents: Document[];
  assignments: TaskAssignment[];
  annotatorId: string;
  roundContexts?: RoundWorkContext[];
  projectProgress: ProjectProgress | null;
  selectedDocumentId: number | null;
  setSelectedDocumentId: (documentId: number | null) => void;
  workbench: AnnotationWorkbench | null;
  setWorkbench: Dispatch<SetStateAction<AnnotationWorkbench | null>>;
  busy: boolean;
  setBusy: (busy: boolean) => void;
  setError: (message: string | null) => void;
  refreshProjectData: (projectId: number) => Promise<void>;
  refreshWorkbench: (documentId: number) => Promise<void>;
  allowAssignmentlessSubmit?: boolean;
  onOpenProjectTab?: (tab: ProjectSetupTab) => void;
  onOpenRound?: (roundId: number) => void;
  /** Legacy caller compatibility; sign-out now belongs to AppShell. */
  onLogout?: () => void;
}

const STATUS_ORDER: QueueStatus[] = ["done", "partial", "review", "todo", "blocked"];
const DONE_STATUSES = new Set<TaskAssignmentStatus>(["submitted", "completed"]);
const REVIEW_STATUSES = new Set<TaskAssignmentStatus>(["adjudication_ready", "adjudicated"]);

const FALLBACK_COLORS = [
  "#4d6e5b",
  "#8e3a3a",
  "#8a6d2a",
  "#4a5a8a",
  "#6a4a7a",
  "#5e6e8a",
  "#c79a47",
];

const FALLBACK_TYPE_SPECS: Record<AnnotationType, AnnotationTypeSpec> = {
  entity: {
    name: "entity",
    requires_span: true,
    requires_head_tail: false,
    description: "Free-form character span with a single label.",
    selection_mode: "character_span",
    renderer_key: "legacy",
    relation_endpoint_allowed: true,
    handler_key: "generic",
  },
  relation: {
    name: "relation",
    requires_span: false,
    requires_head_tail: true,
    description: "Directed link from a head annotation to a tail annotation.",
    selection_mode: "relation",
    renderer_key: "legacy",
    relation_endpoint_allowed: false,
    handler_key: "generic",
  },
  doc_label: {
    name: "doc_label",
    requires_span: false,
    requires_head_tail: false,
    description: "A label on the whole document.",
    selection_mode: "none",
    renderer_key: "legacy",
    relation_endpoint_allowed: false,
    handler_key: "generic",
  },
  sentence_label: {
    name: "sentence_label",
    requires_span: true,
    requires_head_tail: false,
    description: "A label on a sentence span.",
    selection_mode: "character_span",
    renderer_key: "legacy",
    relation_endpoint_allowed: true,
    handler_key: "generic",
  },
  passage_label: {
    name: "passage_label",
    requires_span: true,
    requires_head_tail: false,
    description: "A label on an arbitrary multi-sentence span.",
    selection_mode: "character_span",
    renderer_key: "legacy",
    relation_endpoint_allowed: true,
    handler_key: "generic",
  },
  evidence_block: {
    name: "evidence_block",
    requires_span: false,
    requires_head_tail: false,
    description: "A contiguous range of complete sentences for a versioned evidence target.",
    selection_mode: "sentence_range",
    renderer_key: "evidence_block_v1",
    relation_endpoint_allowed: false,
    handler_key: "evidence_block_v1",
  },
};

const TASK_DISPLAY_NAMES: Record<AnnotationType, string> = {
  entity: "Entity labels",
  relation: "Relation",
  doc_label: "Document label",
  sentence_label: "Sentence label",
  passage_label: "Passage label",
  evidence_block: "Evidence blocks",
};

function isAnnotationType(value: string): value is AnnotationType {
  return value in FALLBACK_TYPE_SPECS;
}

function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

function sortTasks<T extends ProjectTask>(tasks: T[]): T[] {
  return [...tasks].sort((a, b) => a.sort_order - b.sort_order || a.id - b.id);
}

function toWorkbenchTask(
  task: ProjectTask,
  annotationTypeSpec?: AnnotationTypeSpec,
): AnnotationWorkbenchTask {
  return {
    ...task,
    annotation_type_spec: annotationTypeSpec ?? FALLBACK_TYPE_SPECS[task.annotation_type],
  };
}

function schemaTasks(project: Project | null): AnnotationWorkbenchTask[] {
  if (!project) {
    return [];
  }

  return Object.entries(project.annotation_schema.labels).flatMap(([annotationType, labels], index) => {
    if (!isAnnotationType(annotationType)) {
      return [];
    }
    return [
      {
        id: -(index + 1),
        project_id: project.id,
        annotation_type: annotationType,
        display_name: TASK_DISPLAY_NAMES[annotationType],
        description: null,
        enabled: true,
        sort_order: index,
        labels: labels ?? [],
        settings: {},
        annotation_type_spec: FALLBACK_TYPE_SPECS[annotationType],
      },
    ];
  });
}

function effectiveWorkbenchTasks(
  workbench: AnnotationWorkbench | null,
  selectedProject: Project | null,
): AnnotationWorkbenchTask[] {
  if (workbench?.tasks && workbench.tasks.length > 0) {
    return sortTasks(workbench.tasks);
  }

  const project = workbench?.project ?? selectedProject;
  const projectTasks = project?.tasks.filter((task) => task.enabled) ?? [];
  if (projectTasks.length > 0) {
    return sortTasks(projectTasks.map((task) => toWorkbenchTask(task)));
  }

  return sortTasks(schemaTasks(project));
}

function getTaskLabels(task: ProjectTask, project: Project | null): LabelDef[] {
  const taskLabels = task.labels ?? [];
  if (taskLabels.length > 0) {
    return taskLabels;
  }
  return project?.annotation_schema.labels[task.annotation_type] ?? [];
}

function labelColor(label: LabelDef | undefined, fallbackIndex = 0): string {
  return label?.color || FALLBACK_COLORS[fallbackIndex % FALLBACK_COLORS.length];
}

function fallbackLabelColor(annotationType: AnnotationType, labelName: string): string {
  const seed = `${annotationType}:${labelName}`;
  const hash = Array.from(seed).reduce((sum, character) => sum + character.charCodeAt(0), 0);
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
}

function findLabel(
  task: ProjectTask | undefined,
  project: Project | null,
  labelName: string,
): LabelDef | undefined {
  if (!task) {
    return undefined;
  }
  return getTaskLabels(task, project).find((label) => label.name === labelName);
}

function annotationHasOffsets(annotation: Annotation): annotation is Annotation & {
  start_offset: number;
  end_offset: number;
} {
  return (
    annotation.start_offset !== null &&
    annotation.end_offset !== null &&
    annotation.start_offset < annotation.end_offset
  );
}

function compactLabel(label: string): string {
  const words = label
    .replace(/[_-]+/g, " ")
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 0) {
    return "LBL";
  }
  if (words.length === 1) {
    return words[0].slice(0, 4).toUpperCase();
  }
  return words
    .slice(0, 3)
    .map((word) => word[0])
    .join("")
    .toUpperCase();
}

function displayStatus(status: string): string {
  return status.replace(/_/g, " ");
}

function documentTitle(documentItem: Document): string {
  return documentItem.title ?? documentItem.external_id ?? `Document ${documentItem.id}`;
}

function externalId(documentItem: Document): string {
  return documentItem.external_id ?? `DOC-${documentItem.id}`;
}

function statusForAssignments(assignments: TaskAssignment[]): QueueStatus {
  if (assignments.length === 0) {
    return "todo";
  }
  const currentAssignments = assignments.filter(
    (assignment) => assignment.status !== "withdrawn",
  );
  if (currentAssignments.length === 0) {
    return "blocked";
  }
  if (currentAssignments.some((assignment) => assignment.status === "blocked")) {
    return "blocked";
  }
  if (currentAssignments.some((assignment) => REVIEW_STATUSES.has(assignment.status))) {
    return "review";
  }
  if (currentAssignments.some((assignment) => assignment.status === "in_progress")) {
    return "partial";
  }
  if (currentAssignments.every((assignment) => DONE_STATUSES.has(assignment.status))) {
    return "done";
  }
  if (currentAssignments.some((assignment) => DONE_STATUSES.has(assignment.status))) {
    return "partial";
  }
  return "todo";
}

function sentenceRangesFor(documentItem: Document | null): SentenceRange[] {
  if (!documentItem) {
    return [];
  }

  const text = documentItem.text ?? "";
  const ranges = documentItem.sentences
    .map(([start, end], index) => ({
      index,
      start,
      end,
      text: text.slice(start, end),
    }))
    .filter((range) => range.start >= 0 && range.end > range.start && range.end <= text.length);

  if (ranges.length > 0) {
    return ranges;
  }

  return text.length > 0 ? [{ index: 0, start: 0, end: text.length, text }] : [];
}

function hexToRgba(hex: string, alpha: number): string {
  const normalized = hex.replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(normalized)) {
    return `rgba(77, 110, 91, ${alpha})`;
  }
  const red = Number.parseInt(normalized.slice(0, 2), 16);
  const green = Number.parseInt(normalized.slice(2, 4), 16);
  const blue = Number.parseInt(normalized.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function hotkeyForLabel(label: string, index: number): string {
  const letter = label.match(/[A-Za-z0-9]/)?.[0];
  return (letter ?? String(index + 1)).toUpperCase();
}

function isWordCharacter(character: string | undefined): boolean {
  return character !== undefined && /[A-Za-z0-9]/.test(character);
}

function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "recently";
  }
  const diffSeconds = Math.max(0, Math.round((Date.now() - date.getTime()) / 1000));
  if (diffSeconds < 60) {
    return "just now";
  }
  const diffMinutes = Math.round(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes} min ago`;
  }
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours} hr ago`;
  }
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`;
}

function sectionElFromNode(node: Node): HTMLElement | null {
  let element = node.nodeType === Node.TEXT_NODE ? node.parentElement : (node as HTMLElement);
  while (element && !element.dataset.sentenceIndex) {
    element = element.parentElement;
  }
  return element;
}

function charOffsetOf(root: HTMLElement, node: Node, nodeOffset: number): number | null {
  let count = 0;
  let found = false;

  function walk(current: Node): void {
    if (found) {
      return;
    }
    if (current.nodeType === Node.TEXT_NODE) {
      if (current === node) {
        count += nodeOffset;
        found = true;
        return;
      }
      count += current.textContent?.length ?? 0;
      return;
    }

    if (
      current instanceof HTMLElement &&
      (current.classList.contains("aw-span-badge") || current.getAttribute("aria-hidden") === "true")
    ) {
      return;
    }

    current.childNodes.forEach(walk);
  }

  walk(root);
  return found ? count : null;
}

function assignArcLevels(arcs: RelationArc[]): Record<number, number> {
  const sorted = [...arcs].sort((a, b) => Math.min(a.ax, a.bx) - Math.min(b.ax, b.bx));
  const trackEnd: number[] = [];
  const levels: Record<number, number> = {};

  sorted.forEach((arc) => {
    const low = Math.min(arc.ax, arc.bx);
    const high = Math.max(arc.ax, arc.bx);
    let track = 0;
    while (track < trackEnd.length && trackEnd[track] > low - 10) {
      track += 1;
    }
    trackEnd[track] = high;
    levels[arc.id] = track;
  });

  return levels;
}

function renderGuidelineMarkdown(markdown: string | null | undefined): React.ReactElement {
  if (!markdown) {
    return <p className="aw-empty">No active guideline.</p>;
  }

  return (
    <div className="aw-guideline-body">
      {markdown.split(/\n{2,}/).map((block, index) => {
        const trimmed = block.trim();
        if (!trimmed) {
          return null;
        }
        if (trimmed.startsWith("### ")) {
          return <h4 key={index}>{trimmed.slice(4)}</h4>;
        }
        if (trimmed.startsWith("## ")) {
          return <h3 key={index}>{trimmed.slice(3)}</h3>;
        }
        if (trimmed.startsWith("# ")) {
          return <h2 key={index}>{trimmed.slice(2)}</h2>;
        }
        if (trimmed.startsWith("- ")) {
          return (
            <ul key={index}>
              {trimmed.split("\n").map((line) => (
                <li key={line}>{line.replace(/^- /, "")}</li>
              ))}
            </ul>
          );
        }
        return <p key={index}>{trimmed}</p>;
      })}
    </div>
  );
}

export default function AnnotatorWorkspace({
  projects,
  selectedProject,
  selectedProjectId,
  setSelectedProjectId,
  documents,
  assignments,
  annotatorId,
  roundContexts = [],
  projectProgress,
  selectedDocumentId,
  setSelectedDocumentId,
  workbench,
  setWorkbench,
  busy,
  setBusy,
  setError,
  refreshProjectData,
  refreshWorkbench,
  allowAssignmentlessSubmit = false,
  onOpenProjectTab,
  onOpenRound,
}: AnnotatorWorkspaceProps): React.ReactElement {
  const [searchParams, updateSearch] = useSearchState();
  const initialView = allowedSearchValue(
    searchParams,
    "view",
    ["progress", "annotate"] as const,
    "progress",
  ).value;
  const initialTool = allowedSearchValue(
    searchParams,
    "tool",
    ["legacy", "evidence"] as const,
    "legacy",
  ).value;
  const initialPane = allowedSearchValue(
    searchParams,
    "pane",
    ["document", "queue", "tools", "review"] as const,
    "document",
  ).value;
  const [activeTab, setActiveTab] = useState<AnnotatorTab>(initialView);
  const [annotationTool, setAnnotationTool] =
    useState<AnnotationTool>(initialTool);
  const [workspacePane, setWorkspacePane] =
    useState<WorkspacePane>(initialPane);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>("saved");
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("annotations");
  const [activeSpanType, setActiveSpanType] = useState<AnnotationType | null>(null);
  const [activeSpanLabel, setActiveSpanLabel] = useState<string>("");
  const [customLabel, setCustomLabel] = useState<string>("");
  const [pendingSelection, setPendingSelection] = useState<PendingSelection | null>(null);
  const [spanMenu, setSpanMenu] = useState<SpanMenuState | null>(null);
  const [relationSourceId, setRelationSourceId] = useState<number | null>(null);
  const [relationChoice, setRelationChoice] = useState<RelationChoiceState | null>(null);
  const [relationMenu, setRelationMenu] = useState<RelationMenuState | null>(null);
  const [sentenceMenu, setSentenceMenu] = useState<SentenceMenuState | null>(null);
  const [selectedAnnotationId, setSelectedAnnotationId] = useState<number | null>(null);
  const [hoveredRelationId, setHoveredRelationId] = useState<number | null>(null);
  const [arcs, setArcs] = useState<RelationArc[]>([]);
  const [svgSize, setSvgSize] = useState({ width: 0, height: 0 });
  const [layoutTick, setLayoutTick] = useState(0);
  const [submissionAssignmentId, setSubmissionAssignmentId] = useState<number | null>(null);
  const [recentlyFinished, setRecentlyFinished] = useState<{
    documentId: number;
    assignmentId: number | null;
  } | null>(null);
  const [submissionDialogOpen, setSubmissionDialogOpen] = useState(false);
  const [blankSubmissionAcknowledged, setBlankSubmissionAcknowledged] =
    useState(false);
  const [reopenDialogOpen, setReopenDialogOpen] = useState(false);
  const isMobileWorkspace = useMediaQuery("(max-width: 1120px)");
  const previousAssignmentIdRef = useRef<number | null>(null);

  const readPaneRef = useRef<HTMLDivElement | null>(null);
  const spanRefs = useRef<Map<number, HTMLElement>>(new Map());
  const handleActiveEvidenceAssignmentChange = useCallback(
    (assignment: TaskAssignment | null): void => {
      setSubmissionAssignmentId(assignment?.id ?? null);
      updateSearch({ assignment: assignment?.id ?? null }, "replace");
    },
    [updateSearch],
  );

  const workbenchMatchesSelection =
    workbench !== null && workbench.document.id === selectedDocumentId;
  const activeWorkbench = workbenchMatchesSelection ? workbench : null;
  const isDocumentLoading = selectedDocumentId !== null && activeWorkbench === null;

  const tasks = useMemo(
    () => effectiveWorkbenchTasks(activeWorkbench, selectedProject),
    [activeWorkbench, selectedProject],
  );

  const tasksByType = useMemo(
    () => new Map<AnnotationType, AnnotationWorkbenchTask>(tasks.map((task) => [task.annotation_type, task])),
    [tasks],
  );

  const spanTasks = useMemo(
    () =>
      tasks.filter(
        (task) =>
          task.annotation_type_spec.requires_span &&
          task.annotation_type !== "sentence_label",
      ),
    [tasks],
  );

  const relationTasks = useMemo(
    () => tasks.filter((task) => task.annotation_type_spec.requires_head_tail),
    [tasks],
  );

  const sentenceTask = useMemo(
    () => tasks.find((task) => task.annotation_type === "sentence_label") ?? null,
    [tasks],
  );

  const docLabelTask = useMemo(
    () => tasks.find((task) => task.annotation_type === "doc_label") ?? null,
    [tasks],
  );

  const evidenceTask = useMemo(
    () => tasks.find((task) => task.annotation_type === "evidence_block") ?? null,
    [tasks],
  );

  const activeSpanTask = useMemo(
    () =>
      spanTasks.find((task) => task.annotation_type === activeSpanType) ??
      spanTasks[0] ??
      null,
    [activeSpanType, spanTasks],
  );

  const activeRelationTask = relationTasks[0] ?? null;
  const relationConstraints = useMemo(
    () => relationConstraintsOf(activeRelationTask),
    [activeRelationTask],
  );

  const activeProject = activeWorkbench?.project ?? selectedProject;
  const currentDocumentText = activeWorkbench?.document.text ?? "";
  const strictLabelsOnly = activeProject?.annotation_validation_mode === "strict";
  const currentSpanLabel = activeSpanLabel || customLabel.trim();
  const selectedAnnotatorId = annotatorId.trim();

  const currentAnnotatorAssignments = useMemo(
    () =>
      selectedAnnotatorId
        ? assignments.filter((assignment) => assignment.annotator_id === selectedAnnotatorId)
        : assignments,
    [assignments, selectedAnnotatorId],
  );

  const visibleDocuments = useMemo(
    () => visibleDocumentsFor(documents, assignments, selectedAnnotatorId),
    [assignments, documents, selectedAnnotatorId],
  );

  const sentenceRanges = useMemo(
    () => sentenceRangesFor(activeWorkbench?.document ?? null),
    [activeWorkbench],
  );

  const assignmentsByDocument = useMemo(() => {
    const grouped = new Map<number, TaskAssignment[]>();
    currentAnnotatorAssignments.forEach((assignment) => {
      const list = grouped.get(assignment.document_id) ?? [];
      list.push(assignment);
      grouped.set(assignment.document_id, list);
    });
    return grouped;
  }, [currentAnnotatorAssignments]);

  const documentStatuses = useMemo(() => {
    const statuses = new Map<number, QueueStatus>();
    visibleDocuments.forEach((documentItem) => {
      statuses.set(
        documentItem.id,
        statusForAssignments(assignmentsByDocument.get(documentItem.id) ?? []),
      );
    });
    return statuses;
  }, [assignmentsByDocument, visibleDocuments]);

  const selectedDocument =
    documents.find((documentItem) => documentItem.id === selectedDocumentId) ??
    activeWorkbench?.document ??
    null;
  const currentDocumentAssignments = useMemo(
    () => (selectedDocument ? assignmentsByDocument.get(selectedDocument.id) ?? [] : []),
    [assignmentsByDocument, selectedDocument],
  );
  const selectedSubmissionAssignment =
    currentDocumentAssignments.find((assignment) => assignment.id === submissionAssignmentId) ??
    currentDocumentAssignments.find((assignment) => isMutableAssignment(assignment)) ??
    null;
  const annotationEditable = selectedSubmissionAssignment
    ? isMutableAssignment(selectedSubmissionAssignment)
    : allowAssignmentlessSubmit && currentDocumentAssignments.length === 0;
  const annotationGuidelineVersionId =
    selectedSubmissionAssignment?.guideline_version_id ??
    activeWorkbench?.active_guideline?.id ??
    null;
  const annotationStructureVersionId =
    selectedSubmissionAssignment?.structure_version_id ??
    activeWorkbench?.document.active_structure_version_id ??
    null;
  const displayedGuideline =
    selectedSubmissionAssignment?.guideline_version_id
      ? activeWorkbench?.guideline_versions_by_id?.[
          selectedSubmissionAssignment.guideline_version_id
        ] ?? activeWorkbench?.active_guideline ?? null
      : activeWorkbench?.active_guideline ?? null;
  const canSubmitCurrentDocument =
    Boolean(workbenchMatchesSelection && selectedAnnotatorId) &&
    isMutableAssignment(selectedSubmissionAssignment);
  const selectedAssignmentTask =
    tasks.find((task) => task.id === selectedSubmissionAssignment?.task_id) ?? null;
  const selectedTaskAnnotationCount = useMemo(() => {
    if (
      !activeWorkbench ||
      !selectedSubmissionAssignment ||
      !selectedAssignmentTask ||
      !selectedAnnotatorId
    ) {
      return 0;
    }
    return activeWorkbench.annotations.filter((annotation) => {
      if (
        annotation.source !== "human" ||
        annotation.annotator_id !== selectedAnnotatorId ||
        annotation.annotation_type !== selectedAssignmentTask.annotation_type
      ) {
        return false;
      }
      if (
        selectedSubmissionAssignment.structure_version_id !== null &&
        annotation.structure_version_id !==
          selectedSubmissionAssignment.structure_version_id
      ) {
        return false;
      }
      if (
        annotation.guideline_version_id !==
        selectedSubmissionAssignment.guideline_version_id
      ) {
        return false;
      }
      if (selectedSubmissionAssignment.target_version_id !== null) {
        return (
          annotation.evidence_block?.target_version_id ===
          selectedSubmissionAssignment.target_version_id
        );
      }
      return true;
    }).length;
  }, [
    activeWorkbench,
    selectedAnnotatorId,
    selectedAssignmentTask,
    selectedSubmissionAssignment,
  ]);
  const isTaskEditable = useCallback(
    (taskId: number | null | undefined): boolean =>
      isTaskEditableForAssignment(
        selectedSubmissionAssignment,
        taskId,
        allowAssignmentlessSubmit,
        currentDocumentAssignments.length,
      ),
    [
      allowAssignmentlessSubmit,
      currentDocumentAssignments.length,
      selectedSubmissionAssignment,
    ],
  );
  const legacyAnnotationEditable =
    annotationEditable &&
    (selectedSubmissionAssignment === null ||
      (selectedAssignmentTask !== null &&
        selectedAssignmentTask.annotation_type !== "evidence_block"));

  const changeView = useCallback(
    (view: AnnotatorTab, mode: "push" | "replace" = "push"): void => {
      setActiveTab(view);
      updateSearch({ view }, mode);
    },
    [updateSearch],
  );

  const changeWorkspacePane = useCallback(
    (pane: WorkspacePane, mode: "push" | "replace" = "push"): void => {
      setWorkspacePane(pane);
      updateSearch({ pane }, mode);
    },
    [updateSearch],
  );

  const chooseAnnotationTool = useCallback(
    (tool: AnnotationTool): void => {
      const matchingAssignment = currentDocumentAssignments.find((assignment) => {
        const task = tasks.find((item) => item.id === assignment.task_id);
        return tool === "evidence"
          ? task?.annotation_type === "evidence_block"
          : task?.annotation_type !== "evidence_block";
      });
      if (matchingAssignment) {
        setSubmissionAssignmentId(matchingAssignment.id);
        updateSearch({ assignment: matchingAssignment.id }, "replace");
      }
      setAnnotationTool(tool);
      updateSearch({ tool }, "replace");
      changeWorkspacePane("document");
    },
    [
      changeWorkspacePane,
      currentDocumentAssignments,
      tasks,
      updateSearch,
    ],
  );

  useEffect(() => {
    const view = allowedSearchValue(
      searchParams,
      "view",
      ["progress", "annotate"] as const,
      "progress",
    );
    const tool = allowedSearchValue(
      searchParams,
      "tool",
      ["legacy", "evidence"] as const,
      "legacy",
    );
    const pane = allowedSearchValue(
      searchParams,
      "pane",
      ["document", "queue", "tools", "review"] as const,
      "document",
    );
    const invalidUpdate: Record<string, null> = {};
    if (!view.isValid) invalidUpdate.view = null;
    if (!tool.isValid) invalidUpdate.tool = null;
    if (!pane.isValid) invalidUpdate.pane = null;
    if (Object.keys(invalidUpdate).length > 0) {
      updateSearch(invalidUpdate, "replace");
    }

    setActiveTab(view.value);
    if (searchParams.has("tool")) {
      setAnnotationTool(tool.value);
    }
    setWorkspacePane(pane.value);

    const projectId = Number(searchParams.get("project"));
    if (
      Number.isInteger(projectId) &&
      projects.some((project) => project.id === projectId) &&
      projectId !== selectedProjectId
    ) {
      setSelectedProjectId(projectId);
    }
    const documentId = Number(searchParams.get("document"));
    if (
      Number.isInteger(documentId) &&
      visibleDocuments.some((document) => document.id === documentId) &&
      documentId !== selectedDocumentId
    ) {
      setSelectedDocumentId(documentId);
    }
    const assignmentId = Number(searchParams.get("assignment"));
    if (
      Number.isInteger(assignmentId) &&
      currentDocumentAssignments.some(
        (assignment) => assignment.id === assignmentId,
      )
    ) {
      setSubmissionAssignmentId(assignmentId);
    }
  }, [
    currentDocumentAssignments,
    projects,
    searchParams,
    selectedDocumentId,
    selectedProjectId,
    setSelectedDocumentId,
    setSelectedProjectId,
    updateSearch,
    visibleDocuments,
  ]);

  useEffect(() => {
    updateSearch(
      {
        project: selectedProjectId,
        document: selectedDocumentId,
        assignment: submissionAssignmentId,
      },
      "replace",
    );
  }, [
    selectedDocumentId,
    selectedProjectId,
    submissionAssignmentId,
    updateSearch,
  ]);

  useEffect(() => {
    setSubmissionAssignmentId((current) => {
      if (
        allowAssignmentlessSubmit &&
        recentlyFinished?.documentId === selectedDocumentId
      ) {
        const finishedAssignment = currentDocumentAssignments.find(
          (assignment) => assignment.id === recentlyFinished.assignmentId,
        );
        if (finishedAssignment) {
          return finishedAssignment.id;
        }
      }
      return preferredAssignmentId(currentDocumentAssignments, current);
    });
  }, [
    allowAssignmentlessSubmit,
    currentDocumentAssignments,
    recentlyFinished,
    selectedDocumentId,
  ]);

  useEffect(() => {
    setRecentlyFinished(null);
  }, [selectedProjectId]);

  useEffect(() => {
    setSubmissionDialogOpen(false);
    setBlankSubmissionAcknowledged(false);
    setReopenDialogOpen(false);
  }, [selectedDocumentId, submissionAssignmentId]);

  useEffect(() => {
    if (
      isMobileWorkspace &&
      activeTab === "annotate" &&
      selectedAnnotationId !== null
    ) {
      changeWorkspacePane("review", "replace");
    }
  }, [
    activeTab,
    changeWorkspacePane,
    isMobileWorkspace,
    selectedAnnotationId,
  ]);

  useEffect(() => {
    if (legacyAnnotationEditable) {
      return;
    }
    setPendingSelection(null);
    setSpanMenu(null);
    setRelationSourceId(null);
    setRelationChoice(null);
    setRelationMenu(null);
    setSentenceMenu(null);
  }, [legacyAnnotationEditable]);

  useEffect(() => {
    setPendingSelection(null);
    setSpanMenu(null);
    setRelationSourceId(null);
    setRelationChoice(null);
    setRelationMenu(null);
    setSentenceMenu(null);
  }, [selectedSubmissionAssignment?.id]);

  const humanAnnotations = useMemo(() => {
    const annotations = activeWorkbench?.annotations ?? [];
    return annotations.filter(
      (annotation) =>
        isHumanAnnotationOwnedBy(annotation, selectedAnnotatorId) &&
        (selectedSubmissionAssignment === null ||
          (annotation.structure_version_id ===
            selectedSubmissionAssignment.structure_version_id &&
            annotation.guideline_version_id ===
              selectedSubmissionAssignment.guideline_version_id)),
    );
  }, [activeWorkbench, selectedAnnotatorId, selectedSubmissionAssignment]);

  const spanAnnotationTypes = useMemo(
    () => new Set(spanTasks.map((task) => task.annotation_type)),
    [spanTasks],
  );

  const spanAnnotations = useMemo(
    () =>
      humanAnnotations
        .filter(
          (annotation) =>
            spanAnnotationTypes.has(annotation.annotation_type) && annotationHasOffsets(annotation),
        )
        .sort(
          (a, b) =>
            (a.start_offset ?? 0) - (b.start_offset ?? 0) ||
            (a.end_offset ?? 0) - (b.end_offset ?? 0) ||
            a.id - b.id,
        ),
    [humanAnnotations, spanAnnotationTypes],
  );

  const relationAnnotations = useMemo(
    () =>
      humanAnnotations.filter(
        (annotation) =>
          annotation.annotation_type === "relation" &&
          annotation.head_annotation_id !== null &&
          annotation.tail_annotation_id !== null,
      ),
    [humanAnnotations],
  );

  const sentenceLabelAnnotations = useMemo(
    () =>
      humanAnnotations.filter(
        (annotation) => annotation.annotation_type === "sentence_label" && annotationHasOffsets(annotation),
      ),
    [humanAnnotations],
  );

  const docLabelAnnotations = useMemo(
    () => humanAnnotations.filter((annotation) => annotation.annotation_type === "doc_label"),
    [humanAnnotations],
  );

  const correctionLockedIds = useMemo(
    () => new Set(activeWorkbench?.correction_locked_annotation_ids ?? []),
    [activeWorkbench],
  );

  const relationLockedIds = useMemo(() => {
    const locked = new Set<number>();
    relationAnnotations.forEach((relation) => {
      if (relation.head_annotation_id !== null) {
        locked.add(relation.head_annotation_id);
      }
      if (relation.tail_annotation_id !== null) {
        locked.add(relation.tail_annotation_id);
      }
    });
    return locked;
  }, [relationAnnotations]);

  const spanById = useMemo(
    () => new Map(spanAnnotations.map((annotation) => [annotation.id, annotation])),
    [spanAnnotations],
  );

  const labelCounts = useMemo(() => {
    const counts = new Map<string, number>();
    humanAnnotations.forEach((annotation) => {
      const key = `${annotation.annotation_type}:${annotation.label}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    });
    return counts;
  }, [humanAnnotations]);

  const displayLabelsForTask = useCallback(
    (task: ProjectTask): LabelDef[] => {
      const configuredLabels = getTaskLabels(task, activeWorkbench?.project ?? selectedProject);
      if (strictLabelsOnly) {
        return configuredLabels;
      }

      const configuredNames = new Set(configuredLabels.map((label) => label.name));
      const observedLabels = Array.from(
        new Set(
          humanAnnotations
            .filter((annotation) => annotation.annotation_type === task.annotation_type)
            .map((annotation) => annotation.label),
        ),
      )
        .filter((labelName) => !configuredNames.has(labelName))
        .sort((a, b) => a.localeCompare(b))
        .map((labelName) => ({
          name: labelName,
          color: fallbackLabelColor(task.annotation_type, labelName),
          description: "Ad hoc label used on this document.",
        }));

      return [...configuredLabels, ...observedLabels];
    },
    [activeWorkbench, humanAnnotations, selectedProject, strictLabelsOnly],
  );

  const activeSpanLabels = useMemo(
    () => (activeSpanTask ? displayLabelsForTask(activeSpanTask) : []),
    [activeSpanTask, displayLabelsForTask],
  );

  const relationChoiceHead = relationChoice ? spanById.get(relationChoice.fromId) : undefined;
  const relationChoiceTail = relationChoice ? spanById.get(relationChoice.toId) : undefined;
  const relationChoiceLabels = useMemo(() => {
    if (!activeRelationTask) {
      return [];
    }
    const labels = displayLabelsForTask(activeRelationTask);
    if (!relationChoiceHead || !relationChoiceTail) {
      return labels;
    }
    return filterRelationLabels(
      labels,
      relationConstraints,
      relationChoiceHead.label,
      relationChoiceTail.label,
    );
  }, [
    activeRelationTask,
    displayLabelsForTask,
    relationChoiceHead,
    relationChoiceTail,
    relationConstraints,
  ]);

  const sentenceLabelsByRange = useMemo(() => {
    const labels = new Map<string, Annotation>();
    sentenceLabelAnnotations.forEach((annotation) => {
      labels.set(`${annotation.start_offset}:${annotation.end_offset}`, annotation);
    });
    return labels;
  }, [sentenceLabelAnnotations]);

  const progressCounts = useMemo(() => {
    const counts: Record<QueueStatus, number> = {
      todo: 0,
      partial: 0,
      done: 0,
      review: 0,
      blocked: 0,
    };
    visibleDocuments.forEach((documentItem) => {
      counts[documentStatuses.get(documentItem.id) ?? "todo"] += 1;
    });
    return counts;
  }, [documentStatuses, visibleDocuments]);

  const completedDocumentCount = progressCounts.done;
  const progressTotal = visibleDocuments.length;
  const documentBadge =
    progressTotal > 0 ? `${completedDocumentCount}/${progressTotal}` : "0/0";
  const enabledProjectTaskCount = selectedProject?.tasks.filter((task) => task.enabled).length ?? 0;
  const enabledProjectTasks = selectedProject?.tasks.filter((task) => task.enabled) ?? [];
  const hasEnabledEntityTask = enabledProjectTasks.some(
    (task) => task.annotation_type === "entity",
  );
  const personalTaskSetupIssue = !allowAssignmentlessSubmit || !selectedProject
    ? null
    : enabledProjectTasks.find(
          (task) =>
            selectedProject.annotation_validation_mode === "strict" &&
            task.annotation_type !== "evidence_block" &&
            getTaskLabels(task, selectedProject).length === 0,
        )
      ? "Add at least one label to every enabled task before annotating in strict mode."
      : enabledProjectTasks.some(
            (task) => task.annotation_type === "relation" && !hasEnabledEntityTask,
          )
        ? "Add and enable an entity task before using relation annotation."
        : documents.length > 0 &&
            enabledProjectTasks.some(
              (task) =>
                task.annotation_type === "evidence_block" &&
                !currentAnnotatorAssignments.some(
                  (assignment) => assignment.task_id === task.id,
                ),
            )
          ? "Configure an active evidence target before annotating evidence blocks."
          : null;
  const nextOpenDocument =
    visibleDocuments.find((documentItem) => {
      const status = documentStatuses.get(documentItem.id) ?? "todo";
      return status !== "done";
    }) ??
    visibleDocuments[0] ??
    null;
  const nextOpenTaskAssignment = allowAssignmentlessSubmit
    ? nextMutableAssignment(
        currentDocumentAssignments,
        selectedSubmissionAssignment?.id ?? null,
      )
    : null;
  const selectedVisibleDocumentIndex = selectedDocument
    ? visibleDocuments.findIndex((documentItem) => documentItem.id === selectedDocument.id)
    : -1;
  const followingPersonalDocuments =
    selectedVisibleDocumentIndex >= 0
      ? [
          ...visibleDocuments.slice(selectedVisibleDocumentIndex + 1),
          ...visibleDocuments.slice(0, selectedVisibleDocumentIndex),
        ]
      : visibleDocuments;
  const nextPersonalDocument =
    followingPersonalDocuments.find(
      (documentItem) =>
        documentItem.id !== selectedDocumentId &&
        (documentStatuses.get(documentItem.id) ?? "todo") !== "done",
    ) ?? null;
  const isPersonalReview =
    allowAssignmentlessSubmit &&
    selectedDocument !== null &&
    !annotationEditable;
  const isRecentlyFinishedReview =
    isPersonalReview &&
    recentlyFinished?.documentId === selectedDocumentId &&
    (recentlyFinished.assignmentId === null ||
      recentlyFinished.assignmentId === selectedSubmissionAssignment?.id);
  const personalHomeState = !selectedProject
    ? "no_project"
    : documents.length === 0
      ? "no_documents"
      : enabledProjectTaskCount === 0
        ? "no_tasks"
        : personalTaskSetupIssue
          ? "task_setup"
        : progressTotal > 0 && completedDocumentCount >= progressTotal
          ? "complete"
          : "ready";
  const personalNextDetail =
    personalHomeState === "no_project"
      ? "Start a private project for the papers you want to annotate."
      : personalHomeState === "no_documents"
        ? `${selectedProject?.name ?? "This project"} is ready for PMIDs.`
        : personalHomeState === "no_tasks"
          ? "Enable at least one annotation task."
          : personalHomeState === "task_setup"
            ? personalTaskSetupIssue ?? "Complete task setup before annotating."
          : personalHomeState === "complete"
            ? `${completedDocumentCount} of ${progressTotal} documents are finished.`
            : nextOpenDocument
              ? documentTitle(nextOpenDocument)
              : "Your next document is ready.";
  const personalNextButton =
    personalHomeState === "no_project"
      ? "Create project"
      : personalHomeState === "no_documents"
        ? "Import PMIDs"
        : personalHomeState === "no_tasks" || personalHomeState === "task_setup"
          ? "Configure tasks"
          : personalHomeState === "complete"
            ? "Review work"
            : nextOpenDocument
              ? personalDocumentAction(
                  documentStatuses.get(nextOpenDocument.id) ?? "todo",
                )
              : "Start";

  useEffect(() => {
    if (!selectedAnnotatorId) {
      return;
    }
    if (visibleDocuments.length === 0) {
      if (selectedDocumentId !== null) {
        setSelectedDocumentId(null);
      }
      return;
    }
    if (
      selectedDocumentId === null ||
      !visibleDocuments.some((documentItem) => documentItem.id === selectedDocumentId)
    ) {
      setSelectedDocumentId(visibleDocuments[0].id);
    }
  }, [selectedAnnotatorId, selectedDocumentId, setSelectedDocumentId, visibleDocuments]);

  useEffect(() => {
    const firstSpanTask = spanTasks[0] ?? null;
    if (!firstSpanTask) {
      setActiveSpanType(null);
      setActiveSpanLabel("");
      setCustomLabel("");
      return;
    }

    setActiveSpanType((previous) =>
      previous && spanTasks.some((task) => task.annotation_type === previous)
        ? previous
        : firstSpanTask.annotation_type,
    );
  }, [spanTasks]);

  useEffect(() => {
    const assignmentChanged =
      previousAssignmentIdRef.current !== selectedSubmissionAssignment?.id;
    const expectedTool: AnnotationTool =
      selectedAssignmentTask?.annotation_type === "evidence_block"
        ? "evidence"
        : "legacy";
    const currentToolIsIncompatible =
      (annotationTool === "evidence" &&
        selectedAssignmentTask?.annotation_type !== "evidence_block") ||
      (annotationTool === "legacy" &&
        selectedAssignmentTask?.annotation_type === "evidence_block") ||
      (annotationTool === "evidence" && !evidenceTask);

    if (assignmentChanged || currentToolIsIncompatible) {
      setAnnotationTool(expectedTool);
      updateSearch({ tool: expectedTool }, "replace");
    }
    previousAssignmentIdRef.current = selectedSubmissionAssignment?.id ?? null;
  }, [
    annotationTool,
    evidenceTask,
    selectedAssignmentTask?.annotation_type,
    selectedSubmissionAssignment?.id,
    updateSearch,
  ]);

  useEffect(() => {
    if (!activeSpanTask) {
      return;
    }
    const labels = displayLabelsForTask(activeSpanTask);
    setActiveSpanLabel((previous) =>
      labels.some((label) => label.name === previous) ? previous : labels[0]?.name ?? "",
    );
    if (strictLabelsOnly) {
      setCustomLabel("");
    }
  }, [activeSpanTask, displayLabelsForTask, strictLabelsOnly]);

  const runMutation = useCallback(
    async (
      operation: () => Promise<void>,
      failureMessage: string,
      taskId?: number | null,
    ): Promise<void> => {
      if (!annotationEditable) {
        setError("This annotation round is read-only because it has already been submitted or closed.");
        return;
      }
      if (taskId !== undefined && !isTaskEditable(taskId)) {
        setError("Select an open assignment for this annotation task before editing it.");
        return;
      }
      setBusy(true);
      setSaveStatus("saving");
      setError(null);
      try {
        await operation();
        setSaveStatus("saved");
      } catch (error) {
        setSaveStatus("error");
        setError(error instanceof Error ? error.message : failureMessage);
      } finally {
        setBusy(false);
      }
    },
    [annotationEditable, isTaskEditable, setBusy, setError],
  );

  const patchWorkbenchAnnotation = useCallback(
    (annotation: Annotation): void => {
      setWorkbench((current) =>
        current && current.document.id === selectedDocumentId
          ? {
              ...current,
              annotations: current.annotations.map((item) =>
                item.id === annotation.id ? annotation : item,
              ),
            }
          : current,
      );
    },
    [selectedDocumentId, setWorkbench],
  );

  const addWorkbenchAnnotation = useCallback(
    (annotation: Annotation): void => {
      setWorkbench((current) =>
        current && current.document.id === selectedDocumentId
          ? {
              ...current,
              annotations: [annotation, ...current.annotations],
            }
          : current,
      );
    },
    [selectedDocumentId, setWorkbench],
  );

  const removeWorkbenchAnnotations = useCallback(
    (ids: number[]): void => {
      const idSet = new Set(ids);
      setWorkbench((current) =>
        current && current.document.id === selectedDocumentId
          ? {
              ...current,
              annotations: current.annotations.filter((annotation) => !idSet.has(annotation.id)),
            }
          : current,
      );
    },
    [selectedDocumentId, setWorkbench],
  );

  const applyEvidenceAnnotationChanges = useCallback(
    (upserted: Annotation[], removedIds: number[] = []): void => {
      const removed = new Set(removedIds);
      const upsertById = new Map(upserted.map((annotation) => [annotation.id, annotation]));
      setWorkbench((current) => {
        if (!current || current.document.id !== selectedDocumentId) {
          return current;
        }
        const retained = current.annotations
          .filter((annotation) => !removed.has(annotation.id) && !upsertById.has(annotation.id));
        return { ...current, annotations: [...upserted, ...retained] };
      });
    },
    [selectedDocumentId, setWorkbench],
  );

  function setSpanRef(annotationId: number, element: HTMLElement | null): void {
    if (element) {
      spanRefs.current.set(annotationId, element);
    } else {
      spanRefs.current.delete(annotationId);
    }
  }

  const taskColor = useCallback((annotationType: AnnotationType, labelName: string): string => {
    const task = tasksByType.get(annotationType);
    const labels = task ? getTaskLabels(task, activeProject) : [];
    const foundIndex = labels.findIndex((label) => label.name === labelName);
    if (foundIndex >= 0) {
      return labelColor(labels[foundIndex], foundIndex);
    }
    return fallbackLabelColor(annotationType, labelName);
  }, [activeProject, tasksByType]);

  function offsetText(annotation: Annotation): string {
    if (annotation.start_offset === null || annotation.end_offset === null) {
      return "whole doc";
    }
    return `${annotation.start_offset}-${annotation.end_offset}`;
  }

  function spanText(annotation: Annotation): string {
    if (annotation.text_span) {
      return annotation.text_span;
    }
    if (activeWorkbench?.document && annotationHasOffsets(annotation)) {
      return activeWorkbench.document.text.slice(annotation.start_offset, annotation.end_offset);
    }
    return annotation.label;
  }

  function sentenceAnnotation(range: SentenceRange): Annotation | undefined {
    return sentenceLabelsByRange.get(`${range.start}:${range.end}`);
  }

  function handleSelection(): void {
    if (
      !isTaskEditable(activeSpanTask?.id) ||
      relationSourceId !== null ||
      !activeSpanTask ||
      !activeWorkbench
    ) {
      return;
    }

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed || selection.rangeCount === 0) {
      setPendingSelection(null);
      return;
    }

    const range = selection.getRangeAt(0);
    const startSection = sectionElFromNode(range.startContainer);
    const endSection = sectionElFromNode(range.endContainer);
    if (!startSection || !endSection || startSection !== endSection) {
      setPendingSelection(null);
      return;
    }

    const sentenceIndex = Number(startSection.dataset.sentenceIndex);
    const sentence = sentenceRanges.find((item) => item.index === sentenceIndex);
    if (!sentence) {
      setPendingSelection(null);
      return;
    }

    const localStart = charOffsetOf(startSection, range.startContainer, range.startOffset);
    const localEnd = charOffsetOf(endSection, range.endContainer, range.endOffset);
    if (localStart === null || localEnd === null || localStart === localEnd) {
      setPendingSelection(null);
      return;
    }

    let start = sentence.start + Math.min(localStart, localEnd);
    let end = sentence.start + Math.max(localStart, localEnd);
    const text = activeWorkbench.document.text;
    while (start < end && /\s/.test(text[start])) {
      start += 1;
    }
    while (end > start && /\s/.test(text[end - 1])) {
      end -= 1;
    }
    while (
      start > sentence.start &&
      isWordCharacter(text[start - 1]) &&
      isWordCharacter(text[start])
    ) {
      start -= 1;
    }
    while (
      end < sentence.end &&
      isWordCharacter(text[end - 1]) &&
      isWordCharacter(text[end])
    ) {
      end += 1;
    }
    if (start >= end) {
      setPendingSelection(null);
      return;
    }

    const rect = range.getBoundingClientRect();
    setPendingSelection({
      annotationType: activeSpanTask.annotation_type,
      start,
      end,
      text: text.slice(start, end),
      left: rect.left + rect.width / 2,
      top: rect.top,
    });
  }

  const createSpan = useCallback(async (annotationType: AnnotationType, labelName: string): Promise<void> => {
    if (!activeWorkbench || !pendingSelection || !labelName) {
      return;
    }
    if (!selectedAnnotatorId) {
      setError("Select an annotator before creating annotations.");
      return;
    }

    await runMutation(async () => {
      const created = await createAnnotation({
        project_id: activeWorkbench.project.id,
        document_id: activeWorkbench.document.id,
        annotation_type: annotationType,
        label: labelName,
        start_offset: pendingSelection.start,
        end_offset: pendingSelection.end,
        text_span: pendingSelection.text,
        source: "human",
        status: "draft",
        annotator_id: selectedAnnotatorId,
        guideline_version_id: annotationGuidelineVersionId,
        structure_version_id: annotationStructureVersionId,
      });
      addWorkbenchAnnotation(created);
      setSelectedAnnotationId(created.id);
      setActiveSpanType(annotationType);
      setActiveSpanLabel(labelName);
      setCustomLabel("");
      setPendingSelection(null);
      window.getSelection()?.removeAllRanges();
    }, "Unable to create annotation", tasksByType.get(annotationType)?.id ?? null);
  }, [activeWorkbench, addWorkbenchAnnotation, annotationGuidelineVersionId, annotationStructureVersionId, pendingSelection, runMutation, selectedAnnotatorId, setError, tasksByType]);

  async function relabelSpan(annotationId: number, nextLabel: string): Promise<void> {
    const annotation = spanById.get(annotationId);
    const taskId = annotation ? tasksByType.get(annotation.annotation_type)?.id : null;
    await runMutation(async () => {
      const updated = await updateAnnotation(annotationId, { label: nextLabel });
      patchWorkbenchAnnotation(updated);
      setActiveSpanLabel(nextLabel);
      setSpanMenu(null);
    }, "Unable to update annotation", taskId);
  }

  const deleteSpan = useCallback(async (annotationId: number): Promise<void> => {
    if (correctionLockedIds.has(annotationId)) {
      setError("This annotation is referenced by a correction and cannot be deleted.");
      return;
    }
    if (relationLockedIds.has(annotationId)) {
      setError("Delete the referencing relation from its own assignment before deleting this annotation.");
      return;
    }
    const annotation = spanById.get(annotationId);
    const taskId = annotation ? tasksByType.get(annotation.annotation_type)?.id : null;

    await runMutation(async () => {
      await deleteAnnotation(annotationId);
      removeWorkbenchAnnotations([annotationId]);
      setSpanMenu(null);
      setSelectedAnnotationId(null);
      setRelationSourceId(null);
      setRelationChoice(null);
      setRelationMenu(null);
    }, "Unable to delete annotation", taskId);
  }, [correctionLockedIds, relationLockedIds, removeWorkbenchAnnotations, runMutation, setError, spanById, tasksByType]);

  async function deleteRelation(annotationId: number): Promise<void> {
    await runMutation(async () => {
      await deleteAnnotation(annotationId);
      removeWorkbenchAnnotations([annotationId]);
      setRelationMenu(null);
      if (hoveredRelationId === annotationId) {
        setHoveredRelationId(null);
      }
    }, "Unable to delete relation", activeRelationTask?.id ?? null);
  }

  async function relabelRelation(annotationId: number, nextLabel: string): Promise<void> {
    await runMutation(async () => {
      const updated = await updateAnnotation(annotationId, { label: nextLabel });
      patchWorkbenchAnnotation(updated);
      setRelationMenu(null);
    }, "Unable to update relation", activeRelationTask?.id ?? null);
  }

  async function swapRelationDirection(relation: Annotation): Promise<void> {
    await runMutation(async () => {
      const updated = await updateAnnotation(relation.id, {
        head_annotation_id: relation.tail_annotation_id,
        tail_annotation_id: relation.head_annotation_id,
      });
      patchWorkbenchAnnotation(updated);
      setRelationMenu(null);
    }, "Unable to swap relation direction", activeRelationTask?.id ?? null);
  }

  const createRelation = useCallback(async (labelName: string): Promise<void> => {
    if (!activeWorkbench || !relationChoice || !labelName) {
      return;
    }
    if (!selectedAnnotatorId) {
      setError("Select an annotator before creating annotations.");
      return;
    }

    await runMutation(async () => {
      const created = await createAnnotation({
        project_id: activeWorkbench.project.id,
        document_id: activeWorkbench.document.id,
        annotation_type: "relation",
        label: labelName,
        head_annotation_id: relationChoice.fromId,
        tail_annotation_id: relationChoice.toId,
        source: "human",
        status: "draft",
        annotator_id: selectedAnnotatorId,
        guideline_version_id: annotationGuidelineVersionId,
        structure_version_id: annotationStructureVersionId,
      });
      addWorkbenchAnnotation(created);
      setSelectedAnnotationId(relationChoice.toId);
      setRelationSourceId(null);
      setRelationChoice(null);
    }, "Unable to create relation", activeRelationTask?.id ?? null);
  }, [activeRelationTask, activeWorkbench, addWorkbenchAnnotation, annotationGuidelineVersionId, annotationStructureVersionId, relationChoice, runMutation, selectedAnnotatorId, setError]);

  async function setDocumentLabel(labelName: string): Promise<void> {
    if (!activeWorkbench || !docLabelTask) {
      return;
    }
    if (!selectedAnnotatorId) {
      setError("Select an annotator before creating annotations.");
      return;
    }

    const active = docLabelAnnotations.find((annotation) => annotation.label === labelName);
    await runMutation(async () => {
      if (active) {
        await deleteAnnotation(active.id);
        removeWorkbenchAnnotations([active.id]);
        return;
      }

      const previous = docLabelAnnotations.map((annotation) => annotation.id);
      for (const annotationId of previous) {
        await deleteAnnotation(annotationId);
      }
      const created = await createAnnotation({
        project_id: activeWorkbench.project.id,
        document_id: activeWorkbench.document.id,
        annotation_type: docLabelTask.annotation_type,
        label: labelName,
        source: "human",
        status: "draft",
        annotator_id: selectedAnnotatorId,
        guideline_version_id: annotationGuidelineVersionId,
        structure_version_id: annotationStructureVersionId,
      });
      setWorkbench((current) =>
        current && current.document.id === activeWorkbench.document.id
          ? {
              ...current,
              annotations: [
                created,
                ...current.annotations.filter((annotation) => !previous.includes(annotation.id)),
              ],
            }
          : current,
      );
    }, "Unable to update document label", docLabelTask.id);
  }

  async function setSentenceLabel(range: SentenceRange, labelName: string | null): Promise<void> {
    if (!activeWorkbench || !sentenceTask) {
      return;
    }
    if (!selectedAnnotatorId) {
      setError("Select an annotator before creating annotations.");
      return;
    }

    const existing = sentenceAnnotation(range);
    await runMutation(async () => {
      if (existing) {
        await deleteAnnotation(existing.id);
      }
      if (labelName === null) {
        if (existing) {
          removeWorkbenchAnnotations([existing.id]);
        }
        setSentenceMenu(null);
        return;
      }

      const created = await createAnnotation({
        project_id: activeWorkbench.project.id,
        document_id: activeWorkbench.document.id,
        annotation_type: sentenceTask.annotation_type,
        label: labelName,
        start_offset: range.start,
        end_offset: range.end,
        text_span: range.text,
        source: "human",
        status: "draft",
        annotator_id: selectedAnnotatorId,
        guideline_version_id: annotationGuidelineVersionId,
        structure_version_id: annotationStructureVersionId,
      });
      setWorkbench((current) =>
        current && current.document.id === activeWorkbench.document.id
          ? {
              ...current,
              annotations: [
                created,
                ...current.annotations.filter((annotation) => annotation.id !== existing?.id),
              ],
            }
          : current,
      );
      setSentenceMenu(null);
    }, "Unable to update sentence label", sentenceTask.id);
  }

  async function submitCurrentDocument(): Promise<void> {
    if (!activeWorkbench || selectedProjectId === null) {
      return;
    }
    if (activeWorkbench.document.id !== selectedDocumentId) {
      setError("The selected document is still loading.");
      return;
    }
    if (!selectedAnnotatorId) {
      setError(
        allowAssignmentlessSubmit
          ? "Sign in before finishing this task."
          : "Select an annotator before submitting.",
      );
      return;
    }
    if (!allowAssignmentlessSubmit && currentDocumentAssignments.length === 0) {
      setError("This document is not assigned to the selected annotator.");
      return;
    }
    if (selectedSubmissionAssignment === null) {
      setError(
        allowAssignmentlessSubmit
          ? "Select the paper task you want to submit."
          : "Select the target or task assignment to submit.",
      );
      return;
    }
    const finishedAssignmentId = selectedSubmissionAssignment?.id ?? null;
    const finishedDocumentId = activeWorkbench.document.id;
    await runMutation(async () => {
      await createSubmission(selectedProjectId, activeWorkbench.document.id, {
        annotator_id: selectedAnnotatorId,
        assignment_id: finishedAssignmentId,
      });
      setSubmissionDialogOpen(false);
      setBlankSubmissionAcknowledged(false);
      await refreshProjectData(selectedProjectId);
      await refreshWorkbench(activeWorkbench.document.id);
      if (allowAssignmentlessSubmit) {
        setSubmissionAssignmentId(finishedAssignmentId);
        setRecentlyFinished({
          documentId: finishedDocumentId,
          assignmentId: finishedAssignmentId,
        });
      }
    }, allowAssignmentlessSubmit ? "Unable to finish task" : "Unable to submit document");
  }

  async function reopenCurrentPaperTask(): Promise<void> {
    if (
      !allowAssignmentlessSubmit ||
      selectedProjectId === null ||
      selectedSubmissionAssignment === null ||
      selectedSubmissionAssignment.status !== "submitted" ||
      selectedDocumentId === null
    ) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await reopenPersonalTaskAssignment(
        selectedProjectId,
        selectedSubmissionAssignment.id,
      );
      await refreshProjectData(selectedProjectId);
      await refreshWorkbench(selectedDocumentId);
      setRecentlyFinished(null);
      setSubmissionAssignmentId(selectedSubmissionAssignment.id);
      setReopenDialogOpen(false);
    } catch (error) {
      setError(
        error instanceof Error
          ? error.message
          : "Unable to reopen this paper task for editing.",
      );
    } finally {
      setBusy(false);
    }
  }

  function selectDocument(documentId: number, nextTab: AnnotatorTab = activeTab): void {
    setRecentlyFinished(null);
    setSelectedDocumentId(documentId);
    setActiveTab(nextTab);
    setWorkspacePane("document");
    updateSearch(
      {
        document: documentId,
        view: nextTab,
        pane: "document",
      },
      "push",
    );
    setPendingSelection(null);
    setSpanMenu(null);
    setRelationChoice(null);
    setRelationMenu(null);
    setRelationSourceId(null);
    setSelectedAnnotationId(null);
  }

  function moveDocument(direction: 1 | -1): void {
    if (visibleDocuments.length === 0 || selectedDocumentId === null) {
      return;
    }
    const currentIndex = Math.max(
      0,
      visibleDocuments.findIndex((documentItem) => documentItem.id === selectedDocumentId),
    );
    const nextIndex = Math.min(visibleDocuments.length - 1, Math.max(0, currentIndex + direction));
    selectDocument(visibleDocuments[nextIndex].id, "annotate");
  }

  function openNextPersonalWork(): void {
    setRecentlyFinished(null);
    if (nextOpenTaskAssignment) {
      setSubmissionAssignmentId(nextOpenTaskAssignment.id);
      updateSearch(
        { assignment: nextOpenTaskAssignment.id },
        "replace",
      );
      return;
    }
    if (nextPersonalDocument) {
      selectDocument(nextPersonalDocument.id, "annotate");
      return;
    }
    changeView("progress");
  }

  function runPersonalNextAction(): void {
    if (personalHomeState === "no_project") {
      onOpenProjectTab?.("overview");
      return;
    }
    if (personalHomeState === "no_documents") {
      onOpenProjectTab?.("documents");
      return;
    }
    if (personalHomeState === "no_tasks" || personalHomeState === "task_setup") {
      onOpenProjectTab?.("tasks");
      return;
    }
    if (personalHomeState === "complete") {
      if (visibleDocuments[0]) {
        selectDocument(visibleDocuments[0].id, "annotate");
      }
      return;
    }
    if (nextOpenDocument) {
      selectDocument(nextOpenDocument.id, "annotate");
    }
  }

  function handleSpanClick(annotationId: number, event: React.MouseEvent<HTMLElement>): void {
    event.stopPropagation();
    setSelectedAnnotationId(annotationId);
    setRelationMenu(null);
    if (relationSourceId !== null) {
      if (!isTaskEditable(activeRelationTask?.id)) {
        return;
      }
      if (relationSourceId !== annotationId) {
        setRelationChoice({
          fromId: relationSourceId,
          toId: annotationId,
          x: event.clientX,
          y: event.clientY,
        });
      }
      return;
    }
    const annotation = spanById.get(annotationId);
    const taskId = annotation ? tasksByType.get(annotation.annotation_type)?.id : null;
    if (isTaskEditable(taskId)) {
      setSpanMenu({ annotationId, x: event.clientX, y: event.clientY });
    }
  }

  function handleSpanKeyDown(
    annotationId: number,
    event: React.KeyboardEvent<HTMLElement>,
  ): void {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    event.stopPropagation();
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = bounds.left + Math.min(bounds.width / 2, 24);
    const y = bounds.bottom;
    setSelectedAnnotationId(annotationId);
    setRelationMenu(null);
    if (relationSourceId !== null) {
      if (isTaskEditable(activeRelationTask?.id) && relationSourceId !== annotationId) {
        setRelationChoice({
          fromId: relationSourceId,
          toId: annotationId,
          x,
          y,
        });
      }
      return;
    }
    const annotation = spanById.get(annotationId);
    const taskId = annotation ? tasksByType.get(annotation.annotation_type)?.id : null;
    if (isTaskEditable(taskId)) {
      setSpanMenu({ annotationId, x, y });
    }
  }

  function renderSectionText(range: SentenceRange): React.ReactNode {
    const sectionSpans = spanAnnotations
      .filter(
        (annotation) =>
          annotationHasOffsets(annotation) &&
          annotation.start_offset < range.end &&
          annotation.end_offset > range.start,
      )
      .sort((a, b) => (a.start_offset ?? 0) - (b.start_offset ?? 0) || a.id - b.id);

    if (sectionSpans.length === 0) {
      return currentDocumentText.slice(range.start, range.end);
    }

    const boundaries = new Set<number>([range.start, range.end]);
    sectionSpans.forEach((annotation) => {
      if (annotationHasOffsets(annotation)) {
        boundaries.add(Math.max(range.start, annotation.start_offset));
        boundaries.add(Math.min(range.end, annotation.end_offset));
      }
    });

    const points = Array.from(boundaries).sort((a, b) => a - b);
    return points.flatMap((start, index) => {
      const end = points[index + 1];
      if (end === undefined || start === end) {
        return [];
      }

      const covering = sectionSpans
        .filter(
          (annotation) =>
            annotationHasOffsets(annotation) &&
            annotation.start_offset <= start &&
            annotation.end_offset >= end,
        )
        .sort((a, b) => {
          const aLength = (a.end_offset ?? 0) - (a.start_offset ?? 0);
          const bLength = (b.end_offset ?? 0) - (b.start_offset ?? 0);
          return aLength - bLength || b.id - a.id;
        });

      if (covering.length === 0) {
        return [<span key={`text-${start}-${end}`}>{currentDocumentText.slice(start, end)}</span>];
      }

      const primary = covering[0];
      const color = taskColor(primary.annotation_type, primary.label);
      return [
        <span
          className={cx(
            "aw-entity",
            covering.length > 1 && "stacked",
            covering.some((annotation) => selectedAnnotationId === annotation.id) && "selected",
            covering.some((annotation) => relationSourceId === annotation.id) && "relation-source",
          )}
          key={`ann-${start}-${end}-${covering.map((annotation) => annotation.id).join("-")}`}
          ref={(element) => {
            covering.forEach((annotation) => setSpanRef(annotation.id, element));
          }}
          style={
            {
              "--aw-label-color": color,
              backgroundColor: hexToRgba(color, 0.12),
              borderBottomColor: color,
            } as React.CSSProperties
          }
          role="button"
          tabIndex={0}
          aria-label={`${primary.label}: ${currentDocumentText.slice(start, end)}`}
          onClick={(event) => handleSpanClick(primary.id, event)}
          onKeyDown={(event) => handleSpanKeyDown(primary.id, event)}
          onContextMenu={(event) => {
            event.preventDefault();
            event.stopPropagation();
            if (
              relationSourceId === null &&
              isTaskEditable(tasksByType.get(primary.annotation_type)?.id)
            ) {
              setSelectedAnnotationId(primary.id);
              setSpanMenu({ annotationId: primary.id, x: event.clientX, y: event.clientY });
            }
          }}
          title={covering.map((annotation) => annotation.label).join(", ")}
        >
          <span className="aw-span-badge" aria-hidden="true">
            {compactLabel(primary.label)}
            {covering.length > 1 ? ` +${covering.length - 1}` : ""}
          </span>
          {currentDocumentText.slice(start, end)}
        </span>,
      ];
    });
  }

  useLayoutEffect(() => {
    const container = readPaneRef.current;
    if (!container) {
      return;
    }

    const containerRect = container.getBoundingClientRect();
    const nextArcs = relationAnnotations.flatMap((relation) => {
      if (relation.head_annotation_id === null || relation.tail_annotation_id === null) {
        return [];
      }
      const head = spanRefs.current.get(relation.head_annotation_id);
      const tail = spanRefs.current.get(relation.tail_annotation_id);
      if (!head || !tail) {
        return [];
      }
      const headRect = head.getBoundingClientRect();
      const tailRect = tail.getBoundingClientRect();
      const color = taskColor(relation.annotation_type, relation.label);
      return [
        {
          id: relation.id,
          label: relation.label,
          color,
          fromId: relation.head_annotation_id,
          toId: relation.tail_annotation_id,
          ax: headRect.left - containerRect.left + container.scrollLeft + headRect.width / 2,
          ay: headRect.top - containerRect.top + container.scrollTop,
          bx: tailRect.left - containerRect.left + container.scrollLeft + tailRect.width / 2,
          by: tailRect.top - containerRect.top + container.scrollTop,
        },
      ];
    });

    setArcs((previous) => {
      const same =
        previous.length === nextArcs.length &&
        previous.every((item, index) => {
          const next = nextArcs[index];
          return (
            next &&
            item.id === next.id &&
            item.ax === next.ax &&
            item.ay === next.ay &&
            item.bx === next.bx &&
            item.by === next.by
          );
        });
      return same ? previous : nextArcs;
    });
    setSvgSize((previous) =>
      previous.width === container.scrollWidth && previous.height === container.scrollHeight
        ? previous
        : { width: container.scrollWidth, height: container.scrollHeight },
    );
  }, [layoutTick, relationAnnotations, selectedDocumentId, spanAnnotations, taskColor]);

  useEffect(() => {
    const container = readPaneRef.current;
    if (!container) {
      return undefined;
    }
    const bump = (): void => setLayoutTick((tick) => tick + 1);
    const resizeObserver = new ResizeObserver(bump);
    resizeObserver.observe(container);
    container.addEventListener("scroll", bump);
    window.addEventListener("resize", bump);
    return () => {
      resizeObserver.disconnect();
      container.removeEventListener("scroll", bump);
      window.removeEventListener("resize", bump);
    };
  }, [activeTab, selectedDocumentId]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent): void {
      if (submissionDialogOpen || reopenDialogOpen) {
        return;
      }
      const activeElement = document.activeElement;
      const tagName = activeElement?.tagName ?? "";
      const inTextInput = tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
      if (inTextInput) {
        return;
      }
      const selectedSpan =
        selectedAnnotationId !== null ? spanById.get(selectedAnnotationId) ?? null : null;

      if (event.key === "1" && !pendingSelection && !relationChoice) {
        changeView("annotate");
        return;
      }
      if (event.key === "2" && !pendingSelection && !relationChoice) {
        changeView("progress");
        return;
      }

      if (event.key === "Escape") {
        setPendingSelection(null);
        setSpanMenu(null);
        setRelationSourceId(null);
        setRelationChoice(null);
        setRelationMenu(null);
        setSentenceMenu(null);
        window.getSelection()?.removeAllRanges();
        return;
      }

      if (
        isTaskEditable(activeRelationTask?.id) &&
        (event.key === "r" || event.key === "R") &&
        !pendingSelection &&
        !relationChoice &&
        relationSourceId === null &&
        selectedAnnotationId !== null &&
        spanById.has(selectedAnnotationId) &&
        activeRelationTask !== null
      ) {
        event.preventDefault();
        setRelationSourceId(selectedAnnotationId);
        setSpanMenu(null);
        setRelationMenu(null);
        return;
      }

      if (isTaskEditable(activeSpanTask?.id) && pendingSelection && activeSpanTask) {
        const label = displayLabelsForTask(activeSpanTask).find(
          (item, index) => hotkeyForLabel(item.name, index).toLowerCase() === event.key.toLowerCase(),
        );
        if (label) {
          event.preventDefault();
          void createSpan(activeSpanTask.annotation_type, label.name);
        }
        return;
      }

      if (isTaskEditable(activeRelationTask?.id) && relationChoice && activeRelationTask) {
        const label = relationChoiceLabels.find(
          (item, index) => hotkeyForLabel(item.name, index).toLowerCase() === event.key.toLowerCase(),
        );
        if (label) {
          event.preventDefault();
          void createRelation(label.name);
        }
        return;
      }

      if (
        selectedSpan !== null &&
        isTaskEditable(tasksByType.get(selectedSpan.annotation_type)?.id) &&
        (event.key === "Backspace" || event.key === "Delete") &&
        selectedAnnotationId !== null
      ) {
        event.preventDefault();
        void deleteSpan(selectedAnnotationId);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    activeRelationTask,
    activeSpanTask,
    changeView,
    createRelation,
    createSpan,
    deleteSpan,
    displayLabelsForTask,
    isTaskEditable,
    pendingSelection,
    relationChoice,
    relationChoiceLabels,
    relationSourceId,
    reopenDialogOpen,
    selectedAnnotationId,
    spanById,
    submissionDialogOpen,
    tasksByType,
  ]);

  const currentSpanMenuAnnotation = spanMenu ? spanById.get(spanMenu.annotationId) : undefined;
  const spanMenuTask = currentSpanMenuAnnotation
    ? tasksByType.get(currentSpanMenuAnnotation.annotation_type)
    : undefined;
  const sentenceMenuRange =
    sentenceMenu !== null
      ? sentenceRanges.find((range) => range.index === sentenceMenu.sentenceIndex)
      : undefined;
  const relationMenuAnnotation = relationMenu
    ? relationAnnotations.find((relation) => relation.id === relationMenu.relationId)
    : undefined;
  const relationMenuHead =
    relationMenuAnnotation?.head_annotation_id != null
      ? spanById.get(relationMenuAnnotation.head_annotation_id)
      : undefined;
  const relationMenuTail =
    relationMenuAnnotation?.tail_annotation_id != null
      ? spanById.get(relationMenuAnnotation.tail_annotation_id)
      : undefined;

  const activeDocLabel = docLabelAnnotations[0] ?? null;
  const relationLevels = assignArcLevels(arcs);

  return (
    <div
      className="aw"
      data-density="balanced"
    >
      <div className="aw-tabs-row">
        {allowAssignmentlessSubmit ? (
          <nav className="aw-tabs" aria-label="Personal work navigation">
            {activeTab === "annotate" ? (
              <button
                className="aw-tab"
                type="button"
                onClick={() => changeView("progress")}
              >
                Back to My Work
                <span className="aw-tab-badge">{documentBadge}</span>
              </button>
            ) : (
              <span className="aw-tabs-spacer" aria-hidden="true" />
            )}
          </nav>
        ) : (
          <div className="aw-tabs" role="group" aria-label="Annotator views">
            <button
              className={cx("aw-tab", activeTab === "annotate" && "active")}
              id="aw-tab-annotate"
              aria-pressed={activeTab === "annotate"}
              type="button"
              onClick={() => changeView("annotate")}
            >
              Annotate
              <kbd>1</kbd>
            </button>
            <button
              className={cx("aw-tab", activeTab === "progress" && "active")}
              id="aw-tab-progress"
              aria-pressed={activeTab === "progress"}
              type="button"
              onClick={() => changeView("progress")}
            >
              My Progress
              <span className="aw-tab-badge">{documentBadge}</span>
              <kbd>2</kbd>
            </button>
          </div>
        )}

        <div className="aw-context">
          <select
            aria-label="Project"
            className="aw-project-select"
            value={selectedProjectId ?? ""}
            onChange={(event) => {
              const nextProjectId = Number(event.target.value);
              if (Number.isFinite(nextProjectId)) {
                setSelectedProjectId(nextProjectId);
                updateSearch(
                  {
                    project: nextProjectId,
                    document: null,
                    assignment: null,
                  },
                  "push",
                );
              }
            }}
            disabled={projects.length === 0 || busy}
          >
            {projects.map((project) => (
              <option key={project.id} value={project.id}>
                {project.name}
              </option>
            ))}
          </select>
          {activeTab === "annotate" ? (
            <>
              <span>/</span>
              <strong>{selectedDocument ? documentTitle(selectedDocument) : "No document selected"}</strong>
              {selectedDocument ? <span className="aw-pmid">{externalId(selectedDocument)}</span> : null}
            </>
          ) : (
            <>
              <span>/</span>
              <strong>{allowAssignmentlessSubmit ? "My documents" : "My assignments"}</strong>
              <span>
                {allowAssignmentlessSubmit
                  ? `${completedDocumentCount} of ${progressTotal} done`
                  : `${currentAnnotatorAssignments.length} of ${
                      projectProgress?.total ?? currentAnnotatorAssignments.length
                    } tasks`}
              </span>
            </>
          )}
          <span
            className={cx("aw-save-pill", saveStatus)}
            role="status"
            aria-live="polite"
            aria-atomic="true"
          >
            <span className="aw-save-dot" />
            {saveStatus === "saving"
              ? "Saving…"
              : saveStatus === "error"
                ? "Save failed"
                : "Saved"}
          </span>
        </div>

        <div className="aw-primary-actions">
          {activeTab === "annotate" ? (
            <>
              {!isPersonalReview ? (
                <>
                  <button type="button" onClick={() => moveDocument(-1)} disabled={busy || visibleDocuments.length <= 1}>
                    Prev
                  </button>
                  <button type="button" onClick={() => moveDocument(1)} disabled={busy || visibleDocuments.length <= 1}>
                    Next
                  </button>
                </>
              ) : null}
              {currentDocumentAssignments.length > 0 ? (
                <label className="aw-submit-control">
                  <span>{allowAssignmentlessSubmit ? "Task" : "Assignment"}</span>
                  <select
                    aria-label={allowAssignmentlessSubmit ? "Task" : "Assignment to submit"}
                    className="aw-submit-assignment"
                    value={submissionAssignmentId ?? ""}
                    onChange={(event) => {
                      setRecentlyFinished(null);
                      const assignmentId = Number(event.target.value);
                      setSubmissionAssignmentId(assignmentId);
                      updateSearch({ assignment: assignmentId }, "replace");
                    }}
                    disabled={busy}
                  >
                    {currentDocumentAssignments.map((assignment) => {
                      const task = tasks.find((item) => item.id === assignment.task_id);
                      const target = assignment.target_version_id
                        ? ` · target v#${assignment.target_version_id}`
                        : "";
                      const status = allowAssignmentlessSubmit
                        ? personalAssignmentStatusLabel(assignment.status)
                        : assignment.status;
                      return (
                        <option key={assignment.id} value={assignment.id}>
                          {task?.display_name ?? `Task #${assignment.task_id}`}{target} · {status}
                        </option>
                      );
                    })}
                  </select>
                </label>
              ) : null}
              {isPersonalReview ? (
                <>
                  <button type="button" onClick={() => changeView("progress")}>
                    Back to My Work
                  </button>
                  {selectedSubmissionAssignment?.status === "submitted" ? (
                    <button
                      type="button"
                      onClick={() => setReopenDialogOpen(true)}
                      disabled={busy}
                    >
                      Edit this paper task
                    </button>
                  ) : null}
                  {nextOpenTaskAssignment || nextPersonalDocument ? (
                    <button
                      className="primary"
                      type="button"
                      onClick={openNextPersonalWork}
                      disabled={busy}
                    >
                      {nextOpenTaskAssignment ? "Next task" : "Next document"}
                    </button>
                  ) : null}
                </>
              ) : (
                <button
                  className="primary"
                  type="button"
                  onClick={() => {
                    setBlankSubmissionAcknowledged(false);
                    setSubmissionDialogOpen(true);
                  }}
                  disabled={busy || !canSubmitCurrentDocument}
                >
                  {allowAssignmentlessSubmit
                    ? "Submit task for this paper"
                    : "Submit assignment"}
                </button>
              )}
            </>
          ) : !allowAssignmentlessSubmit ? (
            <button
              className="primary"
              type="button"
              onClick={() => {
                if (selectedDocumentId !== null) {
                  changeView("annotate");
                }
              }}
              disabled={
                allowAssignmentlessSubmit ? selectedDocumentId === null : selectedDocumentId === null || !selectedAnnotatorId
              }
            >
              Resume
            </button>
          ) : null}
        </div>
      </div>

      <main id="main-content" className="aw-main" tabIndex={-1}>
      {activeTab === "progress" ? (
          <div
            className="aw-progress"
            data-workspace-kind={
              allowAssignmentlessSubmit ? "personal" : "team"
            }
            id={!allowAssignmentlessSubmit ? "aw-panel-progress" : undefined}
            role={!allowAssignmentlessSubmit ? "region" : undefined}
            aria-labelledby={!allowAssignmentlessSubmit ? "aw-tab-progress" : undefined}
          >
            <div className="aw-progress-wrap">
              <div className={cx("aw-progress-head", allowAssignmentlessSubmit && "personal")}>
                <div>
                  <h1>{allowAssignmentlessSubmit ? "My Work" : "My Progress"}</h1>
                  <p>
                    {allowAssignmentlessSubmit
                      ? personalNextDetail
                      : `${selectedProject?.name ?? "Project"} - ${completedDocumentCount} of ${progressTotal} documents done`}
                  </p>
                </div>
                <div className="aw-progress-head-actions">
                  {allowAssignmentlessSubmit ? (
                    selectedProject ? (
                      <a
                        className="aw-secondary-action"
                        href={`/projects/${selectedProject.id}/overview`}
                        onClick={(event) => {
                          if (!shouldHandleSpaClick(event)) {
                            return;
                          }
                          event.preventDefault();
                          onOpenProjectTab?.("overview");
                        }}
                      >
                        Project setup
                      </a>
                    ) : null
                  ) : null}
                  <button
                    className="aw-resume"
                    type="button"
                    onClick={() => {
                      if (allowAssignmentlessSubmit) {
                        runPersonalNextAction();
                      } else {
                        changeView("annotate");
                      }
                    }}
                    disabled={
                      allowAssignmentlessSubmit
                        ? personalHomeState === "ready" && nextOpenDocument === null
                        : selectedDocumentId === null || !selectedAnnotatorId
                    }
                  >
                    {allowAssignmentlessSubmit ? personalNextButton : "Resume annotating"}
                  </button>
                </div>
              </div>

              <div className="aw-stat-grid">
                <article className="aw-stat-card">
                  <span>{allowAssignmentlessSubmit ? "Documents" : "Documents done"}</span>
                  <strong>
                    {completedDocumentCount} <small>/ {progressTotal}</small>
                  </strong>
                  <em>
                    {allowAssignmentlessSubmit
                      ? `${documents.length} imported`
                      : `${currentAnnotatorAssignments.length} assignments`}
                  </em>
                </article>
                <article className="aw-stat-card">
                  <span>{allowAssignmentlessSubmit ? "Tasks" : "Annotations on document"}</span>
                  {allowAssignmentlessSubmit ? (
                    <>
                      <strong>{enabledProjectTaskCount}</strong>
                      <em>enabled tasks</em>
                    </>
                  ) : (
                    <>
                      <strong>{humanAnnotations.length}</strong>
                      <em>
                        {selectedDocument
                          ? externalId(selectedDocument)
                          : "no document selected"}
                      </em>
                    </>
                  )}
                </article>
                <article className="aw-stat-card">
                  <span>{allowAssignmentlessSubmit ? "Current document" : "Agreement"}</span>
                  {allowAssignmentlessSubmit ? (
                    <>
                      <strong>{humanAnnotations.length}</strong>
                      <em>{selectedDocument ? "annotations" : "no document selected"}</em>
                    </>
                  ) : (
                    <>
                      <strong>Not available</strong>
                      <em>No agreement result loaded</em>
                    </>
                  )}
                </article>
              </div>

              {roundContexts.length ? (
                <ActiveRoundQueue
                  contexts={roundContexts}
                  onOpenRound={(roundId) => onOpenRound?.(roundId)}
                />
              ) : null}

              <div className="aw-progress-grid">
                <section className="aw-panel aw-batch-panel">
                  <h2>
                    {allowAssignmentlessSubmit ? "Document queue" : "Your batch"}{" "}
                    <span>{visibleDocuments.length} documents</span>
                  </h2>
                  <p>
                    {allowAssignmentlessSubmit
                      ? selectedProject?.name ?? "No project selected"
                      : "Pick up where you left off, or jump to any document."}
                  </p>
                <div
                  className="aw-meter"
                  role="progressbar"
                  aria-label="Batch completion"
                  aria-valuemin={0}
                  aria-valuemax={Math.max(progressTotal, 1)}
                  aria-valuenow={completedDocumentCount}
                  aria-valuetext={`${completedDocumentCount} of ${progressTotal} documents complete`}
                >
                  {STATUS_ORDER.map((status) => {
                    const width =
                      progressTotal > 0 ? (progressCounts[status] / progressTotal) * 100 : 0;
                    return width > 0 ? (
                      <i className={status} key={status} style={{ width: `${width}%` }} />
                    ) : null;
                  })}
                </div>
                <div className="aw-legend">
                  {STATUS_ORDER.filter((status) => status !== "blocked" || progressCounts.blocked > 0).map(
                    (status) => (
                      <span key={status}>
                        <i className={status} />
                        {allowAssignmentlessSubmit
                          ? personalDocumentStatusLabel(status)
                          : status === "partial"
                            ? "In progress"
                            : status}
                        <b>{progressCounts[status]}</b>
                      </span>
                    ),
                  )}
                </div>

                  <div className="aw-progress-docs">
                    {visibleDocuments.length === 0 ? (
                      <p className="aw-empty">
                        {allowAssignmentlessSubmit
                          ? selectedProject
                            ? "No documents imported."
                            : "No project selected."
                          : "No assignments for this annotator."}
                      </p>
                    ) : (
                      visibleDocuments.map((documentItem, index) => {
                        const status = documentStatuses.get(documentItem.id) ?? "todo";
                        const selected = selectedDocumentId === documentItem.id;
                        const docAssignments = assignmentsByDocument.get(documentItem.id) ?? [];
                        return (
                          <button
                            className={cx("aw-progress-doc", selected && "selected")}
                            key={documentItem.id}
                            type="button"
                            onClick={() =>
                              selectDocument(
                                documentItem.id,
                                allowAssignmentlessSubmit ? "annotate" : "progress",
                              )
                            }
                          >
                            <span className="aw-doc-check">
                              {selected ? "OK" : String(index + 1).padStart(2, "0")}
                            </span>
                            <span className="aw-doc-info">
                              <strong>{documentTitle(documentItem)}</strong>
                              <small>{externalId(documentItem)}</small>
                            </span>
                            <span className="aw-mini-status">
                              {allowAssignmentlessSubmit
                                ? personalDocumentAction(status)
                                : docAssignments.length === 0
                                  ? "No assignments"
                                  : `${docAssignments.length} assignment${docAssignments.length === 1 ? "" : "s"}`}
                            </span>
                            <span className={cx("aw-status-dot", status)} />
                          </button>
                        );
                      })
                    )}
                  </div>
                </section>

                <section className="aw-panel aw-feed">
                  <h2>{allowAssignmentlessSubmit ? "Recent annotations" : "Recent activity"}</h2>
                  <p>
                    {allowAssignmentlessSubmit
                      ? selectedDocument
                        ? documentTitle(selectedDocument)
                        : "No document selected"
                      : selectedDocument
                        ? documentTitle(selectedDocument)
                        : "No document selected"}
                  </p>
                {(humanAnnotations.length > 0 ? humanAnnotations.slice(0, 5) : []).map((annotation) => (
                  <article className="aw-event" key={annotation.id}>
                    <span className="aw-event-rail">
                      <i />
                    </span>
                    <span>
                      <strong>
                        {annotation.annotation_type === "relation"
                          ? `Created relation ${annotation.label}`
                          : `Labeled "${spanText(annotation)}" as ${annotation.label}`}
                      </strong>
                      <small>
                        {formatRelativeTime(annotation.updated_at)} -{" "}
                        {selectedDocument ? externalId(selectedDocument) : "selected document"}
                      </small>
                    </span>
                  </article>
                ))}
                  {humanAnnotations.length === 0 && allowAssignmentlessSubmit ? (
                    <p className="aw-feed-empty">No annotations on the selected document yet.</p>
                  ) : null}
                  {humanAnnotations.length === 0 && !allowAssignmentlessSubmit ? (
                    <p className="aw-feed-empty">
                      No recent annotation activity is available.
                    </p>
                  ) : null}
                </section>
              </div>
          </div>
        </div>
      ) : (
        <div
          className="aw-workspace"
          id={!allowAssignmentlessSubmit ? "aw-panel-annotate" : undefined}
          role={!allowAssignmentlessSubmit ? "region" : undefined}
          aria-labelledby={!allowAssignmentlessSubmit ? "aw-tab-annotate" : undefined}
        >
          <div className="aw-workspace-heading">
            <h1>
              {selectedDocument
                ? documentTitle(selectedDocument)
                : "Annotation workspace"}
            </h1>
          </div>
          <div className="aw-pane-tabs" role="tablist" aria-label="Workspace panes">
            {WORKSPACE_PANES.map(
              (pane) => (
                <button
                  key={pane}
                  id={`aw-pane-tab-${pane}`}
                  type="button"
                  role="tab"
                  aria-selected={workspacePane === pane}
                  tabIndex={workspacePane === pane ? 0 : -1}
                  aria-controls={
                    !isMobileWorkspace || workspacePane === pane
                      ? `aw-pane-${pane}`
                      : undefined
                  }
                  onClick={() => changeWorkspacePane(pane)}
                  onKeyDown={(event) =>
                    handleTabKey(
                      event,
                      WORKSPACE_PANES,
                      workspacePane,
                      changeWorkspacePane,
                      "aw-pane-tab",
                    )
                  }
                >
                  {pane[0].toUpperCase() + pane.slice(1)}
                </button>
              ),
            )}
          </div>
          <div
            className={cx(
              "aw-annotate",
              annotationTool === "evidence" && "evidence-mode",
            )}
          >
          {!isMobileWorkspace || workspacePane === "queue" ? (
          <aside
            id="aw-pane-queue"
            className="aw-queue"
            role="tabpanel"
            aria-labelledby="aw-pane-tab-queue"
          >
            <h2>
              Queue <span>{documentBadge}</span>
            </h2>
            <div className="aw-queue-list">
              {visibleDocuments.map((documentItem, index) => {
                const status = documentStatuses.get(documentItem.id) ?? "todo";
                return (
                  <button
                    className={cx("aw-queue-row", selectedDocumentId === documentItem.id && "active")}
                    key={documentItem.id}
                    type="button"
                    aria-current={selectedDocumentId === documentItem.id ? "true" : undefined}
                    aria-label={`${documentTitle(documentItem)} · ${personalDocumentStatusLabel(status)}`}
                    onClick={() => selectDocument(documentItem.id, "annotate")}
                  >
                    <span>{String(index + 1).padStart(2, "0")}</span>
                    <strong>{documentTitle(documentItem)}</strong>
                    <i className={cx("aw-status-dot", status)} aria-hidden="true" />
                  </button>
                );
              })}
            </div>
            <button className="aw-queue-manage" type="button" onClick={() => changeView("progress")}>
              {allowAssignmentlessSubmit ? "View My Work queue" : "Manage queue in My Progress"}
            </button>
          </aside>
          ) : null}

          {!isMobileWorkspace || workspacePane === "tools" ? (
          <aside
            id="aw-pane-tools"
            className="aw-labels"
            role="tabpanel"
            aria-labelledby="aw-pane-tab-tools"
            onClickCapture={(event) => {
              if (
                isMobileWorkspace &&
                event.target instanceof Element &&
                event.target.closest("button")
              ) {
                changeWorkspacePane("document", "replace");
              }
            }}
          >
            <h2 className="aw-pane-heading">Tools</h2>
            <section className="aw-label-group aw-tool-switch" aria-label="Annotation tool">
              <h3>Annotation tool</h3>
              <button
                className={cx("aw-label-row", annotationTool === "legacy" && "active")}
                type="button"
                aria-pressed={annotationTool === "legacy"}
                onClick={() => chooseAnnotationTool("legacy")}
              >
                <span>Spans &amp; relations</span>
                {annotationTool === "legacy" ? <b>ON</b> : null}
              </button>
              <button
                className={cx("aw-label-row", annotationTool === "evidence" && "active")}
                type="button"
                aria-pressed={annotationTool === "evidence"}
                onClick={() => chooseAnnotationTool("evidence")}
                disabled={!evidenceTask}
              >
                <span>Evidence blocks</span>
                {annotationTool === "evidence" ? <b>ON</b> : null}
              </button>
            </section>
            {tasks.length === 0 ? <p className="aw-empty">No enabled annotation tasks.</p> : null}
            {tasks.map((task) => {
              const labels = displayLabelsForTask(task);
              const heading = task.display_name || displayStatus(task.annotation_type);
              if (task.annotation_type_spec.requires_head_tail) {
                return (
                  <section className="aw-label-group" key={task.id}>
                    <h3>{heading}</h3>
                    {labels.length === 0 ? <p className="aw-label-hint">No labels configured.</p> : null}
                    {labels.map((label, index) => (
                      <button
                        className="aw-label-row"
                        key={label.name}
                        type="button"
                        onClick={() => {
                          if (relationChoice) {
                            void createRelation(label.name);
                          }
                        }}
                        disabled={
                          !isTaskEditable(task.id) ||
                          (relationChoice !== null &&
                            !relationChoiceLabels.some((item) => item.name === label.name))
                        }
                      >
                        <i style={{ backgroundColor: labelColor(label, index) }} />
                        <span>{label.name}</span>
                        <kbd>{hotkeyForLabel(label.name, index)}</kbd>
                      </button>
                    ))}
                  </section>
                );
              }

              if (task.annotation_type === "doc_label") {
                return (
                  <section className="aw-label-group" key={task.id}>
                    <h3>{heading}</h3>
                    {labels.map((label, index) => (
                      <button
                        className={cx("aw-label-row", activeDocLabel?.label === label.name && "active")}
                        key={label.name}
                        type="button"
                        aria-pressed={activeDocLabel?.label === label.name}
                        onClick={() => void setDocumentLabel(label.name)}
                        disabled={!isTaskEditable(task.id)}
                      >
                        <i style={{ backgroundColor: labelColor(label, index) }} />
                        <span>{label.name}</span>
                        {activeDocLabel?.label === label.name ? <b>OK</b> : <kbd>Alt {index + 1}</kbd>}
                      </button>
                    ))}
                  </section>
                );
              }

              if (task.annotation_type === "sentence_label") {
                return (
                  <section className="aw-label-group" key={task.id}>
                    <h3>{heading}</h3>
                    {labels.map((label, index) => (
                      <div className="aw-label-row static" key={label.name}>
                        <i style={{ backgroundColor: labelColor(label, index) }} />
                        <span>{label.name}</span>
                        <span className="aw-count">
                          {labelCounts.get(`${task.annotation_type}:${label.name}`) ?? 0}
                        </span>
                      </div>
                    ))}
                  </section>
                );
              }

              if (task.annotation_type_spec.requires_span) {
                return (
                  <section className="aw-label-group" key={task.id}>
                    <h3>{heading}</h3>
                    {labels.map((label, index) => (
                      <button
                        className={cx(
                          "aw-label-row",
                          activeSpanTask?.annotation_type === task.annotation_type &&
                            currentSpanLabel === label.name &&
                            "active",
                        )}
                        key={label.name}
                        type="button"
                        aria-pressed={
                          activeSpanTask?.annotation_type === task.annotation_type &&
                          currentSpanLabel === label.name
                        }
                        onClick={() => {
                          setActiveSpanType(task.annotation_type);
                          setActiveSpanLabel(label.name);
                          setCustomLabel("");
                          if (pendingSelection) {
                            void createSpan(task.annotation_type, label.name);
                          }
                        }}
                        disabled={!isTaskEditable(task.id)}
                      >
                        <i style={{ backgroundColor: labelColor(label, index) }} />
                        <span>{label.name}</span>
                        <span className="aw-label-tail">
                          <span className="aw-count">
                            {labelCounts.get(`${task.annotation_type}:${label.name}`) ?? 0}
                          </span>
                          <kbd>{hotkeyForLabel(label.name, index)}</kbd>
                        </span>
                      </button>
                    ))}
                    {!strictLabelsOnly ? (
                      <label className="aw-custom-label">
                        <span>+ Custom label</span>
                        <input
                          value={activeSpanTask?.annotation_type === task.annotation_type ? customLabel : ""}
                          onChange={(event) => {
                            setActiveSpanType(task.annotation_type);
                            setActiveSpanLabel("");
                            setCustomLabel(event.target.value);
                          }}
                          placeholder="e.g., Outcome…"
                          disabled={!isTaskEditable(task.id)}
                        />
                      </label>
                    ) : null}
                  </section>
                );
              }

              return null;
            })}
          </aside>
          ) : null}

          {!isMobileWorkspace || workspacePane === "document" ? (
          <section
            id="aw-pane-document"
            className={cx(
              "aw-center",
              !legacyAnnotationEditable && selectedDocument && "has-read-state",
              isPersonalReview && "review-mode",
            )}
            role="tabpanel"
            aria-labelledby="aw-pane-tab-document"
          >
            {annotationTool === "evidence" && evidenceTask && activeWorkbench ? (
              <EvidenceBlockCanvas
                projectId={activeWorkbench.project.id}
                document={activeWorkbench.document}
                task={evidenceTask}
                assignments={currentAnnotatorAssignments}
                annotations={activeWorkbench.annotations}
                annotatorId={selectedAnnotatorId}
                allowAssignmentlessEditing={allowAssignmentlessSubmit}
                busy={busy}
                setBusy={setBusy}
                setError={setError}
                onAnnotationsChanged={applyEvidenceAnnotationChanges}
                onRefreshAnnotations={() => refreshWorkbench(activeWorkbench.document.id)}
                onActiveAssignmentChange={handleActiveEvidenceAssignmentChange}
                onSaveStatusChange={setSaveStatus}
              />
            ) : (
              <>
            {!legacyAnnotationEditable && selectedDocument ? (
              <header className="aw-read-state" role="status">
                <strong>
                  {allowAssignmentlessSubmit
                    ? isRecentlyFinishedReview
                      ? "Paper task submitted"
                      : selectedSubmissionAssignment
                        ? personalAssignmentStatusLabel(selectedSubmissionAssignment.status)
                        : "Review mode"
                    : selectedSubmissionAssignment
                      ? `${selectedSubmissionAssignment.status} annotation round`
                      : "No open annotation round"}
                </strong>
                <span>
                  {allowAssignmentlessSubmit
                    ? isRecentlyFinishedReview
                      ? "Only this task for this paper was submitted. Your saved annotations are shown below."
                      : "This task is read-only. Review its saved annotations below."
                    : "Read-only"}
                </span>
              </header>
            ) : null}
            <div
              className={cx(
                "aw-reading",
                relationSourceId !== null && "relation-mode",
                !legacyAnnotationEditable && "read-only",
              )}
              ref={readPaneRef}
              onMouseUp={handleSelection}
              onClick={() => {
                if (relationSourceId === null) {
                  setSelectedAnnotationId(null);
                }
              }}
            >
              {arcs.length > 0 ? (
                <svg className="aw-relation-svg" width={svgSize.width} height={svgSize.height}>
                  {arcs.map((arc) => {
                    const level = relationLevels[arc.id] ?? 0;
                    const peak = Math.min(arc.ay, arc.by) - 16 - level * 17;
                    const path = `M ${arc.ax} ${arc.ay} C ${arc.ax} ${peak}, ${arc.bx} ${peak}, ${arc.bx} ${arc.by}`;
                    const active =
                      hoveredRelationId === arc.id ||
                      selectedAnnotationId === arc.fromId ||
                      selectedAnnotationId === arc.toId;
                    const middleX = (arc.ax + arc.bx) / 2;
                    const label = arc.label.toUpperCase();
                    const pillWidth = label.length * 6.5 + 18;
                    return (
                      <g
                        className={cx("aw-arc", active && "active")}
                        key={arc.id}
                        role={
                          isTaskEditable(activeRelationTask?.id) ? "button" : undefined
                        }
                        tabIndex={isTaskEditable(activeRelationTask?.id) ? 0 : undefined}
                        aria-label={`Relation ${arc.label}`}
                        onMouseEnter={() => setHoveredRelationId(arc.id)}
                        onMouseLeave={() => setHoveredRelationId(null)}
                        onClick={(event) => {
                          event.stopPropagation();
                          if (isTaskEditable(activeRelationTask?.id)) {
                            setRelationMenu({ relationId: arc.id, x: event.clientX, y: event.clientY });
                          }
                        }}
                        onKeyDown={(event) => {
                          if (
                            (event.key === "Enter" || event.key === " ") &&
                            isTaskEditable(activeRelationTask?.id)
                          ) {
                            event.preventDefault();
                            const bounds = event.currentTarget.getBoundingClientRect();
                            setRelationMenu({
                              relationId: arc.id,
                              x: bounds.left + bounds.width / 2,
                              y: bounds.top + bounds.height / 2,
                            });
                          }
                        }}
                      >
                        <path className="aw-arc-hit" d={path} />
                        <path d={path} stroke={arc.color} />
                        <polygon
                          className="aw-arc-head"
                          points={`${arc.bx},${arc.by} ${arc.bx - 4},${arc.by - 7} ${arc.bx + 4},${arc.by - 7}`}
                          fill={arc.color}
                        />
                        {active ? (
                          <g>
                            <rect
                              x={middleX - pillWidth / 2}
                              y={peak - 9}
                              width={pillWidth}
                              height={18}
                              rx={9}
                              stroke={arc.color}
                            />
                            <text x={middleX} y={peak + 3} fill={arc.color} textAnchor="middle">
                              {label}
                            </text>
                          </g>
                        ) : null}
                      </g>
                    );
                  })}
                </svg>
              ) : null}

              {isDocumentLoading ? (
                <p className="aw-empty">Loading document text...</p>
              ) : sentenceRanges.length === 0 ? (
                <p className="aw-empty">Select a project document to begin.</p>
              ) : (
                sentenceRanges.map((range) => {
                  const sentenceLabel = sentenceAnnotation(range);
                  const sentenceLabelDef = findLabel(sentenceTask ?? undefined, activeWorkbench?.project ?? selectedProject, sentenceLabel?.label ?? "");
                  return (
                    <section className="aw-sentence" key={range.index}>
                      <div className="aw-sentence-meta">
                        <span>Sec {range.index + 1}</span>
                        {sentenceTask ? (
                          <button
                            className={cx("aw-sentence-chip", !sentenceLabel && "empty")}
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              setSentenceMenu({
                                sentenceIndex: range.index,
                                x: event.clientX,
                                y: event.clientY,
                              });
                            }}
                            disabled={!isTaskEditable(sentenceTask?.id)}
                          >
                            {sentenceLabel ? (
                              <>
                                <i
                                  style={{
                                    backgroundColor: labelColor(sentenceLabelDef, range.index),
                                  }}
                                />
                                {sentenceLabel.label}
                              </>
                            ) : (
                              "+ label"
                            )}
                          </button>
                        ) : null}
                      </div>
                      <div className="aw-sentence-body" data-sentence-index={range.index}>
                        {renderSectionText(range)}
                      </div>
                    </section>
                  );
                })
              )}

              {relationSourceId !== null && !relationChoice ? (
                <div className="aw-relation-banner">
                  Linking from <strong>{spanText(spanById.get(relationSourceId)!)}</strong>. Click a target span.
                  <span>Esc cancels</span>
                </div>
              ) : null}
            </div>

            <footer className="aw-read-status">
              <span>
                <strong>{spanAnnotations.length}</strong> spans -{" "}
                <strong>{relationAnnotations.length}</strong> relations
              </span>
              <span>
                Doc: <strong>{activeDocLabel?.label ?? "unset"}</strong>
              </span>
              <span>
                Sentences: <strong>{sentenceLabelAnnotations.length}/{sentenceRanges.length}</strong>
              </span>
              <span className="right">
                {isPersonalReview ? (
                  "Saved snapshot · review only"
                ) : (
                  <>
                    <kbd>R</kbd> relate - <kbd>Esc</kbd> cancel - <kbd>Delete</kbd> remove
                  </>
                )}
              </span>
            </footer>
              </>
            )}
          </section>
          ) : null}

          {!isMobileWorkspace || workspacePane === "review" ? (
          <aside
            id="aw-pane-review"
            className="aw-inspector"
            role="tabpanel"
            aria-labelledby="aw-pane-tab-review"
          >
            <h2 className="aw-pane-heading">Review</h2>
            <div
              className="aw-inspector-tabs"
              role="tablist"
              aria-label="Review views"
            >
              {INSPECTOR_TABS.map((tab) => (
                <button
                  className={cx(inspectorTab === tab && "active")}
                  id={`aw-inspector-tab-${tab}`}
                  key={tab}
                  role="tab"
                  aria-controls={`aw-inspector-panel-${tab}`}
                  aria-selected={inspectorTab === tab}
                  tabIndex={inspectorTab === tab ? 0 : -1}
                  type="button"
                  onClick={() => setInspectorTab(tab)}
                  onKeyDown={(event) =>
                    handleTabKey(
                      event,
                      INSPECTOR_TABS,
                      inspectorTab,
                      setInspectorTab,
                      "aw-inspector-tab",
                    )
                  }
                >
                  {tab}
                  {tab === "annotations" ? <span>{spanAnnotations.length}</span> : null}
                  {tab === "relations" ? <span>{relationAnnotations.length}</span> : null}
                  {tab === "sentence" ? <span>{sentenceLabelAnnotations.length}</span> : null}
                </button>
              ))}
            </div>

            {inspectorTab === "annotations" ? (
              <div
                className="aw-inspector-list"
                id="aw-inspector-panel-annotations"
                role="tabpanel"
                aria-labelledby="aw-inspector-tab-annotations"
              >
                {spanAnnotations.length === 0 ? (
                  <p className="aw-empty">No annotations yet. Select text to create one.</p>
                ) : (
                  spanAnnotations.map((annotation) => (
                    <article
                      className={cx("aw-annotation-card", selectedAnnotationId === annotation.id && "selected")}
                      key={annotation.id}
                      role="button"
                      tabIndex={0}
                      aria-label={`${annotation.label}: ${spanText(annotation)}`}
                      onClick={(event) => {
                        setSelectedAnnotationId(annotation.id);
                        if (isTaskEditable(tasksByType.get(annotation.annotation_type)?.id)) {
                          setSpanMenu({
                            annotationId: annotation.id,
                            x: event.clientX,
                            y: event.clientY,
                          });
                        }
                      }}
                      onKeyDown={(event) => {
                        if (event.key !== "Enter" && event.key !== " ") {
                          return;
                        }
                        event.preventDefault();
                        const bounds = event.currentTarget.getBoundingClientRect();
                        setSelectedAnnotationId(annotation.id);
                        if (isTaskEditable(tasksByType.get(annotation.annotation_type)?.id)) {
                          setSpanMenu({
                            annotationId: annotation.id,
                            x: bounds.left + Math.min(bounds.width / 2, 24),
                            y: bounds.bottom,
                          });
                        }
                      }}
                    >
                      <div>
                        <span
                          className="aw-type-pill"
                          style={
                            {
                              "--aw-label-color": taskColor(annotation.annotation_type, annotation.label),
                            } as React.CSSProperties
                          }
                        >
                          {annotation.label}
                        </span>
                        <code>{offsetText(annotation)}</code>
                      </div>
                      <p>{spanText(annotation)}</p>
                      <small>
                        {displayStatus(annotation.annotation_type)} - {annotation.status}
                        {correctionLockedIds.has(annotation.id) ? " - correction locked" : ""}
                        {relationLockedIds.has(annotation.id) ? " - relation target" : ""}
                      </small>
                    </article>
                  ))
                )}
              </div>
            ) : null}

            {inspectorTab === "relations" ? (
              <div
                className="aw-rel-list"
                id="aw-inspector-panel-relations"
                role="tabpanel"
                aria-labelledby="aw-inspector-tab-relations"
              >
                {relationAnnotations.length === 0 ? (
                  <p className="aw-empty">No relations yet. Open a span menu to start one.</p>
                ) : (
                  relationAnnotations.map((relation) => {
                    const head = relation.head_annotation_id ? spanById.get(relation.head_annotation_id) : null;
                    const tail = relation.tail_annotation_id ? spanById.get(relation.tail_annotation_id) : null;
                    return (
                      <article
                        className="aw-relation-row"
                        key={relation.id}
                        onMouseEnter={() => setHoveredRelationId(relation.id)}
                        onMouseLeave={() => setHoveredRelationId(null)}
                      >
                        <button
                          type="button"
                          className="aw-relation-select"
                          aria-label={`Edit relation ${relation.label}: ${
                            head ? spanText(head) : "Missing head"
                          } to ${tail ? spanText(tail) : "Missing tail"}`}
                          onClick={(event) => {
                            if (isTaskEditable(activeRelationTask?.id)) {
                              setRelationMenu({
                                relationId: relation.id,
                                x: event.clientX,
                                y: event.clientY,
                              });
                            }
                          }}
                          disabled={!isTaskEditable(activeRelationTask?.id)}
                        >
                          <span>{head ? spanText(head) : "Missing head"}</span>
                          <b>{relation.label}</b>
                          <span>{tail ? spanText(tail) : "Missing tail"}</span>
                        </button>
                        <button
                          type="button"
                          className="aw-relation-delete"
                          aria-label={`Delete relation ${relation.label}`}
                          title={`Delete relation ${relation.label}`}
                          onClick={(event) => {
                            event.stopPropagation();
                            void deleteRelation(relation.id);
                          }}
                          disabled={!isTaskEditable(activeRelationTask?.id)}
                        >
                          <Trash2 size={15} aria-hidden="true" />
                        </button>
                      </article>
                    );
                  })
                )}
              </div>
            ) : null}

            {inspectorTab === "sentence" ? (
              <div
                className="aw-inspector-list"
                id="aw-inspector-panel-sentence"
                role="tabpanel"
                aria-labelledby="aw-inspector-tab-sentence"
              >
                {sentenceRanges.map((range) => {
                  const annotation = sentenceAnnotation(range);
                  return (
                    <article className="aw-annotation-card" key={range.index}>
                      <div>
                        <span className="aw-type-pill">Sentence {range.index + 1}</span>
                        <code>
                          {range.start}-{range.end}
                        </code>
                      </div>
                      <p>{annotation?.label ?? "Unlabeled"}</p>
                    </article>
                  );
                })}
              </div>
            ) : null}

            {inspectorTab === "guideline" ? (
              <div
                className="aw-guideline"
                id="aw-inspector-panel-guideline"
                role="tabpanel"
                aria-labelledby="aw-inspector-tab-guideline"
              >
                {renderGuidelineMarkdown(displayedGuideline?.markdown)}
              </div>
            ) : null}
          </aside>
          ) : null}

          {isTaskEditable(activeSpanTask?.id) && pendingSelection ? (
            <div className="aw-popover" style={{ left: pendingSelection.left, top: pendingSelection.top - 12 }}>
              <span>"{pendingSelection.text.length > 24 ? `${pendingSelection.text.slice(0, 24)}...` : pendingSelection.text}"</span>
              {activeSpanLabels.map((label, index) => (
                <button
                  key={label.name}
                  type="button"
                  onClick={() => void createSpan(pendingSelection.annotationType, label.name)}
                >
                  <i style={{ backgroundColor: labelColor(label, index) }} />
                  {label.name}
                  <kbd>{hotkeyForLabel(label.name, index)}</kbd>
                </button>
              ))}
              {!strictLabelsOnly && currentSpanLabel && !activeSpanLabels.some((label) => label.name === currentSpanLabel) ? (
                <button type="button" onClick={() => void createSpan(pendingSelection.annotationType, currentSpanLabel)}>
                  <i />
                  {currentSpanLabel}
                </button>
              ) : null}
            </div>
          ) : null}

          {isTaskEditable(spanMenuTask?.id) && spanMenu && currentSpanMenuAnnotation && spanMenuTask ? (
            <>
              <button className="aw-menu-backdrop" type="button" onClick={() => setSpanMenu(null)} aria-label="Close menu" />
              <div
                className="aw-context-menu"
                style={{
                  left: Math.min(spanMenu.x, window.innerWidth - 250),
                  top: Math.min(spanMenu.y, window.innerHeight - 360),
                }}
              >
                <header>
                  <strong>"{spanText(currentSpanMenuAnnotation)}"</strong>
                  <code>{offsetText(currentSpanMenuAnnotation)}</code>
                </header>
                <span className="aw-menu-heading">Change label</span>
                {displayLabelsForTask(spanMenuTask).map((label, index) => (
                  <button
                    className={cx(currentSpanMenuAnnotation.label === label.name && "current")}
                    key={label.name}
                    type="button"
                    onClick={() => void relabelSpan(currentSpanMenuAnnotation.id, label.name)}
                  >
                    <i style={{ backgroundColor: labelColor(label, index) }} />
                    {label.name}
                    {currentSpanMenuAnnotation.label === label.name ? <b>OK</b> : <kbd>{hotkeyForLabel(label.name, index)}</kbd>}
                  </button>
                ))}
                <hr />
                <button
                  type="button"
                  onClick={() => {
                    setRelationSourceId(currentSpanMenuAnnotation.id);
                    setSpanMenu(null);
                  }}
                  disabled={!isTaskEditable(activeRelationTask?.id)}
                >
                  Start relation from here
                </button>
                <button
                  className="danger"
                  type="button"
                  onClick={() => void deleteSpan(currentSpanMenuAnnotation.id)}
                  disabled={
                    correctionLockedIds.has(currentSpanMenuAnnotation.id) ||
                    relationLockedIds.has(currentSpanMenuAnnotation.id)
                  }
                >
                  Delete annotation
                </button>
              </div>
            </>
          ) : null}

          {isTaskEditable(activeRelationTask?.id) && relationChoice && activeRelationTask ? (
            <>
              <button
                className="aw-menu-backdrop"
                type="button"
                onClick={() => {
                  setRelationChoice(null);
                  setRelationSourceId(null);
                }}
                aria-label="Close relation picker"
              />
              <div
                className="aw-popover relation"
                style={{
                  left: Math.min(relationChoice.x, window.innerWidth - 230),
                  top: relationChoice.y + 14,
                }}
              >
                <span>Relation type</span>
                {relationChoiceLabels.length === 0 ? (
                  <p className="aw-empty">
                    No relation types allow {relationChoiceHead?.label ?? "this head"} -&gt;{" "}
                    {relationChoiceTail?.label ?? "this tail"}.
                  </p>
                ) : null}
                {relationChoiceLabels.map((label, index) => (
                  <button key={label.name} type="button" onClick={() => void createRelation(label.name)}>
                    <i style={{ backgroundColor: labelColor(label, index) }} />
                    {label.name}
                    <kbd>{hotkeyForLabel(label.name, index)}</kbd>
                  </button>
                ))}
              </div>
            </>
          ) : null}

          {isTaskEditable(activeRelationTask?.id) && relationMenu && relationMenuAnnotation && activeRelationTask ? (
            <>
              <button
                className="aw-menu-backdrop"
                type="button"
                onClick={() => setRelationMenu(null)}
                aria-label="Close relation menu"
              />
              <div
                className="aw-context-menu"
                style={{
                  left: Math.min(relationMenu.x, window.innerWidth - 250),
                  top: Math.min(relationMenu.y, window.innerHeight - 360),
                }}
              >
                <header>
                  <strong>
                    {relationMenuHead ? spanText(relationMenuHead) : "missing head"} -&gt;{" "}
                    {relationMenuTail ? spanText(relationMenuTail) : "missing tail"}
                  </strong>
                  <code>{relationMenuAnnotation.label}</code>
                </header>
                <span className="aw-menu-heading">Change relation type</span>
                {(relationMenuHead && relationMenuTail
                  ? filterRelationLabels(
                      displayLabelsForTask(activeRelationTask),
                      relationConstraints,
                      relationMenuHead.label,
                      relationMenuTail.label,
                    )
                  : displayLabelsForTask(activeRelationTask)
                ).map((label, index) => (
                  <button
                    className={cx(relationMenuAnnotation.label === label.name && "current")}
                    key={label.name}
                    type="button"
                    onClick={() => void relabelRelation(relationMenuAnnotation.id, label.name)}
                  >
                    <i style={{ backgroundColor: labelColor(label, index) }} />
                    {label.name}
                    {relationMenuAnnotation.label === label.name ? (
                      <b>OK</b>
                    ) : (
                      <kbd>{hotkeyForLabel(label.name, index)}</kbd>
                    )}
                  </button>
                ))}
                <hr />
                <button type="button" onClick={() => void swapRelationDirection(relationMenuAnnotation)}>
                  Swap direction
                </button>
                <button
                  className="danger"
                  type="button"
                  onClick={() => void deleteRelation(relationMenuAnnotation.id)}
                >
                  Delete relation
                </button>
              </div>
            </>
          ) : null}

          {isTaskEditable(sentenceTask?.id) && sentenceMenu && sentenceMenuRange && sentenceTask ? (
            <>
              <button
                className="aw-menu-backdrop"
                type="button"
                onClick={() => setSentenceMenu(null)}
                aria-label="Close sentence menu"
              />
              <div
                className="aw-context-menu"
                style={{
                  left: Math.min(sentenceMenu.x, window.innerWidth - 240),
                  top: Math.min(sentenceMenu.y, window.innerHeight - 280),
                }}
              >
                <span className="aw-menu-heading">Sentence label</span>
                {displayLabelsForTask(sentenceTask).map((label, index) => {
                  const current = sentenceAnnotation(sentenceMenuRange)?.label === label.name;
                  return (
                    <button
                      className={cx(current && "current")}
                      key={label.name}
                      type="button"
                      onClick={() => void setSentenceLabel(sentenceMenuRange, label.name)}
                    >
                      <i style={{ backgroundColor: labelColor(label, index) }} />
                      {label.name}
                      {current ? <b>OK</b> : <kbd>{index + 1}</kbd>}
                    </button>
                  );
                })}
                <hr />
                <button type="button" onClick={() => void setSentenceLabel(sentenceMenuRange, null)}>
                  Clear label
                </button>
              </div>
            </>
          ) : null}
          </div>
        </div>
      )}
      </main>

      {submissionDialogOpen &&
      selectedDocument &&
      selectedSubmissionAssignment &&
      selectedAssignmentTask ? (
        <DialogFrame
          busy={busy}
          labelledBy="aw-submit-dialog-title"
          backdropClassName="aw-dialog-backdrop"
          dialogClassName="aw-dialog"
          dialogElement="section"
          initialFocusSelector="[data-dialog-initial-focus]"
          portal={false}
          onDismiss={() => {
            setSubmissionDialogOpen(false);
            setBlankSubmissionAcknowledged(false);
          }}
        >
          <header>
            <h2 id="aw-submit-dialog-title">
              {allowAssignmentlessSubmit
                ? "Submit this paper task?"
                : "Submit this assignment?"}
            </h2>
            <p>
              {allowAssignmentlessSubmit
                ? "This submits only the selected task for this paper. It does not complete that task for your other papers."
                : "This makes the selected assignment read-only."}
            </p>
          </header>
          <dl className="aw-dialog-scope">
            <div>
              <dt>Paper</dt>
              <dd>{documentTitle(selectedDocument)}</dd>
            </div>
            <div>
              <dt>Task</dt>
              <dd>{selectedAssignmentTask.display_name}</dd>
            </div>
            <div>
              <dt>Annotations</dt>
              <dd>{selectedTaskAnnotationCount}</dd>
            </div>
          </dl>
          {selectedTaskAnnotationCount === 0 ? (
            <div className="aw-dialog-warning">
              <strong>This paper task has no annotations.</strong>
              <p>
                A blank result can be valid, but submitting it will make this
                paper task read-only until you reopen it.
              </p>
              <label>
                <input
                  type="checkbox"
                  checked={blankSubmissionAcknowledged}
                  onChange={(event) =>
                    setBlankSubmissionAcknowledged(event.target.checked)
                  }
                />
                I confirm that this paper has no annotations for this task.
              </label>
            </div>
          ) : null}
          <div className="aw-dialog-actions">
            <button
              type="button"
              data-dialog-initial-focus
              onClick={() => {
                setSubmissionDialogOpen(false);
                setBlankSubmissionAcknowledged(false);
              }}
              disabled={busy}
            >
              Keep editing
            </button>
            <button
              className="primary"
              type="button"
              onClick={() => void submitCurrentDocument()}
              disabled={
                busy ||
                (selectedTaskAnnotationCount === 0 &&
                  !blankSubmissionAcknowledged)
              }
            >
              {selectedTaskAnnotationCount === 0
                ? "Submit with no annotations"
                : allowAssignmentlessSubmit
                  ? "Submit this paper task"
                  : "Confirm submission"}
            </button>
          </div>
        </DialogFrame>
      ) : null}

      {reopenDialogOpen &&
      allowAssignmentlessSubmit &&
      selectedDocument &&
      selectedSubmissionAssignment &&
      selectedAssignmentTask ? (
        <DialogFrame
          busy={busy}
          labelledBy="aw-reopen-dialog-title"
          backdropClassName="aw-dialog-backdrop"
          dialogClassName="aw-dialog"
          dialogElement="section"
          initialFocusSelector="[data-dialog-initial-focus]"
          portal={false}
          onDismiss={() => setReopenDialogOpen(false)}
        >
          <header>
            <h2 id="aw-reopen-dialog-title">
              Reopen this paper task for editing?
            </h2>
            <p>
              Your earlier submission will remain saved. You can edit the
              annotations and submit a corrected version when ready.
            </p>
          </header>
          <dl className="aw-dialog-scope">
            <div>
              <dt>Paper</dt>
              <dd>{documentTitle(selectedDocument)}</dd>
            </div>
            <div>
              <dt>Task</dt>
              <dd>{selectedAssignmentTask.display_name}</dd>
            </div>
          </dl>
          <div className="aw-dialog-actions">
            <button
              type="button"
              data-dialog-initial-focus
              onClick={() => setReopenDialogOpen(false)}
              disabled={busy}
            >
              Keep read-only
            </button>
            <button
              className="primary"
              type="button"
              onClick={() => void reopenCurrentPaperTask()}
              disabled={busy}
            >
              Reopen and edit
            </button>
          </div>
        </DialogFrame>
      ) : null}
    </div>
  );
}
