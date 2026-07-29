import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { CheckboxInput } from "@astryxdesign/core/CheckboxInput";
import { EmptyState as XDSEmptyState } from "@astryxdesign/core/EmptyState";
import { SegmentedControl, SegmentedControlItem } from "@astryxdesign/core/SegmentedControl";
import { Selector } from "@astryxdesign/core/Selector";
import { StatusDot } from "@astryxdesign/core/StatusDot";
import { Tab, TabList } from "@astryxdesign/core/TabList";
import { TextArea } from "@astryxdesign/core/TextArea";
import { TextInput } from "@astryxdesign/core/TextInput";

import {
  approveJoinRequest,
  activateEvidenceTarget,
  adjudicateEvidence,
  createEvidenceTarget,
  createEvidenceTargetVersion,
  createGuidelineVersion,
  createProject,
  createProjectTask,
  createWorkspaceInvite,
  createSubmission,
  createTaskAssignment,
  deleteWorkspaceMember,
  deleteProjectTask,
  deleteSubmission,
  deleteTaskAssignment,
  deactivateEvidenceTarget,
  downloadSubmission,
  listGuidelineVersions,
  listEvidenceTargets,
  getEvidenceAdjudication,
  getProjectIaa,
  listSubmissions,
  listWorkspaceJoinRequests,
  listWorkspaceMembers,
  rejectJoinRequest,
  updateProject,
  updateProjectTask,
  updateTaskAssignment,
  updateWorkspaceMemberRole,
} from "@/api/client";
import BrandLogo from "@/components/BrandLogo";
import ConfirmDialog from "@/components/ConfirmDialog";
import DialogFrame from "@/components/DialogFrame";
import PubmedImportPanel from "@/components/PubmedImportPanel";
import type { PubmedImportPanelHandle } from "@/components/PubmedImportPanel";
import { relationConstraintsOf } from "@/lib/relationConstraints";
import { shouldHandleSpaClick } from "@/components/ModuleSwitcher";
import {
  useManagerConsoleStore,
  type AdminSection,
  type LoopPhase,
  type ManagerTab,
} from "@/store/managerConsoleStore";
import { useProjectWorkspaceStore } from "@/store/projectWorkspaceStore";
import type {
  AnnotationSubmission,
  AnnotationType,
  AnnotationValidationMode,
  Document,
  EvidenceAdjudicationRead,
  EvidenceBlockTaskSettingsV1,
  EvidenceTarget,
  GuidelineVersion,
  IaaReport,
  Project,
  ProjectTask,
  TaskAssignment,
  TaskAssignmentStatus,
  WorkspaceJoinRequest,
  WorkspaceMember,
  WorkspaceRole,
} from "@/types/api";

interface ManagerConsoleProps {
  onMyWorkNavigate: () => void;
  pathname: string;
  onNavigate: (pathname: string, mode?: "push" | "replace") => void;
  activeWorkspaceId: number | null;
  canUseTraining: boolean;
  isPersonalWorkspace: boolean;
  workspaceRole: WorkspaceRole | null;
  currentUsername?: string | null;
  moduleSwitcher?: ReactNode;
  workspaceSwitcher?: ReactNode;
  onLogout: () => void;
  embedded?: boolean;
}

interface ProjectFormState {
  name: string;
  description: string;
  annotationValidationMode: AnnotationValidationMode;
  starterTaskTypes: AnnotationType[];
}

interface TaskDraftState {
  annotationType: AnnotationType;
  displayName: string;
}

interface LabelDraftState {
  name: string;
  color: string;
  description: string;
}

interface AssignmentDraftState {
  taskId: string;
  documentId: string;
  assigneeUserId: string;
  status: TaskAssignmentStatus;
  targetVersionId: string;
}

interface EvidenceTargetDraftState {
  key: string;
  name: string;
  text: string;
  guidance: string;
}

type EvidenceSettingsUpdate = Omit<
  Partial<EvidenceBlockTaskSettingsV1>,
  "keyboard_shortcuts"
> & {
  keyboard_shortcuts?: Partial<EvidenceBlockTaskSettingsV1["keyboard_shortcuts"]>;
};

type EvidenceAdjudicationStrategy = "a" | "b" | "union" | "intersection" | "custom";

interface EvidenceAdjudicationDraftState {
  documentId: string;
  targetVersionId: string;
  guidelineVersionId: string;
  strategy: EvidenceAdjudicationStrategy;
  selectedAnnotationIds: number[];
  customStartSentenceId: string;
  customEndSentenceId: string;
  note: string;
}

interface PendingConfirmation {
  title: string;
  description: string;
  confirmLabel: string;
  action: () => Promise<void>;
}

interface GuidelineDraftState {
  versionLabel: string;
  markdown: string;
}

type QueueStatus = "todo" | "partial" | "review" | "done" | "blocked";

export type PersonalReadinessStatus = "complete" | "warning" | "blocked" | "ready";

export interface PersonalReadinessStep {
  id: "tasks" | "documents" | "annotate";
  title: string;
  detail: string;
  status: PersonalReadinessStatus;
}

export interface PersonalProjectReadiness {
  canStart: boolean;
  usesFreeFormLabels: boolean;
  nextAction: "tasks" | "documents" | "my-work";
  steps: [PersonalReadinessStep, PersonalReadinessStep, PersonalReadinessStep];
}

const TABS: Array<{ id: ManagerTab; label: string; icon: IconName }> = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "documents", label: "Documents", icon: "docs" },
  { id: "tasks", label: "Tasks", icon: "tasks" },
  { id: "progress", label: "Progress", icon: "chart" },
  { id: "loop", label: "Learning Loop", icon: "loop" },
  { id: "guidelines", label: "Guidelines", icon: "book" },
  { id: "export", label: "Export", icon: "export" },
];

const ADMIN_SECTIONS: Array<{ id: AdminSection; label: string }> = [
  { id: "users", label: "Users" },
  { id: "plugins", label: "Plugins" },
  { id: "health", label: "Health" },
  { id: "audit", label: "Audit" },
  { id: "settings", label: "Settings" },
];

const LOOP_PHASES: Array<{ id: LoopPhase; label: string }> = [
  { id: "select", label: "Select" },
  { id: "adjudicate", label: "Adjudicate" },
  { id: "reflect", label: "Reflect" },
];

const ANNOTATION_TYPE_OPTIONS: Array<{ value: AnnotationType; label: string }> = [
  { value: "doc_label", label: "Document Annotation" },
  { value: "entity", label: "Entity Annotation" },
  { value: "relation", label: "Relation Annotation" },
  { value: "sentence_label", label: "Sentence Annotation" },
  { value: "passage_label", label: "Passage Annotation" },
  { value: "evidence_block", label: "Evidence Block Identification" },
];

const EVIDENCE_SHORTCUT_LABELS: Array<{
  key: keyof EvidenceBlockTaskSettingsV1["keyboard_shortcuts"];
  label: string;
}> = [
  { key: "create", label: "Create" },
  { key: "expand_start", label: "Expand start" },
  { key: "expand_end", label: "Expand end" },
  { key: "contract_start", label: "Contract start" },
  { key: "contract_end", label: "Contract end" },
  { key: "merge", label: "Merge" },
  { key: "split", label: "Split" },
  { key: "delete", label: "Delete" },
  { key: "mark_reviewed", label: "Mark reviewed" },
  { key: "cancel", label: "Cancel" },
];

const ASSIGNMENT_STATUS_OPTIONS: TaskAssignmentStatus[] = [
  "assigned",
  "in_progress",
  "submitted",
  "adjudication_ready",
  "adjudicated",
  "completed",
  "blocked",
  "withdrawn",
];

const WORKSPACE_ROLE_OPTIONS: WorkspaceRole[] = ["annotator", "trainer", "manager", "admin"];

const FALLBACK_COLORS = ["#4d6e5b", "#8e3a3a", "#8a6d2a", "#4a5a8a", "#6a4a7a", "#5e6e8a"];

const EMPTY_PROJECT_FORM: ProjectFormState = {
  name: "",
  description: "",
  annotationValidationMode: "relaxed",
  starterTaskTypes: ["entity"],
};

const EMPTY_ASSIGNMENT_DRAFT: AssignmentDraftState = {
  taskId: "",
  documentId: "",
  assigneeUserId: "",
  status: "assigned",
  targetVersionId: "",
};

const EMPTY_EVIDENCE_TARGET_DRAFT: EvidenceTargetDraftState = {
  key: "",
  name: "",
  text: "",
  guidance: "",
};

const EMPTY_EVIDENCE_ADJUDICATION_DRAFT: EvidenceAdjudicationDraftState = {
  documentId: "",
  targetVersionId: "",
  guidelineVersionId: "",
  strategy: "union",
  selectedAnnotationIds: [],
  customStartSentenceId: "",
  customEndSentenceId: "",
  note: "",
};

const EMPTY_LABEL_DRAFT: LabelDraftState = {
  name: "",
  color: FALLBACK_COLORS[0],
  description: "",
};

const EMPTY_GUIDELINE_DRAFT: GuidelineDraftState = {
  versionLabel: "v1",
  markdown: "# Annotation guideline\n\nAdd project-specific instructions here.",
};

const DONE_STATUSES = new Set<TaskAssignmentStatus>(["submitted", "completed"]);
const REVIEW_STATUSES = new Set<TaskAssignmentStatus>(["adjudication_ready", "adjudicated"]);

const VALID_TABS = new Set<ManagerTab>(TABS.map((tab) => tab.id));
const VALID_ADMIN_SECTIONS = new Set<AdminSection>(ADMIN_SECTIONS.map((section) => section.id));

type IconName =
  | "admin"
  | "book"
  | "chart"
  | "check"
  | "cpu"
  | "docs"
  | "export"
  | "grid"
  | "loop"
  | "plus"
  | "search"
  | "tasks"
  | "users";

function Icon({ name, size = 15 }: { name: IconName; size?: number }): React.ReactElement {
  const common = {
    width: size,
    height: size,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 2,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };

  const paths: Record<IconName, React.ReactNode> = {
    admin: (
      <>
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.6 1.6 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.6 1.6 0 0 0-1.8-.3 1.6 1.6 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.2a1.6 1.6 0 0 0-1-1.5 1.6 1.6 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.6 1.6 0 0 0 4.6 15a1.6 1.6 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.2a1.6 1.6 0 0 0 1.5-1 1.6 1.6 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.6 1.6 0 0 0 1.8.3h.1a1.6 1.6 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.2a1.6 1.6 0 0 0 1 1.5h.1a1.6 1.6 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.6 1.6 0 0 0-.3 1.8v.1a1.6 1.6 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.2a1.6 1.6 0 0 0-1.4 1Z" />
      </>
    ),
    book: (
      <>
        <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
        <path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15Z" />
      </>
    ),
    chart: (
      <>
        <path d="M3 3v18h18" />
        <path d="m7 14 4-4 3 3 5-7" />
      </>
    ),
    check: <path d="m4 12 5 5L20 6" />,
    cpu: (
      <>
        <rect x="7" y="7" width="10" height="10" rx="2" />
        <path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 15h3M1 9h3M1 15h3" />
      </>
    ),
    docs: (
      <>
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" />
        <path d="M14 2v6h6M8 13h8M8 17h5" />
      </>
    ),
    export: (
      <>
        <path d="M12 3v12" />
        <path d="m7 10 5 5 5-5" />
        <path d="M5 21h14" />
      </>
    ),
    grid: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),
    loop: (
      <>
        <path d="M17 2v6h-6" />
        <path d="M7 22v-6h6" />
        <path d="M20 11a8 8 0 0 0-13.5-5.8L4 8" />
        <path d="M4 13a8 8 0 0 0 13.5 5.8L20 16" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-3.5-3.5" />
      </>
    ),
    tasks: (
      <>
        <path d="m4 6 2 2 4-4" />
        <path d="M12 7h8" />
        <path d="m4 14 2 2 4-4" />
        <path d="M12 15h8" />
      </>
    ),
    users: (
      <>
        <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M22 21v-2a4 4 0 0 0-3-3.8M16 3.1a4 4 0 0 1 0 7.8" />
      </>
    ),
  };

  return <svg {...common}>{paths[name]}</svg>;
}

function sortTasks(tasks: ProjectTask[]): ProjectTask[] {
  return [...tasks].sort((a, b) => a.sort_order - b.sort_order || a.id - b.id);
}

function documentTitle(documentItem: Document): string {
  return documentItem.title ?? documentItem.external_id ?? `Document ${documentItem.id}`;
}

function formatStatus(status: string): string {
  return status.replace(/_/g, " ");
}

function toNullableText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function getTaskDisplayName(annotationType: AnnotationType): string {
  return (
    ANNOTATION_TYPE_OPTIONS.find((option) => option.value === annotationType)?.label ??
    annotationType.replace(/_/g, " ")
  );
}

function effectiveTaskLabelCount(project: Project, task: ProjectTask): number {
  if (task.labels.length > 0) {
    return task.labels.length;
  }
  return project.annotation_schema.labels[task.annotation_type]?.length ?? 0;
}

export function managerTabsForWorkspace(
  isPersonalWorkspace: boolean,
): Array<{ id: ManagerTab; label: string; icon: IconName }> {
  return TABS.map((tab) =>
    isPersonalWorkspace && tab.id === "loop"
      ? { ...tab, label: "Gold review" }
      : tab,
  );
}

export function getPersonalProjectReadiness(
  project: Project,
  documentCount: number,
  evidenceTargets: EvidenceTarget[],
): PersonalProjectReadiness {
  const enabledTasks = sortTasks(project.tasks.filter((task) => task.enabled));
  const taskBlockers: string[] = [];
  const taskWarnings: string[] = [];
  const hasEnabledEntityTask = enabledTasks.some(
    (task) => task.annotation_type === "entity",
  );

  if (enabledTasks.length === 0) {
    taskBlockers.push("Enable at least one annotation task.");
  }

  for (const task of enabledTasks) {
    if (task.annotation_type === "evidence_block") {
      const hasActiveTarget = evidenceTargets.some(
        (target) =>
          target.task_id === task.id &&
          target.is_active &&
          target.active_version_id !== null,
      );
      if (!hasActiveTarget) {
        taskBlockers.push(`${task.display_name} needs an active evidence target.`);
      }
      continue;
    }

    if (task.annotation_type === "relation" && !hasEnabledEntityTask) {
      taskBlockers.push(`${task.display_name} needs an enabled entity task first.`);
    }

    if (effectiveTaskLabelCount(project, task) === 0) {
      if (project.annotation_validation_mode === "strict") {
        taskBlockers.push(
          `${task.display_name} needs at least one label in strict validation.`,
        );
      } else {
        taskWarnings.push(
          `${task.display_name} has no preset labels; configure labels or explicitly use free-form labels.`,
        );
      }
    }
  }

  const taskStatus: PersonalReadinessStatus =
    taskBlockers.length > 0
      ? "blocked"
      : taskWarnings.length > 0
        ? "warning"
        : "complete";
  const taskDetail =
    taskBlockers.length > 0
      ? taskBlockers.join(" ")
      : taskWarnings.length > 0
        ? taskWarnings.join(" ")
        : `${enabledTasks.length} enabled ${
            enabledTasks.length === 1 ? "task is" : "tasks are"
          } ready.`;
  const hasDocuments = documentCount > 0;
  const canStart = taskBlockers.length === 0 && hasDocuments;
  const usesFreeFormLabels = canStart && taskWarnings.length > 0;
  const nextAction =
    taskBlockers.length > 0
      ? "tasks"
      : !hasDocuments
        ? "documents"
        : "my-work";

  return {
    canStart,
    usesFreeFormLabels,
    nextAction,
    steps: [
      {
        id: "tasks",
        title: "Configure annotation tasks",
        detail: taskDetail,
        status: taskStatus,
      },
      {
        id: "documents",
        title: "Import documents",
        detail: hasDocuments
          ? `${documentCount} ${documentCount === 1 ? "document is" : "documents are"} ready.`
          : "Import at least one PMID before starting annotation.",
        status: hasDocuments ? "complete" : "blocked",
      },
      {
        id: "annotate",
        title: "Start annotating",
        detail: canStart
          ? usesFreeFormLabels
            ? "Ready to start with free-form labels. You can add preset labels at any time."
            : "Your project is ready for annotation."
          : taskBlockers.length > 0
            ? "Resolve the task setup blockers before starting."
            : "Import documents before starting.",
        status: canStart ? "ready" : "blocked",
      },
    ],
  };
}

function statusToQueueStatus(status: TaskAssignmentStatus): QueueStatus {
  if (DONE_STATUSES.has(status)) {
    return "done";
  }
  if (REVIEW_STATUSES.has(status)) {
    return "review";
  }
  if (status === "in_progress") {
    return "partial";
  }
  if (status === "blocked" || status === "withdrawn") {
    return "blocked";
  }
  return "todo";
}

function statusCount(assignments: TaskAssignment[]): Record<QueueStatus, number> {
  return assignments.reduce<Record<QueueStatus, number>>(
    (counts, assignment) => {
      counts[statusToQueueStatus(assignment.status)] += 1;
      return counts;
    },
    { todo: 0, partial: 0, review: 0, done: 0, blocked: 0 },
  );
}

export function documentQueueStatusFromAssignments(
  documentAssignments: TaskAssignment[],
): QueueStatus {
  const currentAssignments = documentAssignments.filter(
    (assignment) => assignment.status !== "withdrawn",
  );
  if (currentAssignments.length === 0 && documentAssignments.length > 0) {
    return "blocked";
  }
  if (currentAssignments.some((assignment) => assignment.status === "blocked")) {
    return "blocked";
  }
  if (currentAssignments.some((assignment) => REVIEW_STATUSES.has(assignment.status))) {
    return "review";
  }
  if (
    currentAssignments.length > 0 &&
    currentAssignments.every((assignment) => DONE_STATUSES.has(assignment.status))
  ) {
    return "done";
  }
  if (currentAssignments.some((assignment) => assignment.status === "in_progress")) {
    return "partial";
  }
  if (currentAssignments.some((assignment) => DONE_STATUSES.has(assignment.status))) {
    return "partial";
  }
  return "todo";
}

function percent(done: number, total: number): number {
  if (total <= 0) {
    return 0;
  }
  return Math.round((done / total) * 100);
}

function wordCount(documentItem: Document): number {
  const text = documentItem.text.trim();
  return text ? text.split(/\s+/).length : 0;
}

function projectCode(project: Project): string {
  const fromWords = project.name
    .split(/\s+/)
    .map((part) => part[0])
    .join("")
    .slice(0, 4)
    .toUpperCase();
  return fromWords || `P${project.id}`;
}

function projectAccent(project: Project): string {
  return FALLBACK_COLORS[project.id % FALLBACK_COLORS.length];
}

function userInitials(username: string | null | undefined): string {
  const normalized = username?.trim() ?? "";
  if (!normalized) {
    return "AM";
  }
  const parts = normalized.split(/[\s._-]+/).filter(Boolean);
  if (parts.length > 1) {
    return `${parts[0][0]}${parts[1][0]}`.toUpperCase();
  }
  return normalized.slice(0, 2).toUpperCase();
}

function normalizeTab(value: string | undefined): ManagerTab {
  return VALID_TABS.has(value as ManagerTab) ? (value as ManagerTab) : "overview";
}

function normalizeAdminSection(value: string | undefined): AdminSection {
  return VALID_ADMIN_SECTIONS.has(value as AdminSection) ? (value as AdminSection) : "users";
}

function nextGuidelineVersionLabel(guidelines: GuidelineVersion[]): string {
  const maxVersion = guidelines.reduce((max, guideline) => {
    const match = /^v(\d+)$/i.exec(guideline.version_label.trim());
    return match ? Math.max(max, Number(match[1])) : max;
  }, 0);
  return maxVersion > 0 ? `v${maxVersion + 1}` : `v${guidelines.length + 1}`;
}

function activeGuideline(guidelines: GuidelineVersion[]): GuidelineVersion | null {
  return guidelines.find((guideline) => guideline.status === "active") ?? guidelines[0] ?? null;
}

function isHotkeyTargetIgnored(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return Boolean(
    target.closest(
      "input, textarea, select, button, a, [role='button'], [contenteditable='true']",
    ),
  );
}

function StatusMeter({
  counts,
  label = "Assignment status meter",
}: {
  counts: Record<QueueStatus, number>;
  label?: string;
}): React.ReactElement {
  const actualTotal = Object.values(counts).reduce(
    (sum, count) => sum + count,
    0,
  );
  const total = actualTotal || 1;
  return (
    <div
      className="mc-meter"
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={counts.done}
      aria-valuetext={`${counts.done} complete, ${counts.partial} in progress, ${counts.review} ready for review, ${counts.todo} to do, ${counts.blocked} blocked`}
    >
      <i className="done" style={{ width: `${(counts.done / total) * 100}%` }} />
      <i className="partial" style={{ width: `${(counts.partial / total) * 100}%` }} />
      <i className="review" style={{ width: `${(counts.review / total) * 100}%` }} />
      <i className="todo" style={{ width: `${(counts.todo / total) * 100}%` }} />
      <i className="blocked" style={{ width: `${(counts.blocked / total) * 100}%` }} />
    </div>
  );
}

function StatusBadge({ status }: { status: QueueStatus }): React.ReactElement {
  const variant =
    status === "done"
      ? "success"
      : status === "review"
        ? "error"
        : status === "partial"
          ? "warning"
          : "neutral";
  const dotVariant =
    status === "done"
      ? "success"
      : status === "review"
        ? "error"
        : status === "partial"
          ? "warning"
          : status === "blocked"
            ? "error"
            : "neutral";
  const label =
    status === "partial" ? "In progress" : status === "todo" ? "To do" : formatStatus(status);

  return (
    <Badge
      className={`mc-badge ${status}`}
      variant={variant}
      label={
        <span className="mc-badge-label">
          <StatusDot variant={dotVariant} label={`${label} status`} />
          {label}
        </span>
      }
    />
  );
}

function EmptyState({ title, detail }: { title: string; detail: string }): React.ReactElement {
  return (
    <XDSEmptyState
      className="mc-empty"
      title={title}
      description={detail}
      headingLevel={3}
      isCompact
    />
  );
}

function initials(value: string): string {
  return value
    .split(/[\s._-]+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function Avatar({ id }: { id: string }): React.ReactElement {
  const hash = Array.from(id).reduce((sum, character) => sum + character.charCodeAt(0), 0);
  return (
    <span className="mc-avatar-sm" style={{ background: FALLBACK_COLORS[hash % FALLBACK_COLORS.length] }}>
      {initials(id) || "U"}
    </span>
  );
}

function memberLabel(member: WorkspaceMember | undefined, fallback: string): string {
  if (!member) {
    return fallback;
  }
  return member.display_name.trim() || member.username;
}

export default function ManagerConsole({
  activeWorkspaceId,
  canUseTraining,
  isPersonalWorkspace,
  currentUsername,
  moduleSwitcher,
  workspaceSwitcher,
  onLogout,
  onNavigate,
  onMyWorkNavigate,
  pathname,
  workspaceRole,
  embedded = false,
}: ManagerConsoleProps): React.ReactElement {
  const {
    role,
    appMode,
    currentTab,
    loopPhase,
    tasksSubView,
    adminSection,
    modal,
    setRole,
    setAppMode,
    setCurrentTab,
    setLoopPhase,
    setTasksSubView,
    setAdminSection,
    setModal,
    openProjectTab,
  } = useManagerConsoleStore();

  useEffect(() => {
    setRole(workspaceRole === "admin" ? "admin" : "manager");
  }, [setRole, workspaceRole]);

  const {
    projects,
    selectedProjectId,
    documents,
    assignments,
    projectProgress: progress,
    loading,
    busy,
    error,
    setSelectedProjectId,
    clearProjectData,
    loadProjects,
    loadProject,
    loadProjectData,
    replaceProject,
    setBusy,
    setError,
  } = useProjectWorkspaceStore();
  const [guidelines, setGuidelines] = useState<GuidelineVersion[]>([]);
  const [projectForm, setProjectForm] = useState<ProjectFormState>(EMPTY_PROJECT_FORM);
  const [taskDraft, setTaskDraft] = useState<TaskDraftState>({
    annotationType: "doc_label",
    displayName: "",
  });
  const [isTaskComposerOpen, setIsTaskComposerOpen] = useState(false);
  const [labelDrafts, setLabelDrafts] = useState<Record<number, LabelDraftState>>({});
  const [assignmentDraft, setAssignmentDraft] = useState<AssignmentDraftState>(EMPTY_ASSIGNMENT_DRAFT);
  const [evidenceTargets, setEvidenceTargets] = useState<EvidenceTarget[]>([]);
  const [evidenceTargetDraft, setEvidenceTargetDraft] = useState<EvidenceTargetDraftState>(
    EMPTY_EVIDENCE_TARGET_DRAFT,
  );
  const [evidenceVersionDrafts, setEvidenceVersionDrafts] = useState<Record<number, string>>({});
  const [evidenceAdjudicationDraft, setEvidenceAdjudicationDraft] =
    useState<EvidenceAdjudicationDraftState>(EMPTY_EVIDENCE_ADJUDICATION_DRAFT);
  const [evidenceComparison, setEvidenceComparison] =
    useState<EvidenceAdjudicationRead | null>(null);
  const [evidenceIaa, setEvidenceIaa] = useState<IaaReport | null>(null);
  const [evidenceIaaLoading, setEvidenceIaaLoading] = useState(false);
  const [evidenceAdjudicationLoading, setEvidenceAdjudicationLoading] = useState(false);
  const [guidelineDraft, setGuidelineDraft] = useState<GuidelineDraftState>(EMPTY_GUIDELINE_DRAFT);
  const [documentFilter, setDocumentFilter] = useState<QueueStatus | "all">("all");
  const [toast, setToast] = useState<string | null>(null);
  const [submissions, setSubmissions] = useState<AnnotationSubmission[]>([]);
  const [workspaceMembers, setWorkspaceMembers] = useState<WorkspaceMember[]>([]);
  const [joinRequests, setJoinRequests] = useState<WorkspaceJoinRequest[]>([]);
  const [inviteRole, setInviteRole] = useState<WorkspaceRole>("annotator");
  const [lastInviteToken, setLastInviteToken] = useState<string | null>(null);
  const [submissionDocumentFilter, setSubmissionDocumentFilter] = useState<number | "all">("all");
  const [submissionAnnotatorFilter, setSubmissionAnnotatorFilter] = useState<string>("all");
  const [pendingConfirmation, setPendingConfirmation] =
    useState<PendingConfirmation | null>(null);
  const [confirmationBusy, setConfirmationBusy] = useState(false);
  const toastTimerRef = useRef<number | null>(null);
  const pubmedImportPanelRef = useRef<PubmedImportPanelHandle>(null);
  const taskComposerRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (isPersonalWorkspace) {
      setEvidenceAdjudicationDraft((current) => ({ ...current, strategy: "custom" }));
    }
  }, [isPersonalWorkspace]);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );

  const enabledTasks = useMemo(
    () => sortTasks(selectedProject?.tasks.filter((task) => task.enabled) ?? []),
    [selectedProject],
  );

  const tasksById = useMemo(
    () => new Map(selectedProject?.tasks.map((task) => [task.id, task]) ?? []),
    [selectedProject],
  );

  const evidenceVersionById = useMemo(
    () =>
      new Map(
        evidenceTargets.flatMap((target) =>
          target.versions.map((version) => [version.id, { target, version }] as const),
        ),
      ),
    [evidenceTargets],
  );

  const adjudicationTargetVersions = useMemo(
    () =>
      evidenceTargets.flatMap((target) =>
        target.versions
          .filter((version) => target.is_active || version.id === target.active_version_id)
          .map((version) => ({ target, version })),
      ),
    [evidenceTargets],
  );

  const selectedAssignmentTask = useMemo(() => {
    const taskId = Number(assignmentDraft.taskId || enabledTasks[0]?.id);
    return enabledTasks.find((task) => task.id === taskId) ?? null;
  }, [assignmentDraft.taskId, enabledTasks]);

  const assignmentEvidenceVersions = useMemo(
    () =>
      evidenceTargets
        .filter((target) => target.task_id === selectedAssignmentTask?.id && target.is_active)
        .flatMap((target) =>
          target.versions
            .filter((version) => version.id === target.active_version_id)
            .map((version) => ({ target, version })),
        ),
    [evidenceTargets, selectedAssignmentTask],
  );

  const entityLabelNames = useMemo(() => {
    if (!selectedProject) {
      return [];
    }
    const entityTask = selectedProject.tasks.find((task) => task.annotation_type === "entity");
    const names = [
      ...(entityTask?.labels ?? []).map((label) => label.name),
      ...(selectedProject.annotation_schema.labels.entity ?? []).map((label) => label.name),
    ];
    return Array.from(new Set(names));
  }, [selectedProject]);

  const documentsById = useMemo(
    () => new Map(documents.map((documentItem) => [documentItem.id, documentItem])),
    [documents],
  );

  const workspaceMembersByUserId = useMemo(
    () => new Map(workspaceMembers.map((member) => [member.user_id, member])),
    [workspaceMembers],
  );

  const assignableMembers = useMemo(
    () =>
      workspaceMembers
        .filter((member) => member.is_active)
        .sort((a, b) => a.username.localeCompare(b.username)),
    [workspaceMembers],
  );

  const assignmentsByDocument = useMemo(() => {
    const grouped = new Map<number, TaskAssignment[]>();
    for (const assignment of assignments) {
      const existing = grouped.get(assignment.document_id) ?? [];
      existing.push(assignment);
      grouped.set(assignment.document_id, existing);
    }
    return grouped;
  }, [assignments]);

  const adjudicationGuidelineVersions = useMemo(() => {
    const documentId = Number(evidenceAdjudicationDraft.documentId);
    const targetVersionId = Number(evidenceAdjudicationDraft.targetVersionId);
    const pinnedIds = new Set(
      assignments
        .filter(
          (assignment) =>
            assignment.document_id === documentId &&
            assignment.target_version_id === targetVersionId &&
            assignment.guideline_version_id !== null,
        )
        .map((assignment) => assignment.guideline_version_id as number),
    );
    return guidelines.filter((guideline) => pinnedIds.has(guideline.id));
  }, [assignments, evidenceAdjudicationDraft.documentId, evidenceAdjudicationDraft.targetVersionId, guidelines]);

  const availableTaskTypes = useMemo(() => {
    const existingTypes = new Set(selectedProject?.tasks.map((task) => task.annotation_type) ?? []);
    return ANNOTATION_TYPE_OPTIONS.filter((option) => !existingTypes.has(option.value));
  }, [selectedProject]);

  const assignmentCounts = useMemo(() => statusCount(assignments), [assignments]);
  const personalDocumentCounts = useMemo(
    () =>
      documents.reduce<Record<QueueStatus, number>>(
        (nextCounts, documentItem) => {
          const status = documentQueueStatusFromAssignments(
            assignmentsByDocument.get(documentItem.id) ?? [],
          );
          nextCounts[status] += 1;
          return nextCounts;
        },
        { todo: 0, partial: 0, review: 0, done: 0, blocked: 0 },
      ),
    [assignmentsByDocument, documents],
  );
  const counts = isPersonalWorkspace ? personalDocumentCounts : assignmentCounts;
  const donePct = percent(
    counts.done,
    isPersonalWorkspace ? documents.length : assignments.length,
  );
  const activeGuide = activeGuideline(guidelines);
  const annotatorRows = progress?.by_annotator ?? [];
  const visibleTabs = useMemo(
    () => managerTabsForWorkspace(isPersonalWorkspace),
    [isPersonalWorkspace],
  );
  const activeTab = visibleTabs.some((tab) => tab.id === currentTab) ? currentTab : "overview";
  const filteredSubmissions = useMemo(
    () =>
      submissions.filter(
        (submission) =>
            (submissionDocumentFilter === "all" ||
              submission.document_id === submissionDocumentFilter) &&
            (isPersonalWorkspace ||
              submissionAnnotatorFilter === "all" ||
              (submission.annotator_id ?? "") === submissionAnnotatorFilter),
        ),
      [isPersonalWorkspace, submissionAnnotatorFilter, submissionDocumentFilter, submissions],
    );
  const submissionAnnotators = useMemo(
    () =>
      Array.from(
        new Set(submissions.map((submission) => submission.annotator_id ?? "")),
      ).sort(),
    [submissions],
  );

  const showToast = useCallback((message: string): void => {
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    setToast(message);
    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
      toastTimerRef.current = null;
    }, 2400);
  }, []);

  const focusPubmedImportPanel = useCallback((): void => {
    pubmedImportPanelRef.current?.focusInput();
  }, []);

  const refreshWorkspaceUsers = useCallback(async (): Promise<void> => {
    if (activeWorkspaceId === null) {
      setWorkspaceMembers([]);
      setJoinRequests([]);
      return;
    }
    try {
      const [members, requests] = await Promise.all([
        listWorkspaceMembers(activeWorkspaceId),
        listWorkspaceJoinRequests(activeWorkspaceId),
      ]);
      setWorkspaceMembers(members);
      setJoinRequests(requests);
    } catch (loadError) {
      showToast(loadError instanceof Error ? loadError.message : "Unable to load team users");
    }
  }, [activeWorkspaceId, showToast]);

  const refreshSubmissions = useCallback(async (): Promise<void> => {
    if (selectedProjectId === null) {
      setSubmissions([]);
      return;
    }
    try {
      setSubmissions(await listSubmissions(selectedProjectId));
    } catch (loadError) {
      showToast(loadError instanceof Error ? loadError.message : "Unable to load submissions");
    }
  }, [selectedProjectId, showToast]);

  const closeModal = useCallback((): void => {
    setModal(null);
  }, [setModal]);

  const navigateProject = useCallback((projectId: number, tab: ManagerTab = currentTab, phase?: LoopPhase): void => {
    const path = `/projects/${projectId}/${tab}`;
    if (pathname !== path) {
      onNavigate(path);
    }
    setSelectedProjectId(projectId);
    openProjectTab(tab, phase);
  }, [currentTab, onNavigate, openProjectTab, pathname, setSelectedProjectId]);

  const navigateAdmin = useCallback((section: AdminSection): void => {
    const path = `/admin/${section}`;
    if (pathname !== path) {
      onNavigate(path);
    }
    setAdminSection(section);
    setAppMode("admin");
  }, [onNavigate, pathname, setAdminSection, setAppMode]);

  const refreshProjects = useCallback(async (
    preferredProjectId?: number | null,
    force = true,
  ): Promise<number | null> => {
    const urlMatch = pathname.match(/^\/projects\/(\d+)\/?([^/]*)?/);
    const routeProjectId = urlMatch ? Number(urlMatch[1]) : null;
    return loadProjects(preferredProjectId ?? routeProjectId, force, activeWorkspaceId);
  }, [activeWorkspaceId, loadProjects, pathname]);

  const refreshProjectData = useCallback(async (projectId: number, force = true): Promise<void> => {
    const [, guidelineList, targetList] = await Promise.all([
      loadProjectData(projectId, force),
      listGuidelineVersions(projectId),
      listEvidenceTargets(projectId),
    ]);
    setGuidelines(guidelineList);
    setEvidenceTargets(targetList);
  }, [loadProjectData]);

  const refreshProjectConfig = useCallback(async (projectId: number): Promise<void> => {
    await loadProject(projectId);
  }, [loadProject]);

  const refreshGuidelines = useCallback(async (projectId: number): Promise<GuidelineVersion[]> => {
    const guidelineList = await listGuidelineVersions(projectId);
    setGuidelines(guidelineList);
    return guidelineList;
  }, []);

  async function reExportSubmission(submission: AnnotationSubmission): Promise<void> {
    if (selectedProjectId === null) {
      return;
    }
    try {
      await createSubmission(selectedProjectId, submission.document_id, {
        kind: "re_export",
        annotator_id: submission.annotator_id,
      });
      await refreshSubmissions();
      showToast("Re-export created");
    } catch (actionError) {
      showToast(actionError instanceof Error ? actionError.message : "Re-export failed");
    }
  }

  async function saveSubmissionFile(submission: AnnotationSubmission): Promise<void> {
    try {
      const blob = await downloadSubmission(submission.id);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      try {
        anchor.href = objectUrl;
        anchor.download = submission.file_name;
        anchor.hidden = true;
        document.body.appendChild(anchor);
        anchor.click();
      } finally {
        anchor.remove();
        URL.revokeObjectURL(objectUrl);
      }
    } catch (actionError) {
      showToast(actionError instanceof Error ? actionError.message : "Download failed");
    }
  }

  async function removeSubmission(submission: AnnotationSubmission): Promise<void> {
    try {
      await deleteSubmission(submission.id);
      await refreshSubmissions();
      showToast("Submission file deleted");
    } catch (actionError) {
      showToast(actionError instanceof Error ? actionError.message : "Delete failed");
    }
  }

  useEffect(() => {
    return () => {
      if (toastTimerRef.current !== null) {
        window.clearTimeout(toastTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    void refreshWorkspaceUsers();
  }, [refreshWorkspaceUsers]);

  useEffect(() => {
    function applyRoute(nextPathname: string): void {
      const projectMatch = nextPathname.match(/^\/projects\/(\d+)\/?([^/]*)?/);
      if (projectMatch) {
        const nextProjectId = Number(projectMatch[1]);
        if (projectMatch[2] === "models") {
          setSelectedProjectId(nextProjectId);
          if (canUseTraining) {
            onNavigate(`/models?projectId=${nextProjectId}`, "replace");
            return;
          }
          const fallbackPath = `/projects/${nextProjectId}/overview`;
          if (nextPathname !== fallbackPath) {
            onNavigate(fallbackPath, "replace");
          }
          openProjectTab("overview");
          return;
        }
        const nextTab = normalizeTab(projectMatch[2]);
        setSelectedProjectId(nextProjectId);
        openProjectTab(nextTab);
        return;
      }

      const adminMatch = nextPathname.match(/^\/admin\/?([^/]*)?/);
      if (adminMatch) {
        if (workspaceRole !== "admin") {
          const fallbackPath = selectedProjectId
            ? `/projects/${selectedProjectId}/${currentTab}`
            : "/manager/projects";
          if (nextPathname !== fallbackPath) {
            onNavigate(fallbackPath, "replace");
          }
          setAppMode("project");
          return;
        }
        setAdminSection(normalizeAdminSection(adminMatch[1]));
        setAppMode("admin");
      }
    }

    applyRoute(pathname);
  }, [
    canUseTraining,
    currentTab,
    onNavigate,
    openProjectTab,
    pathname,
    selectedProjectId,
    setAdminSection,
    setAppMode,
    setSelectedProjectId,
    workspaceRole,
  ]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent): void => {
      if (
        event.defaultPrevented ||
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        event.shiftKey ||
        modal ||
        appMode === "admin" ||
        !selectedProjectId ||
        isHotkeyTargetIgnored(event.target)
      ) {
        return;
      }

      if (!/^[1-8]$/.test(event.key)) {
        return;
      }

      const tabIndex = Number(event.key) - 1;
      const targetTab = TABS[tabIndex]?.id;
      if (targetTab) {
        navigateProject(selectedProjectId, targetTab);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [appMode, modal, navigateProject, selectedProjectId]);

  useEffect(() => {
    let cancelled = false;

    async function loadProjectsForConsole(): Promise<void> {
      setError(null);
      try {
        const nextProjectId = await refreshProjects(undefined, false);
        if (
          !cancelled &&
          nextProjectId !== null &&
          pathname === "/manager/projects"
        ) {
          const path = `/projects/${nextProjectId}/overview`;
          onNavigate(path, "replace");
          setCurrentTab("overview");
        }
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Unable to load projects");
        }
      }
    }

    void loadProjectsForConsole();
    return () => {
      cancelled = true;
    };
  }, [onNavigate, pathname, refreshProjects, setCurrentTab, setError]);

  useEffect(() => {
    setAssignmentDraft(EMPTY_ASSIGNMENT_DRAFT);
    if (selectedProjectId === null) {
      clearProjectData();
      setGuidelines([]);
      setEvidenceTargets([]);
      setSubmissions([]);
      return;
    }

    const projectId = selectedProjectId;
    let cancelled = false;
    async function loadProjectDataForConsole(): Promise<void> {
      setError(null);
      try {
        await refreshProjectData(projectId, false);
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof Error ? caught.message : "Unable to load project data");
        }
      }
    }

    void loadProjectDataForConsole();
    return () => {
      cancelled = true;
    };
  }, [clearProjectData, refreshProjectData, selectedProjectId, setError, setAssignmentDraft]);

  useEffect(() => {
    if (currentTab === "export") {
      void refreshSubmissions();
    }
  }, [currentTab, refreshSubmissions]);

  useEffect(() => {
    setEvidenceAdjudicationDraft((current) => {
      const currentDocumentIsValid = documents.some(
        (item) => String(item.id) === current.documentId,
      );
      const currentTargetIsValid = adjudicationTargetVersions.some(
        (item) => String(item.version.id) === current.targetVersionId,
      );
      return {
        ...current,
        documentId: currentDocumentIsValid
          ? current.documentId
          : String(documents[0]?.id ?? ""),
        targetVersionId: currentTargetIsValid
          ? current.targetVersionId
          : String(adjudicationTargetVersions[0]?.version.id ?? ""),
        selectedAnnotationIds: [],
      };
    });
    setEvidenceComparison(null);
  }, [adjudicationTargetVersions, documents]);

  useEffect(() => {
    setEvidenceAdjudicationDraft((current) => {
      const currentGuidelineIsValid = adjudicationGuidelineVersions.some(
        (guideline) => String(guideline.id) === current.guidelineVersionId,
      );
      return {
        ...current,
        guidelineVersionId: currentGuidelineIsValid
          ? current.guidelineVersionId
          : String(adjudicationGuidelineVersions[0]?.id ?? ""),
        selectedAnnotationIds: [],
      };
    });
    setEvidenceComparison(null);
    setEvidenceIaa(null);
  }, [adjudicationGuidelineVersions]);

  useEffect(() => {
    const firstAvailableType = availableTaskTypes[0]?.value;
    if (firstAvailableType && !availableTaskTypes.some((type) => type.value === taskDraft.annotationType)) {
      setTaskDraft({ annotationType: firstAvailableType, displayName: "" });
    }
  }, [availableTaskTypes, taskDraft.annotationType]);

  useEffect(() => {
    setIsTaskComposerOpen(false);
  }, [selectedProjectId]);

  useEffect(() => {
    if (!isTaskComposerOpen) {
      return;
    }
    const animationFrame = window.requestAnimationFrame(() => {
      taskComposerRef.current?.scrollIntoView?.({ block: "center" });
      taskComposerRef.current
        ?.querySelector<HTMLElement>('[role="combobox"], input, button')
        ?.focus();
    });
    return () => window.cancelAnimationFrame(animationFrame);
  }, [isTaskComposerOpen]);

  async function handleCreateProject(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    const name = projectForm.name.trim();
    if (!name) {
      setError("Project name is required");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const created = await createProject({
        name,
        description: toNullableText(projectForm.description),
        annotation_validation_mode: projectForm.annotationValidationMode,
        annotation_schema: { labels: {} },
        workspace_id: activeWorkspaceId,
        tasks: projectForm.starterTaskTypes.map((annotationType, index) => ({
          annotation_type: annotationType,
          display_name: getTaskDisplayName(annotationType),
          sort_order: index,
          labels: [],
          enabled: true,
          settings: {},
        })),
        settings: {},
      });
      setProjectForm(EMPTY_PROJECT_FORM);
      closeModal();
      await refreshProjects(created.id);
      await refreshProjectData(created.id);
      navigateProject(created.id, "overview");
      showToast(
        isPersonalWorkspace
          ? `Project ${created.name} created. Complete setup to start annotating.`
          : `Project ${created.name} created`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create project");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateProjectValidation(mode: AnnotationValidationMode): Promise<void> {
    if (!selectedProject || selectedProject.annotation_validation_mode === mode) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const updated = await updateProject(selectedProject.id, { annotation_validation_mode: mode });
      replaceProject(updated);
      showToast("Project validation updated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update project");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateTask(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!selectedProject || availableTaskTypes.length === 0) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await createProjectTask(selectedProject.id, {
        annotation_type: taskDraft.annotationType,
        display_name: taskDraft.displayName.trim() || getTaskDisplayName(taskDraft.annotationType),
        enabled: true,
        sort_order: selectedProject.tasks.length,
        labels: [],
        settings: {},
      });
      setTaskDraft({
        annotationType:
          availableTaskTypes.find((option) => option.value !== taskDraft.annotationType)?.value ?? "entity",
        displayName: "",
      });
      await refreshProjectConfig(selectedProject.id);
      setIsTaskComposerOpen(false);
      showToast("Task added");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to add task");
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleTask(task: ProjectTask): Promise<void> {
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await updateProjectTask(selectedProject.id, task.id, { enabled: !task.enabled });
      await refreshProjectConfig(selectedProject.id);
      showToast(task.enabled ? "Task disabled" : "Task enabled");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update task");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteTask(task: ProjectTask): Promise<void> {
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await deleteProjectTask(selectedProject.id, task.id);
      await refreshProjectConfig(selectedProject.id);
      await loadProjectData(selectedProject.id, true);
      showToast("Task deleted");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to delete task");
    } finally {
      setBusy(false);
    }
  }

  async function handleAddLabel(task: ProjectTask): Promise<void> {
    if (!selectedProject) {
      return;
    }
    const draft = labelDrafts[task.id] ?? EMPTY_LABEL_DRAFT;
    const name = draft.name.trim();
    if (!name) {
      setError("Label name is required");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await updateProjectTask(selectedProject.id, task.id, {
        labels: [
          ...task.labels,
          {
            name,
            color: draft.color || FALLBACK_COLORS[0],
            description: toNullableText(draft.description),
          },
        ],
      });
      setLabelDrafts((previous) => {
        const next = { ...previous };
        delete next[task.id];
        return next;
      });
      await refreshProjectConfig(selectedProject.id);
      showToast("Label added");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to add label");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemoveLabel(task: ProjectTask, labelIndex: number): Promise<void> {
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await updateProjectTask(selectedProject.id, task.id, {
        labels: task.labels.filter((_label, index) => index !== labelIndex),
      });
      await refreshProjectConfig(selectedProject.id);
      showToast("Label removed");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to remove label");
    } finally {
      setBusy(false);
    }
  }

  async function handleToggleRelationConstraint(
    task: ProjectTask,
    relationLabel: string,
    side: "head" | "tail",
    entityLabel: string,
  ): Promise<void> {
    if (!selectedProject) {
      return;
    }

    const constraints = relationConstraintsOf(task);
    const current = constraints[relationLabel] ?? { head: [], tail: [] };
    const list = current[side];
    const nextList = list.includes(entityLabel)
      ? list.filter((item) => item !== entityLabel)
      : [...list, entityLabel];
    const nextConstraints = {
      ...constraints,
      [relationLabel]: { ...current, [side]: nextList },
    };

    setBusy(true);
    setError(null);
    try {
      await updateProjectTask(selectedProject.id, task.id, {
        settings: { ...task.settings, relation_constraints: nextConstraints },
      });
      await refreshProjectConfig(selectedProject.id);
      showToast("Relation constraints updated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update constraints");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateEvidenceSettings(
    task: ProjectTask,
    updates: EvidenceSettingsUpdate,
  ): Promise<void> {
    if (!selectedProject) {
      return;
    }
    const current = task.settings as Partial<EvidenceBlockTaskSettingsV1>;
    const settings = {
      ...task.settings,
      ...updates,
      keyboard_shortcuts: updates.keyboard_shortcuts
        ? { ...(current.keyboard_shortcuts ?? {}), ...updates.keyboard_shortcuts }
        : current.keyboard_shortcuts,
    };
    setBusy(true);
    setError(null);
    try {
      await updateProjectTask(selectedProject.id, task.id, { settings });
      await refreshProjectConfig(selectedProject.id);
      showToast("Evidence annotation settings updated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update evidence settings");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateEvidenceTarget(task: ProjectTask): Promise<void> {
    if (!selectedProject) {
      return;
    }
    const key = evidenceTargetDraft.key.trim().toLowerCase().replace(/\s+/g, "-");
    const name = evidenceTargetDraft.name.trim();
    const text = evidenceTargetDraft.text.trim();
    if (!key || !name || !text) {
      setError("Target key, name, and target text are required");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      const created = await createEvidenceTarget(selectedProject.id, {
        task_id: task.id,
        key,
        name,
        initial_version: {
          text,
          guidance: toNullableText(evidenceTargetDraft.guidance),
        },
      });
      const firstVersion = created.versions[0];
      if (firstVersion) {
        await activateEvidenceTarget(selectedProject.id, created.id, firstVersion.id);
      }
      setEvidenceTargets(await listEvidenceTargets(selectedProject.id));
      await refreshProjectConfig(selectedProject.id);
      setEvidenceTargetDraft(EMPTY_EVIDENCE_TARGET_DRAFT);
      showToast("Evidence target created and activated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create evidence target");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateEvidenceTargetVersion(target: EvidenceTarget): Promise<void> {
    if (!selectedProject) {
      return;
    }
    const text = (evidenceVersionDrafts[target.id] ?? "").trim();
    if (!text) {
      setError("New target text is required");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await createEvidenceTargetVersion(selectedProject.id, target.id, { text });
      setEvidenceTargets(await listEvidenceTargets(selectedProject.id));
      setEvidenceVersionDrafts((current) => ({ ...current, [target.id]: "" }));
      showToast("Immutable target version created");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create target version");
    } finally {
      setBusy(false);
    }
  }

  async function handleActivateEvidenceTarget(
    target: EvidenceTarget,
    versionId: number,
  ): Promise<void> {
    if (!selectedProject) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await activateEvidenceTarget(selectedProject.id, target.id, versionId);
      setEvidenceTargets(await listEvidenceTargets(selectedProject.id));
      await refreshProjectConfig(selectedProject.id);
      showToast("Evidence target version activated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to activate target version");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeactivateEvidenceTarget(target: EvidenceTarget): Promise<void> {
    if (!selectedProject) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await deactivateEvidenceTarget(selectedProject.id, target.id);
      setEvidenceTargets(await listEvidenceTargets(selectedProject.id));
      await refreshProjectConfig(selectedProject.id);
      showToast("Evidence target deactivated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to deactivate target");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateAssignment(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!selectedProject) {
      return;
    }

    const taskId = Number(assignmentDraft.taskId || enabledTasks[0]?.id);
    const documentId = Number(assignmentDraft.documentId || documents[0]?.id);
    const assigneeUserId = Number(assignmentDraft.assigneeUserId || assignableMembers[0]?.user_id);
    const targetVersionId = Number(
      assignmentDraft.targetVersionId || assignmentEvidenceVersions[0]?.version.id,
    );
    if (!taskId || !documentId || !assigneeUserId) {
      setError("Task, document, and annotator are required");
      return;
    }
    if (selectedAssignmentTask?.annotation_type === "evidence_block" && !targetVersionId) {
      setError("An active evidence target is required for evidence-block assignments");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await createTaskAssignment(selectedProject.id, {
        task_id: taskId,
        document_id: documentId,
        assignee_user_id: assigneeUserId,
        status: assignmentDraft.status,
        target_version_id:
          selectedAssignmentTask?.annotation_type === "evidence_block" ? targetVersionId : null,
      });
      setAssignmentDraft(EMPTY_ASSIGNMENT_DRAFT);
      await loadProjectData(selectedProject.id, true);
      showToast("Assignment created");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create assignment");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateAssignmentStatus(
    assignment: TaskAssignment,
    status: TaskAssignmentStatus,
  ): Promise<void> {
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await updateTaskAssignment(selectedProject.id, assignment.id, { status });
      await loadProjectData(selectedProject.id, true);
      showToast("Assignment updated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update assignment");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteAssignment(assignment: TaskAssignment): Promise<void> {
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await deleteTaskAssignment(selectedProject.id, assignment.id);
      await loadProjectData(selectedProject.id, true);
      showToast("Assignment deleted");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to delete assignment");
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateMemberRole(member: WorkspaceMember, nextRole: WorkspaceRole): Promise<void> {
    if (activeWorkspaceId === null || member.role === nextRole) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await updateWorkspaceMemberRole(activeWorkspaceId, member.user_id, nextRole);
      await refreshWorkspaceUsers();
      showToast("Member role updated");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update member role");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemoveMember(member: WorkspaceMember): Promise<void> {
    if (activeWorkspaceId === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await deleteWorkspaceMember(activeWorkspaceId, member.user_id);
      await Promise.all([
        refreshWorkspaceUsers(),
        selectedProject
          ? loadProjectData(selectedProject.id, true)
          : Promise.resolve(),
      ]);
      showToast("Member removed and open assignments withdrawn");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to remove member");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateInvite(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (activeWorkspaceId === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const invite = await createWorkspaceInvite(activeWorkspaceId, inviteRole);
      setLastInviteToken(invite.token);
      showToast("Invite created");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create invite");
    } finally {
      setBusy(false);
    }
  }

  async function handleDecideJoinRequest(
    request: WorkspaceJoinRequest,
    approve: boolean,
  ): Promise<void> {
    setBusy(true);
    setError(null);
    try {
      if (approve) {
        await approveJoinRequest(request.id);
      } else {
        await rejectJoinRequest(request.id);
      }
      await refreshWorkspaceUsers();
      showToast(approve ? "Join request approved" : "Join request rejected");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to update join request");
    } finally {
      setBusy(false);
    }
  }

  async function handleCreateGuideline(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (!selectedProject) {
      return;
    }

    setBusy(true);
    setError(null);
    try {
      await createGuidelineVersion({
        project_id: selectedProject.id,
        version_label: guidelineDraft.versionLabel.trim() || nextGuidelineVersionLabel(guidelines),
        markdown: guidelineDraft.markdown,
        author_id: "manager",
        status: "active",
      });
      const guidelineList = await refreshGuidelines(selectedProject.id);
      setGuidelineDraft({
        versionLabel: nextGuidelineVersionLabel(guidelineList),
        markdown: guidelineDraft.markdown,
      });
      showToast("Guideline version created");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to create guideline");
    } finally {
      setBusy(false);
    }
  }

  function toggleStarterTask(annotationType: AnnotationType): void {
    setProjectForm((previous) => {
      const exists = previous.starterTaskTypes.includes(annotationType);
      return {
        ...previous,
        starterTaskTypes: exists
          ? previous.starterTaskTypes.filter((taskType) => taskType !== annotationType)
          : [...previous.starterTaskTypes, annotationType],
      };
    });
  }

  function documentQueueStatus(documentItem: Document): QueueStatus {
    return documentQueueStatusFromAssignments(
      assignmentsByDocument.get(documentItem.id) ?? [],
    );
  }

  function renderTopBar(): React.ReactElement {
    const brandPath = selectedProject
      ? `/projects/${selectedProject.id}/overview`
      : "/manager/projects";
    return (
      <header className="mc-top">
        <div className="mc-top-left">
          <a
            className="mc-brand"
            href={brandPath}
            onClick={(event) => {
              if (!selectedProject || !shouldHandleSpaClick(event)) {
                return;
              }
              event.preventDefault();
              navigateProject(selectedProject.id, "overview");
            }}
          >
            <BrandLogo />
            <span>
              <strong>AL-MedLit</strong>
              <small>
                {isPersonalWorkspace ? "Project setup" : role === "admin" ? "Admin console" : "Manager console"}
              </small>
            </span>
          </a>
          {moduleSwitcher}

          {selectedProject && appMode === "project" ? (
            <div className="mc-project-picker">
              <span style={{ background: projectAccent(selectedProject) }}>
                {projectCode(selectedProject)[0]}
              </span>
              <Selector
                label="Project"
                isLabelHidden
                value={String(selectedProject.id)}
                onChange={(value) => navigateProject(Number(value), "overview")}
                placement="below"
                options={projects.map((project) => ({
                  value: String(project.id),
                  label: project.name,
                }))}
                width="100%"
              />
            </div>
          ) : (
            <span className="mc-admin-context">
              {isPersonalWorkspace ? <Icon name="docs" /> : <Icon name="admin" />}
              {isPersonalWorkspace ? "Project setup" : "Administration"}
            </span>
          )}
        </div>

        <div className="mc-top-right">
          {workspaceSwitcher}
          <span className="mc-user" aria-label={currentUsername ?? "Current user"}>
            {userInitials(currentUsername)}
          </span>
          <Button className="logout-button" label="Sign out" size="sm" onClick={onLogout} />
        </div>
      </header>
    );
  }

  function renderTabStrip(): React.ReactElement {
    if (appMode === "admin") {
      return (
        <div className="mc-adminbar">
          <Button
            label="Back to projects"
            size="sm"
            onClick={() => selectedProject && navigateProject(selectedProject.id, currentTab)}
          />
          <strong>Administration</strong>
        </div>
      );
    }

    return (
      <div className="mc-tabstrip">
        <TabList
          className="mc-tabs"
          value={activeTab}
          onChange={(value) => selectedProject && navigateProject(selectedProject.id, value as ManagerTab)}
          size="md"
        >
          {visibleTabs.map((tab) => {
            const badge =
              tab.id === "documents"
                ? documents.length
                : tab.id === "progress"
                  ? `${counts.done}/${isPersonalWorkspace ? documents.length : assignments.length}`
                  : tab.id === "tasks"
                    ? selectedProject?.tasks.length
                    : tab.id === "guidelines"
                    ? guidelines.length
                    : null;
            return (
              <Tab
                key={tab.id}
                value={tab.id}
                label={tab.label}
                href={
                  selectedProject
                    ? `/projects/${selectedProject.id}/${tab.id}`
                    : "/manager/projects"
                }
                onClick={(event) => {
                  if (!selectedProject || !shouldHandleSpaClick(event)) {
                    return;
                  }
                  event.preventDefault();
                  navigateProject(selectedProject.id, tab.id);
                }}
                icon={<Icon name={tab.icon} />}
                endContent={
                  badge !== null && badge !== undefined ? (
                    <Badge className="mc-tab-badge" variant="neutral" label={badge} />
                  ) : undefined
                }
              />
            );
          })}
        </TabList>
        <span className="mc-context-label">{selectedProject?.name ?? "No project"}</span>
      </div>
    );
  }

  function renderOverview(): React.ReactElement {
    if (!selectedProject) {
      return <EmptyState title="No project selected" detail="Create or select a project to open the console." />;
    }

    const reviewCount = counts.review;
    const blockedCount = counts.blocked;
    const unassignedDocuments = documents.filter(
      (documentItem) => (assignmentsByDocument.get(documentItem.id) ?? []).length === 0,
    ).length;
    const personalReadiness = getPersonalProjectReadiness(
      selectedProject,
      documents.length,
      evidenceTargets,
    );
    const runPersonalNextAction = (
      action: PersonalProjectReadiness["nextAction"] = personalReadiness.nextAction,
    ): void => {
      if (action === "tasks") {
        navigateProject(selectedProject.id, "tasks");
      } else if (action === "documents") {
        navigateProject(selectedProject.id, "documents");
      } else {
        onMyWorkNavigate();
      }
    };
    const personalPrimaryLabel =
      personalReadiness.nextAction === "tasks"
        ? "Configure tasks"
        : personalReadiness.nextAction === "documents"
          ? "Import documents"
          : personalReadiness.usesFreeFormLabels
            ? "Start with free-form labels"
            : "Start annotating";

    return (
      <div className="mc-page">
        <div className="mc-pagehead">
          <div>
            <h1>{selectedProject.name}</h1>
            <p>
              {selectedProject.description ??
                (isPersonalWorkspace
                  ? "Project setup and document progress."
                  : "Project operations and annotation quality dashboard.")}
            </p>
          </div>
          <div className="mc-actions">
            <Button
              label="New Project"
              icon={<Icon name="plus" />}
              onClick={() => setModal("newProject")}
            />
            {isPersonalWorkspace ? (
              <Button
                label={personalPrimaryLabel}
                variant="primary"
                onClick={() => runPersonalNextAction()}
              />
            ) : null}
            <Button label="Export" icon={<Icon name="export" />} onClick={() => openProjectTab("export")} />
            {!isPersonalWorkspace ? (
              <Button
                variant="primary"
                label="Run AL cycle"
                icon={<Icon name="loop" />}
                onClick={() => openProjectTab("loop", "select")}
              />
            ) : null}
          </div>
        </div>

        <div className="mc-stats">
          <article className="mc-stat">
            <span>Documents done</span>
            <strong>
              {counts.done}
              <small> / {documents.length}</small>
            </strong>
          </article>
          <article className="mc-stat">
            <span>Completion</span>
            <strong>{donePct}%</strong>
          </article>
          <article className="mc-stat">
            <span>{isPersonalWorkspace ? "Tasks" : "Assignments"}</span>
            <strong>{isPersonalWorkspace ? enabledTasks.length : assignments.length}</strong>
          </article>
          <article className="mc-stat">
            <span>Guidelines</span>
            <strong>{guidelines.length}</strong>
          </article>
        </div>

        <section className="mc-panel">
          <div className="mc-panel-head">
            <h2>{isPersonalWorkspace ? "Project Setup" : "Needs Your Attention"}</h2>
            <span className="mc-badge neutral">
              {isPersonalWorkspace
                ? personalReadiness.canStart
                  ? "Ready"
                  : "Setup required"
                : `${reviewCount + blockedCount + unassignedDocuments} items`}
            </span>
          </div>
          {isPersonalWorkspace ? (
            <div className="mc-progress-list" aria-label="Personal project readiness">
              {personalReadiness.steps.map((step, index) => {
                const statusLabel =
                  step.status === "complete"
                    ? "Complete"
                    : step.status === "ready"
                      ? "Ready"
                      : step.status === "warning"
                        ? "Optional setup"
                        : "Blocked";
                const statusClass =
                  step.status === "complete" || step.status === "ready"
                    ? "done"
                    : step.status === "warning"
                      ? "partial"
                      : "blocked";
                return (
                  <div key={step.id}>
                    <span className="mc-primary-cell">
                      <strong>{index + 1}. {step.title}</strong>
                      <span>{step.detail}</span>
                    </span>
                    <span className="mc-row-actions">
                      <span className={`mc-badge ${statusClass}`}>{statusLabel}</span>
                      {step.id === "tasks" ? (
                        <Button
                          label={step.status === "complete" ? "Review tasks" : "Configure tasks"}
                          size="sm"
                          onClick={() => runPersonalNextAction("tasks")}
                        />
                      ) : step.id === "documents" ? (
                        <Button
                          label={documents.length > 0 ? "Manage documents" : "Import documents"}
                          size="sm"
                          onClick={() => runPersonalNextAction("documents")}
                        />
                      ) : personalReadiness.canStart ? (
                        <Button
                          label={
                            personalReadiness.usesFreeFormLabels
                              ? "Start with free-form labels"
                              : "Start annotating"
                          }
                          size="sm"
                          variant="primary"
                          onClick={() => runPersonalNextAction("my-work")}
                        />
                      ) : null}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="mc-attention-grid">
              <button type="button" onClick={() => openProjectTab("loop", "adjudicate")}>
                <strong>{reviewCount}</strong>
                <span>Assignments in review</span>
              </button>
              <button type="button" onClick={() => openProjectTab("documents")}>
                <strong>{unassignedDocuments}</strong>
                <span>Documents without assignments</span>
              </button>
              <button type="button" onClick={() => openProjectTab("tasks", "reflect")}>
                <strong>{blockedCount}</strong>
                <span>Blocked assignments</span>
              </button>
              <button type="button" onClick={() => openProjectTab("guidelines")}>
                <strong>{activeGuide ? activeGuide.version_label : "-"}</strong>
                <span>Active guideline</span>
              </button>
            </div>
          )}
        </section>

        <div className="mc-grid mc-grid-lower">
          <section className="mc-panel">
            <div className="mc-panel-head">
              <h2>Annotation Progress</h2>
              <button type="button" onClick={() => openProjectTab("progress")}>
                Open Progress
              </button>
            </div>
            <StatusMeter
              counts={counts}
              label={isPersonalWorkspace ? "Document status meter" : "Assignment status meter"}
            />
            <div className="mc-legend">
              {(["done", "partial", "review", "todo", "blocked"] as QueueStatus[]).map((status) => (
                <span key={status}>
                  <i className={status} />
                  {formatStatus(status)} {counts[status]}
                </span>
              ))}
            </div>
            <div className="mc-divider" />
            <div className="mc-progress-list">
              {(progress?.by_task ?? []).map((item) => (
                <div key={item.task_id}>
                  <span>{item.display_name}</span>
                  <b>{item.total}</b>
                </div>
              ))}
              {(progress?.by_task ?? []).length === 0 ? <p className="mc-muted">No task progress yet.</p> : null}
            </div>
          </section>

          {!isPersonalWorkspace ? (
            <section className="mc-panel">
              <div className="mc-panel-head">
                <h2>Team</h2>
                <button type="button" onClick={() => openProjectTab("progress")}>
                  All Annotators
                </button>
              </div>
              <table className="mc-table">
                <thead>
                  <tr>
                    <th>Annotator</th>
                    <th>Assigned</th>
                    <th>Done</th>
                  </tr>
                </thead>
                <tbody>
                  {annotatorRows.map((row) => (
                    <tr key={row.assignee_user_id ?? row.annotator_id}>
                      <td>
                        <span className="mc-avatar-cell">
                          <Avatar id={row.annotator_id} />
                          {memberLabel(
                            row.assignee_user_id !== null
                              ? workspaceMembersByUserId.get(row.assignee_user_id)
                              : undefined,
                            row.annotator_id,
                          )}
                        </span>
                      </td>
                      <td>{row.total}</td>
                      <td>{row.by_status.completed ?? row.by_status.submitted ?? 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {annotatorRows.length === 0 ? (
                <EmptyState title="No annotators yet" detail="Assignments will populate this table." />
              ) : null}
            </section>
          ) : (
            <section className="mc-panel">
              <div className="mc-panel-head">
                <h2>Personal Workspace</h2>
                <button type="button" onClick={onMyWorkNavigate}>
                  Open My Work
                </button>
              </div>
              <div className="mc-progress-list">
                <div>
                  <span>Documents</span>
                  <b>{documents.length}</b>
                </div>
                <div>
                  <span>Tasks</span>
                  <b>{enabledTasks.length}</b>
                </div>
                <div>
                  <span>Completed</span>
                  <b>{counts.done}</b>
                </div>
              </div>
            </section>
          )}
        </div>
      </div>
    );
  }

  function renderDocuments(): React.ReactElement {
    const filteredDocuments = documents.filter((documentItem) => {
      if (documentFilter === "all") {
        return true;
      }
      return documentQueueStatus(documentItem) === documentFilter;
    });

    return (
      <div className="mc-page wide">
        <div className="mc-pagehead">
          <div>
            <h1>Documents</h1>
            <p>
              {isPersonalWorkspace
                ? `Project documents for ${selectedProject?.name ?? "the selected project"}.`
                : `Corpus management for ${selectedProject?.name ?? "the selected project"}.`}
            </p>
          </div>
          <div className="mc-actions">
            <Button label="Freeze snapshot" isDisabled />
            <Button label="Import documents" variant="primary" onClick={focusPubmedImportPanel} />
          </div>
        </div>

        <div className="mc-two-col">
          <section className="mc-panel">
            <SegmentedControl
              className="mc-segmented"
              label="Document status filter"
              value={documentFilter}
              onChange={(value) => setDocumentFilter(value as QueueStatus | "all")}
              size="sm"
            >
              {(["all", "todo", "partial", "review", "done", "blocked"] as Array<QueueStatus | "all">).map((status) => (
                <SegmentedControlItem
                  key={status}
                  value={status}
                  label={status === "all" ? "All" : formatStatus(status)}
                />
              ))}
            </SegmentedControl>
            <div className="mc-table-scroll">
              <table className="mc-table">
                <thead>
                  <tr>
                    <th>Document</th>
                    <th>Source</th>
                    <th>Words</th>
                    <th>Status</th>
                    {!isPersonalWorkspace ? <th>Annotators</th> : null}
                  </tr>
                </thead>
                <tbody>
                  {filteredDocuments.map((documentItem) => {
                    const documentAssignments = assignmentsByDocument.get(documentItem.id) ?? [];
                    const queueStatus = documentQueueStatus(documentItem);
                    return (
                      <tr key={documentItem.id}>
                        <td>
                          <div className="mc-primary-cell">
                            <strong>{documentTitle(documentItem)}</strong>
                            <span>{documentItem.external_id ?? `Document ${documentItem.id}`}</span>
                          </div>
                        </td>
                        <td>{documentItem.source ?? "manual"}</td>
                        <td>{wordCount(documentItem)}</td>
                        <td>
                          <StatusBadge status={queueStatus} />
                        </td>
                        {!isPersonalWorkspace ? (
                          <td>
                            <div className="mc-avatar-stack">
                              {documentAssignments.slice(0, 4).map((assignment) => (
                                <Avatar
                                  id={
                                    memberLabel(
                                      workspaceMembersByUserId.get(assignment.assignee_user_id),
                                      assignment.annotator_id,
                                    )
                                  }
                                  key={assignment.id}
                                />
                              ))}
                            </div>
                          </td>
                        ) : null}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {filteredDocuments.length === 0 ? <EmptyState title="No documents" detail="Import documents to start corpus setup." /> : null}
          </section>

          <aside className="mc-rail">
            {selectedProject ? (
              <PubmedImportPanel
                ref={pubmedImportPanelRef}
                projectId={selectedProject.id}
                variant={isPersonalWorkspace ? "personal" : "team"}
                onImported={async () => {
                  await loadProjectData(selectedProject.id, true);
                }}
                onOpenWork={onMyWorkNavigate}
              />
            ) : null}
            <section className="mc-panel">
              <div className="mc-panel-head">
                <h2>Source Breakdown</h2>
              </div>
              <div className="mc-progress-list">
                {Object.entries(
                  documents.reduce<Record<string, number>>((accumulator, documentItem) => {
                    const source = documentItem.source ?? "manual";
                    accumulator[source] = (accumulator[source] ?? 0) + 1;
                    return accumulator;
                  }, {}),
                ).map(([source, count]) => (
                  <div key={source}>
                    <span>{source}</span>
                    <b>{count}</b>
                  </div>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </div>
    );
  }

  function renderTasks(): React.ReactElement {
    return (
      <div className="mc-page wide">
        <div className="mc-pagehead">
          <div>
            <h1>Tasks</h1>
            <p>
              {isPersonalWorkspace
                ? "Schema-driven annotation configuration."
                : "Schema-driven annotation configuration and assignment."}
            </p>
          </div>
          <Selector
            className="mc-inline-select"
            label="Validation"
            value={selectedProject?.annotation_validation_mode ?? "relaxed"}
            onChange={(value) => void handleUpdateProjectValidation(value as AnnotationValidationMode)}
            placement="below"
            isDisabled={!selectedProject || busy}
            options={[
              { value: "relaxed", label: "Relaxed" },
              { value: "strict", label: "Strict" },
            ]}
          />
        </div>

        {!isPersonalWorkspace ? (
          <SegmentedControl
            className="mc-segmented"
            label="Task view"
            value={tasksSubView}
            onChange={(value) => setTasksSubView(value as typeof tasksSubView)}
          >
            <SegmentedControlItem value="schema" label="Schema" />
            <SegmentedControlItem value="assign" label="Assign" />
          </SegmentedControl>
        ) : null}

        {isPersonalWorkspace || tasksSubView === "schema" ? renderTaskSchema() : renderAssignments()}
      </div>
    );
  }

  function renderTaskSchema(): React.ReactElement {
    if (!selectedProject) {
      return <EmptyState title="No project selected" detail="Select a project to edit its schema." />;
    }
    const personalReadiness = getPersonalProjectReadiness(
      selectedProject,
      documents.length,
      evidenceTargets,
    );
    const hasEnabledEntityTask = selectedProject.tasks.some(
      (task) => task.enabled && task.annotation_type === "entity",
    );

    return (
      <section className="mc-panel">
        <div className="mc-panel-head mc-task-panel-head">
          <div className="mc-panel-title">
            <h2>Annotation Tasks</h2>
            <span className="mc-badge neutral">{selectedProject.tasks.length}</span>
          </div>
          <Button
            label={isTaskComposerOpen ? "Cancel" : "Add task"}
            icon={isTaskComposerOpen ? undefined : <Icon name="plus" />}
            size="sm"
            variant={isTaskComposerOpen ? "ghost" : "primary"}
            onClick={() => setIsTaskComposerOpen((isOpen) => !isOpen)}
            isDisabled={busy || availableTaskTypes.length === 0}
            aria-expanded={isTaskComposerOpen}
            aria-controls="mc-task-composer"
          />
        </div>

        {isPersonalWorkspace &&
        personalReadiness.canStart &&
        personalReadiness.usesFreeFormLabels ? (
          <div className="mc-progress-list">
            <div>
              <span className="mc-primary-cell">
                <strong>Free-form labels are available</strong>
                <span>
                  Relaxed validation lets you start now and add preset labels later.
                </span>
              </span>
              <Button
                label="Start with free-form labels"
                size="sm"
                variant="primary"
                onClick={onMyWorkNavigate}
              />
            </div>
          </div>
        ) : null}

        <div className="mc-task-stack">
          {sortTasks(selectedProject.tasks).map((task) => {
            const draft = labelDrafts[task.id] ?? EMPTY_LABEL_DRAFT;
            const evidenceSettings = task.settings as Partial<EvidenceBlockTaskSettingsV1>;
            const effectiveLabelCount = effectiveTaskLabelCount(selectedProject, task);
            const hasActiveEvidenceTarget = evidenceTargets.some(
              (target) =>
                target.task_id === task.id &&
                target.is_active &&
                target.active_version_id !== null,
            );
            return (
              <article className="mc-task-card" key={task.id}>
                <header>
                  <div>
                    <h3>{task.display_name}</h3>
                    <span>{task.annotation_type}</span>
                  </div>
                  <div className="mc-row-actions">
                    <button type="button" onClick={() => void handleToggleTask(task)} disabled={busy}>
                      {task.enabled ? "Disable" : "Enable"}
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setPendingConfirmation({
                          title: "Delete task?",
                          description: `Delete “${task.display_name}”? Its labels and assignments will be removed. This cannot be undone.`,
                          confirmLabel: "Delete task",
                          action: () => handleDeleteTask(task),
                        })
                      }
                      disabled={busy}
                    >
                      Delete
                    </button>
                  </div>
                </header>
                <div className="mc-label-list">
                  {task.labels.map((label, index) => (
                    <span className="mc-label-chip" key={`${task.id}-${label.name}-${index}`}>
                      <i style={{ background: label.color }} />
                      {label.name}
                      <button type="button" onClick={() => void handleRemoveLabel(task, index)} aria-label={`Remove ${label.name}`}>
                        x
                      </button>
                    </span>
                  ))}
                </div>
                {task.enabled &&
                task.annotation_type !== "evidence_block" &&
                effectiveLabelCount === 0 ? (
                  <p className="mc-muted">
                    {selectedProject.annotation_validation_mode === "strict"
                      ? "Required: add at least one label before annotation."
                      : "Optional: add preset labels, or explicitly start with free-form labels."}
                  </p>
                ) : null}
                {task.enabled &&
                task.annotation_type === "relation" &&
                !hasEnabledEntityTask ? (
                  <p className="mc-muted">
                    Required: enable an entity task before annotating relations.
                  </p>
                ) : null}
                {task.enabled &&
                task.annotation_type === "evidence_block" &&
                !hasActiveEvidenceTarget ? (
                  <p className="mc-muted">
                    Required: add and activate an evidence target before annotation.
                  </p>
                ) : null}
                {task.annotation_type === "relation" && task.labels.length > 0 ? (
                  <div className="mc-constraints">
                    <h4>Type constraints</h4>
                    {entityLabelNames.length === 0 ? (
                      <p className="mc-muted">Add entity labels first to constrain relation endpoints.</p>
                    ) : (
                      task.labels.map((relationLabel) => {
                        const constraint =
                          relationConstraintsOf(task)[relationLabel.name] ?? { head: [], tail: [] };
                        return (
                          <div className="mc-constraint-row" key={`constraint-${relationLabel.name}`}>
                            <strong>{relationLabel.name}</strong>
                            {(["head", "tail"] as const).map((side) => (
                              <div className="mc-constraint-side" key={side}>
                                <span>{side}</span>
                                {entityLabelNames.map((entityLabel) => (
                                  <button
                                    className={`mc-chip-toggle${
                                      constraint[side].includes(entityLabel) ? " active" : ""
                                    }`}
                                    key={entityLabel}
                                    type="button"
                                    onClick={() =>
                                      void handleToggleRelationConstraint(
                                        task,
                                        relationLabel.name,
                                        side,
                                        entityLabel,
                                      )
                                    }
                                    disabled={busy}
                                  >
                                    {entityLabel}
                                  </button>
                                ))}
                                {constraint[side].length === 0 ? <em>any</em> : null}
                              </div>
                            ))}
                          </div>
                        );
                      })
                    )}
                  </div>
                ) : null}
                {task.annotation_type === "evidence_block" ? (
                  <div className="mc-evidence-targets">
                    <header>
                      <h4>Versioned evidence targets</h4>
                      <span>
                        {evidenceTargets.filter((target) => target.task_id === task.id && target.is_active).length} active
                      </span>
                    </header>
                    <details className="mc-evidence-settings">
                      <summary>Annotation policies and shortcuts</summary>
                      <div className="mc-evidence-policy-grid">
                        {([
                          ["multi_paragraph_allowed", "Allow multi-paragraph blocks"],
                          ["cross_section_allowed", "Allow cross-section blocks"],
                          ["same_target_overlap_allowed", "Allow same-target overlap"],
                          ["adjacency_allowed", "Allow adjacent blocks"],
                        ] as const).map(([key, label]) => (
                          <label key={key}>
                            <input
                              type="checkbox"
                              checked={
                                evidenceSettings[key] ??
                                (key === "multi_paragraph_allowed" || key === "adjacency_allowed")
                              }
                              onChange={(event) =>
                                void handleUpdateEvidenceSettings(task, {
                                  [key]: event.target.checked,
                                })
                              }
                              disabled={busy}
                            />
                            {label}
                          </label>
                        ))}
                        <label>
                          <span>Soft token warning</span>
                          <input
                            type="number"
                            min="1"
                            defaultValue={evidenceSettings.soft_token_warning ?? 3072}
                            onBlur={(event) => {
                              const value = Number(event.target.value);
                              if (Number.isInteger(value) && value > 0) {
                                void handleUpdateEvidenceSettings(task, {
                                  soft_token_warning: value,
                                });
                              }
                            }}
                            disabled={busy}
                          />
                        </label>
                      </div>
                      <div className="mc-shortcut-grid">
                        {EVIDENCE_SHORTCUT_LABELS.map(({ key, label }) => (
                          <label key={key}>
                            <span>{label}</span>
                            <input
                              defaultValue={evidenceSettings.keyboard_shortcuts?.[key] ?? ""}
                              onBlur={(event) => {
                                const value = event.target.value.trim();
                                if (value) {
                                  void handleUpdateEvidenceSettings(task, {
                                    keyboard_shortcuts: { [key]: value },
                                  });
                                }
                              }}
                              disabled={busy}
                            />
                          </label>
                        ))}
                      </div>
                    </details>
                    {evidenceTargets
                      .filter((target) => target.task_id === task.id)
                      .map((target) => (
                        <article key={target.id}>
                          <div className="mc-evidence-target-head">
                            <span>
                              <strong>{target.name}</strong>
                              <code>{target.key}</code>
                            </span>
                            {target.is_active ? (
                              <button type="button" onClick={() => void handleDeactivateEvidenceTarget(target)} disabled={busy}>
                                Deactivate
                              </button>
                            ) : null}
                          </div>
                          <div className="mc-target-versions">
                            {target.versions.map((version) => (
                              <button
                                className={target.active_version_id === version.id ? "active" : ""}
                                key={version.id}
                                type="button"
                                onClick={() => void handleActivateEvidenceTarget(target, version.id)}
                                disabled={busy || target.active_version_id === version.id}
                                title={version.text}
                              >
                                v{version.version_number}
                                {target.active_version_id === version.id ? " · active" : " · activate"}
                              </button>
                            ))}
                          </div>
                          <p>
                            {target.versions.find((version) => version.id === target.active_version_id)?.text ??
                              target.versions[target.versions.length - 1]?.text}
                          </p>
                          <div className="mc-target-version-form">
                            <input
                              value={evidenceVersionDrafts[target.id] ?? ""}
                              onChange={(event) =>
                                setEvidenceVersionDrafts((current) => ({
                                  ...current,
                                  [target.id]: event.target.value,
                                }))
                              }
                            />
                            <button type="button" onClick={() => void handleCreateEvidenceTargetVersion(target)} disabled={busy}>
                              Create version
                            </button>
                          </div>
                        </article>
                      ))}
                    <div className="mc-target-create-form">
                      <input
                        value={evidenceTargetDraft.key}
                        onChange={(event) => setEvidenceTargetDraft((current) => ({ ...current, key: event.target.value }))}
                        placeholder="e.g., primary-outcome…"
                        aria-label="Evidence target key"
                      />
                      <input
                        value={evidenceTargetDraft.name}
                        onChange={(event) => setEvidenceTargetDraft((current) => ({ ...current, name: event.target.value }))}
                        aria-label="Evidence target name"
                      />
                      <textarea
                        value={evidenceTargetDraft.text}
                        onChange={(event) => setEvidenceTargetDraft((current) => ({ ...current, text: event.target.value }))}
                        aria-label="Evidence target text"
                      />
                      <textarea
                        value={evidenceTargetDraft.guidance}
                        onChange={(event) => setEvidenceTargetDraft((current) => ({ ...current, guidance: event.target.value }))}
                        aria-label="Evidence target guidance"
                      />
                      <button type="button" onClick={() => void handleCreateEvidenceTarget(task)} disabled={busy}>
                        Add and activate target
                      </button>
                    </div>
                  </div>
                ) : null}
                <div className="mc-label-form">
                  <input
                    value={draft.name}
                    onChange={(event) => setLabelDrafts((previous) => ({ ...previous, [task.id]: { ...draft, name: event.target.value } }))}
                  />
                  <input
                    value={draft.color}
                    onChange={(event) => setLabelDrafts((previous) => ({ ...previous, [task.id]: { ...draft, color: event.target.value } }))}
                    type="color"
                    aria-label="Label color"
                  />
                  <input
                    value={draft.description}
                    onChange={(event) => setLabelDrafts((previous) => ({ ...previous, [task.id]: { ...draft, description: event.target.value } }))}
                  />
                  <button type="button" onClick={() => void handleAddLabel(task)} disabled={busy}>
                    Add Label
                  </button>
                </div>
              </article>
            );
          })}
        </div>

        {isTaskComposerOpen ? (
          <form
            ref={taskComposerRef}
            id="mc-task-composer"
            className="mc-task-create-form"
            aria-label="Add annotation task"
            onSubmit={handleCreateTask}
          >
            <div className="mc-task-create-copy">
              <h3>Add annotation task</h3>
              <p>Choose a task type and optionally customize its display name.</p>
            </div>
            <div className="mc-task-create-fields">
              <Selector
                label="Task type"
                value={taskDraft.annotationType}
                onChange={(value) =>
                  setTaskDraft((previous) => ({
                    ...previous,
                    annotationType: value as AnnotationType,
                  }))
                }
                placement="below"
                isDisabled={availableTaskTypes.length === 0}
                options={availableTaskTypes.map((option) => ({
                  value: option.value,
                  label: option.label,
                }))}
                width="100%"
              />
              <TextInput
                label="Display name"
                value={taskDraft.displayName}
                onChange={(value) =>
                  setTaskDraft((previous) => ({ ...previous, displayName: value }))
                }
                placeholder={getTaskDisplayName(taskDraft.annotationType)}
                isDisabled={availableTaskTypes.length === 0}
                width="100%"
              />
            </div>
            <div className="mc-task-create-actions">
              <Button
                label="Cancel"
                type="button"
                variant="ghost"
                onClick={() => {
                  setTaskDraft((previous) => ({ ...previous, displayName: "" }));
                  setIsTaskComposerOpen(false);
                }}
                isDisabled={busy}
              />
              <Button
                label="Add task"
                type="submit"
                variant="primary"
                isDisabled={busy || availableTaskTypes.length === 0}
                isLoading={busy}
              />
            </div>
          </form>
        ) : null}
      </section>
    );
  }

  function renderAssignments(): React.ReactElement {
    return (
      <section className="mc-panel">
        <div className="mc-panel-head">
          <h2>Assignments</h2>
          <span className="mc-badge neutral">{assignments.length}</span>
        </div>
        <form className="mc-inline-form assignment" onSubmit={handleCreateAssignment}>
          <label>
            <span>Task</span>
            <select
              value={assignmentDraft.taskId || String(enabledTasks[0]?.id ?? "")}
              onChange={(event) =>
                setAssignmentDraft((previous) => ({
                  ...previous,
                  taskId: event.target.value,
                  targetVersionId: "",
                }))
              }
              disabled={enabledTasks.length === 0}
            >
              {enabledTasks.map((task) => (
                <option key={task.id} value={task.id}>{task.display_name}</option>
              ))}
            </select>
          </label>
          {selectedAssignmentTask?.annotation_type === "evidence_block" ? (
            <label>
              <span>Evidence target</span>
              <select
                value={assignmentDraft.targetVersionId || String(assignmentEvidenceVersions[0]?.version.id ?? "")}
                onChange={(event) =>
                  setAssignmentDraft((previous) => ({ ...previous, targetVersionId: event.target.value }))
                }
                disabled={assignmentEvidenceVersions.length === 0}
              >
                {assignmentEvidenceVersions.length === 0 ? <option value="">No active targets</option> : null}
                {assignmentEvidenceVersions.map(({ target, version }) => (
                  <option key={version.id} value={version.id}>
                    {target.name} (v{version.version_number})
                  </option>
                ))}
              </select>
            </label>
          ) : null}
          <label>
            <span>Document</span>
            <select
              value={assignmentDraft.documentId || String(documents[0]?.id ?? "")}
              onChange={(event) => setAssignmentDraft((previous) => ({ ...previous, documentId: event.target.value }))}
              disabled={documents.length === 0}
            >
              {documents.map((documentItem) => (
                <option key={documentItem.id} value={documentItem.id}>{documentTitle(documentItem)}</option>
              ))}
            </select>
          </label>
          <label>
            <span>Annotator</span>
            <select
              value={assignmentDraft.assigneeUserId || String(assignableMembers[0]?.user_id ?? "")}
              onChange={(event) =>
                setAssignmentDraft((previous) => ({
                  ...previous,
                  assigneeUserId: event.target.value,
                }))
              }
              disabled={assignableMembers.length === 0}
            >
              {assignableMembers.map((member) => (
                <option key={member.user_id} value={member.user_id}>
                  {memberLabel(member, member.username)}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Status</span>
            <select
              value={assignmentDraft.status}
              onChange={(event) => setAssignmentDraft((previous) => ({ ...previous, status: event.target.value as TaskAssignmentStatus }))}
            >
              {ASSIGNMENT_STATUS_OPTIONS.map((status) => (
                <option key={status} value={status} disabled={status === "withdrawn"}>
                  {formatStatus(status)}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            disabled={
              busy ||
              enabledTasks.length === 0 ||
              documents.length === 0 ||
              assignableMembers.length === 0
            }
          >
            Assign
          </button>
        </form>

        <div className="mc-table-scroll">
          <table className="mc-table">
            <thead>
              <tr>
                <th>Task</th>
                <th>Document</th>
                <th>Target</th>
                <th>Annotator</th>
                <th>Status</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {assignments.map((assignment) => (
                <tr key={assignment.id}>
                  <td>{tasksById.get(assignment.task_id)?.display_name ?? `Task ${assignment.task_id}`}</td>
                  <td>{documentTitle(documentsById.get(assignment.document_id) ?? { id: assignment.document_id, project_id: assignment.project_id, external_id: null, title: null, text: "", source: null, metadata_: {}, sentences: [], active_structure_version_id: null })}</td>
                  <td>
                    {assignment.target_version_id
                      ? (() => {
                          const entry = evidenceVersionById.get(assignment.target_version_id);
                          return entry
                            ? `${entry.target.name} (v${entry.version.version_number})`
                            : `Target version ${assignment.target_version_id}`;
                        })()
                      : "Document"}
                  </td>
                  <td>
                    {memberLabel(
                      workspaceMembersByUserId.get(assignment.assignee_user_id),
                      assignment.annotator_id,
                    )}
                  </td>
                  <td>
                    <select
                      value={assignment.status}
                      onChange={(event) => void handleUpdateAssignmentStatus(assignment, event.target.value as TaskAssignmentStatus)}
                      disabled={busy || assignment.status === "withdrawn"}
                    >
                      {ASSIGNMENT_STATUS_OPTIONS.map((status) => (
                        <option key={status} value={status} disabled={status === "withdrawn"}>
                          {formatStatus(status)}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="mc-table-action">
                    <button
                      type="button"
                      onClick={() =>
                        setPendingConfirmation({
                          title: "Delete assignment?",
                          description:
                            "Delete this assignment? The annotator will lose access to this task. Existing annotation data is not deleted.",
                          confirmLabel: "Delete assignment",
                          action: () => handleDeleteAssignment(assignment),
                        })
                      }
                      disabled={busy || assignment.status === "withdrawn"}
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {assignments.length === 0 ? <EmptyState title="No assignments" detail="Assign an enabled task to a document and annotator." /> : null}
      </section>
    );
  }

  function renderProgress(): React.ReactElement {
    return (
      <div className="mc-page">
        <div className="mc-pagehead">
          <div>
            <h1>Progress</h1>
            <p>
              {isPersonalWorkspace
                ? "Document completion and task status."
                : "Project-wide completion, assignment status, and annotator workload."}
            </p>
          </div>
        </div>
        <div className="mc-stats">
          <article className="mc-stat">
            <span>Completion</span>
            <strong>{donePct}%</strong>
          </article>
          <article className="mc-stat">
            <span>Done</span>
            <strong>{counts.done}</strong>
          </article>
          <article className="mc-stat">
            <span>{isPersonalWorkspace ? "In progress" : "In review"}</span>
            <strong>{isPersonalWorkspace ? counts.partial : counts.review}</strong>
          </article>
          <article className="mc-stat">
            <span>{isPersonalWorkspace ? "Documents" : "Blocked"}</span>
            <strong>{isPersonalWorkspace ? documents.length : counts.blocked}</strong>
          </article>
        </div>
        <section className="mc-panel">
          <div className="mc-panel-head">
            <h2>Overall Status</h2>
          </div>
          <StatusMeter
            counts={counts}
            label={isPersonalWorkspace ? "Document status meter" : "Assignment status meter"}
          />
          <div className="mc-legend">
            {(["done", "partial", "review", "todo", "blocked"] as QueueStatus[]).map((status) => (
              <span key={status}>
                <i className={status} />
                {formatStatus(status)} {counts[status]}
              </span>
            ))}
          </div>
        </section>
        <div className="mc-grid">
          {!isPersonalWorkspace ? (
            <section className="mc-panel">
              <div className="mc-panel-head">
                <h2>By Annotator</h2>
              </div>
              <table className="mc-table">
                <thead>
                  <tr>
                    <th>Annotator</th>
                    <th>Total</th>
                    <th>In progress</th>
                    <th>Review</th>
                    <th>Done</th>
                  </tr>
                </thead>
                <tbody>
                  {annotatorRows.map((row) => (
                    <tr key={row.assignee_user_id ?? row.annotator_id}>
                      <td>
                        <span className="mc-avatar-cell">
                          <Avatar id={row.annotator_id} />
                          {memberLabel(
                            row.assignee_user_id !== null
                              ? workspaceMembersByUserId.get(row.assignee_user_id)
                              : undefined,
                            row.annotator_id,
                          )}
                        </span>
                      </td>
                      <td>{row.total}</td>
                      <td>{row.by_status.in_progress ?? 0}</td>
                      <td>{(row.by_status.adjudication_ready ?? 0) + (row.by_status.adjudicated ?? 0)}</td>
                      <td>{(row.by_status.submitted ?? 0) + (row.by_status.completed ?? 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {annotatorRows.length === 0 ? (
                <EmptyState title="No annotator progress" detail="Progress appears after assignments are created." />
              ) : null}
            </section>
          ) : (
            <section className="mc-panel">
              <div className="mc-panel-head">
                <h2>By Document</h2>
              </div>
              <div className="mc-progress-list">
                {documents.map((documentItem) => (
                  <div key={documentItem.id}>
                    <span>{documentTitle(documentItem)}</span>
                    <b>{formatStatus(documentQueueStatus(documentItem))}</b>
                  </div>
                ))}
                {documents.length === 0 ? <p className="mc-muted">No documents yet.</p> : null}
              </div>
            </section>
          )}
          <section className="mc-panel">
            <div className="mc-panel-head">
              <h2>By Task</h2>
            </div>
            <div className="mc-progress-list">
              {(progress?.by_task ?? []).map((row) => (
                <div key={row.task_id}>
                  <span>{row.display_name}</span>
                  <b>{row.total}</b>
                </div>
              ))}
            </div>
          </section>
        </div>
      </div>
    );
  }

  function evidenceAdjudicationStructureVersionId(
    documentId: number,
    targetVersionId: number,
  ): number | null {
    const pinned = assignments.find(
      (assignment) =>
        assignment.document_id === documentId &&
        assignment.target_version_id === targetVersionId &&
        assignment.structure_version_id !== null &&
        assignment.structure_version_id !== undefined,
    );
    return (
      pinned?.structure_version_id ??
      documentsById.get(documentId)?.active_structure_version_id ??
      null
    );
  }

  async function loadEvidenceIaa(): Promise<void> {
    if (selectedProjectId === null) {
      return;
    }
    const documentId = Number(evidenceAdjudicationDraft.documentId);
    const targetVersionId = Number(evidenceAdjudicationDraft.targetVersionId);
    const guidelineVersionId = Number(evidenceAdjudicationDraft.guidelineVersionId);
    const structureVersionId = evidenceAdjudicationStructureVersionId(
      documentId,
      targetVersionId,
    );
    if (
      !Number.isInteger(documentId) ||
      !Number.isInteger(targetVersionId) ||
      !Number.isInteger(guidelineVersionId) ||
      structureVersionId === null
    ) {
      setError("Choose a document, target, and pinned structure before computing IAA.");
      return;
    }
    setEvidenceIaaLoading(true);
    setError(null);
    try {
      setEvidenceIaa(
        await getProjectIaa(selectedProjectId, {
          annotationType: "evidence_block",
          documentId,
          targetVersionId,
          structureVersionId,
          guidelineVersionId,
        }),
      );
    } catch (caught) {
      setEvidenceIaa(null);
      setError(caught instanceof Error ? caught.message : "Unable to compute evidence agreement.");
    } finally {
      setEvidenceIaaLoading(false);
    }
  }

  async function loadEvidenceComparison(): Promise<void> {
    if (selectedProjectId === null) {
      return;
    }
    const documentId = Number(evidenceAdjudicationDraft.documentId);
    const targetVersionId = Number(evidenceAdjudicationDraft.targetVersionId);
    const guidelineVersionId = Number(evidenceAdjudicationDraft.guidelineVersionId);
    const structureVersionId = evidenceAdjudicationStructureVersionId(
      documentId,
      targetVersionId,
    );
    if (
      !Number.isInteger(documentId) ||
      !Number.isInteger(targetVersionId) ||
      !Number.isInteger(guidelineVersionId)
    ) {
      setError("Choose an evidence document, target version, and pinned guideline.");
      return;
    }
    if (structureVersionId === null) {
      setError("This document has no active or assignment-pinned structure version.");
      return;
    }

    setEvidenceAdjudicationLoading(true);
    setError(null);
    try {
      const comparison = await getEvidenceAdjudication(
        selectedProjectId,
        documentId,
        targetVersionId,
        structureVersionId,
        guidelineVersionId,
      );
      setEvidenceComparison(comparison);
      setEvidenceAdjudicationDraft((current) => ({
        ...current,
        selectedAnnotationIds: current.selectedAnnotationIds.filter((annotationId) =>
          comparison.blocks.some(
            (block) => block.annotation_id === annotationId && block.status !== "gold",
          ),
        ),
      }));
    } catch (caught) {
      setEvidenceComparison(null);
      setError(caught instanceof Error ? caught.message : "Unable to load evidence comparison.");
    } finally {
      setEvidenceAdjudicationLoading(false);
    }
  }

  function toggleAdjudicationSource(annotationId: number): void {
    setEvidenceAdjudicationDraft((current) => ({
      ...current,
      selectedAnnotationIds: current.selectedAnnotationIds.includes(annotationId)
        ? current.selectedAnnotationIds.filter((id) => id !== annotationId)
        : [...current.selectedAnnotationIds, annotationId],
    }));
  }

  async function createEvidenceGold(): Promise<void> {
    if (selectedProjectId === null || evidenceComparison === null) {
      setError("Load a comparison before adjudicating.");
      return;
    }
    const strategy: EvidenceAdjudicationStrategy = isPersonalWorkspace
      ? "custom"
      : evidenceAdjudicationDraft.strategy;
    const sourceIds = evidenceAdjudicationDraft.selectedAnnotationIds;
    if (!isPersonalWorkspace) {
      const sourceAnnotators = new Set(
        sourceIds
          .map(
            (sourceId) =>
              evidenceComparison.blocks.find((block) => block.annotation_id === sourceId)
                ?.annotator_user_id,
          )
          .filter((userId): userId is number => userId !== null && userId !== undefined),
      );
      if (sourceAnnotators.size < 2) {
        setError("Team adjudication requires reviewed sources from two distinct annotators.");
        return;
      }
    }
    if (strategy !== "custom" && sourceIds.length === 0) {
      setError("Select at least one reviewed source block.");
      return;
    }
    if (["b", "union", "intersection"].includes(strategy) && sourceIds.length < 2) {
      setError(`${strategy === "b" ? "Strategy B" : formatStatus(strategy)} requires two source blocks.`);
      return;
    }
    const startSentenceId = Number(evidenceAdjudicationDraft.customStartSentenceId);
    const endSentenceId = Number(evidenceAdjudicationDraft.customEndSentenceId);
    if (
      strategy === "custom" &&
      (!Number.isInteger(startSentenceId) || !Number.isInteger(endSentenceId))
    ) {
      setError("Custom adjudication requires valid start and end sentence IDs.");
      return;
    }

    setEvidenceAdjudicationLoading(true);
    setError(null);
    try {
      const created = await adjudicateEvidence(
        selectedProjectId,
        evidenceComparison.document_id,
        {
          target_version_id: evidenceComparison.target_version_id,
          structure_version_id: evidenceComparison.structure_version_id,
          guideline_version_id: evidenceComparison.guideline_version_id,
          strategy,
          source_annotation_ids: sourceIds,
          start_sentence_id: strategy === "custom" ? startSentenceId : null,
          end_sentence_id: strategy === "custom" ? endSentenceId : null,
          note: toNullableText(evidenceAdjudicationDraft.note),
          solo_gold: isPersonalWorkspace,
        },
      );
      showToast(`Gold evidence block #${created.id} created`);
      setEvidenceAdjudicationDraft((current) => ({
        ...current,
        selectedAnnotationIds: [],
        note: "",
      }));
      await loadEvidenceComparison();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Evidence adjudication failed.");
    } finally {
      setEvidenceAdjudicationLoading(false);
    }
  }

  function renderLoop(): React.ReactElement {
    return (
      <div className="mc-page">
        <div className="mc-pagehead">
          <div>
            <h1>{isPersonalWorkspace ? "Gold review" : "Learning Loop"}</h1>
            <p>
              {isPersonalWorkspace
                ? "Explicitly promote your reviewed evidence into a locked gold record."
                : "Active learning, adjudication, and co-learning workspace."}
            </p>
          </div>
          {!isPersonalWorkspace ? (
            <Button
              label="Run cycle"
              variant="primary"
              onClick={() => showToast("Active-learning API scaffold is not connected yet")}
            />
          ) : null}
        </div>
        {!isPersonalWorkspace ? (
          <SegmentedControl
            className="mc-phase-ribbon"
            label="Learning loop phase"
            value={loopPhase}
            onChange={(value) => setLoopPhase(value as LoopPhase)}
            layout="fill"
            size="lg"
          >
            {LOOP_PHASES.map((phase) => (
              <SegmentedControlItem
                key={phase.id}
                value={phase.id}
                label={phase.label}
              />
            ))}
          </SegmentedControl>
        ) : null}
        {!isPersonalWorkspace && loopPhase === "select" ? (
          <section className="mc-panel">
            <div className="mc-panel-head"><h2>Active Learning Selection</h2></div>
            <EmptyState title="No active-learning cycle yet" detail="The product scaffold will connect ranked pools, strategies, and queue promotion when the backend module is added." />
          </section>
        ) : null}
        {isPersonalWorkspace || loopPhase === "adjudicate" ? (
          <section className="mc-panel">
            <div className="mc-panel-head">
              <h2>{isPersonalWorkspace ? "Completed evidence" : "Adjudication Queue"}</h2>
            </div>
            <div className="mc-stats compact">
              <article className="mc-stat"><span>Open review</span><strong>{counts.review}</strong></article>
              <article className="mc-stat"><span>Done</span><strong>{counts.done}</strong></article>
              <article className="mc-stat">
                <span>{isPersonalWorkspace ? "Total tasks" : "Total assignments"}</span>
                <strong>{assignments.length}</strong>
              </article>
            </div>
            <div className="mc-adjudication-controls">
              <div className="mc-inline-form assignment">
                <label>
                  <span>Document</span>
                  <select
                    value={evidenceAdjudicationDraft.documentId}
                    onChange={(event) => {
                      setEvidenceAdjudicationDraft((current) => ({
                        ...current,
                        documentId: event.target.value,
                        selectedAnnotationIds: [],
                      }));
                      setEvidenceComparison(null);
                      setEvidenceIaa(null);
                    }}
                    disabled={documents.length === 0}
                  >
                    {documents.map((item) => (
                      <option key={item.id} value={item.id}>{documentTitle(item)}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Target version</span>
                  <select
                    value={evidenceAdjudicationDraft.targetVersionId}
                    onChange={(event) => {
                      setEvidenceAdjudicationDraft((current) => ({
                        ...current,
                        targetVersionId: event.target.value,
                        selectedAnnotationIds: [],
                      }));
                      setEvidenceComparison(null);
                      setEvidenceIaa(null);
                    }}
                    disabled={adjudicationTargetVersions.length === 0}
                  >
                    {adjudicationTargetVersions.map(({ target, version }) => (
                      <option key={version.id} value={version.id}>
                        {target.name} (v{version.version_number})
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Guideline version</span>
                  <select
                    value={evidenceAdjudicationDraft.guidelineVersionId}
                    onChange={(event) => {
                      setEvidenceAdjudicationDraft((current) => ({
                        ...current,
                        guidelineVersionId: event.target.value,
                        selectedAnnotationIds: [],
                      }));
                      setEvidenceComparison(null);
                      setEvidenceIaa(null);
                    }}
                    disabled={adjudicationGuidelineVersions.length === 0}
                  >
                    {adjudicationGuidelineVersions.map((guideline) => (
                      <option key={guideline.id} value={guideline.id}>
                        {guideline.version_label}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  onClick={() => void loadEvidenceComparison()}
                  disabled={
                    evidenceAdjudicationLoading ||
                    documents.length === 0 ||
                    adjudicationTargetVersions.length === 0 ||
                    adjudicationGuidelineVersions.length === 0
                  }
                >
                  {evidenceAdjudicationLoading
                    ? "Loading…"
                    : isPersonalWorkspace
                      ? "Load review"
                      : "Load comparison"}
                </button>
              </div>

              {!isPersonalWorkspace ? (
                <div className="mc-iaa-panel">
                  <div className="mc-panel-head">
                    <h3>Reviewed-region agreement</h3>
                    <button
                      type="button"
                      onClick={() => void loadEvidenceIaa()}
                      disabled={evidenceIaaLoading}
                    >
                      {evidenceIaaLoading ? "Computing…" : "Compute IAA"}
                    </button>
                  </div>
                  {evidenceIaa ? (
                    evidenceIaa.status === "ok" && evidenceIaa.evidence_metrics ? (
                      <div className="mc-table-scroll">
                        <table className="mc-table">
                          <thead>
                            <tr>
                              <th scope="col">Annotators</th>
                              <th scope="col">Reviewed</th>
                              <th scope="col">Sentence F1</th>
                              <th scope="col">Exact F1</th>
                              <th scope="col">IoU F1 @ .50</th>
                              <th scope="col">Coverage</th>
                              <th scope="col">Overreach</th>
                            </tr>
                          </thead>
                          <tbody>
                            {evidenceIaa.evidence_metrics.pairs.map((pair) => (
                              <tr key={`${pair.left_annotator_id}:${pair.right_annotator_id}`}>
                                <td>{pair.left_annotator_id} / {pair.right_annotator_id}</td>
                                <td>{pair.reviewed_sentence_count} sentences</td>
                                <td>{(pair.sentence_f1 * 100).toFixed(1)}%</td>
                                <td>{(pair.exact_f1 * 100).toFixed(1)}%</td>
                                <td>{((pair.iou_f1["0.50"] ?? 0) * 100).toFixed(1)}%</td>
                                <td>{(pair.coverage * 100).toFixed(1)}%</td>
                                <td>{(pair.overreach * 100).toFixed(1)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : (
                      <p className="mc-muted">
                        Agreement is unavailable: {formatStatus(evidenceIaa.status)}.
                      </p>
                    )
                  ) : (
                    <p className="mc-muted">
                      IAA uses only the intersection of reviewed coverage for this exact target,
                      structure, and guideline scope.
                    </p>
                  )}
                </div>
              ) : null}

              {evidenceComparison === null ? (
                <EmptyState
                  title={
                    isPersonalWorkspace
                      ? "Choose completed evidence to review"
                      : "Choose evidence work to compare"
                  }
                  detail={
                    isPersonalWorkspace
                      ? "Only your reviewed blocks for the selected document, target, structure, and guideline are loaded."
                      : "Only reviewed blocks pinned to the selected document, target, structure, and guideline versions are loaded."
                  }
                />
              ) : (
                <>
                  <div className="mc-table-scroll">
                    <table className="mc-table">
                      <thead>
                        <tr>
                          <th scope="col">Use</th>
                          <th scope="col">{isPersonalWorkspace ? "Source" : "Annotator"}</th>
                          <th scope="col">Sentence range</th>
                          <th scope="col">Status</th>
                          <th scope="col">Labels / note</th>
                        </tr>
                      </thead>
                      <tbody>
                        {evidenceComparison.blocks.map((block) => {
                          const isGold = block.status === "gold";
                          const sourcePosition =
                            evidenceAdjudicationDraft.selectedAnnotationIds.indexOf(
                              block.annotation_id,
                            );
                          return (
                            <tr key={block.annotation_id}>
                              <td>
                                <label className="mc-adjudication-source">
                                  <input
                                    type="checkbox"
                                    checked={sourcePosition >= 0}
                                    disabled={isGold || evidenceAdjudicationLoading}
                                    onChange={() => toggleAdjudicationSource(block.annotation_id)}
                                    aria-label={`Use evidence block ${block.annotation_id}`}
                                  />
                                  {sourcePosition >= 0 ? String.fromCharCode(65 + sourcePosition) : "—"}
                                </label>
                              </td>
                              <td>{block.annotator_id ?? `User #${block.annotator_user_id ?? "?"}`}</td>
                              <td>
                                {block.start_sentence_ordinal + 1}–{block.end_sentence_ordinal + 1}
                              </td>
                              <td>
                                <span className="mc-badge neutral">
                                  {formatStatus(block.status)}
                                </span>
                              </td>
                              <td>
                                {[...block.labels, block.note].filter(Boolean).join(" · ") || "—"}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  {evidenceComparison.blocks.length === 0 ? (
                    <p className="mc-muted">No evidence blocks exist in this exact scope.</p>
                  ) : null}

                  <div className="mc-inline-form assignment mc-adjudication-decision">
                    <label>
                      <span>Gold boundary</span>
                      <select
                        value={isPersonalWorkspace ? "custom" : evidenceAdjudicationDraft.strategy}
                        onChange={(event) =>
                          setEvidenceAdjudicationDraft((current) => ({
                            ...current,
                            strategy: event.target.value as EvidenceAdjudicationStrategy,
                          }))
                        }
                      >
                        {isPersonalWorkspace ? (
                          <option value="custom">Reviewed custom boundary</option>
                        ) : (
                          <>
                            <option value="a">Use source A</option>
                            <option value="b">Use source B</option>
                            <option value="union">Union</option>
                            <option value="intersection">Intersection</option>
                            <option value="custom">Custom sentence IDs</option>
                          </>
                        )}
                      </select>
                    </label>
                    {isPersonalWorkspace || evidenceAdjudicationDraft.strategy === "custom" ? (
                      <>
                        <label>
                          <span>Start sentence ID</span>
                          <input
                            type="number"
                            min="1"
                            value={evidenceAdjudicationDraft.customStartSentenceId}
                            onChange={(event) =>
                              setEvidenceAdjudicationDraft((current) => ({
                                ...current,
                                customStartSentenceId: event.target.value,
                              }))
                            }
                          />
                        </label>
                        <label>
                          <span>End sentence ID</span>
                          <input
                            type="number"
                            min="1"
                            value={evidenceAdjudicationDraft.customEndSentenceId}
                            onChange={(event) =>
                              setEvidenceAdjudicationDraft((current) => ({
                                ...current,
                                customEndSentenceId: event.target.value,
                              }))
                            }
                          />
                        </label>
                      </>
                    ) : null}
                    <label>
                      <span>Gold note</span>
                      <input
                        value={evidenceAdjudicationDraft.note}
                        onChange={(event) =>
                          setEvidenceAdjudicationDraft((current) => ({
                            ...current,
                            note: event.target.value,
                          }))
                        }
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => void createEvidenceGold()}
                      disabled={evidenceAdjudicationLoading}
                    >
                      {evidenceAdjudicationLoading ? "Saving…" : "Create gold block"}
                    </button>
                  </div>
                  <p className="mc-muted">
                    {isPersonalWorkspace
                      ? "Solo gold is allowed only inside your reviewed coverage and is created only by this explicit promotion."
                      : "Sources must be submitted by two distinct annotators and inside matching reviewed coverage. Gold is created only by this explicit adjudication action."}
                  </p>
                </>
              )}
            </div>
          </section>
        ) : null}
        {!isPersonalWorkspace && loopPhase === "reflect" ? (
          <section className="mc-panel">
            <div className="mc-panel-head"><h2>Reflect</h2></div>
            <EmptyState title="No co-learning review queue yet" detail="Existing error-guideline records can be surfaced here once the review queue contract is added." />
          </section>
        ) : null}
      </div>
    );
  }

  function renderModels(): React.ReactElement {
    return (
      <div className="mc-page">
        <div className="mc-pagehead">
          <div>
            <h1>Models & Training</h1>
            <p>Project-scoped model registry and training job tracking.</p>
          </div>
          <Button
            label="Open training workspace"
            variant="primary"
            onClick={() => onNavigate("/training")}
          />
        </div>
        <section className="mc-panel">
          <div className="mc-panel-head"><h2>Shared training workspace</h2></div>
          <EmptyState
            title="Results, comparisons, and model files are in Training"
            detail="Managers and model trainers use the same project-scoped workspace. Manager-only champion, infrastructure, and retention controls remain role protected."
          />
        </section>
      </div>
    );
  }

  function renderGuidelines(): React.ReactElement {
    return (
      <div className="mc-page">
        <div className="mc-pagehead">
          <div>
            <h1>Guidelines</h1>
            <p>Versioned project guidance rendered from Markdown.</p>
          </div>
        </div>
        <div className="mc-two-col">
          <section className="mc-panel guideline-doc">
            <div className="mc-panel-head">
              <h2>{activeGuide?.version_label ?? "No active guideline"}</h2>
              <span className="mc-badge neutral">{activeGuide?.status ?? "empty"}</span>
            </div>
            {activeGuide ? <pre>{activeGuide.markdown}</pre> : <EmptyState title="No guideline yet" detail="Create a guideline version to give annotators project-specific instructions." />}
          </section>
          <aside className="mc-rail">
            <section className="mc-panel">
              <div className="mc-panel-head"><h2>New Version</h2></div>
              <form className="mc-form-stack" onSubmit={handleCreateGuideline}>
                <TextInput
                  label="Version"
                  value={guidelineDraft.versionLabel}
                  onChange={(value) =>
                    setGuidelineDraft((previous) => ({ ...previous, versionLabel: value }))
                  }
                />
                <TextArea
                  label="Markdown"
                  value={guidelineDraft.markdown}
                  onChange={(value) =>
                    setGuidelineDraft((previous) => ({ ...previous, markdown: value }))
                  }
                  rows={8}
                />
                <Button
                  label="Create Version"
                  variant="primary"
                  type="submit"
                  isDisabled={busy || !selectedProject}
                  isLoading={busy}
                />
              </form>
            </section>
            <section className="mc-panel">
              <div className="mc-panel-head"><h2>History</h2></div>
              <div className="mc-progress-list">
                {guidelines.map((guideline) => (
                  <div key={guideline.id}>
                    <span>{guideline.version_label}</span>
                    <b>{guideline.status}</b>
                  </div>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </div>
    );
  }

  function renderExport(): React.ReactElement {
    return (
      <div className="mc-page wide">
        <div className="mc-pagehead">
          <div>
            <h1>Export & Lineage</h1>
            <p>
              {isPersonalWorkspace
                ? "Saved annotation snapshot files per document."
                : "Saved annotation snapshot files per document submission."}
            </p>
          </div>
          <Button label="Refresh" variant="primary" onClick={() => void refreshSubmissions()} />
        </div>
        <section className="mc-panel">
          <div className="mc-panel-head">
            <h2>Submission Files</h2>
            <div className="mc-filters">
              <select
                value={submissionDocumentFilter === "all" ? "all" : String(submissionDocumentFilter)}
                onChange={(event) =>
                  setSubmissionDocumentFilter(
                    event.target.value === "all" ? "all" : Number(event.target.value),
                  )
                }
              >
                <option value="all">All documents</option>
                {documents.map((documentItem) => (
                  <option key={documentItem.id} value={documentItem.id}>
                    {documentTitle(documentItem)}
                  </option>
                ))}
              </select>
              {!isPersonalWorkspace ? (
                <select
                  value={submissionAnnotatorFilter}
                  onChange={(event) => setSubmissionAnnotatorFilter(event.target.value)}
                >
                  <option value="all">All annotators</option>
                  {submissionAnnotators.map((annotator) => (
                    <option key={annotator || "(none)"} value={annotator}>
                      {annotator || "No annotator"}
                    </option>
                  ))}
                </select>
              ) : null}
            </div>
          </div>
          {filteredSubmissions.length === 0 ? (
            <EmptyState
              title="No submission files"
              detail={
                isPersonalWorkspace
                  ? "Files appear here when you submit documents."
                  : "Files appear here when annotators submit documents from the workspace."
              }
            />
          ) : (
            <div className="mc-table-scroll">
              <table className="mc-table">
                <thead>
                  <tr>
                    <th>File</th>
                    <th>Document</th>
                    {!isPersonalWorkspace ? <th>Annotator</th> : null}
                    <th>Kind</th>
                    <th>Annotations</th>
                    <th>Size</th>
                    <th>Created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {filteredSubmissions.map((submission) => {
                    const submissionDocument = documentsById.get(submission.document_id);
                    return (
                      <tr key={submission.id}>
                        <td>
                          <div className="mc-primary-cell">
                            <strong>{submission.file_name}</strong>
                            <span>{submission.checksum_sha256.slice(0, 12)}</span>
                          </div>
                        </td>
                        <td>
                          {submissionDocument
                            ? documentTitle(submissionDocument)
                            : `Document ${submission.document_id}`}
                        </td>
                        {!isPersonalWorkspace ? <td>{submission.annotator_id ?? "None"}</td> : null}
                        <td>{submission.kind === "re_export" ? "Re-export" : "Submission"}</td>
                        <td>{submission.annotation_count}</td>
                        <td>{`${(submission.size_bytes / 1024).toFixed(1)} KB`}</td>
                        <td>{new Date(submission.created_at).toLocaleString()}</td>
                        <td>
                          <div className="mc-row-actions">
                            <button
                              type="button"
                              onClick={() => void saveSubmissionFile(submission)}
                            >
                              Download
                            </button>
                            <button
                              type="button"
                              onClick={() => void reExportSubmission(submission)}
                            >
                              Re-export
                            </button>
                            <button
                              type="button"
                              onClick={() =>
                                setPendingConfirmation({
                                  title: "Delete submission file?",
                                  description: `Delete “${submission.file_name}”? The exported file will be removed, but annotations in the database will remain.`,
                                  confirmLabel: "Delete file",
                                  action: () => removeSubmission(submission),
                                })
                              }
                            >
                              Delete
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    );
  }

  function renderAdminUsers(): React.ReactElement {
    if (activeWorkspaceId === null) {
      return <EmptyState title="No workspace selected" detail="Select a workspace to manage users." />;
    }

    return (
      <div className="mc-two-col">
        <section className="mc-panel">
          <div className="mc-panel-head">
            <h2>Members</h2>
            <button type="button" onClick={() => void refreshWorkspaceUsers()} disabled={busy}>
              Refresh
            </button>
          </div>
          <div className="mc-table-scroll">
            <table className="mc-table">
              <thead>
                <tr>
                  <th>User</th>
                  <th>Email</th>
                  <th>Status</th>
                  <th>Role</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {workspaceMembers.map((member) => (
                  <tr key={member.user_id}>
                    <td>
                      <span className="mc-avatar-cell">
                        <Avatar id={member.username} />
                        <span>
                          {memberLabel(member, member.username)}
                          <small>{member.username}</small>
                        </span>
                      </span>
                    </td>
                    <td>{member.email ?? "None"}</td>
                    <td>{member.is_active ? "Active" : "Inactive"}</td>
                    <td>
                      <select
                        value={member.role}
                        onChange={(event) =>
                          void handleUpdateMemberRole(member, event.target.value as WorkspaceRole)
                        }
                        disabled={busy}
                      >
                        {WORKSPACE_ROLE_OPTIONS.map((option) => (
                          <option key={option} value={option}>
                            {formatStatus(option)}
                          </option>
                        ))}
                      </select>
                    </td>
                    <td className="mc-table-action">
                      <button
                        type="button"
                        onClick={() =>
                          setPendingConfirmation({
                            title: "Remove workspace member?",
                            description: `Remove ${member.username} from this workspace? Their open assignments will be withdrawn and they will lose workspace access.`,
                            confirmLabel: "Remove member",
                            action: () => handleRemoveMember(member),
                          })
                        }
                        disabled={busy}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {workspaceMembers.length === 0 ? (
            <EmptyState title="No members" detail="Members appear here after users join this workspace." />
          ) : null}
        </section>

        <aside className="mc-rail">
          <section className="mc-panel">
            <div className="mc-panel-head">
              <h2>Invite</h2>
            </div>
            <form className="mc-form-stack" onSubmit={handleCreateInvite}>
              <label>
                <span>Role</span>
                <select
                  value={inviteRole}
                  onChange={(event) => setInviteRole(event.target.value as WorkspaceRole)}
                  disabled={busy}
                >
                  {WORKSPACE_ROLE_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {formatStatus(option)}
                    </option>
                  ))}
                </select>
              </label>
              <button className="primary" type="submit" disabled={busy}>
                Create Invite
              </button>
            </form>
            {lastInviteToken ? (
              <div className="status">
                Invite token: <code>{lastInviteToken}</code>
              </div>
            ) : null}
          </section>

          <section className="mc-panel">
            <div className="mc-panel-head">
              <h2>Join Requests</h2>
              <span className="mc-badge neutral">{joinRequests.length}</span>
            </div>
            <div className="mc-progress-list">
              {joinRequests.map((request) => (
                <div key={request.id}>
                  <span>User {request.user_id}</span>
                  <b>{request.message ?? "No message"}</b>
                  <div className="mc-row-actions">
                    <button
                      type="button"
                      onClick={() => void handleDecideJoinRequest(request, true)}
                      disabled={busy}
                    >
                      Approve
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDecideJoinRequest(request, false)}
                      disabled={busy}
                    >
                      Reject
                    </button>
                  </div>
                </div>
              ))}
            </div>
            {joinRequests.length === 0 ? (
              <EmptyState title="No pending requests" detail="Join requests appear here when users apply by code." />
            ) : null}
          </section>
        </aside>
      </div>
    );
  }

  function renderAdmin(): React.ReactElement {
    return (
      <div className="mc-page wide">
        <div className="mc-pagehead">
          <div>
            <h1>Administration</h1>
            <p>Workspace controls for administrators.</p>
          </div>
        </div>
        <nav className="mc-admin-tabs" aria-label="Administration section">
          {ADMIN_SECTIONS.map((section) => (
            <a
              key={section.id}
              href={`/admin/${section.id}`}
              aria-current={adminSection === section.id ? "page" : undefined}
              onClick={(event) => {
                if (!shouldHandleSpaClick(event)) {
                  return;
                }
                event.preventDefault();
                navigateAdmin(section.id);
              }}
            >
              {section.label}
            </a>
          ))}
        </nav>
        {adminSection === "users" ? (
          renderAdminUsers()
        ) : (
          <section className="mc-panel">
            <div className="mc-panel-head"><h2>{ADMIN_SECTIONS.find((section) => section.id === adminSection)?.label}</h2></div>
            {adminSection === "health" ? (
              <div className="mc-health-grid">
                {["FastAPI", "PostgreSQL", "MinIO", "Celery ml", "Celery llm", "vLLM"].map((service, index) => (
                  <article key={service}>
                    <span>{service}</span>
                    <strong>{index < 3 ? "ok" : "not configured"}</strong>
                  </article>
                ))}
              </div>
            ) : (
              <EmptyState title="Admin API not connected" detail="This section is scaffolded until dev JWT, audit, plugins, and settings endpoints are added." />
            )}
          </section>
        )}
      </div>
    );
  }

  function renderProjectBody(): React.ReactElement {
    if (loading) {
      return (
        <main id="main-content" className="mc-body" tabIndex={-1}>
          <div className="status" role="status" aria-live="polite">
            Loading…
          </div>
        </main>
      );
    }

    if (projects.length === 0) {
      return (
        <main id="main-content" className="mc-body" tabIndex={-1}>
          <div className="mc-page">
            <EmptyState
              title={isPersonalWorkspace ? "No personal projects" : "No projects"}
              detail={
                isPersonalWorkspace
                  ? "Create a private project, import PMIDs, and start annotating from My Work."
                  : "Create a project to open the manager console."
              }
            />
            <Button
              className="mc-create-first"
              label={isPersonalWorkspace ? "Create Project" : "New Project"}
              variant="primary"
              icon={<Icon name="plus" />}
              onClick={() => setModal("newProject")}
            />
          </div>
        </main>
      );
    }

    const views: Record<ManagerTab, () => React.ReactElement> = {
      overview: renderOverview,
      documents: renderDocuments,
      tasks: renderTasks,
      progress: renderProgress,
      loop: renderLoop,
      models: renderModels,
      guidelines: renderGuidelines,
      export: renderExport,
    };

    return (
      <main id="main-content" className="mc-body" tabIndex={-1}>
        {views[activeTab]()}
      </main>
    );
  }

  function renderModal(): React.ReactElement | null {
    if (modal !== "newProject") {
      return null;
    }

    return (
      <DialogFrame
        ariaLabel="New project"
        busy={busy}
        error={error}
        backdropClassName="mc-modal-backdrop"
        dialogClassName="mc-modal"
        dialogElement="section"
        initialFocusSelector="form input:not([disabled]), form select:not([disabled]), form textarea:not([disabled])"
        portal={false}
        onDismiss={closeModal}
      >
        <div className="mc-panel-head">
          <h2>{isPersonalWorkspace ? "New Personal Project" : "New Project"}</h2>
          <Button label="Close" size="sm" isDisabled={busy} onClick={closeModal} />
        </div>
        <form className="mc-form-stack" onSubmit={handleCreateProject}>
          <TextInput
            label="Name"
            value={projectForm.name}
            onChange={(value) => setProjectForm((previous) => ({ ...previous, name: value }))}
            placeholder="e.g., Neonatal therapy review…"
            isRequired
          />
          <TextArea
            label="Description"
            value={projectForm.description}
            onChange={(value) =>
              setProjectForm((previous) => ({ ...previous, description: value }))
            }
            rows={3}
          />
          <Selector
            label="Validation"
            value={projectForm.annotationValidationMode}
            onChange={(value) =>
              setProjectForm((previous) => ({
                ...previous,
                annotationValidationMode: value as AnnotationValidationMode,
              }))
            }
            placement="below"
            options={[
              { value: "relaxed", label: "Relaxed" },
              { value: "strict", label: "Strict" },
            ]}
          />
          <fieldset className="mc-fieldset">
            <legend>{isPersonalWorkspace ? "Annotation tasks" : "Starter tasks"}</legend>
            {ANNOTATION_TYPE_OPTIONS.map((option) => (
              <CheckboxInput
                key={option.value}
                label={option.label}
                value={projectForm.starterTaskTypes.includes(option.value)}
                onChange={() => toggleStarterTask(option.value)}
                size="sm"
              />
            ))}
          </fieldset>
          <Button
            label={isPersonalWorkspace ? "Create project" : "Create Project"}
            variant="primary"
            type="submit"
            isDisabled={busy}
            isLoading={busy}
          />
        </form>
      </DialogFrame>
    );
  }

  return (
    <div className="mc">
      {!embedded ? (
        <>
          <a className="skip-link" href="#main-content">
            Skip to main content
          </a>
          {renderTopBar()}
          {renderTabStrip()}
        </>
      ) : null}
      {error ? (
        <Banner
          className="shell-banner"
          status="error"
          title="Console error"
          description={error}
          container="section"
        />
      ) : null}
      {busy && !loading ? (
        <Banner
          className="shell-banner"
          status="info"
          title="Saving…"
          container="section"
        />
      ) : null}
      {appMode === "admin" ? (
        <main id="main-content" className="mc-body" tabIndex={-1}>
          {renderAdmin()}
        </main>
      ) : (
        renderProjectBody()
      )}
      {renderModal()}
      <ConfirmDialog
        open={pendingConfirmation !== null}
        title={pendingConfirmation?.title ?? ""}
        description={pendingConfirmation?.description ?? ""}
        confirmLabel={pendingConfirmation?.confirmLabel ?? "Confirm"}
        busy={confirmationBusy}
        onCancel={() => setPendingConfirmation(null)}
        onConfirm={async () => {
          if (!pendingConfirmation) {
            return;
          }
          setConfirmationBusy(true);
          try {
            await pendingConfirmation.action();
            setPendingConfirmation(null);
          } finally {
            setConfirmationBusy(false);
          }
        }}
      />
      {toast ? (
        <div className="mc-toast" role="status" aria-live="polite" aria-atomic="true">
          <Icon name="check" />
          {toast}
        </div>
      ) : null}
    </div>
  );
}
