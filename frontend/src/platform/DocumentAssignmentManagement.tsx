import { useRef, useState } from "react";

import { createProjectTask, createTaskAssignment, updateProjectTask, updateTaskAssignment } from "@/api/client";
import type { DocumentSummary, EvidenceTarget, GuidelineVersion, ProjectTask, TaskAssignment, TaskAssignmentStatus, WorkspaceMember } from "@/types/api";

import { formatStatus, PlatformStatus } from "./components";

export interface DocumentAssignmentManagementProps {
  projectId: number;
  tasks: ProjectTask[];
  documents: DocumentSummary[];
  members: WorkspaceMember[];
  targets: EvidenceTarget[];
  guidelines: GuidelineVersion[];
  assignments: TaskAssignment[];
  busy?: boolean;
  onChanged: () => Promise<void>;
  onOpenAssignment?: (assignment: TaskAssignment) => void;
}

const TASK_KINDS = [
  { value: "evidence_block", label: "Evidence blocks" },
  { value: "entity", label: "Entity spans" },
  { value: "relation", label: "Relations" },
] as const;
type NewTaskKind = typeof TASK_KINDS[number]["value"];

function documentName(document: DocumentSummary): string {
  return document.title?.trim() || document.external_id || `Document ${document.id}`;
}

function nextAction(status: TaskAssignmentStatus): { label: string; status: TaskAssignmentStatus } | null {
  switch (status) {
    case "assigned":
    case "in_progress": return { label: "Block", status: "blocked" };
    case "blocked": return { label: "Resume", status: "assigned" };
    case "submitted": return { label: "Mark ready for adjudication", status: "adjudication_ready" };
    case "adjudication_ready": return { label: "Mark adjudicated", status: "adjudicated" };
    case "adjudicated": return { label: "Mark completed", status: "completed" };
    default: return null;
  }
}

function useDocumentMutation(busy: boolean, onChanged: () => Promise<void>) {
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [refreshFailed, setRefreshFailed] = useState(false);
  const inFlight = useRef(false);
  async function refresh(): Promise<void> {
    try {
      await onChanged();
      setRefreshFailed(false);
    } catch (caught) {
      setRefreshFailed(true);
      setError(`Saved, but the project could not refresh. ${caught instanceof Error ? caught.message : "Try refreshing the project."}`);
    }
  }
  async function mutate(operation: () => Promise<void>): Promise<void> {
    if (busy || refreshFailed || inFlight.current) return;
    inFlight.current = true;
    setSaving(true);
    setError(null);
    setMessage(null);
    try {
      await operation();
      await refresh();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The change could not be saved. Try again.");
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  }
  async function retryRefresh(): Promise<void> {
    if (busy || inFlight.current) return;
    inFlight.current = true;
    setSaving(true);
    setError(null);
    try {
      await refresh();
    } finally {
      inFlight.current = false;
      setSaving(false);
    }
  }
  return { working: busy || saving || refreshFailed, refreshing: busy || saving, refreshFailed, error, message, setMessage, mutate, retryRefresh };
}

export function DocumentTaskManagement({ projectId, tasks, busy = false, onChanged }: Pick<DocumentAssignmentManagementProps, "projectId" | "tasks" | "busy" | "onChanged">): React.ReactElement {
  const [taskKind, setTaskKind] = useState<NewTaskKind>("evidence_block");
  const [taskName, setTaskName] = useState("");
  const [taskLabels, setTaskLabels] = useState("");
  // Remember confirmed creations if the following refresh fails.
  const [savedTasks, setSavedTasks] = useState<ProjectTask[]>([]);
  const { working, refreshing, refreshFailed, error, message, setMessage, mutate, retryRefresh } = useDocumentMutation(busy, onChanged);
  const projectTasks = [...tasks, ...savedTasks.filter((saved) => !tasks.some((task) => task.id === saved.id))]
    .filter((task) => task.project_id === projectId);
  const availableKinds = TASK_KINDS.filter((kind) => !projectTasks.some((task) => task.annotation_type === kind.value));
  const newTaskKind = availableKinds.find((kind) => kind.value === taskKind)?.value ?? availableKinds[0]?.value ?? "evidence_block";

  async function addTask(): Promise<void> {
    if (!availableKinds.some((kind) => kind.value === newTaskKind)) throw new Error("This document task type is already configured.");
    if (!taskName.trim()) throw new Error("Enter a document task name.");
    if (taskName.trim().length > 120) throw new Error("Document task names must be 120 characters or fewer.");
    const labels = [...new Set(taskLabels.split(",").map((label) => label.trim()).filter(Boolean))];
    if (newTaskKind !== "evidence_block" && !labels.length) throw new Error("Enter at least one task label.");
    const task = await createProjectTask(projectId, {
      annotation_type: newTaskKind, display_name: taskName.trim(), enabled: true,
      labels: newTaskKind === "evidence_block" ? [] : labels.map((name) => ({ name, color: "#2563eb", description: null })),
      settings: {},
    });
    setSavedTasks((current) => [...current, task]);
    setTaskName("");
    setTaskLabels("");
    setMessage(`Document task created: ${task.display_name}.`);
  }

  return (
    <section className="platform-panel" aria-label="Document annotation tasks">
      <h3>Document annotation tasks</h3>
      <p>Enable the document tools your annotators will use.</p>
      {error ? <p role="alert" className="platform-dialog-error">{error}</p> : null}
      {refreshFailed ? <button type="button" disabled={refreshing} onClick={() => void retryRefresh()}>Retry refresh</button> : null}
      {message ? <p role="status">{message}</p> : null}
      {projectTasks.length ? <ul>{projectTasks.map((task) => <li key={task.id}>
        {task.display_name} · {formatStatus(task.annotation_type)} · {task.enabled ? "Enabled" : "Disabled"}{" "}
        <button type="button" disabled={working} aria-label={`${task.enabled ? "Disable" : "Enable"} task ${task.id}`} onClick={() => {
          void mutate(async () => {
            await updateProjectTask(projectId, task.id, { enabled: !task.enabled });
            setMessage(`${task.display_name}: ${task.enabled ? "disabled" : "enabled"}.`);
          });
        }}>{task.enabled ? "Disable" : "Enable"}</button>
      </li>)}</ul> : <p>No document annotation tasks have been configured.</p>}
      {availableKinds.length ? (
        <form className="platform-dialog-form" aria-label="Create document annotation task" onSubmit={(event) => { event.preventDefault(); void mutate(addTask); }}>
          <fieldset disabled={working}>
            <legend>Create document task</legend>
            <label><span>Annotation type</span><select value={newTaskKind} onChange={(event) => setTaskKind(event.target.value as NewTaskKind)}>{availableKinds.map((kind) => <option key={kind.value} value={kind.value}>{kind.label}</option>)}</select></label>
            <label><span>Task name</span><input required maxLength={120} value={taskName} onChange={(event) => setTaskName(event.target.value)} /></label>
            {newTaskKind !== "evidence_block" ? <label><span>Task labels</span><input required pattern={".*[^\\s,].*"} value={taskLabels} placeholder="Drug, Disease" onChange={(event) => setTaskLabels(event.target.value)} /></label> : null}
            <button type="submit" disabled={working}>Create document task</button>
          </fieldset>
        </form>
      ) : <p>Evidence, entity, and relation tasks are already configured.</p>}
    </section>
  );
}

export default function DocumentAssignmentManagement({
  projectId, tasks, documents, members, targets, guidelines, assignments, busy = false,
  onChanged, onOpenAssignment,
}: DocumentAssignmentManagementProps): React.ReactElement {
  const [taskId, setTaskId] = useState(0);
  const [documentId, setDocumentId] = useState(0);
  const [assigneeId, setAssigneeId] = useState(0);
  const [targetVersionId, setTargetVersionId] = useState(0);
  const [notes, setNotes] = useState("");
  const [savedAssignments, setSavedAssignments] = useState<TaskAssignment[]>([]);
  const { working, refreshing, refreshFailed, error, message, setMessage, mutate, retryRefresh } = useDocumentMutation(busy, onChanged);
  const projectTasks = tasks.filter((task) => task.project_id === projectId);
  const enabledTasks = projectTasks.filter((task) => task.enabled);
  const projectDocuments = documents.filter((document) => document.project_id === projectId);
  const activeMembers = members.filter((member) => member.is_active);
  const selectedTask = enabledTasks.find((task) => task.id === taskId);
  const selectedDocument = projectDocuments.find((document) => document.id === documentId);
  const evidenceTask = selectedTask?.annotation_type === "evidence_block";
  const availableTargets = targets.filter((target) => target.project_id === projectId &&
    target.task_id === selectedTask?.id && target.is_active && target.active_version_id !== null &&
    target.versions.some((version) => version.id === target.active_version_id));
  const selectedTarget = availableTargets.find((target) => target.active_version_id === targetVersionId);
  const activeGuideline = guidelines.filter((guideline) => guideline.project_id === projectId && guideline.status === "active")
    .sort((left, right) => right.id - left.id)[0];
  const projectAssignments = [...assignments, ...savedAssignments.filter((saved) => !assignments.some((assignment) => assignment.id === saved.id))]
    .filter((assignment) => assignment.project_id === projectId);

  async function assignDocument(): Promise<void> {
    if (!selectedTask) throw new Error("Select an enabled document task.");
    if (!selectedDocument) throw new Error("Select a document from this project.");
    if (!activeMembers.some((member) => member.user_id === assigneeId)) throw new Error("Select an active workspace annotator.");
    if (evidenceTask && !selectedTarget) throw new Error("Select an active evidence target version for this task.");
    if (evidenceTask && !selectedDocument.active_structure_version_id) throw new Error("Import or process this document to create its active structure before assigning evidence.");
    if (evidenceTask && !activeGuideline) throw new Error("Create an active evidence guideline above before assigning documents.");
    const targetPin = evidenceTask ? selectedTarget!.active_version_id : null;
    const structurePin = selectedDocument.active_structure_version_id;
    const guidelinePin = activeGuideline?.id ?? null;
    const existing = projectAssignments.find((assignment) => assignment.task_id === selectedTask.id &&
      assignment.document_id === selectedDocument.id && assignment.assignee_user_id === assigneeId &&
      assignment.target_version_id === targetPin && assignment.structure_version_id === structurePin &&
      assignment.guideline_version_id === guidelinePin);
    if (existing) throw new Error(`Assignment ${existing.id} already exists for this document, task, annotator, and pinned versions.`);
    const assignment = await createTaskAssignment(projectId, {
      task_id: selectedTask.id, document_id: selectedDocument.id, assignee_user_id: assigneeId,
      ...(evidenceTask ? { target_version_id: targetPin, structure_version_id: structurePin, guideline_version_id: guidelinePin } : {}),
      notes: notes.trim() || null,
    });
    setSavedAssignments((current) => [...current, assignment]);
    setNotes("");
    setMessage(`Assignment ${assignment.id} created.`);
  }

  return (
    <section className="platform-panel" aria-label="Document assignments">
      <h3>Assign documents</h3>
      <p>Assign a paper and its evidence version to an annotator.</p>
      {error ? <p role="alert" className="platform-dialog-error">{error}</p> : null}
      {refreshFailed ? <button type="button" disabled={refreshing} onClick={() => void retryRefresh()}>Retry refresh</button> : null}
      {message ? <p role="status">{message}</p> : null}
      <form className="platform-dialog-form" aria-label="Assign a document" onSubmit={(event) => { event.preventDefault(); void mutate(assignDocument); }}>
        <fieldset disabled={working}>
          <legend>Assign document</legend>
          <label><span>Document task</span><select required value={selectedTask?.id ?? 0} onChange={(event) => { setTaskId(Number(event.target.value)); setTargetVersionId(0); }}><option value={0}>Select a document task</option>{enabledTasks.map((task) => <option key={task.id} value={task.id}>{task.display_name}</option>)}</select></label>
          <label><span>Document</span><select required value={selectedDocument?.id ?? 0} onChange={(event) => setDocumentId(Number(event.target.value))}><option value={0}>Select a document</option>{projectDocuments.map((document) => <option key={document.id} value={document.id}>{documentName(document)}</option>)}</select></label>
          <label><span>Annotator</span><select required value={activeMembers.some((member) => member.user_id === assigneeId) ? assigneeId : 0} onChange={(event) => setAssigneeId(Number(event.target.value))}><option value={0}>Select an annotator</option>{activeMembers.map((member) => <option key={member.user_id} value={member.user_id}>{member.display_name || member.username}</option>)}</select></label>
          {evidenceTask ? <>
            <label><span>Evidence target version</span><select required value={selectedTarget?.active_version_id ?? 0} onChange={(event) => setTargetVersionId(Number(event.target.value))}><option value={0}>Select an active target version</option>{availableTargets.map((target) => <option key={target.id} value={target.active_version_id!}>{target.name} · v{target.versions.find((version) => version.id === target.active_version_id)?.version_number}</option>)}</select></label>
            {!availableTargets.length ? <p>Activate an evidence target above before assigning documents.</p> : null}
            {!activeGuideline ? <p>Create an active evidence guideline above before assigning documents.</p> : <p>Assignments use active guideline {activeGuideline.version_label} (version {activeGuideline.id}).</p>}
            {selectedDocument && !selectedDocument.active_structure_version_id ? <p>Import or process this document to prepare its structure before assigning evidence.</p> : null}
          </> : null}
          {!enabledTasks.length ? <p>Create or enable a document task before assigning documents.</p> : null}
          {!projectDocuments.length ? <p>Import papers from the Data page before creating document assignments.</p> : null}
          {!activeMembers.length ? <p>Add an active workspace member before assigning documents.</p> : null}
          <label><span>Assignment notes</span><textarea value={notes} onChange={(event) => setNotes(event.target.value)} /></label>
          <button type="submit" disabled={working}>Assign document</button>
        </fieldset>
      </form>
      <h3>Document assignments</h3>
      {projectAssignments.length ? <div className="platform-table-wrap"><table className="platform-table">
        <thead><tr><th>Document and task</th><th>Annotator</th><th>Status</th><th>Pinned versions</th><th>Actions</th></tr></thead>
        <tbody>{projectAssignments.map((assignment) => {
          const document = projectDocuments.find((item) => item.id === assignment.document_id);
          const action = nextAction(assignment.status);
          const closeIncomplete = projectTasks.find((task) => task.id === assignment.task_id)?.annotation_type === "evidence_block" &&
            [assignment.target_version_id, assignment.structure_version_id, assignment.guideline_version_id].includes(null) &&
            ["assigned", "in_progress", "blocked"].includes(assignment.status);
          return <tr key={assignment.id}>
            <td>{document ? documentName(document) : `Document ${assignment.document_id}`}<br />{projectTasks.find((task) => task.id === assignment.task_id)?.display_name ?? `Task ${assignment.task_id}`}</td>
            <td>{members.find((member) => member.user_id === assignment.assignee_user_id)?.display_name || assignment.annotator_id}</td>
            <td><PlatformStatus value={assignment.status} /></td>
            <td>Target version {assignment.target_version_id ?? "none"}; structure {assignment.structure_version_id ?? "none"}; guideline {assignment.guideline_version_id ?? "none"}
              {closeIncomplete ? <p className="platform-form-warning">This assignment is missing pinned versions. Close it before creating a replacement; recorded work is retained.</p> : null}</td>
            <td>{onOpenAssignment ? <button type="button" disabled={working} aria-label={`Open assignment ${assignment.id}`} onClick={() => onOpenAssignment(assignment)}>Open</button> : null}
              {action ? <button type="button" disabled={working} aria-label={`${action.label} assignment ${assignment.id}`} onClick={() => { void mutate(async () => { await updateTaskAssignment(projectId, assignment.id, { status: action.status }); setMessage(`Assignment ${assignment.id}: ${formatStatus(action.status)}.`); }); }}>{action.label}</button> : null}
              {closeIncomplete ? <button type="button" disabled={working} aria-label={`Close incomplete assignment ${assignment.id}`} onClick={() => {
                void mutate(async () => {
                  await updateTaskAssignment(projectId, assignment.id, { status: "completed" });
                  setMessage(`Incomplete assignment ${assignment.id} closed. Its recorded work and pinned history are retained.`);
                });
              }}>Close incomplete assignment</button> : null}</td>
          </tr>;
        })}</tbody>
      </table></div> : <p>No document assignments yet.</p>}
    </section>
  );
}
