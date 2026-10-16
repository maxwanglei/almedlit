import { useCallback, useEffect, useRef, useState } from "react";

import {
  listDocuments,
  listEvidenceTargets,
  listGuidelineVersions,
  listProjectTasks,
  listTaskAssignments,
  listWorkspaceMembers,
  rebuildDocumentStructure,
} from "@/api/client";
import type {
  Document,
  EvidenceTarget,
  GuidelineVersion,
  ProjectTask,
  TaskAssignment,
  WorkspaceMember,
} from "@/types/api";

import DocumentAssignmentManagement, { DocumentTaskManagement } from "./DocumentAssignmentManagement";
import EvidenceGuidelineManagement from "./EvidenceGuidelineManagement";
import EvidenceTargetManagement from "./EvidenceTargetManagement";
import { PlatformSection } from "./components";

interface SetupData {
  tasks: ProjectTask[];
  targets: EvidenceTarget[];
  documents: Document[];
  members: WorkspaceMember[];
  assignments: TaskAssignment[];
  guidelines: GuidelineVersion[];
}

interface EvidenceSetupProps {
  projectId: number;
  workspaceId: number | null;
  busy?: boolean;
  onChanged: () => Promise<unknown>;
  onNavigate: (path: string) => void;
}

export default function EvidenceSetupPanel(props: EvidenceSetupProps): React.ReactElement {
  return <ProjectEvidenceSetup key={`${props.workspaceId}:${props.projectId}`} {...props} />;
}

function ProjectEvidenceSetup({ projectId, workspaceId, busy = false, onChanged, onNavigate }: EvidenceSetupProps): React.ReactElement {
  const [data, setData] = useState<SetupData | null>(null);
  const [loading, setLoading] = useState(true);
  const [preparing, setPreparing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const mounted = useRef(false);
  const requestId = useRef(0);

  const reload = useCallback(async (): Promise<void> => {
    if (!mounted.current) return;
    const currentRequest = ++requestId.current;
    const current = (): boolean => mounted.current && currentRequest === requestId.current;
    setLoading(true);
    setError(null);
    try {
      if (workspaceId === null) throw new Error("Select a workspace to configure evidence annotation.");
      const [tasks, targets, documents, members, assignments, guidelines] = await Promise.all([
        listProjectTasks(projectId),
        listEvidenceTargets(projectId),
        listDocuments(projectId, { scope: "all" }),
        listWorkspaceMembers(workspaceId),
        listTaskAssignments(projectId, { scope: "all" }),
        listGuidelineVersions(projectId),
      ]);
      if (current()) setData({ tasks, targets, documents, members, assignments, guidelines });
    } catch (caught) {
      if (current()) setError(caught instanceof Error ? caught.message : "Evidence setup could not be loaded.");
      throw caught;
    } finally {
      if (current()) setLoading(false);
    }
  }, [projectId, workspaceId]);

  useEffect(() => {
    mounted.current = true;
    void reload().catch(() => undefined);
    return () => {
      mounted.current = false;
      requestId.current += 1;
    };
  }, [reload]);

  async function refresh(): Promise<void> {
    if (!mounted.current) return;
    await reload();
    if (mounted.current) await onChanged();
  }

  async function prepareDocument(document: Document): Promise<void> {
    if (busy || loading || preparing) return;
    setPreparing(true);
    setError(null);
    setStatus(null);
    let saved = false;
    try {
      const structure = await rebuildDocumentStructure(document.id, true);
      if (!mounted.current) return;
      const activeVersionId = structure.active_structure_version_id;
      if (activeVersionId === null) throw new Error("The document has no active structure after preparation. Try preparing it again.");
      saved = true;
      setData((previous) => previous ? {
        ...previous,
        documents: previous.documents.map((item) => item.id === document.id
          ? { ...item, active_structure_version_id: activeVersionId }
          : item),
      } : previous);
      setStatus(`${document.title || document.external_id || `Document ${document.id}`} is ready for evidence annotation.`);
      await refresh();
    } catch (caught) {
      if (mounted.current) {
        const message = caught instanceof Error ? caught.message : "The document could not be prepared.";
        setError(saved ? `Document prepared, but setup could not be refreshed. ${message}` : message);
      }
    } finally {
      if (mounted.current) setPreparing(false);
    }
  }

  const disabled = busy || loading || preparing;
  const hasGuideline = data?.guidelines.some((guideline) => guideline.project_id === projectId && guideline.status === "active") ?? false;

  return (
    <div id="evidence-setup" className="platform-page" aria-busy={loading}>
      <PlatformSection title="Evidence setup" description="Add a document task, save its guidelines, activate an evidence target, then assign documents to annotators. Review submitted evidence in Quality & Review.">
        <div className="platform-page-actions">
          <button type="button" className="platform-text-action" onClick={() => onNavigate(`/projects/${projectId}/data?tab=source`)}>Import documents</button>
          <button type="button" className="platform-text-action" onClick={() => onNavigate(`/my-work?project=${projectId}&view=annotate&tool=evidence`)}>Open My Work</button>
          <button type="button" className="platform-text-action" onClick={() => onNavigate(`/projects/${projectId}/quality`)}>Review evidence</button>
        </div>
        {loading ? <p role="status">Loading evidence setup…</p> : null}
        {error ? <div role="alert"><p>{error}</p><button type="button" disabled={disabled} onClick={() => void refresh().catch(() => undefined)}>Retry evidence setup</button></div> : null}
        {status ? <p role="status">{status}</p> : null}
        {data && data.documents.length === 0 ? <p>Import documents in Data before assigning evidence annotation.</p> : null}
        {data?.documents.filter((document) => document.active_structure_version_id === null).map((document) => (
          <p key={document.id}>
            {document.title || document.external_id || `Document ${document.id}`} needs preparation for evidence annotation.{" "}
            <button type="button" disabled={disabled} onClick={() => void prepareDocument(document)}>Prepare document {document.id}</button>
          </p>
        ))}
      </PlatformSection>
      {data ? <>
        <DocumentTaskManagement projectId={projectId} tasks={data.tasks} busy={disabled} onChanged={refresh} />
        <EvidenceGuidelineManagement projectId={projectId} guidelines={data.guidelines} busy={disabled} onChanged={refresh} />
        <EvidenceTargetManagement projectId={projectId} tasks={data.tasks} targets={data.targets} busy={disabled} onChanged={refresh}
          activationBlockedReason={hasGuideline ? undefined : "Create an active evidence guideline before activating targets so assignments can be adjudicated."} />
        <DocumentAssignmentManagement projectId={projectId} tasks={data.tasks} documents={data.documents} members={data.members}
          targets={data.targets} assignments={data.assignments} guidelines={data.guidelines} busy={disabled} onChanged={refresh} />
      </> : null}
    </div>
  );
}
