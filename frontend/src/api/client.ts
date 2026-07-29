import type {
  Annotation,
  AnnotationCreate,
  AnnotationSet,
  AnnotationSetCreate,
  AnnotationSubmission,
  AnnotationSubmissionCreate,
  AnnotationUpdate,
  AnnotationWorkbench,
  ArtifactPackage,
  BaseModelAsset,
  BaseModelImportWrite,
  BaseModelReadinessWrite,
  BaseModelUploadWrite,
  CorpusSnapshot,
  CorpusSnapshotCreate,
  Document,
  DocumentStructureRead,
  EvidenceAdjudicationCreate,
  EvidenceAdjudicationRead,
  EvidenceCommandResult,
  EvidenceCommandSummary,
  EvidenceMergeRequest,
  EvidenceReviewCoverage,
  EvidenceReviewIntervalWrite,
  EvidenceSplitRequest,
  EvidenceTarget,
  EvidenceTargetCreate,
  EvidenceTargetVersion,
  EvidenceTargetVersionCreate,
  ExportArtifact,
  ExportCreate,
  ExportFormat,
  GuidelineVersion,
  GuidelineVersionCreate,
  ImportPreviewResponse,
  ImportResponse,
  InferenceRun,
  InferenceRunCreate,
  InferenceWindow,
  IaaReport,
  LineageGraph,
  EvidenceCandidatePrediction,
  PredictionReview,
  PredictionReviewCreate,
  PredictionStatus,
  Project,
  ProjectCreate,
  ProjectProgress,
  ProjectTask,
  ProjectTaskCreate,
  ProjectTaskUpdate,
  ProjectUpdate,
  SubmissionKind,
  TaskAssignment,
  TaskAssignmentCreate,
  TaskAssignmentStatus,
  TaskAssignmentUpdate,
  WorkspaceInvite,
  WorkspaceJoinRequest,
  WorkspaceMember,
  WorkspaceRole,
} from "@/types/api";
import { clearToken, getToken, setToken } from "@/auth/session";
import {
  normalizeArtifactPackage,
  normalizeArtifactPackages,
} from "@/api/artifactPackagesAdapter";

const API_BASE = "/api";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestResponse(path: string, init?: RequestInit): Promise<Response> {
  const requestToken = getToken();
  const multipartBody = typeof FormData !== "undefined" && init?.body instanceof FormData;
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(multipartBody ? {} : { "Content-Type": "application/json" }),
      ...(requestToken ? { Authorization: `Bearer ${requestToken}` } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = response.statusText;
    const responseClone = response.clone();
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") {
        message = body.detail;
      }
    } catch {
      try {
        const text = await responseClone.text();
        if (text.trim()) {
          message = text.trim();
        }
      } catch {
        // Keep the HTTP status text when the body cannot be read.
      }
    }

    if (response.status >= 500 && message === response.statusText) {
      message = "Backend API is unavailable or returned an internal error.";
    }
    if (response.status === 401 && getToken() === requestToken) {
      clearToken();
    }
    throw new ApiError(response.status, message);
  }

  return response;
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await requestResponse(path, init);

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export type AssignmentScope = "mine" | "all";

export function listProjects(workspaceId?: number | null): Promise<Project[]> {
  const params = new URLSearchParams();
  if (workspaceId !== undefined && workspaceId !== null) {
    params.set("workspace_id", String(workspaceId));
  }
  const query = params.toString();
  return request<Project[]>(`/projects${query ? `?${query}` : ""}`);
}

export function listMyWorkProjects(workspaceId: number): Promise<Project[]> {
  const params = new URLSearchParams({
    workspace_id: String(workspaceId),
  });
  return request<Project[]>(`/projects/my-work?${params.toString()}`);
}

export function getProject(projectId: number): Promise<Project> {
  return request<Project>(`/projects/${projectId}`);
}

export function createProject(payload: ProjectCreate): Promise<Project> {
  return request<Project>("/projects", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProject(projectId: number, payload: ProjectUpdate): Promise<Project> {
  return request<Project>(`/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function listDocuments(
  projectId: number,
  options: { scope?: AssignmentScope } = {},
): Promise<Document[]> {
  const params = new URLSearchParams({ project_id: String(projectId) });
  if (options.scope !== undefined) {
    params.set("scope", options.scope);
  }
  return request<Document[]>(`/documents?${params.toString()}`);
}

export interface DocumentStructureQuery {
  versionId?: number;
  sentenceStart?: number;
  sentenceLimit?: number;
}

export function getDocumentStructure(
  documentId: number,
  query: DocumentStructureQuery = {},
): Promise<DocumentStructureRead> {
  const params = new URLSearchParams();
  if (query.versionId !== undefined) {
    params.set("version_id", String(query.versionId));
  }
  if (query.sentenceStart !== undefined) {
    params.set("sentence_start", String(query.sentenceStart));
  }
  if (query.sentenceLimit !== undefined) {
    params.set("sentence_limit", String(query.sentenceLimit));
  }
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return request<DocumentStructureRead>(`/documents/${documentId}/structure${suffix}`);
}

export function rebuildDocumentStructure(
  documentId: number,
  activate = true,
): Promise<DocumentStructureRead> {
  return request<DocumentStructureRead>(`/documents/${documentId}/structure/rebuild`, {
    method: "POST",
    body: JSON.stringify({ activate }),
  });
}

export function activateDocumentStructure(
  documentId: number,
  versionId: number,
): Promise<DocumentStructureRead> {
  return request<DocumentStructureRead>(
    `/documents/${documentId}/structure/${versionId}/activate`,
    { method: "POST" },
  );
}

export function previewPubmedImport(
  projectId: number,
  pmids: string[],
): Promise<ImportPreviewResponse> {
  return request<ImportPreviewResponse>(`/projects/${projectId}/import/pubmed/preview`, {
    method: "POST",
    body: JSON.stringify({ pmids }),
  });
}

export function importPubmed(
  projectId: number,
  pmids: string[],
  includeAbstractOnly: boolean,
): Promise<ImportResponse> {
  return request<ImportResponse>(`/projects/${projectId}/import/pubmed`, {
    method: "POST",
    body: JSON.stringify({ pmids, include_abstract_only: includeAbstractOnly }),
  });
}

export function listProjectTasks(projectId: number, enabledOnly = false): Promise<ProjectTask[]> {
  const params = new URLSearchParams({ enabled_only: String(enabledOnly) });
  return request<ProjectTask[]>(`/projects/${projectId}/tasks?${params.toString()}`);
}

export function createProjectTask(
  projectId: number,
  payload: ProjectTaskCreate,
): Promise<ProjectTask> {
  return request<ProjectTask>(`/projects/${projectId}/tasks`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateProjectTask(
  projectId: number,
  taskId: number,
  payload: ProjectTaskUpdate,
): Promise<ProjectTask> {
  return request<ProjectTask>(`/projects/${projectId}/tasks/${taskId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteProjectTask(projectId: number, taskId: number): Promise<void> {
  return request<void>(`/projects/${projectId}/tasks/${taskId}`, {
    method: "DELETE",
  });
}

export function listEvidenceTargets(projectId: number): Promise<EvidenceTarget[]> {
  return request<EvidenceTarget[]>(`/projects/${projectId}/evidence-targets`);
}

export function createEvidenceTarget(
  projectId: number,
  payload: EvidenceTargetCreate,
): Promise<EvidenceTarget> {
  return request<EvidenceTarget>(`/projects/${projectId}/evidence-targets`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createEvidenceTargetVersion(
  projectId: number,
  targetId: number,
  payload: EvidenceTargetVersionCreate,
): Promise<EvidenceTargetVersion> {
  return request<EvidenceTargetVersion>(
    `/projects/${projectId}/evidence-targets/${targetId}/versions`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function activateEvidenceTarget(
  projectId: number,
  targetId: number,
  versionId: number,
): Promise<EvidenceTarget> {
  return request<EvidenceTarget>(
    `/projects/${projectId}/evidence-targets/${targetId}/activate`,
    { method: "POST", body: JSON.stringify({ version_id: versionId }) },
  );
}

export function deactivateEvidenceTarget(
  projectId: number,
  targetId: number,
): Promise<EvidenceTarget> {
  return request<EvidenceTarget>(
    `/projects/${projectId}/evidence-targets/${targetId}/deactivate`,
    { method: "POST" },
  );
}

export interface TaskAssignmentFilters {
  documentId?: number;
  taskId?: number;
  assigneeUserId?: number;
  annotatorId?: string;
  status?: TaskAssignmentStatus;
  scope?: AssignmentScope;
}

export function listTaskAssignments(
  projectId: number,
  filters: TaskAssignmentFilters = {},
): Promise<TaskAssignment[]> {
  const params = new URLSearchParams();
  if (filters.documentId !== undefined) {
    params.set("document_id", String(filters.documentId));
  }
  if (filters.taskId !== undefined) {
    params.set("task_id", String(filters.taskId));
  }
  if (filters.assigneeUserId !== undefined) {
    params.set("assignee_user_id", String(filters.assigneeUserId));
  }
  if (filters.annotatorId !== undefined) {
    params.set("annotator_id", filters.annotatorId);
  }
  if (filters.status !== undefined) {
    params.set("status", filters.status);
  }
  if (filters.scope !== undefined) {
    params.set("scope", filters.scope);
  }
  const query = params.toString();
  return request<TaskAssignment[]>(
    `/projects/${projectId}/assignments${query ? `?${query}` : ""}`,
  );
}

export function createTaskAssignment(
  projectId: number,
  payload: TaskAssignmentCreate,
): Promise<TaskAssignment> {
  return request<TaskAssignment>(`/projects/${projectId}/assignments`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateTaskAssignment(
  projectId: number,
  assignmentId: number,
  payload: TaskAssignmentUpdate,
): Promise<TaskAssignment> {
  return request<TaskAssignment>(`/projects/${projectId}/assignments/${assignmentId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function reopenPersonalTaskAssignment(
  projectId: number,
  assignmentId: number,
): Promise<TaskAssignment> {
  return request<TaskAssignment>(
    `/projects/${projectId}/assignments/${assignmentId}/reopen`,
    { method: "POST" },
  );
}

export function deleteTaskAssignment(projectId: number, assignmentId: number): Promise<void> {
  return request<void>(`/projects/${projectId}/assignments/${assignmentId}`, {
    method: "DELETE",
  });
}

export function getProjectProgress(
  projectId: number,
  scope?: AssignmentScope,
): Promise<ProjectProgress> {
  const params = new URLSearchParams();
  if (scope !== undefined) {
    params.set("scope", scope);
  }
  const query = params.toString();
  return request<ProjectProgress>(`/projects/${projectId}/progress${query ? `?${query}` : ""}`);
}

export function listGuidelineVersions(projectId: number): Promise<GuidelineVersion[]> {
  const params = new URLSearchParams({ project_id: String(projectId) });
  return request<GuidelineVersion[]>(`/guidelines?${params.toString()}`);
}

export function createGuidelineVersion(
  payload: GuidelineVersionCreate,
): Promise<GuidelineVersion> {
  return request<GuidelineVersion>("/guidelines", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getAnnotationWorkbench(documentId: number): Promise<AnnotationWorkbench> {
  return request<AnnotationWorkbench>(`/annotation-workbench/documents/${documentId}`);
}

export function createAnnotation(payload: AnnotationCreate): Promise<Annotation> {
  return request<Annotation>("/annotations", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getAnnotation(annotationId: number): Promise<Annotation> {
  return request<Annotation>(`/annotations/${annotationId}`);
}

export function updateAnnotation(
  annotationId: number,
  payload: AnnotationUpdate,
): Promise<Annotation> {
  return request<Annotation>(`/annotations/${annotationId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteAnnotation(annotationId: number, expectedRevision?: number): Promise<void> {
  const suffix = expectedRevision === undefined ? "" : `?expected_revision=${expectedRevision}`;
  return request<void>(`/annotations/${annotationId}${suffix}`, {
    method: "DELETE",
  });
}

export function mergeEvidenceBlocks(payload: EvidenceMergeRequest): Promise<Annotation> {
  return request<Annotation>("/annotations/evidence-blocks/merge", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function splitEvidenceBlock(
  annotationId: number,
  payload: EvidenceSplitRequest,
): Promise<Annotation[]> {
  return request<Annotation[]>(`/annotations/${annotationId}/split`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listEvidenceCommands(
  projectId: number,
  filters: {
    documentId?: number;
    targetVersionId?: number;
    structureVersionId?: number;
    guidelineVersionId?: number | null;
  } = {},
): Promise<EvidenceCommandSummary[]> {
  const params = new URLSearchParams({ project_id: String(projectId) });
  if (filters.documentId !== undefined) {
    params.set("document_id", String(filters.documentId));
  }
  if (filters.targetVersionId !== undefined) {
    params.set("target_version_id", String(filters.targetVersionId));
  }
  if (filters.structureVersionId !== undefined) {
    params.set("structure_version_id", String(filters.structureVersionId));
  }
  if (filters.guidelineVersionId !== null && filters.guidelineVersionId !== undefined) {
    params.set("guideline_version_id", String(filters.guidelineVersionId));
  }
  return request<EvidenceCommandSummary[]>(
    `/annotations/evidence-blocks/commands?${params.toString()}`,
  );
}

export function undoEvidenceCommand(commandGroupKey: string): Promise<EvidenceCommandResult> {
  return request<EvidenceCommandResult>(
    `/annotations/evidence-blocks/commands/${encodeURIComponent(commandGroupKey)}/undo`,
    { method: "POST" },
  );
}

export function redoEvidenceCommand(commandGroupKey: string): Promise<EvidenceCommandResult> {
  return request<EvidenceCommandResult>(
    `/annotations/evidence-blocks/commands/${encodeURIComponent(commandGroupKey)}/redo`,
    { method: "POST" },
  );
}

export function getEvidenceReviewCoverage(
  projectId: number,
  documentId: number,
  targetVersionId: number,
  structureVersionId: number,
  guidelineVersionId?: number | null,
): Promise<EvidenceReviewCoverage> {
  const params = new URLSearchParams({
    target_version_id: String(targetVersionId),
    structure_version_id: String(structureVersionId),
  });
  if (guidelineVersionId !== null && guidelineVersionId !== undefined) {
    params.set("guideline_version_id", String(guidelineVersionId));
  }
  return request<EvidenceReviewCoverage>(
    `/projects/${projectId}/documents/${documentId}/evidence-review-coverage?${params.toString()}`,
  );
}

export function markEvidenceReviewed(
  projectId: number,
  documentId: number,
  payload: EvidenceReviewIntervalWrite,
): Promise<EvidenceReviewCoverage> {
  return request<EvidenceReviewCoverage>(
    `/projects/${projectId}/documents/${documentId}/evidence-review-coverage/mark-reviewed`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function reopenEvidenceReview(
  projectId: number,
  documentId: number,
  payload: EvidenceReviewIntervalWrite,
): Promise<EvidenceReviewCoverage> {
  return request<EvidenceReviewCoverage>(
    `/projects/${projectId}/documents/${documentId}/evidence-review-coverage/reopen`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function getEvidenceAdjudication(
  projectId: number,
  documentId: number,
  targetVersionId: number,
  structureVersionId: number,
  guidelineVersionId: number,
): Promise<EvidenceAdjudicationRead> {
  const params = new URLSearchParams({
    target_version_id: String(targetVersionId),
    structure_version_id: String(structureVersionId),
    guideline_version_id: String(guidelineVersionId),
  });
  return request<EvidenceAdjudicationRead>(
    `/projects/${projectId}/documents/${documentId}/evidence-adjudication?${params.toString()}`,
  );
}

export function adjudicateEvidence(
  projectId: number,
  documentId: number,
  payload: EvidenceAdjudicationCreate,
): Promise<Annotation> {
  return request<Annotation>(
    `/projects/${projectId}/documents/${documentId}/evidence-adjudication`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function getProjectIaa(
  projectId: number,
  params: {
    annotationType: "evidence_block";
    documentId: number;
    targetVersionId: number;
    structureVersionId: number;
    guidelineVersionId?: number | null;
  },
): Promise<IaaReport> {
  const query = new URLSearchParams({
    annotation_type: params.annotationType,
    document_id: String(params.documentId),
    target_version_id: String(params.targetVersionId),
    structure_version_id: String(params.structureVersionId),
  });
  if (params.guidelineVersionId === null) {
    query.set("legacy_guideline", "true");
  } else if (params.guidelineVersionId !== undefined) {
    query.set("guideline_version_id", String(params.guidelineVersionId));
  }
  return request<IaaReport>(`/projects/${projectId}/iaa?${query.toString()}`);
}

export function createSubmission(
  projectId: number,
  documentId: number,
  payload: AnnotationSubmissionCreate,
): Promise<AnnotationSubmission> {
  return request<AnnotationSubmission>(
    `/projects/${projectId}/documents/${documentId}/submissions`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export function listSubmissions(
  projectId: number,
  filters: {
    documentId?: number;
    annotatorUserId?: number;
    annotatorId?: string;
    kind?: SubmissionKind;
    scope?: AssignmentScope;
  } = {},
): Promise<AnnotationSubmission[]> {
  const params = new URLSearchParams();
  if (filters.documentId !== undefined) {
    params.set("document_id", String(filters.documentId));
  }
  if (filters.annotatorUserId !== undefined) {
    params.set("annotator_user_id", String(filters.annotatorUserId));
  }
  if (filters.annotatorId !== undefined) {
    params.set("annotator_id", filters.annotatorId);
  }
  if (filters.kind !== undefined) {
    params.set("kind", filters.kind);
  }
  if (filters.scope !== undefined) {
    params.set("scope", filters.scope);
  }
  const query = params.toString();
  return request<AnnotationSubmission[]>(
    `/projects/${projectId}/submissions${query ? `?${query}` : ""}`,
  );
}

export function deleteSubmission(submissionId: number): Promise<void> {
  return request<void>(`/submissions/${submissionId}`, {
    method: "DELETE",
  });
}

export async function downloadSubmission(submissionId: number): Promise<Blob> {
  const response = await requestResponse(`/submissions/${submissionId}/download`);
  return response.blob();
}

export function createCorpusSnapshot(
  projectId: number,
  payload: CorpusSnapshotCreate,
): Promise<CorpusSnapshot> {
  return request<CorpusSnapshot>(`/projects/${projectId}/corpus-snapshots`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listCorpusSnapshots(projectId: number): Promise<CorpusSnapshot[]> {
  return request<CorpusSnapshot[]>(`/projects/${projectId}/corpus-snapshots`);
}

export function getCorpusSnapshot(snapshotId: number): Promise<CorpusSnapshot> {
  return request<CorpusSnapshot>(`/corpus-snapshots/${snapshotId}`);
}

export function createAnnotationSet(
  projectId: number,
  payload: AnnotationSetCreate,
): Promise<AnnotationSet> {
  return request<AnnotationSet>(`/projects/${projectId}/annotation-sets`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listAnnotationSets(projectId: number): Promise<AnnotationSet[]> {
  return request<AnnotationSet[]>(`/projects/${projectId}/annotation-sets`);
}

export function getAnnotationSet(annotationSetId: number): Promise<AnnotationSet> {
  return request<AnnotationSet>(`/annotation-sets/${annotationSetId}`);
}

export function getLineageGraph(artifactId: number): Promise<LineageGraph> {
  return request<LineageGraph>(`/lineage/artifacts/${artifactId}/graph`);
}

export function listExportFormats(): Promise<ExportFormat[]> {
  return request<ExportFormat[]>("/export-formats");
}

export function createExport(projectId: number, payload: ExportCreate): Promise<ExportArtifact> {
  return request<ExportArtifact>(`/projects/${projectId}/exports`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listExports(projectId: number): Promise<ExportArtifact[]> {
  return request<ExportArtifact[]>(`/projects/${projectId}/exports`);
}

export async function downloadExport(exportId: number): Promise<Blob> {
  const response = await requestResponse(`/exports/${exportId}/download`);
  return response.blob();
}

export async function listArtifactPackages(projectId: number): Promise<ArtifactPackage[]> {
  const response = await request<unknown>(
    `/projects/${projectId}/artifact-packages`,
  );
  return normalizeArtifactPackages(response);
}

export async function getArtifactPackage(packageId: number): Promise<ArtifactPackage> {
  const response = await request<unknown>(`/artifact-packages/${packageId}`);
  return normalizeArtifactPackage(response);
}

function normalizeBaseModelAsset(value: unknown): BaseModelAsset {
  const item = value as Omit<BaseModelAsset, "package"> & { package: unknown };
  return { ...item, package: normalizeArtifactPackage(item.package) };
}

export async function listBaseModels(
  projectId: number,
  options: { readiness?: string; includeArchived?: boolean } = {},
): Promise<BaseModelAsset[]> {
  const params = new URLSearchParams();
  if (options.readiness) params.set("readiness", options.readiness);
  if (options.includeArchived) params.set("include_archived", "true");
  const query = params.toString();
  const response = await request<unknown[]>(
    `/projects/${projectId}/base-models${query ? `?${query}` : ""}`,
  );
  return response.map(normalizeBaseModelAsset);
}

export async function importBaseModel(
  projectId: number,
  payload: BaseModelImportWrite,
): Promise<BaseModelAsset> {
  const response = await request<unknown>(`/projects/${projectId}/base-models/import`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  return normalizeBaseModelAsset(response);
}

export async function uploadBaseModel(
  projectId: number,
  metadata: BaseModelUploadWrite,
  files: File[],
): Promise<BaseModelAsset> {
  const body = new FormData();
  body.append("metadata", JSON.stringify(metadata));
  for (const file of files) body.append("files", file, file.name);
  const response = await request<unknown>(`/projects/${projectId}/base-models/upload`, {
    method: "POST",
    body,
  });
  return normalizeBaseModelAsset(response);
}

export async function setBaseModelReadiness(
  assetId: number,
  payload: BaseModelReadinessWrite,
): Promise<BaseModelAsset> {
  const response = await request<unknown>(`/base-models/${assetId}/readiness`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  return normalizeBaseModelAsset(response);
}

export async function archiveBaseModel(assetId: number): Promise<BaseModelAsset> {
  const response = await request<unknown>(`/base-models/${assetId}/archive`, {
    method: "POST",
  });
  return normalizeBaseModelAsset(response);
}

export async function downloadArtifactPackageFile(
  packageId: number,
  fileId: number,
): Promise<Blob> {
  const response = await requestResponse(
    `/artifact-packages/${packageId}/files/${fileId}/download`,
  );
  return response.blob();
}

export function archiveArtifactPackage(packageId: number): Promise<{
  package_id: number;
  archived_at: string;
  purge_after: string;
}> {
  return request(`/artifact-packages/${packageId}/archive`, { method: "POST" });
}

export function purgeArtifactPackage(packageId: number, reason: string): Promise<{
  package_id: number;
  archived_at: string;
  purge_after: string;
  purged_at: string | null;
}> {
  return request(`/artifact-packages/${packageId}/purge`, {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export function launchInferenceRun(
  projectId: number,
  payload: InferenceRunCreate,
): Promise<InferenceRun> {
  return request<InferenceRun>(`/projects/${projectId}/inference/runs`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listInferenceRuns(projectId: number): Promise<InferenceRun[]> {
  return request<InferenceRun[]>(`/projects/${projectId}/inference/runs`);
}

export function getInferenceRun(runId: number): Promise<InferenceRun> {
  return request<InferenceRun>(`/inference/runs/${runId}`);
}

export function cancelInferenceRun(runId: number): Promise<InferenceRun> {
  return request<InferenceRun>(`/inference/runs/${runId}/cancel`, {
    method: "POST",
  });
}

export function listInferenceWindows(runId: number): Promise<InferenceWindow[]> {
  return request<InferenceWindow[]>(`/inference/runs/${runId}/windows`);
}

export interface InferencePredictionFilters {
  documentId?: number;
  targetVersionId?: number;
  status?: PredictionStatus;
}

export function listInferencePredictions(
  runId: number,
  filters: InferencePredictionFilters = {},
): Promise<EvidenceCandidatePrediction[]> {
  const params = new URLSearchParams();
  if (filters.documentId !== undefined) {
    params.set("document_id", String(filters.documentId));
  }
  if (filters.targetVersionId !== undefined) {
    params.set("target_version_id", String(filters.targetVersionId));
  }
  if (filters.status !== undefined) {
    params.set("status", filters.status);
  }
  const query = params.toString();
  return request<EvidenceCandidatePrediction[]>(
    `/inference/runs/${runId}/predictions${query ? `?${query}` : ""}`,
  );
}

export function getInferencePrediction(
  predictionId: number,
): Promise<EvidenceCandidatePrediction> {
  return request<EvidenceCandidatePrediction>(`/inference/predictions/${predictionId}`);
}

export function reviewInferencePrediction(
  predictionId: number,
  payload: PredictionReviewCreate,
): Promise<PredictionReview> {
  return request<PredictionReview>(`/inference/predictions/${predictionId}/review`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export interface WorkspaceCapabilities {
  preset: string;
  overrides: string[];
  effective: string[];
  blocked: Record<string, string>;
}

export interface MeMembership {
  workspace_id: number;
  workspace_name: string;
  workspace_kind: string;
  role: string;
}

export interface MeResponse {
  user: {
    id: number;
    username: string;
    display_name: string | null;
    is_active: boolean;
    is_superuser: boolean;
  };
  memberships: MeMembership[];
}

export function getWorkspaceCapabilities(workspaceId: number): Promise<WorkspaceCapabilities> {
  return request<WorkspaceCapabilities>(`/workspaces/${workspaceId}/capabilities`);
}

export function setWorkspaceCapability(
  workspaceId: number,
  preset: string,
  overrides: string[] = [],
): Promise<WorkspaceCapabilities> {
  return request<WorkspaceCapabilities>(`/workspaces/${workspaceId}/capability`, {
    method: "PATCH",
    body: JSON.stringify({ preset, overrides }),
  });
}

export function listWorkspaceMembers(workspaceId: number): Promise<WorkspaceMember[]> {
  return request<WorkspaceMember[]>(`/workspaces/${workspaceId}/members`);
}

export function updateWorkspaceMemberRole(
  workspaceId: number,
  userId: number,
  role: WorkspaceRole,
): Promise<WorkspaceMember> {
  return request<WorkspaceMember>(`/workspaces/${workspaceId}/members/${userId}`, {
    method: "PATCH",
    body: JSON.stringify({ role }),
  });
}

export function deleteWorkspaceMember(workspaceId: number, userId: number): Promise<void> {
  return request<void>(`/workspaces/${workspaceId}/members/${userId}`, {
    method: "DELETE",
  });
}

export function createWorkspaceInvite(
  workspaceId: number,
  role: WorkspaceRole,
): Promise<WorkspaceInvite> {
  return request<WorkspaceInvite>(`/workspaces/${workspaceId}/invites`, {
    method: "POST",
    body: JSON.stringify({ role }),
  });
}

export function listWorkspaceJoinRequests(
  workspaceId: number,
): Promise<WorkspaceJoinRequest[]> {
  return request<WorkspaceJoinRequest[]>(`/workspaces/${workspaceId}/join-requests`);
}

export function approveJoinRequest(requestId: number): Promise<WorkspaceJoinRequest> {
  return request<WorkspaceJoinRequest>(`/join-requests/${requestId}/approve`, {
    method: "POST",
  });
}

export function rejectJoinRequest(requestId: number): Promise<WorkspaceJoinRequest> {
  return request<WorkspaceJoinRequest>(`/join-requests/${requestId}/reject`, {
    method: "POST",
  });
}

export async function login(username: string, password: string): Promise<void> {
  const response = await request<{ access_token: string }>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(response.access_token);
}

export interface RegisterOptions {
  displayName?: string;
  workspaceKind: "individual" | "team";
  workspaceName?: string;
}

export async function register(
  username: string,
  password: string,
  options: RegisterOptions,
): Promise<void> {
  const response = await request<{ access_token: string }>("/auth/register", {
    method: "POST",
    body: JSON.stringify({
      username,
      password,
      display_name: options.displayName,
      workspace_kind: options.workspaceKind,
      workspace_name: options.workspaceName,
    }),
  });
  setToken(response.access_token);
}

export function getMe(): Promise<MeResponse> {
  return request<MeResponse>("/auth/me");
}
