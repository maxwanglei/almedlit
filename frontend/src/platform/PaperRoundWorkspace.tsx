import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type * as legacyApi from "@/api/client";
import { useSearchState } from "@/hooks/useSearchState";
import AnnotatorWorkspace from "@/pages/AnnotatorWorkspace";
import type { Annotation, AnnotationCreate, AnnotationSubmission, AnnotationTypeSpec, AnnotationWorkbench, AnnotationWorkbenchTask, Document, Project, TaskAssignment } from "@/types/api";

import { addRoundDecision, loadRoundWork, submitRoundDecisions, type RoundWorkData } from "./api";
import { paperAnnotationCompatibility, paperTaskLabels } from "./paperTaskContracts";
import RoundWorkbench from "./RoundWorkbench";
import type { AnnotationDecision, RoundSubmission, RoundWorkContext, RoundWorkItemIdentity } from "./types";

type PaperTaskType = "doc_label" | "entity";
interface EntityOutput { start: number; end: number; label: string }
export interface LoadedPaperRound { context: RoundWorkContext; work: RoundWorkData }
export interface PaperAnnotationApi {
  createAnnotation: typeof legacyApi.createAnnotation;
  updateAnnotation: typeof legacyApi.updateAnnotation;
  deleteAnnotation: typeof legacyApi.deleteAnnotation;
  createSubmission: typeof legacyApi.createSubmission;
  reopenPersonalTaskAssignment: typeof legacyApi.reopenPersonalTaskAssignment;
  setDocumentLabel: (documentId: number, label: string) => Promise<Annotation>;
}

const COLORS = ["#4d6e5b", "#8e3a3a", "#4a5a8a", "#8a6d2a", "#6a4a7a"];

/** Only contracts the existing paper editor can represent without transforming inputs. */
export function paperRoundAnnotationType(context: RoundWorkContext): PaperTaskType | null {
  const compatibility = paperAnnotationCompatibility(context.task_version);
  return compatibility.editor === "classification" ? "doc_label" : compatibility.editor === "entities" ? "entity" : null;
}

/** One current round per task; opening an older round deliberately keeps that round. */
export function selectPaperRoundContexts(context: RoundWorkContext, contexts: RoundWorkContext[]): RoundWorkContext[] {
  const eligible = (item: RoundWorkContext) => item.project.id === context.project.id &&
    item.round.dataset_version_id === context.round.dataset_version_id && item.round.status === "open" &&
    item.round.assistance_policy === "blind" && paperRoundAnnotationType(item) !== null;
  if (!eligible(context)) return [];
  const byTask = new Map<number, RoundWorkContext>();
  for (const candidate of [...contexts].filter(eligible).sort((left, right) => left.round.sequence - right.round.sequence || left.round.id - right.round.id)) {
    byTask.set(candidate.task.id, candidate);
  }
  byTask.set(context.task.id, context);
  return [...byTask.values()].sort((left, right) => Number(right.round.id === context.round.id) - Number(left.round.id === context.round.id) || right.round.sequence - left.round.sequence);
}

function ownLatest(round: LoadedPaperRound, itemId: number, userId: number): AnnotationDecision | undefined {
  const decisions = round.work.decisions.filter((decision) => decision.round_item_id === itemId && decision.annotator_user_id === userId);
  const superseded = new Set(decisions.map((decision) => decision.supersedes_decision_id));
  return decisions.filter((decision) => !superseded.has(decision.id)).sort((left, right) => right.id - left.id)[0];
}

function submissionFor(round: LoadedPaperRound, decision: AnnotationDecision | undefined, userId: number): RoundSubmission | undefined {
  return decision ? round.work.submissions.find((submission) => submission.annotator_user_id === userId && submission.decision_ids.includes(decision.id)) : undefined;
}

function entitiesOf(output: unknown): EntityOutput[] {
  if (output === undefined) return [];
  if (!output || typeof output !== "object" || !Array.isArray((output as { entities?: unknown }).entities)) throw new Error("This NER decision does not use the supported entity-span format.");
  return (output as { entities: EntityOutput[] }).entities;
}

function sourceDocument(round: LoadedPaperRound, item: RoundWorkItemIdentity): Document {
  const source = round.work.datasetItems.find((candidate) => candidate.id === item.dataset_item_id);
  if (!source || source.dataset_version_id !== round.context.round.dataset_version_id || source.project_id !== round.context.project.id ||
    !Number.isInteger(source.payload.document_id) || Number(source.payload.document_id) <= 0 || typeof source.payload.text !== "string" || !source.payload.text.trim()) {
    throw new Error("This round needs a saved paper collection containing original document IDs and text. Open task setup to choose a compatible source.");
  }
  return {
    id: Number(source.payload.document_id), project_id: round.context.project.id,
    external_id: typeof source.payload.external_id === "string" ? source.payload.external_id : null,
    title: typeof source.payload.title === "string" ? source.payload.title : null,
    text: source.payload.text, source: "saved_collection", sentences: [[0, source.payload.text.length]], active_structure_version_id: null,
    metadata_: { ...(source.payload.metadata && typeof source.payload.metadata === "object" ? source.payload.metadata : {}), dataset_item_id: source.id, dataset_version_id: source.dataset_version_id },
  };
}

function taskLabels(context: RoundWorkContext): string[] {
  return paperTaskLabels(context.task_version);
}

function taskSpec(type: PaperTaskType): AnnotationTypeSpec {
  return { name: type, requires_span: type === "entity", requires_head_tail: false,
    description: type === "entity" ? "Label entity spans in the saved paper text." : "Assign one label to this paper.",
    selection_mode: type === "entity" ? "character_span" : "document", renderer_key: "legacy", relation_endpoint_allowed: false, handler_key: "generic" };
}

function projectedTask(round: LoadedPaperRound, index: number): AnnotationWorkbenchTask {
  const type = paperRoundAnnotationType(round.context)!;
  return { id: round.context.task_version.id, project_id: round.context.project.id, annotation_type: type,
    display_name: round.context.task.name, description: `Round: ${round.context.round.name}`, enabled: true, sort_order: index,
    labels: taskLabels(round.context).map((name, labelIndex) => ({ name, color: COLORS[labelIndex % COLORS.length], description: null })),
    settings: { round_id: round.context.round.id, dataset_version_id: round.context.round.dataset_version_id }, annotation_type_spec: taskSpec(type) };
}

function projectedAssignment(round: LoadedPaperRound, item: RoundWorkItemIdentity, userId: number, annotatorId: string): TaskAssignment {
  const latest = ownLatest(round, item.id, userId);
  return { id: item.id, project_id: round.context.project.id, task_id: round.context.task_version.id,
    document_id: sourceDocument(round, item).id, assignee_user_id: userId, annotator_id: annotatorId,
    status: submissionFor(round, latest, userId) ? "submitted" : latest ? "in_progress" : "assigned",
    assigned_by_user_id: null, assigned_by: null, notes: null, target_version_id: null, structure_version_id: null, guideline_version_id: null,
    assignment_scope_key: `round:${round.context.round.id}:item:${item.id}`, metadata_: { round_id: round.context.round.id, dataset_item_id: item.dataset_item_id } };
}

export function createPaperAnnotationIds(): (roundItemId: number, index: number) => number {
  const ids = new Map<string, number>();
  return (roundItemId, index) => {
    const key = `${roundItemId}:${index}`;
    if (!ids.has(key)) ids.set(key, -(ids.size + 1));
    return ids.get(key)!;
  };
}

export function projectPaperRoundWork(rounds: LoadedPaperRound[], userId: number, annotatorId: string, annotationId: ReturnType<typeof createPaperAnnotationIds>) {
  const documents = new Map<number, Document>();
  const assignments: TaskAssignment[] = [];
  const annotations: Annotation[] = [];
  const tasks = rounds.map(projectedTask);
  for (const round of rounds) {
    const seenDocuments = new Set<number>();
    for (const item of round.work.roundItems) {
      if (item.annotation_round_id !== round.context.round.id || item.project_id !== round.context.project.id) throw new Error("The server returned paper items from a different annotation round.");
      const document = sourceDocument(round, item);
      if (seenDocuments.has(document.id)) throw new Error("The saved source contains duplicate records for one paper. Choose a collection with one record per paper.");
      seenDocuments.add(document.id);
      const existing = documents.get(document.id);
      if (existing && (existing.text !== document.text || existing.metadata_.dataset_item_id !== document.metadata_.dataset_item_id)) throw new Error("These rounds refer to different saved versions of the same paper. Open them separately.");
      documents.set(document.id, document);
      assignments.push(projectedAssignment(round, item, userId, annotatorId));
      const latest = ownLatest(round, item.id, userId);
      if (!latest) continue;
      const type = paperRoundAnnotationType(round.context)!;
      if (type === "doc_label" && typeof latest.output !== "string") throw new Error("This classification decision does not contain a single label.");
      const outputs = type === "doc_label" ? [{ label: latest.output as string, start: null, end: null }] : entitiesOf(latest.output);
      outputs.forEach((output, index) => {
        if (typeof output.label !== "string" || (type === "entity" && (!Number.isInteger(output.start) || !Number.isInteger(output.end) || output.start === null || output.end === null || output.start < 0 || output.end <= output.start || output.end > document.text.length))) throw new Error("This saved entity decision contains invalid offsets for the pinned paper text.");
        annotations.push({
        id: annotationId(item.id, index), project_id: round.context.project.id, document_id: document.id, annotation_type: type,
        label: output.label, start_offset: output.start, end_offset: output.end,
        text_span: output.start === null || output.end === null ? null : document.text.slice(output.start, output.end),
        source: "human", status: submissionFor(round, latest, userId) ? "accepted" : "draft", confidence: null,
        annotator_user_id: userId, annotator_id: annotatorId, model_checkpoint_id: null, guideline_version_id: null,
        structure_version_id: null, head_annotation_id: null, tail_annotation_id: null,
        evidence: {}, attributes: { round_item_id: item.id, round_id: round.context.round.id, decision_id: latest.id, entity_index: index, dataset_item_id: item.dataset_item_id },
        revision: latest.id, created_at: latest.created_at ?? "", updated_at: latest.created_at ?? "",
        });
      });
    }
  }
  return { documents: [...documents.values()], assignments, annotations, tasks };
}

/** All mutations append canonical decisions; projected legacy-shaped records never enter legacy storage. */
export function createPaperRoundAnnotationApi(options: {
  getRounds: () => LoadedPaperRound[]; setRounds: (rounds: LoadedPaperRound[]) => void;
  getVisibleRounds: () => LoadedPaperRound[]; userId: number; annotatorId: string;
  annotationId: ReturnType<typeof createPaperAnnotationIds>;
}): PaperAnnotationApi {
  const { userId, annotatorId, annotationId } = options;
  const pending = new Map<number, Promise<unknown>>();
  const serial = <T,>(itemId: number, action: () => Promise<T>): Promise<T> => {
    const next = (pending.get(itemId) ?? Promise.resolve()).catch(() => undefined).then(action);
    pending.set(itemId, next);
    void next.finally(() => { if (pending.get(itemId) === next) pending.delete(itemId); }).catch(() => undefined);
    return next;
  };
  const locate = (itemId: number): { round: LoadedPaperRound; item: RoundWorkItemIdentity } => {
    for (const round of options.getRounds()) {
      const item = round.work.roundItems.find((candidate) => candidate.id === itemId);
      if (item) return { round, item };
    }
    throw new Error("This paper task is not in the current saved annotation round.");
  };
  const findTask = (documentId: number, type: PaperTaskType) => {
    const matches = options.getVisibleRounds().filter((round) => paperRoundAnnotationType(round.context) === type).flatMap((round) => round.work.roundItems.filter((item) => sourceDocument(round, item).id === documentId).map((item) => ({ round, item })));
    if (matches.length !== 1) throw new Error("Choose one annotation task for this paper before saving.");
    return matches[0];
  };
  const projection = () => projectPaperRoundWork(options.getVisibleRounds(), userId, annotatorId, annotationId);
  const findAnnotation = (id: number) => {
    const annotation = projection().annotations.find((candidate) => candidate.id === id);
    if (!annotation) throw new Error("This annotation changed. Refresh the paper before editing it.");
    return annotation;
  };
  const save = async (itemId: number, output: unknown, reopen = false): Promise<AnnotationDecision> => {
    const { round, item } = locate(itemId);
    const latest = ownLatest(round, item.id, userId);
    if (!reopen && submissionFor(round, latest, userId)) throw new Error("Reopen this submitted paper task before changing its annotations.");
    const decision = await addRoundDecision(round.context.project.id, { roundItemId: item.id, output,
      supersedesDecisionId: latest?.id ?? null, decisionKind: "annotation", isInitialCheckpoint: false, rationale: "" });
    options.setRounds(options.getRounds().map((current) => current.context.round.id === round.context.round.id ? { ...current, work: { ...current.work, decisions: [...current.work.decisions, decision] } } : current));
    return decision;
  };
  const validateEntity = (entity: EntityOutput, document: Document, round: LoadedPaperRound) => {
    if (!Number.isInteger(entity.start) || !Number.isInteger(entity.end) || entity.start < 0 || entity.end <= entity.start || entity.end > document.text.length || !entity.label.trim()) throw new Error("Select a valid span in this paper and choose an entity label.");
    const labels = taskLabels(round.context);
    if (labels.length && !labels.includes(entity.label)) throw new Error("Choose an entity label configured for this task.");
  };
  const validateProject = (projectId: number, round: LoadedPaperRound) => {
    if (projectId !== round.context.project.id) throw new Error("This annotation belongs to another project.");
  };
  const setDocumentLabel = async (documentId: number, label: string): Promise<Annotation> => {
    const { item } = findTask(documentId, "doc_label");
    return serial(item.id, async () => {
      const { round } = locate(item.id);
      const values = taskLabels(round.context);
      if (!label.trim() || (values.length && !values.includes(label))) throw new Error("Choose a classification label configured for this task.");
      const latest = ownLatest(round, item.id, userId);
      if (latest?.output !== label) await save(item.id, label);
      return projection().annotations.find((annotation) => annotation.id === annotationId(item.id, 0))!;
    });
  };
  const createAnnotation = async (payload: AnnotationCreate): Promise<Annotation> => {
    if (payload.annotation_type !== "entity" && payload.annotation_type !== "doc_label") throw new Error("This round supports classification and named entity annotations only.");
    const { round, item } = findTask(payload.document_id, payload.annotation_type);
    validateProject(payload.project_id, round);
    if (payload.annotation_type === "doc_label") return setDocumentLabel(payload.document_id, payload.label);
    return serial(item.id, async () => {
      const current = locate(item.id);
      const entity = { start: payload.start_offset ?? -1, end: payload.end_offset ?? -1, label: payload.label };
      validateEntity(entity, sourceDocument(current.round, current.item), current.round);
      const entities = [...entitiesOf(ownLatest(current.round, item.id, userId)?.output), entity];
      await save(item.id, { entities });
      return findAnnotation(annotationId(item.id, entities.length - 1));
    });
  };
  const updateAnnotation: PaperAnnotationApi["updateAnnotation"] = async (id, payload) => {
    const before = findAnnotation(id);
    if (before.annotation_type === "doc_label") return setDocumentLabel(before.document_id, payload.label ?? before.label);
    const itemId = Number(before.attributes.round_item_id);
    return serial(itemId, async () => {
      const currentAnnotation = findAnnotation(id);
      if (currentAnnotation.revision !== before.revision) throw new Error("This annotation changed. Refresh the paper before editing it.");
      const { round, item } = locate(itemId);
      const entities = [...entitiesOf(ownLatest(round, itemId, userId)?.output)];
      const index = Number(currentAnnotation.attributes.entity_index);
      const entity = { start: payload.start_offset ?? entities[index].start, end: payload.end_offset ?? entities[index].end, label: payload.label ?? entities[index].label };
      validateEntity(entity, sourceDocument(round, item), round);
      entities[index] = entity;
      await save(itemId, { entities });
      return findAnnotation(id);
    });
  };
  const deleteAnnotation: PaperAnnotationApi["deleteAnnotation"] = async (id) => {
    const before = findAnnotation(id);
    if (before.annotation_type !== "entity") throw new Error("Choose a replacement classification label instead of deleting the decision.");
    const itemId = Number(before.attributes.round_item_id);
    await serial(itemId, async () => {
      const currentAnnotation = findAnnotation(id);
      if (currentAnnotation.revision !== before.revision) throw new Error("This annotation changed. Refresh the paper before editing it.");
      const { round } = locate(itemId);
      const entities = entitiesOf(ownLatest(round, itemId, userId)?.output).filter((_, index) => index !== Number(currentAnnotation.attributes.entity_index));
      await save(itemId, { entities });
    });
  };
  const createSubmission: PaperAnnotationApi["createSubmission"] = async (projectId, documentId, payload) => {
    if (!payload.assignment_id) throw new Error("Choose the paper task to submit.");
    return serial(payload.assignment_id, async () => {
      let { round, item } = locate(payload.assignment_id!);
      validateProject(projectId, round);
      if (sourceDocument(round, item).id !== documentId) throw new Error("The selected paper does not match this annotation task.");
      let latest = ownLatest(round, item.id, userId);
      if (!latest && paperRoundAnnotationType(round.context) === "entity") {
        latest = await save(item.id, { entities: [] });
        ({ round, item } = locate(item.id));
      }
      if (!latest) throw new Error("Choose a classification label before submitting this paper.");
      const submission = submissionFor(round, latest, userId) ?? await submitRoundDecisions(projectId, round.context.round.id, [latest.id]);
      options.setRounds(options.getRounds().map((current) => current.context.round.id === round.context.round.id && !current.work.submissions.some((existing) => existing.id === submission.id) ? { ...current, work: { ...current.work, submissions: [...current.work.submissions, submission] } } : current));
      const result: AnnotationSubmission = { id: submission.id, project_id: projectId, document_id: documentId, annotator_user_id: userId, annotator_id: annotatorId,
        kind: "submission", storage_key: "", file_name: "", content_type: "application/json", size_bytes: 0, checksum_sha256: submission.content_hash,
        annotation_count: paperRoundAnnotationType(round.context) === "entity" ? entitiesOf(latest.output).length : 1,
        metadata_: { round_id: round.context.round.id, round_item_id: item.id, decision_ids: [latest.id], dataset_item_id: item.dataset_item_id }, created_at: submission.submitted_at, updated_at: submission.submitted_at };
      return result;
    });
  };
  const reopenPersonalTaskAssignment: PaperAnnotationApi["reopenPersonalTaskAssignment"] = async (projectId, itemId) => serial(itemId, async () => {
    const { round } = locate(itemId);
    validateProject(projectId, round);
    const latest = ownLatest(round, itemId, userId);
    if (!latest || !submissionFor(round, latest, userId)) throw new Error("Only a submitted paper task can be reopened.");
    await save(itemId, latest.output, true);
    const current = locate(itemId);
    return projectedAssignment(current.round, current.item, userId, annotatorId);
  });
  return { createAnnotation, updateAnnotation, deleteAnnotation, createSubmission, reopenPersonalTaskAssignment, setDocumentLabel };
}

export default function PaperRoundWorkspace({ context, contexts, currentUserId, annotatorId, onClose, onOpenProjectTasks, canManage = false, defaultView = "annotate", overviewControls, taskInventory }: {
  context: RoundWorkContext; contexts: RoundWorkContext[]; currentUserId: number; annotatorId: string;
  onClose: () => void; onOpenProjectTasks: () => void; canManage?: boolean;
  defaultView?: "progress" | "annotate";
  overviewControls?: React.ReactNode;
  taskInventory?: React.ReactNode;
}): React.ReactElement {
  const selectedContexts = selectPaperRoundContexts(context, contexts);
  const loadKey = selectedContexts.map((item) => `${item.round.id}:${item.task_version.content_hash}`).join(",");
  const selectedContextsRef = useRef(selectedContexts);
  selectedContextsRef.current = selectedContexts;
  const [rounds, setRounds] = useState<LoadedPaperRound[]>([]);
  const roundsRef = useRef(rounds);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState<number | null>(null);
  const [genericRound, setGenericRound] = useState(false);
  const [searchParams, updateSearch] = useSearchState();
  const requestedDocumentRef = useRef(Number(searchParams.get("document")));
  requestedDocumentRef.current = Number(searchParams.get("document"));
  const classificationTaskVersionId = Number(searchParams.get("classificationTaskVersionId"));
  const entityTaskVersionId = Number(searchParams.get("entityTaskVersionId"));
  const annotationId = useMemo(createPaperAnnotationIds, [context.round.id]);
  const revision = useRef(0);
  const updateRounds = useCallback((next: LoadedPaperRound[]) => { revision.current += 1; roundsRef.current = next; setRounds(next); }, []);
  const reload = useCallback(async () => {
    const startedAt = ++revision.current;
    try {
      if (!selectedContextsRef.current.length) throw new Error("This round cannot use the paper editor. It needs a supported paper task and an unassisted open round.");
      const loaded = await Promise.all(selectedContextsRef.current.map(async (item) => ({ context: item, work: await loadRoundWork(item.project.id, item.round) })));
      const primary = loaded.find((item) => item.context.round.id === context.round.id)!;
      let externalSource = false;
      for (const item of primary.work.roundItems) {
        const source = primary.work.datasetItems.find((candidate) => candidate.id === item.dataset_item_id);
        if (!source || item.project_id !== context.project.id || item.annotation_round_id !== context.round.id || source.project_id !== context.project.id || source.dataset_version_id !== context.round.dataset_version_id) throw new Error("The server returned source records from a different project or dataset version.");
        if (!Number.isInteger(source.payload.document_id) || Number(source.payload.document_id) <= 0 || typeof source.payload.text !== "string") externalSource = true;
      }
      if (externalSource) {
        if (startedAt === revision.current) { setGenericRound(true); setError(null); }
        return;
      }
      const loadedProjection = projectPaperRoundWork(loaded, currentUserId, annotatorId, annotationId);
      if (startedAt === revision.current) {
        const requestedDocument = loadedProjection.documents.find((document) => document.id === requestedDocumentRef.current);
        setSelectedDocumentId((current) => requestedDocument?.id ?? (loadedProjection.documents.some((document) => document.id === current) ? current : loadedProjection.documents[0]?.id ?? null));
        roundsRef.current = loaded; setRounds(loaded); setGenericRound(false); setError(null);
      }
    } catch (caught) { if (startedAt === revision.current) setError(caught instanceof Error ? caught.message : "Could not load saved paper annotations."); }
    finally { if (startedAt === revision.current) setLoading(false); }
  }, [annotationId, annotatorId, currentUserId, context.project.id, context.round.id, context.round.dataset_version_id]);
  useEffect(() => { setLoading(true); setRounds([]); roundsRef.current = []; setGenericRound(false); void reload(); return () => { revision.current += 1; }; }, [loadKey, reload]);

  // The paper editor has one panel per annotation family. Extra tasks remain explicitly selectable.
  const visibleRounds = useMemo(() => (["doc_label", "entity"] as const).flatMap((type) => {
    const available = rounds.filter((round) => paperRoundAnnotationType(round.context) === type);
    const selectedId = type === "doc_label" ? classificationTaskVersionId : entityTaskVersionId;
    const selected = available.find((round) => round.context.task_version.id === selectedId) ?? available[0];
    return selected ? [selected] : [];
  }).sort((left, right) => Number(right.context.round.id === context.round.id) - Number(left.context.round.id === context.round.id)), [rounds, classificationTaskVersionId, entityTaskVersionId, context.round.id]);
  const visibleRef = useRef(visibleRounds);
  visibleRef.current = visibleRounds;
  const projection = useMemo(() => projectPaperRoundWork(visibleRounds, currentUserId, annotatorId, annotationId), [visibleRounds, currentUserId, annotatorId, annotationId]);
  const project: Project = useMemo(() => ({ id: context.project.id, name: context.project.name, description: null, workspace_id: null,
    tasks: projection.tasks, annotation_schema: { labels: Object.fromEntries(projection.tasks.map((task) => [task.annotation_type, task.labels])) },
    annotation_validation_mode: projection.tasks.every((task) => task.labels.length) ? "strict" : "relaxed", settings: {} }), [context.project.id, context.project.name, projection.tasks]);
  const workbench = useMemo<AnnotationWorkbench | null>(() => {
    const document = projection.documents.find((item) => item.id === selectedDocumentId) ?? projection.documents[0];
    if (!document) return null;
    return { project, document, tasks: projection.tasks, annotation_type_specs: projection.tasks.map((task) => task.annotation_type_spec),
      annotations: projection.annotations.filter((item) => item.document_id === document.id), assignments: projection.assignments.filter((item) => item.document_id === document.id),
      active_guideline: null, guideline_versions_by_id: {}, correction_locked_annotation_ids: [] };
  }, [project, projection, selectedDocumentId]);
  useEffect(() => { if (workbench && workbench.document.id !== selectedDocumentId) setSelectedDocumentId(workbench.document.id); }, [workbench, selectedDocumentId]);
  const annotationApi = useMemo(() => createPaperRoundAnnotationApi({ getRounds: () => roundsRef.current, setRounds: updateRounds,
    getVisibleRounds: () => visibleRef.current.map((visible) => roundsRef.current.find((round) => round.context.round.id === visible.context.round.id)!).filter(Boolean),
    userId: currentUserId, annotatorId, annotationId }), [annotationId, annotatorId, currentUserId, updateRounds]);

  if (genericRound && !loading) return <RoundWorkbench round={context.round} task={context.task_version} currentUserId={currentUserId} canManage={canManage} onClose={onClose} onRefresh={reload} />;

  return <div className="paper-round-workspace">
    {error ? <div role="alert" className="platform-error">{error}<button type="button" onClick={() => { setLoading(true); void reload(); }}>Retry loading papers</button></div> : null}
    {loading ? <p role="status">Loading saved papers and task decisions…</p> : null}
    {!loading && rounds.length > 0 ? <>
      <div className="platform-form-grid">{(["doc_label", "entity"] as const).map((type) => {
        const available = rounds.filter((round) => paperRoundAnnotationType(round.context) === type);
        return available.length > 1 ? <label key={type}>{type === "entity" ? "Entity task" : "Classification task"}<select disabled={busy} value={visibleRounds.find((round) => paperRoundAnnotationType(round.context) === type)?.context.task_version.id} onChange={(event) => updateSearch({ [type === "doc_label" ? "classificationTaskVersionId" : "entityTaskVersionId"]: Number(event.target.value), assignment: null })}>{available.map((round) => <option key={round.context.round.id} value={round.context.task_version.id}>{round.context.task.name} — {round.context.round.name}</option>)}</select></label> : null;
      })}</div>
      <AnnotatorWorkspace projects={[project]} selectedProject={project} selectedProjectId={project.id} setSelectedProjectId={() => undefined}
        documents={projection.documents} assignments={projection.assignments} annotatorId={annotatorId} projectProgress={null}
        selectedDocumentId={selectedDocumentId} setSelectedDocumentId={setSelectedDocumentId} workbench={workbench} setWorkbench={() => undefined}
        busy={busy} setBusy={setBusy} setError={setError} refreshProjectData={reload} refreshWorkbench={async () => undefined}
        allowAssignmentlessSubmit annotationApi={annotationApi} defaultView={defaultView} onOpenProjectTab={onOpenProjectTasks}
        overviewControls={overviewControls ?? <p>{context.project.name}</p>} taskInventory={taskInventory}
        onBackToMyWork={defaultView === "annotate" ? onClose : undefined} canManageTasks={canManage} />
    </> : null}
  </div>;
}
