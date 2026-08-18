export type AnnotationType =
  | "entity"
  | "relation"
  | "doc_label"
  | "sentence_label"
  | "passage_label"
  | "evidence_block";

export type AnnotationSource = "human" | "model" | "llm";
export type AnnotationStatus = "draft" | "accepted" | "rejected" | "gold";
export type AnnotationValidationMode = "strict" | "relaxed";
export type TaskAssignmentStatus =
  | "assigned"
  | "in_progress"
  | "submitted"
  | "adjudication_ready"
  | "adjudicated"
  | "completed"
  | "blocked"
  | "withdrawn";

export interface LabelDef {
  name: string;
  color: string;
  description: string | null;
}

export interface RelationConstraint {
  head: string[];
  tail: string[];
}

export type RelationConstraintMap = Record<string, RelationConstraint>;

export interface LabelSet {
  labels: Partial<Record<AnnotationType, LabelDef[]>>;
}

export interface ProjectTask {
  id: number;
  project_id: number;
  annotation_type: AnnotationType;
  display_name: string;
  description: string | null;
  enabled: boolean;
  sort_order: number;
  labels: LabelDef[];
  settings: Record<string, unknown>;
}

export interface EvidenceBlockTaskSettingsV1 {
  schema_version: "1";
  active_target_ids: number[];
  sentence_boundaries: boolean;
  multi_paragraph_allowed: boolean;
  cross_section_allowed: boolean;
  same_target_overlap_allowed: boolean;
  adjacency_allowed: boolean;
  soft_token_warning: number;
  model_context_tokens: number;
  window_overlap_tokens: number;
  review_scope: "document";
  keyboard_shortcuts: {
    create: string;
    expand_start: string;
    expand_end: string;
    contract_start: string;
    contract_end: string;
    merge: string;
    split: string;
    delete: string;
    mark_reviewed: string;
    cancel: string;
  };
}

export interface ProjectTaskCreate {
  annotation_type: AnnotationType;
  display_name?: string | null;
  description?: string | null;
  enabled?: boolean;
  sort_order?: number;
  labels?: LabelDef[];
  settings?: Record<string, unknown>;
}

export interface ProjectTaskUpdate {
  display_name?: string | null;
  description?: string | null;
  enabled?: boolean;
  sort_order?: number;
  labels?: LabelDef[];
  settings?: Record<string, unknown>;
}

export interface Project {
  id: number;
  name: string;
  description: string | null;
  annotation_schema: LabelSet;
  annotation_validation_mode: AnnotationValidationMode;
  tasks: ProjectTask[];
  settings: Record<string, unknown>;
  workspace_id: number | null;
}

export interface ProjectCreate {
  name: string;
  description?: string | null;
  annotation_schema?: LabelSet;
  annotation_validation_mode?: AnnotationValidationMode;
  tasks?: ProjectTaskCreate[];
  settings?: Record<string, unknown>;
  workspace_id?: number | null;
}

export interface ProjectUpdate {
  name?: string;
  description?: string | null;
  annotation_validation_mode?: AnnotationValidationMode;
  settings?: Record<string, unknown>;
}

export interface Document {
  id: number;
  project_id: number;
  external_id: string | null;
  title: string | null;
  text: string;
  source: string | null;
  metadata_: Record<string, unknown>;
  sentences: number[][];
  active_structure_version_id: number | null;
}

export type DocumentStructureStatus = "pending" | "ready" | "failed" | "superseded";

export interface DocumentStructureVersion {
  id: number;
  document_id: number;
  version: number;
  segmenter_name: string;
  segmenter_version: string;
  source_hash: string;
  text_length: number;
  status: DocumentStructureStatus;
  created_at: string;
}

export interface DocumentStructureSection {
  id: number;
  ordinal: number;
  title: string | null;
  path: string[];
  kind: string;
  start_offset: number;
  end_offset: number;
  locator: Record<string, unknown> | null;
}

export interface DocumentStructureParagraph {
  id: number;
  section_id: number | null;
  ordinal: number;
  section_ordinal: number;
  start_offset: number;
  end_offset: number;
  locator: Record<string, unknown> | null;
}

export interface DocumentStructureSentence {
  id: number;
  section_id: number | null;
  paragraph_id: number;
  ordinal: number;
  paragraph_ordinal: number;
  start_offset: number;
  end_offset: number;
  text: string;
}

export interface DocumentStructureRead {
  document_id: number;
  active_structure_version_id: number | null;
  structure_version: DocumentStructureVersion;
  range: {
    start_ordinal: number;
    end_ordinal: number;
    total_sentences: number;
    has_more: boolean;
  };
  sections: DocumentStructureSection[];
  paragraphs: DocumentStructureParagraph[];
  sentences: DocumentStructureSentence[];
}

export interface GuidelineVersion {
  id: number;
  project_id: number;
  version_label: string;
  markdown: string;
  author_id: string | null;
  status: string;
}

export interface GuidelineVersionCreate {
  project_id: number;
  version_label?: string;
  markdown?: string;
  author_id?: string | null;
  status?: string;
}

export interface AnnotationTypeSpec {
  name: AnnotationType;
  requires_span: boolean;
  requires_head_tail: boolean;
  description: string;
  selection_mode: string;
  renderer_key: string;
  relation_endpoint_allowed: boolean;
  handler_key: string;
}

export interface AnnotationWorkbenchTask extends ProjectTask {
  annotation_type_spec: AnnotationTypeSpec;
}

export interface Annotation {
  id: number;
  project_id: number;
  document_id: number;
  annotation_type: AnnotationType;
  label: string;
  start_offset: number | null;
  end_offset: number | null;
  text_span: string | null;
  source: AnnotationSource;
  status: AnnotationStatus;
  confidence: number | null;
  annotator_user_id: number | null;
  annotator_id: string | null;
  model_checkpoint_id: string | null;
  guideline_version_id: number | null;
  structure_version_id: number | null;
  head_annotation_id: number | null;
  tail_annotation_id: number | null;
  evidence: Record<string, unknown>;
  attributes: Record<string, unknown>;
  revision?: number;
  evidence_block?: EvidenceBlockPayloadV1 | null;
  created_at: string;
  updated_at: string;
}

export interface TaskAssignment {
  id: number;
  project_id: number;
  task_id: number;
  document_id: number;
  assignee_user_id: number;
  annotator_id: string;
  status: TaskAssignmentStatus;
  assigned_by_user_id: number | null;
  assigned_by: string | null;
  notes: string | null;
  metadata_: Record<string, unknown>;
  target_version_id: number | null;
  structure_version_id: number | null;
  guideline_version_id: number | null;
  assignment_scope_key: string;
}

export interface TaskAssignmentCreate {
  task_id: number;
  document_id: number;
  assignee_user_id?: number | null;
  annotator_id?: string | null;
  status?: TaskAssignmentStatus;
  assigned_by_user_id?: number | null;
  assigned_by?: string | null;
  notes?: string | null;
  metadata_?: Record<string, unknown>;
  target_version_id?: number | null;
  structure_version_id?: number | null;
  guideline_version_id?: number | null;
}

export interface TaskAssignmentUpdate {
  assignee_user_id?: number | null;
  annotator_id?: string | null;
  status?: TaskAssignmentStatus;
  notes?: string | null;
  metadata_?: Record<string, unknown>;
}

export interface EvidenceTargetVersion {
  id: number;
  target_id: number;
  version_number: number;
  text: string;
  guidance: string | null;
  inclusion_guidance: string | null;
  exclusion_guidance: string | null;
  metadata_: Record<string, unknown>;
  created_by_user_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceTarget {
  id: number;
  project_id: number;
  task_id: number;
  key: string;
  name: string;
  description: string | null;
  is_active: boolean;
  active_version_id: number | null;
  versions: EvidenceTargetVersion[];
  created_by_user_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface EvidenceTargetVersionCreate {
  text: string;
  guidance?: string | null;
  inclusion_guidance?: string | null;
  exclusion_guidance?: string | null;
  metadata_?: Record<string, unknown>;
}

export interface EvidenceTargetCreate {
  task_id: number;
  key: string;
  name: string;
  description?: string | null;
  initial_version: EvidenceTargetVersionCreate;
}

export interface EvidenceBlockPayloadV1 {
  annotation_id?: number;
  structure_version_id: number;
  target_version_id: number;
  start_sentence_id: number;
  end_sentence_id: number;
  start_sentence_ordinal?: number;
  end_sentence_ordinal?: number;
  start_offset?: number;
  end_offset?: number;
  labels: string[];
  note: string | null;
  boundary_policy: "sentence";
  revision?: number;
  locked?: boolean;
  last_command_group_key?: string | null;
  created_at?: string;
  updated_at?: string;
}

export interface EvidenceBlockWrite {
  structure_version_id: number;
  target_version_id: number;
  start_sentence_id: number;
  end_sentence_id: number;
  labels?: string[];
  note?: string | null;
  boundary_policy?: "sentence";
}

export interface EvidenceReviewInterval {
  id: number;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  start_sentence_id: number;
  end_sentence_id: number;
  created_at: string;
  updated_at: string;
}

export interface EvidenceReviewEvent {
  id: number;
  action: "mark_reviewed" | "reopen";
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  actor_user_id?: number | null;
  start_sentence_id: number;
  end_sentence_id: number;
  reason: string | null;
  metadata_: Record<string, unknown>;
  created_at: string;
}

export interface EvidenceReviewCoverage {
  project_id: number;
  document_id: number;
  target_version_id: number;
  structure_version_id: number;
  guideline_version_id: number | null;
  reviewer_user_id: number;
  intervals: EvidenceReviewInterval[];
  events: EvidenceReviewEvent[];
  fully_reviewed: boolean;
}

export interface EvidenceReviewIntervalWrite {
  target_version_id: number;
  structure_version_id: number;
  guideline_version_id?: number | null;
  start_sentence_id: number;
  end_sentence_id: number;
  reason?: string | null;
}

export interface EvidenceMergeRequest {
  annotation_ids: number[];
  expected_revisions: Record<number, number>;
  labels?: string[] | null;
  note?: string | null;
  boundary_policy?: "sentence";
}

export interface EvidenceSplitRequest {
  expected_revision: number;
  split_before_sentence_id: number;
}

export interface EvidenceCommandSummary {
  command_group_key: string;
  operation: string;
  status: "applied" | "undone";
  project_id: number;
  document_id: number;
  target_version_id: number;
  structure_version_id: number;
  guideline_version_id: number | null;
  actor_user_id: number | null;
  created_at: string;
}

export interface EvidenceCommandResult {
  command_group_key: string;
  status: "applied" | "undone";
  annotations: Annotation[];
}

export interface EvidenceComparisonBlock {
  annotation_id: number;
  annotator_user_id: number | null;
  annotator_id: string | null;
  status: AnnotationStatus;
  start_sentence_id: number;
  end_sentence_id: number;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  labels: string[];
  note: string | null;
}

export interface EvidenceAdjudicationRead {
  project_id: number;
  document_id: number;
  target_version_id: number;
  structure_version_id: number;
  guideline_version_id: number;
  blocks: EvidenceComparisonBlock[];
}

export interface EvidenceAdjudicationCreate {
  target_version_id: number;
  structure_version_id: number;
  guideline_version_id: number;
  strategy: "a" | "b" | "union" | "intersection" | "custom";
  source_annotation_ids?: number[];
  start_sentence_id?: number | null;
  end_sentence_id?: number | null;
  labels?: string[];
  note?: string | null;
  solo_gold?: boolean;
}

export interface EvidencePairIaa {
  left_annotator_id: string;
  right_annotator_id: string;
  reviewed_sentence_count: number;
  left_block_count: number;
  right_block_count: number;
  sentence_precision: number;
  sentence_recall: number;
  sentence_f1: number;
  exact_f1: number;
  iou_f1: Record<string, number>;
  coverage: number;
  overreach: number;
  mean_start_boundary_deviation: number | null;
  mean_end_boundary_deviation: number | null;
  document_presence_agreement: boolean;
}

export interface EvidenceIaaMetrics {
  target_version_id: number;
  structure_version_id: number | null;
  guideline_version_id: number | null;
  pairs: EvidencePairIaa[];
  aggregate: Record<string, number | boolean | null>;
}

export interface IaaReport {
  project_id: number;
  annotation_type: AnnotationType;
  document_id: number | null;
  status: "ok" | "insufficient_annotators" | "no_items";
  annotator_ids: string[];
  item_count: number;
  percent_agreement: number | null;
  cohens_kappa: number | null;
  fleiss_kappa: number | null;
  span_detection_f1: number | null;
  evidence_metrics: EvidenceIaaMetrics | null;
}

export interface StatusCount {
  total: number;
  by_status: Record<string, number>;
}

export interface TaskProgress extends StatusCount {
  task_id: number;
  annotation_type: AnnotationType;
  display_name: string;
}

export interface DocumentProgress extends StatusCount {
  document_id: number;
}

export interface AnnotatorProgress extends StatusCount {
  assignee_user_id: number | null;
  annotator_id: string;
}

export interface ProjectProgress extends StatusCount {
  project_id: number;
  by_task: TaskProgress[];
  by_document: DocumentProgress[];
  by_annotator: AnnotatorProgress[];
}

export interface AnnotationWorkbench {
  project: Project;
  document: Document;
  active_guideline: GuidelineVersion | null;
  guideline_versions_by_id: Record<number, GuidelineVersion>;
  tasks: AnnotationWorkbenchTask[];
  annotation_type_specs: AnnotationTypeSpec[];
  annotations: Annotation[];
  assignments: TaskAssignment[];
  correction_locked_annotation_ids: number[];
}

export interface AnnotationCreate {
  project_id: number;
  document_id: number;
  annotation_type: AnnotationType;
  label: string;
  start_offset?: number | null;
  end_offset?: number | null;
  text_span?: string | null;
  source?: AnnotationSource;
  status?: AnnotationStatus;
  confidence?: number | null;
  annotator_user_id?: number | null;
  annotator_id?: string | null;
  model_checkpoint_id?: string | null;
  guideline_version_id?: number | null;
  structure_version_id?: number | null;
  head_annotation_id?: number | null;
  tail_annotation_id?: number | null;
  evidence?: Record<string, unknown>;
  attributes?: Record<string, unknown>;
  evidence_block?: EvidenceBlockWrite;
}

export type ImportStatus = "full_text" | "abstract_only" | "not_found" | "error";

export interface ImportPreviewItem {
  pmid: string;
  title: string;
  journal: string;
  year: string;
  pmcid: string | null;
  status: ImportStatus;
  has_full_text: boolean;
  has_abstract: boolean;
}

export interface ImportPreviewResponse {
  items: ImportPreviewItem[];
}

export interface ImportOutcome {
  pmid: string;
  status: ImportStatus;
  title: string;
  document_id: number | null;
  reason: string | null;
}

export interface ImportResponse {
  created: ImportOutcome[];
  skipped: ImportOutcome[];
}

export interface AnnotationUpdate {
  label?: string | null;
  start_offset?: number | null;
  end_offset?: number | null;
  text_span?: string | null;
  status?: AnnotationStatus | null;
  confidence?: number | null;
  head_annotation_id?: number | null;
  tail_annotation_id?: number | null;
  evidence?: Record<string, unknown> | null;
  attributes?: Record<string, unknown> | null;
  expected_revision?: number;
  evidence_block?: EvidenceBlockWrite | null;
}

export type SubmissionKind = "submission" | "re_export";

export interface AnnotationSubmissionCreate {
  annotator_user_id?: number | null;
  annotator_id?: string | null;
  kind?: SubmissionKind;
  metadata_?: Record<string, unknown>;
  assignment_id?: number | null;
}

export interface AnnotationSubmission {
  id: number;
  project_id: number;
  document_id: number;
  annotator_user_id: number | null;
  annotator_id: string | null;
  kind: SubmissionKind;
  storage_key: string;
  file_name: string;
  content_type: string;
  size_bytes: number;
  checksum_sha256: string;
  annotation_count: number;
  metadata_: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export type DatasetSplit = "train" | "validation" | "test" | "pool";

export interface CorpusSnapshotDocumentCreate {
  document_id: number;
  structure_version_id: number;
  split?: DatasetSplit;
  group_key: string;
  source_hash: string;
  metadata_?: Record<string, unknown>;
}

export interface CorpusSnapshotCreate {
  name: string;
  split_strategy?: string;
  split_seed?: number;
  documents: CorpusSnapshotDocumentCreate[];
  metadata_?: Record<string, unknown>;
}

export interface CorpusSnapshot {
  id: number;
  project_id: number;
  artifact_id: number;
  name: string;
  split_strategy: string;
  split_seed: number;
  document_count: number;
  metadata_: Record<string, unknown>;
}

export interface AnnotationSetItemCreate {
  source_annotation_id?: number | null;
  document_id: number;
  structure_version_id: number;
  target_version_id: number;
  guideline_version_id?: number | null;
  start_sentence_id: number;
  end_sentence_id: number;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  start_char: number;
  end_char: number;
  block_text: string;
  section_paths?: string[][];
  labels?: string[];
  source?: "adjudicated" | "solo_gold";
  metadata_?: Record<string, unknown>;
}

export interface AnnotationSetReviewRegionCreate {
  document_id: number;
  structure_version_id: number;
  target_version_id: number;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  metadata_?: Record<string, unknown>;
}

export interface AnnotationSetCreate {
  name: string;
  corpus_snapshot_id: number;
  target_version_ids: number[];
  guideline_version_ids?: number[];
  items?: AnnotationSetItemCreate[];
  reviewed_regions?: AnnotationSetReviewRegionCreate[];
  metadata_?: Record<string, unknown>;
}

export interface AnnotationSet {
  id: number;
  project_id: number;
  artifact_id: number;
  corpus_snapshot_id: number;
  name: string;
  target_version_ids: number[];
  guideline_version_ids: number[];
  block_count: number;
  reviewed_region_count: number;
  metadata_: Record<string, unknown>;
}

export interface LineageArtifact {
  id: number;
  project_id: number;
  artifact_type: string;
  schema_version: string;
  content_hash: string;
  storage_key: string;
  content_type: string;
  size_bytes: number;
  manifest: Record<string, unknown>;
}

export interface LineageGraph {
  artifacts: LineageArtifact[];
  edges: Array<{
    id: number;
    upstream_artifact_id: number;
    downstream_artifact_id: number;
    relationship_type: string;
    metadata: Record<string, unknown>;
  }>;
}

export interface ExportCreate {
  format_key: string;
  annotation_set_id?: number | null;
  corpus_snapshot_id?: number | null;
  file_name?: string | null;
  options?: Record<string, unknown>;
  metadata_?: Record<string, unknown>;
}

export interface ExportArtifact {
  id: number;
  project_id: number;
  artifact_id: number;
  corpus_snapshot_id: number | null;
  annotation_set_id: number | null;
  format_key: string;
  file_name: string;
  row_count: number;
  metadata_: Record<string, unknown>;
}

export interface ExportFormat {
  key: string;
  content_type: string;
  extension: string;
}

export type ModelFamily = "conventional_ml" | "deep_learning" | "llm_finetune";

export interface ArtifactPackageFile {
  id: number;
  relative_path: string;
  role: string;
  content_type: string;
  size_bytes: number;
  checksum_sha256: string;
  downloadable: boolean;
}

export interface ArtifactPackageReference {
  relationship_type: string;
  target_package_id: number | null;
  manifest_digest: string;
}

export interface ArtifactPackage {
  id: number;
  project_id: number;
  kind: string;
  format: string;
  schema_version: string;
  model_family: ModelFamily | null;
  model_type: string | null;
  readiness: string;
  deployable: boolean;
  manifest_digest: string;
  logical_size_bytes: number;
  file_count: number;
  checkpoint_id?: number | null;
  training_job_id?: number | null;
  pinned?: boolean;
  champion?: boolean;
  retention_expires_at?: string | null;
  archived_at?: string | null;
  purge_after?: string | null;
  purged_at?: string | null;
  legal_hold?: boolean;
  created_at: string;
  files?: ArtifactPackageFile[];
  references?: ArtifactPackageReference[];
}

export type BaseModelAccessMode = "downloadable" | "execution_only" | "manager_only";
export type BaseModelReadiness = "ready" | "quarantined" | "failed" | "archived";

export interface BaseModelAsset {
  id: number;
  project_id: number;
  package_id: number;
  provider: string;
  source_model_id: string;
  exact_revision: string;
  display_name: string;
  model_family: ModelFamily;
  model_type: string;
  license_name: string;
  license_url: string | null;
  license_terms_sha256: string | null;
  access_mode: BaseModelAccessMode;
  readiness: BaseModelReadiness;
  archived_at: string | null;
  metadata: Record<string, unknown>;
  package: ArtifactPackage;
  created_at: string;
}

export interface BaseModelReadinessWrite {
  readiness: Exclude<BaseModelReadiness, "archived">;
  reason: string;
}

export interface BaseModelCatalogWrite {
  provider: string;
  source_model_id: string;
  exact_revision: string;
  display_name: string;
  model_family: ModelFamily;
  model_type: string;
  license_name: string;
  license_url?: string | null;
  license_terms_sha256?: string | null;
  access_mode: BaseModelAccessMode;
  metadata?: Record<string, unknown>;
}

export interface BaseModelImportWrite extends BaseModelCatalogWrite {
  source_package_id: number;
}

export interface BaseModelUploadFileWrite {
  relative_path: string;
  role?: string;
  content_type?: string;
  checksum_sha256?: string | null;
  size_bytes?: number | null;
  metadata?: Record<string, unknown>;
}

export interface BaseModelUploadWrite extends BaseModelCatalogWrite {
  package_format: string;
  task_contract?: Record<string, unknown>;
  runtime?: Record<string, unknown>;
  files: BaseModelUploadFileWrite[];
}

export interface InferenceWindowConfig {
  max_tokens: number;
  overlap_tokens: number;
  aggregation: "mean";
}

export interface InferenceDecoderConfig {
  version: "evidence-block-decoder-v1";
  block_threshold: number;
  allow_cross_section: boolean;
  merge_adjacent: false;
}

export interface InferenceRunCreate {
  name: string;
  corpus_snapshot_id: number;
  checkpoint_id: number;
  compute_profile_id: number;
  target_version_ids: number[];
  window_config?: InferenceWindowConfig;
  decoder_config?: InferenceDecoderConfig;
  idempotency_key: string;
}

export interface InferenceRun {
  id: number;
  project_id: number;
  corpus_snapshot_id: number;
  checkpoint_id: number;
  compute_profile_id: number;
  name: string;
  target_version_ids: number[];
  window_config: Record<string, unknown>;
  decoder_config: Record<string, unknown>;
  status: string;
  idempotency_key: string;
  external_job_id: string | null;
  diagnostics_artifact_id: number | null;
  started_at: string | null;
  completed_at: string | null;
  failure_reason: string | null;
  metrics: Record<string, unknown>;
}

export interface InferenceWindow {
  id: number;
  run_id: number;
  document_id: number;
  structure_version_id: number;
  target_version_id: number;
  stable_key: string;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  token_count: number;
  status: string;
  sentence_contribution_counts: Record<string, number>;
  diagnostics_artifact_id: number | null;
}

export type PredictionReviewAction = "accept" | "modify" | "reject";
export type PredictionStatus = "pending" | "accepted" | "modified" | "rejected";

export interface PredictionReviewCreate {
  action: PredictionReviewAction;
  start_sentence_id?: number | null;
  end_sentence_id?: number | null;
  labels?: string[];
  note?: string | null;
  metadata_?: Record<string, unknown>;
}

export interface PredictionReview {
  id: number;
  prediction_id: number;
  reviewer_user_id: number;
  action: PredictionReviewAction;
  revision: number;
  resulting_annotation_id: number | null;
  selected_boundaries: Record<string, number> | null;
  note: string | null;
  metadata_: Record<string, unknown>;
  created_at: string;
}

export interface EvidenceCandidatePrediction {
  id: number;
  project_id: number;
  run_id: number;
  checkpoint_id: number;
  document_id: number;
  structure_version_id: number;
  target_version_id: number;
  start_sentence_id: number;
  end_sentence_id: number;
  start_sentence_ordinal: number;
  end_sentence_ordinal: number;
  start_char: number;
  end_char: number;
  block_confidence: number;
  boundary_confidence: Record<string, number>;
  uncertainty: number;
  decoder_version: string;
  source_window_ids: number[];
  status: PredictionStatus;
  review_status: PredictionStatus;
  diagnostics_artifact_id: number | null;
  metadata_: Record<string, unknown>;
  reviews: PredictionReview[];
}

export type WorkspaceRole = "annotator" | "trainer" | "manager" | "admin";

export interface WorkspaceMember {
  id: number;
  workspace_id: number;
  user_id: number;
  username: string;
  display_name: string;
  email: string | null;
  is_active: boolean;
  role: WorkspaceRole;
}

export interface WorkspaceInvite {
  id: number;
  token: string;
  role: WorkspaceRole;
  workspace_id: number;
  expires_at: string | null;
}

export interface WorkspaceInviteSummary {
  id: number;
  workspace_id: number;
  role: WorkspaceRole;
  created_by: number;
  created_by_username: string;
  expires_at: string | null;
  created_at: string;
}

export interface InvitePreview {
  workspace_name: string;
  workspace_kind: string;
  role: WorkspaceRole;
  expires_at: string | null;
}

export interface InviteAcceptPayload {
  username?: string;
  password?: string;
  display_name?: string;
}

export interface WorkspaceJoinRequest {
  id: number;
  workspace_id: number;
  user_id: number;
  username: string;
  display_name: string;
  email: string | null;
  status: string;
  message: string | null;
  created_at: string;
}

export interface WorkspaceGovernance {
  workspace_id: number;
  workspace_kind: string;
  join_code: string | null;
  default_invite_expiry_minutes: number;
}

export interface AdminMembership {
  workspace_id: number;
  workspace_name: string;
  workspace_kind: string;
  role: WorkspaceRole;
}

export interface AdminUserSummary {
  id: number;
  username: string;
  display_name: string;
  email: string | null;
  is_active: boolean;
  is_initialized: boolean;
  is_superuser: boolean;
  last_login_at: string | null;
  membership_count: number;
  created_at?: string;
}

export interface AdminUserDetail extends AdminUserSummary {
  memberships: AdminMembership[];
}

export interface AdminUserListParams {
  search?: string;
  status?: "active" | "inactive" | "all";
  workspaceId?: number;
  page?: number;
  pageSize?: number;
}

export interface AdminUserCreate {
  username: string;
  display_name?: string;
  email?: string;
}

export interface AdminUserPage {
  items: AdminUserSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface AccountActionLink {
  url: string;
  expires_at: string;
  purpose: AccountActionPurpose;
}

export interface AdminUserCreateResult {
  user: AdminUserSummary;
  action: AccountActionLink;
}

export interface AdminSettings {
  allow_self_registration: boolean;
  default_invite_expiry_minutes: number;
  account_action_expiry_minutes: number;
  deployment_profile: string;
  storage_backend: string;
  storage_encryption: string;
  task_execution: string;
  jwt_lifetime_minutes: number;
}

export type AdminSettingsUpdate = Pick<
  AdminSettings,
  | "allow_self_registration"
  | "default_invite_expiry_minutes"
  | "account_action_expiry_minutes"
>;

export type AccountActionPurpose = "activation" | "password_reset";

export interface AccountActionPreview {
  purpose: AccountActionPurpose;
  username: string;
  display_name: string;
  expires_at: string;
}
