import type { BaseModelAsset, Project, WorkspaceMember } from "@/types/api";

export type TaskKind =
  | "regression"
  | "classification"
  | "multilabel_classification"
  | "token_labeling"
  | "span_extraction"
  | "relation_extraction"
  | "ranking"
  | "generation"
  | "instruction_tuning";

export type AssistancePolicy =
  | "blind"
  | "reveal_after_first_pass"
  | "immediate_suggestions"
  | "critique"
  | "micro_question";

export interface VersionedResource {
  id: number;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface TaskDefinition extends VersionedResource {
  project_id: number;
  key: string;
  name: string;
  description: string | null;
}

export interface TaskVersion extends VersionedResource {
  project_id: number;
  task_definition_id: number;
  version_number: number;
  task_kind: TaskKind;
  input_schema: Record<string, unknown>;
  output_schema: Record<string, unknown>;
  label_rules: Record<string, unknown>;
  annotation_ui: Record<string, unknown>;
  metrics: string[];
  trainer_compatibility: string[];
  content_hash: string;
}

export interface Dataset extends VersionedResource {
  project_id: number;
  name: string;
  description: string | null;
  source_type: "upload" | "public_registry" | "project_corpus" | "generated" | "other";
}

export interface DatasetVersion extends VersionedResource {
  project_id: number;
  dataset_id: number;
  version_number: number;
  source_uri: string | null;
  source_revision: string;
  source_format: string;
  data_schema: Record<string, unknown>;
  provenance: Record<string, unknown>;
  license_info: Record<string, unknown>;
  content_hash: string;
  item_count: number;
  artifact_package_id: number | null;
}

export interface DatasetItem extends VersionedResource {
  project_id: number;
  dataset_version_id: number;
  stable_key: string;
  group_key: string | null;
  payload: Record<string, unknown>;
  content_hash: string;
}

export interface LabelSetVersion extends VersionedResource {
  project_id: number;
  dataset_version_id: number;
  task_version_id: number;
  parent_version_id: number | null;
  name: string;
  source_kind: "imported" | "human" | "adjudicated" | "derived";
  composition_policy: "replace" | "inherit" | "exclude";
  labels: Record<string, unknown>;
  version_number: number;
  label_count: number;
  content_hash: string;
}

export interface SplitMap extends VersionedResource {
  project_id: number;
  dataset_version_id: number;
  name: string;
  strategy: string;
  seed: number;
  group_key_field: string | null;
  assignments: Record<string, "pool" | "train" | "validation" | "test">;
  protected_splits: string[];
  content_hash: string;
}

export interface TrainingDatasetVersion extends VersionedResource {
  project_id: number;
  name: string;
  dataset_version_id: number;
  task_version_id: number;
  label_set_version_ids: number[];
  split_map_id: number;
  composition: Record<string, unknown>[];
  preprocessing: Record<string, unknown>;
  content_hash: string;
}

export type LearningFeedbackProducer =
  | "registered_model"
  | "external_llm"
  | "rule"
  | "dictionary"
  | "ensemble"
  | "human_disagreement";

export interface LearningFeedbackSource {
  producer_type: LearningFeedbackProducer;
  name: string;
  provider: string | null;
  external_model_id: string | null;
  exact_revision: string | null;
  configuration: Record<string, unknown>;
  data_egress_policy: Record<string, unknown>;
}

export interface FeedbackRun extends VersionedResource {
  project_id: number;
  dataset_version_id: number;
  task_version_id: number;
  producer_type:
    | "registered_model"
    | "external_llm"
    | "rule"
    | "dictionary"
    | "ensemble"
    | "human_disagreement"
    | "none";
  cycle_id: number | null;
  model_version_id: number | null;
  provider: string | null;
  external_model_id: string | null;
  exact_revision: string | null;
  prompt_template_hash: string | null;
  configuration: Record<string, unknown>;
  data_egress_policy: Record<string, unknown>;
  status: "planned" | "queued" | "running" | "completed" | "failed";
  output_feedback_set_version_id: number | null;
  failure_code: string | null;
  failure_reason: string | null;
  started_at: string | null;
  heartbeat_at: string | null;
  completed_at: string | null;
}

export interface FeedbackSetVersion extends VersionedResource {
  project_id: number;
  feedback_run_id: number;
  dataset_version_id: number;
  task_version_id: number;
  version_number: number;
  output_schema: Record<string, unknown>;
  candidate_count: number;
  content_hash: string;
  artifact_package_id: number | null;
}

export interface LearningCycle extends VersionedResource {
  project_id: number;
  name: string;
  sequence: number;
  goal?: string;
  parent_cycle_id?: number | null;
  status: string;
  current_stage: string;
  task_version_id: number;
  source_dataset_version_id: number;
  baseline_model_version_id: number | null;
  feedback_sources: LearningFeedbackSource[];
  output_training_dataset_version_id: number | null;
  output_model_version_id: number | null;
  metadata: Record<string, unknown>;
}

export interface AnnotationRound extends VersionedResource {
  project_id: number;
  cycle_id: number | null;
  parent_round_id?: number | null;
  name: string;
  sequence: number;
  dataset_version_id: number;
  task_version_id: number;
  guideline_revision_id: number | null;
  selection_set_version_id: number | null;
  feedback_set_version_id: number | null;
  assistance_policy: AssistancePolicy;
  reannotation_mode: "targeted_subset" | "full_dataset";
  annotator_user_ids: number[];
  open_to_all_annotators: boolean;
  reason: string | null;
  status: string;
  opened_at: string | null;
  closed_at: string | null;
}

export interface RoundWorkRound extends VersionedResource {
  project_id: number;
  name: string;
  sequence: number;
  dataset_version_id: number;
  task_version_id: number;
  assistance_policy: AssistancePolicy;
  feedback_available: boolean;
  status: string;
  opened_at: string | null;
  closed_at: string | null;
}

export interface RoundWorkContext {
  project: {
    id: number;
    name: string;
  };
  round: RoundWorkRound;
  task: {
    id: number;
    key: string;
    name: string;
  };
  task_version: TaskVersion;
  cycle: {
    id: number;
    name: string;
    sequence: number;
  } | null;
  guideline: {
    guideline_id: number;
    guideline_revision_id: number;
    name: string;
    version_number: number;
    status: string;
  } | null;
}

export interface RoundItem extends VersionedResource {
  project_id: number;
  annotation_round_id: number;
  dataset_item_id: number;
  selection_rank: number | null;
  selection_score: number | null;
  selection_probability: number | null;
  selection_reason: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

export interface RoundWorkItemIdentity extends VersionedResource {
  project_id: number;
  annotation_round_id: number;
  dataset_item_id: number;
  selection_rank: number | null;
  selection_score: number | null;
  selection_reason: Record<string, unknown>;
}

export interface AnnotationDecision extends VersionedResource {
  project_id: number;
  round_item_id: number;
  supersedes_decision_id: number | null;
  output: unknown;
  decision_kind: "annotation" | "adjudication" | "correction";
  is_initial_checkpoint: boolean;
  rationale: string | null;
  annotator_user_id: number;
  content_hash: string;
}

export interface RoundSubmission extends VersionedResource {
  project_id: number;
  annotation_round_id: number;
  decision_ids: number[];
  annotator_user_id: number;
  sequence: number;
  content_hash: string;
  submitted_at: string;
}

export interface FeedbackCandidate extends VersionedResource {
  project_id: number;
  feedback_set_version_id: number;
  dataset_item_id: number;
  candidate_key: string;
  output: unknown;
  score: number | null;
  explanation: Record<string, unknown>;
  content_hash: string;
}

export interface FeedbackExposure extends VersionedResource {
  project_id: number;
  round_item_id: number;
  feedback_candidate_id: number;
  exposure_mode: AssistancePolicy;
  context: Record<string, unknown>;
  user_id: number;
  exposed_at: string;
}

export interface FeedbackReveal {
  exposure: FeedbackExposure;
  candidate: FeedbackCandidate;
}

export interface RegisteredModel extends VersionedResource {
  project_id: number;
  name: string;
  description: string | null;
  lifecycle_status: string;
  created_by_user_id?: number | null;
}

export interface ModelVersion extends VersionedResource {
  project_id: number;
  registered_model_id: number;
  version_number: number;
  parent_version_id: number | null;
  task_version_id: number;
  training_dataset_version_id: number | null;
  family: string;
  framework: string;
  base_model: Record<string, unknown>;
  training_method: string;
  recipe_key: string;
  recipe_version: string;
  parameters: Record<string, unknown>;
  metrics: Record<string, unknown>;
  runtime_digest: string;
  content_hash: string;
  code_digest?: string;
  seed?: number;
  checkpoint_package_id?: number | null;
  created_by_user_id?: number | null;
  training_run_id?: number | null;
  source_dataset_version_id?: number | null;
  source_dataset_version_number?: number | null;
  runtime_id?: number | null;
  runtime_name?: string | null;
  storage_policy_id?: number | null;
  storage_policy_name?: string | null;
  creator_username?: string | null;
  creator_display_name?: string | null;
}

export interface ModelEvaluation extends VersionedResource {
  project_id: number;
  training_run_id: number;
  model_version_id: number;
  task_version_id: number;
  training_dataset_version_id: number;
  dataset_version_id: number;
  split_map_id: number;
  artifact_package_id: number | null;
  split_name: string;
  status: "succeeded" | "unsupported" | "failed";
  evaluator_key: string | null;
  evaluator_version: string | null;
  row_count: number;
  requested_metrics: string[];
  metrics: Record<string, number | null>;
  report: Record<string, unknown>;
  evaluation_plan: Record<string, unknown>;
  runtime_digest: string;
  code_digest: string;
  status_reason: string | null;
  content_hash: string;
}

export interface Guideline extends VersionedResource {
  project_id: number;
  task_definition_id: number;
  name: string;
  description: string | null;
}

export interface GuidelineRevision extends VersionedResource {
  project_id: number;
  guideline_id: number;
  task_version_id: number;
  parent_revision_id: number | null;
  version_number: number;
  markdown: string;
  rationale: string | null;
  diff_summary: Record<string, unknown>;
  source_proposal_ids: number[];
  content_hash: string;
  status: "draft" | "pilot" | "active" | "retired";
  approved_by_user_id: number | null;
  approved_at: string | null;
}

export interface GuidelineProposal extends VersionedResource {
  project_id: number;
  guideline_id: number;
  base_revision_id: number | null;
  feedback_event_ids: number[];
  proposed_change: Record<string, unknown>;
  rationale: string;
  status: string;
  resulting_revision_id: number | null;
}

export interface GuidelineImpact extends VersionedResource {
  project_id: number;
  guideline_revision_id: number;
  pilot_round_id: number;
  protected_split_map_id: number | null;
  status: string;
  metrics: Record<string, unknown>;
  passed: boolean | null;
  completed_at: string | null;
}

export interface FeedbackEvent extends VersionedResource {
  project_id: number;
  event_type:
    | "correction"
    | "iaa_disagreement"
    | "model_disagreement"
    | "adjudication"
    | "drift"
    | "evaluation_failure"
    | "llm_critique";
  cycle_id: number | null;
  annotation_round_id: number | null;
  round_item_id: number | null;
  feedback_candidate_id: number | null;
  severity: "low" | "medium" | "high" | "critical";
  payload: Record<string, unknown>;
  occurred_at: string;
  created_by_user_id: number | null;
}

export interface ReviewCase extends VersionedResource {
  project_id: number;
  feedback_event_id: number;
  case_type: string;
  assigned_to_user_id: number | null;
  status: string;
  resolution: Record<string, unknown>;
  resolved_at: string | null;
}

export interface TrainingRecipeDescriptor {
  schema_version: "training-recipe-descriptor-v1";
  key: string;
  version: string;
  label: string;
  description: string;
  model_family: string;
  architecture_family: string;
  parameterization: "full" | "head_only" | "lora" | "qlora";
  supported_task_kinds: TaskKind[];
  trainer_key: string;
  implementation_status: "implemented" | "experimental";
  environment: {
    runtime_class: string;
    packages: string[];
    devices: string[];
    minimum_memory_gb: number;
    requires_verified_environment: boolean;
    setup_hint: string;
  };
  config_schema: Record<string, unknown>;
  artifact_formats: string[];
}

export interface TrainingRecipe extends VersionedResource {
  project_id: number;
  key: string;
  name: string;
  description: string | null;
}

export interface TrainingRecipeVersion extends VersionedResource {
  project_id: number;
  training_recipe_id: number;
  version_number: number;
  trainer_plugin_key: string;
  trainer_plugin_version: string;
  compatible_task_kinds: TaskKind[];
  environment_class: string;
  config_schema: Record<string, unknown>;
  default_config: Record<string, unknown>;
  evaluation_defaults: Record<string, unknown>;
  content_hash: string;
}

export interface ExecutionEnvironment extends VersionedResource {
  project_id: number;
  name: string;
  environment_class: string;
  image_digest: string | null;
  package_manifest: Record<string, unknown>;
  hardware_constraints: Record<string, unknown>;
  status: string;
  verification_report: Record<string, unknown>;
  verified_at: string | null;
  created_by_user_id?: number | null;
}

export interface StoragePolicy extends VersionedResource {
  project_id: number;
  name: string;
  backend: "s3" | "minio" | "local";
  artifact_prefix: string;
  retention_class: string;
  encryption: Record<string, unknown>;
  cache_policy: Record<string, unknown>;
  is_default: boolean;
  created_by_user_id?: number | null;
}

export interface TrainingRun extends VersionedResource {
  project_id: number;
  registered_model_id: number;
  task_version_id: number;
  training_dataset_version_id: number;
  recipe_version_id: number;
  environment_id: number;
  storage_policy_id: number;
  idempotency_key: string;
  status: string;
  seed: number;
  config: Record<string, unknown>;
  evaluation_plan: Record<string, unknown>;
  output_model_version_id: number | null;
  failure_code?: string | null;
  failure_reason?: string | null;
  parent_model_version_id?: number | null;
  artifact_reservation_id?: number | null;
  artifact_reservation_bytes?: number | null;
  launch_hash?: string;
  runtime_snapshot?: Record<string, unknown>;
  storage_snapshot?: Record<string, unknown>;
  started_at?: string | null;
  completed_at?: string | null;
  created_by_user_id?: number | null;
}

export interface WorkspacePage<T> {
  items: T[];
  next_cursor: number | null;
}

export interface WorkspaceTrainingContext {
  project_id: number;
  project_name: string;
  project_description: string | null;
  training_only: boolean;
  effective_modules: ProjectModule[];
  task_version_count: number;
  training_dataset_version_count: number;
  environment_count: number;
  available_environment_count: number;
  storage_policy_count: number;
}

export interface WorkspaceTrainingRun extends TrainingRun {
  project_name: string;
  project_description: string | null;
  model_name: string;
  family: string;
  framework: string;
  base_model: Record<string, unknown>;
  training_method: string;
  task_name: string;
  task_kind: TaskKind;
  training_dataset_name: string;
  dataset_version_number: number | null;
  recipe_name: string;
  runtime_name: string;
  storage_policy_name: string;
  evaluation_status: string | null;
  evaluation_split: string | null;
  evaluation_metrics: Record<string, number | null>;
  creator_username: string | null;
  creator_display_name: string | null;
}

export type WorkspaceTrainingRunPage = WorkspacePage<WorkspaceTrainingRun>;

export interface WorkspaceModel extends RegisteredModel {
  project_name: string;
  project_description: string | null;
  latest_version: ModelVersion | null;
  task_name: string | null;
  task_kind: TaskKind | null;
  training_dataset_name: string | null;
  evaluation_status: string | null;
  evaluation_split: string | null;
  evaluation_metrics: Record<string, number | null>;
  creator_username: string | null;
  creator_display_name: string | null;
}

export interface WorkspaceTrainingFilters {
  projectId?: number;
  status?: string;
  creatorUserId?: number;
  taskVersionId?: number;
  family?: string;
  cursor?: number;
  limit?: number;
}

export interface WorkspaceTrainingProject extends Project {
  training_context: WorkspaceTrainingContext;
}

export type ProjectModule =
  | "data"
  | "annotate"
  | "learning"
  | "train"
  | "models"
  | "guidelines"
  | "activity";

export interface ProjectModules {
  project_id: number;
  selected: ProjectModule[];
  effective: ProjectModule[];
  workspace_capabilities: string[];
}

export interface PlatformProjectData {
  workspaceMembers: WorkspaceMember[];
  projectModules: ProjectModules;
  taskDefinitions: TaskDefinition[];
  taskVersions: TaskVersion[];
  datasets: Dataset[];
  datasetVersions: DatasetVersion[];
  labelSets: LabelSetVersion[];
  splitMaps: SplitMap[];
  trainingDatasets: TrainingDatasetVersion[];
  cycles: LearningCycle[];
  rounds: AnnotationRound[];
  feedbackRuns: FeedbackRun[];
  feedbackSets: FeedbackSetVersion[];
  models: RegisteredModel[];
  modelVersions: ModelVersion[];
  modelEvaluations: ModelEvaluation[];
  guidelines: Guideline[];
  guidelineRevisions: GuidelineRevision[];
  guidelineProposals: GuidelineProposal[];
  guidelineImpacts: GuidelineImpact[];
  feedbackEvents: FeedbackEvent[];
  reviewCases: ReviewCase[];
  trainingRuns: TrainingRun[];
  recipes: TrainingRecipeDescriptor[];
  baseModels: BaseModelAsset[];
  projectRecipes: TrainingRecipe[];
  recipeVersions: TrainingRecipeVersion[];
  environments: ExecutionEnvironment[];
  storagePolicies: StoragePolicy[];
}

export const EMPTY_PLATFORM_PROJECT_DATA: PlatformProjectData = {
  workspaceMembers: [],
  projectModules: {
    project_id: 0,
    selected: [],
    effective: [],
    workspace_capabilities: [],
  },
  taskDefinitions: [],
  taskVersions: [],
  datasets: [],
  datasetVersions: [],
  labelSets: [],
  splitMaps: [],
  trainingDatasets: [],
  cycles: [],
  rounds: [],
  feedbackRuns: [],
  feedbackSets: [],
  models: [],
  modelVersions: [],
  modelEvaluations: [],
  guidelines: [],
  guidelineRevisions: [],
  guidelineProposals: [],
  guidelineImpacts: [],
  feedbackEvents: [],
  reviewCases: [],
  trainingRuns: [],
  recipes: [],
  baseModels: [],
  projectRecipes: [],
  recipeVersions: [],
  environments: [],
  storagePolicies: [],
};
