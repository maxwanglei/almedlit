import type { Page, Route } from "@playwright/test";

import {
  installEvidenceApiMock,
  type EvidenceApiMockState,
} from "./mockEvidenceApi";

export interface PlatformApiRequest {
  method: string;
  pathname: string;
  search: string;
  body?: unknown;
}

export interface PlatformApiMockOptions {
  role?: "personal" | "trainer" | "manager";
  includeSecondWorkspace?: boolean;
  trainingReleaseGate?: boolean;
}

async function json(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

const timestamp = "2026-07-27T14:00:00Z";
const hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const releaseGateHash =
  "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210";

const platformProject = {
  id: 1,
  name: "Evidence Review Demo",
  description: "Deterministic Playwright fixture",
  annotation_schema: { labels: { evidence_block: [] } },
  annotation_validation_mode: "strict",
  tasks: [
    {
      id: 11,
      project_id: 1,
      annotation_type: "evidence_block",
      display_name: "Evidence blocks",
      description: "Find complete-sentence evidence blocks.",
      enabled: true,
      sort_order: 0,
      labels: [],
      settings: {},
    },
  ],
  settings: {
    modules: [
      "data",
      "annotate",
      "models",
      "train",
      "activity",
    ],
  },
  workspace_id: 1,
};

const taskDefinitions = [
  {
    id: 11,
    project_id: 1,
    key: "abstract_relevance",
    name: "Abstract relevance",
    description: "Classify abstracts for evidence review.",
    created_at: timestamp,
  },
];

const taskVersions = [
  {
    id: 12,
    project_id: 1,
    task_definition_id: 11,
    version_number: 3,
    task_kind: "classification",
    input_schema: { type: "object", properties: { text: { type: "string" } } },
    output_schema: { type: "string", enum: ["include", "exclude"] },
    label_rules: { values: ["include", "exclude"], closed_set: true },
    annotation_ui: { preset: "classification" },
    metrics: ["f1", "precision", "recall"],
    trainer_compatibility: ["tfidf_logistic_regression"],
    content_hash: hash,
    created_at: timestamp,
  },
];

const datasets = [
  {
    id: 21,
    project_id: 1,
    name: "PubMed screening corpus",
    description: "Pinned abstracts and imported screening labels.",
    source_type: "upload",
    created_at: timestamp,
  },
];

const datasetVersions = [
  {
    id: 22,
    project_id: 1,
    dataset_id: 21,
    version_number: 2,
    source_uri: "upload://sha256/demo",
    source_revision: hash,
    source_format: "csv",
    data_schema: { type: "object" },
    provenance: { original_file_name: "screening.csv" },
    license_info: { identifier: "CC-BY-4.0" },
    content_hash: hash,
    item_count: 240,
    artifact_package_id: null,
    created_at: timestamp,
  },
];

const labelSets = [
  {
    id: 31,
    project_id: 1,
    dataset_version_id: 22,
    task_version_id: 12,
    parent_version_id: null,
    name: "Imported screening decisions",
    source_kind: "imported",
    composition_policy: "replace",
    labels: {},
    version_number: 1,
    label_count: 240,
    content_hash: hash,
    created_at: timestamp,
  },
];

const splitMaps = [
  {
    id: 41,
    project_id: 1,
    dataset_version_id: 22,
    name: "Patient-group split",
    strategy: "deterministic_group_hash",
    seed: 42,
    group_key_field: "patient_id",
    assignments: {},
    protected_splits: ["test"],
    content_hash: hash,
    created_at: timestamp,
  },
];

const trainingDatasets = [
  {
    id: 51,
    project_id: 1,
    name: "Relevance training set",
    dataset_version_id: 22,
    task_version_id: 12,
    label_set_version_ids: [31],
    split_map_id: 41,
    composition: [{ label_set_version_id: 31, policy: "replace" }],
    preprocessing: { input_field: "text", target_field: "label" },
    content_hash: hash,
    created_at: timestamp,
  },
];

const cycles = [
  {
    id: 61,
    project_id: 1,
    name: "Error remediation cycle",
    sequence: 2,
    goal: "error_remediation",
    parent_cycle_id: null,
    status: "active",
    current_stage: "annotate",
    task_version_id: 12,
    source_dataset_version_id: 22,
    baseline_model_version_id: 82,
    feedback_sources: [
      {
        producer_type: "external_llm",
        name: "Pinned disagreement reviewer",
        provider: "openai",
        external_model_id: "review-model",
        exact_revision: "2026-07-15",
        configuration: { temperature: 0 },
        data_egress_policy: { mode: "deidentified_only" },
      },
      {
        producer_type: "human_disagreement",
        name: "Adjudication disagreements",
        provider: null,
        external_model_id: null,
        exact_revision: null,
        configuration: { minimum_annotators: 2 },
        data_egress_policy: {},
      },
    ],
    output_training_dataset_version_id: null,
    output_model_version_id: null,
    metadata: {},
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const rounds = [
  {
    id: 71,
    project_id: 1,
    cycle_id: 61,
    parent_round_id: null,
    name: "False-negative review",
    sequence: 4,
    dataset_version_id: 22,
    task_version_id: 12,
    guideline_revision_id: 92,
    selection_set_version_id: 72,
    feedback_set_version_id: 73,
    assistance_policy: "reveal_after_first_pass",
    reannotation_mode: "targeted_subset",
    annotator_user_ids: [2],
    open_to_all_annotators: false,
    reason: "Review high-confidence false negatives.",
    status: "open",
    opened_at: timestamp,
    closed_at: null,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const roundWorkContext = {
  project: {
    id: platformProject.id,
    name: platformProject.name,
  },
  round: {
    id: rounds[0].id,
    project_id: rounds[0].project_id,
    name: rounds[0].name,
    sequence: rounds[0].sequence,
    dataset_version_id: rounds[0].dataset_version_id,
    task_version_id: rounds[0].task_version_id,
    assistance_policy: rounds[0].assistance_policy,
    feedback_available: true,
    status: rounds[0].status,
    opened_at: rounds[0].opened_at,
    closed_at: rounds[0].closed_at,
    created_at: rounds[0].created_at,
    updated_at: rounds[0].updated_at,
  },
  task: {
    id: taskDefinitions[0].id,
    key: taskDefinitions[0].key,
    name: taskDefinitions[0].name,
  },
  task_version: taskVersions[0],
  cycle: {
    id: cycles[0].id,
    name: cycles[0].name,
    sequence: cycles[0].sequence,
  },
  guideline: {
    guideline_id: 91,
    guideline_revision_id: 92,
    name: "Abstract relevance guidance",
    version_number: 2,
    status: "active",
  },
};

const feedbackRuns = [
  {
    id: 72,
    project_id: 1,
    dataset_version_id: 22,
    task_version_id: 12,
    producer_type: "external_llm",
    cycle_id: 61,
    model_version_id: null,
    provider: "openai",
    external_model_id: "review-model",
    exact_revision: "2026-07-15",
    prompt_template_hash: hash,
    configuration: { temperature: 0 },
    data_egress_policy: { mode: "deidentified_only" },
    status: "succeeded",
    created_by_user_id: 1,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const feedbackSets = [
  {
    id: 73,
    project_id: 1,
    feedback_run_id: 72,
    dataset_version_id: 22,
    task_version_id: 12,
    version_number: 1,
    output_schema: {
      type: "object",
      properties: {
        suggested_label: { type: "string" },
        explanation: { type: "string" },
      },
    },
    candidate_count: 48,
    content_hash: hash,
    artifact_package_id: 273,
    created_by_user_id: 1,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const registeredModels = [
  {
    id: 81,
    project_id: 1,
    name: "Relevance baseline",
    description: "TF-IDF screening baseline.",
    lifecycle_status: "active",
    created_at: timestamp,
  },
];

const modelVersions = [
  {
    id: 82,
    project_id: 1,
    registered_model_id: 81,
    version_number: 2,
    parent_version_id: null,
    task_version_id: 12,
    training_dataset_version_id: 51,
    family: "linear",
    framework: "scikit-learn",
    base_model: {},
    training_method: "supervised",
    recipe_key: "tfidf_logistic_regression",
    recipe_version: "1.0.0",
    parameters: {},
    metrics: { f1: 0.84 },
    runtime_digest: hash,
    content_hash: hash,
    created_at: timestamp,
  },
];

const modelEvaluations = [
  {
    id: 83,
    project_id: 1,
    training_run_id: 151,
    model_version_id: 82,
    task_version_id: 12,
    training_dataset_version_id: 51,
    dataset_version_id: 22,
    split_map_id: 41,
    artifact_package_id: 281,
    split_name: "test",
    status: "succeeded",
    evaluator_key: "classification_metrics",
    evaluator_version: "1.0.0",
    row_count: 36,
    requested_metrics: ["f1", "precision", "recall"],
    metrics: { f1: 0.84, precision: 0.86, recall: 0.82 },
    report: {
      confusion_matrix: [[16, 3], [3, 14]],
      labels: ["exclude", "include"],
    },
    evaluation_plan: {
      splits: ["test"],
      metrics: ["f1", "precision", "recall"],
    },
    runtime_digest: hash,
    code_digest: hash,
    status_reason: null,
    content_hash: hash,
    created_by_user_id: 1,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const guidelines = [
  {
    id: 91,
    project_id: 1,
    task_definition_id: 11,
    name: "Abstract screening guidance",
    description: "Inclusion and exclusion guidance.",
    created_at: timestamp,
  },
];

const guidelineRevisions = [
  {
    id: 92,
    project_id: 1,
    guideline_id: 91,
    task_version_id: 12,
    parent_revision_id: null,
    version_number: 5,
    markdown: "# Screening guidance",
    rationale: "Clarify mixed-population studies.",
    diff_summary: { changed_sections: ["Population"] },
    source_proposal_ids: [93],
    content_hash: hash,
    status: "pilot",
    approved_by_user_id: null,
    approved_at: null,
    created_at: timestamp,
  },
];

const guidelineProposals = [
  {
    id: 93,
    project_id: 1,
    guideline_id: 91,
    base_revision_id: 92,
    feedback_event_ids: [101],
    proposed_change: { section: "Population" },
    rationale: "Clarify how to classify mixed-population studies.",
    status: "pending",
    reviewed_by_user_id: null,
    reviewed_at: null,
    resulting_revision_id: null,
    created_by_user_id: 1,
    created_at: timestamp,
  },
];

const guidelineImpacts = [
  {
    id: 94,
    project_id: 1,
    guideline_revision_id: 92,
    pilot_round_id: 71,
    protected_split_map_id: 41,
    status: "completed",
    metrics: { agreement_delta: 0.08 },
    passed: true,
    completed_at: timestamp,
    reviewed_by_user_id: 1,
    created_by_user_id: 1,
    created_at: timestamp,
  },
];

const feedbackEvents = [
  {
    id: 101,
    project_id: 1,
    event_type: "model_disagreement",
    cycle_id: 61,
    annotation_round_id: 71,
    round_item_id: 701,
    feedback_candidate_id: 702,
    severity: "high",
    payload: { summary: "Model and adjudicated label disagree." },
    occurred_at: timestamp,
    created_by_user_id: 1,
    created_at: timestamp,
  },
];

const reviewCases = [
  {
    id: 102,
    project_id: 1,
    feedback_event_id: 101,
    case_type: "false_negative_review",
    assigned_to_user_id: 1,
    status: "open",
    resolution: {},
    resolved_at: null,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const recipeDescriptors = [
  {
    schema_version: "training-recipe-descriptor-v1",
    key: "tfidf_logistic_regression",
    version: "1.0.0",
    label: "TF-IDF logistic regression",
    description: "Fast linear text classification baseline.",
    model_family: "conventional_ml",
    architecture_family: "linear_classifier",
    parameterization: "full",
    supported_task_kinds: ["classification"],
    trainer_key: "sklearn_tfidf",
    implementation_status: "implemented",
    environment: {
      runtime_class: "classical-cpu",
      packages: ["scikit-learn"],
      devices: ["cpu"],
      minimum_memory_gb: 4,
      requires_verified_environment: true,
      setup_hint: "Use the verified CPU environment.",
    },
    config_schema: { type: "object" },
    artifact_formats: ["joblib"],
  },
  {
    schema_version: "training-recipe-descriptor-v1",
    key: "transformer_sequence_classification",
    version: "1.0.0",
    label: "Transformer sequence classification",
    description: "Encoder fine-tuning for document classification.",
    model_family: "deep_learning",
    architecture_family: "transformer_encoder",
    parameterization: "full",
    supported_task_kinds: ["classification"],
    trainer_key: "huggingface_sequence",
    implementation_status: "implemented",
    environment: {
      runtime_class: "transformer-cpu",
      packages: ["torch", "transformers", "safetensors"],
      devices: ["cpu"],
      minimum_memory_gb: 8,
      requires_verified_environment: true,
      setup_hint: "Use a verified transformer environment.",
    },
    config_schema: {
      type: "object",
      required: ["base_model_asset_id"],
    },
    artifact_formats: ["safetensors", "huggingface_json", "tokenizer"],
  },
];

function baseModelPackage(
  id: number,
  modelFamily: "deep_learning" | "llm_finetune",
  modelType: string,
  readiness = "ready",
) {
  return {
    id,
    project_id: 1,
    kind: "base_model",
    format: "safetensors",
    schema_version: "1",
    model_family: modelFamily,
    model_type: modelType,
    readiness,
    deployable: true,
    manifest_digest: hash,
    logical_size_bytes: 1024,
    file_count: 1,
    archived_at: null,
    purged_at: null,
    created_at: timestamp,
    files: [],
    references: [],
  };
}

const baseModels = [
  {
    id: 161,
    project_id: 1,
    package_id: 261,
    provider: "hugging_face",
    source_model_id: "medical/abstract-encoder",
    exact_revision: "encoder-commit-42",
    display_name: "Medical abstract encoder",
    model_family: "deep_learning",
    model_type: "bert",
    license_name: "Apache-2.0",
    license_url: null,
    license_terms_sha256: null,
    access_mode: "downloadable",
    readiness: "ready",
    archived_at: null,
    metadata: {
      architecture_family: "transformer_encoder",
      supported_task_kinds: ["classification"],
    },
    package: baseModelPackage(261, "deep_learning", "bert"),
    created_at: timestamp,
  },
  {
    id: 162,
    project_id: 1,
    package_id: 262,
    provider: "hugging_face",
    source_model_id: "medical/generative-model",
    exact_revision: "causal-commit-17",
    display_name: "Medical causal model",
    model_family: "llm_finetune",
    model_type: "causal_lm",
    license_name: "Apache-2.0",
    license_url: null,
    license_terms_sha256: null,
    access_mode: "execution_only",
    readiness: "ready",
    archived_at: null,
    metadata: { architecture_family: "causal_lm" },
    package: baseModelPackage(262, "llm_finetune", "causal_lm"),
    created_at: timestamp,
  },
  {
    id: 163,
    project_id: 1,
    package_id: 263,
    provider: "hugging_face",
    source_model_id: "medical/quarantined-encoder",
    exact_revision: "quarantined-commit",
    display_name: "Quarantined encoder",
    model_family: "deep_learning",
    model_type: "bert",
    license_name: "Apache-2.0",
    license_url: null,
    license_terms_sha256: null,
    access_mode: "downloadable",
    readiness: "quarantined",
    archived_at: null,
    metadata: { architecture_family: "transformer_encoder" },
    package: baseModelPackage(263, "deep_learning", "bert", "quarantined"),
    created_at: timestamp,
  },
];

const projectRecipes = [
  {
    id: 121,
    project_id: 1,
    key: "tfidf_logistic_regression",
    name: "TF-IDF logistic regression",
    description: "Fast linear text classification baseline.",
    created_at: timestamp,
  },
];

const recipeVersions = [
  {
    id: 122,
    project_id: 1,
    training_recipe_id: 121,
    version_number: 1,
    trainer_plugin_key: "sklearn_tfidf",
    trainer_plugin_version: "1.0.0",
    compatible_task_kinds: ["classification"],
    environment_class: "classical-cpu",
    config_schema: { type: "object" },
    default_config: {},
    evaluation_defaults: { splits: ["test"] },
    content_hash: hash,
    created_at: timestamp,
  },
];

const environments = [
  {
    id: 131,
    project_id: 1,
    name: "Verified CPU",
    environment_class: "classical-cpu",
    image_digest:
      "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    package_manifest: { packages: ["scikit-learn"] },
    hardware_constraints: { devices: ["cpu"] },
    status: "available",
    verification_report: { passed: true },
    verified_at: timestamp,
    created_at: timestamp,
  },
];

const storagePolicies = [
  {
    id: 141,
    project_id: 1,
    name: "Project artifacts",
    backend: "minio",
    artifact_prefix: "projects/1",
    retention_class: "standard",
    encryption: {
      enabled: true,
      algorithm: "AES-256-GCM",
      key_management: "platform_managed",
    },
    cache_policy: { enabled: true },
    is_default: true,
    created_at: timestamp,
  },
];

const trainingRuns = [
  {
    id: 151,
    project_id: 1,
    registered_model_id: 81,
    task_version_id: 12,
    training_dataset_version_id: 51,
    recipe_version_id: 122,
    environment_id: 131,
    storage_policy_id: 141,
    idempotency_key: "fixture-run",
    status: "succeeded",
    seed: 42,
    config: {},
    evaluation_plan: { splits: ["test"] },
    output_model_version_id: 82,
    failure_code: null,
    failure_reason: null,
    created_at: timestamp,
    updated_at: timestamp,
  },
];

const workspaceTrainingContexts = {
  items: [
    {
      project_id: 1,
      project_name: platformProject.name,
      project_description: platformProject.description,
      training_only: false,
      effective_modules: [
        "data",
        "annotate",
        "models",
        "train",
        "activity",
      ],
      task_version_count: 1,
      training_dataset_version_count: 1,
      environment_count: 1,
      available_environment_count: 1,
      storage_policy_count: 1,
    },
  ],
  next_cursor: null,
};

const workspaceTrainingRuns = {
  items: [
    {
      ...trainingRuns[0],
      created_by_user_id: 2,
      project_name: platformProject.name,
      project_description: platformProject.description,
      model_name: registeredModels[0].name,
      family: modelVersions[0].family,
      framework: modelVersions[0].framework,
      base_model: modelVersions[0].base_model,
      training_method: modelVersions[0].training_method,
      task_name: taskDefinitions[0].name,
      task_kind: taskVersions[0].task_kind,
      training_dataset_name: trainingDatasets[0].name,
      dataset_version_number: datasetVersions[0].version_number,
      recipe_name: recipeDescriptors[0].label,
      runtime_name: environments[0].name,
      storage_policy_name: storagePolicies[0].name,
      evaluation_status: modelEvaluations[0].status,
      evaluation_split: modelEvaluations[0].split_name,
      evaluation_metrics: modelEvaluations[0].metrics,
      creator_username: "alice",
      creator_display_name: "Alice Trainer",
    },
  ],
  next_cursor: null,
};

const workspaceModels = {
  items: [
    {
      ...registeredModels[0],
      created_by_user_id: 2,
      project_name: platformProject.name,
      project_description: platformProject.description,
      latest_version: {
        ...modelVersions[0],
        created_by_user_id: 2,
      },
      task_name: taskDefinitions[0].name,
      task_kind: taskVersions[0].task_kind,
      training_dataset_name: trainingDatasets[0].name,
      evaluation_status: modelEvaluations[0].status,
      evaluation_split: modelEvaluations[0].split_name,
      evaluation_metrics: modelEvaluations[0].metrics,
      creator_username: "alice",
      creator_display_name: "Alice Trainer",
    },
  ],
  next_cursor: null,
};

export async function installPlatformApiMock(
  page: Page,
  options: PlatformApiMockOptions = {},
): Promise<{
  requests: PlatformApiRequest[];
  evidence: EvidenceApiMockState;
}> {
  const role = options.role ?? "personal";
  const evidence = await installEvidenceApiMock(
    page,
    role === "manager" ? "manager" : "personal",
  );
  const requests: PlatformApiRequest[] = [];
  const releaseGateEnabled = options.trainingReleaseGate === true;
  let releaseProject: Record<string, unknown> | null = null;
  let releaseDataset: Record<string, unknown> | null = null;
  let releaseDatasetVersion: Record<string, unknown> | null = null;
  let releaseTask: Record<string, unknown> | null = null;
  let releaseTaskVersion: Record<string, unknown> | null = null;
  let releaseLabelSet: Record<string, unknown> | null = null;
  let releaseSplitMap: Record<string, unknown> | null = null;
  let releaseTrainingDataset: Record<string, unknown> | null = null;
  let releaseRecipe: Record<string, unknown> | null = null;
  let releaseRecipeVersion: Record<string, unknown> | null = null;
  let releaseModel: Record<string, unknown> | null = null;
  let releaseModelVersion: Record<string, unknown> | null = null;
  let releaseEvaluation: Record<string, unknown> | null = null;
  let releaseRun: Record<string, unknown> | null = null;

  const releaseEnvironment = {
    id: 331,
    project_id: 2,
    name: "Verified classical CPU",
    environment_class: "classical-cpu",
    image_digest:
      "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    package_manifest: { packages: ["scikit-learn==1.8.0"] },
    hardware_constraints: { device: "cpu", memory_gb: 8 },
    status: "available",
    verification_report: {
      verified: true,
      image_digest:
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    },
    verified_at: timestamp,
    created_by_user_id: 2,
    created_at: timestamp,
  };
  const releaseStorage = {
    id: 341,
    project_id: 2,
    name: "Immutable MinIO artifacts",
    backend: "minio",
    artifact_prefix: "projects/2/training",
    retention_class: "standard",
    encryption: {
      enabled: true,
      algorithm: "AES-256-GCM",
      key_management: "platform_managed",
    },
    cache_policy: { enabled: true },
    is_default: true,
    created_by_user_id: 2,
    created_at: timestamp,
  };

  const releaseContext = (): Record<string, unknown> | null =>
    releaseProject
      ? {
          project_id: 2,
          project_name: releaseProject.name,
          project_description: releaseProject.description,
          training_only: true,
          effective_modules: ["data", "train", "models", "activity"],
          task_version_count: releaseTaskVersion ? 1 : 0,
          training_dataset_version_count: releaseTrainingDataset ? 1 : 0,
          environment_count: 1,
          available_environment_count: 1,
          storage_policy_count: 1,
        }
      : null;

  const releaseWorkspaceRun = (): Record<string, unknown> | null => {
    if (
      !releaseProject ||
      !releaseRun ||
      !releaseModel ||
      !releaseTask ||
      !releaseTaskVersion ||
      !releaseTrainingDataset ||
      !releaseDatasetVersion ||
      !releaseModelVersion ||
      !releaseEvaluation
    ) {
      return null;
    }
    return {
      ...releaseRun,
      created_by_user_id: 2,
      project_name: releaseProject.name,
      project_description: releaseProject.description,
      model_name: releaseModel.name,
      family: releaseModelVersion.family,
      framework: releaseModelVersion.framework,
      base_model: releaseModelVersion.base_model,
      training_method: releaseModelVersion.training_method,
      task_name: releaseTask.name,
      task_kind: releaseTaskVersion.task_kind,
      training_dataset_name: releaseTrainingDataset.name,
      dataset_version_number: releaseDatasetVersion.version_number,
      recipe_name: recipeDescriptors[0].label,
      runtime_name: releaseEnvironment.name,
      storage_policy_name: releaseStorage.name,
      evaluation_status: releaseEvaluation.status,
      evaluation_split: releaseEvaluation.split_name,
      evaluation_metrics: releaseEvaluation.metrics,
      creator_username: "alice",
      creator_display_name: "Alice Owner",
    };
  };

  const releaseWorkspaceModel = (): Record<string, unknown> | null => {
    if (
      !releaseProject ||
      !releaseModel ||
      !releaseModelVersion ||
      !releaseTask ||
      !releaseTaskVersion ||
      !releaseTrainingDataset ||
      !releaseEvaluation
    ) {
      return null;
    }
    return {
      ...releaseModel,
      created_by_user_id: 2,
      project_name: releaseProject.name,
      project_description: releaseProject.description,
      latest_version: releaseModelVersion,
      task_name: releaseTask.name,
      task_kind: releaseTaskVersion.task_kind,
      training_dataset_name: releaseTrainingDataset.name,
      evaluation_status: releaseEvaluation.status,
      evaluation_split: releaseEvaluation.split_name,
      evaluation_metrics: releaseEvaluation.metrics,
      creator_username: "alice",
      creator_display_name: "Alice Owner",
    };
  };

  await page.route(/^https?:\/\/[^/]+\/api\//, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "GET" && url.pathname === "/api/auth/me") {
      const personal = role === "personal";
      const manager = role === "manager";
      const memberships = [
        {
          workspace_id: 1,
          workspace_name: personal ? "Alice's Workspace" : "Evidence Team",
          workspace_kind: personal ? "individual" : "team",
          role: personal ? "admin" : role,
        },
        ...(options.includeSecondWorkspace
          ? [
              {
                workspace_id: 2,
                workspace_name: "Screening Team",
                workspace_kind: "team",
                role: "annotator",
              },
            ]
          : []),
      ];
      requests.push({
        method: request.method(),
        pathname: url.pathname,
        search: url.search,
      });
      await json(route, {
        user: {
          id: manager ? 1 : 2,
          username: manager ? "manager" : "alice",
          display_name: manager
            ? "Project Manager"
            : role === "trainer"
              ? "Alice Trainer"
              : "Alice Owner",
          is_active: true,
          is_superuser: false,
        },
        memberships,
      });
      return;
    }
    if (
      request.method() === "GET" &&
      /^\/api\/workspaces\/[12]\/capabilities$/.test(url.pathname)
    ) {
      const workspaceId = Number(url.pathname.split("/")[3]);
      requests.push({
        method: request.method(),
        pathname: url.pathname,
        search: url.search,
      });
      await json(route, {
        preset: workspaceId === 1 ? "full_loop" : "annotation",
        overrides: [],
        effective:
          workspaceId === 1
            ? ["annotation", "lineage", "export", "training", "inference"]
            : ["annotation", "lineage", "export"],
        blocked: {},
      });
      return;
    }
    if (
      request.method() === "GET" &&
      /^\/api\/workspaces\/[12]\/my-work\/rounds$/.test(url.pathname)
    ) {
      const workspaceId = Number(url.pathname.split("/")[3]);
      requests.push({
        method: request.method(),
        pathname: url.pathname,
        search: url.search,
      });
      await json(route, workspaceId === 1 ? [roundWorkContext] : []);
      return;
    }
    if (
      request.method() === "GET" &&
      url.pathname === "/api/rounds/71/work-context"
    ) {
      requests.push({
        method: request.method(),
        pathname: url.pathname,
        search: url.search,
      });
      await json(route, roundWorkContext);
      return;
    }
    if (request.method() === "GET" && url.pathname === "/api/projects") {
      const workspaceId = Number(url.searchParams.get("workspace_id") ?? "1");
      requests.push({
        method: request.method(),
        pathname: url.pathname,
        search: url.search,
      });
      await json(
        route,
        workspaceId === 1
          ? [platformProject, ...(releaseProject ? [releaseProject] : [])]
          : [],
      );
      return;
    }
    if (request.method() === "POST") {
      let requestBody: unknown = null;
      try {
        requestBody = request.postDataJSON();
      } catch {
        requestBody = request.postData();
      }
      const payload =
        requestBody !== null &&
        typeof requestBody === "object" &&
        !Array.isArray(requestBody)
          ? (requestBody as Record<string, unknown>)
          : {};
      const capture = (): void => {
        requests.push({
          method: request.method(),
          pathname: url.pathname,
          search: url.search,
          body: requestBody,
        });
      };

      if (releaseGateEnabled && url.pathname === "/api/projects") {
        releaseProject = {
          id: 2,
          name: String(payload.name ?? "Public benchmark training"),
          description:
            typeof payload.description === "string"
              ? payload.description
              : null,
          annotation_schema: payload.annotation_schema ?? { labels: {} },
          annotation_validation_mode:
            payload.annotation_validation_mode ?? "strict",
          tasks: [],
          settings: payload.settings ?? {
            modules: ["data", "train", "models", "activity"],
            project_purpose: "training_only",
          },
          workspace_id: Number(payload.workspace_id ?? 1),
        };
        capture();
        await json(route, releaseProject, 201);
        return;
      }
      if (
        releaseGateEnabled &&
        releaseProject &&
        url.pathname === "/api/datasets" &&
        Number(payload.project_id) === 2
      ) {
        releaseDataset = {
          id: 221,
          project_id: 2,
          name: String(payload.name ?? "IMDb pinned benchmark"),
          description:
            typeof payload.description === "string"
              ? payload.description
              : null,
          source_type: payload.source_type ?? "public_registry",
          created_at: timestamp,
        };
        capture();
        await json(route, releaseDataset, 201);
        return;
      }
      if (
        releaseGateEnabled &&
        releaseDataset &&
        url.pathname ===
          "/api/projects/2/datasets/221/versions/public-registry-snapshot"
      ) {
        releaseDatasetVersion = {
          id: 222,
          project_id: 2,
          dataset_id: 221,
          version_number: 1,
          source_uri: "hf://stanfordnlp/imdb",
          source_revision: "2fdd8b9bcadd6e7055e742a7069a522d8f4cce56",
          source_format: "jsonl",
          data_schema: {
            type: "object",
            properties: {
              text: { type: "string" },
              label: { type: "string" },
            },
          },
          provenance: {
            registry: "hugging_face",
            registry_dataset_id: "stanfordnlp/imdb",
            exact_revision:
              "2fdd8b9bcadd6e7055e742a7069a522d8f4cce56",
            original_file_name: "imdb-pinned.jsonl",
          },
          license_info: { identifier: "apache-2.0" },
          content_hash: releaseGateHash,
          item_count: 50_000,
          artifact_package_id: 421,
          created_at: timestamp,
        };
        capture();
        await json(route, releaseDatasetVersion, 201);
        return;
      }
      if (
        releaseGateEnabled &&
        releaseProject &&
        url.pathname === "/api/tasks" &&
        Number(payload.project_id) === 2
      ) {
        releaseTask = {
          id: 211,
          project_id: 2,
          key: String(payload.key ?? "imdb_sentiment"),
          name: String(payload.name ?? "IMDb sentiment"),
          description:
            typeof payload.description === "string"
              ? payload.description
              : null,
          created_at: timestamp,
        };
        capture();
        await json(route, releaseTask, 201);
        return;
      }
      if (
        releaseGateEnabled &&
        releaseTask &&
        url.pathname === "/api/tasks/versions" &&
        Number(payload.project_id) === 2
      ) {
        releaseTaskVersion = {
          ...payload,
          id: 212,
          project_id: 2,
          task_definition_id: 211,
          version_number: 1,
          task_kind: payload.task_kind ?? "classification",
          content_hash: releaseGateHash,
          created_at: timestamp,
        };
        capture();
        await json(route, releaseTaskVersion, 201);
        return;
      }
      if (
        releaseGateEnabled &&
        releaseDatasetVersion &&
        releaseTaskVersion &&
        url.pathname === "/api/datasets/training-versions/compose" &&
        Number(payload.project_id) === 2
      ) {
        releaseLabelSet = {
          id: 231,
          project_id: 2,
          dataset_version_id: 222,
          task_version_id: 212,
          parent_version_id: null,
          name: "Imported label field · label",
          source_kind: "imported",
          composition_policy: "replace",
          labels: {},
          version_number: 1,
          label_count: 50_000,
          content_hash: releaseGateHash,
          created_at: timestamp,
        };
        releaseSplitMap = {
          id: 241,
          project_id: 2,
          dataset_version_id: 222,
          name: "Deterministic protected split",
          strategy: "deterministic_group_hash",
          seed: Number(payload.seed ?? 42),
          group_key_field: null,
          assignments: {},
          protected_splits: ["test"],
          content_hash: releaseGateHash,
          created_at: timestamp,
        };
        releaseTrainingDataset = {
          id: 251,
          project_id: 2,
          name: String(payload.name ?? "IMDb sentiment training set"),
          dataset_version_id: 222,
          task_version_id: 212,
          label_set_version_ids: [231],
          split_map_id: 241,
          composition: [
            { label_set_version_id: 231, policy: "replace" },
          ],
          preprocessing: {
            input_field: String(payload.input_field ?? "text"),
            target_field: String(payload.label_field ?? "label"),
          },
          content_hash: releaseGateHash,
          created_at: timestamp,
        };
        capture();
        await json(
          route,
          {
            training_dataset_version: releaseTrainingDataset,
            label_set_version_id: 231,
            split_map_id: 241,
            split_counts: {
              train: 40_000,
              validation: 5_000,
              test: 5_000,
            },
            group_count: 50_000,
          },
          201,
        );
        return;
      }

      let response: unknown;
      if (
        url.pathname.startsWith("/api/training/recipes/") &&
        url.pathname.endsWith("/validate-configuration")
      ) {
        response = {
          valid: true,
          normalized_config: payload.config ?? {},
          errors: [],
        };
      } else if (
        url.pathname.startsWith("/api/training-recipes/trusted/")
      ) {
        if (
          releaseGateEnabled &&
          Number(url.searchParams.get("project_id")) === 2
        ) {
          releaseRecipe = {
            ...projectRecipes[0],
            id: 321,
            project_id: 2,
            created_at: timestamp,
          };
          releaseRecipeVersion = {
            ...recipeVersions[0],
            id: 322,
            project_id: 2,
            training_recipe_id: 321,
            content_hash: releaseGateHash,
            created_at: timestamp,
          };
          response = releaseRecipeVersion;
        } else {
          response = recipeVersions[0];
        }
      } else if (url.pathname === "/api/models") {
        if (releaseGateEnabled && Number(payload.project_id) === 2) {
          releaseModel = {
            id: 281,
            project_id: 2,
            name: String(payload.name ?? "IMDb sentiment linear"),
            description: null,
            lifecycle_status: "candidate",
            created_by_user_id: 2,
            created_at: timestamp,
          };
          response = releaseModel;
        } else {
          response = {
            ...registeredModels[0],
            id: 84,
            name: String(payload.name ?? "New model"),
          };
        }
      } else if (url.pathname === "/api/training-runs") {
        if (
          releaseGateEnabled &&
          releaseProject &&
          releaseDatasetVersion &&
          releaseTrainingDataset &&
          releaseTaskVersion &&
          releaseRecipeVersion &&
          releaseModel &&
          Number(payload.project_id) === 2
        ) {
          releaseRun = {
            id: 252,
            ...payload,
            project_id: 2,
            registered_model_id: 281,
            task_version_id: 212,
            training_dataset_version_id: 251,
            recipe_version_id: 322,
            environment_id: 331,
            storage_policy_id: 341,
            status: "succeeded",
            output_model_version_id: 282,
            failure_code: null,
            failure_reason: null,
            created_by_user_id: 2,
            started_at: timestamp,
            completed_at: timestamp,
            created_at: timestamp,
            updated_at: timestamp,
          };
          releaseModelVersion = {
            id: 282,
            project_id: 2,
            registered_model_id: 281,
            version_number: 1,
            parent_version_id: null,
            task_version_id: 212,
            training_dataset_version_id: 251,
            family: "linear",
            framework: "scikit-learn",
            base_model: {},
            training_method: "full",
            recipe_key: "tfidf_logistic_regression",
            recipe_version: "1.0.0",
            parameters: payload.config ?? {},
            metrics: { macro_f1: 0.912, accuracy: 0.915 },
            runtime_digest: releaseEnvironment.image_digest,
            code_digest: releaseGateHash,
            content_hash: releaseGateHash,
            seed: Number(payload.seed ?? 42),
            checkpoint_package_id: 481,
            created_by_user_id: 2,
            training_run_id: 252,
            source_dataset_version_id: 222,
            source_dataset_version_number: 1,
            runtime_id: 331,
            runtime_name: releaseEnvironment.name,
            storage_policy_id: 341,
            storage_policy_name: releaseStorage.name,
            creator_username: "alice",
            creator_display_name: "Alice Owner",
            created_at: timestamp,
          };
          releaseEvaluation = {
            id: 283,
            project_id: 2,
            training_run_id: 252,
            model_version_id: 282,
            task_version_id: 212,
            training_dataset_version_id: 251,
            dataset_version_id: 222,
            split_map_id: 241,
            artifact_package_id: 482,
            split_name: "test",
            status: "succeeded",
            evaluator_key: "classification_metrics",
            evaluator_version: "1.0.0",
            row_count: 5_000,
            requested_metrics: ["macro_f1", "precision", "recall", "accuracy"],
            metrics: {
              macro_f1: 0.912,
              precision: 0.918,
              recall: 0.907,
              accuracy: 0.915,
            },
            report: {},
            evaluation_plan: payload.evaluation_plan ?? {
              splits: ["test"],
            },
            runtime_digest: releaseEnvironment.image_digest,
            code_digest: releaseGateHash,
            status_reason: null,
            content_hash: releaseGateHash,
            created_by_user_id: 2,
            created_at: timestamp,
            updated_at: timestamp,
          };
          response = releaseRun;
        } else {
          response = {
            ...trainingRuns[0],
            ...payload,
            id: 152,
            status: "queued",
            output_model_version_id: null,
          };
        }
      }
      if (response !== undefined) {
        capture();
        await json(route, response);
        return;
      }
    }
    if (request.method() !== "GET") {
      await route.fallback();
      return;
    }

    if (releaseGateEnabled) {
      let releaseResponse: unknown;
      if (
        url.pathname === "/api/workspaces/1/training-contexts" &&
        releaseProject
      ) {
        releaseResponse = {
          items: [
            ...workspaceTrainingContexts.items,
            ...(releaseContext() ? [releaseContext()] : []),
          ],
          next_cursor: null,
        };
      } else if (url.pathname === "/api/workspaces/1/training-runs") {
        const projectFilter = Number(url.searchParams.get("project_id") ?? 0);
        const gateRun = releaseWorkspaceRun();
        releaseResponse = {
          items: [
            ...(projectFilter === 2 ? [] : workspaceTrainingRuns.items),
            ...(gateRun && projectFilter !== 1 ? [gateRun] : []),
          ],
          next_cursor: null,
        };
      } else if (url.pathname === "/api/workspaces/1/models") {
        const projectFilter = Number(url.searchParams.get("project_id") ?? 0);
        const gateModel = releaseWorkspaceModel();
        releaseResponse = {
          items: [
            ...(projectFilter === 2 ? [] : workspaceModels.items),
            ...(gateModel && projectFilter !== 1 ? [gateModel] : []),
          ],
          next_cursor: null,
        };
      } else if (url.pathname === "/api/projects/2" && releaseProject) {
        releaseResponse = releaseProject;
      } else if (
        url.pathname === "/api/projects/2/modules" &&
        releaseProject
      ) {
        releaseResponse = {
          project_id: 2,
          selected: ["data", "train", "models", "activity"],
          effective: ["data", "train", "models", "activity"],
          workspace_capabilities: [
            "training",
            "lineage",
            "export",
            "inference",
          ],
        };
      } else if (url.pathname === "/api/projects/2/base-models") {
        releaseResponse = [];
      } else if (
        url.pathname === "/api/models/281/versions/282/evaluations"
      ) {
        releaseResponse = releaseEvaluation ? [releaseEvaluation] : [];
      } else if (url.pathname === "/api/training-runs/252/evaluations") {
        releaseResponse = releaseEvaluation ? [releaseEvaluation] : [];
      } else if (Number(url.searchParams.get("project_id")) === 2) {
        const releaseCollections = new Map<string, unknown>([
          ["/api/tasks", releaseTask ? [releaseTask] : []],
          [
            "/api/tasks/versions",
            releaseTaskVersion ? [releaseTaskVersion] : [],
          ],
          ["/api/datasets", releaseDataset ? [releaseDataset] : []],
          [
            "/api/datasets/versions",
            releaseDatasetVersion ? [releaseDatasetVersion] : [],
          ],
          [
            "/api/datasets/label-sets",
            releaseLabelSet ? [releaseLabelSet] : [],
          ],
          [
            "/api/datasets/split-maps",
            releaseSplitMap ? [releaseSplitMap] : [],
          ],
          [
            "/api/datasets/training-versions",
            releaseTrainingDataset ? [releaseTrainingDataset] : [],
          ],
          ["/api/models", releaseModel ? [releaseModel] : []],
          [
            "/api/models/versions",
            releaseModelVersion ? [releaseModelVersion] : [],
          ],
          ["/api/training-runs", releaseRun ? [releaseRun] : []],
          [
            "/api/training-recipes",
            releaseRecipe ? [releaseRecipe] : [],
          ],
          [
            "/api/training-recipes/versions",
            releaseRecipeVersion ? [releaseRecipeVersion] : [],
          ],
          ["/api/environments", [releaseEnvironment]],
          ["/api/storage/policies", [releaseStorage]],
        ]);
        if (releaseCollections.has(url.pathname)) {
          releaseResponse = releaseCollections.get(url.pathname);
        }
      }
      if (releaseResponse !== undefined) {
        requests.push({
          method: request.method(),
          pathname: url.pathname,
          search: url.search,
        });
        await json(route, releaseResponse);
        return;
      }
    }

    const responses = new Map<string, unknown>([
      ["/api/workspaces/1/training-contexts", workspaceTrainingContexts],
      ["/api/workspaces/1/training-runs", workspaceTrainingRuns],
      ["/api/workspaces/1/models", workspaceModels],
      [
        "/api/projects/1/modules",
        {
          project_id: 1,
          selected: [
            "data",
            "annotate",
            "learning",
            "train",
            "models",
            "guidelines",
            "activity",
          ],
          effective: [
            "data",
            "annotate",
            "learning",
            "train",
            "models",
            "guidelines",
            "activity",
          ],
          workspace_capabilities: [
            "annotation",
            "training",
            "lineage",
            "active_learning",
            "co_learning",
          ],
        },
      ],
      ["/api/tasks", taskDefinitions],
      ["/api/tasks/versions", taskVersions],
      ["/api/datasets", datasets],
      ["/api/datasets/versions", datasetVersions],
      ["/api/datasets/label-sets", labelSets],
      ["/api/datasets/split-maps", splitMaps],
      ["/api/datasets/training-versions", trainingDatasets],
      ["/api/cycles", cycles],
      ["/api/rounds", rounds],
      ["/api/feedback-runs", feedbackRuns],
      ["/api/feedback-runs/sets", feedbackSets],
      ["/api/models", registeredModels],
      ["/api/models/versions", modelVersions],
      ["/api/models/81/versions/82/evaluations", modelEvaluations],
      ["/api/training-runs/151/evaluations", modelEvaluations],
      ["/api/workflow-guidelines", guidelines],
      ["/api/workflow-guidelines/revisions", guidelineRevisions],
      ["/api/workflow-guidelines/proposals", guidelineProposals],
      ["/api/workflow-guidelines/impact-evaluations", guidelineImpacts],
      ["/api/feedback-runs/events", feedbackEvents],
      ["/api/feedback-runs/review-cases", reviewCases],
      ["/api/training-runs", trainingRuns],
      ["/api/training/recipes", recipeDescriptors],
      ["/api/projects/1/base-models", baseModels],
      ["/api/training-recipes", projectRecipes],
      ["/api/training-recipes/versions", recipeVersions],
      ["/api/environments", environments],
      ["/api/storage/policies", storagePolicies],
    ]);
    if (!responses.has(url.pathname)) {
      await route.fallback();
      return;
    }
    requests.push({
      method: request.method(),
      pathname: url.pathname,
      search: url.search,
    });
    await json(route, responses.get(url.pathname));
  });
  return { requests, evidence };
}
