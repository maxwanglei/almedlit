"""API contracts for the canonical learning workflow."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskKind = Literal[
    "regression",
    "classification",
    "multilabel_classification",
    "token_labeling",
    "span_extraction",
    "relation_extraction",
    "ranking",
    "generation",
    "instruction_tuning",
]
AssistancePolicy = Literal[
    "blind",
    "reveal_after_first_pass",
    "immediate_suggestions",
    "critique",
    "micro_question",
]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadModel(BaseModel):
    id: int
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class ProjectModulesUpdate(InputModel):
    selected: list[
        Literal[
            "data",
            "annotate",
            "learning",
            "train",
            "models",
            "guidelines",
            "activity",
        ]
    ]


class ProjectModulesRead(InputModel):
    project_id: int
    selected: list[str]
    effective: list[str]
    workspace_capabilities: list[str]


class TrainingRecipeConfigurationCreate(InputModel):
    config: dict = Field(default_factory=dict)


class TaskDefinitionCreate(InputModel):
    project_id: int
    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class TaskDefinitionRead(TaskDefinitionCreate, ReadModel):
    created_by_user_id: int | None = None


class TaskVersionCreate(InputModel):
    project_id: int
    task_definition_id: int
    task_kind: TaskKind
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    label_rules: dict = Field(default_factory=dict)
    annotation_ui: dict = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    trainer_compatibility: list[str] = Field(default_factory=list)


class TaskVersionRead(TaskVersionCreate, ReadModel):
    version_number: int
    content_hash: str
    created_by_user_id: int | None = None


class DatasetCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    source_type: Literal["upload", "public_registry", "project_corpus", "generated", "other"]


class DatasetRead(DatasetCreate, ReadModel):
    created_by_user_id: int | None = None


class DatasetItemCreate(InputModel):
    stable_key: str = Field(min_length=1, max_length=255)
    group_key: str | None = Field(default=None, max_length=255)
    payload: dict


class DatasetItemRead(DatasetItemCreate, ReadModel):
    project_id: int
    dataset_version_id: int
    content_hash: str


class DatasetVersionCreate(InputModel):
    project_id: int
    dataset_id: int
    source_uri: str | None = Field(default=None, max_length=1024)
    source_revision: str = Field(min_length=1, max_length=255)
    source_format: Literal["csv", "jsonl", "parquet", "project_corpus", "other"]
    data_schema: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    license_info: dict = Field(default_factory=dict)
    artifact_package_id: int | None = None
    items: list[DatasetItemCreate] = Field(default_factory=list)


class DatasetVersionRead(ReadModel):
    project_id: int
    dataset_id: int
    version_number: int
    source_uri: str | None
    source_revision: str
    source_format: str
    data_schema: dict
    provenance: dict
    license_info: dict
    content_hash: str
    item_count: int
    artifact_package_id: int | None
    created_by_user_id: int | None


class PublicRegistryDatasetVersionCreate(InputModel):
    provider: Literal["hugging_face"] = "hugging_face"
    registry_dataset_id: str = Field(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$",
    )
    exact_revision: str = Field(
        min_length=40,
        max_length=40,
        pattern=r"^[0-9a-fA-F]{40}$",
    )
    config_name: str | None = Field(
        default=None,
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    source_format: Literal["csv", "jsonl", "parquet", "other"] = "other"
    data_schema: dict = Field(default_factory=dict)
    provenance: dict = Field(default_factory=dict)
    license_info: dict = Field(default_factory=dict)
    expected_content_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-fA-F]{64}$",
    )


class LabelSetVersionCreate(InputModel):
    project_id: int
    dataset_version_id: int
    task_version_id: int
    parent_version_id: int | None = None
    name: str = Field(min_length=1, max_length=255)
    source_kind: Literal["imported", "human", "adjudicated", "derived"]
    composition_policy: Literal["replace", "inherit", "exclude"] = "replace"
    labels: dict[str, Any] = Field(default_factory=dict)
    artifact_package_id: int | None = None


class LabelSetVersionRead(LabelSetVersionCreate, ReadModel):
    version_number: int
    label_count: int
    content_hash: str
    source_annotation_round_id: int | None
    source_submission_ids: list[int]
    source_decision_ids: list[int]
    created_by_user_id: int | None


class ImportedLabelSetFromFieldCreate(InputModel):
    project_id: int
    dataset_version_id: int
    task_version_id: int
    parent_version_id: int | None = None
    name: str = Field(min_length=1, max_length=255)
    label_field: str = Field(min_length=1, max_length=255)
    composition_policy: Literal["replace", "inherit"] = "replace"


class RoundLabelSetCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    source_kind: Literal["human", "adjudicated"]
    submission_ids: list[int] = Field(min_length=1)
    parent_version_id: int | None = None
    composition_policy: Literal["replace", "inherit"] = "replace"


class SplitMapCreate(InputModel):
    project_id: int
    dataset_version_id: int
    name: str = Field(min_length=1, max_length=255)
    strategy: str = Field(min_length=1, max_length=60)
    seed: int = 42
    group_key_field: str | None = Field(default=None, max_length=255)
    assignments: dict[str, Literal["pool", "train", "validation", "test"]]
    protected_splits: list[str] = Field(default_factory=lambda: ["test"])


class SplitMapRead(SplitMapCreate, ReadModel):
    content_hash: str
    created_by_user_id: int | None


class TrainingDatasetVersionCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: int
    task_version_id: int
    label_set_version_ids: list[int] = Field(min_length=1)
    split_map_id: int
    composition: list[dict] = Field(default_factory=list)
    preprocessing: dict = Field(default_factory=dict)
    artifact_package_id: int | None = None


class TrainingDatasetVersionRead(TrainingDatasetVersionCreate, ReadModel):
    content_hash: str
    created_by_user_id: int | None


class TrainingDatasetComposeCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: int
    task_version_id: int
    input_field: str = Field(min_length=1, max_length=255)
    label_field: str | None = Field(default=None, min_length=1, max_length=255)
    label_set_version_id: int | None = Field(default=None, ge=1)
    train_percent: float = Field(gt=0, lt=100)
    validation_percent: float = Field(gt=0, lt=100)
    seed: int = 42

    @field_validator("name", "input_field")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value cannot be blank")
        return normalized

    @field_validator("label_field")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("label_field cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_label_source_and_percentages(self):
        if (self.label_field is None) == (self.label_set_version_id is None):
            raise ValueError(
                "Exactly one of label_field or label_set_version_id is required"
            )
        if self.train_percent + self.validation_percent >= 100:
            raise ValueError(
                "Train and validation percentages must leave a positive test split"
            )
        return self


class TrainingDatasetComposeRead(InputModel):
    training_dataset_version: TrainingDatasetVersionRead
    label_set_version_id: int
    split_map_id: int
    split_counts: dict[str, int]
    group_count: int


class ComposedTrainingLabelsRead(InputModel):
    training_dataset_version_id: int
    labels: dict
    label_count: int
    content_hash: str
    composition: list[dict]


class LearningFeedbackSource(InputModel):
    producer_type: Literal[
        "external_llm",
        "rule",
        "dictionary",
        "ensemble",
        "human_disagreement",
    ]
    name: str = Field(min_length=1, max_length=255)
    provider: str | None = Field(default=None, max_length=120)
    external_model_id: str | None = Field(default=None, max_length=255)
    exact_revision: str | None = Field(default=None, max_length=255)
    configuration: dict = Field(default_factory=dict)
    data_egress_policy: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_external_source(self):
        if self.producer_type != "external_llm":
            return self
        if not all((self.provider, self.external_model_id, self.exact_revision)):
            raise ValueError(
                "External LLM cycle sources require provider, external_model_id, "
                "and exact_revision"
            )
        if not self.data_egress_policy:
            raise ValueError(
                "External LLM cycle sources require an approved data-egress policy"
            )
        return self


class LearningCycleCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    task_version_id: int
    source_dataset_version_id: int
    parent_cycle_id: int | None = None
    goal: Literal[
        "expand_pool",
        "reannotate",
        "guideline_pilot",
        "error_remediation",
    ] = "reannotate"
    baseline_model_version_id: int | None = None
    feedback_sources: list[LearningFeedbackSource] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_feedback_source(self):
        if self.baseline_model_version_id is None and not self.feedback_sources:
            raise ValueError(
                "A learning cycle requires a baseline model or another feedback source"
            )
        return self


class LearningCycleRead(ReadModel):
    project_id: int
    parent_cycle_id: int | None
    name: str
    sequence: int
    goal: str
    status: str
    current_stage: str
    task_version_id: int
    source_dataset_version_id: int
    baseline_model_version_id: int | None
    feedback_sources: list[LearningFeedbackSource]
    output_training_dataset_version_id: int | None
    output_model_version_id: int | None
    metadata: dict = Field(validation_alias="metadata_")
    created_by_user_id: int | None


class CycleTransition(InputModel):
    status: Literal["planned", "active", "completed", "cancelled"]
    current_stage: Literal[
        "pool",
        "select",
        "annotate",
        "resolve",
        "reflect",
        "publish_guideline",
        "train",
        "evaluate",
        "complete",
    ]
    output_training_dataset_version_id: int | None = None
    output_model_version_id: int | None = None


class SelectionRunCreate(InputModel):
    project_id: int
    dataset_version_id: int
    task_version_id: int
    cycle_id: int | None = None
    feedback_set_version_id: int | None = None
    split_map_id: int | None = None
    strategy: Literal[
        "all",
        "random",
        "uncertainty",
        "diversity",
        "disagreement",
        "error_based",
        "hybrid_uncertainty_diversity",
    ]
    parameters: dict = Field(default_factory=dict)
    eligibility_filter: dict = Field(default_factory=dict)
    seed: int = 42


class SelectionRunRead(SelectionRunCreate, ReadModel):
    status: str
    created_by_user_id: int | None


class SelectionItem(InputModel):
    dataset_item_id: int
    rank: int = Field(gt=0)
    score: float | None = None
    probability: float | None = Field(default=None, ge=0, le=1)
    reason: dict = Field(default_factory=dict)


class SelectionSetCreate(InputModel):
    project_id: int
    selection_run_id: int
    items: list[SelectionItem] = Field(min_length=1)


class SelectionSetRead(ReadModel):
    project_id: int
    selection_run_id: int
    version_number: int
    items: list[dict]
    item_count: int
    content_hash: str
    created_by_user_id: int | None


class FeedbackRunCreate(InputModel):
    project_id: int
    dataset_version_id: int
    task_version_id: int
    producer_type: Literal[
        "registered_model",
        "external_llm",
        "rule",
        "dictionary",
        "ensemble",
        "human_disagreement",
        "none",
    ]
    cycle_id: int | None = None
    model_version_id: int | None = None
    provider: str | None = Field(default=None, max_length=120)
    external_model_id: str | None = Field(default=None, max_length=255)
    exact_revision: str | None = Field(default=None, max_length=255)
    prompt_template_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    configuration: dict = Field(default_factory=dict)
    data_egress_policy: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_producer_identity(self):
        if self.producer_type == "registered_model" and self.model_version_id is None:
            raise ValueError("model_version_id is required for registered_model feedback")
        if self.producer_type == "external_llm":
            required = (
                self.provider,
                self.external_model_id,
                self.exact_revision,
                self.prompt_template_hash,
            )
            if not all(required):
                raise ValueError(
                    "provider, external_model_id, exact_revision, and "
                    "prompt_template_hash are required for external_llm feedback"
                )
            if not self.configuration:
                raise ValueError(
                    "external_llm feedback requires recorded decoding configuration"
                )
            if not self.data_egress_policy:
                raise ValueError(
                    "external_llm feedback requires a recorded data-egress policy"
                )
        return self


class FeedbackRunRead(FeedbackRunCreate, ReadModel):
    status: Literal["planned", "queued", "running", "completed", "failed"]
    output_feedback_set_version_id: int | None
    failure_code: str | None
    failure_reason: str | None
    started_at: datetime | None
    heartbeat_at: datetime | None
    completed_at: datetime | None
    created_by_user_id: int | None


class FeedbackCandidateCreate(InputModel):
    dataset_item_id: int
    candidate_key: str = Field(default="primary", min_length=1, max_length=120)
    output: Any
    score: float | None = None
    explanation: dict = Field(default_factory=dict)


class FeedbackSetCreate(InputModel):
    project_id: int
    feedback_run_id: int
    output_schema: dict = Field(default_factory=dict)
    artifact_package_id: int | None = None
    candidates: list[FeedbackCandidateCreate] = Field(default_factory=list)


class FeedbackSetRead(ReadModel):
    project_id: int
    feedback_run_id: int
    dataset_version_id: int
    task_version_id: int
    version_number: int
    output_schema: dict
    candidate_count: int
    content_hash: str
    artifact_package_id: int | None
    created_by_user_id: int | None


class FeedbackCandidateRead(FeedbackCandidateCreate, ReadModel):
    project_id: int
    feedback_set_version_id: int
    content_hash: str


class AnnotationRoundCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: int
    task_version_id: int
    parent_round_id: int | None = None
    cycle_id: int | None = None
    guideline_revision_id: int | None = None
    selection_set_version_id: int | None = None
    feedback_set_version_id: int | None = None
    assistance_policy: AssistancePolicy = "reveal_after_first_pass"
    reannotation_mode: Literal["targeted_subset", "full_dataset"] = "targeted_subset"
    annotator_user_ids: list[int] = Field(default_factory=list)
    open_to_all_annotators: bool = False
    reason: str | None = None


class AnnotationRoundRead(AnnotationRoundCreate, ReadModel):
    sequence: int
    status: str
    opened_at: datetime | None
    closed_at: datetime | None
    created_by_user_id: int | None


class RoundWorkRoundRead(ReadModel):
    project_id: int
    name: str
    sequence: int
    dataset_version_id: int
    task_version_id: int
    assistance_policy: AssistancePolicy
    feedback_available: bool
    status: str
    opened_at: datetime | None
    closed_at: datetime | None


class RoundWorkTaskLabel(InputModel):
    id: int
    key: str
    name: str


class RoundWorkCycleLabel(InputModel):
    id: int
    name: str
    sequence: int


class RoundWorkGuidelineLabel(InputModel):
    guideline_id: int
    guideline_revision_id: int
    name: str
    version_number: int
    status: str


class RoundWorkProjectLabel(InputModel):
    id: int
    name: str


class RoundWorkContextRead(InputModel):
    project: RoundWorkProjectLabel
    round: RoundWorkRoundRead
    task: RoundWorkTaskLabel
    task_version: TaskVersionRead
    cycle: RoundWorkCycleLabel | None = None
    guideline: RoundWorkGuidelineLabel | None = None


class AnnotationRoundTransition(InputModel):
    status: Literal["open", "closed", "cancelled"]


class RoundItemCreate(InputModel):
    dataset_item_id: int
    selection_rank: int | None = Field(default=None, gt=0)
    selection_score: float | None = None
    selection_probability: float | None = Field(default=None, ge=0, le=1)
    selection_reason: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


class RoundItemRead(ReadModel):
    project_id: int
    annotation_round_id: int
    dataset_item_id: int
    selection_rank: int | None
    selection_score: float | None
    selection_probability: float | None
    selection_reason: dict
    metadata: dict = Field(validation_alias="metadata_")


class RoundWorkItemIdentityRead(ReadModel):
    project_id: int
    annotation_round_id: int
    dataset_item_id: int
    selection_rank: int | None
    selection_score: float | None
    selection_reason: dict


class RoundWorkItemRead(InputModel):
    round_item: RoundWorkItemIdentityRead
    dataset_item: DatasetItemRead


class AnnotationDecisionCreate(InputModel):
    project_id: int
    round_item_id: int
    supersedes_decision_id: int | None = None
    output: Any
    decision_kind: Literal["annotation", "adjudication", "correction"] = "annotation"
    is_initial_checkpoint: bool = False
    rationale: str | None = None


class AnnotationDecisionRead(AnnotationDecisionCreate, ReadModel):
    annotator_user_id: int
    content_hash: str


class RoundSubmissionCreate(InputModel):
    project_id: int
    annotation_round_id: int
    decision_ids: list[int] = Field(min_length=1)


class RoundSubmissionRead(RoundSubmissionCreate, ReadModel):
    annotator_user_id: int
    sequence: int
    content_hash: str
    submitted_at: datetime


class FeedbackExposureCreate(InputModel):
    project_id: int
    round_item_id: int
    feedback_candidate_id: int
    exposure_mode: AssistancePolicy
    context: dict = Field(default_factory=dict)


class FeedbackExposureRead(FeedbackExposureCreate, ReadModel):
    user_id: int
    exposed_at: datetime


class FeedbackRevealRequest(InputModel):
    project_id: int
    candidate_key: str = Field(default="primary", min_length=1, max_length=120)
    context: dict = Field(default_factory=dict)


class FeedbackRevealRead(InputModel):
    exposure: FeedbackExposureRead
    candidate: FeedbackCandidateRead


class FeedbackDecisionCreate(InputModel):
    project_id: int
    feedback_exposure_id: int
    round_annotation_decision_id: int | None = None
    decision: Literal["accepted", "modified", "rejected", "ignored"]
    details: dict = Field(default_factory=dict)


class FeedbackDecisionRead(FeedbackDecisionCreate, ReadModel):
    user_id: int
    decided_at: datetime


class FeedbackEventCreate(InputModel):
    project_id: int
    event_type: Literal[
        "correction",
        "iaa_disagreement",
        "model_disagreement",
        "adjudication",
        "drift",
        "evaluation_failure",
        "llm_critique",
    ]
    cycle_id: int | None = None
    annotation_round_id: int | None = None
    round_item_id: int | None = None
    feedback_candidate_id: int | None = None
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    payload: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_structured_lineage(self):
        if not any(
            (
                self.cycle_id,
                self.annotation_round_id,
                self.round_item_id,
                self.feedback_candidate_id,
            )
        ):
            raise ValueError(
                "Feedback events require structured cycle, round, item, or candidate lineage"
            )
        raw_record_keys = {
            "text",
            "input",
            "output",
            "example",
            "examples",
            "record",
            "records",
            "prompt",
            "completion",
            "content",
        }

        def contains_raw_record(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    str(key).lower() in raw_record_keys
                    or contains_raw_record(item)
                    for key, item in value.items()
                )
            if isinstance(value, list):
                return any(contains_raw_record(item) for item in value)
            return False

        if contains_raw_record(self.payload):
            raise ValueError(
                "Feedback payloads cannot embed raw records; reference normalized lineage"
            )
        return self


class FeedbackEventRead(FeedbackEventCreate, ReadModel):
    occurred_at: datetime
    created_by_user_id: int | None


class ReviewCaseCreate(InputModel):
    project_id: int
    feedback_event_id: int
    case_type: str = Field(min_length=1, max_length=60)
    assigned_to_user_id: int | None = None


class ReviewCaseRead(ReviewCaseCreate, ReadModel):
    status: str
    resolution: dict
    resolved_at: datetime | None


class ReviewCaseResolve(InputModel):
    resolution: dict


class RegisteredModelCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Model name cannot be blank")
        return normalized


class RegisteredModelRead(RegisteredModelCreate, ReadModel):
    lifecycle_status: str
    created_by_user_id: int | None


class ModelVersionCreate(InputModel):
    project_id: int
    registered_model_id: int
    task_version_id: int
    training_dataset_version_id: int | None = None
    parent_version_id: int | None = None
    legacy_training_experiment_id: int | None = None
    family: str = Field(min_length=1, max_length=80)
    framework: str = Field(min_length=1, max_length=80)
    base_model: dict = Field(default_factory=dict)
    training_method: str = Field(min_length=1, max_length=60)
    recipe_key: str = Field(min_length=1, max_length=120)
    recipe_version: str = Field(min_length=1, max_length=60)
    parameters: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict)
    runtime_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = 42
    checkpoint_package_id: int | None = None


class ModelVersionRead(ModelVersionCreate, ReadModel):
    version_number: int
    content_hash: str
    created_by_user_id: int | None


class WorkspaceModelVersionRead(ModelVersionRead):
    training_run_id: int | None = None
    source_dataset_version_id: int | None = None
    source_dataset_version_number: int | None = None
    runtime_id: int | None = None
    runtime_name: str | None = None
    storage_policy_id: int | None = None
    storage_policy_name: str | None = None
    creator_username: str | None = None
    creator_display_name: str | None = None


class TrainingRecipeCreate(InputModel):
    project_id: int
    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class TrainingRecipeRead(TrainingRecipeCreate, ReadModel):
    created_by_user_id: int | None


class TrainingRecipeVersionCreate(InputModel):
    project_id: int
    training_recipe_id: int
    trainer_plugin_key: str = Field(min_length=1, max_length=120)
    trainer_plugin_version: str = Field(min_length=1, max_length=60)
    compatible_task_kinds: list[TaskKind] = Field(min_length=1)
    environment_class: str = Field(min_length=1, max_length=80)
    config_schema: dict = Field(default_factory=dict)
    default_config: dict = Field(default_factory=dict)
    evaluation_defaults: dict = Field(default_factory=dict)


class TrainingRecipeVersionRead(TrainingRecipeVersionCreate, ReadModel):
    version_number: int
    content_hash: str
    created_by_user_id: int | None


class ExecutionEnvironmentCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    environment_class: str = Field(min_length=1, max_length=80)
    image_digest: str = Field(
        min_length=64,
        max_length=71,
        pattern=r"^(?:sha256:)?[0-9a-fA-F]{64}$",
    )
    package_manifest: dict = Field(default_factory=dict)
    hardware_constraints: dict = Field(default_factory=dict)


class ExecutionEnvironmentRead(ExecutionEnvironmentCreate, ReadModel):
    status: str
    verification_report: dict
    verified_at: datetime | None
    created_by_user_id: int | None


class EnvironmentVerification(InputModel):
    status: Literal["available", "unavailable"]
    verification_report: dict


class StoragePolicyCreate(InputModel):
    project_id: int
    name: str = Field(min_length=1, max_length=255)
    backend: Literal["minio", "local"] = "minio"
    artifact_prefix: str = Field(min_length=1, max_length=512)
    retention_class: Literal["indefinite", "resume_14d"] = "indefinite"
    encryption: dict = Field(default_factory=dict)
    cache_policy: dict = Field(default_factory=dict)
    is_default: bool = False


class StoragePolicyRead(StoragePolicyCreate, ReadModel):
    created_by_user_id: int | None


class TrainingRunCreate(InputModel):
    project_id: int
    registered_model_id: int
    task_version_id: int
    training_dataset_version_id: int
    recipe_version_id: int
    environment_id: int
    storage_policy_id: int
    idempotency_key: str = Field(min_length=1, max_length=160)
    parent_model_version_id: int | None = None
    evaluation_plan: dict = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)
    seed: int = 42
    artifact_reservation_bytes: int | None = Field(default=None, ge=1)


class TrainingRunRead(TrainingRunCreate, ReadModel):
    artifact_reservation_id: int | None
    status: str
    launch_hash: str
    output_model_version_id: int | None
    runtime_snapshot: dict
    storage_snapshot: dict
    failure_code: str | None
    failure_reason: str | None
    started_at: datetime | None
    completed_at: datetime | None
    created_by_user_id: int | None


class TrainingRunTransition(InputModel):
    status: Literal["running", "succeeded", "failed", "cancelled"]
    output_model_version_id: int | None = None
    runtime_snapshot: dict | None = None
    storage_snapshot: dict | None = None
    failure_code: str | None = None
    failure_reason: str | None = None


class TrainingRunCancellation(InputModel):
    status: Literal["cancelled"]


class ModelEvaluationRead(ReadModel):
    project_id: int
    training_run_id: int
    model_version_id: int
    task_version_id: int
    training_dataset_version_id: int
    dataset_version_id: int
    split_map_id: int
    artifact_package_id: int | None
    split_name: str
    status: Literal["succeeded", "unsupported", "failed"]
    evaluator_key: str | None
    evaluator_version: str | None
    row_count: int
    requested_metrics: list[str]
    metrics: dict[str, float | None]
    report: dict
    evaluation_plan: dict
    runtime_digest: str
    code_digest: str
    status_reason: str | None
    content_hash: str
    created_by_user_id: int | None


class WorkspaceTrainingContextRead(InputModel):
    project_id: int
    project_name: str
    project_description: str | None
    training_only: bool
    effective_modules: list[str]
    task_version_count: int
    training_dataset_version_count: int
    environment_count: int
    available_environment_count: int
    storage_policy_count: int


class WorkspaceTrainingContextPageRead(InputModel):
    items: list[WorkspaceTrainingContextRead]
    next_cursor: int | None = None


class WorkspaceTrainingRunRead(TrainingRunRead):
    project_name: str
    project_description: str | None
    model_name: str
    family: str
    framework: str
    base_model: dict
    training_method: str
    task_name: str
    task_kind: TaskKind
    training_dataset_name: str
    dataset_version_number: int
    recipe_name: str
    runtime_name: str
    storage_policy_name: str
    evaluation_status: Literal["succeeded", "unsupported", "failed"] | None = None
    evaluation_split: str | None = None
    evaluation_metrics: dict[str, float | None] = Field(default_factory=dict)
    creator_username: str | None = None
    creator_display_name: str | None = None


class WorkspaceTrainingRunPageRead(InputModel):
    items: list[WorkspaceTrainingRunRead]
    next_cursor: int | None = None


class WorkspaceRegisteredModelRead(RegisteredModelRead):
    project_name: str
    project_description: str | None
    latest_version: WorkspaceModelVersionRead | None = None
    task_name: str | None = None
    task_kind: TaskKind | None = None
    training_dataset_name: str | None = None
    evaluation_status: Literal["succeeded", "unsupported", "failed"] | None = None
    evaluation_split: str | None = None
    evaluation_metrics: dict[str, float | None] = Field(default_factory=dict)
    creator_username: str | None = None
    creator_display_name: str | None = None


class WorkspaceRegisteredModelPageRead(InputModel):
    items: list[WorkspaceRegisteredModelRead]
    next_cursor: int | None = None


class GuidelineCreate(InputModel):
    project_id: int
    task_definition_id: int
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None


class GuidelineRead(GuidelineCreate, ReadModel):
    created_by_user_id: int | None


class GuidelineRevisionCreate(InputModel):
    project_id: int
    guideline_id: int
    task_version_id: int
    parent_revision_id: int | None = None
    markdown: str
    rationale: str | None = None
    diff_summary: dict = Field(default_factory=dict)
    source_proposal_ids: list[int] = Field(default_factory=list)


class GuidelineRevisionRead(GuidelineRevisionCreate, ReadModel):
    version_number: int
    content_hash: str
    status: str
    approved_by_user_id: int | None
    approved_at: datetime | None
    created_by_user_id: int | None


class GuidelineRevisionTransition(InputModel):
    status: Literal["pilot", "active", "retired"]


class GuidelineProposalCreate(InputModel):
    project_id: int
    guideline_id: int
    base_revision_id: int | None = None
    feedback_event_ids: list[int] = Field(min_length=1)
    proposed_change: dict
    rationale: str = Field(min_length=1)


class GuidelineProposalRead(GuidelineProposalCreate, ReadModel):
    status: str
    reviewed_by_user_id: int | None
    reviewed_at: datetime | None
    resulting_revision_id: int | None
    created_by_user_id: int | None


class GuidelineProposalReview(InputModel):
    decision: Literal["accepted", "rejected"]


class GuidelineImpactCreate(InputModel):
    project_id: int
    guideline_revision_id: int
    pilot_round_id: int
    protected_split_map_id: int | None = None


class GuidelineImpactRead(GuidelineImpactCreate, ReadModel):
    status: str
    metrics: dict
    passed: bool | None
    completed_at: datetime | None
    reviewed_by_user_id: int | None
    created_by_user_id: int | None


class GuidelineImpactComplete(InputModel):
    minimum_agreement: float = Field(default=0.8, ge=0, le=1)
