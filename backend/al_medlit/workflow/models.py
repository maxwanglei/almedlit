"""Persistent models for the canonical learning workflow."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.orm import Mapped, mapped_column

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin, utc_now
from al_medlit.core.types import JSONType
from al_medlit.lineage.models import ImmutableRecordError


class TaskDefinition(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "task_definitions"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_task_definitions_project_key"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    key: Mapped[str] = mapped_column(String(120))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class TaskVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "task_versions"
    __table_args__ = (
        UniqueConstraint(
            "task_definition_id",
            "version_number",
            name="uq_task_versions_definition_version",
        ),
        UniqueConstraint(
            "task_definition_id",
            "content_hash",
            name="uq_task_versions_definition_hash",
        ),
        CheckConstraint("version_number > 0", name="ck_task_versions_positive_version"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    task_definition_id: Mapped[int] = mapped_column(
        ForeignKey("task_definitions.id"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    task_kind: Mapped[str] = mapped_column(String(60), index=True)
    input_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    output_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    label_rules: Mapped[dict] = mapped_column(JSONType, default=dict)
    annotation_ui: Mapped[dict] = mapped_column(JSONType, default=dict)
    metrics: Mapped[list] = mapped_column(JSONType, default=list)
    trainer_compatibility: Mapped[list] = mapped_column(JSONType, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class Dataset(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "datasets"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_datasets_project_name"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(40), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class DatasetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "dataset_versions"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "version_number",
            name="uq_dataset_versions_dataset_version",
        ),
        UniqueConstraint(
            "dataset_id",
            "content_hash",
            name="uq_dataset_versions_dataset_hash",
        ),
        CheckConstraint("version_number > 0", name="ck_dataset_versions_positive_version"),
        CheckConstraint("item_count >= 0", name="ck_dataset_versions_nonnegative_count"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    source_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    source_revision: Mapped[str] = mapped_column(String(255))
    source_format: Mapped[str] = mapped_column(String(40))
    data_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    provenance: Mapped[dict] = mapped_column(JSONType, default=dict)
    license_info: Mapped[dict] = mapped_column(JSONType, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class DatasetItem(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "dataset_items"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "stable_key",
            name="uq_dataset_items_version_stable_key",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    stable_key: Mapped[str] = mapped_column(String(255))
    group_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class LabelSetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "label_set_versions"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "task_version_id",
            "name",
            "version_number",
            name="uq_label_set_versions_identity",
        ),
        CheckConstraint("version_number > 0", name="ck_label_set_versions_positive_version"),
        CheckConstraint("label_count >= 0", name="ck_label_set_versions_nonnegative_count"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("label_set_versions.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    version_number: Mapped[int] = mapped_column(Integer)
    source_kind: Mapped[str] = mapped_column(String(40), index=True)
    composition_policy: Mapped[str] = mapped_column(String(20), default="replace")
    labels: Mapped[dict] = mapped_column(JSONType, default=dict)
    label_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    source_annotation_round_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_rounds.id"), nullable=True, index=True
    )
    source_submission_ids: Mapped[list] = mapped_column(JSONType, default=list)
    source_decision_ids: Mapped[list] = mapped_column(JSONType, default=list)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class SplitMap(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "split_maps"
    __table_args__ = (
        UniqueConstraint(
            "dataset_version_id",
            "name",
            name="uq_split_maps_dataset_name",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    strategy: Mapped[str] = mapped_column(String(60))
    seed: Mapped[int] = mapped_column(Integer, default=42)
    group_key_field: Mapped[str | None] = mapped_column(String(255), nullable=True)
    assignments: Mapped[dict] = mapped_column(JSONType, default=dict)
    protected_splits: Mapped[list] = mapped_column(JSONType, default=lambda: ["test"])
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class TrainingDatasetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_dataset_versions"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "content_hash",
            name="uq_training_dataset_versions_project_hash",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    label_set_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    split_map_id: Mapped[int] = mapped_column(ForeignKey("split_maps.id"), index=True)
    composition: Mapped[list] = mapped_column(JSONType, default=list)
    preprocessing: Mapped[dict] = mapped_column(JSONType, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class LearningCycle(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "learning_cycles"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "sequence",
            name="uq_learning_cycles_project_sequence",
        ),
        CheckConstraint("sequence > 0", name="ck_learning_cycles_positive_sequence"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    parent_cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("learning_cycles.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    goal: Mapped[str] = mapped_column(String(40), default="reannotate", index=True)
    status: Mapped[str] = mapped_column(String(30), default="planned", index=True)
    current_stage: Mapped[str] = mapped_column(String(40), default="pool", index=True)
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    source_dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    baseline_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    feedback_sources: Mapped[list] = mapped_column(JSONType, default=list)
    output_training_dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_dataset_versions.id"), nullable=True, index=True
    )
    output_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class SelectionRun(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "selection_runs"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("learning_cycles.id"), nullable=True, index=True
    )
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    feedback_set_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("feedback_set_versions.id"), nullable=True, index=True
    )
    split_map_id: Mapped[int | None] = mapped_column(
        ForeignKey("split_maps.id"), nullable=True, index=True
    )
    strategy: Mapped[str] = mapped_column(String(60), index=True)
    parameters: Mapped[dict] = mapped_column(JSONType, default=dict)
    eligibility_filter: Mapped[dict] = mapped_column(JSONType, default=dict)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String(30), default="planned", index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class SelectionSetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "selection_set_versions"
    __table_args__ = (
        UniqueConstraint(
            "selection_run_id",
            "version_number",
            name="uq_selection_set_versions_run_version",
        ),
        CheckConstraint(
            "version_number > 0", name="ck_selection_set_versions_positive_version"
        ),
        CheckConstraint("item_count >= 0", name="ck_selection_set_versions_nonnegative_count"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    selection_run_id: Mapped[int] = mapped_column(
        ForeignKey("selection_runs.id"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    items: Mapped[list] = mapped_column(JSONType, default=list)
    item_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class FeedbackRun(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feedback_runs"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("learning_cycles.id"), nullable=True, index=True
    )
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    producer_type: Mapped[str] = mapped_column(String(40), index=True)
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    provider: Mapped[str | None] = mapped_column(String(120), nullable=True)
    external_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exact_revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_template_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    configuration: Mapped[dict] = mapped_column(JSONType, default=dict)
    data_egress_policy: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="planned", index=True)
    output_feedback_set_version_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "feedback_set_versions.id",
            name="fk_feedback_runs_output_feedback_set_version_id",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class FeedbackSetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feedback_set_versions"
    __table_args__ = (
        UniqueConstraint(
            "feedback_run_id",
            "version_number",
            name="uq_feedback_set_versions_run_version",
        ),
        CheckConstraint(
            "version_number > 0", name="ck_feedback_set_versions_positive_version"
        ),
        CheckConstraint("candidate_count >= 0", name="ck_feedback_sets_nonnegative_count"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    feedback_run_id: Mapped[int] = mapped_column(ForeignKey("feedback_runs.id"), index=True)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    output_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class FeedbackCandidate(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "feedback_candidates"
    __table_args__ = (
        UniqueConstraint(
            "feedback_set_version_id",
            "dataset_item_id",
            "candidate_key",
            name="uq_feedback_candidates_set_item_key",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    feedback_set_version_id: Mapped[int] = mapped_column(
        ForeignKey("feedback_set_versions.id"), index=True
    )
    dataset_item_id: Mapped[int] = mapped_column(ForeignKey("dataset_items.id"), index=True)
    candidate_key: Mapped[str] = mapped_column(String(120), default="primary")
    output: Mapped[dict] = mapped_column(JSONType, default=dict)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    explanation: Mapped[dict] = mapped_column(JSONType, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class AnnotationRound(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "annotation_rounds"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "cycle_id",
            "sequence",
            name="uq_annotation_rounds_cycle_sequence",
        ),
        CheckConstraint("sequence > 0", name="ck_annotation_rounds_positive_sequence"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    parent_round_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_rounds.id"), nullable=True, index=True
    )
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("learning_cycles.id"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    sequence: Mapped[int] = mapped_column(Integer)
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    guideline_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_revisions.id"), nullable=True, index=True
    )
    selection_set_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("selection_set_versions.id"), nullable=True, index=True
    )
    feedback_set_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("feedback_set_versions.id"), nullable=True, index=True
    )
    assistance_policy: Mapped[str] = mapped_column(
        String(40), default="reveal_after_first_pass", index=True
    )
    reannotation_mode: Mapped[str] = mapped_column(
        String(30), default="targeted_subset", index=True
    )
    annotator_user_ids: Mapped[list] = mapped_column(JSONType, default=list)
    open_to_all_annotators: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        index=True,
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="draft", index=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class RoundItem(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "round_items"
    __table_args__ = (
        UniqueConstraint(
            "annotation_round_id",
            "dataset_item_id",
            name="uq_round_items_round_item",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    annotation_round_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_rounds.id"), index=True
    )
    dataset_item_id: Mapped[int] = mapped_column(ForeignKey("dataset_items.id"), index=True)
    selection_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selection_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    selection_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    selection_reason: Mapped[dict] = mapped_column(JSONType, default=dict)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)


class RoundAnnotationDecision(Base, IntPrimaryKeyMixin):
    __tablename__ = "round_annotation_decisions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    round_item_id: Mapped[int] = mapped_column(ForeignKey("round_items.id"), index=True)
    annotator_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    supersedes_decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("round_annotation_decisions.id"), nullable=True, index=True
    )
    output: Mapped[dict] = mapped_column(JSONType, default=dict)
    decision_kind: Mapped[str] = mapped_column(String(30), default="annotation", index=True)
    is_initial_checkpoint: Mapped[bool] = mapped_column(Boolean, default=False)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )


class RoundSubmission(Base, IntPrimaryKeyMixin):
    __tablename__ = "round_submissions"
    __table_args__ = (
        UniqueConstraint(
            "annotation_round_id",
            "annotator_user_id",
            "sequence",
            name="uq_round_submissions_round_annotator_sequence",
        ),
        CheckConstraint("sequence > 0", name="ck_round_submissions_positive_sequence"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    annotation_round_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_rounds.id"), index=True
    )
    annotator_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    sequence: Mapped[int] = mapped_column(Integer, default=1)
    decision_ids: Mapped[list] = mapped_column(JSONType, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    submitted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )


class FeedbackExposure(Base, IntPrimaryKeyMixin):
    __tablename__ = "feedback_exposures"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    round_item_id: Mapped[int] = mapped_column(ForeignKey("round_items.id"), index=True)
    feedback_candidate_id: Mapped[int] = mapped_column(
        ForeignKey("feedback_candidates.id"), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    exposure_mode: Mapped[str] = mapped_column(String(40), index=True)
    context: Mapped[dict] = mapped_column(JSONType, default=dict)
    exposed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )


class FeedbackDecision(Base, IntPrimaryKeyMixin):
    __tablename__ = "feedback_decisions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    feedback_exposure_id: Mapped[int] = mapped_column(
        ForeignKey("feedback_exposures.id"), index=True
    )
    round_annotation_decision_id: Mapped[int | None] = mapped_column(
        ForeignKey("round_annotation_decisions.id"), nullable=True, index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    decision: Mapped[str] = mapped_column(String(30), index=True)
    details: Mapped[dict] = mapped_column(JSONType, default=dict)
    decided_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )


class FeedbackEvent(Base, IntPrimaryKeyMixin):
    __tablename__ = "feedback_events"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    cycle_id: Mapped[int | None] = mapped_column(
        ForeignKey("learning_cycles.id"), nullable=True, index=True
    )
    annotation_round_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_rounds.id"), nullable=True, index=True
    )
    round_item_id: Mapped[int | None] = mapped_column(
        ForeignKey("round_items.id"), nullable=True, index=True
    )
    feedback_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("feedback_candidates.id"), nullable=True, index=True
    )
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="medium", index=True)
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class ReviewCase(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "review_cases"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    feedback_event_id: Mapped[int] = mapped_column(ForeignKey("feedback_events.id"), index=True)
    assigned_to_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    case_type: Mapped[str] = mapped_column(String(60), index=True)
    status: Mapped[str] = mapped_column(String(30), default="open", index=True)
    resolution: Mapped[dict] = mapped_column(JSONType, default=dict)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class RegisteredModel(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "registered_models"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "name",
            name="uq_registered_models_project_name",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    lifecycle_status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class ModelVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "model_versions"
    __table_args__ = (
        UniqueConstraint(
            "registered_model_id",
            "version_number",
            name="uq_model_versions_model_version",
        ),
        CheckConstraint("version_number > 0", name="ck_model_versions_positive_version"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    registered_model_id: Mapped[int] = mapped_column(
        ForeignKey("registered_models.id"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    parent_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    training_dataset_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_dataset_versions.id"), nullable=True, index=True
    )
    legacy_training_experiment_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_experiments.id"), nullable=True, index=True
    )
    family: Mapped[str] = mapped_column(String(80), index=True)
    framework: Mapped[str] = mapped_column(String(80), index=True)
    base_model: Mapped[dict] = mapped_column(JSONType, default=dict)
    training_method: Mapped[str] = mapped_column(String(60), index=True)
    recipe_key: Mapped[str] = mapped_column(String(120))
    recipe_version: Mapped[str] = mapped_column(String(60))
    parameters: Mapped[dict] = mapped_column(JSONType, default=dict)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    runtime_digest: Mapped[str] = mapped_column(String(64))
    code_digest: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int] = mapped_column(Integer, default=42)
    checkpoint_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class TrainingRecipe(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_recipes"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_training_recipes_project_key"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    key: Mapped[str] = mapped_column(String(120))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class TrainingRecipeVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_recipe_versions"
    __table_args__ = (
        UniqueConstraint(
            "training_recipe_id",
            "version_number",
            name="uq_training_recipe_versions_recipe_version",
        ),
        UniqueConstraint(
            "training_recipe_id",
            "content_hash",
            name="uq_training_recipe_versions_recipe_hash",
        ),
        CheckConstraint(
            "version_number > 0", name="ck_training_recipe_versions_positive_version"
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    training_recipe_id: Mapped[int] = mapped_column(
        ForeignKey("training_recipes.id"), index=True
    )
    version_number: Mapped[int] = mapped_column(Integer)
    trainer_plugin_key: Mapped[str] = mapped_column(String(120), index=True)
    trainer_plugin_version: Mapped[str] = mapped_column(String(60))
    compatible_task_kinds: Mapped[list] = mapped_column(JSONType, default=list)
    environment_class: Mapped[str] = mapped_column(String(80), index=True)
    config_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    default_config: Mapped[dict] = mapped_column(JSONType, default=dict)
    evaluation_defaults: Mapped[dict] = mapped_column(JSONType, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class ExecutionEnvironment(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "execution_environments"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "name",
            name="uq_execution_environments_project_name",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    environment_class: Mapped[str] = mapped_column(String(80), index=True)
    image_digest: Mapped[str | None] = mapped_column(String(255), nullable=True)
    package_manifest: Mapped[dict] = mapped_column(JSONType, default=dict)
    hardware_constraints: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="unverified", index=True)
    verification_report: Mapped[dict] = mapped_column(JSONType, default=dict)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class StoragePolicy(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "storage_policies"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_storage_policies_project_name"),
        CheckConstraint(
            "backend IN ('minio', 'local')",
            name="ck_storage_policies_executable_backend",
        ),
        CheckConstraint(
            "retention_class IN ('indefinite', 'resume_14d')",
            name="ck_storage_policies_supported_retention",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    backend: Mapped[str] = mapped_column(String(40), default="minio", index=True)
    artifact_prefix: Mapped[str] = mapped_column(String(512))
    retention_class: Mapped[str] = mapped_column(String(40), default="indefinite")
    encryption: Mapped[dict] = mapped_column(JSONType, default=dict)
    cache_policy: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class TrainingRun(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_runs"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_training_runs_project_idempotency",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    registered_model_id: Mapped[int] = mapped_column(
        ForeignKey("registered_models.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    training_dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("training_dataset_versions.id"), index=True
    )
    recipe_version_id: Mapped[int] = mapped_column(
        ForeignKey("training_recipe_versions.id"), index=True
    )
    parent_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    environment_id: Mapped[int] = mapped_column(
        ForeignKey("execution_environments.id"), index=True
    )
    storage_policy_id: Mapped[int] = mapped_column(
        ForeignKey("storage_policies.id"), index=True
    )
    artifact_reservation_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_storage_reservations.id"),
        nullable=True,
        unique=True,
        index=True,
    )
    evaluation_plan: Mapped[dict] = mapped_column(JSONType, default=dict)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160))
    launch_hash: Mapped[str] = mapped_column(String(64), index=True)
    output_model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True, index=True
    )
    runtime_snapshot: Mapped[dict] = mapped_column(JSONType, default=dict)
    storage_snapshot: Mapped[dict] = mapped_column(JSONType, default=dict)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class ModelEvaluation(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "model_evaluations"
    __table_args__ = (
        UniqueConstraint(
            "training_run_id",
            "split_name",
            name="uq_model_evaluations_run_split",
        ),
        UniqueConstraint(
            "artifact_package_id",
            name="uq_model_evaluations_artifact_package",
        ),
        CheckConstraint(
            "row_count >= 0",
            name="ck_model_evaluations_nonnegative_row_count",
        ),
        CheckConstraint(
            "status IN ('succeeded', 'unsupported', 'failed')",
            name="ck_model_evaluations_status",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id"), index=True
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), index=True
    )
    task_version_id: Mapped[int] = mapped_column(
        ForeignKey("task_versions.id"), index=True
    )
    training_dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("training_dataset_versions.id"), index=True
    )
    dataset_version_id: Mapped[int] = mapped_column(
        ForeignKey("dataset_versions.id"), index=True
    )
    split_map_id: Mapped[int] = mapped_column(
        ForeignKey("split_maps.id"), index=True
    )
    artifact_package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    split_name: Mapped[str] = mapped_column(String(40), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    evaluator_key: Mapped[str | None] = mapped_column(
        String(120), nullable=True, index=True
    )
    evaluator_version: Mapped[str | None] = mapped_column(
        String(60), nullable=True
    )
    row_count: Mapped[int] = mapped_column(Integer)
    requested_metrics: Mapped[list] = mapped_column(JSONType, default=list)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    report: Mapped[dict] = mapped_column(JSONType, default=dict)
    evaluation_plan: Mapped[dict] = mapped_column(JSONType, default=dict)
    runtime_digest: Mapped[str] = mapped_column(String(64), index=True)
    code_digest: Mapped[str] = mapped_column(String(64), index=True)
    status_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class Guideline(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guidelines"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "task_definition_id",
            "name",
            name="uq_guidelines_project_task_name",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    task_definition_id: Mapped[int] = mapped_column(
        ForeignKey("task_definitions.id"), index=True
    )
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class GuidelineRevision(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guideline_revisions"
    __table_args__ = (
        UniqueConstraint(
            "guideline_id",
            "version_number",
            name="uq_guideline_revisions_guideline_version",
        ),
        UniqueConstraint(
            "guideline_id",
            "content_hash",
            name="uq_guideline_revisions_guideline_hash",
        ),
        CheckConstraint("version_number > 0", name="ck_guideline_revisions_positive_version"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    guideline_id: Mapped[int] = mapped_column(ForeignKey("guidelines.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    parent_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_revisions.id"), nullable=True, index=True
    )
    task_version_id: Mapped[int] = mapped_column(ForeignKey("task_versions.id"), index=True)
    markdown: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    diff_summary: Mapped[dict] = mapped_column(JSONType, default=dict)
    source_proposal_ids: Mapped[list] = mapped_column(JSONType, default=list)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)
    approved_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class GuidelineChangeProposal(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guideline_change_proposals"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    guideline_id: Mapped[int] = mapped_column(ForeignKey("guidelines.id"), index=True)
    base_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_revisions.id"), nullable=True, index=True
    )
    feedback_event_ids: Mapped[list] = mapped_column(JSONType, default=list)
    proposed_change: Mapped[dict] = mapped_column(JSONType, default=dict)
    rationale: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    reviewed_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resulting_revision_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_revisions.id"), nullable=True, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


class GuidelineImpactEvaluation(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guideline_impact_evaluations"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    guideline_revision_id: Mapped[int] = mapped_column(
        ForeignKey("guideline_revisions.id"), index=True
    )
    pilot_round_id: Mapped[int] = mapped_column(ForeignKey("annotation_rounds.id"), index=True)
    protected_split_map_id: Mapped[int | None] = mapped_column(
        ForeignKey("split_maps.id"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewed_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )


_FULLY_IMMUTABLE_TYPES = (
    TaskVersion,
    DatasetVersion,
    DatasetItem,
    LabelSetVersion,
    SplitMap,
    TrainingDatasetVersion,
    SelectionSetVersion,
    FeedbackSetVersion,
    FeedbackCandidate,
    RoundItem,
    RoundAnnotationDecision,
    RoundSubmission,
    FeedbackExposure,
    FeedbackDecision,
    FeedbackEvent,
    ModelVersion,
    ModelEvaluation,
    TrainingRecipeVersion,
)


def _reject_immutable_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(
        f"{type(target).__name__} {getattr(target, 'id', None)} is immutable"
    )


for _immutable_type in _FULLY_IMMUTABLE_TYPES:
    event.listen(_immutable_type, "before_update", _reject_immutable_mutation)
    event.listen(_immutable_type, "before_delete", _reject_immutable_mutation)


_GUIDELINE_MUTABLE_FIELDS = {
    "status",
    "approved_by_user_id",
    "approved_at",
    "updated_at",
}


def _protect_guideline_content(_mapper, _connection, target: GuidelineRevision) -> None:
    state = inspect(target)
    changed = {
        attribute.key
        for attribute in state.mapper.column_attrs
        if state.attrs[attribute.key].history.has_changes()
    }
    immutable_changes = changed - _GUIDELINE_MUTABLE_FIELDS
    if immutable_changes:
        fields = ", ".join(sorted(immutable_changes))
        raise ImmutableRecordError(
            f"GuidelineRevision {target.id} has immutable changes: {fields}"
        )


event.listen(GuidelineRevision, "before_update", _protect_guideline_content)
event.listen(GuidelineRevision, "before_delete", _reject_immutable_mutation)
