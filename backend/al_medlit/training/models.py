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
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin, utc_now
from al_medlit.core.types import JSONType
from al_medlit.lineage.models import ImmutableRecordError


class ComputeProfile(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "compute_profiles"
    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_compute_profiles_project_name"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    backend: Mapped[str] = mapped_column(String(40), index=True)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="active", index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )


class TrainingExperiment(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_experiments"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_training_experiments_project_idempotency",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    annotation_set_id: Mapped[int] = mapped_column(ForeignKey("annotation_sets.id"), index=True)
    dataset_export_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("export_artifacts.id"),
        nullable=True,
        index=True,
    )
    compute_profile_id: Mapped[int] = mapped_column(ForeignKey("compute_profiles.id"))
    name: Mapped[str] = mapped_column(String(255))
    model_type: Mapped[str] = mapped_column(String(100), index=True)
    model_family: Mapped[str] = mapped_column(
        String(40), default="deep_learning", index=True
    )
    task_contract_key: Mapped[str] = mapped_column(
        String(120), default="evidence_blocks", index=True
    )
    task_contract_version: Mapped[str] = mapped_column(String(40), default="1")
    target_scope_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    dataset_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    preprocessing_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    configuration_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    seed: Mapped[int] = mapped_column(Integer, default=42)
    result_schema_version: Mapped[str] = mapped_column(
        String(40), default="training-result-v2"
    )
    mode: Mapped[str] = mapped_column(String(30), index=True)
    target_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(120))
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    annotation_set = relationship("AnnotationSet")
    dataset_export_artifact = relationship("ExportArtifact")
    compute_profile = relationship("ComputeProfile")
    jobs = relationship(
        "TrainingJob",
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="TrainingJob.id",
    )


class TrainingJob(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_jobs"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_training_jobs_idempotency_key"),
        CheckConstraint(
            "metric_sequence >= 0",
            name="ck_training_jobs_metric_sequence_nonnegative",
        ),
        CheckConstraint(
            "event_sequence >= 0",
            name="ck_training_jobs_event_sequence_nonnegative",
        ),
    )

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("training_experiments.id"),
        index=True,
    )
    target_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("evidence_target_versions.id"),
        nullable=True,
        index=True,
    )
    compute_profile_id: Mapped[int] = mapped_column(
        ForeignKey("compute_profiles.id"),
        index=True,
    )
    artifact_reservation_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_storage_reservations.id"),
        nullable=True,
        unique=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160))
    external_job_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    bundle_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        nullable=True,
        index=True,
    )
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finalizing_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finalization_claim_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    finalization_claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    state_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    metric_sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    event_sequence: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    failure_code: Mapped[str | None] = mapped_column(String(80), nullable=True, index=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    runtime_metadata: Mapped[dict] = mapped_column(JSONType, default=dict)

    experiment = relationship("TrainingExperiment", back_populates="jobs")
    compute_profile = relationship("ComputeProfile")
    checkpoints = relationship(
        "ModelCheckpoint",
        back_populates="training_job",
        cascade="all, delete-orphan",
    )
    metric_points = relationship(
        "TrainingMetricPoint",
        back_populates="training_job",
        cascade="all, delete-orphan",
        order_by="TrainingMetricPoint.sequence",
    )
    events = relationship(
        "TrainingJobEvent",
        back_populates="training_job",
        cascade="all, delete-orphan",
        order_by="TrainingJobEvent.sequence",
    )
    artifact_links = relationship(
        "TrainingJobArtifact",
        back_populates="training_job",
        cascade="all, delete-orphan",
    )

    # PostgreSQL row locks are used around lifecycle transitions, while this
    # version column also gives the SQLite test path a real compare-and-swap
    # guard against stale ORM writes.
    __mapper_args__ = {"version_id_col": state_version}


class ModelCheckpoint(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "model_checkpoints"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    training_job_id: Mapped[int] = mapped_column(ForeignKey("training_jobs.id"), index=True)
    artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        unique=True,
        index=True,
    )
    model_type: Mapped[str] = mapped_column(String(100), index=True)
    training_mode: Mapped[str] = mapped_column(String(30), index=True)
    trained_target_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    base_model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    base_model_revision: Mapped[str | None] = mapped_column(String(255), nullable=True)
    max_context_tokens: Mapped[int] = mapped_column(Integer)
    manifest: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_primary: Mapped[bool] = mapped_column(default=True)
    readiness: Mapped[str] = mapped_column(String(40), default="ready", index=True)
    package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )
    task_contract_key: Mapped[str] = mapped_column(
        String(120), default="evidence_blocks", index=True
    )
    task_contract_version: Mapped[str] = mapped_column(String(40), default="1")
    target_scope_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    dataset_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    evaluation_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    selection_metric_key: Mapped[str | None] = mapped_column(String(120), nullable=True)
    selection_metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_step: Mapped[int | None] = mapped_column(Integer, nullable=True)

    training_job = relationship("TrainingJob", back_populates="checkpoints")
    artifact = relationship("LineageArtifact")
    package = relationship("ArtifactPackage")
    evaluations = relationship(
        "EvaluationResult",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )
    pins = relationship(
        "ModelCheckpointPin",
        back_populates="checkpoint",
        cascade="all, delete-orphan",
    )


class TrainingMetricPoint(Base, IntPrimaryKeyMixin):
    __tablename__ = "training_metric_points"
    __table_args__ = (
        UniqueConstraint(
            "training_job_id", "sequence", name="uq_training_metric_points_job_sequence"
        ),
        CheckConstraint(
            "sequence > 0",
            name="ck_training_metric_points_sequence_positive",
        ),
    )

    training_job_id: Mapped[int] = mapped_column(
        ForeignKey("training_jobs.id"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    phase: Mapped[str] = mapped_column(String(40), default="train", index=True)
    split: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    epoch: Mapped[float | None] = mapped_column(Float, nullable=True)
    step: Mapped[int | None] = mapped_column(Integer, nullable=True)
    values: Mapped[dict] = mapped_column(JSONType, default=dict)

    training_job = relationship("TrainingJob", back_populates="metric_points")


class TrainingJobEvent(Base, IntPrimaryKeyMixin):
    __tablename__ = "training_job_events"
    __table_args__ = (
        UniqueConstraint(
            "training_job_id", "sequence", name="uq_training_job_events_job_sequence"
        ),
        CheckConstraint(
            "sequence > 0",
            name="ck_training_job_events_sequence_positive",
        ),
    )

    training_job_id: Mapped[int] = mapped_column(
        ForeignKey("training_jobs.id"), index=True
    )
    sequence: Mapped[int] = mapped_column(Integer)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False, index=True
    )
    event_type: Mapped[str] = mapped_column(String(60), index=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[dict] = mapped_column(JSONType, default=dict)

    training_job = relationship("TrainingJob", back_populates="events")


class EvaluationResult(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_evaluation_results"
    __table_args__ = (
        UniqueConstraint(
            "checkpoint_id", "split", name="uq_training_evaluation_checkpoint_split"
        ),
    )

    training_job_id: Mapped[int] = mapped_column(
        ForeignKey("training_jobs.id"), index=True
    )
    checkpoint_id: Mapped[int] = mapped_column(
        ForeignKey("model_checkpoints.id"), index=True
    )
    split: Mapped[str] = mapped_column(String(20), index=True)
    evaluator_key: Mapped[str] = mapped_column(String(120), index=True)
    evaluator_version: Mapped[str] = mapped_column(String(40))
    metric_schema_version: Mapped[str] = mapped_column(String(40))
    prediction_schema_version: Mapped[str] = mapped_column(String(40))
    evaluation_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    controlled_cohort_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    task_contract_key: Mapped[str] = mapped_column(String(120), index=True)
    task_contract_version: Mapped[str] = mapped_column(String(40))
    target_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    preprocessing_fingerprint: Mapped[str] = mapped_column(String(64))
    decoder_fingerprint: Mapped[str] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    supports: Mapped[dict] = mapped_column(JSONType, default=dict)
    confusion_matrix: Mapped[dict] = mapped_column(JSONType, default=dict)
    confidence_intervals: Mapped[dict] = mapped_column(JSONType, default=dict)
    diagnostics: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_selection_split: Mapped[bool] = mapped_column(Boolean, default=False)

    checkpoint = relationship("ModelCheckpoint", back_populates="evaluations")


class TrainingJobArtifact(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_job_artifacts"
    __table_args__ = (
        UniqueConstraint(
            "training_job_id",
            "role",
            name="uq_training_job_artifacts_role",
        ),
        CheckConstraint(
            "artifact_id IS NOT NULL OR package_id IS NOT NULL",
            name="ck_training_job_artifacts_has_reference",
        ),
    )

    training_job_id: Mapped[int] = mapped_column(
        ForeignKey("training_jobs.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(60), index=True)
    artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("lineage_artifacts.id"), nullable=True, index=True
    )
    package_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifact_packages.id"), nullable=True, index=True
    )

    training_job = relationship("TrainingJob", back_populates="artifact_links")
    artifact = relationship("LineageArtifact")
    package = relationship("ArtifactPackage")


class ModelCheckpointPin(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "model_checkpoint_pins"
    __table_args__ = (
        UniqueConstraint(
            "checkpoint_id", "user_id", name="uq_model_checkpoint_pins_checkpoint_user"
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    checkpoint_id: Mapped[int] = mapped_column(
        ForeignKey("model_checkpoints.id"), index=True
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    reason: Mapped[str | None] = mapped_column(String(255), nullable=True)

    checkpoint = relationship("ModelCheckpoint", back_populates="pins")


class ModelReleaseAlias(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "model_release_aliases"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "task_contract_key",
            "task_contract_version",
            "target_scope_hash",
            name="uq_model_release_aliases_scope",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    task_contract_key: Mapped[str] = mapped_column(String(120), index=True)
    task_contract_version: Mapped[str] = mapped_column(String(40), default="1", index=True)
    target_scope_hash: Mapped[str] = mapped_column(String(64), index=True)
    checkpoint_id: Mapped[int] = mapped_column(
        ForeignKey("model_checkpoints.id"), index=True
    )
    changed_by_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    state: Mapped[str] = mapped_column(String(30), default="champion", index=True)

    checkpoint = relationship("ModelCheckpoint")


class ModelReleaseEvent(Base, IntPrimaryKeyMixin):
    __tablename__ = "model_release_events"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    release_alias_id: Mapped[int] = mapped_column(
        ForeignKey("model_release_aliases.id"), index=True
    )
    checkpoint_id: Mapped[int] = mapped_column(
        ForeignKey("model_checkpoints.id"), index=True
    )
    actor_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(40), index=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    details: Mapped[dict] = mapped_column(JSONType, default=dict)


def _reject_checkpoint_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(f"ModelCheckpoint {target.id} is immutable")


event.listen(ModelCheckpoint, "before_update", _reject_checkpoint_mutation)
event.listen(ModelCheckpoint, "before_delete", _reject_checkpoint_mutation)


def _reject_append_only_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(
        f"{type(target).__name__} {getattr(target, 'id', None)} is append-only"
    )


for _append_only_type in (
    TrainingMetricPoint,
    TrainingJobEvent,
    EvaluationResult,
    TrainingJobArtifact,
    ModelReleaseEvent,
):
    event.listen(_append_only_type, "before_update", _reject_append_only_mutation)
    event.listen(_append_only_type, "before_delete", _reject_append_only_mutation)
