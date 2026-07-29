from datetime import datetime

from sqlalchemy import (
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
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType
from al_medlit.lineage.models import ImmutableRecordError


class ModelPrediction(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Stores predictions from BERT/SLM/LLM/dictionary systems.

    These predictions are compared against human/gold annotations to create
    disagreement records and error patterns.
    """

    __tablename__ = "model_predictions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)

    model_name: Mapped[str] = mapped_column(String(255), index=True)
    model_version: Mapped[str | None] = mapped_column(String(255), nullable=True)

    task_type: Mapped[str] = mapped_column(String(80), index=True)
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    evidence: Mapped[dict] = mapped_column(JSONType, default=dict)
    raw_output: Mapped[dict] = mapped_column(JSONType, default=dict)


class InferenceRun(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "inference_runs"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "idempotency_key",
            name="uq_inference_runs_project_idempotency",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    corpus_snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("corpus_snapshots.id"),
        index=True,
    )
    checkpoint_id: Mapped[int] = mapped_column(ForeignKey("model_checkpoints.id"), index=True)
    compute_profile_id: Mapped[int] = mapped_column(
        ForeignKey("compute_profiles.id"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    target_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    window_config: Mapped[dict] = mapped_column(JSONType, default=dict)
    decoder_config: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="queued", index=True)
    idempotency_key: Mapped[str] = mapped_column(String(120))
    external_job_id: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    diagnostics_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        nullable=True,
        index=True,
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)

    corpus_snapshot = relationship("CorpusSnapshot")
    checkpoint = relationship("ModelCheckpoint")
    compute_profile = relationship("ComputeProfile")
    diagnostics_artifact = relationship("LineageArtifact")
    windows = relationship(
        "InferenceWindow",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by=lambda: (
            InferenceWindow.document_id,
            InferenceWindow.start_sentence_ordinal,
        ),
    )
    candidates = relationship(
        "EvidenceCandidatePrediction",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class InferenceWindow(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "inference_windows"
    __table_args__ = (
        UniqueConstraint("run_id", "stable_key", name="uq_inference_windows_run_key"),
    )

    run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        index=True,
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"),
        index=True,
    )
    stable_key: Mapped[str] = mapped_column(String(80))
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    sentence_contribution_counts: Mapped[dict] = mapped_column(JSONType, default=dict)
    diagnostics_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        nullable=True,
        index=True,
    )

    run = relationship("InferenceRun", back_populates="windows")


class EvidenceCandidatePrediction(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "evidence_candidate_predictions"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "document_id",
            "target_version_id",
            "start_sentence_ordinal",
            "end_sentence_ordinal",
            name="uq_evidence_candidate_run_scope_boundary",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("inference_runs.id"), index=True)
    checkpoint_id: Mapped[int] = mapped_column(ForeignKey("model_checkpoints.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        index=True,
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"),
        index=True,
    )
    start_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    end_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    block_confidence: Mapped[float] = mapped_column(Float)
    boundary_confidence: Mapped[dict] = mapped_column(JSONType, default=dict)
    uncertainty: Mapped[float] = mapped_column(Float)
    decoder_version: Mapped[str] = mapped_column(String(80))
    source_window_ids: Mapped[list] = mapped_column(JSONType, default=list)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    diagnostics_artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        nullable=True,
        index=True,
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    run = relationship("InferenceRun", back_populates="candidates")
    checkpoint = relationship("ModelCheckpoint")
    reviews = relationship(
        "EvidencePredictionReview",
        back_populates="prediction",
        cascade="all, delete-orphan",
        order_by="EvidencePredictionReview.revision",
    )


class EvidencePredictionReview(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "evidence_prediction_reviews"
    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            "revision",
            name="uq_evidence_prediction_review_revision",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_candidate_predictions.id"),
        index=True,
    )
    assignment_id: Mapped[int | None] = mapped_column(
        ForeignKey("task_assignments.id"),
        nullable=True,
        index=True,
    )
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"),
        nullable=True,
        index=True,
    )
    reviewer_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(30), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    resulting_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
        index=True,
    )
    selected_boundaries: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    prediction = relationship("EvidenceCandidatePrediction", back_populates="reviews")
    assignment = relationship("TaskAssignment")
    guideline_version = relationship("GuidelineVersion")
    resulting_annotation = relationship("Annotation")


def _reject_review_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(f"EvidencePredictionReview {target.id} is append-only")


event.listen(EvidencePredictionReview, "before_update", _reject_review_mutation)
event.listen(EvidencePredictionReview, "before_delete", _reject_review_mutation)
