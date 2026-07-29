from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint, event
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class EvidenceTarget(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A stable logical question/criterion used to condition evidence annotation."""

    __tablename__ = "evidence_targets"
    __table_args__ = (
        UniqueConstraint("project_id", "key", name="uq_evidence_targets_project_key"),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("project_tasks.id"), index=True)
    key: Mapped[str] = mapped_column(String(120))
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_version_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "evidence_target_versions.id",
            name="fk_evidence_targets_active_version_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        nullable=True,
        index=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )

    project = relationship("Project", back_populates="evidence_targets")
    task = relationship("ProjectTask", back_populates="evidence_targets")
    versions = relationship(
        "EvidenceTargetVersion",
        back_populates="target",
        cascade="all, delete-orphan",
        foreign_keys="EvidenceTargetVersion.target_id",
        order_by="EvidenceTargetVersion.version_number",
    )
    active_version = relationship(
        "EvidenceTargetVersion",
        foreign_keys=[active_version_id],
        post_update=True,
    )
    created_by_user = relationship("User", foreign_keys=[created_by_user_id])


class EvidenceTargetVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Immutable target wording and annotation guidance."""

    __tablename__ = "evidence_target_versions"
    __table_args__ = (
        UniqueConstraint(
            "target_id",
            "version_number",
            name="uq_evidence_target_versions_target_number",
        ),
    )

    target_id: Mapped[int] = mapped_column(ForeignKey("evidence_targets.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    inclusion_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    exclusion_guidance: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )

    target = relationship(
        "EvidenceTarget",
        back_populates="versions",
        foreign_keys=[target_id],
    )
    created_by_user = relationship("User", foreign_keys=[created_by_user_id])


class EvidenceBlockAnnotation(Base, TimestampMixin):
    """Sentence-aligned, target-scoped payload for an Annotation parent row."""

    __tablename__ = "evidence_block_annotations"

    annotation_id: Mapped[int] = mapped_column(
        ForeignKey("annotations.id", ondelete="CASCADE"), primary_key=True
    )
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"), index=True
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"), index=True
    )
    start_sentence_id: Mapped[int] = mapped_column(
        ForeignKey("document_sentences.id"), index=True
    )
    end_sentence_id: Mapped[int] = mapped_column(
        ForeignKey("document_sentences.id"), index=True
    )
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    labels: Mapped[list] = mapped_column(JSONType, default=list)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary_policy: Mapped[str] = mapped_column(String(50), default="sentence")
    revision: Mapped[int] = mapped_column(Integer, default=1)
    locked: Mapped[bool] = mapped_column(Boolean, default=False)
    last_command_group_key: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )

    annotation = relationship("Annotation", back_populates="evidence_block")
    structure_version = relationship("DocumentStructureVersion")
    target_version = relationship("EvidenceTargetVersion")
    start_sentence = relationship("DocumentSentence", foreign_keys=[start_sentence_id])
    end_sentence = relationship("DocumentSentence", foreign_keys=[end_sentence_id])

    @property
    def start_offset(self) -> int:
        return self.start_sentence.start_offset

    @property
    def end_offset(self) -> int:
        return self.end_sentence.end_offset


class EvidenceBlockRevision(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Append-only audit record for evidence-block boundary commands."""

    __tablename__ = "evidence_block_revisions"

    annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id", ondelete="SET NULL"), nullable=True, index=True
    )
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"), index=True
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"), index=True
    )
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    operation: Mapped[str] = mapped_column(String(50), index=True)
    before: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    after: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    command_group_key: Mapped[str] = mapped_column(String(64), index=True)
    is_undone: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class EvidenceReviewCoverage(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A normalized, effective reviewed interval for one user and target."""

    __tablename__ = "evidence_review_coverage"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"), index=True
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"), index=True
    )
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"), nullable=True, index=True
    )
    reviewer_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    start_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    end_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)


class EvidenceReviewEvent(Base, IntPrimaryKeyMixin, TimestampMixin):
    """Append-only mark/reopen history; effective coverage lives in coverage rows."""

    __tablename__ = "evidence_review_events"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"), index=True
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"), index=True
    )
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"), nullable=True, index=True
    )
    actor_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(50), index=True)
    start_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    end_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)


class ImmutableEvidenceTargetVersionError(ValueError):
    pass


class ImmutableEvidenceReviewEventError(ValueError):
    pass


def _reject_target_version_mutation(_mapper, _connection, target) -> None:
    raise ImmutableEvidenceTargetVersionError(
        f"EvidenceTargetVersion {target.id} is immutable"
    )


def _reject_review_event_mutation(_mapper, _connection, target) -> None:
    raise ImmutableEvidenceReviewEventError(
        f"EvidenceReviewEvent {target.id} is append-only"
    )


event.listen(EvidenceTargetVersion, "before_update", _reject_target_version_mutation)
event.listen(EvidenceTargetVersion, "before_delete", _reject_target_version_mutation)
event.listen(EvidenceReviewEvent, "before_update", _reject_review_event_mutation)
event.listen(EvidenceReviewEvent, "before_delete", _reject_review_event_mutation)
