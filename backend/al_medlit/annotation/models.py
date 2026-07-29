from sqlalchemy import Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class Annotation(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "annotations"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)

    # entity, relation, doc_label
    annotation_type: Mapped[str] = mapped_column(String(80), index=True)
    label: Mapped[str] = mapped_column(String(120), index=True)

    start_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_offset: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_span: Mapped[str | None] = mapped_column(Text, nullable=True)

    # human, model, llm
    source: Mapped[str] = mapped_column(String(50), default="human")
    # draft, accepted, rejected, gold
    status: Mapped[str] = mapped_column(String(50), default="draft")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    annotator_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    annotator_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model_checkpoint_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"),
        nullable=True,
        index=True,
    )
    structure_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        nullable=True,
        index=True,
    )
    head_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
        index=True,
    )
    tail_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
        index=True,
    )

    evidence: Mapped[dict] = mapped_column(JSONType, default=dict)
    attributes: Mapped[dict] = mapped_column(JSONType, default=dict)

    document = relationship("Document", back_populates="annotations")
    annotator_user = relationship("User")
    head = relationship(
        "Annotation",
        foreign_keys=[head_annotation_id],
        remote_side="Annotation.id",
        uselist=False,
    )
    tail = relationship(
        "Annotation",
        foreign_keys=[tail_annotation_id],
        remote_side="Annotation.id",
        uselist=False,
    )
    evidence_block = relationship(
        "EvidenceBlockAnnotation",
        back_populates="annotation",
        uselist=False,
        cascade="all, delete-orphan",
    )


class AnnotationCorrection(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A human/model correction event that can feed error-guideline learning."""

    __tablename__ = "annotation_corrections"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )

    original_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
    )
    corrected_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
    )

    # human, adjudication, model
    correction_source: Mapped[str] = mapped_column(String(50), default="human")
    correction_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Optional direct error phenotype supplied by adjudicator/UI.
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    severity: Mapped[str] = mapped_column(String(50), default="medium")

    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    created_by_user = relationship("User")
