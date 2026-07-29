from sqlalchemy import (
    BigInteger,
    CheckConstraint,
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


class ImmutableRecordError(RuntimeError):
    """Raised when code attempts to rewrite frozen lineage state."""


class LineageArtifact(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "lineage_artifacts"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(80), index=True)
    schema_version: Mapped[str] = mapped_column(String(40), default="1.0")
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    content_type: Mapped[str] = mapped_column(String(120))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    manifest: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )


class LineageEdge(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "lineage_edges"
    __table_args__ = (
        UniqueConstraint(
            "upstream_artifact_id",
            "downstream_artifact_id",
            "relationship_type",
            name="uq_lineage_edge_artifacts_relationship",
        ),
        CheckConstraint(
            "upstream_artifact_id <> downstream_artifact_id",
            name="ck_lineage_edge_not_self_referential",
        ),
    )

    upstream_artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        index=True,
    )
    downstream_artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        index=True,
    )
    relationship_type: Mapped[str] = mapped_column(String(80), default="derived_from")
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    upstream = relationship("LineageArtifact", foreign_keys=[upstream_artifact_id])
    downstream = relationship("LineageArtifact", foreign_keys=[downstream_artifact_id])


class CorpusSnapshot(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "corpus_snapshots"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        unique=True,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    split_strategy: Mapped[str] = mapped_column(String(80), default="document")
    split_seed: Mapped[int] = mapped_column(Integer, default=42)
    document_count: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    artifact = relationship("LineageArtifact")
    documents = relationship(
        "CorpusSnapshotDocument",
        back_populates="snapshot",
        cascade="all, delete-orphan",
        order_by="CorpusSnapshotDocument.document_id",
    )


class CorpusSnapshotDocument(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "corpus_snapshot_documents"
    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "document_id",
            name="uq_corpus_snapshot_document",
        ),
    )

    snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("corpus_snapshots.id"),
        index=True,
    )
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        index=True,
    )
    split: Mapped[str] = mapped_column(String(20), index=True)
    group_key: Mapped[str] = mapped_column(String(255), index=True)
    source_hash: Mapped[str] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    snapshot = relationship("CorpusSnapshot", back_populates="documents")


class AnnotationSet(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "annotation_sets"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        unique=True,
        index=True,
    )
    corpus_snapshot_id: Mapped[int] = mapped_column(
        ForeignKey("corpus_snapshots.id"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    target_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    guideline_version_ids: Mapped[list] = mapped_column(JSONType, default=list)
    block_count: Mapped[int] = mapped_column(Integer)
    reviewed_region_count: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    artifact = relationship("LineageArtifact")
    corpus_snapshot = relationship("CorpusSnapshot")
    items = relationship(
        "AnnotationSetItem",
        back_populates="annotation_set",
        cascade="all, delete-orphan",
        order_by=lambda: (
            AnnotationSetItem.document_id,
            AnnotationSetItem.start_sentence_ordinal,
        ),
    )
    reviewed_regions = relationship(
        "AnnotationSetReviewRegion",
        back_populates="annotation_set",
        cascade="all, delete-orphan",
    )


class AnnotationSetItem(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "annotation_set_items"

    annotation_set_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_sets.id"),
        index=True,
    )
    source_annotation_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotations.id"),
        nullable=True,
        index=True,
    )
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        index=True,
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"),
        index=True,
    )
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"),
        nullable=True,
        index=True,
    )
    start_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    end_sentence_id: Mapped[int] = mapped_column(ForeignKey("document_sentences.id"))
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    block_text: Mapped[str] = mapped_column(Text)
    section_paths: Mapped[list] = mapped_column(JSONType, default=list)
    labels: Mapped[list] = mapped_column(JSONType, default=list)
    source: Mapped[str] = mapped_column(String(30), default="adjudicated")
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    annotation_set = relationship("AnnotationSet", back_populates="items")


class AnnotationSetReviewRegion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "annotation_set_review_regions"

    annotation_set_id: Mapped[int] = mapped_column(
        ForeignKey("annotation_sets.id"),
        index=True,
    )
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id"),
        index=True,
    )
    target_version_id: Mapped[int] = mapped_column(
        ForeignKey("evidence_target_versions.id"),
        index=True,
    )
    start_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    end_sentence_ordinal: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    annotation_set = relationship("AnnotationSet", back_populates="reviewed_regions")


class ExportArtifact(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "export_artifacts"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    artifact_id: Mapped[int] = mapped_column(
        ForeignKey("lineage_artifacts.id"),
        unique=True,
        index=True,
    )
    corpus_snapshot_id: Mapped[int | None] = mapped_column(
        ForeignKey("corpus_snapshots.id"),
        nullable=True,
        index=True,
    )
    annotation_set_id: Mapped[int | None] = mapped_column(
        ForeignKey("annotation_sets.id"),
        nullable=True,
        index=True,
    )
    format_key: Mapped[str] = mapped_column(String(100), index=True)
    file_name: Mapped[str] = mapped_column(String(255))
    row_count: Mapped[int] = mapped_column(Integer)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    artifact = relationship("LineageArtifact")


_IMMUTABLE_TYPES = (
    LineageArtifact,
    LineageEdge,
    CorpusSnapshot,
    CorpusSnapshotDocument,
    AnnotationSet,
    AnnotationSetItem,
    AnnotationSetReviewRegion,
    ExportArtifact,
)


def _reject_frozen_mutation(_mapper, _connection, target) -> None:
    raise ImmutableRecordError(
        f"{type(target).__name__} {getattr(target, 'id', None)} is immutable"
    )


for _immutable_type in _IMMUTABLE_TYPES:
    event.listen(_immutable_type, "before_update", _reject_frozen_mutation)
    event.listen(_immutable_type, "before_delete", _reject_frozen_mutation)
