from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    inspect,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin, utc_now
from al_medlit.core.types import JSONType
from al_medlit.corpus.segmentation import segment_sentences


class Document(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "documents"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    external_id: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)
    active_structure_version_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "document_structure_versions.id",
            name="fk_documents_active_structure_version_id",
            ondelete="SET NULL",
            use_alter=True,
        ),
        index=True,
        nullable=True,
    )

    project = relationship("Project", back_populates="documents")
    annotations = relationship(
        "Annotation",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    structure_versions = relationship(
        "DocumentStructureVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        foreign_keys="DocumentStructureVersion.document_id",
        order_by="DocumentStructureVersion.version",
    )
    active_structure_version = relationship(
        "DocumentStructureVersion",
        foreign_keys=[active_structure_version_id],
        post_update=True,
    )

    @property
    def sentences(self) -> list[list[int]]:
        """Compatibility view used by the legacy annotation canvas."""
        active = self.active_structure_version
        if active is not None:
            return [
                [sentence.start_offset, sentence.end_offset]
                for sentence in active.sentences
            ]
        return [list(span) for span in segment_sentences(self.text or "")]


class DocumentStructureVersion(Base, IntPrimaryKeyMixin):
    __tablename__ = "document_structure_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "version",
            name="uq_document_structure_versions_document_version",
        ),
        CheckConstraint("version >= 1", name="ck_document_structure_versions_version"),
        CheckConstraint("text_length >= 0", name="ck_document_structure_versions_text_length"),
    )

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer)
    segmenter_name: Mapped[str] = mapped_column(String(80))
    segmenter_version: Mapped[str] = mapped_column(String(40))
    source_hash: Mapped[str] = mapped_column(String(64), index=True)
    text_length: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40), default="ready", index=True)
    source_metadata: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    document = relationship(
        "Document",
        back_populates="structure_versions",
        foreign_keys=[document_id],
    )
    sections = relationship(
        "DocumentSection",
        back_populates="structure_version",
        cascade="all, delete-orphan",
        order_by="DocumentSection.ordinal",
    )
    paragraphs = relationship(
        "DocumentParagraph",
        order_by="DocumentParagraph.ordinal",
        viewonly=True,
    )
    sentences = relationship(
        "DocumentSentence",
        order_by="DocumentSentence.ordinal",
        viewonly=True,
    )


class DocumentSection(Base, IntPrimaryKeyMixin):
    __tablename__ = "document_sections"
    __table_args__ = (
        UniqueConstraint(
            "structure_version_id",
            "ordinal",
            name="uq_document_sections_structure_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ck_document_sections_ordinal"),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_sections_offsets",
        ),
    )

    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id", ondelete="CASCADE"),
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    path: Mapped[list] = mapped_column(JSONType, default=list)
    kind: Mapped[str] = mapped_column(String(40), default="unknown")
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    locator: Mapped[dict] = mapped_column(JSONType, default=dict)

    structure_version = relationship("DocumentStructureVersion", back_populates="sections")
    paragraphs = relationship(
        "DocumentParagraph",
        back_populates="section",
        cascade="all, delete-orphan",
        order_by="DocumentParagraph.section_ordinal",
    )
    sentences = relationship(
        "DocumentSentence",
        order_by="DocumentSentence.ordinal",
        viewonly=True,
    )


class DocumentParagraph(Base, IntPrimaryKeyMixin):
    __tablename__ = "document_paragraphs"
    __table_args__ = (
        UniqueConstraint(
            "structure_version_id",
            "ordinal",
            name="uq_document_paragraphs_structure_ordinal",
        ),
        UniqueConstraint(
            "section_id",
            "section_ordinal",
            name="uq_document_paragraphs_section_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ck_document_paragraphs_ordinal"),
        CheckConstraint(
            "section_ordinal >= 0",
            name="ck_document_paragraphs_section_ordinal",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset >= start_offset",
            name="ck_document_paragraphs_offsets",
        ),
    )

    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id", ondelete="CASCADE"),
        index=True,
    )
    section_id: Mapped[int] = mapped_column(
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    section_ordinal: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    locator: Mapped[dict] = mapped_column(JSONType, default=dict)

    structure_version = relationship("DocumentStructureVersion")
    section = relationship("DocumentSection", back_populates="paragraphs")
    sentences = relationship(
        "DocumentSentence",
        back_populates="paragraph",
        cascade="all, delete-orphan",
        order_by="DocumentSentence.paragraph_ordinal",
    )


class DocumentSentence(Base, IntPrimaryKeyMixin):
    __tablename__ = "document_sentences"
    __table_args__ = (
        UniqueConstraint(
            "structure_version_id",
            "ordinal",
            name="uq_document_sentences_structure_ordinal",
        ),
        UniqueConstraint(
            "paragraph_id",
            "paragraph_ordinal",
            name="uq_document_sentences_paragraph_ordinal",
        ),
        CheckConstraint("ordinal >= 0", name="ck_document_sentences_ordinal"),
        CheckConstraint(
            "paragraph_ordinal >= 0",
            name="ck_document_sentences_paragraph_ordinal",
        ),
        CheckConstraint(
            "start_offset >= 0 AND end_offset > start_offset",
            name="ck_document_sentences_offsets",
        ),
    )

    structure_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_structure_versions.id", ondelete="CASCADE"),
        index=True,
    )
    section_id: Mapped[int] = mapped_column(
        ForeignKey("document_sections.id", ondelete="CASCADE"),
        index=True,
    )
    paragraph_id: Mapped[int] = mapped_column(
        ForeignKey("document_paragraphs.id", ondelete="CASCADE"),
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    paragraph_ordinal: Mapped[int] = mapped_column(Integer)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    text_hash: Mapped[str] = mapped_column(String(64))

    structure_version = relationship("DocumentStructureVersion")
    section = relationship("DocumentSection")
    paragraph = relationship("DocumentParagraph", back_populates="sentences")


class ImmutableDocumentStructureError(ValueError):
    """Raised when persisted structure data is changed in place."""


def _reject_persisted_structure_update(mapper, _connection, target) -> None:
    state = inspect(target)
    changed_columns = [
        attribute.key
        for attribute in mapper.column_attrs
        if state.attrs[attribute.key].history.has_changes()
    ]
    if changed_columns:
        fields = ", ".join(changed_columns)
        raise ImmutableDocumentStructureError(
            f"{type(target).__name__} is immutable; changed fields: {fields}"
        )


for _immutable_model in (
    DocumentStructureVersion,
    DocumentSection,
    DocumentParagraph,
    DocumentSentence,
):
    event.listen(_immutable_model, "before_update", _reject_persisted_structure_update)
