from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class AnnotationSubmission(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A saved snapshot file of a document's annotations at submission time."""

    __tablename__ = "annotation_submissions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    assignment_id: Mapped[int | None] = mapped_column(
        ForeignKey("task_assignments.id"), nullable=True, index=True
    )
    annotator_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    annotator_id: Mapped[str | None] = mapped_column(
        String(120),
        nullable=True,
        index=True,
    )

    # "submission" finalizes assignments; "re_export" leaves assignments alone.
    kind: Mapped[str] = mapped_column(String(20), default="submission", index=True)

    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    file_name: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100), default="application/json")
    size_bytes: Mapped[int] = mapped_column(Integer)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    annotation_count: Mapped[int] = mapped_column(Integer, default=0)

    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    document = relationship("Document")
    annotator_user = relationship("User")
    assignment = relationship("TaskAssignment")
