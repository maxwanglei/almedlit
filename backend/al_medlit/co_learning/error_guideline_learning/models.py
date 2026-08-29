from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class ErrorPattern(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "error_patterns"
    __table_args__ = (
        Index(
            "uq_error_patterns_active_unlabeled",
            "project_id",
            "task_type",
            "error_type",
            unique=True,
            postgresql_where=text("status = 'active' AND label_type IS NULL"),
            sqlite_where=text("status = 'active' AND label_type IS NULL"),
        ),
        Index(
            "uq_error_patterns_active_labeled",
            "project_id",
            "task_type",
            "error_type",
            "label_type",
            unique=True,
            postgresql_where=text("status = 'active' AND label_type IS NOT NULL"),
            sqlite_where=text("status = 'active' AND label_type IS NOT NULL"),
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)

    # entity, relation, doc_label
    task_type: Mapped[str] = mapped_column(String(80), index=True)
    error_type: Mapped[str] = mapped_column(String(120), index=True)
    label_type: Mapped[str | None] = mapped_column(String(120), nullable=True)

    description: Mapped[str] = mapped_column(Text)
    example_count: Mapped[int] = mapped_column(Integer, default=1)
    severity: Mapped[str] = mapped_column(String(50), default="medium")
    detected_from: Mapped[str] = mapped_column(String(120), default="manual")
    status: Mapped[str] = mapped_column(String(50), default="active")

    example_ids: Mapped[list] = mapped_column(JSONType, default=list)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)


class GuidelineAtom(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guideline_atoms"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"),
        nullable=True,
        index=True,
    )
    error_pattern_id: Mapped[int | None] = mapped_column(
        ForeignKey("error_patterns.id"),
        nullable=True,
        index=True,
    )

    task_type: Mapped[str] = mapped_column(String(80), index=True)
    error_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    rule_text: Mapped[str] = mapped_column(Text)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    positive_examples: Mapped[list] = mapped_column(JSONType, default=list)
    negative_examples: Mapped[list] = mapped_column(JSONType, default=list)
    applies_to: Mapped[list] = mapped_column(JSONType, default=list)

    # pending, accepted, rejected
    status: Mapped[str] = mapped_column(String(50), default="pending")
    approved_by: Mapped[str | None] = mapped_column(String(120), nullable=True)


class TrainingAction(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "training_actions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    error_pattern_id: Mapped[int | None] = mapped_column(
        ForeignKey("error_patterns.id"),
        nullable=True,
        index=True,
    )
    guideline_atom_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_atoms.id"),
        nullable=True,
        index=True,
    )

    action_type: Mapped[str] = mapped_column(String(120), index=True)
    target_model: Mapped[str] = mapped_column(String(255))
    example_ids: Mapped[list] = mapped_column(JSONType, default=list)

    priority: Mapped[str] = mapped_column(String(50), default="medium")
    status: Mapped[str] = mapped_column(String(50), default="planned")
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class MicroQuestionTemplate(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "micro_question_templates"

    project_id: Mapped[int | None] = mapped_column(
        ForeignKey("projects.id"),
        nullable=True,
        index=True,
    )
    guideline_atom_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_atoms.id"),
        nullable=True,
        index=True,
    )

    error_type: Mapped[str] = mapped_column(String(120), index=True)
    target_annotation_type: Mapped[str] = mapped_column(String(80), default="entity")
    question_text: Mapped[str] = mapped_column(Text)
    answer_options: Mapped[list] = mapped_column(JSONType, default=list)
    status: Mapped[str] = mapped_column(String(50), default="active")
