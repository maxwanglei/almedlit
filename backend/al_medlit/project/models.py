from sqlalchemy import Boolean, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class Project(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "projects"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "name",
            name="uq_projects_workspace_name",
        ),
    )

    name: Mapped[str] = mapped_column(String(255), index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    annotation_schema: Mapped[dict] = mapped_column(JSONType, default=dict)
    annotation_validation_mode: Mapped[str] = mapped_column(String(20), default="relaxed")
    settings: Mapped[dict] = mapped_column(JSONType, default=dict)
    workspace_id: Mapped[int] = mapped_column(
        ForeignKey("workspaces.id"),
        index=True,
        nullable=False,
    )

    workspace = relationship("Workspace", back_populates="projects")
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    tasks = relationship(
        "ProjectTask",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="ProjectTask.sort_order",
    )
    task_assignments = relationship(
        "TaskAssignment",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    guideline_versions = relationship(
        "GuidelineVersion",
        back_populates="project",
        cascade="all, delete-orphan",
    )
    evidence_targets = relationship(
        "EvidenceTarget",
        back_populates="project",
        cascade="all, delete-orphan",
        foreign_keys="EvidenceTarget.project_id",
    )


class ProjectTask(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "project_tasks"
    __table_args__ = (
        UniqueConstraint(
            "project_id",
            "annotation_type",
            name="uq_project_tasks_project_annotation_type",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    annotation_type: Mapped[str] = mapped_column(String(80), index=True)
    display_name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    labels: Mapped[list] = mapped_column(JSONType, default=list)
    settings: Mapped[dict] = mapped_column(JSONType, default=dict)

    project = relationship("Project", back_populates="tasks")
    assignments = relationship(
        "TaskAssignment",
        back_populates="task",
        cascade="all, delete-orphan",
    )
    evidence_targets = relationship(
        "EvidenceTarget",
        back_populates="task",
        cascade="all, delete-orphan",
        foreign_keys="EvidenceTarget.task_id",
    )


class TaskAssignment(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "task_assignments"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "document_id",
            "assignee_user_id",
            "assignment_scope_key",
            name="uq_task_assignments_task_document_assignee_scope",
        ),
    )

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("project_tasks.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    assignee_user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    target_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("evidence_target_versions.id"), nullable=True, index=True
    )
    structure_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("document_structure_versions.id"), nullable=True, index=True
    )
    guideline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("guideline_versions.id"), nullable=True, index=True
    )
    assignment_scope_key: Mapped[str] = mapped_column(
        String(160), default="document", index=True
    )
    annotator_id: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(50), default="assigned", index=True)
    assigned_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    assigned_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType, default=dict)

    project = relationship("Project", back_populates="task_assignments")
    task = relationship("ProjectTask", back_populates="assignments")
    document = relationship("Document")
    assignee = relationship("User", foreign_keys=[assignee_user_id])
    assigned_by_user = relationship("User", foreign_keys=[assigned_by_user_id])
    target_version = relationship("EvidenceTargetVersion")
    structure_version = relationship("DocumentStructureVersion")
    guideline_version = relationship("GuidelineVersion")
