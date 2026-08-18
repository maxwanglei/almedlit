from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin
from al_medlit.core.types import JSONType


class Workspace(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "workspaces"
    __table_args__ = (
        UniqueConstraint(
            "created_by",
            "name",
            "kind",
            name="uq_workspaces_creator_name_kind",
        ),
        Index(
            "uq_workspaces_system_name_kind",
            "name",
            "kind",
            unique=True,
            postgresql_where=text("created_by IS NULL"),
            sqlite_where=text("created_by IS NULL"),
        ),
    )

    name: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(40), default="individual", index=True)
    join_code: Mapped[str | None] = mapped_column(
        String(40),
        unique=True,
        index=True,
        nullable=True,
    )
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    capability_preset: Mapped[str] = mapped_column(String(40), default="annotate")
    capability_overrides: Mapped[list] = mapped_column(JSONType, default=list)

    members = relationship(
        "WorkspaceMember",
        back_populates="workspace",
        cascade="all, delete-orphan",
    )
    projects = relationship("Project", back_populates="workspace")


class WorkspaceMember(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint(
            "workspace_id",
            "user_id",
            name="uq_workspace_members_workspace_user",
        ),
    )

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(40), default="annotator", index=True)

    workspace = relationship("Workspace", back_populates="members")
    user = relationship("User", back_populates="memberships")

    @property
    def username(self) -> str:
        return self.user.username if self.user is not None else ""

    @property
    def display_name(self) -> str:
        return self.user.display_name if self.user is not None else ""

    @property
    def email(self) -> str | None:
        return self.user.email if self.user is not None else None

    @property
    def is_active(self) -> bool:
        return self.user.is_active if self.user is not None else False


class WorkspaceInvite(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "workspace_invites"

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role: Mapped[str] = mapped_column(String(40), default="annotator", index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    accepted_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)

    creator = relationship("User", foreign_keys=[created_by])

    @property
    def created_by_username(self) -> str:
        return self.creator.username if self.creator is not None else ""


class WorkspaceJoinRequest(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "workspace_join_requests"
    __table_args__ = (
        Index(
            "uq_workspace_join_requests_pending_user",
            "workspace_id",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
            sqlite_where=text("status = 'pending'"),
        ),
    )

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    decided_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    user = relationship("User", foreign_keys=[user_id])

    @property
    def username(self) -> str:
        return self.user.username if self.user is not None else ""

    @property
    def display_name(self) -> str:
        return self.user.display_name if self.user is not None else ""

    @property
    def email(self) -> str | None:
        return self.user.email if self.user is not None else None
