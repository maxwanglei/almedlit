from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, event
from sqlalchemy.orm import Mapped, mapped_column

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin, utc_now
from al_medlit.core.types import JSONType


class InstancePolicy(Base, TimestampMixin):
    """The singleton set of deployment policy overrides.

    A nullable self-registration value preserves the environment setting on
    upgrade. Once an administrator changes it, the database value becomes the
    live override.
    """

    __tablename__ = "instance_policies"
    __table_args__ = (
        CheckConstraint("id = 1", name="ck_instance_policies_singleton"),
        CheckConstraint(
            "default_invite_expiry_minutes BETWEEN 60 AND 43200",
            name="ck_instance_policies_invite_expiry_range",
        ),
        CheckConstraint(
            "account_action_expiry_minutes BETWEEN 15 AND 1440",
            name="ck_instance_policies_account_action_expiry_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    allow_self_registration: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    default_invite_expiry_minutes: Mapped[int] = mapped_column(
        Integer,
        default=10_080,
        nullable=False,
    )
    account_action_expiry_minutes: Mapped[int] = mapped_column(
        Integer,
        default=60,
        nullable=False,
    )
    updated_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
    )


class AccountActionToken(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "account_action_tokens"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    purpose: Mapped[str] = mapped_column(String(40), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    consumed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )


class AdminAuditEvent(Base, IntPrimaryKeyMixin):
    __tablename__ = "admin_audit_events"

    event_type: Mapped[str] = mapped_column(String(120), index=True)
    # Deliberately retain identifiers without foreign keys. Audit history must
    # survive any future out-of-band account/workspace removal workflow.
    actor_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    target_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    workspace_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    details: Mapped[dict] = mapped_column(JSONType, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )


def _reject_audit_event_mutation(_mapper, _connection, _target) -> None:
    raise ValueError("Administrative audit events are append-only")


event.listen(AdminAuditEvent, "before_update", _reject_audit_event_mutation)
event.listen(AdminAuditEvent, "before_delete", _reject_audit_event_mutation)
