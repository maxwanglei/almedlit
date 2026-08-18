"""add system administration

Revision ID: 71b9d2e4f6a8
Revises: 6f2a9d4c8b31
Create Date: 2026-07-31 00:00:00.000000
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

batch_alter_table: Any = op.batch_alter_table

revision: str = "71b9d2e4f6a8"
down_revision: str | None = "6f2a9d4c8b31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "session_version",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.add_column(
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True)
        )

    with batch_alter_table("workspace_invites", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(sa.Column("revoked_by", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_workspace_invites_revoked_by_users",
            "users",
            ["revoked_by"],
            ["id"],
        )

    op.create_table(
        "instance_policies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("allow_self_registration", sa.Boolean(), nullable=True),
        sa.Column(
            "default_invite_expiry_minutes",
            sa.Integer(),
            nullable=False,
            server_default="10080",
        ),
        sa.Column(
            "account_action_expiry_minutes",
            sa.Integer(),
            nullable=False,
            server_default="60",
        ),
        sa.Column("updated_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("id = 1", name="ck_instance_policies_singleton"),
        sa.CheckConstraint(
            "default_invite_expiry_minutes BETWEEN 60 AND 43200",
            name="ck_instance_policies_invite_expiry_range",
        ),
        sa.CheckConstraint(
            "account_action_expiry_minutes BETWEEN 15 AND 1440",
            name="ck_instance_policies_account_action_expiry_range",
        ),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    instance_policies = sa.table(
        "instance_policies",
        sa.column("id", sa.Integer()),
        sa.column("allow_self_registration", sa.Boolean()),
        sa.column("default_invite_expiry_minutes", sa.Integer()),
        sa.column("account_action_expiry_minutes", sa.Integer()),
        sa.column("updated_by", sa.Integer()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    op.execute(
        instance_policies.insert().values(
            id=1,
            allow_self_registration=None,
            default_invite_expiry_minutes=10_080,
            account_action_expiry_minutes=60,
            updated_by=None,
            created_at=sa.func.now(),
            updated_at=sa.func.now(),
        )
    )

    op.create_table(
        "account_action_tokens",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(length=40), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("account_action_tokens", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_account_action_tokens_id"), ["id"])
        batch_op.create_index(
            batch_op.f("ix_account_action_tokens_user_id"),
            ["user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_account_action_tokens_purpose"),
            ["purpose"],
        )
        batch_op.create_index(
            batch_op.f("ix_account_action_tokens_token_hash"),
            ["token_hash"],
            unique=True,
        )
        batch_op.create_index(
            batch_op.f("ix_account_action_tokens_created_by"),
            ["created_by"],
        )
        batch_op.create_index(
            batch_op.f("ix_account_action_tokens_expires_at"),
            ["expires_at"],
        )

    op.create_table(
        "admin_audit_events",
        sa.Column("event_type", sa.String(length=120), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("target_user_id", sa.Integer(), nullable=True),
        sa.Column("workspace_id", sa.Integer(), nullable=True),
        sa.Column("details", JSONType(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("admin_audit_events", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_admin_audit_events_id"), ["id"])
        batch_op.create_index(
            batch_op.f("ix_admin_audit_events_event_type"),
            ["event_type"],
        )
        batch_op.create_index(
            batch_op.f("ix_admin_audit_events_actor_user_id"),
            ["actor_user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_admin_audit_events_target_user_id"),
            ["target_user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_admin_audit_events_workspace_id"),
            ["workspace_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_admin_audit_events_created_at"),
            ["created_at"],
        )
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            """
            CREATE FUNCTION reject_admin_audit_event_mutation()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'Administrative audit events are append-only';
                RETURN NULL;
            END;
            $$
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_admin_audit_events_append_only
            BEFORE UPDATE OR DELETE ON admin_audit_events
            FOR EACH ROW EXECUTE FUNCTION reject_admin_audit_event_mutation()
            """
        )
    elif bind.dialect.name == "sqlite":
        op.execute(
            """
            CREATE TRIGGER trg_admin_audit_events_no_update
            BEFORE UPDATE ON admin_audit_events
            BEGIN
                SELECT RAISE(ABORT, 'Administrative audit events are append-only');
            END
            """
        )
        op.execute(
            """
            CREATE TRIGGER trg_admin_audit_events_no_delete
            BEFORE DELETE ON admin_audit_events
            BEGIN
                SELECT RAISE(ABORT, 'Administrative audit events are append-only');
            END
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "DROP TRIGGER trg_admin_audit_events_append_only ON admin_audit_events"
        )
        op.execute("DROP FUNCTION reject_admin_audit_event_mutation()")
    elif bind.dialect.name == "sqlite":
        op.execute("DROP TRIGGER trg_admin_audit_events_no_update")
        op.execute("DROP TRIGGER trg_admin_audit_events_no_delete")

    with batch_alter_table("admin_audit_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_created_at"))
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_target_user_id"))
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_actor_user_id"))
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_event_type"))
        batch_op.drop_index(batch_op.f("ix_admin_audit_events_id"))
    op.drop_table("admin_audit_events")

    with batch_alter_table("account_action_tokens", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_expires_at"))
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_created_by"))
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_token_hash"))
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_purpose"))
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_user_id"))
        batch_op.drop_index(batch_op.f("ix_account_action_tokens_id"))
    op.drop_table("account_action_tokens")

    op.drop_table("instance_policies")

    with batch_alter_table("workspace_invites", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_workspace_invites_revoked_by_users",
            type_="foreignkey",
        )
        batch_op.drop_column("revoked_by")
        batch_op.drop_column("revoked_at")

    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_column("last_login_at")
        batch_op.drop_column("session_version")
