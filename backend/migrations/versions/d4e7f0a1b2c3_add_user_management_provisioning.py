# pyright: reportAttributeAccessIssue=false

"""add user management provisioning

Revision ID: d4e7f0a1b2c3
Revises: a3f6b8c1d2e4
Create Date: 2026-06-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

batch_alter_table: Any = op.batch_alter_table


revision: str = "d4e7f0a1b2c3"
down_revision: str | None = "a3f6b8c1d2e4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    # The preceding schema allowed nullable 255-character display names. Normalize
    # those rows before narrowing the column so a populated PostgreSQL deployment
    # cannot fail its ALTER when legacy values exceed the new application limit.
    bind.execute(
        sa.text(
            "UPDATE users "
            "SET display_name = SUBSTR(COALESCE(display_name, ''), 1, 120) "
            "WHERE display_name IS NULL OR LENGTH(display_name) > 120"
        )
    )

    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.add_column(sa.Column("email", sa.String(length=255), nullable=True))
        batch_op.alter_column(
            "display_name",
            existing_type=sa.String(length=255),
            type_=sa.String(length=120),
            nullable=False,
            server_default="",
        )

    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.add_column(sa.Column("join_code", sa.String(length=40), nullable=True))
        batch_op.create_index(
            batch_op.f("ix_workspaces_join_code"),
            ["join_code"],
            unique=True,
        )

    op.create_table(
        "workspace_invites",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(length=64), nullable=False),
        sa.Column(
            "role",
            sa.String(length=40),
            nullable=False,
            server_default="annotator",
        ),
        sa.Column("created_by", sa.Integer(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("accepted_by", sa.Integer(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["accepted_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("workspace_invites", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_workspace_invites_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_workspace_invites_role"), ["role"])
        batch_op.create_index(
            batch_op.f("ix_workspace_invites_token"),
            ["token"],
            unique=True,
        )
        batch_op.create_index(
            batch_op.f("ix_workspace_invites_workspace_id"),
            ["workspace_id"],
        )

    op.create_table(
        "workspace_join_requests",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("decided_by", sa.Integer(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["decided_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("workspace_join_requests", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_workspace_join_requests_id"), ["id"])
        batch_op.create_index(
            batch_op.f("ix_workspace_join_requests_status"),
            ["status"],
        )
        batch_op.create_index(
            batch_op.f("ix_workspace_join_requests_user_id"),
            ["user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_workspace_join_requests_workspace_id"),
            ["workspace_id"],
        )


def downgrade() -> None:
    with batch_alter_table("workspace_join_requests", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_workspace_join_requests_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_workspace_join_requests_user_id"))
        batch_op.drop_index(batch_op.f("ix_workspace_join_requests_status"))
        batch_op.drop_index(batch_op.f("ix_workspace_join_requests_id"))
    op.drop_table("workspace_join_requests")

    with batch_alter_table("workspace_invites", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_workspace_invites_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_workspace_invites_token"))
        batch_op.drop_index(batch_op.f("ix_workspace_invites_role"))
        batch_op.drop_index(batch_op.f("ix_workspace_invites_id"))
    op.drop_table("workspace_invites")

    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_workspaces_join_code"))
        batch_op.drop_column("join_code")

    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.alter_column(
            "display_name",
            existing_type=sa.String(length=120),
            type_=sa.String(length=255),
            nullable=True,
            server_default=None,
        )
        batch_op.drop_column("email")
