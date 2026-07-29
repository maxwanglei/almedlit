# pyright: reportAttributeAccessIssue=false

"""add workspace capabilities

Revision ID: a3f6b8c1d2e4
Revises: 9b2d4f6a8c10
Create Date: 2026-06-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

batch_alter_table: Any = op.batch_alter_table


revision: str = "a3f6b8c1d2e4"
down_revision: str | None = "9b2d4f6a8c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("username", sa.String(length=120), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_users_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_users_is_active"), ["is_active"])
        batch_op.create_index(batch_op.f("ix_users_is_superuser"), ["is_superuser"])
        batch_op.create_index(batch_op.f("ix_users_username"), ["username"], unique=True)

    op.create_table(
        "workspaces",
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "kind",
            sa.String(length=40),
            nullable=False,
            server_default="individual",
        ),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column(
            "capability_preset",
            sa.String(length=40),
            nullable=False,
            server_default="annotate",
        ),
        sa.Column(
            "capability_overrides",
            JSONType(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_workspaces_created_by"), ["created_by"])
        batch_op.create_index(batch_op.f("ix_workspaces_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_workspaces_kind"), ["kind"])
        batch_op.create_index(batch_op.f("ix_workspaces_name"), ["name"])

    op.create_table(
        "workspace_members",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "role",
            sa.String(length=40),
            nullable=False,
            server_default="annotator",
        ),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_id",
            "user_id",
            name="uq_workspace_members_workspace_user",
        ),
    )
    with batch_alter_table("workspace_members", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_workspace_members_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_workspace_members_role"), ["role"])
        batch_op.create_index(batch_op.f("ix_workspace_members_user_id"), ["user_id"])
        batch_op.create_index(
            batch_op.f("ix_workspace_members_workspace_id"),
            ["workspace_id"],
        )

    with batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(sa.Column("workspace_id", sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f("ix_projects_workspace_id"), ["workspace_id"])
        batch_op.create_foreign_key(
            "fk_projects_workspace_id_workspaces",
            "workspaces",
            ["workspace_id"],
            ["id"],
        )

    bind = op.get_bind()
    now = datetime.now(UTC)
    workspaces = sa.table(
        "workspaces",
        sa.column("name", sa.String()),
        sa.column("kind", sa.String()),
        sa.column("created_by", sa.Integer()),
        sa.column("capability_preset", sa.String()),
        sa.column("capability_overrides", JSONType()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    bind.execute(
        workspaces.insert().values(
            name="Default",
            kind="team",
            created_by=None,
            capability_preset="full",
            capability_overrides=[],
            created_at=now,
            updated_at=now,
        )
    )
    default_ws_id = bind.execute(
        sa.text(
            "SELECT id FROM workspaces WHERE name = 'Default' "
            "AND kind = 'team' ORDER BY id LIMIT 1"
        )
    ).scalar_one()

    bind.execute(
        sa.text("UPDATE projects SET workspace_id = :w WHERE workspace_id IS NULL"),
        {"w": default_ws_id},
    )

    with batch_alter_table("projects", schema=None) as batch_op:
        batch_op.alter_column(
            "workspace_id",
            existing_type=sa.Integer(),
            nullable=False,
        )


def downgrade() -> None:
    with batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_projects_workspace_id_workspaces",
            type_="foreignkey",
        )
        batch_op.drop_index(batch_op.f("ix_projects_workspace_id"))
        batch_op.drop_column("workspace_id")

    with batch_alter_table("workspace_members", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_workspace_members_workspace_id"))
        batch_op.drop_index(batch_op.f("ix_workspace_members_user_id"))
        batch_op.drop_index(batch_op.f("ix_workspace_members_role"))
        batch_op.drop_index(batch_op.f("ix_workspace_members_id"))
    op.drop_table("workspace_members")

    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_workspaces_name"))
        batch_op.drop_index(batch_op.f("ix_workspaces_kind"))
        batch_op.drop_index(batch_op.f("ix_workspaces_id"))
        batch_op.drop_index(batch_op.f("ix_workspaces_created_by"))
    op.drop_table("workspaces")

    with batch_alter_table("users", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_users_username"))
        batch_op.drop_index(batch_op.f("ix_users_is_superuser"))
        batch_op.drop_index(batch_op.f("ix_users_is_active"))
        batch_op.drop_index(batch_op.f("ix_users_id"))
    op.drop_table("users")
