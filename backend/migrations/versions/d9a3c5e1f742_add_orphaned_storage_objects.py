"""add orphaned storage object reclaim queue

Revision ID: d9a3c5e1f742
Revises: a6d1f9c3e8b2
Create Date: 2026-08-30 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "d9a3c5e1f742"
down_revision: str | None = "a6d1f9c3e8b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "orphaned_storage_objects",
        sa.Column("storage_key", sa.String(length=512), nullable=False),
        sa.Column("origin", sa.String(length=100), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("last_attempted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "attempts >= 0",
            name="ck_orphaned_storage_objects_nonnegative_attempts",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "storage_key",
            name="uq_orphaned_storage_objects_storage_key",
        ),
    )
    with op.batch_alter_table("orphaned_storage_objects") as batch_op:
        batch_op.create_index(batch_op.f("ix_orphaned_storage_objects_id"), ["id"])
        batch_op.create_index(
            batch_op.f("ix_orphaned_storage_objects_origin"),
            ["origin"],
        )
        batch_op.create_index(
            batch_op.f("ix_orphaned_storage_objects_next_attempt_at"),
            ["next_attempt_at"],
        )


def downgrade() -> None:
    with op.batch_alter_table("orphaned_storage_objects") as batch_op:
        batch_op.drop_index(batch_op.f("ix_orphaned_storage_objects_next_attempt_at"))
        batch_op.drop_index(batch_op.f("ix_orphaned_storage_objects_origin"))
        batch_op.drop_index(batch_op.f("ix_orphaned_storage_objects_id"))
    op.drop_table("orphaned_storage_objects")
