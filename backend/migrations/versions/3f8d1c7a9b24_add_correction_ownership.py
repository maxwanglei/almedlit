# pyright: reportAttributeAccessIssue=false

"""add correction ownership

Revision ID: 3f8d1c7a9b24
Revises: f2c9d1e4a6b8
Create Date: 2026-07-09 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "3f8d1c7a9b24"
down_revision: str | None = "f2c9d1e4a6b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _backfill_unambiguous_owners() -> None:
    corrections = sa.table(
        "annotation_corrections",
        sa.column("original_annotation_id", sa.Integer()),
        sa.column("corrected_annotation_id", sa.Integer()),
        sa.column("created_by_user_id", sa.Integer()),
    )
    annotations = sa.table(
        "annotations",
        sa.column("id", sa.Integer()),
        sa.column("annotator_user_id", sa.Integer()),
    )

    inferred_owner = (
        sa.select(
            sa.case(
                (
                    sa.func.count(sa.distinct(annotations.c.annotator_user_id)) == 1,
                    sa.func.min(annotations.c.annotator_user_id),
                ),
                else_=sa.null(),
            )
        )
        .where(
            annotations.c.annotator_user_id.is_not(None),
            sa.or_(
                annotations.c.id == corrections.c.original_annotation_id,
                annotations.c.id == corrections.c.corrected_annotation_id,
            ),
        )
        .correlate(corrections)
        .scalar_subquery()
    )

    op.get_bind().execute(
        sa.update(corrections).values(created_by_user_id=inferred_owner)
    )


def upgrade() -> None:
    with op.batch_alter_table("annotation_corrections", schema=None) as batch_op:
        batch_op.add_column(sa.Column("created_by_user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_annotation_corrections_created_by_user_id_users",
            "users",
            ["created_by_user_id"],
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_annotation_corrections_created_by_user_id"),
            ["created_by_user_id"],
        )

    _backfill_unambiguous_owners()


def downgrade() -> None:
    with op.batch_alter_table("annotation_corrections", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_annotation_corrections_created_by_user_id"))
        batch_op.drop_constraint(
            "fk_annotation_corrections_created_by_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("created_by_user_id")
