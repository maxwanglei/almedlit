"""add feedback scoring execution state

Revision ID: 82c4d6e8f1a3
Revises: 71b9d2e4f6a8
Create Date: 2026-08-18 00:00:00.000000
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

batch_alter_table: Any = op.batch_alter_table

revision: str = "82c4d6e8f1a3"
down_revision: str | None = "71b9d2e4f6a8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with batch_alter_table("feedback_runs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("output_feedback_set_version_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("failure_code", sa.String(length=80), nullable=True))
        batch_op.add_column(sa.Column("failure_reason", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_foreign_key(
            "fk_feedback_runs_output_feedback_set_version_id",
            "feedback_set_versions",
            ["output_feedback_set_version_id"],
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_feedback_runs_output_feedback_set_version_id"),
            ["output_feedback_set_version_id"],
            unique=False,
        )

    op.execute(
        """
        UPDATE feedback_runs
        SET output_feedback_set_version_id = (
                SELECT feedback_set_versions.id
                FROM feedback_set_versions
                WHERE feedback_set_versions.feedback_run_id = feedback_runs.id
                ORDER BY feedback_set_versions.version_number DESC,
                         feedback_set_versions.id DESC
                LIMIT 1
            ),
            completed_at = COALESCE(
                completed_at,
                (
                    SELECT feedback_set_versions.created_at
                    FROM feedback_set_versions
                    WHERE feedback_set_versions.feedback_run_id = feedback_runs.id
                    ORDER BY feedback_set_versions.version_number DESC,
                             feedback_set_versions.id DESC
                    LIMIT 1
                )
            )
        WHERE EXISTS (
            SELECT 1
            FROM feedback_set_versions
            WHERE feedback_set_versions.feedback_run_id = feedback_runs.id
        )
        """
    )


def downgrade() -> None:
    with batch_alter_table("feedback_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_feedback_runs_output_feedback_set_version_id"))
        batch_op.drop_constraint(
            "fk_feedback_runs_output_feedback_set_version_id",
            type_="foreignkey",
        )
        batch_op.drop_column("completed_at")
        batch_op.drop_column("heartbeat_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("failure_reason")
        batch_op.drop_column("failure_code")
        batch_op.drop_column("output_feedback_set_version_id")
