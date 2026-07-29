"""add label set decision lineage

Revision ID: a4e8c1f6b903
Revises: f3c6a9d2e5b8
Create Date: 2026-07-27 20:10:00.000000
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

from al_medlit.core.types import JSONType

revision: str = "a4e8c1f6b903"
down_revision: str | None = "f3c6a9d2e5b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("label_set_versions") as batch_op:
        batch_op.add_column(
            sa.Column("source_annotation_round_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "source_submission_ids",
                JSONType(),
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "source_decision_ids",
                JSONType(),
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )
        batch_op.create_index(
            batch_op.f("ix_label_set_versions_source_annotation_round_id"),
            ["source_annotation_round_id"],
        )
        batch_op.create_foreign_key(
            "fk_label_set_versions_source_annotation_round_id",
            "annotation_rounds",
            ["source_annotation_round_id"],
            ["id"],
        )


def downgrade() -> None:
    with op.batch_alter_table("label_set_versions") as batch_op:
        batch_op.drop_constraint(
            "fk_label_set_versions_source_annotation_round_id",
            type_="foreignkey",
        )
        batch_op.drop_index(
            batch_op.f("ix_label_set_versions_source_annotation_round_id")
        )
        batch_op.drop_column("source_decision_ids")
        batch_op.drop_column("source_submission_ids")
        batch_op.drop_column("source_annotation_round_id")
