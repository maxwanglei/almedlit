# pyright: reportAttributeAccessIssue=false

"""add annotation relation fks

Revision ID: 8f4b6d9e2a31
Revises: 11944d2ce893
Create Date: 2026-05-24 00:00:00.000000
"""

from typing import Any, Sequence, Union

from alembic import op
import sqlalchemy as sa


batch_alter_table: Any = getattr(op, "batch_alter_table")


revision: str = "8f4b6d9e2a31"
down_revision: Union[str, None] = "11944d2ce893"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("head_annotation_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("tail_annotation_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_annotations_head_annotation_id",
            "annotations",
            ["head_annotation_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_annotations_tail_annotation_id",
            "annotations",
            ["tail_annotation_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_annotations_head_annotation_id",
            ["head_annotation_id"],
            unique=False,
        )
        batch_op.create_index(
            "ix_annotations_tail_annotation_id",
            ["tail_annotation_id"],
            unique=False,
        )


def downgrade() -> None:
    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.drop_index("ix_annotations_tail_annotation_id")
        batch_op.drop_index("ix_annotations_head_annotation_id")
        batch_op.drop_constraint(
            "fk_annotations_tail_annotation_id",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_annotations_head_annotation_id",
            type_="foreignkey",
        )
        batch_op.drop_column("tail_annotation_id")
        batch_op.drop_column("head_annotation_id")