# pyright: reportAttributeAccessIssue=false

"""add annotation guideline_version_id fk

Revision ID: c4d2f7a8b591
Revises: 8f4b6d9e2a31
Create Date: 2026-05-25 00:00:00.000000
"""

from typing import Any, Sequence, Union

from alembic import op


batch_alter_table: Any = getattr(op, "batch_alter_table")


revision: str = "c4d2f7a8b591"
down_revision: Union[str, None] = "8f4b6d9e2a31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.create_foreign_key(
            "fk_annotations_guideline_version_id",
            "guideline_versions",
            ["guideline_version_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_annotations_guideline_version_id",
            ["guideline_version_id"],
            unique=False,
        )


def downgrade() -> None:
    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.drop_index("ix_annotations_guideline_version_id")
        batch_op.drop_constraint(
            "fk_annotations_guideline_version_id",
            type_="foreignkey",
        )
