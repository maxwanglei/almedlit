"""widen lineage artifact size

Revision ID: e7b4c2d9a106
Revises: c3f7a1e9d5b2
Create Date: 2026-07-27 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "e7b4c2d9a106"
down_revision: str | None = "c3f7a1e9d5b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("lineage_artifacts") as batch_op:
        batch_op.alter_column(
            "size_bytes",
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("lineage_artifacts") as batch_op:
        batch_op.alter_column(
            "size_bytes",
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=False,
        )
