"""add learning cycle feedback sources

Revision ID: e1c4a7b9d203
Revises: d9a3f5b7c102
Create Date: 2026-07-27 18:35:00.000000
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

from al_medlit.core.types import JSONType

revision: str = "e1c4a7b9d203"
down_revision: str | None = "c8d2e4f6a901"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("learning_cycles") as batch_op:
        batch_op.add_column(
            sa.Column(
                "feedback_sources",
                JSONType(),
                nullable=False,
                server_default=sa.text("'[]'"),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("learning_cycles") as batch_op:
        batch_op.drop_column("feedback_sources")
