"""require explicit annotation round access

Revision ID: b5f9d2a7c014
Revises: a4e8c1f6b903
Create Date: 2026-07-27 21:20:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

revision: str = "b5f9d2a7c014"
down_revision: str | None = "a4e8c1f6b903"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("annotation_rounds") as batch_op:
        batch_op.add_column(
            sa.Column(
                "open_to_all_annotators",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )
        batch_op.create_index(
            batch_op.f("ix_annotation_rounds_open_to_all_annotators"),
            ["open_to_all_annotators"],
        )
    rounds = sa.table(
        "annotation_rounds",
        sa.column("id", sa.Integer()),
        sa.column("annotator_user_ids", JSONType()),
        sa.column("open_to_all_annotators", sa.Boolean()),
    )
    connection = op.get_bind()
    for row in connection.execute(
        sa.select(rounds.c.id, rounds.c.annotator_user_ids)
    ).mappings():
        if not row["annotator_user_ids"]:
            connection.execute(
                rounds.update()
                .where(rounds.c.id == row["id"])
                .values(open_to_all_annotators=True)
            )


def downgrade() -> None:
    with op.batch_alter_table("annotation_rounds") as batch_op:
        batch_op.drop_index(
            batch_op.f("ix_annotation_rounds_open_to_all_annotators")
        )
        batch_op.drop_column("open_to_all_annotators")
