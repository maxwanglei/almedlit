"""Enforce round sequences when the round has no learning cycle.

Revision ID: c9e4a7b12d68
Revises: b7c2e4a91d63
"""

import sqlalchemy as sa
from alembic import op

revision = "c9e4a7b12d68"
down_revision = "b7c2e4a91d63"
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    if connection.dialect.name == "postgresql":
        connection.execute(sa.text("LOCK TABLE annotation_rounds IN EXCLUSIVE MODE"))
    connection.execute(
        sa.text("""
        WITH ranked AS (
            SELECT id, project_id, sequence,
                   ROW_NUMBER() OVER (PARTITION BY project_id, sequence ORDER BY id) AS duplicate_rank,
                   MAX(sequence) OVER (PARTITION BY project_id) AS max_sequence
            FROM annotation_rounds WHERE cycle_id IS NULL
        ), duplicates AS (
            SELECT id, max_sequence + ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY id) AS new_sequence
            FROM ranked WHERE duplicate_rank > 1
        )
        UPDATE annotation_rounds
        SET sequence = (SELECT new_sequence FROM duplicates WHERE duplicates.id = annotation_rounds.id)
        WHERE id IN (SELECT id FROM duplicates)
    """)
    )
    op.create_index(
        "uq_annotation_rounds_project_sequence_without_cycle",
        "annotation_rounds",
        ["project_id", "sequence"],
        unique=True,
        postgresql_where=sa.text("cycle_id IS NULL"),
        sqlite_where=sa.text("cycle_id IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_annotation_rounds_project_sequence_without_cycle", table_name="annotation_rounds"
    )
