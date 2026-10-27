"""Persist annotation round creation requests across client retries.

Revision ID: f6a1d8c3e902
Revises: d3f7c9e2a501
"""

import sqlalchemy as sa
from alembic import op

revision = "f6a1d8c3e902"
down_revision = "d3f7c9e2a501"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("annotation_rounds") as batch:
        batch.add_column(sa.Column("request_key", sa.String(120), nullable=True))
        batch.add_column(sa.Column("request_hash", sa.String(64), nullable=True))
        batch.create_unique_constraint(
            "uq_annotation_rounds_actor_request",
            ["project_id", "created_by_user_id", "request_key"],
        )


def downgrade() -> None:
    with op.batch_alter_table("annotation_rounds") as batch:
        batch.drop_constraint("uq_annotation_rounds_actor_request", type_="unique")
        batch.drop_column("request_hash")
        batch.drop_column("request_key")
