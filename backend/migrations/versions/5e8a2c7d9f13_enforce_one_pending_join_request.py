# pyright: reportAttributeAccessIssue=false

"""enforce one pending join request per workspace user

Revision ID: 5e8a2c7d9f13
Revises: 7a2e4c9b1d60
Create Date: 2026-07-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5e8a2c7d9f13"
down_revision: str | None = "7a2e4c9b1d60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INDEX_NAME = "uq_workspace_join_requests_pending_user"


def upgrade() -> None:
    # Preserve the earliest request and close duplicate pending rows before the
    # partial unique index is installed on an existing shared deployment.
    op.execute(
        sa.text(
            """
            UPDATE workspace_join_requests
            SET status = 'rejected',
                decided_at = COALESCE(decided_at, CURRENT_TIMESTAMP)
            WHERE status = 'pending'
              AND id NOT IN (
                  SELECT MIN(id)
                  FROM workspace_join_requests
                  WHERE status = 'pending'
                  GROUP BY workspace_id, user_id
              )
            """
        )
    )
    op.create_index(
        INDEX_NAME,
        "workspace_join_requests",
        ["workspace_id", "user_id"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
        sqlite_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(INDEX_NAME, table_name="workspace_join_requests")
