"""Allow new assignment rounds after workspace offboarding.

Revision ID: b7c2e4a91d63
Revises: c0e6f8a3b7d2
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7c2e4a91d63"
down_revision: str | None = "c0e6f8a3b7d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_CONSTRAINT = "uq_task_assignments_task_document_assignee_scope"
LIVE_INDEX = "uq_task_assignments_live_scope"
SCOPE_COLUMNS = [
    "task_id",
    "document_id",
    "assignee_user_id",
    "assignment_scope_key",
]


def upgrade() -> None:
    with op.batch_alter_table("task_assignments") as batch:
        batch.drop_constraint(OLD_CONSTRAINT, type_="unique")
    op.create_index(
        LIVE_INDEX,
        "task_assignments",
        SCOPE_COLUMNS,
        unique=True,
        postgresql_where=sa.text("status != 'withdrawn'"),
        sqlite_where=sa.text("status != 'withdrawn'"),
    )


def downgrade() -> None:
    duplicate = op.get_bind().execute(
        sa.text(
            """
            SELECT 1 FROM task_assignments
            GROUP BY task_id, document_id, assignee_user_id, assignment_scope_key
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if duplicate is not None:
        raise RuntimeError(
            "Cannot restore assignment uniqueness after multiple rounds in the same scope"
        )
    op.drop_index(LIVE_INDEX, table_name="task_assignments")
    with op.batch_alter_table("task_assignments") as batch:
        batch.create_unique_constraint(OLD_CONSTRAINT, SCOPE_COLUMNS)
