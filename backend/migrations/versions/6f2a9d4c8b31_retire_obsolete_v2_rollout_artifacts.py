"""retire obsolete v2 rollout artifacts

Revision ID: 6f2a9d4c8b31
Revises: 4e8b1c6d9a20
Create Date: 2026-07-30 00:30:00.000000
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "6f2a9d4c8b31"
down_revision: str | None = "4e8b1c6d9a20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

LEGACY_BACKFILL_TABLE = "legacy_v2_backfill_cohorts"
ROLLOUT_COLUMN = "platform_v2_enabled"


def upgrade() -> None:
    connection = op.get_bind()
    inspector = sa.inspect(connection)
    tables = set(inspector.get_table_names())
    has_legacy_backfill = LEGACY_BACKFILL_TABLE in tables

    if has_legacy_backfill:
        row_count = connection.execute(
            sa.text(f"SELECT COUNT(*) FROM {LEGACY_BACKFILL_TABLE}")
        ).scalar_one()
        if row_count:
            raise RuntimeError(
                f"{LEGACY_BACKFILL_TABLE} contains {row_count} provenance "
                "record(s). Export or migrate those records to a canonical "
                "archive before retrying; this migration will not discard them."
            )

    workspace_columns = {column["name"] for column in inspector.get_columns("workspaces")}
    if has_legacy_backfill:
        op.drop_table(LEGACY_BACKFILL_TABLE)
    if ROLLOUT_COLUMN in workspace_columns:
        with op.batch_alter_table("workspaces") as batch_op:
            batch_op.drop_column(ROLLOUT_COLUMN)


def downgrade() -> None:
    # These objects are absent from the checked-in canonical lineage. Recreating
    # an empty ledger or a false rollout flag would not restore historical state.
    pass
