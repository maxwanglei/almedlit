# pyright: reportAttributeAccessIssue=false

"""make timestamp columns timezone aware

Revision ID: 6c1a8f0e9d72
Revises: f7a9c1d2e3b4
Create Date: 2026-06-09 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


batch_alter_table: Any = op.batch_alter_table


revision: str = "6c1a8f0e9d72"
down_revision: str | None = "f7a9c1d2e3b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TIMESTAMP_TABLES = (
    "projects",
    "documents",
    "error_patterns",
    "guideline_versions",
    "annotations",
    "guideline_atoms",
    "model_predictions",
    "annotation_corrections",
    "micro_question_templates",
    "training_actions",
    "project_tasks",
    "task_assignments",
)
TIMESTAMP_COLUMNS = ("created_at", "updated_at")


def _matching_timestamp_columns(
    connection: Any,
    table_name: str,
    *,
    timezone: bool,
) -> list[str]:
    columns_by_name = {
        column["name"]: column for column in inspect(connection).get_columns(table_name)
    }
    return [
        column_name
        for column_name in TIMESTAMP_COLUMNS
        if column_name in columns_by_name
        and bool(getattr(columns_by_name[column_name]["type"], "timezone", False)) == timezone
    ]


def _alter_timestamp_columns(*, to_timezone: bool) -> None:
    connection = op.get_bind()
    dialect_name = connection.dialect.name
    if dialect_name == "sqlite":
        return

    from_timezone = not to_timezone
    for table_name in TIMESTAMP_TABLES:
        for column_name in _matching_timestamp_columns(
            connection,
            table_name,
            timezone=from_timezone,
        ):
            alter_kwargs: dict[str, Any] = {
                "existing_type": sa.DateTime(timezone=from_timezone),
                "type_": sa.DateTime(timezone=to_timezone),
                "existing_nullable": False,
            }
            if dialect_name == "postgresql":
                alter_kwargs["postgresql_using"] = f"{column_name} AT TIME ZONE 'UTC'"

            with batch_alter_table(table_name, schema=None) as batch_op:
                batch_op.alter_column(column_name, **alter_kwargs)


def upgrade() -> None:
    _alter_timestamp_columns(to_timezone=True)


def downgrade() -> None:
    _alter_timestamp_columns(to_timezone=False)
