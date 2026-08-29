"""serialize active error-pattern aggregation

Revision ID: a6d1f9c3e8b2
Revises: 82c4d6e8f1a3
Create Date: 2026-08-29 00:00:00.000000
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a6d1f9c3e8b2"
down_revision: str | None = "82c4d6e8f1a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UNLABELED_INDEX = "uq_error_patterns_active_unlabeled"
LABELED_INDEX = "uq_error_patterns_active_labeled"


def _merge_active_duplicates() -> None:
    connection = op.get_bind()
    metadata = sa.MetaData()
    patterns = sa.Table("error_patterns", metadata, autoload_with=connection)
    guideline_atoms = sa.Table("guideline_atoms", metadata, autoload_with=connection)
    training_actions = sa.Table("training_actions", metadata, autoload_with=connection)

    rows = connection.execute(
        sa.select(
            patterns.c.id,
            patterns.c.project_id,
            patterns.c.task_type,
            patterns.c.error_type,
            patterns.c.label_type,
            patterns.c.example_count,
            patterns.c.example_ids,
        )
        .where(patterns.c.status == "active")
        .order_by(patterns.c.id)
    ).mappings()
    grouped: dict[tuple[int, str, str, str | None], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["project_id"],
            row["task_type"],
            row["error_type"],
            row["label_type"],
        )
        grouped[key].append(dict(row))

    for duplicates in grouped.values():
        if len(duplicates) < 2:
            continue
        keeper = duplicates[0]
        merged_examples = []
        merged_count = 0
        for pattern in duplicates:
            merged_count += int(pattern["example_count"] or 0)
            examples = pattern["example_ids"] or []
            if isinstance(examples, list):
                merged_examples.extend(examples)

        connection.execute(
            patterns.update()
            .where(patterns.c.id == keeper["id"])
            .values(example_count=merged_count, example_ids=merged_examples)
        )
        for duplicate in duplicates[1:]:
            duplicate_id = duplicate["id"]
            connection.execute(
                guideline_atoms.update()
                .where(guideline_atoms.c.error_pattern_id == duplicate_id)
                .values(error_pattern_id=keeper["id"])
            )
            connection.execute(
                training_actions.update()
                .where(training_actions.c.error_pattern_id == duplicate_id)
                .values(error_pattern_id=keeper["id"])
            )
            connection.execute(patterns.delete().where(patterns.c.id == duplicate_id))


def upgrade() -> None:
    _merge_active_duplicates()
    op.create_index(
        UNLABELED_INDEX,
        "error_patterns",
        ["project_id", "task_type", "error_type"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND label_type IS NULL"),
        sqlite_where=sa.text("status = 'active' AND label_type IS NULL"),
    )
    op.create_index(
        LABELED_INDEX,
        "error_patterns",
        ["project_id", "task_type", "error_type", "label_type"],
        unique=True,
        postgresql_where=sa.text("status = 'active' AND label_type IS NOT NULL"),
        sqlite_where=sa.text("status = 'active' AND label_type IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(LABELED_INDEX, table_name="error_patterns")
    op.drop_index(UNLABELED_INDEX, table_name="error_patterns")
