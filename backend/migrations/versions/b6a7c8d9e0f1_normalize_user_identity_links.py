# pyright: reportAttributeAccessIssue=false

"""normalize user identity links

Revision ID: b6a7c8d9e0f1
Revises: e5f8a1b2c3d4
Create Date: 2026-06-16 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op
from passlib.context import CryptContext

batch_alter_table: Any = op.batch_alter_table

revision: str = "b6a7c8d9e0f1"
down_revision: str | None = "e5f8a1b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _table_has_column(bind, table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _distinct_names(bind, table_name: str, column_name: str) -> set[str]:
    if not _table_has_column(bind, table_name, column_name):
        return set()
    rows = bind.execute(
        sa.text(f"SELECT DISTINCT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL")
    ).fetchall()
    return {str(row[0]).strip() for row in rows if row[0] and str(row[0]).strip()}


def _ensure_user_ids(bind, names: set[str]) -> dict[str, int]:
    now = datetime.now(UTC)
    unusable_hash = _pwd.hash("!" + "x" * 16)
    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("username", sa.String),
        sa.column("email", sa.String),
        sa.column("password_hash", sa.String),
        sa.column("display_name", sa.String),
        sa.column("is_active", sa.Boolean),
        sa.column("is_superuser", sa.Boolean),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )
    user_ids: dict[str, int] = {}
    for username in sorted(names):
        existing = bind.execute(
            sa.text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        ).first()
        if existing is not None:
            user_ids[username] = existing[0]
            continue
        result = bind.execute(
            users.insert().values(
                username=username,
                email=None,
                password_hash=unusable_hash,
                display_name=username,
                is_active=False,
                is_superuser=False,
                created_at=now,
                updated_at=now,
            )
        )
        inserted_pk = tuple(result.inserted_primary_key or ())
        user_ids[username] = (
            inserted_pk[0]
            if inserted_pk
            else bind.execute(
                sa.text("SELECT id FROM users WHERE username = :username"),
                {"username": username},
            ).scalar_one()
        )
    return user_ids


def _backfill_column_from_username(
    bind,
    *,
    table_name: str,
    source_column: str,
    target_column: str,
    user_ids: dict[str, int],
) -> None:
    if not _table_has_column(bind, table_name, target_column):
        return
    rows = bind.execute(
        sa.text(f"SELECT id, {source_column} FROM {table_name} WHERE {source_column} IS NOT NULL")
    ).fetchall()
    for row_id, username in rows:
        cleaned = str(username).strip() if username else ""
        user_id = user_ids.get(cleaned)
        if user_id is None:
            continue
        bind.execute(
            sa.text(f"UPDATE {table_name} SET {target_column} = :user_id WHERE id = :row_id"),
            {"user_id": user_id, "row_id": row_id},
        )


def upgrade() -> None:
    bind = op.get_bind()

    with batch_alter_table("task_assignments", schema=None) as batch_op:
        batch_op.add_column(sa.Column("assignee_user_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("assigned_by_user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_task_assignments_assignee_user_id_users",
            "users",
            ["assignee_user_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            "fk_task_assignments_assigned_by_user_id_users",
            "users",
            ["assigned_by_user_id"],
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_task_assignments_assignee_user_id"),
            ["assignee_user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_task_assignments_assigned_by_user_id"),
            ["assigned_by_user_id"],
        )

    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("annotator_user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_annotations_annotator_user_id_users",
            "users",
            ["annotator_user_id"],
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_annotations_annotator_user_id"),
            ["annotator_user_id"],
        )

    with batch_alter_table("annotation_submissions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("annotator_user_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_annotation_submissions_annotator_user_id_users",
            "users",
            ["annotator_user_id"],
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_annotation_submissions_annotator_user_id"),
            ["annotator_user_id"],
        )

    names: set[str] = set()
    for table_name, column_name in [
        ("task_assignments", "annotator_id"),
        ("task_assignments", "assigned_by"),
        ("annotations", "annotator_id"),
        ("annotation_submissions", "annotator_id"),
    ]:
        names.update(_distinct_names(bind, table_name, column_name))
    user_ids = _ensure_user_ids(bind, names)

    _backfill_column_from_username(
        bind,
        table_name="task_assignments",
        source_column="annotator_id",
        target_column="assignee_user_id",
        user_ids=user_ids,
    )
    _backfill_column_from_username(
        bind,
        table_name="task_assignments",
        source_column="assigned_by",
        target_column="assigned_by_user_id",
        user_ids=user_ids,
    )
    _backfill_column_from_username(
        bind,
        table_name="annotations",
        source_column="annotator_id",
        target_column="annotator_user_id",
        user_ids=user_ids,
    )
    _backfill_column_from_username(
        bind,
        table_name="annotation_submissions",
        source_column="annotator_id",
        target_column="annotator_user_id",
        user_ids=user_ids,
    )

    with batch_alter_table("task_assignments", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_task_assignments_task_document_annotator",
            type_="unique",
        )
        batch_op.alter_column(
            "assignee_user_id",
            existing_type=sa.Integer(),
            nullable=False,
        )
        batch_op.create_unique_constraint(
            "uq_task_assignments_task_document_assignee",
            ["task_id", "document_id", "assignee_user_id"],
        )


def downgrade() -> None:
    with batch_alter_table("task_assignments", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_task_assignments_task_document_assignee",
            type_="unique",
        )
        batch_op.create_unique_constraint(
            "uq_task_assignments_task_document_annotator",
            ["task_id", "document_id", "annotator_id"],
        )
        batch_op.drop_index(batch_op.f("ix_task_assignments_assigned_by_user_id"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_assignee_user_id"))
        batch_op.drop_constraint(
            "fk_task_assignments_assigned_by_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            "fk_task_assignments_assignee_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("assigned_by_user_id")
        batch_op.drop_column("assignee_user_id")

    with batch_alter_table("annotation_submissions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_annotation_submissions_annotator_user_id"))
        batch_op.drop_constraint(
            "fk_annotation_submissions_annotator_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("annotator_user_id")

    with batch_alter_table("annotations", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_annotations_annotator_user_id"))
        batch_op.drop_constraint(
            "fk_annotations_annotator_user_id_users",
            type_="foreignkey",
        )
        batch_op.drop_column("annotator_user_id")
