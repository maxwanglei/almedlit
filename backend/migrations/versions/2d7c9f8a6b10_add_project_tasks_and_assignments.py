# pyright: reportAttributeAccessIssue=false

"""add project tasks and assignments

Revision ID: 2d7c9f8a6b10
Revises: c4d2f7a8b591
Create Date: 2026-05-28 00:00:00.000000
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

batch_alter_table: Any = op.batch_alter_table
create_table: Any = op.create_table
drop_table: Any = op.drop_table


revision: str = "2d7c9f8a6b10"
down_revision: str | None = "c4d2f7a8b591"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TASK_DISPLAY_NAMES = {
    "entity": "Entity Annotation",
    "relation": "Relation Annotation",
    "doc_label": "Document Annotation",
    "sentence_label": "Sentence Annotation",
    "passage_label": "Passage Annotation",
}


def _parse_json(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return {}


def _seed_project_tasks_from_annotation_schema() -> None:
    connection = op.get_bind()
    project_tasks = sa.table(
        "project_tasks",
        sa.column("project_id", sa.Integer()),
        sa.column("annotation_type", sa.String()),
        sa.column("display_name", sa.String()),
        sa.column("description", sa.Text()),
        sa.column("enabled", sa.Boolean()),
        sa.column("sort_order", sa.Integer()),
        sa.column("labels", JSONType()),
        sa.column("settings", JSONType()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    projects = connection.execute(
        sa.text("SELECT id, annotation_schema FROM projects")
    ).mappings()
    for project in projects:
        annotation_schema = _parse_json(project["annotation_schema"])
        labels_by_type = annotation_schema.get("labels")
        if not isinstance(labels_by_type, dict):
            continue

        for sort_order, (annotation_type, labels) in enumerate(labels_by_type.items()):
            if annotation_type not in TASK_DISPLAY_NAMES:
                continue
            now = datetime.now(UTC)
            connection.execute(
                project_tasks.insert().values(
                    project_id=project["id"],
                    annotation_type=annotation_type,
                    display_name=TASK_DISPLAY_NAMES[annotation_type],
                    description=None,
                    enabled=True,
                    sort_order=sort_order,
                    labels=labels or [],
                    settings={},
                    created_at=now,
                    updated_at=now,
                )
            )


def upgrade() -> None:
    with batch_alter_table("projects", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "annotation_validation_mode",
                sa.String(length=20),
                nullable=False,
                server_default="relaxed",
            )
        )

    create_table(
        "project_tasks",
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("annotation_type", sa.String(length=80), nullable=False),
        sa.Column("display_name", sa.String(length=120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("labels", JSONType(), nullable=False),
        sa.Column("settings", JSONType(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "project_id",
            "annotation_type",
            name="uq_project_tasks_project_annotation_type",
        ),
    )
    with batch_alter_table("project_tasks", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_project_tasks_annotation_type"), ["annotation_type"])
        batch_op.create_index(batch_op.f("ix_project_tasks_enabled"), ["enabled"])
        batch_op.create_index(batch_op.f("ix_project_tasks_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_project_tasks_project_id"), ["project_id"])

    _seed_project_tasks_from_annotation_schema()

    create_table(
        "task_assignments",
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Integer(), nullable=False),
        sa.Column("annotator_id", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("assigned_by", sa.String(length=120), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("metadata", JSONType(), nullable=False),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["task_id"], ["project_tasks.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "task_id",
            "document_id",
            "annotator_id",
            name="uq_task_assignments_task_document_annotator",
        ),
    )
    with batch_alter_table("task_assignments", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_task_assignments_annotator_id"), ["annotator_id"])
        batch_op.create_index(batch_op.f("ix_task_assignments_document_id"), ["document_id"])
        batch_op.create_index(batch_op.f("ix_task_assignments_id"), ["id"])
        batch_op.create_index(batch_op.f("ix_task_assignments_project_id"), ["project_id"])
        batch_op.create_index(batch_op.f("ix_task_assignments_status"), ["status"])
        batch_op.create_index(batch_op.f("ix_task_assignments_task_id"), ["task_id"])


def downgrade() -> None:
    with batch_alter_table("task_assignments", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_task_assignments_task_id"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_status"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_project_id"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_id"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_document_id"))
        batch_op.drop_index(batch_op.f("ix_task_assignments_annotator_id"))

    drop_table("task_assignments")

    with batch_alter_table("project_tasks", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_project_tasks_project_id"))
        batch_op.drop_index(batch_op.f("ix_project_tasks_id"))
        batch_op.drop_index(batch_op.f("ix_project_tasks_enabled"))
        batch_op.drop_index(batch_op.f("ix_project_tasks_annotation_type"))

    drop_table("project_tasks")

    with batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_column("annotation_validation_mode")
