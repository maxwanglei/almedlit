# pyright: reportAttributeAccessIssue=false

"""backfill project tasks from annotation schema

Revision ID: f7a9c1d2e3b4
Revises: 2d7c9f8a6b10
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


revision: str = "f7a9c1d2e3b4"
down_revision: str | None = "2d7c9f8a6b10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


TASK_DISPLAY_NAMES = {
    "entity": "Entity Annotation",
    "relation": "Relation Annotation",
    "doc_label": "Document Annotation",
    "sentence_label": "Sentence Annotation",
    "passage_label": "Passage Annotation",
}


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _labels_by_type(annotation_schema: dict[str, Any]) -> dict[str, list]:
    labels = annotation_schema.get("labels")
    if isinstance(labels, list):
        return {"entity": labels}
    if isinstance(labels, dict):
        return {
            annotation_type: label_list
            for annotation_type, label_list in labels.items()
            if isinstance(label_list, list)
        }
    return {}


def upgrade() -> None:
    connection = op.get_bind()
    projects = sa.table(
        "projects",
        sa.column("id", sa.Integer()),
        sa.column("annotation_schema", JSONType()),
    )
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

    existing_pairs = {
        (row["project_id"], row["annotation_type"])
        for row in connection.execute(
            sa.select(project_tasks.c.project_id, project_tasks.c.annotation_type)
        ).mappings()
    }

    for project in connection.execute(
        sa.select(projects.c.id, projects.c.annotation_schema)
    ).mappings():
        for sort_order, (annotation_type, labels) in enumerate(
            _labels_by_type(_parse_json(project["annotation_schema"])).items()
        ):
            if annotation_type not in TASK_DISPLAY_NAMES:
                continue
            if (project["id"], annotation_type) in existing_pairs:
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
                    labels=labels,
                    settings={},
                    created_at=now,
                    updated_at=now,
                )
            )
            existing_pairs.add((project["id"], annotation_type))


def downgrade() -> None:
    # No-op: the upgrade only reconstructs missing canonical task rows.
    pass
