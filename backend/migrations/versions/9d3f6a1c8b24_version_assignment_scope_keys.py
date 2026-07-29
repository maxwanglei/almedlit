# pyright: reportAttributeAccessIssue=false

"""include structure and guideline versions in assignment scopes

Revision ID: 9d3f6a1c8b24
Revises: 5e8a2c7d9f13
Create Date: 2026-07-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9d3f6a1c8b24"
down_revision: str | None = "5e8a2c7d9f13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_SCOPE_SQL = """
CASE
    WHEN target_version_id IS NULL THEN 'document'
    ELSE 'target:' || CAST(target_version_id AS TEXT)
END
"""

NEW_SCOPE_SQL = f"""
{OLD_SCOPE_SQL}
|| ':structure:' || COALESCE(CAST(structure_version_id AS TEXT), 'none')
|| ':guideline:' || COALESCE(CAST(guideline_version_id AS TEXT), 'none')
"""


def upgrade() -> None:
    with op.batch_alter_table("annotations") as batch_op:
        batch_op.add_column(
            sa.Column("structure_version_id", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_annotations_structure_version_id_document_structure_versions",
            "document_structure_versions",
            ["structure_version_id"],
            ["id"],
        )
        batch_op.create_index(
            "ix_annotations_structure_version_id",
            ["structure_version_id"],
            unique=False,
        )

    # Evidence annotations have an unambiguous historical structure pin.
    op.execute(
        sa.text(
            """
            UPDATE annotations
            SET structure_version_id = (
                SELECT evidence_block_annotations.structure_version_id
                FROM evidence_block_annotations
                WHERE evidence_block_annotations.annotation_id = annotations.id
            )
            WHERE annotation_type = 'evidence_block'
            """
        )
    )
    # For legacy ordinary annotations, prefer the newest matching assignment
    # round and fall back to the document's active structure when no historical
    # assignment exists.
    op.execute(
        sa.text(
            """
            UPDATE annotations
            SET structure_version_id = COALESCE(
                (
                    SELECT task_assignments.structure_version_id
                    FROM task_assignments
                    JOIN project_tasks
                      ON project_tasks.id = task_assignments.task_id
                    WHERE task_assignments.project_id = annotations.project_id
                      AND task_assignments.document_id = annotations.document_id
                      AND task_assignments.assignee_user_id = annotations.annotator_user_id
                      AND project_tasks.annotation_type = annotations.annotation_type
                      AND (
                          task_assignments.guideline_version_id
                              = annotations.guideline_version_id
                          OR (
                              task_assignments.guideline_version_id IS NULL
                              AND annotations.guideline_version_id IS NULL
                          )
                      )
                    ORDER BY task_assignments.id DESC
                    LIMIT 1
                ),
                (
                    SELECT documents.active_structure_version_id
                    FROM documents
                    WHERE documents.id = annotations.document_id
                )
            )
            WHERE structure_version_id IS NULL
            """
        )
    )
    op.execute(
        sa.text(
            f"UPDATE task_assignments SET assignment_scope_key = {NEW_SCOPE_SQL}"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    duplicate = bind.execute(
        sa.text(
            f"""
            SELECT task_id, document_id, assignee_user_id, {OLD_SCOPE_SQL} AS old_scope
            FROM task_assignments
            GROUP BY task_id, document_id, assignee_user_id, {OLD_SCOPE_SQL}
            HAVING COUNT(*) > 1
            LIMIT 1
            """
        )
    ).first()
    if duplicate is not None:
        raise RuntimeError(
            "Cannot downgrade assignment scope keys after multiple versioned "
            "assignment rounds have been created"
        )
    op.execute(
        sa.text(
            f"UPDATE task_assignments SET assignment_scope_key = {OLD_SCOPE_SQL}"
        )
    )
    with op.batch_alter_table("annotations") as batch_op:
        batch_op.drop_index("ix_annotations_structure_version_id")
        batch_op.drop_constraint(
            "fk_annotations_structure_version_id_document_structure_versions",
            type_="foreignkey",
        )
        batch_op.drop_column("structure_version_id")
