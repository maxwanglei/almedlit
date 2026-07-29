# pyright: reportAttributeAccessIssue=false

"""pin prediction reviews to assignment and guideline rounds

Revision ID: a8c4e7d1f2b6
Revises: 9d3f6a1c8b24
Create Date: 2026-07-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a8c4e7d1f2b6"
down_revision: str | None = "9d3f6a1c8b24"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

ASSIGNMENT_FK_NAME = "fk_evidence_prediction_reviews_assignment_id_task_assignments"
GUIDELINE_VERSION_FK_NAME = "fk_evidence_prediction_reviews_guideline_version_id"


def upgrade() -> None:
    with op.batch_alter_table("evidence_prediction_reviews") as batch_op:
        batch_op.add_column(sa.Column("assignment_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("guideline_version_id", sa.Integer(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_evidence_prediction_reviews_assignment_id"),
            ["assignment_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_evidence_prediction_reviews_guideline_version_id"),
            ["guideline_version_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            ASSIGNMENT_FK_NAME,
            "task_assignments",
            ["assignment_id"],
            ["id"],
        )
        batch_op.create_foreign_key(
            GUIDELINE_VERSION_FK_NAME,
            "guideline_versions",
            ["guideline_version_id"],
            ["id"],
        )

    # Accepted/modified reviews already pin their guideline on the resulting
    # annotation. Recover that unambiguous provenance first.
    op.execute(
        sa.text(
            """
            UPDATE evidence_prediction_reviews
            SET guideline_version_id = (
                SELECT annotations.guideline_version_id
                FROM annotations
                WHERE annotations.id
                    = evidence_prediction_reviews.resulting_annotation_id
            )
            WHERE resulting_annotation_id IS NOT NULL
            """
        )
    )
    # Legacy reject-only reviews have no annotation. Pin a historical review
    # only when exactly one assignment matches its reviewer and immutable
    # prediction scope; ambiguous legacy rows deliberately remain nullable.
    op.execute(
        sa.text(
            """
            UPDATE evidence_prediction_reviews
            SET assignment_id = (
                SELECT CASE
                    WHEN COUNT(*) = 1 THEN MIN(task_assignments.id)
                    ELSE NULL
                END
                FROM task_assignments
                JOIN evidence_candidate_predictions
                  ON evidence_candidate_predictions.id
                    = evidence_prediction_reviews.prediction_id
                JOIN project_tasks
                  ON project_tasks.id = task_assignments.task_id
                WHERE task_assignments.project_id
                        = evidence_candidate_predictions.project_id
                  AND task_assignments.document_id
                        = evidence_candidate_predictions.document_id
                  AND task_assignments.assignee_user_id
                        = evidence_prediction_reviews.reviewer_user_id
                  AND task_assignments.target_version_id
                        = evidence_candidate_predictions.target_version_id
                  AND task_assignments.structure_version_id
                        = evidence_candidate_predictions.structure_version_id
                  AND project_tasks.annotation_type = 'evidence_block'
                  AND (
                      evidence_prediction_reviews.guideline_version_id IS NULL
                      OR task_assignments.guideline_version_id
                          = evidence_prediction_reviews.guideline_version_id
                  )
            )
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE evidence_prediction_reviews
            SET guideline_version_id = (
                SELECT task_assignments.guideline_version_id
                FROM task_assignments
                WHERE task_assignments.id
                    = evidence_prediction_reviews.assignment_id
            )
            WHERE guideline_version_id IS NULL
              AND assignment_id IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("evidence_prediction_reviews") as batch_op:
        batch_op.drop_constraint(
            GUIDELINE_VERSION_FK_NAME,
            type_="foreignkey",
        )
        batch_op.drop_constraint(
            ASSIGNMENT_FK_NAME,
            type_="foreignkey",
        )
        batch_op.drop_index(
            batch_op.f("ix_evidence_prediction_reviews_guideline_version_id")
        )
        batch_op.drop_index(
            batch_op.f("ix_evidence_prediction_reviews_assignment_id")
        )
        batch_op.drop_column("guideline_version_id")
        batch_op.drop_column("assignment_id")
