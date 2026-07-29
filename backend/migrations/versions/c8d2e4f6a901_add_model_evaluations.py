"""add immutable model evaluations

Revision ID: c8d2e4f6a901
Revises: b5f9d2a7c014
Create Date: 2026-07-27 22:10:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

revision: str = "c8d2e4f6a901"
down_revision: str | None = "b5f9d2a7c014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_evaluations",
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("training_run_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("task_version_id", sa.Integer(), nullable=False),
        sa.Column("training_dataset_version_id", sa.Integer(), nullable=False),
        sa.Column("dataset_version_id", sa.Integer(), nullable=False),
        sa.Column("split_map_id", sa.Integer(), nullable=False),
        sa.Column("artifact_package_id", sa.Integer(), nullable=True),
        sa.Column("split_name", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("evaluator_key", sa.String(length=120), nullable=True),
        sa.Column("evaluator_version", sa.String(length=60), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("requested_metrics", JSONType(), nullable=False),
        sa.Column("metrics", JSONType(), nullable=False),
        sa.Column("report", JSONType(), nullable=False),
        sa.Column("evaluation_plan", JSONType(), nullable=False),
        sa.Column("runtime_digest", sa.String(length=64), nullable=False),
        sa.Column("code_digest", sa.String(length=64), nullable=False),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "row_count >= 0",
            name="ck_model_evaluations_nonnegative_row_count",
        ),
        sa.CheckConstraint(
            "status IN ('succeeded', 'unsupported', 'failed')",
            name="ck_model_evaluations_status",
        ),
        sa.ForeignKeyConstraint(["artifact_package_id"], ["artifact_packages.id"]),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.id"]),
        sa.ForeignKeyConstraint(["model_version_id"], ["model_versions.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["split_map_id"], ["split_maps.id"]),
        sa.ForeignKeyConstraint(["task_version_id"], ["task_versions.id"]),
        sa.ForeignKeyConstraint(
            ["training_dataset_version_id"],
            ["training_dataset_versions.id"],
        ),
        sa.ForeignKeyConstraint(["training_run_id"], ["training_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "artifact_package_id",
            name="uq_model_evaluations_artifact_package",
        ),
        sa.UniqueConstraint(
            "training_run_id",
            "split_name",
            name="uq_model_evaluations_run_split",
        ),
    )
    for column in (
        "artifact_package_id",
        "code_digest",
        "content_hash",
        "created_by_user_id",
        "dataset_version_id",
        "evaluator_key",
        "id",
        "model_version_id",
        "project_id",
        "runtime_digest",
        "split_map_id",
        "split_name",
        "status",
        "task_version_id",
        "training_dataset_version_id",
        "training_run_id",
    ):
        op.create_index(
            op.f(f"ix_model_evaluations_{column}"),
            "model_evaluations",
            [column],
        )


def downgrade() -> None:
    op.drop_table("model_evaluations")
