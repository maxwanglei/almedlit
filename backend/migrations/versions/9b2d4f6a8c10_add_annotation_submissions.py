# pyright: reportAttributeAccessIssue=false

"""add annotation submissions

Revision ID: 9b2d4f6a8c10
Revises: 6c1a8f0e9d72
Create Date: 2026-06-11 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from al_medlit.core.types import JSONType

revision: str = "9b2d4f6a8c10"
down_revision: str | None = "6c1a8f0e9d72"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "annotation_submissions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=False),
        sa.Column("annotator_id", sa.String(length=120), nullable=True),
        sa.Column(
            "kind",
            sa.String(length=20),
            nullable=False,
            server_default="submission",
        ),
        sa.Column("storage_key", sa.String(length=512), nullable=False, unique=True),
        sa.Column("file_name", sa.String(length=255), nullable=False),
        sa.Column(
            "content_type",
            sa.String(length=100),
            nullable=False,
            server_default="application/json",
        ),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("annotation_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("metadata", JSONType(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_annotation_submissions_id", "annotation_submissions", ["id"])
    op.create_index(
        "ix_annotation_submissions_project_id",
        "annotation_submissions",
        ["project_id"],
    )
    op.create_index(
        "ix_annotation_submissions_document_id",
        "annotation_submissions",
        ["document_id"],
    )
    op.create_index(
        "ix_annotation_submissions_annotator_id",
        "annotation_submissions",
        ["annotator_id"],
    )
    op.create_index("ix_annotation_submissions_kind", "annotation_submissions", ["kind"])


def downgrade() -> None:
    op.drop_table("annotation_submissions")
