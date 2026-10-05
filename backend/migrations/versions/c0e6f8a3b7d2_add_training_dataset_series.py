"""Add reusable source purposes and named, immutable training dataset series.

Revision ID: c0e6f8a3b7d2
Revises: d9a3c5e1f742
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "c0e6f8a3b7d2"
down_revision: str | None = "d9a3c5e1f742"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    json_type = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
    with op.batch_alter_table("datasets") as batch:
        batch.add_column(sa.Column("purposes", json_type, nullable=False, server_default="[]"))
    op.create_table(
        "training_datasets",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("project_id", sa.Integer(), sa.ForeignKey("projects.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column(
            "task_version_id", sa.Integer(), sa.ForeignKey("task_versions.id"), nullable=False
        ),
        sa.Column("created_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("project_id", "name", name="uq_training_datasets_project_name"),
    )
    for column in ("id", "project_id", "task_version_id", "created_by_user_id"):
        op.create_index(f"ix_training_datasets_{column}", "training_datasets", [column])
    with op.batch_alter_table("training_dataset_versions") as batch:
        batch.add_column(sa.Column("training_dataset_id", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("version_number", sa.Integer(), nullable=False, server_default="1")
        )
        batch.add_column(sa.Column("parent_version_id", sa.Integer(), nullable=True))
        batch.add_column(
            sa.Column("preparation_manifest", json_type, nullable=False, server_default="{}")
        )
        batch.add_column(sa.Column("idempotency_key", sa.String(160), nullable=True))
        batch.add_column(sa.Column("request_hash", sa.String(64), nullable=True))
        batch.create_foreign_key(
            "fk_training_versions_series", "training_datasets", ["training_dataset_id"], ["id"]
        )
        batch.create_foreign_key(
            "fk_training_versions_parent",
            "training_dataset_versions",
            ["parent_version_id"],
            ["id"],
        )
        batch.create_index(
            "ix_training_dataset_versions_training_dataset_id", ["training_dataset_id"]
        )
        batch.create_index("ix_training_dataset_versions_parent_version_id", ["parent_version_id"])
        batch.create_unique_constraint(
            "uq_training_versions_series_number", ["training_dataset_id", "version_number"]
        )
        batch.create_unique_constraint(
            "uq_training_versions_preparation_key", ["project_id", "idempotency_key"]
        )
        batch.create_check_constraint("ck_training_versions_positive_number", "version_number > 0")

    bind = op.get_bind()
    metadata = sa.MetaData()
    versions = sa.Table("training_dataset_versions", metadata, autoload_with=bind)
    series = sa.Table("training_datasets", metadata, autoload_with=bind)
    reserved: set[tuple[int, str]] = set()
    # Each historical row is an independent v1. Do not invent provenance or
    # mutate historical IDs, names, content hashes, or training-run references.
    for row in bind.execute(sa.select(versions).order_by(versions.c.id)).mappings():
        name = row["name"]
        candidate = name
        attempt = 0
        while (row["project_id"], candidate) in reserved:
            suffix = f" (legacy-{row['id']}{'-' + str(attempt) if attempt else ''})"
            candidate = f"{name[: 255 - len(suffix)]}{suffix}"
            attempt += 1
        reserved.add((row["project_id"], candidate))
        inserted = bind.execute(
            series.insert().values(
                project_id=row["project_id"],
                name=candidate,
                task_version_id=row["task_version_id"],
                created_by_user_id=row["created_by_user_id"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
        )
        bind.execute(
            versions.update()
            .where(versions.c.id == row["id"])
            .values(
                training_dataset_id=inserted.inserted_primary_key[0],
                version_number=1,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("training_dataset_versions") as batch:
        batch.drop_constraint("uq_training_versions_series_number", type_="unique")
        batch.drop_constraint("uq_training_versions_preparation_key", type_="unique")
        batch.drop_constraint("ck_training_versions_positive_number", type_="check")
        batch.drop_constraint("fk_training_versions_series", type_="foreignkey")
        batch.drop_constraint("fk_training_versions_parent", type_="foreignkey")
        batch.drop_index("ix_training_dataset_versions_training_dataset_id")
        batch.drop_index("ix_training_dataset_versions_parent_version_id")
        for column in (
            "training_dataset_id",
            "version_number",
            "parent_version_id",
            "preparation_manifest",
            "idempotency_key",
            "request_hash",
        ):
            batch.drop_column(column)
    op.drop_table("training_datasets")
    with op.batch_alter_table("datasets") as batch:
        batch.drop_column("purposes")
