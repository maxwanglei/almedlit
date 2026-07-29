"""add workspace artifact quotas and reservations

Revision ID: f3c6a9d2e5b8
Revises: d1a4f7c9e2b6
Create Date: 2026-07-27 18:05:00.000000
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "f3c6a9d2e5b8"
down_revision: str | None = "f8c5d3e0b217"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workspace_artifact_quotas",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("limit_bytes", sa.BigInteger(), nullable=True),
        sa.Column("used_bytes", sa.BigInteger(), nullable=False),
        sa.Column("reserved_bytes", sa.BigInteger(), nullable=False),
        sa.Column("reservation_ttl_seconds", sa.Integer(), nullable=False),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "limit_bytes IS NULL OR limit_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_limit",
        ),
        sa.CheckConstraint(
            "reserved_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_reserved",
        ),
        sa.CheckConstraint(
            "reservation_ttl_seconds > 0",
            name="ck_workspace_artifact_quotas_positive_ttl",
        ),
        sa.CheckConstraint(
            "used_bytes >= 0",
            name="ck_workspace_artifact_quotas_nonnegative_used",
        ),
        sa.ForeignKeyConstraint(["updated_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "workspace_id",
            name="uq_workspace_artifact_quotas_workspace",
        ),
    )
    with op.batch_alter_table("workspace_artifact_quotas") as batch_op:
        batch_op.create_index(
            batch_op.f("ix_workspace_artifact_quotas_id"),
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_workspace_artifact_quotas_updated_by_user_id"),
            ["updated_by_user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_workspace_artifact_quotas_workspace_id"),
            ["workspace_id"],
        )

    op.create_table(
        "artifact_storage_reservations",
        sa.Column("workspace_id", sa.Integer(), nullable=False),
        sa.Column("project_id", sa.Integer(), nullable=False),
        sa.Column("owner_type", sa.String(length=40), nullable=False),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=False),
        sa.Column("reserved_bytes", sa.BigInteger(), nullable=False),
        sa.Column("committed_bytes", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=30), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("committed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("release_reason", sa.String(length=500), nullable=True),
        sa.Column("artifact_package_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "committed_bytes >= 0",
            name="ck_artifact_storage_reservations_nonnegative_committed",
        ),
        sa.CheckConstraint(
            "reserved_bytes > 0",
            name="ck_artifact_storage_reservations_positive_reserved",
        ),
        sa.ForeignKeyConstraint(
            ["artifact_package_id"],
            ["artifact_packages.id"],
        ),
        sa.ForeignKeyConstraint(["created_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "owner_type",
            "owner_id",
            name="uq_artifact_storage_reservations_owner",
        ),
        sa.UniqueConstraint(
            "workspace_id",
            "idempotency_key",
            name="uq_artifact_storage_reservations_workspace_key",
        ),
    )
    with op.batch_alter_table("artifact_storage_reservations") as batch_op:
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_artifact_package_id"),
            ["artifact_package_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_created_by_user_id"),
            ["created_by_user_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_expires_at"),
            ["expires_at"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_id"),
            ["id"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_owner_id"),
            ["owner_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_owner_type"),
            ["owner_type"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_project_id"),
            ["project_id"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_status"),
            ["status"],
        )
        batch_op.create_index(
            batch_op.f("ix_artifact_storage_reservations_workspace_id"),
            ["workspace_id"],
        )

    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("artifact_reservation_id", sa.Integer(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_training_jobs_artifact_reservation_id"),
            ["artifact_reservation_id"],
            unique=True,
        )
        batch_op.create_foreign_key(
            "fk_training_jobs_artifact_reservation_id",
            "artifact_storage_reservations",
            ["artifact_reservation_id"],
            ["id"],
        )

    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.add_column(
            sa.Column("artifact_reservation_id", sa.Integer(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_training_runs_artifact_reservation_id"),
            ["artifact_reservation_id"],
            unique=True,
        )
        batch_op.create_foreign_key(
            "fk_training_runs_artifact_reservation_id",
            "artifact_storage_reservations",
            ["artifact_reservation_id"],
            ["id"],
        )

    op.execute(
        sa.text(
            """
            INSERT INTO workspace_artifact_quotas (
                workspace_id,
                limit_bytes,
                used_bytes,
                reserved_bytes,
                reservation_ttl_seconds,
                updated_by_user_id,
                created_at,
                updated_at
            )
            SELECT
                workspaces.id,
                NULL,
                COALESCE(SUM(
                    CASE
                        WHEN artifact_blobs.status = 'ready'
                        THEN artifact_blobs.size_bytes
                        ELSE 0
                    END
                ), 0),
                0,
                604800,
                NULL,
                CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
            FROM workspaces
            LEFT JOIN artifact_blobs
                ON artifact_blobs.workspace_id = workspaces.id
            GROUP BY workspaces.id
            """
        )
    )


def downgrade() -> None:
    with op.batch_alter_table("training_runs") as batch_op:
        batch_op.drop_constraint(
            "fk_training_runs_artifact_reservation_id",
            type_="foreignkey",
        )
        batch_op.drop_index(
            batch_op.f("ix_training_runs_artifact_reservation_id")
        )
        batch_op.drop_column("artifact_reservation_id")

    with op.batch_alter_table("training_jobs") as batch_op:
        batch_op.drop_constraint(
            "fk_training_jobs_artifact_reservation_id",
            type_="foreignkey",
        )
        batch_op.drop_index(batch_op.f("ix_training_jobs_artifact_reservation_id"))
        batch_op.drop_column("artifact_reservation_id")

    op.drop_table("artifact_storage_reservations")
    op.drop_table("workspace_artifact_quotas")
