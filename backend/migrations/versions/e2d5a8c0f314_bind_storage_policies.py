"""bind storage policies to executable object-store contracts

Revision ID: e2d5a8c0f314
Revises: e1c4a7b9d203
Create Date: 2026-07-27 20:10:00.000000
"""

from typing import Sequence

from alembic import op
import sqlalchemy as sa

from al_medlit.core.types import JSONType

revision: str = "e2d5a8c0f314"
down_revision: str | None = "e1c4a7b9d203"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _tables():
    projects = sa.table(
        "projects",
        sa.column("id", sa.Integer()),
        sa.column("workspace_id", sa.Integer()),
    )
    policies = sa.table(
        "storage_policies",
        sa.column("id", sa.Integer()),
        sa.column("project_id", sa.Integer()),
        sa.column("backend", sa.String(length=40)),
        sa.column("artifact_prefix", sa.String(length=512)),
        sa.column("encryption", JSONType()),
    )
    return projects, policies


def upgrade() -> None:
    projects, policies = _tables()
    connection = op.get_bind()
    rows = connection.execute(
        sa.select(
            policies.c.id,
            policies.c.backend,
            policies.c.encryption,
            projects.c.workspace_id,
        ).select_from(
            policies.join(projects, policies.c.project_id == projects.c.id)
        )
    ).mappings()
    for row in rows:
        values: dict = {
            "artifact_prefix": (
                f"workspaces/{row['workspace_id']}/artifact-blobs"
            ),
        }
        if row["backend"] == "s3":
            values["backend"] = "minio"
        if not row["encryption"]:
            values["encryption"] = {"mode": "none"}
        connection.execute(
            policies.update()
            .where(policies.c.id == row["id"])
            .values(**values)
        )
    with op.batch_alter_table("storage_policies") as batch_op:
        batch_op.create_check_constraint(
            "ck_storage_policies_executable_backend",
            "backend IN ('minio', 'local')",
        )
        batch_op.create_check_constraint(
            "ck_storage_policies_supported_retention",
            "retention_class IN ('indefinite', 'resume_14d')",
        )


def downgrade() -> None:
    _projects, policies = _tables()
    connection = op.get_bind()
    with op.batch_alter_table("storage_policies") as batch_op:
        batch_op.drop_constraint(
            "ck_storage_policies_supported_retention",
            type_="check",
        )
        batch_op.drop_constraint(
            "ck_storage_policies_executable_backend",
            type_="check",
        )
    rows = connection.execute(
        sa.select(policies.c.id, policies.c.project_id, policies.c.encryption)
    ).mappings()
    for row in rows:
        values: dict = {"artifact_prefix": f"projects/{row['project_id']}"}
        if row["encryption"] == {"mode": "none"}:
            values["encryption"] = {}
        connection.execute(
            policies.update()
            .where(policies.c.id == row["id"])
            .values(**values)
        )
