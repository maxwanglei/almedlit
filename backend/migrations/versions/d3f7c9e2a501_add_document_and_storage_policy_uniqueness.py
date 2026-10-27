"""Enforce document identity and one default storage policy per project."""

import sqlalchemy as sa
from alembic import op

revision = "d3f7c9e2a501"
down_revision = "c9e4a7b12d68"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    for query, description in (
        (
            "SELECT project_id, external_id FROM documents WHERE external_id IS NOT NULL "
            "GROUP BY project_id, external_id HAVING COUNT(*) > 1",
            "duplicate document external IDs",
        ),
        (
            "SELECT project_id FROM storage_policies WHERE is_default = true "
            "GROUP BY project_id HAVING COUNT(*) > 1",
            "multiple default storage policies",
        ),
    ):
        if bind.execute(sa.text(query)).first() is not None:
            raise RuntimeError(f"Resolve {description} before applying this migration")
    with op.batch_alter_table("documents") as batch:
        batch.create_unique_constraint(
            "uq_documents_project_external_id", ["project_id", "external_id"]
        )
    op.create_index(
        "uq_storage_policies_project_default", "storage_policies", ["project_id"],
        unique=True, postgresql_where=sa.text("is_default = true"),
        sqlite_where=sa.text("is_default = 1"),
    )
    with op.batch_alter_table("model_checkpoints") as batch:
        batch.alter_column("training_job_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    if op.get_bind().execute(sa.text(
        "SELECT id FROM model_checkpoints WHERE training_job_id IS NULL LIMIT 1"
    )).first() is not None:
        raise RuntimeError("Remove imported checkpoints before reverting their nullable job link")
    with op.batch_alter_table("model_checkpoints") as batch:
        batch.alter_column("training_job_id", existing_type=sa.Integer(), nullable=False)
    op.drop_index("uq_storage_policies_project_default", table_name="storage_policies")
    with op.batch_alter_table("documents") as batch:
        batch.drop_constraint("uq_documents_project_external_id", type_="unique")
