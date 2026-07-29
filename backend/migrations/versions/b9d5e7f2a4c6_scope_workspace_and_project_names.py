# pyright: reportAttributeAccessIssue=false

"""scope workspace and project display-name uniqueness

Revision ID: b9d5e7f2a4c6
Revises: a8c4e7d1f2b6
Create Date: 2026-07-17 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b9d5e7f2a4c6"
down_revision: str | None = "a8c4e7d1f2b6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MAX_NAME_LENGTH = 255


def _suffixed_name(name: str, entity: str, row_id: int, attempt: int = 0) -> str:
    discriminator = f"{entity}-{row_id}"
    if attempt:
        discriminator = f"{discriminator}-{attempt}"
    suffix = f" ({discriminator})"
    return f"{name[: MAX_NAME_LENGTH - len(suffix)]}{suffix}"


def _disambiguate_for_global_downgrade(
    bind,
    *,
    table_name: str,
    entity: str,
    include_kind: bool,
) -> None:
    columns = "id, name, kind" if include_kind else "id, name"
    rows = bind.execute(sa.text(f"SELECT {columns} FROM {table_name} ORDER BY id")).mappings()
    reserved: set[tuple[str, str | None]] = set()
    for row in rows:
        row_id = int(row["id"])
        kind = str(row["kind"]) if include_kind else None
        original_name = str(row["name"])
        candidate = original_name[:MAX_NAME_LENGTH]
        attempt = 0
        while (candidate, kind) in reserved:
            candidate = _suffixed_name(original_name, entity, row_id, attempt)
            attempt += 1
        reserved.add((candidate, kind))
        if candidate != original_name:
            bind.execute(
                sa.text(f"UPDATE {table_name} SET name = :name WHERE id = :row_id"),
                {"name": candidate, "row_id": row_id},
            )


def upgrade() -> None:
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_constraint("uq_workspaces_name_kind", type_="unique")
        batch_op.create_unique_constraint(
            "uq_workspaces_creator_name_kind",
            ["created_by", "name", "kind"],
        )
    op.create_index(
        "uq_workspaces_system_name_kind",
        "workspaces",
        ["name", "kind"],
        unique=True,
        postgresql_where=sa.text("created_by IS NULL"),
        sqlite_where=sa.text("created_by IS NULL"),
    )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_index("ix_projects_name")
        batch_op.create_index("ix_projects_name", ["name"], unique=False)
        batch_op.create_unique_constraint(
            "uq_projects_workspace_name",
            ["workspace_id", "name"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    # Rows created after this migration may legitimately share display names
    # across tenants. Remove the scoped constraints before renaming so a
    # generated global fallback cannot temporarily collide with an unprocessed
    # row in the same tenant.
    op.drop_index("uq_workspaces_system_name_kind", table_name="workspaces")
    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_constraint(
            "uq_workspaces_creator_name_kind",
            type_="unique",
        )
    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_constraint("uq_projects_workspace_name", type_="unique")

    # Preserve every row while restoring the old global keys.
    _disambiguate_for_global_downgrade(
        bind,
        table_name="workspaces",
        entity="workspace",
        include_kind=True,
    )
    _disambiguate_for_global_downgrade(
        bind,
        table_name="projects",
        entity="project",
        include_kind=False,
    )

    with op.batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_workspaces_name_kind",
            ["name", "kind"],
        )

    with op.batch_alter_table("projects", schema=None) as batch_op:
        batch_op.drop_index("ix_projects_name")
        batch_op.create_index("ix_projects_name", ["name"], unique=True)
