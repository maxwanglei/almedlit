# pyright: reportAttributeAccessIssue=false

"""add workspace name kind unique constraint

Revision ID: c7b2d5f8a611
Revises: b6a7c8d9e0f1
Create Date: 2026-06-18 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op


batch_alter_table: Any = getattr(op, "batch_alter_table")


revision: str = "c7b2d5f8a611"
down_revision: str | None = "b6a7c8d9e0f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MAX_WORKSPACE_NAME_LENGTH = 255


def _name_with_id_suffix(name: str, workspace_id: int, attempt: int = 0) -> str:
    discriminator = f"workspace-{workspace_id}"
    if attempt:
        discriminator = f"{discriminator}-{attempt}"
    suffix = f" ({discriminator})"
    return f"{name[: MAX_WORKSPACE_NAME_LENGTH - len(suffix)]}{suffix}"


def _disambiguate_workspace_names(bind) -> None:
    """Make every legacy ``(name, kind)`` pair deterministic and unique.

    IDs, unlike creator usernames, also distinguish repeated rows owned by the
    same user and system rows whose ``created_by`` value is NULL.
    """

    rows = bind.execute(sa.text("SELECT id, name, kind FROM workspaces ORDER BY id")).mappings()
    reserved: set[tuple[str, str]] = set()
    for row in rows:
        workspace_id = int(row["id"])
        kind = str(row["kind"])
        original_name = str(row["name"])
        candidate = original_name[:MAX_WORKSPACE_NAME_LENGTH]
        attempt = 0
        while (candidate, kind) in reserved:
            candidate = _name_with_id_suffix(original_name, workspace_id, attempt)
            attempt += 1
        reserved.add((candidate, kind))
        if candidate != original_name:
            bind.execute(
                sa.text("UPDATE workspaces SET name = :name WHERE id = :workspace_id"),
                {"name": candidate, "workspace_id": workspace_id},
            )


def upgrade() -> None:
    _disambiguate_workspace_names(op.get_bind())
    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.create_unique_constraint(
            "uq_workspaces_name_kind",
            ["name", "kind"],
        )


def downgrade() -> None:
    with batch_alter_table("workspaces", schema=None) as batch_op:
        batch_op.drop_constraint("uq_workspaces_name_kind", type_="unique")
