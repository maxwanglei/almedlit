# pyright: reportAttributeAccessIssue=false

"""backfill users from annotator ids

Revision ID: e5f8a1b2c3d4
Revises: d4e7f0a1b2c3
Create Date: 2026-06-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from passlib.context import CryptContext

revision: str = "e5f8a1b2c3d4"
down_revision: str | None = "d4e7f0a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _table_has_column(bind, table_name: str, column_name: str) -> bool:
    inspector = sa.inspect(bind)
    if table_name not in inspector.get_table_names():
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    now = datetime.now(UTC)

    names: set[str] = set()
    for table, column in [
        ("task_assignments", "annotator_id"),
        ("task_assignments", "assigned_by"),
        ("annotations", "annotator_id"),
        ("annotation_submissions", "annotator_id"),
    ]:
        if not _table_has_column(bind, table, column):
            continue
        rows = bind.execute(
            sa.text(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL")
        ).fetchall()
        names.update(str(row[0]).strip() for row in rows if row[0] and str(row[0]).strip())

    users = sa.table(
        "users",
        sa.column("id", sa.Integer),
        sa.column("username", sa.String),
        sa.column("email", sa.String),
        sa.column("password_hash", sa.String),
        sa.column("display_name", sa.String),
        sa.column("is_active", sa.Boolean),
        sa.column("is_superuser", sa.Boolean),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    def ensure_user(
        username: str,
        *,
        active: bool,
        superuser: bool,
        password_hash: str,
    ) -> int:
        existing = bind.execute(
            sa.text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        ).first()
        if existing is not None:
            if active or superuser:
                bind.execute(
                    sa.text(
                        "UPDATE users SET is_active = :active, "
                        "is_superuser = :superuser, password_hash = :password_hash, "
                        "updated_at = :updated_at WHERE id = :id"
                    ),
                    {
                        "active": active,
                        "superuser": superuser,
                        "password_hash": password_hash,
                        "updated_at": now,
                        "id": existing[0],
                    },
                )
            return existing[0]

        result = bind.execute(
            users.insert().values(
                username=username,
                email=None,
                password_hash=password_hash,
                display_name=username,
                is_active=active,
                is_superuser=superuser,
                created_at=now,
                updated_at=now,
            )
        )
        inserted_pk = tuple(result.inserted_primary_key or ())
        if inserted_pk:
            return inserted_pk[0]
        return bind.execute(
            sa.text("SELECT id FROM users WHERE username = :username"),
            {"username": username},
        ).scalar_one()

    unusable_hash = _pwd.hash("!" + "x" * 16)
    user_ids = {
        name: ensure_user(
            name,
            active=False,
            superuser=False,
            password_hash=unusable_hash,
        )
        for name in sorted(names)
    }

    default_ws = bind.execute(
        sa.text(
            "SELECT id FROM workspaces WHERE name = 'Default' "
            "AND kind = 'team' ORDER BY id LIMIT 1"
        )
    ).first()
    if default_ws is None:
        workspaces = sa.table(
            "workspaces",
            sa.column("id", sa.Integer),
            sa.column("name", sa.String),
            sa.column("kind", sa.String),
            sa.column("join_code", sa.String),
            sa.column("created_by", sa.Integer),
            sa.column("capability_preset", sa.String),
            sa.column("capability_overrides", sa.JSON),
            sa.column("created_at", sa.DateTime(timezone=True)),
            sa.column("updated_at", sa.DateTime(timezone=True)),
        )
        ws_result = bind.execute(
            workspaces.insert().values(
                name="Default",
                kind="team",
                join_code=None,
                created_by=None,
                capability_preset="full",
                capability_overrides=[],
                created_at=now,
                updated_at=now,
            )
        )
        inserted_ws_pk = tuple(ws_result.inserted_primary_key or ())
        if inserted_ws_pk:
            default_ws_id = inserted_ws_pk[0]
        else:
            default_ws_id = bind.execute(
                sa.text(
                    "SELECT id FROM workspaces WHERE name = 'Default' "
                    "AND kind = 'team' ORDER BY id LIMIT 1"
                )
            ).scalar_one()
    else:
        default_ws_id = default_ws[0]
        bind.execute(
            sa.text(
                "UPDATE workspaces SET capability_preset = 'full', "
                "updated_at = :updated_at WHERE id = :id"
            ),
            {"updated_at": now, "id": default_ws_id},
        )

    members = sa.table(
        "workspace_members",
        sa.column("workspace_id", sa.Integer),
        sa.column("user_id", sa.Integer),
        sa.column("role", sa.String),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    def ensure_member(user_id: int, role: str) -> None:
        existing = bind.execute(
            sa.text(
                "SELECT id FROM workspace_members "
                "WHERE workspace_id = :workspace_id AND user_id = :user_id"
            ),
            {"workspace_id": default_ws_id, "user_id": user_id},
        ).first()
        if existing is not None:
            if role == "admin":
                bind.execute(
                    sa.text(
                        "UPDATE workspace_members SET role = 'admin', updated_at = :updated_at "
                        "WHERE id = :id"
                    ),
                    {"updated_at": now, "id": existing[0]},
                )
            return
        bind.execute(
            members.insert().values(
                workspace_id=default_ws_id,
                user_id=user_id,
                role=role,
                created_at=now,
                updated_at=now,
            )
        )

    for user_id in user_ids.values():
        ensure_member(user_id, "annotator")


def downgrade() -> None:
    bind = op.get_bind()
    default_ws = bind.execute(
        sa.text(
            "SELECT id FROM workspaces WHERE name = 'Default' "
            "AND kind = 'team' ORDER BY id LIMIT 1"
        )
    ).first()
    if default_ws is not None:
        bind.execute(
            sa.text("DELETE FROM workspace_members WHERE workspace_id = :workspace_id"),
            {"workspace_id": default_ws[0]},
        )
