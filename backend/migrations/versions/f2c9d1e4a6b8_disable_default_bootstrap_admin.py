# pyright: reportAttributeAccessIssue=false

"""disable default bootstrap admin password

Revision ID: f2c9d1e4a6b8
Revises: c7b2d5f8a611
Create Date: 2026-07-03 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import sqlalchemy as sa
from alembic import op
from passlib.context import CryptContext
from passlib.exc import UnknownHashError

revision: str = "f2c9d1e4a6b8"
down_revision: str | None = "c7b2d5f8a611"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DEFAULT_BOOTSTRAP_ADMIN_USERNAME = "admin"
DEFAULT_BOOTSTRAP_ADMIN_PASSWORD = "change-me-now"

_pwd = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")


def _matches_default_password(password_hash: str) -> bool:
    try:
        return _pwd.verify(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, password_hash)
    except UnknownHashError:
        return False


def upgrade() -> None:
    bind = op.get_bind()
    if "users" not in sa.inspect(bind).get_table_names():
        return

    admin = bind.execute(
        sa.text(
            "SELECT id, password_hash, is_active, is_superuser "
            "FROM users WHERE username = :username"
        ),
        {"username": DEFAULT_BOOTSTRAP_ADMIN_USERNAME},
    ).first()
    if admin is None:
        return

    admin_id, password_hash, is_active, is_superuser = admin
    if not is_active or not is_superuser or not _matches_default_password(password_hash):
        return

    bind.execute(
        sa.text(
            "UPDATE users SET is_active = :is_active, password_hash = :password_hash, "
            "updated_at = :updated_at WHERE id = :id"
        ),
        {
            "id": admin_id,
            "is_active": False,
            "password_hash": _pwd.hash("!" + "x" * 32),
            "updated_at": datetime.now(UTC),
        },
    )


def downgrade() -> None:
    # Intentionally do not restore a known vulnerable credential.
    return
