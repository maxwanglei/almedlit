import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError

PREVIOUS_REVISION = "6f2a9d4c8b31"
ADMINISTRATION_REVISION = "71b9d2e4f6a8"


def _run_alembic(
    direction: str,
    revision: str,
    *,
    database_url: str,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["AL_MEDLIT_DATABASE_URL"] = database_url
    return subprocess.run(
        [sys.executable, "-m", "alembic", direction, revision],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_administration_migration_preserves_users_and_invites_and_downgrades(tmp_path):
    database_path = tmp_path / "administration-migration.db"
    database_url = f"sqlite:///{database_path}"
    initial = _run_alembic("upgrade", PREVIOUS_REVISION, database_url=database_url)
    assert initial.returncode == 0, initial.stderr

    engine = create_engine(database_url)
    now = datetime.now(UTC)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users "
                "(id, username, password_hash, display_name, is_active, is_superuser, "
                "created_at, updated_at, email) "
                "VALUES (1, 'preserved-admin', 'hash', 'Admin', 1, 1, :now, :now, "
                "'admin@example.test')"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO workspaces "
                "(id, name, kind, created_by, capability_preset, capability_overrides, "
                "created_at, updated_at) "
                "VALUES (99, 'Preserved Team', 'team', 1, 'annotate', '[]', :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO workspace_members "
                "(id, workspace_id, user_id, role, created_at, updated_at) "
                "VALUES (99, 99, 1, 'admin', :now, :now)"
            ),
            {"now": now},
        )
        connection.execute(
            text(
                "INSERT INTO workspace_invites "
                "(id, workspace_id, token, role, created_by, created_at, updated_at) "
                "VALUES (99, 99, 'preserved-token', 'trainer', 1, :now, :now)"
            ),
            {"now": now},
        )

    upgraded = _run_alembic(
        "upgrade",
        ADMINISTRATION_REVISION,
        database_url=database_url,
    )
    assert upgraded.returncode == 0, upgraded.stderr
    inspection = inspect(engine)
    assert {
        "instance_policies",
        "account_action_tokens",
        "admin_audit_events",
    }.issubset(inspection.get_table_names())
    assert {"session_version", "last_login_at"}.issubset(
        {column["name"] for column in inspection.get_columns("users")}
    )
    assert {"revoked_at", "revoked_by"}.issubset(
        {column["name"] for column in inspection.get_columns("workspace_invites")}
    )
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT username, session_version FROM users WHERE id = 1")
        ).one() == ("preserved-admin", 0)
        assert connection.execute(
            text("SELECT role, revoked_at, revoked_by FROM workspace_invites WHERE id = 99")
        ).one() == ("trainer", None, None)
        assert connection.execute(
            text("SELECT workspace_id, user_id, role FROM workspace_members WHERE id = 99")
        ).one() == (99, 1, "admin")
        assert connection.execute(
            text(
                "SELECT allow_self_registration, default_invite_expiry_minutes, "
                "account_action_expiry_minutes FROM instance_policies WHERE id = 1"
            )
        ).one() == (None, 10_080, 60)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO admin_audit_events "
                "(id, event_type, details, created_at) "
                "VALUES (1, 'migration.test', '{}', :now)"
            ),
            {"now": now},
        )
    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE admin_audit_events SET event_type = 'tampered' WHERE id = 1"
                )
            )
    with pytest.raises(DBAPIError, match="append-only"):
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM admin_audit_events WHERE id = 1"))
    with pytest.raises(DBAPIError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE instance_policies "
                    "SET default_invite_expiry_minutes = 59 WHERE id = 1"
                )
            )
    with pytest.raises(DBAPIError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE instance_policies "
                    "SET account_action_expiry_minutes = 1441 WHERE id = 1"
                )
            )

    downgraded = _run_alembic(
        "downgrade",
        PREVIOUS_REVISION,
        database_url=database_url,
    )
    assert downgraded.returncode == 0, downgraded.stderr
    inspection = inspect(engine)
    assert "instance_policies" not in inspection.get_table_names()
    assert "account_action_tokens" not in inspection.get_table_names()
    assert "admin_audit_events" not in inspection.get_table_names()
    assert "session_version" not in {
        column["name"] for column in inspection.get_columns("users")
    }
    assert "revoked_at" not in {
        column["name"] for column in inspection.get_columns("workspace_invites")
    }
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT username FROM users WHERE id = 1")
        ).scalar_one() == "preserved-admin"
        assert connection.execute(
            text("SELECT token FROM workspace_invites WHERE id = 99")
        ).scalar_one() == "preserved-token"
        assert connection.execute(
            text("SELECT workspace_id, user_id, role FROM workspace_members WHERE id = 99")
        ).one() == (99, 1, "admin")
    engine.dispose()
