import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from passlib.context import CryptContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from al_medlit.core.config import DEFAULT_BOOTSTRAP_ADMIN_PASSWORD
from al_medlit.core.database import ensure_schema_ready
from al_medlit.project.models import Project
from scripts.seed_demo import DEMO_PROJECT_NAME

_pwd = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sqlite_url(path: Path) -> str:
    return f"sqlite:///{path.resolve().as_posix()}"


def _run_backend_command(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=_backend_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_postgres(url: str, timeout_seconds: int = 30) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            engine.dispose()
            return
        except SQLAlchemyError:
            time.sleep(1)
    raise AssertionError("Timed out waiting for PostgreSQL to accept connections")


def test_ensure_schema_ready_fails_on_unmigrated_database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        with pytest.raises(RuntimeError, match="alembic upgrade head"):
            ensure_schema_ready(bind=engine)
    finally:
        engine.dispose()


def test_seed_demo_requires_migrated_schema(tmp_path):
    database_url = _sqlite_url(tmp_path / "unmigrated.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}

    result = _run_backend_command("-m", "scripts.seed_demo", env=env)

    assert result.returncode != 0
    assert "Database schema is not initialized" in result.stderr
    assert "alembic upgrade head" in result.stderr


def test_migration_then_seed_bootstrap_sqlite(tmp_path):
    database_url = _sqlite_url(tmp_path / "migrated.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}

    migrate_result = _run_backend_command("-m", "alembic", "upgrade", "head", env=env)
    assert migrate_result.returncode == 0, migrate_result.stderr

    schema_check = _run_backend_command("-m", "alembic", "check", env=env)
    assert schema_check.returncode == 0, schema_check.stderr

    seed_result = _run_backend_command("-m", "scripts.seed_demo", env=env)
    assert seed_result.returncode == 0, seed_result.stderr
    assert "Seeded demo project" in seed_result.stdout

    repeat_seed_result = _run_backend_command("-m", "scripts.seed_demo", env=env)
    assert repeat_seed_result.returncode == 0, repeat_seed_result.stderr
    assert "Demo project already seeded; nothing to do." in repeat_seed_result.stdout

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            admin = connection.execute(
                text(
                    "SELECT password_hash, is_active, is_superuser "
                    "FROM users WHERE username = 'admin'"
                )
            ).first()
            if admin is not None:
                assert not (
                    admin[1]
                    and admin[2]
                    and _pwd.verify(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, admin[0])
                )
            project_count = connection.execute(text("SELECT COUNT(*) FROM projects")).scalar_one()
            correction_count = connection.execute(
                text("SELECT COUNT(*) FROM annotation_corrections")
            ).scalar_one()
            training_action_count = connection.execute(
                text("SELECT COUNT(*) FROM training_actions")
            ).scalar_one()
            micro_question_count = connection.execute(
                text("SELECT COUNT(*) FROM micro_question_templates")
            ).scalar_one()
        assert project_count == 1
        assert correction_count == 1
        assert training_action_count == 1
        assert micro_question_count == 1
    finally:
        engine.dispose()


def test_user_management_migration_normalizes_legacy_display_names(tmp_path):
    database_url = _sqlite_url(tmp_path / "legacy-display-names.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}
    pre_migration = _run_backend_command(
        "-m",
        "alembic",
        "upgrade",
        "a3f6b8c1d2e4",
        env=env,
    )
    assert pre_migration.returncode == 0, pre_migration.stderr

    engine = create_engine(database_url)
    try:
        now = "2026-06-15 00:00:00"
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(username, password_hash, display_name, is_active, is_superuser, "
                    "created_at, updated_at) VALUES "
                    "(:username, 'x', :display_name, 1, 0, :now, :now)"
                ),
                [
                    {
                        "username": "long-display-name",
                        "display_name": "L" * 180,
                        "now": now,
                    },
                    {
                        "username": "null-display-name",
                        "display_name": None,
                        "now": now,
                    },
                ],
            )
    finally:
        engine.dispose()

    migrated = _run_backend_command(
        "-m",
        "alembic",
        "upgrade",
        "d4e7f0a1b2c3",
        env=env,
    )
    assert migrated.returncode == 0, migrated.stderr

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            rows = dict(
                connection.execute(
                    text("SELECT username, display_name FROM users ORDER BY username")
                ).all()
            )
        assert rows["long-display-name"] == "L" * 120
        assert rows["null-display-name"] == ""
    finally:
        engine.dispose()


def test_workspace_name_migration_disambiguates_all_legacy_duplicates(tmp_path):
    database_url = _sqlite_url(tmp_path / "duplicate-workspaces.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}
    pre_migration = _run_backend_command(
        "-m",
        "alembic",
        "upgrade",
        "b6a7c8d9e0f1",
        env=env,
    )
    assert pre_migration.returncode == 0, pre_migration.stderr

    engine = create_engine(database_url)
    try:
        now = "2026-06-18 00:00:00"
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(username, email, password_hash, display_name, is_active, is_superuser, "
                    "created_at, updated_at) VALUES "
                    "('duplicate-owner', NULL, 'x', 'Owner', 1, 0, :now, :now)"
                ),
                {"now": now},
            )
            owner_id = connection.execute(
                text("SELECT id FROM users WHERE username = 'duplicate-owner'")
            ).scalar_one()
            workspace_rows = [
                {
                    "name": "X" * 255,
                    "join_code": f"owner-{index}",
                    "created_by": owner_id,
                    "now": now,
                }
                for index in range(3)
            ] + [
                {
                    "name": "System Duplicate",
                    "join_code": f"system-{index}",
                    "created_by": None,
                    "now": now,
                }
                for index in range(3)
            ]
            connection.execute(
                text(
                    "INSERT INTO workspaces "
                    "(name, kind, join_code, created_by, capability_preset, "
                    "capability_overrides, created_at, updated_at) VALUES "
                    "(:name, 'team', :join_code, :created_by, 'annotate', '[]', :now, :now)"
                ),
                workspace_rows,
            )
    finally:
        engine.dispose()

    migrated = _run_backend_command(
        "-m",
        "alembic",
        "upgrade",
        "c7b2d5f8a611",
        env=env,
    )
    assert migrated.returncode == 0, migrated.stderr

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            names = connection.execute(
                text("SELECT name, kind FROM workspaces ORDER BY id")
            ).all()
        assert len(names) == len(set(names))
        assert all(len(name) <= 255 for name, _kind in names)
        assert sum(name == "X" * 255 for name, _kind in names) == 1
        assert sum(name == "System Duplicate" for name, _kind in names) == 1
    finally:
        engine.dispose()


def test_security_migration_disables_existing_default_bootstrap_admin(tmp_path):
    database_url = _sqlite_url(tmp_path / "vulnerable-admin.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}

    pre_fix_result = _run_backend_command(
        "-m",
        "alembic",
        "upgrade",
        "c7b2d5f8a611",
        env=env,
    )
    assert pre_fix_result.returncode == 0, pre_fix_result.stderr

    engine = create_engine(database_url)
    try:
        now = "2026-07-03 00:00:00"
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(username, email, password_hash, display_name, is_active, "
                    "is_superuser, created_at, updated_at) "
                    "VALUES (:username, NULL, :password_hash, :display_name, "
                    ":is_active, :is_superuser, :created_at, :updated_at)"
                ),
                {
                    "username": "admin",
                    "password_hash": _pwd.hash(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD),
                    "display_name": "admin",
                    "is_active": True,
                    "is_superuser": True,
                    "created_at": now,
                    "updated_at": now,
                },
            )
    finally:
        engine.dispose()

    migrate_result = _run_backend_command("-m", "alembic", "upgrade", "head", env=env)
    assert migrate_result.returncode == 0, migrate_result.stderr

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            admin = connection.execute(
                text("SELECT password_hash, is_active FROM users WHERE username = 'admin'")
            ).one()
        assert not admin[1]
        assert not _pwd.verify(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, admin[0])
    finally:
        engine.dispose()


def test_seed_demo_fails_on_partially_seeded_demo_project(tmp_path):
    database_url = _sqlite_url(tmp_path / "partial-seed.db")
    env = os.environ | {"AL_MEDLIT_DATABASE_URL": database_url}

    migrate_result = _run_backend_command("-m", "alembic", "upgrade", "head", env=env)
    assert migrate_result.returncode == 0, migrate_result.stderr

    engine = create_engine(database_url)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        default_workspace_id = session.execute(
            text(
                "SELECT id FROM workspaces WHERE name = 'Default' "
                "AND kind = 'team' ORDER BY id LIMIT 1"
            )
        ).scalar_one()
        session.add(
            Project(
                name=DEMO_PROJECT_NAME,
                description="partial demo seed",
                annotation_schema={},
                settings={},
                workspace_id=default_workspace_id,
            )
        )
        session.commit()
    finally:
        session.close()
        engine.dispose()

    seed_result = _run_backend_command("-m", "scripts.seed_demo", env=env)
    assert seed_result.returncode != 0
    assert "Demo project already exists but seed is incomplete." in seed_result.stderr
    assert "documents" in seed_result.stderr


@pytest.mark.postgres
def test_postgres_migration_and_seed_via_docker():
    if os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1":
        pytest.skip("Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests")

    port = _free_port()
    container_name = f"al-medlit-postgres-{uuid4().hex[:8]}"
    postgres_url = f"postgresql+psycopg2://al_medlit:secret@127.0.0.1:{port}/al_medlit"
    docker_run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-e",
            "POSTGRES_DB=al_medlit",
            "-e",
            "POSTGRES_USER=al_medlit",
            "-e",
            "POSTGRES_PASSWORD=secret",
            "-p",
            f"{port}:5432",
            "postgres:16-alpine",
        ],
        cwd=_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert docker_run.returncode == 0, docker_run.stderr

    try:
        _wait_for_postgres(postgres_url)

        env = os.environ | {"AL_MEDLIT_DATABASE_URL": postgres_url}
        migrate_result = _run_backend_command("-m", "alembic", "upgrade", "head", env=env)
        assert migrate_result.returncode == 0, migrate_result.stderr

        seed_result = _run_backend_command("-m", "scripts.seed_demo", env=env)
        assert seed_result.returncode == 0, seed_result.stderr

        engine = create_engine(postgres_url)
        try:
            inspector = inspect(engine)
            table_names = set(inspector.get_table_names())
            assert {"projects", "documents", "annotations", "annotation_corrections"}.issubset(
                table_names
            )

            with engine.connect() as connection:
                project_count = connection.execute(
                    text("SELECT COUNT(*) FROM projects")
                ).scalar_one()
                revision = connection.execute(
                    text("SELECT version_num FROM alembic_version")
                ).scalar_one()
            assert project_count == 1
            alembic_cfg = Config(str(_backend_root() / "alembic.ini"))
            expected_head = ScriptDirectory.from_config(alembic_cfg).get_current_head()
            assert revision == expected_head
        finally:
            engine.dispose()
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            cwd=_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
