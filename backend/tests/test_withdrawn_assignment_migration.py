import os
import subprocess
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

PREVIOUS = "c0e6f8a3b7d2"
REVISION = "b7c2e4a91d63"
INDEX = "uq_task_assignments_live_scope"
OLD_CONSTRAINT = "uq_task_assignments_task_document_assignee_scope"
NOW = "2026-10-16 12:00:00"


def _migrate(database_url: str, direction: str, revision: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", direction, revision],
        cwd=Path(__file__).resolve().parents[1],
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": database_url},
        capture_output=True,
        text=True,
        check=False,
    )


def _seed_old_assignment(engine: sa.Engine) -> None:
    statements = [
        "INSERT INTO users (id, username, password_hash, display_name, is_active, "
        "is_superuser, created_at, updated_at) VALUES "
        "(9101, 'migration-assignee', 'unusable', 'Assignee', 1, 0, :now, :now)",
        "INSERT INTO projects (id, name, annotation_schema, settings, "
        "annotation_validation_mode, workspace_id, created_at, updated_at) "
        "VALUES (9101, 'assignment-migration', '{}', '{}', 'relaxed', "
        "(SELECT id FROM workspaces ORDER BY id LIMIT 1), :now, :now)",
        "INSERT INTO documents (id, project_id, text, metadata, created_at, updated_at) "
        "VALUES (9101, 9101, 'Document', '{}', :now, :now)",
        "INSERT INTO project_tasks (id, project_id, annotation_type, display_name, "
        "enabled, sort_order, labels, settings, created_at, updated_at) "
        "VALUES (9101, 9101, 'entity', 'Entities', 1, 0, '[]', '{}', :now, :now)",
        "INSERT INTO task_assignments (id, project_id, task_id, document_id, "
        "assignee_user_id, annotator_id, assignment_scope_key, status, metadata, "
        "created_at, updated_at) VALUES (9101, 9101, 9101, 9101, 9101, "
        "'migration-assignee', 'document', 'withdrawn', :audit, :now, :now)",
        "CREATE TABLE assignment_migration_reference "
        "(assignment_id INTEGER NOT NULL REFERENCES task_assignments(id))",
        "INSERT INTO assignment_migration_reference VALUES (9101)",
    ]
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(sa.text(statement), {"now": NOW, "audit": '{"audit":true}'})


def _insert_assignment(connection: sa.Connection, assignment_id: int, status: str) -> None:
    connection.execute(
        sa.text(
            "INSERT INTO task_assignments (id, project_id, task_id, document_id, "
            "assignee_user_id, annotator_id, assignment_scope_key, status, metadata, "
            "created_at, updated_at) VALUES (:id, 9101, 9101, 9101, 9101, "
            "'migration-assignee', 'document', :status, '{}', :now, :now)"
        ),
        {"id": assignment_id, "status": status, "now": NOW},
    )


@pytest.mark.parametrize("current_status", ["assigned", "submitted", "completed"])
def test_withdrawn_assignment_history_allows_repeatable_rejoin(
    tmp_path: Path, current_status: str
) -> None:
    database_url = f"sqlite:///{tmp_path / 'assignment-history.db'}"
    initial = _migrate(database_url, "upgrade", PREVIOUS)
    assert initial.returncode == 0, initial.stderr
    engine = sa.create_engine(database_url)
    try:
        _seed_old_assignment(engine)
        upgraded = _migrate(database_url, "upgrade", REVISION)
        assert upgraded.returncode == 0, upgraded.stderr
        index_names = {
            index["name"] for index in sa.inspect(engine).get_indexes("task_assignments")
        }
        assert INDEX in index_names

        with engine.begin() as connection:
            _insert_assignment(connection, 9102, "assigned")
            connection.execute(
                sa.text("UPDATE task_assignments SET status = 'withdrawn' WHERE id = 9102")
            )
            _insert_assignment(connection, 9103, current_status)
            rows = connection.execute(
                sa.text(
                    "SELECT id, status, assignment_scope_key, metadata FROM task_assignments "
                    "WHERE id IN (9101, 9102, 9103) ORDER BY id"
                )
            ).all()
            assert rows == [
                (9101, "withdrawn", "document", '{"audit":true}'),
                (9102, "withdrawn", "document", "{}"),
                (9103, current_status, "document", "{}"),
            ]
            assert connection.execute(
                sa.text("SELECT assignment_id FROM assignment_migration_reference")
            ).scalar_one() == 9101
            assert connection.exec_driver_sql(
                "PRAGMA foreign_key_check(assignment_migration_reference)"
            ).all() == []
            with pytest.raises(IntegrityError):
                with connection.begin_nested():
                    _insert_assignment(connection, 9104, "assigned")

        refused = _migrate(database_url, "downgrade", PREVIOUS)
        assert refused.returncode != 0
        assert "Cannot restore assignment uniqueness" in refused.stderr
        with engine.connect() as connection:
            version = connection.execute(sa.text("SELECT version_num FROM alembic_version"))
            assert version.scalar_one() == REVISION
            count = connection.execute(sa.text("SELECT COUNT(*) FROM task_assignments"))
            assert count.scalar_one() == 3
            assert INDEX in {
                index["name"] for index in sa.inspect(connection).get_indexes("task_assignments")
            }
    finally:
        engine.dispose()


def test_withdrawn_assignment_migration_downgrades_without_duplicate_history(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{tmp_path / 'assignment-clean-downgrade.db'}"
    initial = _migrate(database_url, "upgrade", PREVIOUS)
    assert initial.returncode == 0, initial.stderr
    engine = sa.create_engine(database_url)
    try:
        _seed_old_assignment(engine)
        upgraded = _migrate(database_url, "upgrade", REVISION)
        assert upgraded.returncode == 0, upgraded.stderr
        downgraded = _migrate(database_url, "downgrade", PREVIOUS)
        assert downgraded.returncode == 0, downgraded.stderr
        unique_names = {
            constraint["name"]
            for constraint in sa.inspect(engine).get_unique_constraints("task_assignments")
        }
        assert OLD_CONSTRAINT in unique_names
        with engine.begin() as connection:
            assert connection.execute(
                sa.text("SELECT metadata FROM task_assignments WHERE id = 9101")
            ).scalar_one() == '{"audit":true}'
            assert connection.execute(
                sa.text("SELECT assignment_id FROM assignment_migration_reference")
            ).scalar_one() == 9101
            with pytest.raises(IntegrityError):
                with connection.begin_nested():
                    _insert_assignment(connection, 9102, "assigned")
    finally:
        engine.dispose()
