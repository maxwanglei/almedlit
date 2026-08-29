import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

PREVIOUS_REVISION = "82c4d6e8f1a3"
PATTERN_REVISION = "a6d1f9c3e8b2"


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


def _insert_pattern(connection, *, pattern_id, label_type, count, correction_id):
    connection.execute(
        text(
            "INSERT INTO error_patterns ("
            "id, project_id, task_type, error_type, label_type, description, "
            "example_count, severity, detected_from, status, example_ids, metadata, "
            "created_at, updated_at"
            ") VALUES ("
            ":id, 20, 'entity', 'boundary_error', :label_type, 'Boundary error', "
            ":count, 'medium', 'adjudication', 'active', :example_ids, '{}', :now, :now"
            ")"
        ),
        {
            "id": pattern_id,
            "label_type": label_type,
            "count": count,
            "example_ids": json.dumps(
                [{"correction_id": correction_id, "document_id": 30}]
            ),
            "now": "2026-08-29 12:00:00",
        },
    )


def test_error_pattern_migration_merges_duplicates_and_enforces_active_identity(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'error-pattern-migration.db'}"
    initial = _run_alembic("upgrade", PREVIOUS_REVISION, database_url=database_url)
    assert initial.returncode == 0, initial.stderr
    engine = create_engine(database_url)

    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO users (id, username, password_hash, display_name, is_active, "
                "is_superuser, session_version, created_at, updated_at) VALUES "
                "(1, 'pattern-migration', 'hash', 'Pattern Migration', 1, 0, 0, :now, :now)"
            ),
            {"now": "2026-08-29 12:00:00"},
        )
        connection.execute(
            text(
                "INSERT INTO workspaces (id, name, kind, created_by, capability_preset, "
                "capability_overrides, created_at, updated_at) VALUES "
                "(10, 'Pattern Migration', 'team', 1, 'full', '[]', :now, :now)"
            ),
            {"now": "2026-08-29 12:00:00"},
        )
        connection.execute(
            text(
                "INSERT INTO projects (id, workspace_id, name, annotation_schema, settings, "
                "annotation_validation_mode, created_at, updated_at) VALUES "
                "(20, 10, 'Pattern Migration', '{}', '{}', 'relaxed', :now, :now)"
            ),
            {"now": "2026-08-29 12:00:00"},
        )
        _insert_pattern(
            connection,
            pattern_id=100,
            label_type="Drug",
            count=1,
            correction_id=1,
        )
        _insert_pattern(
            connection,
            pattern_id=101,
            label_type="Drug",
            count=2,
            correction_id=2,
        )
        _insert_pattern(
            connection,
            pattern_id=102,
            label_type=None,
            count=1,
            correction_id=3,
        )
        _insert_pattern(
            connection,
            pattern_id=103,
            label_type=None,
            count=3,
            correction_id=4,
        )
        connection.execute(
            text(
                "INSERT INTO guideline_atoms ("
                "id, project_id, error_pattern_id, task_type, error_type, rule_text, "
                "positive_examples, negative_examples, applies_to, status, created_at, updated_at"
                ") VALUES (200, 20, 101, 'entity', 'boundary_error', 'Rule', '[]', '[]', "
                "'[]', 'pending', :now, :now)"
            ),
            {"now": "2026-08-29 12:00:00"},
        )
        connection.execute(
            text(
                "INSERT INTO training_actions ("
                "id, project_id, error_pattern_id, action_type, target_model, example_ids, "
                "priority, status, created_at, updated_at"
                ") VALUES (300, 20, 103, 'retrain', 'model', '[]', 'medium', 'planned', "
                ":now, :now)"
            ),
            {"now": "2026-08-29 12:00:00"},
        )

    upgraded = _run_alembic("upgrade", PATTERN_REVISION, database_url=database_url)
    assert upgraded.returncode == 0, upgraded.stderr

    index_names = {item["name"] for item in inspect(engine).get_indexes("error_patterns")}
    assert {
        "uq_error_patterns_active_labeled",
        "uq_error_patterns_active_unlabeled",
    } <= index_names
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT id, label_type, example_count, example_ids "
                "FROM error_patterns ORDER BY id"
            )
        ).mappings().all()
        assert [row["id"] for row in rows] == [100, 102]
        assert [row["example_count"] for row in rows] == [3, 4]
        assert [
            item["correction_id"]
            for item in json.loads(rows[0]["example_ids"])
        ] == [1, 2]
        assert [
            item["correction_id"]
            for item in json.loads(rows[1]["example_ids"])
        ] == [3, 4]
        assert connection.execute(
            text("SELECT error_pattern_id FROM guideline_atoms WHERE id = 200")
        ).scalar_one() == 100
        assert connection.execute(
            text("SELECT error_pattern_id FROM training_actions WHERE id = 300")
        ).scalar_one() == 102

    for label_type in ("Drug", None):
        with pytest.raises(IntegrityError), engine.begin() as connection:
            _insert_pattern(
                connection,
                pattern_id=400 if label_type else 401,
                label_type=label_type,
                count=1,
                correction_id=5,
            )

    downgraded = _run_alembic("downgrade", PREVIOUS_REVISION, database_url=database_url)
    assert downgraded.returncode == 0, downgraded.stderr
    downgraded_indexes = {
        item["name"] for item in inspect(engine).get_indexes("error_patterns")
    }
    assert "uq_error_patterns_active_labeled" not in downgraded_indexes
    assert "uq_error_patterns_active_unlabeled" not in downgraded_indexes
    engine.dispose()
