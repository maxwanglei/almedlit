import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

PREVIOUS_REVISION = "71b9d2e4f6a8"
SCORING_REVISION = "82c4d6e8f1a3"


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


def test_feedback_scoring_migration_backfills_latest_output_and_downgrades(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'feedback-scoring-migration.db'}"
    initial = _run_alembic("upgrade", PREVIOUS_REVISION, database_url=database_url)
    assert initial.returncode == 0, initial.stderr
    engine = create_engine(database_url)
    statements = [
        "INSERT INTO users (id, username, password_hash, display_name, is_active, "
        "is_superuser, session_version, created_at, updated_at) VALUES "
        "(1, 'scoring-migration', 'hash', 'Scoring Migration', 1, 0, 0, :now, :now)",
        "INSERT INTO workspaces (id, name, kind, created_by, capability_preset, "
        "capability_overrides, created_at, updated_at) VALUES "
        "(10, 'Scoring Migration', 'team', 1, 'full', '[]', :now, :now)",
        "INSERT INTO projects (id, workspace_id, name, annotation_schema, settings, "
        "annotation_validation_mode, created_at, updated_at) VALUES "
        "(20, 10, 'Scoring Migration', '{}', '{}', 'relaxed', :now, :now)",
        "INSERT INTO task_definitions (id, project_id, key, name, created_by_user_id, "
        "created_at, updated_at) VALUES "
        "(30, 20, 'classification', 'Classification', 1, :now, :now)",
        "INSERT INTO task_versions (id, project_id, task_definition_id, version_number, "
        "task_kind, input_schema, output_schema, label_rules, annotation_ui, metrics, "
        "trainer_compatibility, content_hash, created_by_user_id, created_at, updated_at) "
        "VALUES (31, 20, 30, 1, 'classification', '{}', '{}', '{}', '{}', '[]', "
        "'[]', :task_hash, 1, :now, :now)",
        "INSERT INTO datasets (id, project_id, name, source_type, created_by_user_id, "
        "created_at, updated_at) VALUES "
        "(40, 20, 'Dataset', 'upload', 1, :now, :now)",
        "INSERT INTO dataset_versions (id, project_id, dataset_id, version_number, "
        "source_revision, source_format, data_schema, provenance, license_info, "
        "content_hash, item_count, created_by_user_id, created_at, updated_at) VALUES "
        "(41, 20, 40, 1, 'revision', 'jsonl', '{}', '{}', '{}', :dataset_hash, 1, "
        "1, :now, :now)",
        "INSERT INTO feedback_runs (id, project_id, dataset_version_id, task_version_id, "
        "producer_type, configuration, data_egress_policy, status, created_by_user_id, "
        "created_at, updated_at) VALUES "
        "(50, 20, 41, 31, 'rule', '{}', '{}', 'failed', 1, :now, :now)",
        "INSERT INTO feedback_set_versions (id, project_id, feedback_run_id, "
        "dataset_version_id, task_version_id, version_number, output_schema, "
        "candidate_count, content_hash, created_by_user_id, created_at, updated_at) "
        "VALUES (51, 20, 50, 41, 31, 1, '{}', 0, :set_hash_one, 1, :first, :first)",
        "INSERT INTO feedback_set_versions (id, project_id, feedback_run_id, "
        "dataset_version_id, task_version_id, version_number, output_schema, "
        "candidate_count, content_hash, created_by_user_id, created_at, updated_at) "
        "VALUES (52, 20, 50, 41, 31, 2, '{}', 0, :set_hash_two, 1, :latest, :latest)",
    ]
    values = {
        "now": "2026-08-18 10:00:00",
        "first": "2026-08-18 11:00:00",
        "latest": "2026-08-18 12:00:00",
        "task_hash": "1" * 64,
        "dataset_hash": "2" * 64,
        "set_hash_one": "3" * 64,
        "set_hash_two": "4" * 64,
    }
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement), values)

    upgraded = _run_alembic("upgrade", SCORING_REVISION, database_url=database_url)
    assert upgraded.returncode == 0, upgraded.stderr
    columns = {column["name"] for column in inspect(engine).get_columns("feedback_runs")}
    assert {
        "output_feedback_set_version_id",
        "failure_code",
        "failure_reason",
        "started_at",
        "heartbeat_at",
        "completed_at",
    }.issubset(columns)
    foreign_keys = inspect(engine).get_foreign_keys("feedback_runs")
    assert any(
        foreign_key["constrained_columns"] == ["output_feedback_set_version_id"]
        and foreign_key["referred_table"] == "feedback_set_versions"
        for foreign_key in foreign_keys
    )
    with engine.connect() as connection:
        status, output_id, completed_at = connection.execute(
            text(
                "SELECT status, output_feedback_set_version_id, completed_at "
                "FROM feedback_runs WHERE id = 50"
            )
        ).one()
    assert status == "failed"
    assert output_id == 52
    assert str(completed_at).startswith("2026-08-18 12:00:00")

    downgraded = _run_alembic("downgrade", PREVIOUS_REVISION, database_url=database_url)
    assert downgraded.returncode == 0, downgraded.stderr
    downgraded_columns = {
        column["name"] for column in inspect(engine).get_columns("feedback_runs")
    }
    assert "output_feedback_set_version_id" not in downgraded_columns
    with engine.connect() as connection:
        assert connection.execute(
            text("SELECT status FROM feedback_runs WHERE id = 50")
        ).scalar_one() == "failed"
