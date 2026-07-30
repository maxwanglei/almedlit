import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session

from al_medlit.workflow.models import Guideline, TrainingRun

PRE_REPAIR_REVISION = "e2d5a8c0f314"
REPAIR_REVISION = "4e8b1c6d9a20"
CLEANUP_REVISION = "6f2a9d4c8b31"


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_alembic(
    revision: str,
    *,
    database_url: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        cwd=_backend_root(),
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": database_url},
        text=True,
        capture_output=True,
        check=False,
    )


def _upgrade_to_pre_repair(database_url: str) -> None:
    result = _run_alembic(PRE_REPAIR_REVISION, database_url=database_url)
    assert result.returncode == 0, result.stderr


def test_repair_migration_is_a_noop_for_the_canonical_schema(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'canonical.db').as_posix()}"
    _upgrade_to_pre_repair(database_url)

    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)

    assert repaired.returncode == 0, repaired.stderr
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
        assert {"training_runs", "guidelines"} <= tables
        assert {"training_runs_v2", "guidelines_v2"}.isdisjoint(tables)
    finally:
        engine.dispose()


def test_repair_migration_renames_v2_tables_without_losing_rows_or_foreign_keys(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'legacy-v2.db').as_posix()}"
    _upgrade_to_pre_repair(database_url)

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE training_runs RENAME TO training_runs_v2"))
            connection.execute(text("ALTER TABLE guidelines RENAME TO guidelines_v2"))
            connection.execute(
                text(
                    """
                    INSERT INTO training_runs_v2 (
                        project_id,
                        registered_model_id,
                        task_version_id,
                        training_dataset_version_id,
                        recipe_version_id,
                        environment_id,
                        storage_policy_id,
                        evaluation_plan,
                        config,
                        seed,
                        status,
                        idempotency_key,
                        launch_hash,
                        runtime_snapshot,
                        storage_snapshot,
                        id,
                        created_at,
                        updated_at
                    ) VALUES (
                        9101,
                        9101,
                        9101,
                        9101,
                        9101,
                        9101,
                        9101,
                        '{}',
                        '{}',
                        42,
                        'queued',
                        'repair-sentinel',
                        'repair-launch-hash',
                        '{}',
                        '{}',
                        9101,
                        '2026-07-30 00:00:00',
                        '2026-07-30 00:00:00'
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO guidelines_v2 (
                        project_id,
                        task_definition_id,
                        name,
                        id,
                        created_at,
                        updated_at
                    ) VALUES (
                        9101,
                        9101,
                        'Repair sentinel',
                        9101,
                        '2026-07-30 00:00:00',
                        '2026-07-30 00:00:00'
                    )
                    """
                )
            )
    finally:
        engine.dispose()

    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)
    assert repaired.returncode == 0, repaired.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert {"training_runs", "guidelines"} <= tables
        assert {"training_runs_v2", "guidelines_v2"}.isdisjoint(tables)

        evaluation_fk_targets = {
            foreign_key["referred_table"]
            for foreign_key in inspector.get_foreign_keys("model_evaluations")
            if foreign_key["constrained_columns"] == ["training_run_id"]
        }
        revision_fk_targets = {
            foreign_key["referred_table"]
            for foreign_key in inspector.get_foreign_keys("guideline_revisions")
            if foreign_key["constrained_columns"] == ["guideline_id"]
        }
        assert evaluation_fk_targets == {"training_runs"}
        assert revision_fk_targets == {"guidelines"}

        with Session(engine) as session:
            training_run = session.get(TrainingRun, 9101)
            guideline = session.get(Guideline, 9101)
            assert training_run is not None
            assert training_run.idempotency_key == "repair-sentinel"
            assert guideline is not None
            assert guideline.name == "Repair sentinel"
    finally:
        engine.dispose()


def test_repair_migration_rejects_ambiguous_duplicate_tables(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'duplicate.db').as_posix()}"
    _upgrade_to_pre_repair(database_url)

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE TABLE training_runs_v2 (id INTEGER PRIMARY KEY)"))
    finally:
        engine.dispose()

    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)

    assert repaired.returncode != 0
    assert "training_runs_v2/training_runs: both exist" in repaired.stderr


def test_repair_migration_rejects_missing_table_pairs(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'missing.db').as_posix()}"
    _upgrade_to_pre_repair(database_url)

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(text("DROP TABLE model_evaluations"))
            connection.execute(text("DROP TABLE training_runs"))
    finally:
        engine.dispose()

    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)

    assert repaired.returncode != 0
    assert "training_runs_v2/training_runs: neither exists" in repaired.stderr


def test_cleanup_migration_removes_only_empty_v2_rollout_artifacts(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'empty-rollout.db').as_posix()}"
    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)
    assert repaired.returncode == 0, repaired.stderr

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE workspaces ADD COLUMN "
                    "platform_v2_enabled BOOLEAN NOT NULL DEFAULT 0"
                )
            )
            connection.execute(
                text("CREATE TABLE legacy_v2_backfill_cohorts (id INTEGER PRIMARY KEY)")
            )
    finally:
        engine.dispose()

    cleaned = _run_alembic(CLEANUP_REVISION, database_url=database_url)
    assert cleaned.returncode == 0, cleaned.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "legacy_v2_backfill_cohorts" not in inspector.get_table_names()
        assert "platform_v2_enabled" not in {
            column["name"] for column in inspector.get_columns("workspaces")
        }
    finally:
        engine.dispose()


def test_cleanup_migration_preserves_nonempty_backfill_provenance(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'populated-rollout.db').as_posix()}"
    repaired = _run_alembic(REPAIR_REVISION, database_url=database_url)
    assert repaired.returncode == 0, repaired.stderr

    engine = create_engine(database_url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE workspaces ADD COLUMN "
                    "platform_v2_enabled BOOLEAN NOT NULL DEFAULT 0"
                )
            )
            connection.execute(
                text("CREATE TABLE legacy_v2_backfill_cohorts (id INTEGER PRIMARY KEY)")
            )
            connection.execute(text("INSERT INTO legacy_v2_backfill_cohorts (id) VALUES (1)"))
    finally:
        engine.dispose()

    cleaned = _run_alembic(CLEANUP_REVISION, database_url=database_url)
    assert cleaned.returncode != 0
    assert "contains 1 provenance record(s)" in cleaned.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert "legacy_v2_backfill_cohorts" in inspector.get_table_names()
        assert "platform_v2_enabled" in {
            column["name"] for column in inspector.get_columns("workspaces")
        }
    finally:
        engine.dispose()
