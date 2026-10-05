import os
import subprocess
import sys
from pathlib import Path

import sqlalchemy as sa

PREVIOUS = "d9a3c5e1f742"
REVISION = "c0e6f8a3b7d2"


def _migrate(database_url, direction, revision):
    environment = os.environ.copy()
    environment["AL_MEDLIT_DATABASE_URL"] = database_url
    result = subprocess.run(
        [sys.executable, "-m", "alembic", direction, revision],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_series_migration_preserves_existing_versions_and_references(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'training-series.db'}"
    _migrate(database_url, "upgrade", PREVIOUS)
    engine = sa.create_engine(database_url)
    now = "2026-09-14 12:00:00"
    statements = [
        "INSERT INTO users (id, username, password_hash, display_name, is_active, "
        "is_superuser, session_version, created_at, updated_at) VALUES "
        "(1, 'migration', 'hash', 'Migration', 1, 0, 0, :now, :now)",
        "INSERT INTO workspaces (id, name, kind, created_by, capability_preset, "
        "capability_overrides, created_at, updated_at) VALUES "
        "(10, 'Workspace', 'individual', 1, 'full', '[]', :now, :now)",
        "INSERT INTO projects (id, workspace_id, name, annotation_schema, settings, "
        "annotation_validation_mode, created_at, updated_at) VALUES "
        "(20, 10, 'Project', '{}', '{}', 'relaxed', :now, :now)",
        "INSERT INTO task_definitions (id, project_id, key, name, created_by_user_id, "
        "created_at, updated_at) VALUES (30, 20, 'classify', 'Classify', 1, :now, :now)",
        "INSERT INTO task_versions (id, project_id, task_definition_id, version_number, "
        "task_kind, input_schema, output_schema, label_rules, annotation_ui, metrics, "
        "trainer_compatibility, content_hash, created_by_user_id, created_at, updated_at) "
        "VALUES (31, 20, 30, 1, 'classification', '{}', '{}', '{}', '{}', '[]', '[]', "
        ":task_hash, 1, :now, :now)",
        "INSERT INTO datasets (id, project_id, name, source_type, created_by_user_id, "
        "created_at, updated_at) VALUES (40, 20, 'Source', 'upload', 1, :now, :now)",
        "INSERT INTO dataset_versions (id, project_id, dataset_id, version_number, "
        "source_revision, source_format, data_schema, provenance, license_info, "
        "content_hash, item_count, created_by_user_id, created_at, updated_at) VALUES "
        "(41, 20, 40, 1, '1', 'jsonl', '{}', '{}', '{}', :dataset_hash, 1, 1, :now, :now)",
        "INSERT INTO split_maps (id, project_id, dataset_version_id, name, strategy, seed, "
        "assignments, protected_splits, content_hash, created_by_user_id, created_at, updated_at) "
        "VALUES (42, 20, 41, 'Split', 'manual', 42, '{}', '[\"test\"]', "
        ":split_hash, 1, :now, :now)",
        "INSERT INTO training_dataset_versions (id, project_id, name, dataset_version_id, "
        "task_version_id, label_set_version_ids, split_map_id, composition, preprocessing, "
        "content_hash, created_by_user_id, created_at, updated_at) VALUES "
        "(50, 20, 'Same name', 41, 31, '[]', 42, '[]', '{}', :training_one, 1, :now, :now)",
        "INSERT INTO training_dataset_versions (id, project_id, name, dataset_version_id, "
        "task_version_id, label_set_version_ids, split_map_id, composition, preprocessing, "
        "content_hash, created_by_user_id, created_at, updated_at) VALUES "
        "(51, 20, 'Same name', 41, 31, '[]', 42, '[]', '{}', :training_two, 1, :now, :now)",
        "CREATE TABLE preparation_migration_pin "
        "(version_id INTEGER NOT NULL REFERENCES training_dataset_versions(id))",
        "INSERT INTO preparation_migration_pin VALUES (50)",
    ]
    values = {
        "now": now,
        "task_hash": "1" * 64,
        "dataset_hash": "2" * 64,
        "split_hash": "3" * 64,
        "training_one": "4" * 64,
        "training_two": "5" * 64,
    }
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(sa.text(statement), values)
    _migrate(database_url, "upgrade", REVISION)
    with engine.connect() as connection:
        rows = (
            connection.execute(
                sa.text(
                    "SELECT id, name, content_hash, training_dataset_id, "
                    "version_number, parent_version_id "
                    "FROM training_dataset_versions ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
        assert [row["id"] for row in rows] == [50, 51]
        assert [row["name"] for row in rows] == ["Same name", "Same name"]
        assert [row["content_hash"] for row in rows] == ["4" * 64, "5" * 64]
        assert all(row["version_number"] == 1 and row["parent_version_id"] is None for row in rows)
        assert rows[0]["training_dataset_id"] != rows[1]["training_dataset_id"]
        names = (
            connection.execute(sa.text("SELECT name FROM training_datasets ORDER BY id"))
            .scalars()
            .all()
        )
        assert names == ["Same name", "Same name (legacy-51)"]
        assert (
            connection.execute(sa.text("SELECT version_id FROM preparation_migration_pin")).scalar()
            == 50
        )
        assert (
            connection.execute(sa.text("SELECT purposes FROM datasets WHERE id = 40")).scalar()
            == "[]"
        )
        assert not connection.execute(sa.text("PRAGMA foreign_key_check")).all()
    _migrate(database_url, "downgrade", PREVIOUS)
    assert "training_datasets" not in sa.inspect(engine).get_table_names()
    with engine.connect() as connection:
        assert (
            connection.execute(sa.text("SELECT version_id FROM preparation_migration_pin")).scalar()
            == 50
        )
        assert (
            connection.execute(
                sa.text("SELECT content_hash FROM training_dataset_versions WHERE id = 50")
            ).scalar()
            == "4" * 64
        )
        assert not connection.execute(sa.text("PRAGMA foreign_key_check")).all()
    engine.dispose()
