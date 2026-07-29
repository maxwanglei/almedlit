import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

PRE_EVIDENCE_REVISION = "3f8d1c7a9b24"
EVIDENCE_REVISION = "7a2e4c9b1d60"

FEATURE_TABLES = {
    "annotation_set_items",
    "annotation_set_review_regions",
    "annotation_sets",
    "compute_profiles",
    "corpus_snapshot_documents",
    "corpus_snapshots",
    "document_paragraphs",
    "document_sections",
    "document_sentences",
    "document_structure_versions",
    "evidence_block_annotations",
    "evidence_block_revisions",
    "evidence_candidate_predictions",
    "evidence_prediction_reviews",
    "evidence_review_coverage",
    "evidence_review_events",
    "evidence_target_versions",
    "evidence_targets",
    "export_artifacts",
    "inference_runs",
    "inference_windows",
    "lineage_artifacts",
    "lineage_edges",
    "model_checkpoints",
    "training_experiments",
    "training_jobs",
}


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_alembic(
    direction: str,
    revision: str,
    *,
    database_url: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", direction, revision],
        cwd=_backend_root(),
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": database_url},
        text=True,
        capture_output=True,
        check=False,
    )


def _insert_legacy_assignment(database_url: str) -> None:
    engine = create_engine(database_url)
    timestamp = "2026-07-15 00:00:00"
    try:
        with engine.begin() as connection:
            workspace_id = connection.execute(
                text("SELECT id FROM workspaces ORDER BY id LIMIT 1")
            ).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, username, password_hash, display_name, is_active, "
                    "is_superuser, created_at, updated_at) VALUES "
                    "(9101, 'evidence-migration-user', 'unusable', 'Migration User', "
                    "1, 0, :timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, name, annotation_schema, settings, "
                    "annotation_validation_mode, workspace_id, created_at, updated_at) "
                    "VALUES (9101, 'evidence-migration-project', '{}', '{}', "
                    "'relaxed', :workspace_id, :timestamp, :timestamp)"
                ),
                {"workspace_id": workspace_id, "timestamp": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, text, metadata, created_at, updated_at) "
                    "VALUES (9101, 9101, 'Alpha. Beta.', '{}', :timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO project_tasks "
                    "(id, project_id, annotation_type, display_name, enabled, "
                    "sort_order, labels, settings, created_at, updated_at) VALUES "
                    "(9101, 9101, 'entity', 'Entities', 1, 0, '[]', '{}', "
                    ":timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO task_assignments "
                    "(id, project_id, task_id, document_id, annotator_id, status, "
                    "metadata, assignee_user_id, created_at, updated_at) VALUES "
                    "(9101, 9101, 9101, 9101, 'evidence-migration-user', 'assigned', "
                    "'{}', 9101, :timestamp, :timestamp)"
                ),
                {"timestamp": timestamp},
            )
    finally:
        engine.dispose()


def test_evidence_platform_migration_upgrades_existing_assignments_and_downgrades(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'evidence-platform.db').as_posix()}"
    initial = _run_alembic(
        "upgrade",
        PRE_EVIDENCE_REVISION,
        database_url=database_url,
    )
    assert initial.returncode == 0, initial.stderr
    _insert_legacy_assignment(database_url)

    upgraded = _run_alembic("upgrade", EVIDENCE_REVISION, database_url=database_url)
    assert upgraded.returncode == 0, upgraded.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assignment_columns = {
            column["name"] for column in inspector.get_columns("task_assignments")
        }
        with engine.connect() as connection:
            scope_key = connection.execute(
                text(
                    "SELECT assignment_scope_key FROM task_assignments "
                    "WHERE id = 9101"
                )
            ).scalar_one()

        assert FEATURE_TABLES <= tables
        assert {
            "target_version_id",
            "structure_version_id",
            "guideline_version_id",
            "assignment_scope_key",
        } <= assignment_columns
        assert scope_key == "document"
        assert any(
            foreign_key["constrained_columns"] == ["active_version_id"]
            for foreign_key in inspector.get_foreign_keys("evidence_targets")
        )
    finally:
        engine.dispose()

    downgraded = _run_alembic(
        "downgrade",
        PRE_EVIDENCE_REVISION,
        database_url=database_url,
    )
    assert downgraded.returncode == 0, downgraded.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert FEATURE_TABLES.isdisjoint(inspector.get_table_names())
        assert "assignment_scope_key" not in {
            column["name"] for column in inspector.get_columns("task_assignments")
        }
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT COUNT(*) FROM task_assignments WHERE id = 9101")
            ).scalar_one() == 1
    finally:
        engine.dispose()
