import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect, text

from al_medlit.annotation.schemas import (
    AnnotationCorrectionCreate,
    AnnotationCorrectionRead,
)


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_alembic(revision: str, *, database_url: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        cwd=_backend_root(),
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": database_url},
        text=True,
        capture_output=True,
        check=False,
    )


def _run_alembic_downgrade(
    revision: str,
    *,
    database_url: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", revision],
        cwd=_backend_root(),
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": database_url},
        text=True,
        capture_output=True,
        check=False,
    )


def test_correction_owner_is_read_only_in_api_schema():
    assert "created_by_user_id" not in AnnotationCorrectionCreate.model_fields

    correction = AnnotationCorrectionRead(
        id=1,
        project_id=2,
        document_id=3,
        created_by_user_id=4,
    )

    assert correction.created_by_user_id == 4


def test_correction_ownership_migration_backfills_only_unambiguous_owners(tmp_path):
    database_url = f"sqlite:///{(tmp_path / 'correction-ownership.db').as_posix()}"
    pre_migration = _run_alembic("f2c9d1e4a6b8", database_url=database_url)
    assert pre_migration.returncode == 0, pre_migration.stderr

    engine = create_engine(database_url)
    try:
        timestamp = "2026-07-09 00:00:00"
        with engine.begin() as connection:
            workspace_id = connection.execute(
                text(
                    "SELECT id FROM workspaces WHERE name = 'Default' "
                    "AND kind = 'team' ORDER BY id LIMIT 1"
                )
            ).scalar_one()
            connection.execute(
                text(
                    "INSERT INTO users "
                    "(id, username, email, password_hash, display_name, is_active, "
                    "is_superuser, created_at, updated_at) VALUES "
                    "(:id, :username, NULL, :password_hash, :display_name, 1, 0, "
                    ":created_at, :updated_at)"
                ),
                [
                    {
                        "id": 1001,
                        "username": "migration-owner-a",
                        "password_hash": "unusable",
                        "display_name": "Owner A",
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    {
                        "id": 1002,
                        "username": "migration-owner-b",
                        "password_hash": "unusable",
                        "display_name": "Owner B",
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                ],
            )
            connection.execute(
                text(
                    "INSERT INTO projects "
                    "(id, name, description, annotation_schema, settings, "
                    "annotation_validation_mode, workspace_id, created_at, updated_at) "
                    "VALUES (1001, 'correction-migration-project', NULL, '{}', '{}', "
                    "'relaxed', :workspace_id, :created_at, :updated_at)"
                ),
                {
                    "workspace_id": workspace_id,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO documents "
                    "(id, project_id, external_id, title, text, source, metadata, "
                    "created_at, updated_at) VALUES "
                    "(1001, 1001, NULL, NULL, 'migration text', NULL, '{}', "
                    ":created_at, :updated_at)"
                ),
                {"created_at": timestamp, "updated_at": timestamp},
            )
            connection.execute(
                text(
                    "INSERT INTO annotations "
                    "(id, project_id, document_id, annotation_type, label, "
                    "start_offset, end_offset, text_span, source, status, confidence, "
                    "annotator_id, annotator_user_id, model_checkpoint_id, "
                    "guideline_version_id, head_annotation_id, tail_annotation_id, "
                    "evidence, attributes, created_at, updated_at) VALUES "
                    "(:id, 1001, 1001, 'entity', 'Finding', NULL, NULL, NULL, "
                    "'human', 'draft', NULL, NULL, :owner_id, NULL, NULL, NULL, NULL, "
                    "'{}', '{}', :created_at, :updated_at)"
                ),
                [
                    {
                        "id": 1001,
                        "owner_id": 1001,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    {
                        "id": 1002,
                        "owner_id": 1001,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    {
                        "id": 1003,
                        "owner_id": 1002,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    {
                        "id": 1004,
                        "owner_id": None,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                ],
            )
            connection.execute(
                text(
                    "INSERT INTO annotation_corrections "
                    "(id, project_id, document_id, original_annotation_id, "
                    "corrected_annotation_id, correction_source, correction_note, "
                    "error_type, severity, metadata, created_at, updated_at) VALUES "
                    "(:id, 1001, 1001, :original_id, :corrected_id, 'human', NULL, "
                    "NULL, 'medium', '{}', :created_at, :updated_at)"
                ),
                [
                    # Two linked annotations with the same owner.
                    {
                        "id": 1001,
                        "original_id": 1001,
                        "corrected_id": 1002,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    # Conflicting linked owners must remain manager-only.
                    {
                        "id": 1002,
                        "original_id": 1001,
                        "corrected_id": 1003,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    # One owned and one unowned annotation has one distinct owner.
                    {
                        "id": 1003,
                        "original_id": 1001,
                        "corrected_id": 1004,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    # Ownerless and reference-free corrections remain manager-only.
                    {
                        "id": 1004,
                        "original_id": 1004,
                        "corrected_id": None,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    {
                        "id": 1005,
                        "original_id": None,
                        "corrected_id": None,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                    # A single corrected annotation can establish ownership.
                    {
                        "id": 1006,
                        "original_id": None,
                        "corrected_id": 1003,
                        "created_at": timestamp,
                        "updated_at": timestamp,
                    },
                ],
            )
    finally:
        engine.dispose()

    migrated = _run_alembic("3f8d1c7a9b24", database_url=database_url)
    assert migrated.returncode == 0, migrated.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        column_names = {
            column["name"] for column in inspector.get_columns("annotation_corrections")
        }
        index_names = {
            index["name"] for index in inspector.get_indexes("annotation_corrections")
        }
        ownership_fks = [
            foreign_key
            for foreign_key in inspector.get_foreign_keys("annotation_corrections")
            if foreign_key["constrained_columns"] == ["created_by_user_id"]
        ]
        with engine.connect() as connection:
            owners = connection.execute(
                text(
                    "SELECT id, created_by_user_id FROM annotation_corrections "
                    "WHERE id >= 1001 ORDER BY id"
                )
            ).all()

        assert "created_by_user_id" in column_names
        assert "ix_annotation_corrections_created_by_user_id" in index_names
        assert len(ownership_fks) == 1
        assert ownership_fks[0]["referred_table"] == "users"
        assert owners == [
            (1001, 1001),
            (1002, None),
            (1003, 1001),
            (1004, None),
            (1005, None),
            (1006, 1002),
        ]
    finally:
        engine.dispose()

    downgraded = _run_alembic_downgrade("f2c9d1e4a6b8", database_url=database_url)
    assert downgraded.returncode == 0, downgraded.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        column_names = {
            column["name"] for column in inspector.get_columns("annotation_corrections")
        }
        with engine.connect() as connection:
            correction_count = connection.execute(
                text("SELECT COUNT(*) FROM annotation_corrections WHERE id >= 1001")
            ).scalar_one()

        assert "created_by_user_id" not in column_names
        assert correction_count == 6
    finally:
        engine.dispose()
