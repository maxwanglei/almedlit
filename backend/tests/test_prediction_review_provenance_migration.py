import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import postgresql

from migrations.versions.a8c4e7d1f2b6_pin_prediction_reviews_to_assignment_rounds import (
    ASSIGNMENT_FK_NAME,
    GUIDELINE_VERSION_FK_NAME,
)

PRE_PROVENANCE_REVISION = "9d3f6a1c8b24"
PROVENANCE_REVISION = "a8c4e7d1f2b6"


def test_prediction_review_provenance_constraint_names_fit_postgresql() -> None:
    max_identifier_length = postgresql.dialect().max_identifier_length

    assert len(ASSIGNMENT_FK_NAME) <= max_identifier_length
    assert len(GUIDELINE_VERSION_FK_NAME) <= max_identifier_length


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


def test_prediction_review_provenance_migration_adds_round_foreign_keys(
    tmp_path: Path,
) -> None:
    database_url = f"sqlite:///{(tmp_path / 'prediction-provenance.db').as_posix()}"
    initial = _run_alembic(
        "upgrade",
        PRE_PROVENANCE_REVISION,
        database_url=database_url,
    )
    assert initial.returncode == 0, initial.stderr

    upgraded = _run_alembic(
        "upgrade",
        PROVENANCE_REVISION,
        database_url=database_url,
    )
    assert upgraded.returncode == 0, upgraded.stderr

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        columns = {
            column["name"]
            for column in inspector.get_columns("evidence_prediction_reviews")
        }
        foreign_keys = {
            (tuple(foreign_key["constrained_columns"]), foreign_key["referred_table"])
            for foreign_key in inspector.get_foreign_keys(
                "evidence_prediction_reviews"
            )
        }
        indexes = {
            tuple(index["column_names"])
            for index in inspector.get_indexes("evidence_prediction_reviews")
        }
        assert {"assignment_id", "guideline_version_id"} <= columns
        assert (("assignment_id",), "task_assignments") in foreign_keys
        assert (("guideline_version_id",), "guideline_versions") in foreign_keys
        assert ("assignment_id",) in indexes
        assert ("guideline_version_id",) in indexes
    finally:
        engine.dispose()

    downgraded = _run_alembic(
        "downgrade",
        PRE_PROVENANCE_REVISION,
        database_url=database_url,
    )
    assert downgraded.returncode == 0, downgraded.stderr

    engine = create_engine(database_url)
    try:
        columns = {
            column["name"]
            for column in inspect(engine).get_columns("evidence_prediction_reviews")
        }
        assert "assignment_id" not in columns
        assert "guideline_version_id" not in columns
    finally:
        engine.dispose()
