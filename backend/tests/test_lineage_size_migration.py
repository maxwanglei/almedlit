import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import BigInteger, create_engine, inspect

from al_medlit.lineage.models import LineageArtifact

PRE_MIGRATION_REVISION = "c3f7a1e9d5b2"
SIZE_MIGRATION_REVISION = "e7b4c2d9a106"


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


def _size_type_name(database_url: str) -> str:
    engine = create_engine(database_url)
    try:
        columns = inspect(engine).get_columns("lineage_artifacts")
        return next(
            type(column["type"]).__name__
            for column in columns
            if column["name"] == "size_bytes"
        )
    finally:
        engine.dispose()


def test_lineage_model_uses_big_integer_size():
    assert isinstance(LineageArtifact.__table__.c.size_bytes.type, BigInteger)


def test_lineage_size_migration_round_trip(tmp_path: Path):
    database_url = f"sqlite:///{(tmp_path / 'lineage-size.db').as_posix()}"
    initial = _run_alembic(
        "upgrade",
        PRE_MIGRATION_REVISION,
        database_url=database_url,
    )
    assert initial.returncode == 0, initial.stderr
    assert _size_type_name(database_url) == "INTEGER"

    upgraded = _run_alembic(
        "upgrade",
        SIZE_MIGRATION_REVISION,
        database_url=database_url,
    )
    assert upgraded.returncode == 0, upgraded.stderr
    assert _size_type_name(database_url) == "BIGINT"

    downgraded = _run_alembic(
        "downgrade",
        PRE_MIGRATION_REVISION,
        database_url=database_url,
    )
    assert downgraded.returncode == 0, downgraded.stderr
    assert _size_type_name(database_url) == "INTEGER"
