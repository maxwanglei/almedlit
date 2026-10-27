"""The round request identity migration preserves existing user data."""

import runpy
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.exc import IntegrityError


def test_round_request_identity_migration_round_trip():
    migration_path = (
        Path(__file__).parents[1]
        / "migrations/versions/f6a1d8c3e902_add_round_creation_request_identity.py"
    )
    migration = runpy.run_path(str(migration_path))
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            sa.text(
                "CREATE TABLE annotation_rounds (id INTEGER PRIMARY KEY, project_id INTEGER, "
                "created_by_user_id INTEGER, reason TEXT)"
            )
        )
        connection.execute(
            sa.text("INSERT INTO annotation_rounds VALUES (1, 7, 9, 'Original user reason')")
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration["upgrade"]()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns("annotation_rounds")
        }
        assert {"request_key", "request_hash"} <= columns
        connection.execute(
            sa.text(
                "INSERT INTO annotation_rounds (id, project_id, created_by_user_id, request_key) "
                "VALUES (2, 7, 9, 'first'), (3, 7, 10, 'first'), (4, 8, 9, 'first'), "
                "(5, 7, 9, NULL), (6, 7, 9, NULL)"
            )
        )
        with pytest.raises(IntegrityError):
            with connection.begin_nested():
                connection.execute(
                    sa.text(
                        "INSERT INTO annotation_rounds "
                        "(id, project_id, created_by_user_id, request_key) "
                        "VALUES (7, 7, 9, 'first')"
                    )
                )
        with Operations.context(MigrationContext.configure(connection)):
            migration["downgrade"]()
        columns = {
            column["name"] for column in sa.inspect(connection).get_columns("annotation_rounds")
        }
        assert "request_key" not in columns
        assert "request_hash" not in columns
        assert (
            connection.execute(
                sa.text("SELECT reason FROM annotation_rounds WHERE id=1")
            ).scalar_one()
            == "Original user reason"
        )
        assert (
            connection.execute(sa.text("SELECT COUNT(*) FROM annotation_rounds")).scalar_one() == 6
        )
    engine.dispose()
