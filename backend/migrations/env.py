"""Alembic environment configuration for AL-MedLit.

PostgreSQL is the canonical target. Batch mode (``render_as_batch=True``) is
kept on as cheap insurance: it costs nothing on PostgreSQL and keeps SQLite
compatible for the unit-test fixture in ``tests/conftest.py``.
"""

from logging.config import fileConfig

from alembic import context as alembic_context
from sqlalchemy import engine_from_config, pool

from al_medlit.core.config import settings
from al_medlit.core.database import Base, register_models

register_models()

config = getattr(alembic_context, "config")
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    getattr(alembic_context, "configure")(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,  # required for SQLite ALTER TABLE
    )
    with getattr(alembic_context, "begin_transaction")():
        getattr(alembic_context, "run_migrations")()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        getattr(alembic_context, "configure")(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,  # required for SQLite ALTER TABLE
        )
        with getattr(alembic_context, "begin_transaction")():
            getattr(alembic_context, "run_migrations")()


if getattr(alembic_context, "is_offline_mode")():
    run_migrations_offline()
else:
    run_migrations_online()
