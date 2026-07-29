from collections.abc import Generator

from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from al_medlit.core.config import settings


class Base(DeclarativeBase):
    pass


# PostgreSQL is the production target. SQLite is only used in the unit-test
# fixture (tests/conftest.py), which builds its own engine — so we don't need
# SQLite-specific connect_args here.
engine = create_engine(settings.database_url, echo=settings.echo_sql)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def register_models() -> None:
    """Import model modules so SQLAlchemy metadata is fully registered."""

    from al_medlit.annotation import models as _annotation_models  # noqa: F401
    from al_medlit.auth import models as _auth_models  # noqa: F401
    from al_medlit.co_learning.error_guideline_learning import models as _egl_models  # noqa: F401
    from al_medlit.corpus import models as _corpus_models  # noqa: F401
    from al_medlit.evidence import models as _evidence_models  # noqa: F401
    from al_medlit.guideline import models as _guideline_models  # noqa: F401
    from al_medlit.inference import models as _inference_models  # noqa: F401
    from al_medlit.lineage import models as _lineage_models  # noqa: F401
    from al_medlit.model_artifacts import models as _model_artifact_models  # noqa: F401
    from al_medlit.project import models as _project_models  # noqa: F401
    from al_medlit.submission import models as _submission_models  # noqa: F401
    from al_medlit.training import models as _training_models  # noqa: F401
    from al_medlit.workflow import models as _workflow_models  # noqa: F401
    from al_medlit.workspace import models as _workspace_models  # noqa: F401


def init_db(bind: Engine | None = None) -> None:
    """Create all tables for explicit local bootstrap and tests.

    Application startup should not call this directly. Runtime schema management
    belongs to Alembic migrations; this helper remains for explicit setup paths
    such as tests and local seeding.
    """

    register_models()
    Base.metadata.create_all(bind=bind or engine)


def ensure_schema_ready(bind: Engine | None = None) -> None:
    """Verify that the configured database already has the expected schema.

    Runtime startup should not create tables implicitly. Explicit setup paths,
    such as seeding, can use this guard to fail fast with a clear migration
    instruction instead of triggering lower-level database errors later.
    """

    inspection_engine = bind or engine
    register_models()

    existing_tables = set(inspect(inspection_engine).get_table_names())
    expected_tables = set(Base.metadata.tables)
    missing_tables = sorted(expected_tables - existing_tables)

    if missing_tables:
        missing = ", ".join(missing_tables)
        raise RuntimeError(
            "Database schema is not initialized. Run `alembic upgrade head` "
            f"before continuing. Missing tables: {missing}"
        )
