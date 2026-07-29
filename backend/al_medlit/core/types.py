from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON, TypeDecorator


class JSONType(TypeDecorator):
    """Portable JSON column.

    Maps to JSONB on PostgreSQL (production) and to generic JSON on every
    other dialect (used by the SQLite-backed unit-test fixture). Application
    code must never reach for ``JSONB`` directly — use ``JSONType`` so the
    test path stays green.
    """

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect) -> Any:
        return value

    def process_result_value(self, value: Any, dialect) -> Any:
        return value
