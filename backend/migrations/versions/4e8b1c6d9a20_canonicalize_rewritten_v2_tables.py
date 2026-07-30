"""canonicalize table names from the rewritten v2 migration lineage

Revision ID: 4e8b1c6d9a20
Revises: e2d5a8c0f314
Create Date: 2026-07-30 00:00:00.000000
"""

from collections.abc import Iterable
from typing import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.engine import Connection
from sqlalchemy.engine.reflection import Inspector

revision: str = "4e8b1c6d9a20"
down_revision: str | None = "e2d5a8c0f314"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TABLE_RENAMES = (
    ("training_runs_v2", "training_runs"),
    ("guidelines_v2", "guidelines"),
)


def _constraint_names(inspector: Inspector, table_name: str) -> Iterable[str]:
    primary_key = inspector.get_pk_constraint(table_name).get("name")
    if primary_key:
        yield primary_key
    for getter in (
        inspector.get_unique_constraints,
        inspector.get_foreign_keys,
        inspector.get_check_constraints,
    ):
        for item in getter(table_name):
            name = item.get("name")
            if name:
                yield name


def _canonicalize_postgresql_names(
    connection: Connection,
    *,
    legacy_name: str,
    canonical_name: str,
) -> None:
    inspector = sa.inspect(connection)
    constraint_names = tuple(_constraint_names(inspector, canonical_name))
    index_names = tuple(
        index["name"]
        for index in inspector.get_indexes(canonical_name)
        if index.get("name", "").startswith(f"ix_{legacy_name}_")
    )
    quote = connection.dialect.identifier_preparer.quote_identifier
    quoted_table = quote(canonical_name)

    for old_name in constraint_names:
        if legacy_name not in old_name:
            continue
        new_name = old_name.replace(legacy_name, canonical_name)
        op.execute(
            sa.text(
                f"ALTER TABLE {quoted_table} "
                f"RENAME CONSTRAINT {quote(old_name)} TO {quote(new_name)}"
            )
        )

    for old_name in index_names:
        new_name = old_name.replace(legacy_name, canonical_name)
        op.execute(sa.text(f"ALTER INDEX {quote(old_name)} RENAME TO {quote(new_name)}"))

    legacy_sequence = f"{legacy_name}_id_seq"
    sequence = connection.execute(
        sa.text(
            """
            SELECT sequence_schema, sequence_name
            FROM information_schema.sequences
            WHERE sequence_schema = current_schema()
              AND sequence_name = :sequence_name
            """
        ),
        {"sequence_name": legacy_sequence},
    ).one_or_none()
    if sequence is not None:
        schema_name, old_name = sequence
        new_name = old_name.replace(legacy_name, canonical_name)
        op.execute(
            sa.text(
                f"ALTER SEQUENCE {quote(schema_name)}.{quote(old_name)} RENAME TO {quote(new_name)}"
            )
        )


def upgrade() -> None:
    connection = op.get_bind()
    tables = set(sa.inspect(connection).get_table_names())
    invalid_states: list[str] = []

    for legacy_name, canonical_name in TABLE_RENAMES:
        has_legacy = legacy_name in tables
        has_canonical = canonical_name in tables
        if has_legacy == has_canonical:
            state = "both exist" if has_legacy else "neither exists"
            invalid_states.append(f"{legacy_name}/{canonical_name}: {state}")

    if invalid_states:
        details = "; ".join(invalid_states)
        raise RuntimeError(
            "Cannot safely canonicalize rewritten migration tables. "
            f"{details}. Restore a valid pre-migration schema or reconcile "
            "the conflicting tables manually before retrying."
        )

    for legacy_name, canonical_name in TABLE_RENAMES:
        if legacy_name in tables:
            op.rename_table(legacy_name, canonical_name)
        if connection.dialect.name == "postgresql":
            _canonicalize_postgresql_names(
                connection,
                legacy_name=legacy_name,
                canonical_name=canonical_name,
            )


def downgrade() -> None:
    # The preceding e2d5a8c0f314 revision has existed with both legacy and
    # canonical table names, so there is no single safe schema to restore.
    # Keeping the canonical names matches the checked-in migration lineage.
    pass
