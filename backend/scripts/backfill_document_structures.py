"""Idempotently create active structure versions for legacy documents.

Run after the structure Alembic migration has been applied::

    python -m scripts.backfill_document_structures

Individual failed documents are reported and left retryable; successful
documents are committed independently and skipped on subsequent runs.
"""

from __future__ import annotations

import argparse

from al_medlit.core.database import SessionLocal, ensure_schema_ready
from al_medlit.corpus.service import backfill_document_structures


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-id",
        action="append",
        type=int,
        dest="document_ids",
        help="Backfill only this document ID (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    ensure_schema_ready()
    db = SessionLocal()
    try:
        result = backfill_document_structures(db, document_ids=args.document_ids)
    finally:
        db.close()

    print(
        "Document structure backfill: "
        f"created={result.created}, "
        f"activated_existing={result.activated_existing}, "
        f"skipped={result.skipped}, "
        f"failed={len(result.failures)}"
    )
    if result.failures:
        for document_id, reason in result.failures.items():
            print(f"  document {document_id}: {reason}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
