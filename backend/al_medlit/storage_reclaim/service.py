"""Retry storage deletes that were abandoned after the database won.

An object delete and the deletion of the row referencing it span two systems
and cannot commit together. Every caller therefore makes the database
authoritative first and treats the storage delete as best effort; without a
follow-up the failed delete silently leaks the object. ``record_orphaned_object``
queues the key instead, and ``reclaim_orphaned_objects`` — driven by the
``al_medlit.storage.reclaim_orphaned_objects`` beat task — retries until
storage confirms the object is gone.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from al_medlit.core.models import utc_now
from al_medlit.core.storage import ObjectStorage
from al_medlit.storage_reclaim.models import OrphanedStorageObject

logger = logging.getLogger(__name__)

RETRY_BASE_SECONDS = 300
RETRY_MAX_SECONDS = 6 * 60 * 60
# Backlog rows are never dropped; this only escalates the log level so a key
# that storage keeps refusing becomes visible to alerting.
PERSISTENT_FAILURE_ATTEMPTS = 10
MAX_RECORDED_ERROR_CHARS = 2000


@dataclass(frozen=True, slots=True)
class OrphanReclaimResult:
    scanned_count: int
    reclaimed_count: int
    failed_count: int


def _retry_delay(attempts: int) -> timedelta:
    seconds = RETRY_BASE_SECONDS * 2 ** max(attempts - 1, 0)
    return timedelta(seconds=min(seconds, RETRY_MAX_SECONDS))


def _describe(error: BaseException | None) -> str | None:
    if error is None:
        return None
    return f"{type(error).__name__}: {error}"[:MAX_RECORDED_ERROR_CHARS]


def record_orphaned_object(
    db: Session,
    storage_key: str,
    *,
    origin: str,
    error: BaseException | None = None,
    commit: bool = True,
) -> OrphanedStorageObject | None:
    """Queue ``storage_key`` for a later reclaim attempt.

    Callers are already on a failure path that must not be made worse, so this
    never raises: a queueing failure is logged and the key is dropped, which is
    no worse than the leak it was trying to record.
    """

    now = utc_now()
    try:
        entry = (
            db.query(OrphanedStorageObject)
            .filter(OrphanedStorageObject.storage_key == storage_key)
            .one_or_none()
        )
        if entry is None:
            entry = OrphanedStorageObject(storage_key=storage_key, attempts=0)
            db.add(entry)
        entry.origin = origin
        entry.attempts += 1
        entry.last_error = _describe(error)
        entry.last_attempted_at = now
        entry.next_attempt_at = now + _retry_delay(entry.attempts)
        if commit:
            db.commit()
        else:
            db.flush()
        return entry
    except Exception:
        logger.exception(
            "Failed to queue an unreferenced object for reclamation",
            extra={"storage_key": storage_key, "origin": origin},
        )
        try:
            db.rollback()
        except Exception:
            logger.exception("Failed to roll back after a reclaim-queue failure")
        return None


def reclaim_orphaned_objects(
    db: Session,
    storage: ObjectStorage,
    *,
    limit: int = 200,
    now: datetime | None = None,
) -> OrphanReclaimResult:
    """Delete queued objects that are due, dropping each row once storage agrees.

    Rows are locked with ``skip_locked`` so concurrent sweeps take disjoint
    batches. A row that fails again keeps its place in the queue with an
    exponentially backed-off next attempt.
    """

    moment = now or utc_now()
    bounded_limit = min(max(limit, 1), 10_000)
    candidates = (
        db.query(OrphanedStorageObject)
        .filter(OrphanedStorageObject.next_attempt_at <= moment)
        .order_by(OrphanedStorageObject.next_attempt_at, OrphanedStorageObject.id)
        .limit(bounded_limit)
        .with_for_update(skip_locked=True)
        .all()
    )
    reclaimed = 0
    failed = 0
    for entry in candidates:
        try:
            storage.delete(entry.storage_key)
        except Exception as exc:
            failed += 1
            entry.attempts += 1
            entry.last_error = _describe(exc)
            entry.last_attempted_at = moment
            entry.next_attempt_at = moment + _retry_delay(entry.attempts)
            log = (
                logger.error
                if entry.attempts >= PERSISTENT_FAILURE_ATTEMPTS
                else logger.warning
            )
            log(
                "Could not reclaim unreferenced object %s after %s attempts: %s",
                entry.storage_key,
                entry.attempts,
                exc,
            )
            continue
        reclaimed += 1
        db.delete(entry)
    db.flush()
    return OrphanReclaimResult(
        scanned_count=len(candidates),
        reclaimed_count=reclaimed,
        failed_count=failed,
    )
