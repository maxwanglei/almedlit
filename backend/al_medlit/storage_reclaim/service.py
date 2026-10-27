"""Retry storage deletes that were abandoned after the database won.

An object delete and the deletion of the row referencing it span two systems
and cannot commit together. Every caller therefore makes the database
authoritative first and treats the storage delete as best effort; without a
follow-up the failed delete silently leaks the object. ``record_orphaned_object``
queues the key instead, and ``reclaim_orphaned_objects`` — driven by the
``al_medlit.storage.reclaim_orphaned_objects`` beat task or the eager API's
periodic maintenance loop — retries until storage confirms the object is gone.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from al_medlit.core.database import begin_savepoint
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
    logs a queueing failure and leaves the caller's transaction intact. With
    commit=False, errors flushing the caller's own pending changes propagate.
    """

    now = utc_now()
    if not commit:
        db.flush()
    try:
        with begin_savepoint(db):
            entry = (
                db.query(OrphanedStorageObject)
                .filter(OrphanedStorageObject.storage_key == storage_key)
                .with_for_update()
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
            db.flush()
        if commit:
            db.commit()
        return entry
    except Exception:
        logger.exception(
            "Failed to queue an unreferenced object for reclamation",
            extra={"storage_key": storage_key, "origin": origin},
        )
        if commit:
            db.rollback()
        return None


def reclaim_orphaned_objects(
    db: Session,
    storage: ObjectStorage,
    *,
    limit: int = 200,
    now: datetime | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> OrphanReclaimResult:
    """Delete queued objects that are due, dropping each row once storage agrees.

    Rows are locked with ``skip_locked`` so concurrent sweeps take disjoint
    batches. A row that fails again keeps its place in the queue with an
    exponentially backed-off next attempt.
    ``should_stop`` leaves unprocessed objects queued during worker shutdown;
    the caller still commits completed work and releases the batch's locks.
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
        if should_stop is not None and should_stop():
            break
        if storage_key_is_referenced(db, entry.storage_key):
            db.delete(entry)
            continue
        try:
            storage.delete(entry.storage_key)
        except Exception as exc:
            failed += 1
            entry.attempts += 1
            entry.last_error = _describe(exc)
            entry.last_attempted_at = moment
            entry.next_attempt_at = moment + _retry_delay(entry.attempts)
            log = logger.error if entry.attempts >= PERSISTENT_FAILURE_ATTEMPTS else logger.warning
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


def storage_key_is_referenced(
    db: Session,
    storage_key: str,
    *,
    include_purged_blobs: bool = True,
) -> bool:
    from al_medlit.lineage.models import LineageArtifact
    from al_medlit.model_artifacts.models import ArtifactBlob
    from al_medlit.submission.models import AnnotationSubmission

    if any(
        db.query(model.id).filter(model.storage_key == storage_key).first() is not None
        for model in (AnnotationSubmission, LineageArtifact)
    ):
        return True
    blobs = db.query(ArtifactBlob).filter(ArtifactBlob.storage_key == storage_key)
    if include_purged_blobs:
        return blobs.first() is not None
    from al_medlit.model_artifacts.service import _blob_has_live_package_reference

    return any(
        blob.status != "purged" or _blob_has_live_package_reference(db, blob.id)
        for blob in blobs.populate_existing().with_for_update()
    )
