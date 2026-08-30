"""The reclaim queue that keeps failed storage deletes from leaking objects."""

from datetime import UTC, timedelta

from al_medlit.core.models import utc_now
from al_medlit.storage_reclaim.models import OrphanedStorageObject
from al_medlit.storage_reclaim.service import (
    RETRY_BASE_SECONDS,
    reclaim_orphaned_objects,
    record_orphaned_object,
)


def _aware(moment):
    """Normalize a timestamp the SQLite test fixture hands back without a zone."""

    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _queue(db, storage, key: str, *, origin: str = "submission.delete") -> None:
    storage.put_bytes(key, b"{}", content_type="application/json")
    record_orphaned_object(db, key, origin=origin, error=RuntimeError("boom"))


def test_record_orphaned_object_captures_origin_and_backoff(db, object_storage):
    before = utc_now()
    _queue(db, object_storage, "objects/one.json")

    entry = db.query(OrphanedStorageObject).one()
    assert entry.storage_key == "objects/one.json"
    assert entry.origin == "submission.delete"
    assert entry.attempts == 1
    assert entry.last_error == "RuntimeError: boom"
    assert _aware(entry.next_attempt_at) >= before + timedelta(seconds=RETRY_BASE_SECONDS)


def test_recording_the_same_key_twice_keeps_one_row(db, object_storage):
    _queue(db, object_storage, "objects/one.json")
    record_orphaned_object(
        db,
        "objects/one.json",
        origin="submission.create_rollback",
        error=RuntimeError("again"),
    )

    entry = db.query(OrphanedStorageObject).one()
    assert entry.attempts == 2
    assert entry.origin == "submission.create_rollback"


def test_reclaim_deletes_the_object_and_drops_the_row(db, object_storage):
    from al_medlit.core.storage import ObjectNotFoundError

    _queue(db, object_storage, "objects/one.json")
    due = utc_now() + timedelta(seconds=RETRY_BASE_SECONDS + 1)

    result = reclaim_orphaned_objects(db, object_storage, now=due)
    db.commit()

    assert (result.scanned_count, result.reclaimed_count, result.failed_count) == (1, 1, 0)
    assert db.query(OrphanedStorageObject).count() == 0
    try:
        object_storage.get_bytes("objects/one.json")
    except ObjectNotFoundError:
        pass
    else:  # pragma: no cover - only reached on a regression
        raise AssertionError("the reclaimed object is still readable")


def test_reclaim_skips_entries_that_are_not_due_yet(db, object_storage):
    _queue(db, object_storage, "objects/one.json")

    result = reclaim_orphaned_objects(db, object_storage)

    assert result.scanned_count == 0
    assert db.query(OrphanedStorageObject).count() == 1
    assert object_storage.get_bytes("objects/one.json")


def test_reclaim_keeps_and_backs_off_a_key_storage_still_refuses(
    db,
    object_storage,
    monkeypatch,
):
    _queue(db, object_storage, "objects/one.json")
    due = utc_now() + timedelta(seconds=RETRY_BASE_SECONDS + 1)

    def fail_delete(_key):
        raise RuntimeError("storage still down")

    monkeypatch.setattr(object_storage, "delete", fail_delete)
    result = reclaim_orphaned_objects(db, object_storage, now=due)
    db.commit()

    assert (result.reclaimed_count, result.failed_count) == (0, 1)
    entry = db.query(OrphanedStorageObject).one()
    assert entry.attempts == 2
    assert entry.last_error == "RuntimeError: storage still down"
    assert _aware(entry.next_attempt_at) > due
    assert object_storage.get_bytes("objects/one.json")


def test_reclaim_processes_a_bounded_batch(db, object_storage):
    for index in range(3):
        _queue(db, object_storage, f"objects/{index}.json")
    due = utc_now() + timedelta(seconds=RETRY_BASE_SECONDS + 1)

    result = reclaim_orphaned_objects(db, object_storage, limit=2, now=due)
    db.commit()

    assert result.reclaimed_count == 2
    assert db.query(OrphanedStorageObject).count() == 1
