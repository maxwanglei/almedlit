import asyncio
from contextlib import contextmanager
from datetime import timedelta
from threading import Event
from types import SimpleNamespace

import pytest

from al_medlit import main
from al_medlit.core.config import settings
from al_medlit.core.models import utc_now
from al_medlit.core.storage import ObjectNotFoundError
from al_medlit.storage_reclaim import maintenance
from al_medlit.storage_reclaim.models import OrphanedStorageObject
from al_medlit.storage_reclaim.service import record_orphaned_object


def test_eager_api_retries_cleanup_after_outage(
    db, testing_session_factory, object_storage, monkeypatch, caplog,
):
    key = "submissions/orphan.json"
    object_storage.put_bytes(key, b"{}")
    record_orphaned_object(db, key, origin="submission.delete")
    entry = db.query(OrphanedStorageObject).one()
    entry.next_attempt_at = utc_now() - timedelta(seconds=1)
    db.commit()
    monkeypatch.setattr(settings, "celery_task_always_eager", True)
    monkeypatch.setattr(maintenance, "SessionLocal", testing_session_factory)
    monkeypatch.setattr(maintenance, "get_object_storage", lambda: object_storage)
    monkeypatch.setattr(maintenance, "RECLAIM_INTERVAL_SECONDS", 0.01)
    sweep = maintenance.reclaim_once
    attempts = 0

    async def exercise():
        completed = asyncio.Event()
        loop = asyncio.get_running_loop()

        def interrupted_sweep(stop_requested):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary storage outage")
            sweep(stop_requested)
            loop.call_soon_threadsafe(completed.set)

        monkeypatch.setattr(maintenance, "reclaim_once", interrupted_sweep)
        async with main.lifespan(main.create_app()):
            await asyncio.wait_for(completed.wait(), timeout=5)

    asyncio.run(exercise())

    assert attempts >= 2
    assert "will retry" in caplog.text
    assert db.query(OrphanedStorageObject).count() == 0
    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(key)


def test_shutdown_finishes_active_delete_and_closes_session_before_returning(
    db, testing_session_factory, object_storage, monkeypatch,
):
    keys = ["submissions/first.json", "submissions/second.json"]
    for key in keys:
        object_storage.put_bytes(key, b"{}")
        record_orphaned_object(db, key, origin="submission.delete")
    due = utc_now() - timedelta(seconds=1)
    for entry in db.query(OrphanedStorageObject).all():
        entry.next_attempt_at = due
    db.commit()
    monkeypatch.setattr(settings, "celery_task_always_eager", True)
    monkeypatch.setattr(maintenance, "get_object_storage", lambda: object_storage)
    monkeypatch.setattr(maintenance, "RECLAIM_INTERVAL_SECONDS", 0.01)

    release_delete = Event()
    session_closed = Event()
    stop_events = []
    deleted_keys = []
    original_delete = object_storage.delete
    original_sweep = maintenance.reclaim_once

    @contextmanager
    def tracked_transaction():
        try:
            with testing_session_factory.begin() as session:
                yield session
        finally:
            session_closed.set()

    monkeypatch.setattr(
        maintenance, "SessionLocal", SimpleNamespace(begin=tracked_transaction),
    )

    def observed_sweep(stop_requested):
        stop_events.append(stop_requested)
        original_sweep(stop_requested)

    monkeypatch.setattr(maintenance, "reclaim_once", observed_sweep)

    async def exercise():
        started_delete = asyncio.Event()
        loop = asyncio.get_running_loop()

        def blocked_delete(key):
            deleted_keys.append(key)
            loop.call_soon_threadsafe(started_delete.set)
            assert release_delete.wait(timeout=5)
            original_delete(key)

        monkeypatch.setattr(object_storage, "delete", blocked_delete)
        lifespan = main.lifespan(main.create_app())
        await lifespan.__aenter__()
        shutdown = None
        try:
            await asyncio.wait_for(started_delete.wait(), timeout=5)
            shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))

            async def wait_for_stop_request():
                while not stop_events[0].is_set():
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait_for_stop_request(), timeout=5)
            assert not shutdown.done()
            assert not session_closed.is_set()
        finally:
            release_delete.set()
            if shutdown is None:
                await lifespan.__aexit__(None, None, None)
            else:
                await asyncio.wait_for(shutdown, timeout=5)
        assert session_closed.is_set()

    asyncio.run(exercise())

    assert deleted_keys == keys[:1]
    db.expire_all()
    remaining = db.query(OrphanedStorageObject).one()
    assert remaining.storage_key == keys[1]
    assert remaining.attempts == 1
    assert object_storage.get_bytes(keys[1]) == b"{}"
    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(keys[0])


@pytest.mark.parametrize("eager", [False, True])
def test_api_starts_maintenance_only_without_beat_and_stops_on_shutdown(monkeypatch, eager):
    monkeypatch.setattr(settings, "celery_task_always_eager", eager)
    started = False
    stopped = False

    async def waiting_sweep():
        nonlocal started, stopped
        started = True
        try:
            await asyncio.Event().wait()
        finally:
            stopped = True

    monkeypatch.setattr(main, "run_reclaim_loop", waiting_sweep)

    async def exercise():
        async with main.lifespan(main.create_app()):
            await asyncio.sleep(0)
            assert started is eager
            assert stopped is False
        assert stopped is eager

    asyncio.run(exercise())
