import asyncio
from datetime import UTC, datetime, timedelta
from threading import Event

import pytest

from al_medlit import main
from al_medlit.core.config import settings
from al_medlit.training import maintenance
from al_medlit.workflow import models


@pytest.mark.parametrize("eager", [False, True])
def test_api_reconciles_training_without_beat_and_stops_on_shutdown(monkeypatch, eager):
    monkeypatch.setattr(settings, "celery_task_always_eager", eager)
    started = False
    stopped = False

    async def waiting_reconciler():
        nonlocal started, stopped
        started = True
        try:
            await asyncio.Event().wait()
        finally:
            stopped = True

    async def waiting_reclaimer():
        await asyncio.Event().wait()

    monkeypatch.setattr(
        main, "run_training_reconciliation_loop", waiting_reconciler, raising=False
    )
    monkeypatch.setattr(main, "run_reclaim_loop", waiting_reclaimer)

    async def check():
        async with main.lifespan(main.create_app()):
            await asyncio.sleep(0)
            assert started is eager
        assert stopped is eager

    asyncio.run(check())


def test_eager_api_recovers_abandoned_queue_after_transient_outage(
    db, testing_session_factory, monkeypatch
):
    from test_training_run_execution import MemoryObjectStorage, _training_run_fixture

    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix="eager-recovery")
    scope.run.created_at = datetime.now(UTC) - timedelta(days=8)
    db.commit()
    run_id = scope.run.id
    monkeypatch.setattr(settings, "celery_task_always_eager", True)
    monkeypatch.setattr(maintenance, "SessionLocal", testing_session_factory)
    monkeypatch.setattr(maintenance, "RECONCILIATION_INTERVAL_SECONDS", 0.01)
    recovered = Event()
    attempts = []
    reconcile = maintenance.reconcile_stale_training_runs

    def interrupted_reconcile(session, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise OSError("Database temporarily unavailable")
        result = reconcile(session, **kwargs)
        if run_id in result:
            recovered.set()
        return result

    monkeypatch.setattr(maintenance, "reconcile_stale_training_runs", interrupted_reconcile)

    async def check():
        async with main.lifespan(main.create_app()):
            async with asyncio.timeout(5):
                while not recovered.is_set():
                    await asyncio.sleep(0.01)

    asyncio.run(check())
    db.expire_all()
    assert len(attempts) >= 2
    assert db.get(models.TrainingRun, run_id).status == "failed"


def test_shutdown_is_bounded_when_training_reconciliation_is_stalled(monkeypatch):
    started = Event()
    released = Event()
    monkeypatch.setattr(maintenance, "RECONCILIATION_INTERVAL_SECONDS", 0.001)
    monkeypatch.setattr(maintenance, "SHUTDOWN_GRACE_SECONDS", 0.01, raising=False)

    def blocked_reconciliation(*_args):
        started.set()
        released.wait(5)

    monkeypatch.setattr(maintenance, "reconcile_once", blocked_reconciliation)

    async def check():
        task = asyncio.create_task(maintenance.run_training_reconciliation_loop())
        try:
            async with asyncio.timeout(2):
                while not started.is_set():
                    await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.1)
            assert task.done(), "Shutdown still waits for the blocked database sweep"
        finally:
            released.set()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(check())


@pytest.mark.parametrize("status", ["queued", "running"])
def test_shutdown_after_lock_stops_before_training_mutation(db, monkeypatch, status):
    from test_training_run_execution import MemoryObjectStorage, _training_run_fixture

    from al_medlit.model_artifacts.models import ArtifactStorageReservation
    from al_medlit.training.run_execution import _claim_training_run, heartbeat_training_run
    from al_medlit.workflow import service

    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix=f"shutdown-{status}")
    if status == "queued":
        scope.run.created_at = datetime.now(UTC) - timedelta(days=8)
        db.commit()
    else:
        _claim_training_run(db, scope.run.id)
        heartbeat_training_run(db, scope.run.id, now=datetime.now(UTC) - timedelta(minutes=10))
    stop_requested = Event()
    lock_run = service.lock_training_run_for_mutation

    def stop_after_lock(*args, **kwargs):
        result = lock_run(*args, **kwargs)
        stop_requested.set()
        return result

    monkeypatch.setattr(service, "lock_training_run_for_mutation", stop_after_lock)

    assert maintenance.reconcile_stale_training_runs(db, should_stop=stop_requested.is_set) == []
    db.expire_all()
    assert db.get(models.TrainingRun, scope.run.id).status == status
    assert db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id).status == "active"
