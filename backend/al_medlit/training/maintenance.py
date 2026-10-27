"""Recover abandoned training runs in deployments without Celery beat."""

import asyncio
import logging
import time
from threading import Event, Thread

from al_medlit.core.database import SessionLocal
from al_medlit.training.run_execution import reconcile_stale_training_runs

logger = logging.getLogger(__name__)
RECONCILIATION_INTERVAL_SECONDS = 60
SHUTDOWN_GRACE_SECONDS = 5


def _run_eager_schedules(stop_requested: Event) -> None:
    from al_medlit.celery_app import app
    from al_medlit.training import tasks  # noqa: F401

    already_running = {
        "al_medlit.training.reconcile_training_runs",
        "al_medlit.storage.reclaim_orphaned_objects",
    }
    schedules = {
        item["task"]: float(item["schedule"])
        for item in app.conf.beat_schedule.values()
        if item["task"] not in already_running
    }
    due = {name: time.monotonic() + interval for name, interval in schedules.items()}
    while not stop_requested.wait(1):
        for name, interval in schedules.items():
            if stop_requested.is_set():
                return
            if time.monotonic() < due[name]:
                continue
            try:
                app.tasks[name].run()
            except Exception:
                logger.exception("Eager maintenance task %s failed; will retry", name)
            due[name] = time.monotonic() + interval


async def run_eager_scheduled_maintenance() -> None:
    stop_requested = Event()
    thread = Thread(
        target=_run_eager_schedules,
        args=(stop_requested,),
        name="eager-maintenance",
        daemon=True,
    )
    thread.start()
    try:
        await asyncio.Event().wait()
    finally:
        stop_requested.set()
        await asyncio.to_thread(thread.join, SHUTDOWN_GRACE_SECONDS)


def reconcile_once(stop_requested: Event) -> None:
    with SessionLocal() as db:
        reconcile_stale_training_runs(db, should_stop=stop_requested.is_set)


def _reconcile_until_stopped(stop_requested: Event) -> None:
    while not stop_requested.wait(RECONCILIATION_INTERVAL_SECONDS):
        try:
            reconcile_once(stop_requested)
        except Exception:
            logger.exception("Training reconciliation failed; will retry")


async def run_training_reconciliation_loop() -> None:
    stop_requested = Event()
    thread = Thread(
        target=_reconcile_until_stopped,
        args=(stop_requested,),
        name="training-reconciliation",
        daemon=True,
    )
    thread.start()
    try:
        await asyncio.Event().wait()
    finally:
        stop_requested.set()
        await asyncio.to_thread(thread.join, SHUTDOWN_GRACE_SECONDS)
