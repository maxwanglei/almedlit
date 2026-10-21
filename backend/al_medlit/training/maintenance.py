"""Recover abandoned training runs in deployments without Celery beat."""

import asyncio
import logging
from threading import Event, Thread

from al_medlit.core.database import SessionLocal
from al_medlit.training.run_execution import reconcile_stale_training_runs

logger = logging.getLogger(__name__)
RECONCILIATION_INTERVAL_SECONDS = 60
SHUTDOWN_GRACE_SECONDS = 5


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
