"""Periodic orphan cleanup for deployments without a Celery scheduler."""

import asyncio
import logging
from threading import Event

from al_medlit.core.database import SessionLocal
from al_medlit.core.storage import get_object_storage
from al_medlit.storage_reclaim.service import reclaim_orphaned_objects

logger = logging.getLogger(__name__)
RECLAIM_INTERVAL_SECONDS = 15 * 60


def reclaim_once(stop_requested: Event | None = None) -> None:
    # The background thread owns its session and transaction. Request sessions
    # must never be shared with this sweep.
    with SessionLocal.begin() as db:
        reclaim_orphaned_objects(
            db,
            get_object_storage(),
            should_stop=stop_requested.is_set if stop_requested is not None else None,
        )


async def run_reclaim_loop() -> None:
    """Sweep every fifteen minutes; a transient outage leaves the loop alive."""
    stop_requested = Event()
    while True:
        # Defer the first sweep too: API startup must not depend on storage
        # availability, and the durable backlog survives process restarts.
        await asyncio.sleep(RECLAIM_INTERVAL_SECONDS)
        sweep = asyncio.create_task(asyncio.to_thread(reclaim_once, stop_requested))
        try:
            await asyncio.shield(sweep)
        except asyncio.CancelledError:
            # Cancelling to_thread cannot stop its synchronous worker. Finish
            # the current delete and close its transaction before API shutdown,
            # while leaving the rest of the batch queued for another process.
            stop_requested.set()
            while not sweep.done():
                try:
                    await asyncio.shield(sweep)
                except asyncio.CancelledError:
                    # Repeated shutdown requests must not abandon the session.
                    continue
                except Exception:
                    break
            try:
                sweep.result()
            except Exception:
                logger.exception("Orphaned-object reclamation failed during shutdown")
            raise
        except Exception:
            logger.exception("Periodic orphaned-object reclamation failed; will retry")
