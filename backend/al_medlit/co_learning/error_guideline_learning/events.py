from collections.abc import Callable
from typing import cast

from sqlalchemy.orm import Session

from al_medlit.annotation.models import AnnotationCorrection
from al_medlit.co_learning.error_guideline_learning.service import upsert_pattern_from_correction
from al_medlit.core.database import SessionLocal
from al_medlit.core.events import event_bus

SessionFactory = Callable[[], Session]

_state: dict[str, object] = {
    "session_factory": SessionLocal,
    "event_handlers_registered": False,
}


def set_event_session_factory(session_factory: SessionFactory) -> None:
    _state["session_factory"] = session_factory


def reset_event_session_factory() -> None:
    set_event_session_factory(SessionLocal)


def handle_annotation_correction_created(payload: dict) -> None:
    """Event handler that turns a correction into an error pattern.

    Currently synchronous and in-process. Once the lab profile's Celery
    workers are wired in, this becomes a task dispatched onto the ``ml`` queue.
    """

    correction_id = payload.get("correction_id")
    if correction_id is None:
        return

    session_factory = cast(SessionFactory, _state["session_factory"])
    db = session_factory()
    try:
        correction = db.get(AnnotationCorrection, correction_id)
        if correction is not None:
            upsert_pattern_from_correction(db, correction)
    finally:
        db.close()


def register_event_handlers() -> None:
    if _state["event_handlers_registered"]:
        return

    event_bus.subscribe("annotation.correction.created", handle_annotation_correction_created)
    _state["event_handlers_registered"] = True
