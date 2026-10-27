from collections.abc import Callable
from typing import cast

from sqlalchemy.orm import Session

from al_medlit.annotation.models import AnnotationCorrection
from al_medlit.co_learning.error_guideline_learning.service import upsert_pattern_from_correction
from al_medlit.core.database import SessionLocal
from al_medlit.core.events import event_bus
from al_medlit.lineage.models import CorpusSnapshot, CorpusSnapshotDocument
from al_medlit.project.models import Project
from al_medlit.workflow.models import DatasetItem, TaskVersion
from al_medlit.workflow.services.holdouts import protected_dataset_item_ids
from al_medlit.workspace.capability_service import effective_capabilities_for_workspace

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
            project = db.get(Project, correction.project_id)
            if project is None or "co_learning" not in effective_capabilities_for_workspace(
                db, project.workspace_id
            ):
                return
            if db.query(CorpusSnapshotDocument.id).join(CorpusSnapshot).filter(
                CorpusSnapshot.project_id == project.id,
                CorpusSnapshotDocument.document_id == correction.document_id,
                CorpusSnapshotDocument.split == "test",
            ).first() is not None:
                return
            items = db.query(DatasetItem).filter(
                DatasetItem.project_id == project.id,
                DatasetItem.payload["document_id"].as_integer() == correction.document_id,
            ).all()
            if items and any(
                protected_dataset_item_ids(db, items, task_id)
                for (task_id,) in db.query(TaskVersion.id).filter(
                    TaskVersion.project_id == project.id
                )
            ):
                return
            upsert_pattern_from_correction(db, correction)
    finally:
        db.close()


def register_event_handlers() -> None:
    if _state["event_handlers_registered"]:
        return

    event_bus.subscribe("annotation.correction.created", handle_annotation_correction_created)
    _state["event_handlers_registered"] = True
