"""Celery entry points for durable training orchestration."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from threading import Event, Thread

from al_medlit.celery_app import app
from al_medlit.core.database import SessionLocal
from al_medlit.core.exceptions import ValidationError
from al_medlit.core.storage import ObjectStorageError, get_object_storage
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import ArtifactStorageReservation
from al_medlit.training.runtime_profiles import runtime_queue_for_environment_class

logger = logging.getLogger(__name__)
TRAINING_HEARTBEAT_INTERVAL_SECONDS = 30.0
FEEDBACK_SCORING_HEARTBEAT_INTERVAL_SECONDS = 30.0


def _training_heartbeat_loop(training_run_id: int, stop: Event) -> None:
    from al_medlit.training.run_execution import heartbeat_training_run

    while not stop.wait(TRAINING_HEARTBEAT_INTERVAL_SECONDS):
        db = SessionLocal()
        try:
            if not heartbeat_training_run(db, training_run_id):
                return
        except Exception:
            db.rollback()
            logger.exception(
                "Could not renew the training heartbeat for run %s",
                training_run_id,
            )
        finally:
            db.close()


@contextmanager
def _training_heartbeat(training_run_id: int) -> Iterator[None]:
    stop = Event()
    thread = Thread(
        target=_training_heartbeat_loop,
        args=(training_run_id, stop),
        name=f"training-heartbeat-{training_run_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def _feedback_scoring_heartbeat_loop(feedback_run_id: int, stop: Event) -> None:
    from al_medlit.workflow.services.feedback_scoring import (
        heartbeat_feedback_scoring_run,
    )

    while not stop.wait(FEEDBACK_SCORING_HEARTBEAT_INTERVAL_SECONDS):
        db = SessionLocal()
        try:
            if not heartbeat_feedback_scoring_run(db, feedback_run_id):
                return
        except Exception:
            db.rollback()
            logger.exception(
                "Could not renew the feedback-scoring heartbeat for run %s",
                feedback_run_id,
            )
        finally:
            db.close()


@contextmanager
def _feedback_scoring_heartbeat(feedback_run_id: int) -> Iterator[None]:
    stop = Event()
    thread = Thread(
        target=_feedback_scoring_heartbeat_loop,
        args=(feedback_run_id, stop),
        name=f"feedback-scoring-heartbeat-{feedback_run_id}",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


@app.task(
    name="al_medlit.training.execute_training_run",
    acks_late=True,
    reject_on_worker_lost=True,
)
def execute_training_run_task(training_run_id: int) -> int:
    # Keep worker plugins outside the API import graph. Their optional ML
    # dependencies remain lazy until the selected plugin actually trains.
    from al_medlit.training.run_execution import execute_training_run
    from al_medlit.training.trainers import trainer_plugins

    db = SessionLocal()
    try:
        with _training_heartbeat(training_run_id):
            run = execute_training_run(
                db,
                get_object_storage(),
                training_run_id=training_run_id,
                trainer_registry=trainer_plugins,
            )
        return run.id
    finally:
        db.close()


@app.task(name="al_medlit.training.reconcile_training_runs")
def reconcile_training_runs_task() -> list[int]:
    from al_medlit.training.run_execution import reconcile_stale_training_runs

    db = SessionLocal()
    try:
        return reconcile_stale_training_runs(db)
    finally:
        db.close()


@app.task(
    name="al_medlit.workflow.execute_feedback_scoring",
    acks_late=True,
    reject_on_worker_lost=True,
)
def execute_feedback_scoring_task(feedback_run_id: int) -> int:
    from al_medlit.workflow.services.feedback_scoring import (
        execute_feedback_scoring_run,
    )

    db = SessionLocal()
    try:
        with _feedback_scoring_heartbeat(feedback_run_id):
            run = execute_feedback_scoring_run(
                db,
                get_object_storage(),
                feedback_run_id=feedback_run_id,
            )
        return run.id
    finally:
        db.close()


@app.task(name="al_medlit.workflow.reconcile_feedback_scoring")
def reconcile_feedback_scoring_task() -> dict[str, list[int]]:
    from al_medlit.workflow.services.feedback_scoring import (
        reconcile_feedback_scoring_runs,
    )

    db = SessionLocal()
    try:
        result = reconcile_feedback_scoring_runs(db)
    finally:
        db.close()
    for feedback_run_id in result["redispatch_ids"]:
        enqueue_feedback_scoring(feedback_run_id)
    return result


@app.task(
    name="al_medlit.model_artifacts.garbage_collect",
    autoretry_for=(OSError, ObjectStorageError),
    retry_backoff=True,
    max_retries=3,
)
def garbage_collect_artifact_blobs_task(
    workspace_id: int | None = None,
    limit: int = 1000,
) -> dict[str, int]:
    """Reclaim only content-addressed blobs with no live package references."""

    db = SessionLocal()
    try:
        processed_packages = 0
        after_retention_id = 0
        bounded_limit = min(max(limit, 1), 10_000)
        while True:
            payload_result = artifact_service.garbage_collect_purged_package_payloads(
                db,
                get_object_storage(),
                workspace_id=workspace_id,
                limit=limit,
                after_retention_id=after_retention_id,
            )
            db.commit()
            processed_packages += payload_result.processed_package_count
            if (
                payload_result.processed_package_count < bounded_limit
                or payload_result.last_scanned_retention_id is None
            ):
                break
            after_retention_id = payload_result.last_scanned_retention_id
        scanned = 0
        deleted = 0
        reclaimed = 0
        after_blob_id = 0
        while True:
            result = artifact_service.garbage_collect_artifact_blobs(
                db,
                get_object_storage(),
                workspace_id=workspace_id,
                limit=limit,
                after_blob_id=after_blob_id,
            )
            db.commit()
            scanned += result.scanned_blob_count
            deleted += result.deleted_blob_count
            reclaimed += result.reclaimed_bytes
            if result.scanned_blob_count < bounded_limit or result.last_scanned_blob_id is None:
                break
            after_blob_id = result.last_scanned_blob_id
        return {
            "processed_package_count": processed_packages,
            "scanned_blob_count": scanned,
            "deleted_blob_count": deleted,
            "reclaimed_bytes": reclaimed,
        }
    finally:
        db.close()


@app.task(name="al_medlit.model_artifacts.expire_reservations")
def expire_artifact_storage_reservations_task(
    workspace_limit: int = 1000,
) -> dict[str, int]:
    """Release capacity held by abandoned training launches."""

    db = SessionLocal()
    expired_count = 0
    released_bytes = 0
    effective_now = datetime.now(UTC)
    bounded_limit = min(max(workspace_limit, 1), 10_000)
    try:
        workspace_ids = [
            workspace_id
            for (workspace_id,) in (
                db.query(ArtifactStorageReservation.workspace_id)
                .filter(
                    ArtifactStorageReservation.status == "active",
                    ArtifactStorageReservation.expires_at <= effective_now,
                )
                .distinct()
                .order_by(ArtifactStorageReservation.workspace_id)
                .limit(bounded_limit)
                .all()
            )
        ]
        for workspace_id in workspace_ids:
            result = artifact_quota.expire_artifact_reservations(
                db,
                workspace_id=workspace_id,
                now=effective_now,
            )
            db.commit()
            expired_count += result.expired_count
            released_bytes += result.released_bytes
        return {
            "workspace_count": len(workspace_ids),
            "expired_count": expired_count,
            "released_bytes": released_bytes,
        }
    finally:
        db.close()


def enqueue_training_run(
    training_run_id: int,
    *,
    environment_class: str,
) -> None:
    try:
        queue = runtime_queue_for_environment_class(environment_class)
    except KeyError as exc:
        raise ValidationError(str(exc)) from exc
    execute_training_run_task.apply_async(args=(training_run_id,), queue=queue)


def enqueue_feedback_scoring(feedback_run_id: int) -> None:
    execute_feedback_scoring_task.apply_async(
        args=(feedback_run_id,),
        queue=runtime_queue_for_environment_class("classical-cpu"),
    )


def enqueue_artifact_garbage_collection(workspace_id: int) -> None:
    garbage_collect_artifact_blobs_task.apply_async(args=(workspace_id,), queue="control")
