"""Opt-in PostgreSQL coverage for training publication and worker lease locks."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, text
from sqlalchemy.orm import Session, sessionmaker
from test_system_administration_postgres_contention import (
    _LOCK_TIMEOUT_SECONDS,
    _THREAD_TIMEOUT_SECONDS,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_run_execution import (
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.storage import StoredObject
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactStorageReservation,
    WorkspaceArtifactQuota,
)
from al_medlit.training.run_execution import execute_training_run, heartbeat_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow.models import ModelVersion, TrainingRun

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


class PausedPublicationStorage(MemoryObjectStorage):
    """Keep the publisher's quota lock while the heartbeat starts."""

    def __init__(self) -> None:
        super().__init__()
        self.upload_started = threading.Event()
        self.resume_upload = threading.Event()

    def put_stream(
        self,
        key: str,
        stream,
        *,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = 1024 * 1024,
    ) -> StoredObject:
        if not self.upload_started.is_set():
            self.upload_started.set()
            assert self.resume_upload.wait(timeout=_THREAD_TIMEOUT_SECONDS), (
                "The coordinator did not release the publication upload"
            )
        return super().put_stream(
            key,
            stream,
            length=length,
            content_type=content_type,
            chunk_size=chunk_size,
        )


def test_training_publication_and_heartbeat_do_not_deadlock(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
) -> None:
    # The attestation fixture in test_training_run_execution does not apply to
    # callers importing that module's scope builder and fake trainer.
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = PausedPublicationStorage()
    with postgres_session_factory() as db:
        scope = _training_run_fixture(
            db,
            storage=storage,
            suffix=f"publication-race-{uuid4().hex[:8]}",
        )
        run_id = scope.run.id
        workspace_id = scope.workspace.id
        reservation_id = scope.run.artifact_reservation_id

    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    heartbeat_at_quota = threading.Event()
    original_locked_quota = artifact_quota._locked_quota

    def observed_locked_quota(db: Session, workspace_id: int) -> WorkspaceArtifactQuota:
        if db.info.get("publication_race_role") == "heartbeat":
            # The original heartbeat held the run lock at this point. With the
            # fixed order, it waits for quota before acquiring the run lock.
            heartbeat_at_quota.set()
        return original_locked_quota(db, workspace_id)

    monkeypatch.setattr(artifact_quota, "_locked_quota", observed_locked_quota)
    results: dict[str, tuple[str, Any]] = {}

    def worker(role: str, operation: Callable[[Session], Any]) -> None:
        with postgres_session_factory() as db:
            db.info["publication_race_role"] = role
            try:
                # Claiming a run commits its transaction, so session timeouts
                # must survive that commit rather than use SET LOCAL.
                db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
                db.execute(text(f"SET statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))
                results[role] = ("ok", operation(db))
            except Exception as exc:  # noqa: BLE001 - assert concurrent failures below
                db.rollback()
                results[role] = ("error", exc)

    def publish(db: Session) -> str:
        completed = execute_training_run(
            db,
            storage,
            training_run_id=run_id,
            trainer_registry=trainers,
        )
        return completed.status

    publisher = threading.Thread(target=worker, args=("publisher", publish), daemon=True)
    heartbeat = threading.Thread(
        target=worker,
        args=("heartbeat", lambda db: heartbeat_training_run(db, run_id)),
        daemon=True,
    )
    publisher.start()
    try:
        assert storage.upload_started.wait(timeout=_LOCK_TIMEOUT_SECONDS), (
            f"The publisher did not reach storage upload: {results!r}"
        )
        heartbeat.start()
        assert heartbeat_at_quota.wait(timeout=_LOCK_TIMEOUT_SECONDS), (
            f"The heartbeat did not reach its quota lock: {results!r}"
        )
    finally:
        storage.resume_upload.set()
        deadline = time.monotonic() + _THREAD_TIMEOUT_SECONDS
        publisher.join(max(0, deadline - time.monotonic()))
        if heartbeat.ident is not None:
            heartbeat.join(max(0, deadline - time.monotonic()))

    assert not publisher.is_alive() and not heartbeat.is_alive(), (
        f"Concurrent training publication did not finish: {results!r}"
    )
    assert results == {
        "publisher": ("ok", "succeeded"),
        "heartbeat": ("ok", False),
    }, results
    assert trainer.train_calls == 1

    with postgres_session_factory() as db:
        run = db.get(TrainingRun, run_id)
        assert run.status == "succeeded"
        assert run.failure_code is None
        model_version = db.get(ModelVersion, run.output_model_version_id)
        package = db.get(ArtifactPackage, model_version.checkpoint_package_id)
        assert package.deployable is True
        reservation = db.get(ArtifactStorageReservation, reservation_id)
        assert reservation.status == "committed"
        assert reservation.artifact_package_id == package.id
        quota = (
            db.query(WorkspaceArtifactQuota)
            .filter(WorkspaceArtifactQuota.workspace_id == workspace_id)
            .one()
        )
        physical_bytes = db.scalar(
            db.query(func.coalesce(func.sum(ArtifactBlob.size_bytes), 0))
            .filter(
                ArtifactBlob.workspace_id == workspace_id,
                ArtifactBlob.status == "ready",
            )
            .statement
        )
        assert quota.reserved_bytes == 0
        assert quota.used_bytes == reservation.committed_bytes == physical_bytes
        assert physical_bytes > 0
