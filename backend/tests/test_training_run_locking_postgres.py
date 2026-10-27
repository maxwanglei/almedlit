"""PostgreSQL races between publication and stale run mutations."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
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
from test_training_publication_postgres_contention import PausedPublicationStorage
from test_training_run_execution import (
    FakeEvaluator,
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ConflictError
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactStorageReservation,
    WorkspaceArtifactQuota,
)
from al_medlit.training.evaluators.contracts import EvaluatorPluginRegistry
from al_medlit.training.run_execution import (
    execute_training_run,
    heartbeat_training_run,
    reconcile_stale_training_runs,
)
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import schemas, service
from al_medlit.workflow.models import ModelVersion, TrainingRun

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


@pytest.mark.parametrize("mutation", ["reconcile", "cancel"])
def test_publication_refreshes_waiting_run_mutations(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    mutation: str,
) -> None:
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = PausedPublicationStorage()
    with postgres_session_factory() as db:
        scope = _training_run_fixture(
            db,
            storage=storage,
            suffix=f"publication-{mutation}-{uuid4().hex[:8]}",
        )
        run_id = scope.run.id
        project_id = scope.project.id
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
    mutation_at_quota = threading.Event()
    original_locked_quota = artifact_quota._locked_quota

    def observed_locked_quota(db: Session, workspace_id: int) -> WorkspaceArtifactQuota:
        if db.info.get("publication_race_role") == "mutation":
            mutation_at_quota.set()
        return original_locked_quota(db, workspace_id)

    monkeypatch.setattr(artifact_quota, "_locked_quota", observed_locked_quota)
    results: dict[str, tuple[str, Any]] = {}

    def worker(role: str, operation: Callable[[Session], Any]) -> None:
        with postgres_session_factory() as db:
            db.info["publication_race_role"] = role
            try:
                db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
                db.execute(text(f"SET statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))
                results[role] = ("ok", operation(db))
                db.commit()
            except ConflictError as exc:
                db.rollback()
                results[role] = ("conflict", str(exc))
            except Exception as exc:  # noqa: BLE001 - assert concurrent failures below
                db.rollback()
                results[role] = ("error", exc)

    def publish(db: Session) -> str:
        return execute_training_run(
            db,
            storage,
            training_run_id=run_id,
            trainer_registry=trainers,
        ).status

    def mutate(db: Session) -> Any:
        # Keep a strong reference in the identity map: a locking SELECT must
        # refresh this running state after publication commits succeeded.
        preloaded = db.get(TrainingRun, run_id)
        assert preloaded.status == "running"
        if mutation == "reconcile":
            return reconcile_stale_training_runs(
                db,
                now=datetime.now(UTC) + timedelta(minutes=10),
                stale_after_seconds=300,
            )
        return service.transition_training_run(
            db,
            project_id,
            run_id,
            schemas.TrainingRunTransition(status="cancelled"),
        ).status

    publisher = threading.Thread(target=worker, args=("publisher", publish), daemon=True)
    mutator = threading.Thread(target=worker, args=("mutation", mutate), daemon=True)
    publisher.start()
    try:
        assert storage.upload_started.wait(timeout=_LOCK_TIMEOUT_SECONDS), (
            f"The publisher did not reach storage upload: {results!r}"
        )
        mutator.start()
        assert mutation_at_quota.wait(timeout=_LOCK_TIMEOUT_SECONDS), (
            f"The mutation did not reach its quota lock: {results!r}"
        )
    finally:
        storage.resume_upload.set()
        deadline = time.monotonic() + _THREAD_TIMEOUT_SECONDS
        publisher.join(max(0, deadline - time.monotonic()))
        if mutator.ident is not None:
            mutator.join(max(0, deadline - time.monotonic()))

    assert not publisher.is_alive() and not mutator.is_alive(), results
    expected_mutation = (
        ("ok", [])
        if mutation == "reconcile"
        else ("conflict", "Cannot transition training run from succeeded to cancelled")
    )
    assert results == {
        "publisher": ("ok", "succeeded"),
        "mutation": expected_mutation,
    }, results
    assert trainer.train_calls == 1

    with postgres_session_factory() as db:
        run = db.get(TrainingRun, run_id)
        assert run.status == "succeeded"
        assert run.failure_code is None
        assert "worker_recovery" not in run.runtime_snapshot
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


class _HeartbeatDuringEvaluation(FakeEvaluator):
    def __init__(self, *, during_evaluation: Callable[[], None], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.during_evaluation = during_evaluation

    def evaluate(self, **kwargs: Any):
        self.during_evaluation()
        return super().evaluate(**kwargs)


def test_heartbeat_keeps_run_alive_after_candidate_publication(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
) -> None:
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        scope = _training_run_fixture(
            db,
            storage=storage,
            suffix=f"evaluation-heartbeat-{uuid4().hex[:8]}",
            evaluation_plan={"splits": ["validation", "test"], "metrics": ["accuracy"]},
        )
        run_id = scope.run.id
        reservation_id = scope.run.artifact_reservation_id

    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    results: dict[str, Any] = {}

    def heartbeat_then_reconcile() -> None:
        with postgres_session_factory() as db:
            db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
            reservation = db.get(ArtifactStorageReservation, reservation_id)
            results["reservation"] = reservation.status
            claimed_at = db.get(TrainingRun, run_id).started_at
            results["heartbeat"] = heartbeat_training_run(
                db,
                run_id,
                now=claimed_at + timedelta(seconds=290),
            )
        with postgres_session_factory() as db:
            db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
            results["reconciled"] = reconcile_stale_training_runs(
                db,
                now=claimed_at + timedelta(seconds=400),
                stale_after_seconds=300,
            )

    evaluators = EvaluatorPluginRegistry()
    evaluators.register(
        _HeartbeatDuringEvaluation(
            during_evaluation=heartbeat_then_reconcile,
            recipe_key="tfidf_logistic_regression",
            trainer=trainer,
        )
    )
    with postgres_session_factory() as db:
        db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
        try:
            results["executor"] = (
                "ok",
                execute_training_run(
                    db,
                    storage,
                    training_run_id=run_id,
                    trainer_registry=trainers,
                    evaluator_registry=evaluators,
                ).status,
            )
        except ConflictError as exc:
            results["executor"] = ("conflict", str(exc))

    assert results == {
        "reservation": "active",
        "heartbeat": True,
        "reconciled": [],
        "executor": ("ok", "succeeded"),
    }, results
    with postgres_session_factory() as db:
        run = db.get(TrainingRun, run_id)
        assert run.status == "succeeded"
        assert "worker_recovery" not in run.runtime_snapshot
        assert run.runtime_snapshot["protected_test_evaluation"]["status"] == "succeeded"
        reservation = db.get(ArtifactStorageReservation, reservation_id)
        assert reservation.status == "committed"
        version = db.get(ModelVersion, run.output_model_version_id)
        candidate = db.get(ArtifactPackage, reservation.artifact_package_id)
        assert candidate.package_kind == "model_candidate"
        assert candidate.retention.pinned is False
        promoted = db.get(ArtifactPackage, version.checkpoint_package_id)
        assert any(
            reference.target_package_id == candidate.id
            and reference.relationship_type == "promoted_from_candidate"
            for reference in promoted.outgoing_references
        )
