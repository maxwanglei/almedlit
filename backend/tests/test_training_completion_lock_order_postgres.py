"""PostgreSQL coverage for completion after a committed training candidate."""

from __future__ import annotations

import os
import threading
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker
from test_system_administration_postgres_contention import (
    _LOCK_TIMEOUT_SECONDS,
    _THREAD_TIMEOUT_SECONDS,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_run_execution import FakeTrainer, MemoryObjectStorage, _training_run_fixture

from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import ArtifactPackage, WorkspaceArtifactQuota
from al_medlit.training import run_execution
from al_medlit.training.evaluators.contracts import EvaluatorPluginRegistry
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow.models import ModelVersion, RegisteredModel, TrainingRun

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def test_completion_waits_for_quota_before_locking_registered_model(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
) -> None:
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        scope = _training_run_fixture(
            db,
            storage=storage,
            suffix=f"completion-order-{uuid4().hex[:8]}",
            recipe_key="transformer_sequence_classification",
            plugin_key="huggingface_sequence",
            environment_class="transformer-cpu",
            with_base_model=True,
            evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
        )
        run_id = scope.run.id
        workspace_id = scope.workspace.id
        registered_model_id = scope.registered.id

    trainer = FakeTrainer(
        key="huggingface_sequence",
        recipe_key="transformer_sequence_classification",
        runtime_class="transformer-cpu",
        output_name="model.safetensors",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    candidate_committed = threading.Event()
    resume_completion = threading.Event()
    completion_at_quota = threading.Event()
    original_evaluate = run_execution._run_protected_test_evaluation
    original_locked_quota = artifact_quota._locked_quota

    def paused_evaluation(*args, **kwargs):
        result = original_evaluate(*args, **kwargs)
        assert result.status == "unsupported"
        # execute_training_run committed the candidate before this call. Its
        # later completion must reacquire quota before taking the model lock.
        candidate_committed.set()
        assert resume_completion.wait(timeout=_THREAD_TIMEOUT_SECONDS)
        return result

    def observed_locked_quota(db: Session, workspace_id: int) -> WorkspaceArtifactQuota:
        if db.info.get("completion_worker") and candidate_committed.is_set():
            completion_at_quota.set()
        return original_locked_quota(db, workspace_id)

    monkeypatch.setattr(run_execution, "_run_protected_test_evaluation", paused_evaluation)
    monkeypatch.setattr(artifact_quota, "_locked_quota", observed_locked_quota)
    results: dict[str, tuple[str, Any]] = {}

    def complete() -> None:
        with postgres_session_factory() as db:
            db.info["completion_worker"] = True
            try:
                db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
                db.execute(text(f"SET statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))
                run = run_execution.execute_training_run(
                    db,
                    storage,
                    training_run_id=run_id,
                    trainer_registry=trainers,
                    evaluator_registry=EvaluatorPluginRegistry(),
                )
                results["completion"] = ("ok", run.status)
            except Exception as exc:  # noqa: BLE001 - report thread failures below
                db.rollback()
                results["completion"] = ("error", exc)

    executor = threading.Thread(target=complete, daemon=True)
    executor.start()
    try:
        assert candidate_committed.wait(timeout=_THREAD_TIMEOUT_SECONDS), results
        with postgres_session_factory() as holder:
            holder.execute(text(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
            artifact_quota.lock_workspace_artifact_quota(holder, workspace_id=workspace_id)
            resume_completion.set()
            assert completion_at_quota.wait(timeout=_LOCK_TIMEOUT_SECONDS), results
            # A completion waiting on our quota must leave the model available.
            # Reversing those locks raises LockNotAvailable here with NOWAIT.
            holder.query(RegisteredModel).filter(
                RegisteredModel.id == registered_model_id
            ).with_for_update(nowait=True).one()
            holder.rollback()
    finally:
        resume_completion.set()
        executor.join(timeout=_THREAD_TIMEOUT_SECONDS)

    assert not executor.is_alive(), results
    assert results == {"completion": ("ok", "succeeded")}, results
    assert trainer.train_calls == 1
    with postgres_session_factory() as db:
        run = db.get(TrainingRun, run_id)
        assert run.status == "succeeded"
        assert run.runtime_snapshot["protected_test_evaluation"]["status"] == "unsupported"
        version = db.get(ModelVersion, run.output_model_version_id)
        package = db.get(ArtifactPackage, version.checkpoint_package_id)
        assert package.package_kind == "model_candidate"
        assert package.deployable is False
