from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from test_training_run_execution import (
    FailingEvaluator,
    FakeEvaluator,
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import (
    ArtifactPackage,
    ArtifactPackageRetention,
    ArtifactStorageReservation,
)
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate, ArtifactPackageReferenceCreate
from al_medlit.training.evaluators.contracts import EvaluatorPluginRegistry
from al_medlit.training.evaluators.sklearn_tfidf import SklearnTfidfEvaluator
from al_medlit.training.run_execution import execute_training_run, reconcile_stale_training_runs
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, schemas, service


@pytest.fixture(autouse=True)
def attested_worker_image(monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)


def _trainer_registry():
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    registry = TrainerPluginRegistry()
    registry.register(trainer)
    return trainer, registry


class WorkerLost(BaseException):
    pass


class InterruptedEvaluator(FakeEvaluator):
    def evaluate(self, **_kwargs):
        raise WorkerLost()


@pytest.fixture
def abandoned_candidate(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="abandoned-candidate",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    trainer, trainers = _trainer_registry()
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(
        InterruptedEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer)
    )
    with pytest.raises(WorkerLost):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
        )
    db.rollback()
    [candidate] = db.query(ArtifactPackage).filter_by(project_id=scope.project.id).all()
    assert db.get(models.TrainingRun, scope.run.id).status == "running"
    assert candidate.metadata_["automatic_candidate_pin_version"] == 1
    assert candidate.retention.pinned
    return scope, storage, candidate


@pytest.mark.parametrize("metrics", ["accuracy", [True], [" "]])
def test_invalid_protected_metrics_fail_before_training_and_publication(db, metrics):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="invalid-protected-metrics",
        evaluation_plan={"splits": ["test"], "metrics": metrics},
    )
    trainer, trainers = _trainer_registry()

    with pytest.raises(ValidationError, match="Evaluation plan metrics"):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=EvaluatorPluginRegistry(),
        )

    assert trainer.train_calls == 0
    assert db.query(ArtifactPackage).filter_by(project_id=scope.project.id).count() == 0
    assert db.get(models.TrainingRun, scope.run.id).status == "failed"
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert reservation.status == "released"
    snapshot = quota.workspace_artifact_quota_snapshot(db, workspace_id=scope.workspace.id)
    assert snapshot.used_bytes == 0
    assert snapshot.reserved_bytes == 0


@pytest.mark.parametrize(
    ("metrics", "text", "reason"),
    [
        (["auroc"], "test forbidden", "not supported by this evaluator"),
        (["accuracy"], "", "must be non-empty text"),
    ],
)
def test_invalid_sklearn_holdout_input_fails_before_training(db, metrics, text, reason):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="invalid-sklearn-holdout",
        evaluation_plan={"splits": ["test"], "metrics": metrics},
    )
    db.execute(
        update(models.DatasetItem)
        .where(
            models.DatasetItem.dataset_version_id == scope.dataset.id,
            models.DatasetItem.stable_key == "test-1",
        )
        .values(payload={"text": text})
    )
    db.commit()
    trainer, trainers = _trainer_registry()

    class PredictingModel:
        def predict(self, _texts):
            return ["positive"]

    evaluators = EvaluatorPluginRegistry()
    evaluators.register(SklearnTfidfEvaluator(model_loader=lambda _path: PredictingModel()))

    with pytest.raises(ValidationError, match=reason):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
        )

    assert trainer.train_calls == 0
    assert db.query(ArtifactPackage).filter_by(project_id=scope.project.id).count() == 0
    assert db.get(models.TrainingRun, scope.run.id).status == "failed"
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert reservation.status == "released"


def test_failed_protected_evaluation_archives_candidate_and_allows_quota_reclamation(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="failed-evaluation-retention",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    trainer, trainers = _trainer_registry()
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(
        FailingEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer)
    )

    with pytest.raises(ValidationError, match="protected evaluation failed intentionally"):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
        )

    db.expire_all()
    run = db.get(models.TrainingRun, scope.run.id)
    assert run.status == "failed"
    assert run.output_model_version_id is None
    [candidate] = db.query(ArtifactPackage).filter_by(project_id=scope.project.id).all()
    assert candidate.deployable is False
    assert candidate.retention.pinned is False
    assert candidate.retention.archived_at is not None
    assert candidate.retention.purge_after is not None
    before = quota.workspace_artifact_quota_snapshot(db, workspace_id=scope.workspace.id)
    assert before.used_bytes > 0
    assert before.reserved_bytes == 0

    with pytest.raises(ConflictError, match="seven-day package purge grace"):
        artifact_service.approve_artifact_package_purge(
            db,
            package=candidate,
            actor_user_id=scope.actor.id,
            reason="Failed training candidate is no longer needed",
        )
    artifact_service.approve_artifact_package_purge(
        db,
        package=candidate,
        actor_user_id=scope.actor.id,
        reason="Failed training candidate is no longer needed",
        now=datetime.now(UTC) + timedelta(days=8),
    )
    artifact_service.garbage_collect_artifact_blobs(
        db,
        storage,
        workspace_id=scope.workspace.id,
    )
    db.commit()

    after = quota.workspace_artifact_quota_snapshot(db, workspace_id=scope.workspace.id)
    assert after.used_bytes == 0
    assert after.reserved_bytes == 0
    assert after.accounting_consistent


def test_reconciler_archives_candidate_after_worker_disappears(db, abandoned_candidate):
    scope, _storage, candidate = abandoned_candidate

    recovered = reconcile_stale_training_runs(
        db,
        now=datetime.now(UTC) + timedelta(minutes=10),
    )

    assert recovered == [scope.run.id]
    run = db.get(models.TrainingRun, scope.run.id)
    assert run.status == "failed"
    assert run.runtime_snapshot["candidate_cleanup"] == {
        "artifact_package_id": candidate.id,
        "status": "archived",
    }
    assert candidate.retention.pinned is False
    assert candidate.retention.archived_at is not None


def test_cancel_archives_published_candidate(db, abandoned_candidate):
    scope, _storage, candidate = abandoned_candidate

    cancelled = service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status="cancelled"),
    )

    assert cancelled.status == "cancelled"
    assert candidate.retention.pinned is False
    assert candidate.retention.archived_at is not None


def test_committed_legal_hold_survives_cleanup_with_cached_retention(db, abandoned_candidate):
    scope, _storage, candidate = abandoned_candidate
    cached_retention = candidate.retention
    assert cached_retention.legal_hold is False
    with Session(bind=db.get_bind()) as concurrent:
        concurrent_candidate = concurrent.get(ArtifactPackage, candidate.id)
        artifact_service.update_artifact_package_retention(
            concurrent,
            package=concurrent_candidate,
            actor_user_id=scope.actor.id,
            legal_hold=True,
        )
        concurrent.commit()

    failed = service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status="failed"),
    )

    assert failed.status == "failed"
    assert candidate.retention.legal_hold is True
    assert candidate.retention.pinned is True
    assert candidate.retention.archived_at is None
    cleanup = failed.runtime_snapshot["candidate_cleanup"]
    assert cleanup["status"] == "protected"
    assert "legal hold" in cleanup["reason"]


@pytest.mark.parametrize("same_clock_tick", [False, True])
def test_creator_repinning_candidate_is_preserved_by_automatic_cleanup(
    db, abandoned_candidate, monkeypatch, same_clock_tick
):
    scope, _storage, candidate = abandoned_candidate
    assert candidate.retention.pinned is True
    if same_clock_tick:
        retained_at = candidate.retention.created_at.replace(tzinfo=UTC)

        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return retained_at

        monkeypatch.setattr(artifact_service, "datetime", FrozenDatetime)
    artifact_service.update_artifact_package_retention(
        db,
        package=candidate,
        actor_user_id=scope.actor.id,
        pinned=True,
    )
    db.commit()

    failed = service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status="failed"),
    )

    assert failed.status == "failed"
    assert candidate.retention.pinned is True
    assert candidate.retention.archived_at is None
    cleanup = failed.runtime_snapshot["candidate_cleanup"]
    assert cleanup["status"] == "protected"
    assert "package pin" in cleanup["reason"]


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
def test_legacy_equal_timestamp_candidate_pin_stays_protected(db, terminal_status):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="legacy-candidate-pin",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    trainer, trainers = _trainer_registry()
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(
        InterruptedEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer)
    )

    def legacy_publisher(db, storage, *, data, **kwargs):
        metadata = dict(data.metadata)
        metadata.pop("automatic_candidate_pin_version", None)
        return artifact_service.publish_artifact_package(
            db,
            storage,
            data=data.model_copy(update={"metadata": metadata}),
            **kwargs,
        )

    with pytest.raises(WorkerLost):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
            artifact_publisher=legacy_publisher,
        )
    db.rollback()
    [candidate] = db.query(ArtifactPackage).filter_by(project_id=scope.project.id).all()
    assert "automatic_candidate_pin_version" not in candidate.metadata_
    retained_at = candidate.retention.created_at
    db.execute(
        update(ArtifactPackageRetention)
        .where(ArtifactPackageRetention.package_id == candidate.id)
        .values(
            pinned=True,
            updated_by_user_id=scope.actor.id,
            updated_at=retained_at,
        )
    )
    db.commit()
    assert candidate.retention.updated_at == candidate.retention.created_at

    ended = service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status=terminal_status),
    )

    assert ended.status == terminal_status
    assert candidate.retention.pinned is True
    assert candidate.retention.archived_at is None
    cleanup = ended.runtime_snapshot["candidate_cleanup"]
    assert cleanup["status"] == "protected"
    assert "package pin" in cleanup["reason"]


def test_candidate_dependency_blocks_cleanup_without_blocking_terminal_run(db, abandoned_candidate):
    scope, storage, candidate = abandoned_candidate
    artifact_service.publish_artifact_package(
        db,
        storage,
        project_id=scope.project.id,
        data=ArtifactPackageCreate(
            package_kind="report",
            package_format="json",
            references=[
                ArtifactPackageReferenceCreate(
                    target_package_id=candidate.id,
                    relationship_type="evaluates_model",
                )
            ],
        ),
        files=[
            artifact_service.PackageFileUpload(
                relative_path="report.json",
                source=b"{}",
                role="report",
            )
        ],
        actor_user_id=scope.actor.id,
    )
    db.commit()

    failed = service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status="failed"),
    )

    assert failed.status == "failed"
    assert candidate.retention.pinned is True
    assert candidate.retention.archived_at is None
    cleanup = failed.runtime_snapshot["candidate_cleanup"]
    assert cleanup["status"] == "protected"
    assert "package dependency" in cleanup["reason"]


def test_cancel_before_candidate_publication_prevents_a_late_candidate(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="cancel-before-candidate",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )

    class CancellingTrainer(FakeTrainer):
        def train(self, **kwargs):
            output = super().train(**kwargs)
            service.transition_training_run(
                db,
                scope.project.id,
                scope.run.id,
                schemas.TrainingRunTransition(status="cancelled"),
            )
            return output

    trainer = CancellingTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    with pytest.raises(
        ConflictError,
        match="reservation cannot be committed from status 'released'",
    ):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=EvaluatorPluginRegistry(),
        )

    assert db.get(models.TrainingRun, scope.run.id).status == "cancelled"
    assert db.query(ArtifactPackage).filter_by(project_id=scope.project.id).count() == 0
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert reservation.status == "released"
