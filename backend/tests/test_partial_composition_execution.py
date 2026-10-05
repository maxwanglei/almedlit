"""Protected evaluation handles new and historical partial label snapshots."""

import pytest
from test_training_run_execution import (
    FakeEvaluator,
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ValidationError
from al_medlit.model_artifacts.models import ArtifactPackage, ArtifactStorageReservation
from al_medlit.training.evaluators.contracts import EvaluatorPluginRegistry
from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, schemas, service


def test_partial_label_composition_completes_protected_test_evaluation(db, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="partial-composition")
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            name="partially-annotated-source",
            source_kind="imported",
            labels={
                "pool-1": {"label": "negative"},
                "test-1": {"label": "positive"},
                "train-2": {"label": "negative"},
            },
        ),
        scope.actor,
    )
    composition = service.compose_training_dataset_version(
        db,
        schemas.TrainingDatasetComposeCreate(
            project_id=scope.project.id,
            name="partially-labeled-training-data",
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            input_field="text",
            label_set_version_id=labels.id,
            train_percent=80,
            validation_percent=10,
            seed=42,
        ),
        scope.actor,
        storage=storage,
    )
    run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=scope.project.id,
            registered_model_id=scope.registered.id,
            task_version_id=scope.task.id,
            training_dataset_version_id=composition["training_dataset_version"].id,
            recipe_version_id=scope.recipe_version.id,
            environment_id=scope.environment.id,
            storage_policy_id=scope.storage_policy.id,
            idempotency_key="partial-composition-with-test-evaluation",
            evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
            config={"fields": {"input_field": "text", "target_field": "label"}},
            seed=17,
            artifact_reservation_bytes=1024 * 1024,
        ),
        scope.actor,
    )
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    evaluator = FakeEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer)
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(evaluator)

    completed = execute_training_run(
        db,
        storage,
        training_run_id=run.id,
        trainer_registry=trainers,
        evaluator_registry=evaluators,
    )

    assert completed.status == "succeeded"
    assert completed.output_model_version_id is not None
    assert trainer.train_calls == evaluator.calls == 1
    assert len(trainer.train_rows) == len(trainer.validation_rows) == len(evaluator.rows) == 1
    assert {
        row["text"] for row in (*trainer.train_rows, *trainer.validation_rows, *evaluator.rows)
    } == {"pool forbidden", "test forbidden", "train excluded"}
    [evaluation] = service.list_training_run_evaluations(db, scope.project.id, completed.id)
    assert evaluation.status == "succeeded"
    assert evaluation.row_count == 1
    split_map = db.get(models.SplitMap, composition["split_map_id"])
    assert split_map.assignments["train-1"] == "pool"
    assert split_map.assignments["validation-1"] == "pool"


@pytest.mark.parametrize("evaluator_available", [True, False])
def test_historical_missing_test_labels_fail_before_training_and_publication(
    db, monkeypatch, evaluator_available
):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="historical-partial-labels")
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            name="historical-labels-without-test",
            source_kind="imported",
            labels={
                "train-1": {"label": "negative"},
                "train-2": {"label": "positive"},
                "validation-1": {"label": "negative"},
            },
        ),
        scope.actor,
    )
    training_dataset = service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=scope.project.id,
            name="historical-partially-labeled-training-data",
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            label_set_version_ids=[labels.id],
            split_map_id=scope.split_map.id,
        ),
        scope.actor,
    )
    run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=scope.project.id,
            registered_model_id=scope.registered.id,
            task_version_id=scope.task.id,
            training_dataset_version_id=training_dataset.id,
            recipe_version_id=scope.recipe_version.id,
            environment_id=scope.environment.id,
            storage_policy_id=scope.storage_policy.id,
            idempotency_key="historical-partial-labels-with-test-evaluation",
            evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
            config={"fields": {"input_field": "text", "target_field": "label"}},
            seed=17,
            artifact_reservation_bytes=1024 * 1024,
        ),
        scope.actor,
    )
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    evaluator = FakeEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer)
    evaluators = EvaluatorPluginRegistry()
    if evaluator_available:
        evaluators.register(evaluator)

    with pytest.raises(
        ValidationError,
        match="Protected test evaluation requires labels for every test item: test-1",
    ):
        execute_training_run(
            db,
            storage,
            training_run_id=run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
        )

    assert trainer.train_calls == evaluator.calls == 0
    assert db.query(ArtifactPackage).count() == 0
    assert storage.objects == {}
    assert db.get(models.TrainingRun, run.id).status == "failed"
    assert db.get(models.TrainingRun, run.id).output_model_version_id is None
    reservation = db.get(ArtifactStorageReservation, run.artifact_reservation_id)
    assert reservation.status == "released"
