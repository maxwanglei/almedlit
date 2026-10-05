"""Exercise prepared snapshots through the existing trainer and canonical scorer."""

from test_training_run_execution import (
    FakeProbabilityModel,
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.prediction_schemas import PredictionRunCreate
from al_medlit.workflow.preparation_schemas import TrainingPreparationRequest, TrainingSource
from al_medlit.workflow.services.feedback_scoring import (
    execute_feedback_scoring_run,
    request_feedback_run_materialization,
)
from al_medlit.workflow.services.predictions import create_prediction_run, result_rows
from al_medlit.workflow.services.preparation import prepare_training_dataset


def test_prepared_source_runs_through_training_and_independent_inference(db, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="reusable-lifecycle")
    label_set = (
        db.query(models.LabelSetVersion)
        .filter_by(dataset_version_id=scope.dataset.id, name="source-labels")
        .one()
    )
    prepared = prepare_training_dataset(
        db,
        scope.project.id,
        TrainingPreparationRequest(
            name="Reusable training",
            task_version_id=scope.task.id,
            idempotency_key="reusable-training",
            sources=[
                TrainingSource(
                    dataset_version_id=scope.dataset.id,
                    label_set_version_id=label_set.id,
                    split_map_id=scope.split_map.id,
                )
            ],
        ),
        scope.actor,
    )
    version = prepared.training_dataset_version
    assert version.dataset_version_id != scope.dataset.id
    run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=scope.project.id,
            registered_model_id=scope.registered.id,
            task_version_id=scope.task.id,
            training_dataset_version_id=version.id,
            recipe_version_id=scope.recipe_version.id,
            environment_id=scope.environment.id,
            storage_policy_id=scope.storage_policy.id,
            idempotency_key="prepared-training-run",
            evaluation_plan={"splits": ["validation"], "metrics": ["accuracy"]},
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
    registry = TrainerPluginRegistry()
    registry.register(trainer)
    completed = execute_training_run(
        db,
        storage,
        training_run_id=run.id,
        trainer_registry=registry,
    )
    assert completed.status == "succeeded"
    assert {row["text"] for row in trainer.train_rows} == {"train corrected", "train excluded"}
    assert trainer.validation_rows == ({"text": "validation visible", "label": "positive"},)
    assert db.get(models.DatasetVersion, scope.dataset.id).item_count == 5

    scope.project.settings = {"modules": ["data", "models"]}
    db.commit()
    prediction = create_prediction_run(
        db,
        scope.project.id,
        PredictionRunCreate(
            name="Source predictions",
            dataset_version_id=scope.dataset.id,
            model_version_id=completed.output_model_version_id,
            request_key="reusable-predictions",
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=prediction.id,
        actor=scope.actor,
    )
    scored = execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=prediction.id,
        model_loader=lambda _: FakeProbabilityModel(),
    )
    assert scored.status == "completed"
    results = result_rows(db, scored)
    assert len(results) == 5
    assert {row["dataset_item_id"] for row in results} == {
        item.id
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=scope.dataset.id)
    }
    assert {row["stable_key"] for row in results if row["protected"]} == {"test-1"}
    assert label_set.labels["train-1"] == {"label": "negative"}
