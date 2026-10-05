"""Historical training snapshots must respect established prepared holdouts."""

import pytest
from test_training_run_execution import (
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ValidationError
from al_medlit.model_artifacts.models import ArtifactPackage, ArtifactStorageReservation
from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.preparation_schemas import TrainingPreparationRequest, TrainingSource
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workflow.services.preparation import prepare_training_dataset


def test_historical_training_split_cannot_reuse_prepared_test_items(db, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="historical-holdout")
    labels = (
        db.query(models.LabelSetVersion)
        .filter_by(dataset_version_id=scope.dataset.id, name="source-labels")
        .one()
    )
    prepared = prepare_training_dataset(
        db,
        scope.project.id,
        TrainingPreparationRequest(
            name="Established training boundaries",
            task_version_id=scope.task.id,
            idempotency_key="established-prepared-holdout",
            sources=[
                TrainingSource(
                    dataset_version_id=scope.dataset.id,
                    label_set_version_id=labels.id,
                    split_map_id=scope.split_map.id,
                )
            ],
        ),
        scope.actor,
    ).training_dataset_version
    prepared_split = db.get(models.SplitMap, prepared.split_map_id)
    [test_key] = [key for key, split in prepared_split.assignments.items() if split == "test"]
    historical_assignments = {**prepared_split.assignments, test_key: "train"}
    # Manufacture immutable rows to model snapshots created before write guards.
    historical_split = models.SplitMap(
        project_id=scope.project.id,
        dataset_version_id=prepared.dataset_version_id,
        name="Historical split with a reused holdout",
        strategy="legacy",
        assignments=historical_assignments,
        protected_splits=["test"],
        content_hash=_canonical_hash(historical_assignments),
        created_by_user_id=scope.actor.id,
    )
    db.add(historical_split)
    db.flush()
    historical_version = models.TrainingDatasetVersion(
        project_id=scope.project.id,
        name="Historical training that reused a holdout",
        dataset_version_id=prepared.dataset_version_id,
        task_version_id=scope.task.id,
        label_set_version_ids=prepared.label_set_version_ids,
        split_map_id=historical_split.id,
        composition=prepared.composition,
        preprocessing=prepared.preprocessing,
        content_hash=_canonical_hash({"historical_split_id": historical_split.id}),
        created_by_user_id=scope.actor.id,
    )
    db.add(historical_version)
    db.commit()
    run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=scope.project.id,
            registered_model_id=scope.registered.id,
            task_version_id=scope.task.id,
            training_dataset_version_id=historical_version.id,
            recipe_version_id=scope.recipe_version.id,
            environment_id=scope.environment.id,
            storage_policy_id=scope.storage_policy.id,
            idempotency_key="historical-run-with-a-reused-holdout",
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
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    with pytest.raises(ValidationError, match="protected"):
        execute_training_run(
            db,
            storage,
            training_run_id=run.id,
            trainer_registry=trainers,
        )

    assert trainer.train_calls == 0
    assert trainer.train_rows == trainer.validation_rows == ()
    assert db.query(ArtifactPackage).count() == 0
    assert storage.objects == {}
    assert db.get(models.TrainingRun, run.id).status == "failed"
    assert db.get(models.TrainingRun, run.id).output_model_version_id is None
    reservation = db.get(ArtifactStorageReservation, run.artifact_reservation_id)
    assert reservation.status == "released"
