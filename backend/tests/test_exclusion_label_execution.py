"""Historical exclusion masks must fail before reaching the training plugin."""

import pytest
from sqlalchemy import update
from test_training_run_execution import (
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ValidationError
from al_medlit.model_artifacts.models import ArtifactPackage, ArtifactStorageReservation
from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models


@pytest.mark.parametrize("policy", ["replace", "inherit"])
def test_worker_rejects_historical_composition_that_promotes_exclusion_masks(
    db, monkeypatch, policy
):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix=f"exclusion-mask-{policy}")
    label_sets = [
        db.get(models.LabelSetVersion, label_set_id)
        for label_set_id in scope.training_dataset.label_set_version_ids
    ]
    exclusion = next(label_set for label_set in label_sets if label_set.source_kind == "derived")
    assert exclusion.composition_policy == "exclude"
    assert exclusion.labels == {"train-2": True}
    imported, corrections = [label_set for label_set in label_sets if label_set is not exclusion]
    historical_composition = [
        {"label_set_version_id": exclusion.id, "policy": policy},
        {"label_set_version_id": imported.id, "policy": "inherit"},
        {"label_set_version_id": corrections.id, "policy": "replace"},
    ]
    # Seed a snapshot saved before the policy restriction existed. A SQL update
    # represents historical persisted data without relaxing ORM immutability.
    # Placing the mask before inherited labels would turn True into a training
    # target for either override if execution trusted the per-step policy.
    db.execute(
        update(models.TrainingDatasetVersion)
        .where(models.TrainingDatasetVersion.id == scope.training_dataset.id)
        .values(composition=historical_composition)
    )
    db.commit()
    db.expire_all()

    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    registry = TrainerPluginRegistry()
    registry.register(trainer)
    packages_before = db.query(ArtifactPackage).count()

    with pytest.raises(ValidationError, match="[Ee]xclu|[Dd]erived"):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=registry,
        )

    run = db.get(models.TrainingRun, scope.run.id)
    reservation = db.get(ArtifactStorageReservation, run.artifact_reservation_id)
    assert run.status == "failed"
    assert run.output_model_version_id is None
    assert reservation.status == "released"
    assert trainer.train_calls == 0
    assert trainer.train_rows == trainer.validation_rows == ()
    assert db.query(models.ModelVersion).count() == 0
    assert db.query(ArtifactPackage).count() == packages_before
    assert storage.objects == {}
