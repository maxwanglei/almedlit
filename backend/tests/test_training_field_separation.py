import pytest
from sqlalchemy import update
from test_training_run_execution import (
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.core.exceptions import ValidationError
from al_medlit.model_artifacts.models import ArtifactPackage, ArtifactStorageReservation
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, schemas, service


@pytest.fixture(autouse=True)
def _attested_worker_image(monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)


def _recipe_config(recipe_key, input_field, target_field):
    if recipe_key.startswith("tfidf_"):
        return {"fields": {"input_field": input_field, "target_field": target_field}}
    if recipe_key.startswith("transformer_"):
        return {
            "base_model_asset_id": 1,
            "input_field": input_field,
            "target_field": target_field,
        }
    return {
        "base_model_asset_id": 1,
        "prompt_field": input_field,
        "completion_field": target_field,
        "prompt_template_version": "v1",
    }


@pytest.mark.parametrize("recipe_key", [recipe.key for recipe in training_recipes.list()])
def test_every_recipe_rejects_label_as_source_input(recipe_key):
    result = training_recipes.validate(
        recipe_key, _recipe_config(recipe_key, "answer", "answer")
    )

    assert result.valid is False
    assert result.normalized_config is None
    assert any("different" in error for error in result.errors)


@pytest.mark.parametrize("recipe_key", [recipe.key for recipe in training_recipes.list()])
def test_recipe_preserves_distinct_exact_dataset_keys(recipe_key):
    config = _recipe_config(recipe_key, "answer ", "answer")

    result = training_recipes.validate(recipe_key, config)

    assert result.valid is True
    assert result.normalized_config is not None
    fields = result.normalized_config.get("fields", result.normalized_config)
    assert fields.get("input_field", fields.get("prompt_field")) == "answer "


@pytest.mark.parametrize("recipe_key", ["tfidf_linear_regression", "causal_lm_sft_lora"])
def test_configuration_api_rejects_label_as_source_input(client, recipe_key):
    response = client.post(
        f"/api/training/recipes/{recipe_key}/validate-configuration",
        json={"config": _recipe_config(recipe_key, "answer", "answer")},
    )

    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["normalized_config"] is None


def _launch_data(scope, config):
    return schemas.TrainingRunCreate(
        project_id=scope.project.id,
        registered_model_id=scope.registered.id,
        task_version_id=scope.task.id,
        training_dataset_version_id=scope.training_dataset.id,
        recipe_version_id=scope.recipe_version.id,
        environment_id=scope.environment.id,
        storage_policy_id=scope.storage_policy.id,
        idempotency_key="invalid-field-binding",
        evaluation_plan={"splits": ["validation"], "metrics": ["accuracy"]},
        config=config,
        seed=17,
        artifact_reservation_bytes=1024 * 1024,
    )


@pytest.mark.parametrize("config_source", ["run", "recipe_defaults"])
def test_launch_rejects_effective_alias_binding(db, config_source):
    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix=config_source)
    config = {"fields": {"input_field": "text", "target_field": "text"}}
    if config_source == "recipe_defaults":
        db.execute(
            update(models.TrainingRecipeVersion.__table__)
            .where(models.TrainingRecipeVersion.id == scope.recipe_version.id)
            .values(default_config=config)
        )
        db.expire(scope.recipe_version)
        config = {}

    with pytest.raises(ValidationError, match="different"):
        service.create_training_run(db, _launch_data(scope, config), scope.actor)


@pytest.mark.parametrize("config_source", ["run", "recipe_defaults"])
def test_worker_rejects_historical_alias_binding_before_training(db, config_source):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix=config_source)
    alias_config = {"fields": {"input_field": "text", "target_field": "text"}}
    if config_source == "recipe_defaults":
        db.execute(
            update(models.TrainingRecipeVersion.__table__)
            .where(models.TrainingRecipeVersion.id == scope.recipe_version.id)
            .values(default_config=alias_config)
        )
    db.execute(
        update(models.TrainingRun.__table__)
        .where(models.TrainingRun.id == scope.run.id)
        .values(config=alias_config if config_source == "run" else {})
    )
    db.commit()
    db.expire_all()
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    with pytest.raises(ValidationError, match="different"):
        execute_training_run(
            db, storage, training_run_id=scope.run.id, trainer_registry=trainers
        )

    result = db.get(models.TrainingRun, scope.run.id)
    assert trainer.train_calls == 0, trainer.train_rows
    assert result.status == "failed"
    assert "different" in result.failure_reason
    assert result.output_model_version_id is None
    assert db.query(ArtifactPackage).count() == 0
    reservation = db.get(ArtifactStorageReservation, result.artifact_reservation_id)
    assert reservation.status == "released"
