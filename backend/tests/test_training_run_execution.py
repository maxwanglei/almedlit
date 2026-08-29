"""Execution coverage for canonical immutable training runs."""

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import ObjectNotFoundError, StoredObject
from al_medlit.lineage.models import ImmutableRecordError
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import (
    ArtifactPackage,
    ArtifactStorageReservation,
    BaseModelAsset,
    BaseModelAssetState,
)
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate
from al_medlit.project.models import Project
from al_medlit.training.artifact_loading import (
    load_safe_skops_model,
    stage_artifact_package,
)
from al_medlit.training.evaluators.contracts import (
    EvaluationInput,
    EvaluationOutput,
    EvaluatorPluginRegistry,
)
from al_medlit.training.evaluators.sklearn_tfidf import SklearnTfidfEvaluator
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.run_execution import (
    execute_training_run,
    heartbeat_training_run,
    reconcile_stale_training_runs,
)
from al_medlit.training.runtime_profiles import RUNTIME_PROFILES, RuntimeReadinessReport
from al_medlit.training.tasks import enqueue_training_run
from al_medlit.training.trainers.contracts import (
    TrainerPluginRegistry,
    TrainerPreflight,
    TrainingInput,
    TrainingOutput,
    TrainingPlan,
)
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.routes import learning as learning_router
from al_medlit.workflow.routes import training as workflow_router
from al_medlit.workflow.services.feedback_scoring import (
    _prediction_output,
    execute_feedback_scoring_run,
    heartbeat_feedback_scoring_run,
    reconcile_feedback_scoring_runs,
    request_feedback_run_materialization,
)
from al_medlit.workspace import capability_service
from al_medlit.workspace import service as workspace_service


@pytest.fixture(autouse=True)
def _attested_worker_image(monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)


class MemoryObjectStorage:
    backend_name = "minio"
    encryption_mode = "none"
    encryption_key_id = None

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.objects[key] = bytes(data)

    def get_bytes(self, key: str) -> bytes:
        try:
            return self.objects[key]
        except KeyError as exc:
            raise ObjectNotFoundError(f"Object not found: {key}") from exc

    def put_stream(
        self,
        key: str,
        stream,
        *,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = 1024 * 1024,
    ) -> StoredObject:
        data = stream.read()
        if length is not None and len(data) != length:
            raise AssertionError("test storage received the wrong stream length")
        self.objects[key] = data
        return StoredObject(
            key=key,
            size_bytes=len(data),
            checksum_sha256=hashlib.sha256(data).hexdigest(),
            content_type=content_type,
        )

    def iter_bytes(self, key: str, *, chunk_size: int = 1024 * 1024):
        data = self.get_bytes(key)
        for start in range(0, len(data), chunk_size):
            yield data[start : start + chunk_size]

    def put_file(
        self,
        key: str,
        source: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> StoredObject:
        path = Path(source)
        return self.put_stream(
            key,
            io.BytesIO(path.read_bytes()),
            length=path.stat().st_size,
            content_type=content_type,
        )

    def download_file(self, key: str, destination: str | Path) -> StoredObject:
        data = self.get_bytes(key)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return StoredObject(
            key=key,
            size_bytes=len(data),
            checksum_sha256=hashlib.sha256(data).hexdigest(),
            content_type="application/octet-stream",
        )

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class FakeTrainer:
    def __init__(
        self,
        *,
        key: str,
        recipe_key: str,
        runtime_class: str,
        output_name: str,
    ):
        self.key = key
        self.recipe_keys = (recipe_key,)
        self.runtime_class = runtime_class
        self.output_name = output_name
        self.train_calls = 0
        self.train_rows: tuple[dict, ...] = ()
        self.validation_rows: tuple[dict, ...] = ()
        self.base_model_contents: bytes | None = None

    def preflight(self, recipe_key: str | None = None) -> TrainerPreflight:
        return TrainerPreflight(
            ready=True,
            runtime_class=self.runtime_class,
            checks=({"key": "fake-runtime", "status": "pass", "message": "ready"},),
        )

    def plan(
        self,
        *,
        recipe_key: str,
        config,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan:
        return TrainingPlan(
            normalized_config=dict(config),
            manifest={
                "schema_version": "al-medlit-training-plan-v1",
                "recipe": {"key": recipe_key, "version": "1"},
                "trainer": {"key": self.key, "version": "1"},
                "runtime_class": self.runtime_class,
                "dataset_fingerprint": training_input.dataset_fingerprint,
                "split_fingerprint": training_input.split_fingerprint,
                "seed": seed,
            },
        )

    def train(
        self,
        *,
        recipe_key: str,
        config,
        training_input: TrainingInput,
        destination: Path,
        seed: int,
    ) -> TrainingOutput:
        self.train_calls += 1
        self.train_rows = training_input.rows
        self.validation_rows = training_input.validation_rows
        if training_input.base_model_path is not None:
            base_path = Path(training_input.base_model_path) / "config.json"
            self.base_model_contents = base_path.read_bytes()
            assert not bool(base_path.stat().st_mode & 0o222)
        destination.mkdir(parents=True)
        (destination / self.output_name).write_bytes(b"safe model bytes")
        (destination / "recipe.json").write_text(
            json.dumps(dict(config), sort_keys=True),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "al-medlit-training-package-v1",
            "recipe_key": recipe_key,
            "seed": seed,
            "train_count": len(training_input.rows),
            "validation_count": len(training_input.validation_rows),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        return TrainingOutput(
            manifest=manifest,
            validation_metrics={"accuracy": 0.75},
            artifact_paths=(self.output_name, "recipe.json", "manifest.json"),
        )


class FakeEvaluator:
    key = "fake-holdout"
    evaluator_version = "1"

    def __init__(self, *, recipe_key: str, trainer: FakeTrainer):
        self.recipe_keys = (recipe_key,)
        self.trainer = trainer
        self.calls = 0
        self.rows: tuple[dict, ...] = ()

    def evaluate(
        self,
        *,
        recipe_key,
        config,
        evaluation_input,
        model_directory,
    ) -> EvaluationOutput:
        assert recipe_key in self.recipe_keys
        assert config
        assert self.trainer.train_calls == 1
        assert (model_directory / self.trainer.output_name).is_file()
        self.calls += 1
        self.rows = evaluation_input.rows
        return EvaluationOutput(
            metrics={"accuracy": 1.0},
            prediction_count=len(evaluation_input.rows),
            report={
                "phase": "post_training",
                "prediction_count": len(evaluation_input.rows),
            },
        )


class FailingEvaluator(FakeEvaluator):
    key = "failing-holdout"

    def evaluate(
        self,
        *,
        recipe_key,
        config,
        evaluation_input,
        model_directory,
    ) -> EvaluationOutput:
        assert recipe_key in self.recipe_keys
        assert config
        assert self.trainer.train_calls == 1
        assert (model_directory / self.trainer.output_name).is_file()
        self.calls += 1
        self.rows = evaluation_input.rows
        raise ValidationError("protected evaluation failed intentionally")


def _user(db, username: str) -> User:
    user = User(
        username=username,
        password_hash="test-only",
        display_name=username,
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user


def _runtime_report(runtime_class: str, image_digest: str) -> dict:
    descriptor = RUNTIME_PROFILES[runtime_class]
    return RuntimeReadinessReport(
        runtime_profile=runtime_class,
        generated_at=datetime.now(UTC),
        python_version="3.12.0",
        worker_image_digest=image_digest.removeprefix("sha256:"),
        dependency_versions={
            name: "test" for name in descriptor.required_imports
        },
        missing_dependencies=[],
        device=descriptor.required_device,
        device_available=True,
        device_memory_bytes=16 * 1024 * 1024 * 1024,
        scratch_path="/tmp/al-medlit-test",
        scratch_available_bytes=descriptor.minimum_scratch_bytes,
        storage_access_verified=True,
        ready=True,
    ).model_dump(mode="json")


def _base_model_asset(db, storage, *, project: Project, actor: User) -> BaseModelAsset:
    package = artifact_service.publish_artifact_package(
        db,
        storage,
        project_id=project.id,
        data=ArtifactPackageCreate(
            package_kind="base_model",
            package_format="huggingface_json",
            display_name="Pinned tiny base",
            model_family="deep_learning",
            model_type="transformer_encoder",
            readiness="ready",
            deployable=True,
            loader_policy="safe",
            task_contract={},
            license_info={"name": "apache-2.0"},
        ),
        files=[
            artifact_service.PackageFileUpload(
                relative_path="config.json",
                source=b'{"model_type":"tiny"}',
                role="configuration",
                content_type="application/json",
            ),
            artifact_service.PackageFileUpload(
                relative_path="model.safetensors",
                source=b"worker-test-safetensors",
                role="weights",
            ),
        ],
        actor_user_id=actor.id,
    )
    asset = BaseModelAsset(
        project_id=project.id,
        workspace_id=project.workspace_id,
        package_id=package.id,
        provider="uploaded",
        source_model_id="tiny/base",
        exact_revision="revision-001",
        display_name="Pinned tiny base",
        model_family="deep_learning",
        model_type="transformer_encoder",
        license_name="apache-2.0",
        access_mode="execution_only",
        metadata_={},
        created_by_user_id=actor.id,
    )
    db.add(asset)
    db.flush()
    db.add(
        BaseModelAssetState(
            base_model_asset_id=asset.id,
            readiness="ready",
            updated_by_user_id=actor.id,
        )
    )
    db.commit()
    db.refresh(asset)
    return asset


def _training_run_fixture(
    db,
    *,
    storage,
    suffix: str,
    recipe_key: str = "tfidf_logistic_regression",
    plugin_key: str = "sklearn_tfidf",
    environment_class: str = "classical-cpu",
    with_base_model: bool = False,
    evaluation_plan: dict | None = None,
):
    actor = _user(db, f"workflow-executor-{suffix}")
    workspace = workspace_service.create_team_workspace(
        db,
        actor,
        name=f"Executor workspace {suffix}",
    )
    capability_service.set_capability(
        db,
        workspace.id,
        preset="full",
        actor_user_id=actor.id,
    )
    project = Project(name=f"Executor project {suffix}", workspace_id=workspace.id)
    db.add(project)
    db.commit()

    task_definition = service.create_task_definition(
        db,
        schemas.TaskDefinitionCreate(
            project_id=project.id,
            key=f"classification-{suffix}",
            name="Classification",
        ),
        actor,
    )
    task = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project.id,
            task_definition_id=task_definition.id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"label": ["negative", "positive"]},
            label_rules={"labels": ["negative", "positive"]},
            metrics=["accuracy"],
            trainer_compatibility=[plugin_key],
        ),
        actor,
    )
    dataset = service.create_dataset(
        db,
        schemas.DatasetCreate(
            project_id=project.id,
            name="Public reviews",
            source_type="public_registry",
        ),
        actor,
    )
    texts = {
        "train-1": "train corrected",
        "train-2": "train excluded",
        "validation-1": "validation visible",
        "pool-1": "pool forbidden",
        "test-1": "test forbidden",
    }
    dataset_version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project.id,
            dataset_id=dataset.id,
            source_uri="hf://example/reviews",
            source_revision="1" * 40,
            source_format="jsonl",
            data_schema={"text": "string"},
            provenance={"registry": "huggingface"},
            license_info={"name": "apache-2.0"},
            items=[
                schemas.DatasetItemCreate(
                    stable_key=stable_key,
                    group_key=f"group-{stable_key}",
                    payload={"text": text},
                )
                for stable_key, text in texts.items()
            ],
        ),
        actor,
    )
    split_map = service.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project.id,
            dataset_version_id=dataset_version.id,
            name="governed-splits",
            strategy="grouped",
            assignments={
                "train-1": "train",
                "train-2": "train",
                "validation-1": "validation",
                "pool-1": "pool",
                "test-1": "test",
            },
            protected_splits=["test"],
        ),
        actor,
    )
    imported = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project.id,
            dataset_version_id=dataset_version.id,
            task_version_id=task.id,
            name="source-labels",
            source_kind="imported",
            labels={
                "train-1": {"label": "negative"},
                "train-2": {"label": "negative"},
                "validation-1": {"label": "positive"},
                "pool-1": {"label": "negative"},
                "test-1": {"label": "positive"},
            },
        ),
        actor,
    )
    corrections = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project.id,
            dataset_version_id=dataset_version.id,
            task_version_id=task.id,
            name="human-corrections",
            source_kind="imported",
            labels={"train-1": {"label": "positive"}},
        ),
        actor,
    )
    exclusions = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project.id,
            dataset_version_id=dataset_version.id,
            task_version_id=task.id,
            name="excluded-items",
            source_kind="derived",
            composition_policy="exclude",
            labels={"train-2": True},
        ),
        actor,
    )
    training_dataset = service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=project.id,
            name="composed-training-data",
            dataset_version_id=dataset_version.id,
            task_version_id=task.id,
            label_set_version_ids=[imported.id, corrections.id, exclusions.id],
            split_map_id=split_map.id,
            composition=[
                {"label_set_version_id": imported.id, "policy": "inherit"},
                {"label_set_version_id": corrections.id, "policy": "replace"},
                {"label_set_version_id": exclusions.id, "policy": "exclude"},
            ],
            preprocessing={"normalization": "none"},
        ),
        actor,
    )
    registered = service.create_registered_model(
        db,
        schemas.RegisteredModelCreate(
            project_id=project.id,
            name=f"named-model-{suffix}",
        ),
        actor,
    )
    recipe = service.create_training_recipe(
        db,
        schemas.TrainingRecipeCreate(
            project_id=project.id,
            key=recipe_key,
            name=f"Recipe {recipe_key}",
        ),
        actor,
    )
    descriptor = None
    try:
        descriptor = training_recipes.get(recipe_key)
    except Exception:
        pass
    recipe_version = service.create_training_recipe_version(
        db,
        schemas.TrainingRecipeVersionCreate(
            project_id=project.id,
            training_recipe_id=recipe.id,
            trainer_plugin_key=plugin_key,
            trainer_plugin_version="1",
            compatible_task_kinds=["classification"],
            environment_class=environment_class,
            config_schema=descriptor.config_schema if descriptor is not None else {},
        ),
        actor,
    )
    image_digest = "sha256:" + "a" * 64
    environment = service.create_execution_environment(
        db,
        schemas.ExecutionEnvironmentCreate(
            project_id=project.id,
            name=f"Worker {suffix}",
            environment_class=environment_class,
            image_digest=image_digest,
            package_manifest={},
        ),
        actor,
    )
    service.verify_execution_environment(
        db,
        project.id,
        environment.id,
        schemas.EnvironmentVerification(
            status="available",
            verification_report=_runtime_report(environment_class, image_digest),
        ),
    )
    storage_policy = service.create_storage_policy(
        db,
        schemas.StoragePolicyCreate(
            project_id=project.id,
            name="Immutable artifacts",
            backend="minio",
            artifact_prefix=artifact_service.workspace_blob_prefix(workspace.id),
            retention_class="indefinite",
            encryption={"mode": "none"},
        ),
        actor,
    )
    config = {"fields": {"input_field": "text", "target_field": "label"}}
    base_asset = None
    if with_base_model:
        base_asset = _base_model_asset(db, storage, project=project, actor=actor)
        config = {
            "base_model_asset_id": base_asset.id,
            "input_field": "text",
            "target_field": "label",
        }
    run = service.create_training_run(
        db,
        schemas.TrainingRunCreate(
            project_id=project.id,
            registered_model_id=registered.id,
            task_version_id=task.id,
            training_dataset_version_id=training_dataset.id,
            recipe_version_id=recipe_version.id,
            environment_id=environment.id,
            storage_policy_id=storage_policy.id,
            idempotency_key=f"run-{suffix}",
            evaluation_plan=(
                evaluation_plan
                if evaluation_plan is not None
                else {"splits": ["validation"], "metrics": ["accuracy"]}
            ),
            config=config,
            seed=17,
            artifact_reservation_bytes=1024 * 1024,
        ),
        actor,
    )
    return SimpleNamespace(
        actor=actor,
        workspace=workspace,
        project=project,
        task=task,
        dataset=dataset_version,
        split_map=split_map,
        training_dataset=training_dataset,
        registered=registered,
        recipe_version=recipe_version,
        environment=environment,
        storage_policy=storage_policy,
        base_asset=base_asset,
        run=run,
    )


def test_launch_rejects_builtin_recipe_with_untrusted_plugin(db):
    with pytest.raises(ValidationError, match="trusted trainer plugin"):
        _training_run_fixture(
            db,
            storage=MemoryObjectStorage(),
            suffix="untrusted-plugin",
            recipe_key="tfidf_logistic_regression",
            plugin_key="replacement-plugin",
        )


def test_sklearn_holdout_evaluator_computes_aggregate_classification_metrics(
    tmp_path,
):
    class PredictingModel:
        def predict(self, texts):
            assert texts == ["first", "second"]
            return ["positive", "positive"]

    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"worker-generated-model")
    evaluator = SklearnTfidfEvaluator(
        model_loader=lambda path: (
            PredictingModel()
            if path == model_path
            else pytest.fail("unexpected model path")
        )
    )
    result = evaluator.evaluate(
        recipe_key="tfidf_logistic_regression",
        config={
            "fields": {
                "input_field": "text",
                "target_field": "label",
            }
        },
        evaluation_input=EvaluationInput(
            rows=(
                {"text": "first", "label": "positive"},
                {"text": "second", "label": "negative"},
            ),
            split_name="test",
            protected_split=True,
            requested_metrics=("accuracy", "f1"),
            dataset_fingerprint="dataset",
            training_dataset_fingerprint="training-dataset",
            split_fingerprint="split-map",
            model_fingerprint="model",
            task_kind="classification",
            label_vocabulary=("negative", "positive"),
        ),
        model_directory=tmp_path,
    )

    assert result.metrics["accuracy"] == pytest.approx(0.5)
    assert result.metrics["f1"] == pytest.approx(1 / 3)
    assert result.prediction_count == 2
    assert "predictions" not in result.report


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None
    or importlib.util.find_spec("skops") is None,
    reason="classical-cpu optional dependencies are not installed",
)
def test_real_skops_checkpoint_stages_and_loads_from_memory_storage(db, tmp_path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from skops.io import dump

    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="real-skops-stage",
    )
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(random_state=42),
    )
    pipeline.fit(
        ["include this study", "exclude this review", "include trial", "exclude editorial"],
        ["positive", "negative", "positive", "negative"],
    )
    checkpoint = tmp_path / "model.skops"
    dump(pipeline, checkpoint)
    package = artifact_service.publish_artifact_package(
        db,
        storage,
        project_id=scope.project.id,
        data=ArtifactPackageCreate(
            package_kind="model_checkpoint",
            package_format="skops",
            display_name="Real safe sklearn checkpoint",
            model_family="conventional_ml",
            model_type="tfidf_logistic_regression",
            readiness="ready",
            deployable=False,
            loader_policy="safe",
            task_contract={
                "task_version_id": scope.task.id,
                "task_content_hash": scope.task.content_hash,
            },
            license_info={},
        ),
        files=[
            artifact_service.PackageFileUpload(
                relative_path="model.skops",
                source=checkpoint.read_bytes(),
                role="weights",
            )
        ],
        actor_user_id=scope.actor.id,
    )

    staged = stage_artifact_package(
        db,
        storage,
        package=package,
        destination=tmp_path / "staged",
    )
    loaded = load_safe_skops_model(staged / "model.skops")

    assert loaded.predict(["include a randomized trial"]).tolist() == ["positive"]
    assert loaded.predict_proba(["include a randomized trial"]).shape == (1, 2)


def test_safe_skops_loader_rejects_untrusted_types(monkeypatch, tmp_path):
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"untrusted checkpoint")
    skops_module = ModuleType("skops")
    skops_module.__path__ = []
    skops_io_module = ModuleType("skops.io")
    skops_io_module.get_untrusted_types = lambda **_kwargs: ["unsafe.CustomEstimator"]
    skops_io_module.load = lambda *_args, **_kwargs: pytest.fail(
        "untrusted checkpoint was loaded"
    )
    monkeypatch.setitem(sys.modules, "skops", skops_module)
    monkeypatch.setitem(sys.modules, "skops.io", skops_io_module)

    with pytest.raises(ValidationError, match="unsafe.CustomEstimator"):
        load_safe_skops_model(model_path)


def test_artifact_staging_detects_corrupted_object_bytes(db, tmp_path):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="corrupt-stage")
    package = db.get(ArtifactPackage, scope.model_version.checkpoint_package_id)
    model_file = next(
        package_file
        for package_file in package.files
        if package_file.relative_path == "model.skops"
    )
    storage.objects[model_file.blob.storage_key] = b"corrupted bytes"

    with pytest.raises(ConflictError, match="integrity verification"):
        stage_artifact_package(
            db,
            storage,
            package=package,
            destination=tmp_path / "corrupted",
        )


def test_protected_test_evaluation_runs_after_training_and_is_immutable(db, client):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="protected-evaluation",
        evaluation_plan={"splits": ["validation", "test"], "metrics": ["accuracy"]},
    )
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    evaluator = FakeEvaluator(
        recipe_key="tfidf_logistic_regression",
        trainer=trainer,
    )
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(evaluator)

    completed = execute_training_run(
        db,
        storage,
        training_run_id=scope.run.id,
        trainer_registry=trainers,
        evaluator_registry=evaluators,
    )

    assert completed.status == "succeeded"
    assert trainer.train_rows == (
        {"text": "train corrected", "label": "positive"},
    )
    assert trainer.validation_rows == (
        {"text": "validation visible", "label": "positive"},
    )
    assert evaluator.calls == 1
    assert evaluator.rows == (
        {"text": "test forbidden", "label": "positive"},
    )
    assert all(
        row["text"] != "test forbidden"
        for row in (*trainer.train_rows, *trainer.validation_rows)
    )

    evaluations = service.list_training_run_evaluations(
        db,
        scope.project.id,
        completed.id,
    )
    assert len(evaluations) == 1
    evaluation = evaluations[0]
    model_version = db.get(models.ModelVersion, completed.output_model_version_id)
    assert evaluation.status == "succeeded"
    assert evaluation.training_run_id == completed.id
    assert evaluation.model_version_id == model_version.id
    assert evaluation.task_version_id == scope.task.id
    assert evaluation.training_dataset_version_id == scope.training_dataset.id
    assert evaluation.dataset_version_id == scope.dataset.id
    assert evaluation.split_map_id == scope.split_map.id
    assert evaluation.split_name == "test"
    assert evaluation.metrics == {"accuracy": 1.0}
    assert evaluation.row_count == 1
    assert len(evaluation.runtime_digest) == 64
    assert len(evaluation.code_digest) == 64
    assert (
        service.list_model_version_evaluations(
            db,
            scope.project.id,
            scope.registered.id,
            model_version.id,
        )[0].id
        == evaluation.id
    )
    headers = {
        "Authorization": f"Bearer {create_access_token(str(scope.actor.id))}",
    }
    run_response = client.get(
        f"/api/training-runs/{completed.id}/evaluations",
        params={"project_id": scope.project.id},
        headers=headers,
    )
    assert run_response.status_code == 200
    assert run_response.json()[0]["id"] == evaluation.id
    model_response = client.get(
        (
            f"/api/models/{scope.registered.id}/versions/{model_version.id}"
            "/evaluations"
        ),
        params={"project_id": scope.project.id},
        headers=headers,
    )
    assert model_response.status_code == 200
    assert model_response.json()[0]["id"] == evaluation.id
    detail_response = client.get(
        f"/api/evaluations/{evaluation.id}",
        params={"project_id": scope.project.id},
        headers=headers,
    )
    assert detail_response.status_code == 200
    assert detail_response.json()["content_hash"] == evaluation.content_hash

    evaluation_package = db.get(ArtifactPackage, evaluation.artifact_package_id)
    checkpoint_package = db.get(
        ArtifactPackage,
        model_version.checkpoint_package_id,
    )
    assert checkpoint_package.deployable is True
    promotion_references = {
        reference.relationship_type: reference.target_package_id
        for reference in checkpoint_package.outgoing_references
    }
    assert promotion_references["validated_by_evaluation"] == evaluation_package.id
    candidate_package = db.get(
        ArtifactPackage,
        promotion_references["promoted_from_candidate"],
    )
    assert candidate_package.package_kind == "model_candidate"
    assert candidate_package.deployable is False
    assert candidate_package.retention.retention_class == "candidate"
    assert checkpoint_package.metadata_["protected_test_gate"] == {
        "required": True,
        "status": "succeeded",
        "evaluator_key": evaluator.key,
        "evaluator_version": evaluator.evaluator_version,
        "candidate_package_id": candidate_package.id,
        "evaluation_package_id": evaluation_package.id,
    }
    assert evaluation_package.package_kind == "evaluation_report"
    assert evaluation_package.sensitivity == "restricted"
    assert evaluation_package.deployable is False
    assert [
        (reference.relationship_type, reference.target_package_id)
        for reference in evaluation_package.outgoing_references
    ] == [("evaluates_model", candidate_package.id)]
    report_file = evaluation_package.files[0]
    assert report_file.relative_path == "evaluation.json"
    report_bytes = storage.get_bytes(report_file.blob.storage_key)
    report = json.loads(report_bytes)
    assert report["lineage"]["candidate_checkpoint_package_id"] == (
        candidate_package.id
    )
    assert report["lineage"]["candidate_checkpoint_manifest_digest"] == (
        candidate_package.manifest_digest
    )
    assert report["lineage"]["split_map_content_hash"] == scope.split_map.content_hash
    assert report["metrics"] == {"accuracy": 1.0}
    assert b"test forbidden" not in report_bytes

    evaluation.metrics = {"accuracy": 0.0}
    with pytest.raises(ImmutableRecordError, match="ModelEvaluation.*immutable"):
        db.commit()
    db.rollback()
    assert db.get(models.ModelEvaluation, evaluation.id).metrics == {
        "accuracy": 1.0
    }

    evaluation_package = db.get(ArtifactPackage, evaluation_package.id)
    evaluation_package.metadata_ = {"changed": True}
    with pytest.raises(ImmutableRecordError, match="ArtifactPackage.*immutable"):
        db.commit()
    db.rollback()


def test_requested_test_evaluation_marks_unsupported_recipe_without_metrics(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="unsupported-evaluation",
        recipe_key="transformer_sequence_classification",
        plugin_key="huggingface_sequence",
        environment_class="transformer-cpu",
        with_base_model=True,
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    trainer = FakeTrainer(
        key="huggingface_sequence",
        recipe_key="transformer_sequence_classification",
        runtime_class="transformer-cpu",
        output_name="model.safetensors",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    completed = execute_training_run(
        db,
        storage,
        training_run_id=scope.run.id,
        trainer_registry=trainers,
        evaluator_registry=EvaluatorPluginRegistry(),
    )

    assert completed.status == "succeeded"
    [evaluation] = service.list_training_run_evaluations(
        db,
        scope.project.id,
        completed.id,
    )
    assert evaluation.status == "unsupported"
    assert evaluation.metrics == {}
    assert evaluation.artifact_package_id is None
    assert evaluation.evaluator_key is None
    assert "No trusted protected-test evaluator" in evaluation.status_reason
    assert completed.runtime_snapshot["protected_test_evaluation"]["status"] == (
        "unsupported"
    )
    model_version = db.get(models.ModelVersion, completed.output_model_version_id)
    assert model_version.metrics == {"accuracy": 0.75}
    checkpoint_package = db.get(
        ArtifactPackage,
        model_version.checkpoint_package_id,
    )
    assert checkpoint_package.deployable is False
    assert checkpoint_package.package_kind == "model_candidate"
    assert checkpoint_package.metadata_["protected_test_gate"] == {
        "required": True,
        "status": "unsupported",
        "evaluator_key": None,
        "evaluator_version": None,
    }


def test_evaluator_failure_leaves_only_a_non_deployable_candidate(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="failed-evaluation",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)
    evaluator = FailingEvaluator(
        recipe_key="tfidf_logistic_regression",
        trainer=trainer,
    )
    evaluators = EvaluatorPluginRegistry()
    evaluators.register(evaluator)

    with pytest.raises(
        ValidationError,
        match="protected evaluation failed intentionally",
    ):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=trainers,
            evaluator_registry=evaluators,
        )

    db.expire_all()
    failed = db.get(models.TrainingRun, scope.run.id)
    assert failed.status == "failed"
    assert failed.output_model_version_id is None
    assert trainer.train_calls == 1
    assert evaluator.calls == 1
    assert evaluator.rows == (
        {"text": "test forbidden", "label": "positive"},
    )
    assert all(
        row["text"] != "test forbidden"
        for row in (*trainer.train_rows, *trainer.validation_rows)
    )
    packages = (
        db.query(ArtifactPackage)
        .filter(ArtifactPackage.project_id == scope.project.id)
        .all()
    )
    assert len(packages) == 1
    [candidate] = packages
    assert candidate.package_kind == "model_candidate"
    assert candidate.deployable is False
    assert candidate.metadata_["protected_test_gate"]["status"] == "pending"
    assert candidate.retention.retention_class == "candidate"
    assert db.query(models.ModelVersion).filter(
        models.ModelVersion.project_id == scope.project.id
    ).count() == 0
    assert db.query(models.ModelEvaluation).filter(
        models.ModelEvaluation.project_id == scope.project.id
    ).count() == 0
    reservation = db.get(
        ArtifactStorageReservation,
        scope.run.artifact_reservation_id,
    )
    assert reservation.status == "committed"
    assert reservation.artifact_package_id == candidate.id


def test_executor_composes_labels_excludes_pool_and_test_and_is_idempotent(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="classical")
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
        training_run_id=scope.run.id,
        trainer_registry=registry,
    )
    repeated = execute_training_run(
        db,
        storage,
        training_run_id=scope.run.id,
        trainer_registry=registry,
    )

    assert completed.status == "succeeded"
    assert repeated.id == completed.id
    assert repeated.output_model_version_id == completed.output_model_version_id
    assert trainer.train_calls == 1
    assert trainer.train_rows == ({"text": "train corrected", "label": "positive"},)
    assert trainer.validation_rows == (
        {"text": "validation visible", "label": "positive"},
    )
    assert db.query(models.ModelVersion).count() == 1

    version = db.get(models.ModelVersion, completed.output_model_version_id)
    package = db.get(ArtifactPackage, version.checkpoint_package_id)
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert version.recipe_key == "tfidf_logistic_regression"
    assert version.family == "conventional_ml"
    assert version.framework == "scikit-learn"
    assert version.seed == 17
    assert version.parameters["_lineage"]["task_content_hash"] == scope.task.content_hash
    assert (
        version.parameters["_lineage"]["training_dataset_content_hash"]
        == scope.training_dataset.content_hash
    )
    assert version.parameters["_lineage"]["split_map_content_hash"] == scope.split_map.content_hash
    assert version.parameters["_lineage"]["checkpoint_manifest_digest"] == (
        package.manifest_digest
    )
    assert version.parameters["_lineage"]["evaluation_plan"] == {
        "splits": ["validation"],
        "metrics": ["accuracy"],
    }
    assert len(version.runtime_digest) == 64
    assert len(version.code_digest) == 64
    assert reservation.status == "committed"
    assert reservation.artifact_package_id == package.id


def test_executor_rejects_worker_storage_encryption_mismatch_and_releases_quota(
    db,
):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="storage-encryption-mismatch",
    )
    storage.encryption_mode = "sse-s3"
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    registry = TrainerPluginRegistry()
    registry.register(trainer)

    with pytest.raises(ConflictError, match="storage encryption"):
        execute_training_run(
            db,
            storage,
            training_run_id=scope.run.id,
            trainer_registry=registry,
        )

    run = db.get(models.TrainingRun, scope.run.id)
    reservation = db.get(ArtifactStorageReservation, run.artifact_reservation_id)
    assert run.status == "failed"
    assert reservation.status == "released"
    assert trainer.train_calls == 0
    assert db.query(models.ModelVersion).count() == 0


def test_executor_stages_pinned_base_model_and_records_reference(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db,
        storage=storage,
        suffix="transformer",
        recipe_key="transformer_sequence_classification",
        plugin_key="huggingface_sequence",
        environment_class="transformer-cpu",
        with_base_model=True,
    )
    trainer = FakeTrainer(
        key="huggingface_sequence",
        recipe_key="transformer_sequence_classification",
        runtime_class="transformer-cpu",
        output_name="model.safetensors",
    )
    registry = TrainerPluginRegistry()
    registry.register(trainer)

    completed = execute_training_run(
        db,
        storage,
        training_run_id=scope.run.id,
        trainer_registry=registry,
    )

    version = db.get(models.ModelVersion, completed.output_model_version_id)
    package = db.get(ArtifactPackage, version.checkpoint_package_id)
    assert trainer.base_model_contents == b'{"model_type":"tiny"}'
    assert version.base_model["asset_id"] == scope.base_asset.id
    assert version.base_model["exact_revision"] == "revision-001"
    assert version.base_model["package_manifest_digest"] == (
        scope.base_asset.package.manifest_digest
    )
    assert [
        (reference.relationship_type, reference.target_package_id)
        for reference in package.outgoing_references
    ] == [("uses_base_model", scope.base_asset.package_id)]


def test_launch_fails_closed_for_unregistered_custom_recipe(db):
    storage = MemoryObjectStorage()
    with pytest.raises(
        ValidationError,
        match="registered trusted worker contract",
    ):
        _training_run_fixture(
            db,
            storage=storage,
            suffix="custom",
            recipe_key="custom_classifier",
            plugin_key="custom_plugin",
        )
    assert db.query(models.TrainingRun).count() == 0
    assert db.query(ArtifactStorageReservation).count() == 0


def test_worker_heartbeat_and_stale_run_recovery_release_quota(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="worker-recovery")
    service.transition_training_run(
        db,
        scope.project.id,
        scope.run.id,
        schemas.TrainingRunTransition(status="running"),
    )
    heartbeat_at = datetime.now(UTC) - timedelta(minutes=10)
    assert heartbeat_training_run(db, scope.run.id, now=heartbeat_at) is True

    recovered = reconcile_stale_training_runs(
        db,
        now=datetime.now(UTC),
        stale_after_seconds=300,
    )

    assert recovered == [scope.run.id]
    run = db.get(models.TrainingRun, scope.run.id)
    assert run.status == "failed"
    assert run.failure_code == "worker_heartbeat_expired"
    assert run.runtime_snapshot["worker_recovery"]["outcome"] == (
        "failed_safe_to_relaunch"
    )
    reservation = db.get(ArtifactStorageReservation, run.artifact_reservation_id)
    assert reservation.status == "released"
    assert db.query(models.ModelVersion).count() == 0


def test_queue_routing_uses_selected_environment(monkeypatch):
    dispatched = {}

    def fake_apply_async(*, args, queue):
        dispatched.update(args=args, queue=queue)

    monkeypatch.setattr(
        "al_medlit.training.tasks.execute_training_run_task.apply_async",
        fake_apply_async,
    )
    enqueue_training_run(91, environment_class="qlora-cuda")

    assert dispatched == {"args": (91,), "queue": "qlora-cuda"}
    with pytest.raises(ValidationError, match="has no worker queue"):
        enqueue_training_run(92, environment_class="custom-runtime")


def test_feedback_scoring_queue_uses_classical_cpu(monkeypatch):
    from al_medlit.training.tasks import enqueue_feedback_scoring

    dispatched = {}

    def fake_apply_async(*, args, queue):
        dispatched.update(args=args, queue=queue)

    monkeypatch.setattr(
        "al_medlit.training.tasks.execute_feedback_scoring_task.apply_async",
        fake_apply_async,
    )
    enqueue_feedback_scoring(93)

    assert dispatched == {"args": (93,), "queue": "classical-cpu"}


def test_training_launch_route_dispatches_queued_run(monkeypatch):
    dispatched = {}
    queued_run = SimpleNamespace(
        id=41,
        status="queued",
        runtime_snapshot={"environment_class": "transformer-cpu"},
    )
    payload = SimpleNamespace(project_id=7)
    monkeypatch.setattr(workflow_router, "_write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        workflow_router.service,
        "create_training_run",
        lambda *_args, **_kwargs: queued_run,
    )
    monkeypatch.setattr(
        workflow_router,
        "enqueue_training_run",
        lambda run_id, *, environment_class: dispatched.update(
            run_id=run_id,
            environment_class=environment_class,
        ),
    )

    returned = workflow_router.create_training_run(
        payload,
        current_user=object(),
        db=object(),
    )

    assert returned is queued_run
    assert dispatched == {
        "run_id": 41,
        "environment_class": "transformer-cpu",
    }


def test_feedback_materialization_route_dispatches_and_refreshes(monkeypatch):
    dispatched = []
    queued = SimpleNamespace(id=42, status="queued")
    refreshed = SimpleNamespace(id=42, status="completed")

    class FakeDb:
        expired = False

        def expire_all(self):
            self.expired = True

    db = FakeDb()
    monkeypatch.setattr(learning_router, "_write", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        learning_router.service,
        "request_feedback_run_materialization",
        lambda *_args, **_kwargs: SimpleNamespace(
            run=queued,
            should_enqueue=True,
        ),
    )
    monkeypatch.setattr(
        learning_router.service,
        "get_feedback_run",
        lambda *_args, **_kwargs: refreshed,
    )
    monkeypatch.setattr(
        learning_router,
        "enqueue_feedback_scoring",
        dispatched.append,
    )

    result = learning_router.materialize_feedback_run(
        project_id=7,
        feedback_run_id=42,
        current_user=object(),
        db=db,
    )

    assert result is refreshed
    assert dispatched == [42]
    assert db.expired is True


def test_api_import_does_not_load_optional_ml_packages():
    script = """
import sys
from al_medlit.main import create_app
create_app()
optional = {'torch', 'transformers', 'peft', 'bitsandbytes', 'sklearn', 'skops'}
loaded = sorted(optional.intersection(sys.modules))
if loaded:
    raise SystemExit('optional ML imports loaded by API: ' + ','.join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


class FakeProbabilityModel:
    classes_ = ["negative", "positive"]

    def __init__(self):
        self.batches: list[list[str]] = []

    def predict_proba(self, texts):
        values = list(texts)
        self.batches.append(values)
        return [
            [0.9, 0.1] if "forbidden" not in text else [0.4, 0.6]
            for text in values
        ]


def _completed_tfidf_model(db, storage, *, suffix: str):
    scope = _training_run_fixture(db, storage=storage, suffix=suffix)
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
        training_run_id=scope.run.id,
        trainer_registry=registry,
    )
    scope.model_version = db.get(models.ModelVersion, completed.output_model_version_id)
    return scope


def test_registered_model_feedback_scoring_is_deterministic_and_idempotent(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring")
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )

    dispatch = request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    assert dispatch.should_enqueue is True
    assert dispatch.run.status == "queued"
    scope.actor.is_active = False
    db.commit()

    fake_model = FakeProbabilityModel()
    completed = execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=feedback_run.id,
        model_loader=lambda _path: fake_model,
    )
    repeated = execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=feedback_run.id,
        model_loader=lambda _path: pytest.fail("duplicate delivery reloaded the model"),
    )

    assert completed.status == "completed"
    assert completed.output_feedback_set_version_id is not None
    assert completed.completed_at is not None
    assert repeated.id == completed.id
    assert repeated.output_feedback_set_version_id == completed.output_feedback_set_version_id
    feedback_set = db.get(
        models.FeedbackSetVersion,
        completed.output_feedback_set_version_id,
    )
    assert feedback_set.candidate_count == scope.dataset.item_count == 5
    assert feedback_set.created_by_user_id == scope.actor.id
    candidates = (
        db.query(models.FeedbackCandidate)
        .filter(models.FeedbackCandidate.feedback_set_version_id == feedback_set.id)
        .order_by(models.FeedbackCandidate.dataset_item_id)
        .all()
    )
    assert len(candidates) == 5
    assert all(candidate.output in {"negative", "positive"} for candidate in candidates)
    assert all(
        candidate.explanation["uncertainty"] == pytest.approx(1 - candidate.score)
        and candidate.explanation["signal_basis"] == "least_confidence"
        and candidate.explanation["checkpoint_manifest_digest"]
        == scope.model_version.parameters["_lineage"]["checkpoint_manifest_digest"]
        for candidate in candidates
    )
    assert fake_model.batches == [
        [
            "pool forbidden",
            "test forbidden",
            "train corrected",
            "train excluded",
            "validation visible",
        ]
    ]

    selection_run = service.create_selection_run(
        db,
        schemas.SelectionRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            feedback_set_version_id=feedback_set.id,
            split_map_id=scope.split_map.id,
            strategy="uncertainty",
            parameters={"limit": 2},
        ),
        scope.actor,
    )
    selection_set = service.materialize_selection_run(
        db,
        project_id=scope.project.id,
        selection_run_id=selection_run.id,
        actor=scope.actor,
    )
    protected_item = (
        db.query(models.DatasetItem)
        .filter(
            models.DatasetItem.dataset_version_id == scope.dataset.id,
            models.DatasetItem.stable_key == "test-1",
        )
        .one()
    )
    assert protected_item.id in {candidate.dataset_item_id for candidate in candidates}
    assert protected_item.id not in {
        item["dataset_item_id"] for item in selection_set.items
    }

    equivalent_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=equivalent_run.id,
        actor=scope.actor,
    )
    equivalent_completed = execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=equivalent_run.id,
        model_loader=lambda _path: FakeProbabilityModel(),
    )
    equivalent_set = db.get(
        models.FeedbackSetVersion,
        equivalent_completed.output_feedback_set_version_id,
    )
    equivalent_hashes = [
        content_hash
        for (content_hash,) in (
            db.query(models.FeedbackCandidate.content_hash)
            .filter(models.FeedbackCandidate.feedback_set_version_id == equivalent_set.id)
            .order_by(models.FeedbackCandidate.dataset_item_id)
            .all()
        )
    ]
    assert equivalent_hashes == [candidate.content_hash for candidate in candidates]
    assert equivalent_set.content_hash != feedback_set.content_hash


def test_feedback_scoring_crosses_the_fixed_512_item_batch_boundary(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-batches")
    db.bulk_save_objects(
        [
            models.DatasetItem(
                project_id=scope.project.id,
                dataset_version_id=scope.dataset.id,
                stable_key=f"bulk-{index:04d}",
                payload={"text": f"bulk text {index}"},
                content_hash=f"{index + 1:064x}",
            )
            for index in range(508)
        ]
    )
    db.query(models.DatasetVersion).filter(
        models.DatasetVersion.id == scope.dataset.id
    ).update({models.DatasetVersion.item_count: 513})
    db.commit()
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    fake_model = FakeProbabilityModel()

    completed = execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=feedback_run.id,
        model_loader=lambda _path: fake_model,
    )

    feedback_set = db.get(
        models.FeedbackSetVersion,
        completed.output_feedback_set_version_id,
    )
    assert feedback_set.candidate_count == 513
    assert [len(batch) for batch in fake_model.batches] == [512, 1]
    assert fake_model.batches[0][0] == "bulk text 0"
    assert fake_model.batches[-1] == ["validation visible"]


def test_feedback_prediction_serializes_scalar_and_single_field_object_outputs():
    scalar_task = SimpleNamespace(
        output_schema={"type": "string", "enum": ["negative", "positive"]}
    )
    object_task = SimpleNamespace(
        output_schema={
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ["negative", "positive"]}
            },
            "required": ["label"],
            "additionalProperties": False,
        }
    )

    assert _prediction_output(scalar_task, "label", "positive") == "positive"
    assert _prediction_output(object_task, "label", "positive") == {
        "label": "positive"
    }


@pytest.mark.parametrize(
    ("retention_field", "message"),
    (("archived_at", "archived"), ("purged_at", "purged")),
)
def test_feedback_scoring_rejects_unavailable_checkpoint_retention(
    db,
    retention_field,
    message,
):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(
        db,
        storage,
        suffix=f"feedback-scoring-{retention_field}",
    )
    package = db.get(ArtifactPackage, scope.model_version.checkpoint_package_id)
    setattr(package.retention, retention_field, datetime.now(UTC))
    db.commit()
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )

    with pytest.raises(ConflictError, match=message):
        request_feedback_run_materialization(
            db,
            project_id=scope.project.id,
            feedback_run_id=feedback_run.id,
            actor=scope.actor,
        )


def test_feedback_scoring_eager_cap_fails_with_worker_guidance(db, monkeypatch):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-eager-cap")
    db.query(models.DatasetVersion).filter(
        models.DatasetVersion.id == scope.dataset.id
    ).update({models.DatasetVersion.item_count: 5001})
    db.commit()
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    monkeypatch.setattr(
        "al_medlit.workflow.services.feedback_scoring.settings.celery_task_always_eager",
        True,
    )

    with pytest.raises(ValidationError, match="use `make lab-up`"):
        execute_feedback_scoring_run(
            db,
            storage,
            feedback_run_id=feedback_run.id,
            model_loader=lambda _path: pytest.fail("eager cap loaded the model"),
        )

    assert db.get(models.FeedbackRun, feedback_run.id).status == "failed"


def test_feedback_scoring_failure_records_terminal_state_without_partial_set(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-failure")
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )

    class InvalidProbabilityModel(FakeProbabilityModel):
        def predict_proba(self, texts):
            return [[0.2, 0.2] for _text in texts]

    with pytest.raises(ValidationError, match="sum to one"):
        execute_feedback_scoring_run(
            db,
            storage,
            feedback_run_id=feedback_run.id,
            model_loader=lambda _path: InvalidProbabilityModel(),
        )

    db.expire_all()
    failed = db.get(models.FeedbackRun, feedback_run.id)
    assert failed.status == "failed"
    assert failed.failure_code == "ValidationError"
    assert "sum to one" in failed.failure_reason
    assert failed.output_feedback_set_version_id is None
    assert db.query(models.FeedbackSetVersion).filter(
        models.FeedbackSetVersion.feedback_run_id == feedback_run.id
    ).count() == 0

    retried = request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    assert retried.run.status == "queued"
    assert retried.run.failure_code is None
    with pytest.raises(ConflictError, match="cannot accept manual results"):
        service.create_feedback_set(
            db,
            schemas.FeedbackSetCreate(
                project_id=scope.project.id,
                feedback_run_id=feedback_run.id,
                output_schema=scope.task.output_schema,
                candidates=[],
            ),
            scope.actor,
        )


def test_feedback_scoring_rejects_missing_input_text_without_partial_output(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-missing-text")
    missing = (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == scope.dataset.id)
        .order_by(models.DatasetItem.id)
        .first()
    )
    db.query(models.DatasetItem).filter(models.DatasetItem.id == missing.id).update(
        {models.DatasetItem.payload: {"text": " "}}
    )
    db.commit()
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )

    with pytest.raises(ValidationError, match="must be non-empty text"):
        execute_feedback_scoring_run(
            db,
            storage,
            feedback_run_id=feedback_run.id,
            model_loader=lambda _path: FakeProbabilityModel(),
        )

    assert db.get(models.FeedbackRun, feedback_run.id).status == "failed"
    assert (
        db.query(models.FeedbackSetVersion)
        .filter(models.FeedbackSetVersion.feedback_run_id == feedback_run.id)
        .count()
        == 0
    )


def test_feedback_scoring_heartbeat_and_reconciliation(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-heartbeat")
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    now = datetime.now(UTC)
    queued = db.get(models.FeedbackRun, feedback_run.id)
    queued.updated_at = now - timedelta(minutes=6)
    db.commit()
    reconciled = reconcile_feedback_scoring_runs(db, now=now)
    assert reconciled["redispatch_ids"] == [feedback_run.id]

    from al_medlit.workflow.services.feedback_scoring import _claim_feedback_scoring_run

    _claim_feedback_scoring_run(db, feedback_run.id)
    old_heartbeat = now - timedelta(minutes=20)
    assert heartbeat_feedback_scoring_run(
        db,
        feedback_run.id,
        now=old_heartbeat,
    ) is True
    reconciled = reconcile_feedback_scoring_runs(db, now=now)
    assert reconciled["failed_ids"] == [feedback_run.id]
    failed = db.get(models.FeedbackRun, feedback_run.id)
    assert failed.status == "failed"
    assert failed.failure_code == "scoring_worker_stalled"


def test_completed_feedback_scoring_requires_a_consistent_output_pointer(db):
    storage = MemoryObjectStorage()
    scope = _completed_tfidf_model(db, storage, suffix="feedback-scoring-pointer")
    feedback_run = service.create_feedback_run(
        db,
        schemas.FeedbackRunCreate(
            project_id=scope.project.id,
            dataset_version_id=scope.dataset.id,
            task_version_id=scope.task.id,
            producer_type="registered_model",
            model_version_id=scope.model_version.id,
        ),
        scope.actor,
    )
    request_feedback_run_materialization(
        db,
        project_id=scope.project.id,
        feedback_run_id=feedback_run.id,
        actor=scope.actor,
    )
    execute_feedback_scoring_run(
        db,
        storage,
        feedback_run_id=feedback_run.id,
        model_loader=lambda _path: FakeProbabilityModel(),
    )
    db.query(models.FeedbackRun).filter(models.FeedbackRun.id == feedback_run.id).update(
        {models.FeedbackRun.output_feedback_set_version_id: None}
    )
    db.commit()

    with pytest.raises(ConflictError, match="has no output feedback set"):
        request_feedback_run_materialization(
            db,
            project_id=scope.project.id,
            feedback_run_id=feedback_run.id,
            actor=scope.actor,
        )
