"""Opt-in PostgreSQL races for atomic training preparation and version allocation."""

import os
from uuid import uuid4

import pytest
from test_system_administration_postgres_contention import (
    _new_user,
    _run_concurrently,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError
from al_medlit.core.storage import LocalObjectStorage
from al_medlit.project.models import Project
from al_medlit.workflow import models, preparation_schemas, schemas
from al_medlit.workflow.services import datasets, tasks
from al_medlit.workflow.services.composition import compose_training_dataset_version
from al_medlit.workflow.services.preparation import prepare_training_dataset
from al_medlit.workspace.models import Workspace

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _scope(db, *, legacy_input=False):
    user = _new_user(db, "training-preparation")
    workspace = Workspace(name=f"Preparation {uuid4().hex}", created_by=user.id)
    db.add(workspace)
    db.flush()
    project = Project(workspace_id=workspace.id, name="Preparation races")
    db.add(project)
    db.commit()
    definition = tasks.create_task_definition(
        db,
        schemas.TaskDefinitionCreate(project_id=project.id, key="classify", name="Classify"),
        user,
    )
    task = tasks.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project.id,
            task_definition_id=definition.id,
            task_kind="classification",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": legacy_input,
            },
            output_schema={"type": "string", "enum": ["yes", "no"]},
        ),
        user,
    )
    dataset = datasets.create_dataset(
        db,
        schemas.DatasetCreate(project_id=project.id, name="External", source_type="upload"),
        user,
    )
    source = datasets.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project.id,
            dataset_id=dataset.id,
            source_revision="1",
            source_format="jsonl",
            items=[
                schemas.DatasetItemCreate(
                    stable_key=f"paper-{index}",
                    payload={"text": f"paper number {index}", "label": "yes"},
                )
                for index in range(6)
            ],
        ),
        user,
    )
    request = preparation_schemas.TrainingPreparationRequest(
        name="Prepared examples",
        task_version_id=task.id,
        sources=[
            preparation_schemas.TrainingSource(dataset_version_id=source.id, label_field="label")
        ],
        idempotency_key="concurrent-initial",
    )
    return project.id, user.id, request


def test_concurrent_same_key_returns_one_atomic_training_snapshot(postgres_session_factory):
    with postgres_session_factory() as db:
        project_id, user_id, request = _scope(db)

    def prepare(db):
        result = prepare_training_dataset(db, project_id, request, db.get(User, user_id))
        return result.training_dataset_version.id

    left, right = _run_concurrently(postgres_session_factory, prepare, prepare)
    assert left[0] == right[0] == "ok", (left, right)
    assert left[1] == right[1]
    with postgres_session_factory() as db:
        counts = {
            model: db.query(model).filter(model.project_id == project_id).count()
            for model in (
                models.TrainingDataset,
                models.TrainingDatasetVersion,
                models.Dataset,
                models.DatasetVersion,
                models.LabelSetVersion,
                models.SplitMap,
            )
        }
        assert counts == {
            models.TrainingDataset: 1,
            models.TrainingDatasetVersion: 1,
            models.Dataset: 2,
            models.DatasetVersion: 2,
            models.LabelSetVersion: 2,
            models.SplitMap: 1,
        }


def test_concurrent_new_versions_reject_stale_parent_without_partial_rows(
    postgres_session_factory,
):
    with postgres_session_factory() as db:
        project_id, user_id, request = _scope(db)
        first = prepare_training_dataset(db, project_id, request, db.get(User, user_id))
        parent = first.training_dataset_version
        request = request.model_copy(
            update={
                "training_dataset_id": first.training_dataset.id,
                "parent_version_id": parent.id,
            }
        )
        parent_id = parent.id
        first_split = db.get(models.SplitMap, parent.split_map_id).assignments.copy()

    def prepare(key):
        def operation(db):
            result = prepare_training_dataset(
                db,
                project_id,
                request.model_copy(update={"idempotency_key": key}),
                db.get(User, user_id),
            )
            return result.training_dataset_version.id

        return operation

    results = _run_concurrently(
        postgres_session_factory, prepare("version-two-left"), prepare("version-two-right")
    )
    assert sorted(result[0] for result in results) == ["error", "ok"], results
    error = next(result[1] for result in results if result[0] == "error")
    assert isinstance(error, ConflictError), repr(error)
    with postgres_session_factory() as db:
        versions = (
            db.query(models.TrainingDatasetVersion)
            .filter(models.TrainingDatasetVersion.project_id == project_id)
            .order_by(models.TrainingDatasetVersion.version_number)
            .all()
        )
        assert [version.version_number for version in versions] == [1, 2]
        assert versions[1].parent_version_id == parent_id
        assert db.get(models.SplitMap, versions[0].split_map_id).assignments == first_split
        assert db.get(models.SplitMap, versions[1].split_map_id).assignments == first_split
        assert db.query(models.DatasetVersion).filter_by(project_id=project_id).count() == 3
        assert db.query(models.LabelSetVersion).filter_by(project_id=project_id).count() == 3
        assert db.query(models.SplitMap).filter_by(project_id=project_id).count() == 2


def test_legacy_composer_and_preparation_share_project_then_source_lock_order(
    postgres_session_factory, tmp_path
):
    with postgres_session_factory() as db:
        project_id, user_id, request = _scope(db, legacy_input=True)
        legacy_actor_id = _new_user(db, "legacy-composer").id
        db.commit()
    storage = LocalObjectStorage(tmp_path / "legacy-artifacts")

    def prepare(db):
        return prepare_training_dataset(
            db, project_id, request, db.get(User, user_id)
        ).training_dataset_version.id

    def compose(db):
        result = compose_training_dataset_version(
            db,
            schemas.TrainingDatasetComposeCreate(
                project_id=project_id,
                name="Legacy compatible examples",
                dataset_version_id=request.sources[0].dataset_version_id,
                task_version_id=request.task_version_id,
                input_field="text",
                label_field="label",
                train_percent=80,
                validation_percent=10,
                seed=42,
            ),
            db.get(User, legacy_actor_id),
            storage=storage,
        )
        return result["training_dataset_version"].id

    left, right = _run_concurrently(postgres_session_factory, prepare, compose)
    assert left[0] == right[0] == "ok", (left, right)
    assert left[1] != right[1]
    with postgres_session_factory() as db:
        versions = db.query(models.TrainingDatasetVersion).filter_by(project_id=project_id).all()
        assert len(versions) == 2
        assert all(version.training_dataset_id is not None for version in versions)
        assert all(version.version_number == 1 for version in versions)
        assert {version.created_by_user_id for version in versions} == {user_id, legacy_actor_id}
