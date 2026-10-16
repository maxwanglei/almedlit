"""PostgreSQL races must preserve article partitions across training series."""

import os

import pytest
from test_system_administration_postgres_contention import _run_concurrently
from test_training_preparation_postgres_contention import _scope
from test_training_preparation_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.auth.models import User
from al_medlit.workflow import models, preparation_schemas, schemas
from al_medlit.workflow.services import datasets
from al_medlit.workflow.services.preparation import (
    _build_preparation,
    prepare_training_dataset,
)

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _texts_in_split(db, version, partition):
    assignments = db.get(models.SplitMap, version.split_map_id).assignments
    return {
        item.payload["text"]
        for item in db.query(models.DatasetItem).filter_by(
            dataset_version_id=version.dataset_version_id
        )
        if assignments[item.stable_key] == partition
    }


def test_new_series_and_existing_series_update_keep_prior_training_out_of_test(
    postgres_session_factory,
):
    with postgres_session_factory() as db:
        project_id, user_id, request = _scope(db)
        actor = db.get(User, user_id)
        initial = _build_preparation(db, project_id, request)
        prior_training = {
            row.payload["text"]
            for row in initial.examples
            if initial.assignments[row.stable_key] in {"train", "validation"}
        }
        fresh_texts = {f"fresh paper number {index}" for index in range(3)}
        copied_dataset = datasets.create_dataset(
            db,
            schemas.DatasetCreate(
                project_id=project_id, name="Copied training articles", source_type="upload"
            ),
            actor,
        )
        copied_source = datasets.create_dataset_version(
            db,
            schemas.DatasetVersionCreate(
                project_id=project_id,
                dataset_id=copied_dataset.id,
                source_revision="1",
                source_format="jsonl",
                items=[
                    schemas.DatasetItemCreate(
                        stable_key=f"renamed-{index}", payload={"text": text, "label": "yes"}
                    )
                    for index, text in enumerate(sorted(prior_training | fresh_texts))
                ],
            ),
            actor,
        )
        new_request = request.model_copy(
            update={
                "name": "Second training series",
                "sources": [
                    preparation_schemas.TrainingSource(
                        dataset_version_id=copied_source.id, label_field="label"
                    )
                ],
                "idempotency_key": "second-series",
            }
        )
        # Choose an adversarial seed before the first series establishes any
        # history: without cross-series governance it would hold out an article
        # already used for training, even after copying and renaming its key.
        for seed in range(100):
            candidate = new_request.model_copy(update={"seed": seed})
            planned = _build_preparation(db, project_id, candidate)
            if any(
                row.payload["text"] in prior_training
                and planned.assignments[row.stable_key] == "test"
                for row in planned.examples
            ):
                new_request = candidate
                break
        else:
            pytest.fail("The regression fixture must exercise a contaminated test assignment")

        first = prepare_training_dataset(db, project_id, request, actor)
        series_id = first.training_dataset.id
        parent_id = first.training_dataset_version.id
        prior_holdout = _texts_in_split(db, first.training_dataset_version, "test")
        update_request = request.model_copy(
            update={
                "training_dataset_id": series_id,
                "parent_version_id": parent_id,
                "idempotency_key": "first-series-update",
            }
        )

    def prepare(request_data):
        def operation(db):
            return prepare_training_dataset(
                db, project_id, request_data, db.get(User, user_id)
            ).training_dataset_version.id

        return operation

    new_result, update_result = _run_concurrently(
        postgres_session_factory, prepare(new_request), prepare(update_request)
    )
    assert new_result[0] == update_result[0] == "ok", (new_result, update_result)
    with postgres_session_factory() as db:
        new_version = db.get(models.TrainingDatasetVersion, new_result[1])
        new_holdout = _texts_in_split(db, new_version, "test")
        assert new_holdout and new_holdout <= fresh_texts
        assert new_holdout.isdisjoint(prior_training)
        updated = db.get(models.TrainingDatasetVersion, update_result[1])
        assert updated.parent_version_id == parent_id
        assert _texts_in_split(db, updated, "test") == prior_holdout
        # A later update must also work when the new series won the race last.
        subsequent = prepare_training_dataset(
            db,
            project_id,
            update_request.model_copy(
                update={
                    "parent_version_id": updated.id,
                    "idempotency_key": "first-series-subsequent-update",
                }
            ),
            db.get(User, user_id),
        ).training_dataset_version
        assert subsequent.version_number == 3
        assert _texts_in_split(db, subsequent, "test") == prior_holdout
