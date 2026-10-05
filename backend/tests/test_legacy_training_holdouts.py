"""Legacy training entry points must honor holdouts established by preparation."""

import pytest
from test_training_preparation import _context, _prepare, _preview, _records, _request, _source

from al_medlit.auth.models import User
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels
from al_medlit.workflow.services.common import _canonical_hash


def _prepared_source(client, db):
    project, task = _context(client)
    records = _records("source", 20)
    source = _source(
        client,
        project,
        "Reusable training source",
        [{"key": row["key"], "text": row["text"]} for row in records],
    )
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    published = labels.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project["id"],
            dataset_version_id=source["id"],
            task_version_id=task["id"],
            name="Published source labels",
            source_kind="imported",
            labels={row["key"]: row["label"] for row in records},
        ),
        actor,
    )
    payload = _request(
        task,
        sources=[{"dataset_version_id": source["id"], "label_set_version_id": published.id}],
    )
    prepared = _prepare(client, project, payload)
    version = prepared["training_dataset_version"]
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    split = db.get(models.SplitMap, version["split_map_id"])
    source_assignments = {
        origin["stable_key"]: split.assignments[key]
        for key, provenance in generated.provenance["items"].items()
        for origin in provenance["origins"]
        if origin["dataset_version_id"] == source["id"]
    }
    return project, task, source, published, prepared, payload, source_assignments


def _legacy_version_request(project, task, source, published, split_id):
    return {
        "project_id": project["id"],
        "dataset_version_id": source["id"],
        "task_version_id": task["id"],
        "name": "Legacy training examples",
        "split_map_id": split_id,
        "label_set_version_ids": [published.id],
        "preprocessing": {"input_field": "text"},
    }


@pytest.mark.parametrize("conflicting_split", ["train", "validation", "pool"])
def test_split_creation_rejects_prepared_holdout_reassignment(
    client, db, conflicting_split
):
    project, task, source, published, prepared, payload, assignments = _prepared_source(client, db)
    held_out_key = next(key for key, split in assignments.items() if split == "test")
    assignments[held_out_key] = conflicting_split
    counts = [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ]
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": source["id"],
            "name": "Conflicting legacy split",
            "strategy": "manual",
            "assignments": assignments,
        },
    )
    assert response.status_code == 422, response.text
    assert "protected" in response.text.lower()
    assert [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ] == counts
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-rejected-split",
    }
    assert _preview(client, project, update)["ready"]
    assert _prepare(client, project, update)["training_dataset_version"]["version_number"] == 2


@pytest.mark.parametrize("conflicting_split", ["train", "validation", "pool"])
def test_direct_legacy_version_rejects_historical_prepared_holdout_reassignment(
    client, db, conflicting_split
):
    project, task, source, published, prepared, payload, assignments = _prepared_source(client, db)
    held_out_key = next(key for key, split in assignments.items() if split == "test")
    assignments[held_out_key] = conflicting_split
    assert "test" in assignments.values()
    # Existing databases can contain maps created before these safeguards.
    split = models.SplitMap(
        project_id=project["id"],
        dataset_version_id=source["id"],
        name="Historical conflicting legacy split",
        strategy="manual",
        assignments=assignments,
        protected_splits=["test"],
        content_hash=_canonical_hash(assignments),
    )
    db.add(split)
    db.commit()
    counts = [
        db.query(model).count() for model in (models.TrainingDataset, models.TrainingDatasetVersion)
    ]
    response = client.post(
        "/api/datasets/training-versions",
        json=_legacy_version_request(project, task, source, published, split.id),
    )
    assert response.status_code == 422, response.text
    assert "protected" in response.text.lower()
    assert [
        db.query(model).count() for model in (models.TrainingDataset, models.TrainingDatasetVersion)
    ] == counts
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-rejected-legacy-version",
    }
    assert _preview(client, project, update)["ready"]
    assert _prepare(client, project, update)["training_dataset_version"]["version_number"] == 2


def test_split_creation_cannot_newly_protect_a_prepared_training_item(client, db):
    project, task, source, published, prepared, payload, assignments = _prepared_source(client, db)
    training_key = next(key for key, split in assignments.items() if split == "train")
    assignments[training_key] = "test"
    counts = [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ]
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": source["id"],
            "name": "Legacy split with an extra protected test item",
            "strategy": "manual",
            "assignments": assignments,
        },
    )
    assert response.status_code == 422, response.text
    assert "protected" in response.text.lower()
    assert [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ] == counts
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-rejected-extra-holdout",
    }
    assert _preview(client, project, update)["ready"]
    assert _prepare(client, project, update)["training_dataset_version"]["version_number"] == 2


@pytest.mark.parametrize("protected_split", ["train", "validation"])
def test_split_creation_cannot_declare_prepared_learning_partition_protected(
    client, db, protected_split
):
    project, task, source, published, prepared, payload, assignments = _prepared_source(client, db)
    counts = [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ]
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": source["id"],
            "name": "Legacy split declaring learning data protected",
            "strategy": "manual",
            "assignments": assignments,
            "protected_splits": ["test", protected_split],
        },
    )
    assert response.status_code == 422, response.text
    assert "protected" in response.text.lower()
    assert [
        db.query(model).count()
        for model in (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    ] == counts
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-rejected-protected-learning-partition",
    }
    assert _preview(client, project, update)["ready"]


def test_direct_legacy_version_can_reuse_consistent_prepared_source_splits(client, db):
    project, task, source, published, prepared, payload, assignments = _prepared_source(client, db)
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": source["id"],
            "name": "Consistent legacy split",
            "strategy": "manual",
            "assignments": assignments,
        },
    )
    assert response.status_code == 201, response.text
    response = client.post(
        "/api/datasets/training-versions",
        json=_legacy_version_request(project, task, source, published, response.json()["id"]),
    )
    assert response.status_code == 201, response.text
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-consistent-legacy-version",
    }
    assert _preview(client, project, update)["ready"]
    assert _prepare(client, project, update)["training_dataset_version"]["version_number"] == 2
