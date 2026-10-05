"""Legacy composition must respect the governed partitions of prepared series."""

import pytest
from test_legacy_training_holdouts import _prepared_source
from test_training_preparation import _preview, _source

from al_medlit.auth.models import User
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels as label_service


@pytest.mark.parametrize("new_source_revision", [False, True])
def test_legacy_composition_retains_prepared_test_and_does_not_poison_update(
    client, db, new_source_revision,
):
    project, task, source, labels, prepared, payload, source_splits = _prepared_source(client, db)
    if new_source_revision:
        records = [
            {"key": item.stable_key, "group": f"new-{item.stable_key}", **item.payload}
            for item in db.query(models.DatasetItem)
            .filter(models.DatasetItem.dataset_version_id == source["id"])
            .all()
        ]
        source = _source(
            client, project, "Source revision", records,
            dataset_id=source["dataset_id"], revision="2",
        )
        actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
        labels = label_service.create_label_set_version(
            db,
            schemas.LabelSetVersionCreate(
                project_id=project["id"], dataset_version_id=source["id"],
                task_version_id=task["id"], name="Revised labels", source_kind="imported",
                labels=labels.labels,
            ),
            actor,
        )
    response = client.post("/api/datasets/training-versions/compose", json={
        "project_id": project["id"],
        "name": "Legacy composition",
        "dataset_version_id": source["id"],
        "task_version_id": task["id"],
        "input_field": "text",
        "label_set_version_id": labels.id,
        "train_percent": 80,
        "validation_percent": 10,
        "seed": 42,
    })
    assert response.status_code == 201, response.text
    legacy_split = db.get(models.SplitMap, response.json()["split_map_id"])
    assert {key for key, split in legacy_split.assignments.items() if split == "test"} == {
        key for key, split in source_splits.items() if split == "test"
    }
    update = {
        **payload,
        "sources": [{"dataset_version_id": source["id"], "label_set_version_id": labels.id}],
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "prepared-next",
    }
    preview = _preview(client, project, update)
    assert preview["ready"], preview["issues"]


def test_legacy_composition_does_not_apply_another_tasks_prepared_holdouts(client, db):
    project, task, source, labels, prepared, payload, source_splits = _prepared_source(client, db)
    response = client.post("/api/tasks/versions", json={
        "project_id": project["id"],
        "task_definition_id": task["task_definition_id"],
        "task_kind": "classification",
        "input_schema": task["input_schema"],
        "output_schema": {"type": "string", "enum": ["yes", "no", "unsure"]},
    })
    assert response.status_code == 201, response.text
    other_task = response.json()
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    other_labels = label_service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project["id"], dataset_version_id=source["id"],
            task_version_id=other_task["id"], name="Other task labels", source_kind="imported",
            labels=labels.labels,
        ),
        actor,
    )
    response = client.post("/api/datasets/training-versions/compose", json={
        "project_id": project["id"], "name": "Other task composition",
        "dataset_version_id": source["id"], "task_version_id": other_task["id"],
        "input_field": "text", "label_set_version_id": other_labels.id,
        "train_percent": 80, "validation_percent": 10, "seed": 42,
    })
    assert response.status_code == 201, response.text
    other_split = db.get(models.SplitMap, response.json()["split_map_id"])
    assert {key for key, split in other_split.assignments.items() if split == "test"} != {
        key for key, split in source_splits.items() if split == "test"
    }
    update = {
        **payload,
        "training_dataset_id": prepared["training_dataset"]["id"],
        "parent_version_id": prepared["training_dataset_version"]["id"],
        "idempotency_key": "update-after-other-task-composition",
    }
    preview = _preview(client, project, update)
    assert preview["ready"], preview["issues"]
