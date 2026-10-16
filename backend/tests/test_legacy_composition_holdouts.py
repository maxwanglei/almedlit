"""Legacy composition must preserve held-out source identities as datasets change."""

import pytest
from test_legacy_training_holdouts import _prepared_source
from test_training_preparation import _context, _preview, _records, _source

from al_medlit.auth.models import User
from al_medlit.corpus.models import Document
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


def _legacy_context(client):
    project, task = _context(client)
    response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": task["task_definition_id"],
        "task_kind": task["task_kind"], "output_schema": task["output_schema"],
        "input_schema": {**task["input_schema"], "additionalProperties": True},
    })
    assert response.status_code == 201, response.text
    return project, response.json()


def _compose_source(client, db, project, task, source, name, **settings):
    items = db.query(models.DatasetItem).filter_by(dataset_version_id=source["id"]).all()
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    published = label_service.create_label_set_version(db, schemas.LabelSetVersionCreate(
        project_id=project["id"], dataset_version_id=source["id"],
        task_version_id=task["id"], name=f"{name} labels", source_kind="imported",
        labels={item.stable_key: "yes" for item in items},
    ), actor)
    response = client.post("/api/datasets/training-versions/compose", json={
        "project_id": project["id"], "name": name,
        "dataset_version_id": source["id"], "task_version_id": task["id"],
        "input_field": "text", "label_set_version_id": published.id,
        "train_percent": 80, "validation_percent": 10, "seed": 42, **settings,
    })
    assert response.status_code == 201, response.text
    return dict(db.get(models.SplitMap, response.json()["split_map_id"]).assignments)


def _corpus_source(client, project, name, document_ids, *, dataset_id=None):
    if dataset_id is None:
        response = client.post("/api/datasets", json={
            "project_id": project["id"], "name": name, "source_type": "project_corpus",
        })
        assert response.status_code == 201, response.text
        dataset_id = response.json()["id"]
    response = client.post(
        f"/api/projects/{project['id']}/datasets/{dataset_id}/versions/project-corpus",
        json={"document_ids": document_ids},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize("source_kind", [
    "upload-revision", "upload-regrouped-and-reseeded", "corpus-revision", "new-corpus-source",
    "new-corpus-source-edited",
])
def test_legacy_composition_retains_test_documents_when_source_grows(
    client, db, source_kind,
):
    project, task = _legacy_context(client)
    original = _records("source", 5)
    added = _records("new", 5)
    if source_kind.startswith("upload"):
        first = _source(client, project, "Growing source", original)
    else:
        document_ids = []
        for record in original:
            response = client.post("/api/documents", json={
                "project_id": project["id"], "external_id": record["key"],
                "text": record["text"], "source": "test",
            })
            assert response.status_code == 200, response.text
            document_ids.append(response.json()["id"])
        first = _corpus_source(client, project, "Growing corpus", document_ids)
    first_assignments = _compose_source(client, db, project, task, first, "First legacy snapshot")
    held_out = {key for key, split in first_assignments.items() if split == "test"}
    assert held_out

    if source_kind.startswith("upload"):
        revised = list(reversed([*original, *added]))
        if source_kind == "upload-regrouped-and-reseeded":
            revised = [{**record, "group": f"regrouped-{record['key']}"} for record in revised]
        second = _source(
            client, project, "Growing source", revised,
            dataset_id=first["dataset_id"], revision="2",
        )
    else:
        for record in added:
            response = client.post("/api/documents", json={
                "project_id": project["id"], "external_id": record["key"],
                "text": record["text"], "source": "test",
            })
            assert response.status_code == 200, response.text
            document_ids.append(response.json()["id"])
        if source_kind == "new-corpus-source-edited":
            for key in held_out:
                document = db.get(Document, int(key.removeprefix("project-document:")))
                document.text += " Updated source text."
                document.metadata_ = {"revision": "changed-after-holdout"}
            db.commit()
        second = _corpus_source(
            client, project, "Grown corpus", list(reversed(document_ids)),
            dataset_id=first["dataset_id"] if source_kind == "corpus-revision" else None,
        )
        if source_kind == "new-corpus-source-edited":
            for key in held_out:
                versions = db.query(models.DatasetItem).filter(
                    models.DatasetItem.dataset_version_id.in_([first["id"], second["id"]]),
                    models.DatasetItem.stable_key == key,
                ).all()
                assert len({item.content_hash for item in versions}) == 2
    settings = (
        {"seed": 91, "train_percent": 60, "validation_percent": 20}
        if source_kind == "upload-regrouped-and-reseeded" else {}
    )
    second_assignments = _compose_source(
        client, db, project, task, second, "Second legacy snapshot", **settings,
    )
    assert {key: second_assignments[key] for key in held_out} == {
        key: "test" for key in held_out
    }
    assert {key for key, split in second_assignments.items() if split == "test"} == held_out


def test_legacy_composition_retains_uploaded_test_documents_when_positional_keys_shift(
    client, db,
):
    project, task = _legacy_context(client)
    response = client.post("/api/datasets", json={
        "project_id": project["id"], "name": "Positional CSV", "source_type": "upload",
    })
    assert response.status_code == 201, response.text
    dataset_id = response.json()["id"]
    path = f"/api/projects/{project['id']}/datasets/{dataset_id}/versions/upload"
    first = client.post(path, files={
        "file": ("papers.csv", b"text,label\nPaper A,yes\nPaper B,yes\nPaper C,yes\n", "text/csv"),
    })
    assert first.status_code == 201, first.text
    first_assignments = _compose_source(
        client, db, project, task, first.json(), "Original CSV training", seed=0,
    )
    first_by_text = {
        item.payload["text"]: first_assignments[item.stable_key]
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=first.json()["id"])
    }
    held_out = {text for text, split in first_by_text.items() if split == "test"}
    assert held_out
    second = client.post(path, files={
        "file": (
            "papers.csv", b"text,label\nNew paper,yes\nPaper A,yes\nPaper B,yes\nPaper C,yes\n",
            "text/csv",
        ),
    })
    assert second.status_code == 201, second.text
    second_assignments = _compose_source(
        client, db, project, task, second.json(), "Updated CSV training", seed=0,
    )
    second_by_text = {
        item.payload["text"]: second_assignments[item.stable_key]
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=second.json()["id"])
    }
    assert {text: second_by_text[text] for text in held_out} == {text: "test" for text in held_out}


def test_legacy_composition_does_not_apply_other_task_holdouts_to_copied_source(client, db):
    project, task = _legacy_context(client)
    original = _records("source", 5)
    first = _source(client, project, "Original task source", original)
    first_assignments = _compose_source(client, db, project, task, first, "Original task training")
    held_out = {key for key, split in first_assignments.items() if split == "test"}
    response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": task["task_definition_id"],
        "task_kind": task["task_kind"], "input_schema": task["input_schema"],
        "output_schema": {"type": "string", "enum": ["yes", "no", "unsure"]},
    })
    assert response.status_code == 201, response.text
    second = _source(
        client, project, "Different task source", list(reversed([*original, *_records("new", 5)])),
    )
    second_assignments = _compose_source(
        client, db, project, response.json(), second, "Different task training",
    )
    assert held_out
    assert {second_assignments[key] for key in held_out} == {"train"}


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
