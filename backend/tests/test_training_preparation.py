"""User journeys and invariants for immutable multi-source training preparation."""

import copy

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.lineage.models import ImmutableRecordError
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workflow.services.preparation import is_source_item_in_prepared_holdout
from al_medlit.workspace.models import WorkspaceMember


def _context(client):
    response = client.post("/api/projects", json={"name": "Preparation"})
    assert response.status_code == 200, response.text
    project = response.json()
    definition = client.post(
        "/api/tasks",
        json={"project_id": project["id"], "key": "classify", "name": "Classification"},
    )
    assert definition.status_code == 201, definition.text
    task = client.post(
        "/api/tasks/versions",
        json={
            "project_id": project["id"],
            "task_definition_id": definition.json()["id"],
            "task_kind": "classification",
            "input_schema": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string", "minLength": 1}},
                "additionalProperties": False,
            },
            "output_schema": {"type": "string", "enum": ["yes", "no"]},
        },
    )
    assert task.status_code == 201, task.text
    return project, task.json()


def _source(client, project, name, records, *, dataset_id=None, revision="1"):
    if dataset_id is None:
        response = client.post(
            "/api/datasets",
            json={"project_id": project["id"], "name": name, "source_type": "upload"},
        )
        assert response.status_code == 201, response.text
        dataset_id = response.json()["id"]
    response = client.post(
        "/api/datasets/versions",
        json={
            "project_id": project["id"],
            "dataset_id": dataset_id,
            "source_revision": revision,
            "source_format": "jsonl",
            "items": [
                {
                    "stable_key": record["key"],
                    "group_key": record.get("group"),
                    "payload": {
                        key: value for key, value in record.items() if key not in {"key", "group"}
                    },
                }
                for record in records
            ],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _records(prefix, size):
    return [
        {"key": f"{prefix}-{index}", "text": f"{prefix} paper {index}", "label": "yes"}
        for index in range(size)
    ]


def _request(task, *sources, **extra):
    return {
        "name": "Training examples",
        "task_version_id": task["id"],
        "sources": [
            {"dataset_version_id": source["id"], "label_field": "label"} for source in sources
        ],
        "idempotency_key": "prepare-once",
        **extra,
    }


def _preview(client, project, payload):
    response = client.post(f"/api/projects/{project['id']}/training-datasets/preview", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _prepare(client, project, payload):
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _generated_items(db, result):
    return list(
        db.query(models.DatasetItem)
        .filter(
            models.DatasetItem.dataset_version_id
            == result["training_dataset_version"]["dataset_version_id"]
        )
        .order_by(models.DatasetItem.stable_key)
    )


def test_two_small_sources_combine_and_preview_has_no_writes(client, db):
    project, task = _context(client)
    one = _source(client, project, "Source A", _records("A", 2))
    two = _source(client, project, "Source B", _records("B", 2))
    assert not _preview(client, project, _request(task, one))["ready"]
    tracked = (
        models.TrainingDataset,
        models.TrainingDatasetVersion,
        models.DatasetVersion,
        models.LabelSetVersion,
        models.SplitMap,
    )
    before = [db.query(model).count() for model in tracked]
    payload = _request(task, one, two)
    preview = _preview(client, project, payload)
    assert preview["ready"] and preview["group_count"] == 4
    assert [db.query(model).count() for model in tracked] == before
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    result = _prepare(client, project, payload)
    version = result["training_dataset_version"]
    assert version["version_number"] == 1 and version["parent_version_id"] is None
    assert result["preview"]["item_count"] == 4
    assert all(
        result["preview"]["split_counts"][split] for split in ("train", "validation", "test")
    )
    rows = _generated_items(db, result)
    assert all(set(item.payload) == {"text"} for item in rows)
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    assert {
        origin["dataset_version_id"]
        for value in generated.provenance["items"].values()
        for origin in value["origins"]
    } == {one["id"], two["id"]}
    assert {source["source_kind"] for source in version["preparation_manifest"]["sources"]} == {
        "imported"
    }
    assert (
        db.get(models.LabelSetVersion, version["label_set_version_ids"][0]).source_kind
        == "composed"
    )
    datasets = client.get("/api/datasets", params={"project_id": project["id"]})
    assert datasets.status_code == 200, datasets.text
    listed = client.get(f"/api/projects/{project['id']}/training-datasets")
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["latest_version_id"] == version["id"]
    assert listed.json()[0]["versions"][0]["id"] == version["id"]


def test_external_missing_null_and_blank_labels_are_excluded_but_invalid_labels_are_reported(
    client,
):
    project, task = _context(client)
    source = _source(
        client,
        project,
        "Partially labeled source",
        [
            *_records("labeled", 3),
            {"key": "missing", "text": "not annotated yet"},
            {"key": "null", "text": "empty JSON field", "label": None},
            {"key": "blank", "text": "empty CSV cell", "label": ""},
            {"key": "spaces", "text": "whitespace CSV cell", "label": " \t"},
        ],
    )
    preview = _preview(client, project, _request(task, source))
    assert preview["ready"]
    assert preview["input_count"] == 7 and preview["labeled_count"] == 3
    assert preview["excluded_unlabeled_count"] == 4 and preview["item_count"] == 3
    invalid = _source(
        client,
        project,
        "Invalid label",
        [{"key": "invalid", "text": "a labeled example", "label": "unexpected"}],
    )
    preview = _preview(client, project, _request(task, source, invalid))
    assert not preview["ready"]
    assert "incompatible_example" in {issue["code"] for issue in preview["issues"]}
    assert preview["source_counts"][1]["labeled_count"] == 1
    assert preview["source_counts"][1]["excluded_unlabeled_count"] == 0


def test_preparation_retries_return_same_version_and_changed_key_inputs_conflict(client, db):
    project, task = _context(client)
    source = _source(client, project, "External", _records("external", 5))
    payload = _request(task, source)
    first = _prepare(client, project, payload)
    second = _prepare(client, project, payload)
    assert first["training_dataset_version"]["id"] == second["training_dataset_version"]["id"]
    assert db.query(models.TrainingDataset).count() == 1
    assert db.query(models.TrainingDatasetVersion).count() == 1
    response = client.post(
        f"/api/projects/{project['id']}/training-datasets/prepare", json={**payload, "seed": 9}
    )
    assert response.status_code == 409


def test_existing_label_snapshot_name_cannot_substitute_different_labels(client, db):
    project, task = _context(client)
    records = _records("source", 3)
    source = _source(client, project, "External", records)
    expected = {row["key"]: row["label"] for row in records}
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    collision = labels.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project["id"],
            dataset_version_id=source["id"],
            task_version_id=task["id"],
            name=f"Prepared source {_canonical_hash(expected)[:16]}",
            source_kind="imported",
            labels={row["key"]: "no" for row in records},
        ),
        actor,
    )
    result = _prepare(client, project, _request(task, source))
    ref = result["training_dataset_version"]["preparation_manifest"]["sources"][0]
    snapshot = db.get(models.LabelSetVersion, ref["label_set_version_id"])
    assert snapshot.id != collision.id
    assert snapshot.labels == expected
    assert snapshot.version_number == collision.version_number + 1


def test_uploaded_provenance_does_not_assert_prepared_source_ancestry(client, db):
    project, task = _context(client)
    dataset = client.post(
        "/api/datasets",
        json={"project_id": project["id"], "name": "External", "source_type": "upload"},
    ).json()
    response = client.post(
        "/api/datasets/versions",
        json={
            "project_id": project["id"],
            "dataset_id": dataset["id"],
            "source_revision": "1",
            "source_format": "jsonl",
            "provenance": {
                "ingestion": "training_preparation_v1",
                "items": {"source-0": {"origins": [{"dataset_item_id": 999999}]}},
            },
            "items": [
                {"stable_key": row["key"], "payload": {"text": row["text"], "label": row["label"]}}
                for row in _records("source", 3)
            ],
        },
    )
    assert response.status_code == 201, response.text
    source = response.json()
    result = _prepare(client, project, _request(task, source))
    generated = db.get(
        models.DatasetVersion, result["training_dataset_version"]["dataset_version_id"]
    )
    assert all(
        origin["dataset_item_id"] != 999999
        for value in generated.provenance["items"].values()
        for origin in value["origins"]
    )


def test_duplicates_keep_both_origins_and_conflicts_block_atomically(client, db):
    project, task = _context(client)
    one = _source(client, project, "One", _records("one", 3))
    duplicate = {"key": "copy", "text": "one paper 0", "label": "yes"}
    two = _source(client, project, "Two", [duplicate, *_records("two", 1)])
    result = _prepare(client, project, _request(task, one, two))
    assert result["preview"]["duplicate_count"] == 1
    assert result["preview"]["item_count"] == 4
    provenance = db.get(
        models.DatasetVersion, result["training_dataset_version"]["dataset_version_id"]
    ).provenance
    assert any(len(item["origins"]) == 2 for item in provenance["items"].values())
    conflict = _source(client, project, "Conflicts", [{**duplicate, "label": "no"}])
    bad = _request(task, one, conflict, name="Conflict", idempotency_key="conflict")
    preview = _preview(client, project, bad)
    assert not preview["ready"]
    assert "label_conflict" in {issue["code"] for issue in preview["issues"]}
    before = db.query(models.DatasetVersion).count()
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=bad)
    assert response.status_code == 422
    assert db.query(models.DatasetVersion).count() == before
    assert db.query(models.TrainingDataset).count() == 1


def _submit_subset(client, project, task, source, labels, *, excluded_keys=()):
    user_id = client.get("/api/auth/me").json()["user"]["id"]
    created = client.post(
        "/api/rounds",
        json={
            "project_id": project["id"],
            "name": "Ongoing annotation",
            "dataset_version_id": source["id"],
            "task_version_id": task["id"],
            "assistance_policy": "blind",
            "reannotation_mode": "full_dataset",
            "annotator_user_ids": [user_id],
        },
    )
    assert created.status_code == 201, created.text
    round_id = created.json()["id"]
    opened = client.post(
        f"/api/rounds/{round_id}/transition",
        params={"project_id": project["id"]},
        json={"status": "open"},
    )
    assert opened.status_code == 200, opened.text
    items = client.get(
        f"/api/rounds/{round_id}/work-items", params={"project_id": project["id"]}
    ).json()
    items = [item for item in items if item["dataset_item"]["stable_key"] not in excluded_keys]
    decisions = []
    for item, label in zip(items, labels, strict=False):
        response = client.post(
            "/api/rounds/decisions",
            json={
                "project_id": project["id"],
                "round_item_id": item["round_item"]["id"],
                "output": label,
            },
        )
        assert response.status_code == 201, response.text
        decisions.append(response.json())
    submitted = client.post(
        "/api/rounds/submissions",
        json={
            "project_id": project["id"],
            "annotation_round_id": round_id,
            "decision_ids": [decision["id"] for decision in decisions],
        },
    )
    assert submitted.status_code == 201, submitted.text
    return round_id, submitted.json(), decisions


def test_open_round_subset_combines_with_external_labels_without_closing(client, db):
    project, task = _context(client)
    source = _source(
        client,
        project,
        "Annotating",
        [{"key": f"a{i}", "text": f"annotating {i}"} for i in range(4)],
    )
    external = _source(client, project, "External", _records("external", 2))
    round_id, submission, decisions = _submit_subset(client, project, task, source, ["yes", "no"])
    payload = _request(task, external)
    payload["sources"].append({"dataset_version_id": source["id"], "annotation_round_id": round_id})
    before = db.query(models.LabelSetVersion).count()
    preview = _preview(client, project, payload)
    assert preview["ready"] and preview["excluded_unlabeled_count"] == 2
    assert db.query(models.LabelSetVersion).count() == before
    payload["sources"] = preview["resolved_sources"]
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    result = _prepare(client, project, payload)
    assert db.get(models.AnnotationRound, round_id).status == "open"
    assert result["preview"]["item_count"] == 4
    refs = result["training_dataset_version"]["preparation_manifest"]["sources"]
    human_ref = next(ref for ref in refs if ref["source_kind"] == "human")
    human = db.get(models.LabelSetVersion, human_ref["label_set_version_id"])
    assert human.source_submission_ids == [submission["id"]]
    assert human.source_decision_ids == sorted(decision["id"] for decision in decisions)
    assert human.label_count == 2


def test_new_versions_retain_prior_holdout_and_immutable_previous_data(client, db):
    project, task = _context(client)
    records = _records("paper", 5)
    source = _source(client, project, "Sources", records)
    first = _prepare(client, project, _request(task, source))
    old = first["training_dataset_version"]
    old_hash = old["content_hash"]
    old_split = db.get(models.SplitMap, old["split_map_id"])
    old_assignments = copy.deepcopy(old_split.assignments)
    updated_source = _source(
        client,
        project,
        "Sources",
        [*records, *_records("new", 3)],
        dataset_id=source["dataset_id"],
        revision="2",
    )
    payload = _request(
        task,
        updated_source,
        training_dataset_id=first["training_dataset"]["id"],
        parent_version_id=old["id"],
        idempotency_key="version-two",
    )
    second = _prepare(client, project, payload)
    new = second["training_dataset_version"]
    assert new["version_number"] == 2 and new["parent_version_id"] == old["id"]
    new_assignments = db.get(models.SplitMap, new["split_map_id"]).assignments
    assert all(new_assignments[key] == split for key, split in old_assignments.items())
    assert {key for key, split in new_assignments.items() if split == "test"} == {
        key for key, split in old_assignments.items() if split == "test"
    }
    db.expire_all()
    assert db.get(models.TrainingDatasetVersion, old["id"]).content_hash == old_hash
    assert db.get(models.SplitMap, old["split_map_id"]).assignments == old_assignments
    stale = client.post(
        f"/api/projects/{project['id']}/training-datasets/prepare",
        json={**payload, "idempotency_key": "stale-parent"},
    )
    assert stale.status_code == 409
    immutable = db.get(models.TrainingDatasetVersion, old["id"])
    immutable.name = "Overwritten"
    with pytest.raises(ImmutableRecordError):
        db.flush()
    db.rollback()


@pytest.mark.parametrize("duplicate_name", [False, True])
def test_legacy_training_series_updates_across_source_versions(client, db, duplicate_name):
    project, task = _context(client)
    records = _records("legacy", 5)
    source = _source(
        client,
        project,
        "Legacy source",
        [{"key": row["key"], "text": row["text"]} for row in records],
    )
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    published = labels.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project["id"],
            dataset_version_id=source["id"],
            task_version_id=task["id"],
            name="Legacy published labels",
            source_kind="imported",
            labels={row["key"]: row["label"] for row in records},
        ),
        actor,
    )
    assignments = {
        "legacy-0": "train",
        "legacy-1": "train",
        "legacy-2": "train",
        "legacy-3": "validation",
        "legacy-4": "test",
    }
    split = labels.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project["id"],
            dataset_version_id=source["id"],
            name="Legacy split",
            strategy="manual",
            assignments=assignments,
        ),
        actor,
    )
    legacy_request = {
        "project_id": project["id"],
        "dataset_version_id": source["id"],
        "task_version_id": task["id"],
        "name": "Training examples",
        "split_map_id": split.id,
        "label_set_version_ids": [published.id],
        "preprocessing": {"input_field": "text"},
    }
    response = client.post("/api/datasets/training-versions", json=legacy_request)
    assert response.status_code == 201, response.text
    if duplicate_name:
        legacy_request["preprocessing"]["legacy_copy"] = True
        response = client.post("/api/datasets/training-versions", json=legacy_request)
        assert response.status_code == 201, response.text
    legacy = response.json()
    assert legacy["training_dataset_id"] is not None
    assert legacy["version_number"] == 1
    canonical_name = db.get(models.TrainingDataset, legacy["training_dataset_id"]).name
    assert (canonical_name != legacy["name"]) == duplicate_name
    updated_source = _source(
        client,
        project,
        "Legacy source",
        [*records, *_records("new", 3)],
        dataset_id=source["dataset_id"],
        revision="2",
    )
    payload = _request(
        task,
        updated_source,
        name=legacy["name"],
        training_dataset_id=legacy["training_dataset_id"],
        parent_version_id=legacy["id"],
    )
    arbitrary_rename = client.post(
        f"/api/projects/{project['id']}/training-datasets/preview",
        json={**payload, "name": "Unrelated new name"},
    )
    assert arbitrary_rename.status_code == 422
    preview = _preview(client, project, payload)
    assert preview["ready"]
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    result = _prepare(client, project, payload)
    current = result["training_dataset_version"]
    assert current["version_number"] == 2 and current["parent_version_id"] == legacy["id"]
    assert current["name"] == result["training_dataset"]["name"] == canonical_name
    assert _prepare(client, project, payload)["training_dataset_version"]["id"] == current["id"]
    generated = db.get(models.DatasetVersion, current["dataset_version_id"])
    current_split = db.get(models.SplitMap, current["split_map_id"])
    for key, provenance in generated.provenance["items"].items():
        for origin in provenance["origins"]:
            old_split = assignments.get(origin["stable_key"])
            if old_split:
                assert current_split.assignments[key] == old_split
    assert sum(value == "test" for value in current_split.assignments.values()) == 1
    assert (
        db.get(models.TrainingDatasetVersion, legacy["id"]).content_hash == legacy["content_hash"]
    )
    assert db.get(models.TrainingDatasetVersion, legacy["id"]).name == legacy["name"]


@pytest.mark.parametrize("change", ["remove", "remap"])
def test_new_version_rejects_removed_or_remapped_test_inputs(client, db, change):
    project, task = _context(client)
    records = [
        {**record, "alternate": f"different input {index}"}
        for index, record in enumerate(_records("paper", 5))
    ]
    source = _source(client, project, "Source", records)
    first = _prepare(client, project, _request(task, source))
    parent = first["training_dataset_version"]
    payload = _request(
        task,
        source,
        training_dataset_id=first["training_dataset"]["id"],
        parent_version_id=parent["id"],
        idempotency_key="changed-input",
    )
    if change == "remap":
        payload["sources"][0]["input_mapping"] = {"text": "alternate"}
    else:
        generated = db.get(models.DatasetVersion, parent["dataset_version_id"])
        split = db.get(models.SplitMap, parent["split_map_id"])
        protected_keys = {
            origin["stable_key"]
            for key, provenance in generated.provenance["items"].items()
            if split.assignments[key] == "test"
            for origin in provenance["origins"]
        }
        updated_source = _source(
            client,
            project,
            "Source",
            [row for row in records if row["key"] not in protected_keys],
            dataset_id=source["dataset_id"],
            revision="2",
        )
        payload["sources"][0]["dataset_version_id"] = updated_source["id"]
    preview = _preview(client, project, payload)
    assert not preview["ready"]
    assert "missing_holdout" in {issue["code"] for issue in preview["issues"]}
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 422, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 1


def test_shared_article_stays_together_and_prepared_holdout_maps_to_source(client, db):
    project, task = _context(client)
    one = _source(
        client,
        project,
        "First",
        [
            {"key": "a", "text": "first passage", "pmid": "123", "label": "yes"},
            {"key": "b", "text": "second paper", "pmid": "456", "label": "yes"},
        ],
    )
    two = _source(
        client,
        project,
        "Second",
        [
            {"key": "c", "text": "another passage", "pmid": "123", "label": "yes"},
            {"key": "d", "text": "third paper", "pmid": "789", "label": "yes"},
        ],
    )
    result = _prepare(client, project, _request(task, one, two))
    assert result["preview"]["group_count"] == 3
    generated = db.get(
        models.DatasetVersion, result["training_dataset_version"]["dataset_version_id"]
    )
    split = db.get(models.SplitMap, result["training_dataset_version"]["split_map_id"])
    common_splits = set()
    for key, origin in generated.provenance["items"].items():
        if "pmid:123" in origin["identity_tokens"]:
            common_splits.add(split.assignments[key])
        for source in origin["origins"]:
            raw = db.get(models.DatasetItem, source["dataset_item_id"])
            assert is_source_item_in_prepared_holdout(db, raw, task["id"]) == (
                split.assignments[key] == "test"
            )
            assert not is_source_item_in_prepared_holdout(db, raw, task["id"] + 1000)
    assert len(common_splits) == 1


def test_source_split_holdout_survives_source_revision_without_explicit_map_selection(client, db):
    project, task = _context(client)
    records = [{**record, "group": record["key"]} for record in _records("paper", 5)]
    old_source = _source(client, project, "Source", records)
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": old_source["id"],
            "name": "Reserved source test",
            "strategy": "manual",
            "assignments": {
                row["key"]: "test" if row["key"] == "paper-4" else "pool" for row in records
            },
        },
    )
    assert response.status_code == 201, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 0
    source = _source(
        client,
        project,
        "Source",
        [
            *[{**record, "group": f"revised-{record['key']}"} for record in records],
            *_records("new", 2),
        ],
        dataset_id=old_source["dataset_id"],
        revision="2",
    )
    result = _prepare(client, project, _request(task, source))
    generated = db.get(
        models.DatasetVersion, result["training_dataset_version"]["dataset_version_id"]
    )
    assignments = db.get(models.SplitMap, result["training_dataset_version"]["split_map_id"])
    held_out_keys = {
        key
        for key, provenance in generated.provenance["items"].items()
        if any(origin["stable_key"] == "paper-4" for origin in provenance["origins"])
    }
    assert len(held_out_keys) == 1
    assert {assignments.assignments[key] for key in held_out_keys} == {"test"}
    source_item = (
        db.query(models.DatasetItem)
        .filter_by(dataset_version_id=source["id"], stable_key="paper-4")
        .one()
    )
    assert is_source_item_in_prepared_holdout(db, source_item, task["id"])
    missing_test_label = _source(
        client,
        project,
        "Source",
        [{**row, "label": None if row["key"] == "paper-4" else "yes"} for row in records],
        dataset_id=old_source["dataset_id"],
        revision="3",
    )
    preview = _preview(client, project, _request(task, missing_test_label))
    assert not preview["ready"]
    assert "unlabeled_holdout" in {issue["code"] for issue in preview["issues"]}


@pytest.mark.parametrize("holdout_source_first", [True, False])
@pytest.mark.parametrize("matching_input", [True, False])
def test_unlabeled_holdout_uses_label_from_another_source(
    client, db, holdout_source_first, matching_input
):
    project, task = _context(client)
    protected = {"key": "protected", "text": "protected paper", "pmid": "123"}
    source = _source(client, project, "Partially labeled", [*_records("paper", 3), protected])
    split = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": source["id"],
            "name": "Protected test",
            "strategy": "manual",
            "assignments": {
                "paper-0": "train",
                "paper-1": "train",
                "paper-2": "validation",
                protected["key"]: "test",
            },
        },
    )
    assert split.status_code == 201, split.text
    labeled_source = _source(
        client,
        project,
        "Holdout labels",
        [
            {
                **protected,
                "key": "labeled-copy",
                "text": protected["text"]
                if matching_input
                else "different text from the same paper",
                "label": "no",
            }
        ],
    )
    sources = (source, labeled_source) if holdout_source_first else (labeled_source, source)
    payload = _request(task, *sources)
    next(
        selected
        for selected in payload["sources"]
        if selected["dataset_version_id"] == source["id"]
    )["split_map_id"] = split.json()["id"]
    preview = _preview(client, project, payload)
    if not matching_input:
        assert not preview["ready"]
        issue = next(issue for issue in preview["issues"] if issue["code"] == "unlabeled_holdout")
        assert issue["item_key"] == protected["key"]
        assert issue["source_index"] == (0 if holdout_source_first else 1)
        response = client.post(
            f"/api/projects/{project['id']}/training-datasets/prepare", json=payload
        )
        assert response.status_code == 422, response.text
        assert db.query(models.TrainingDatasetVersion).count() == 0
        return
    assert preview["ready"], preview["issues"]
    assert preview["item_count"] == 4
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    result = _prepare(client, project, payload)
    version = result["training_dataset_version"]
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    assignments = db.get(models.SplitMap, version["split_map_id"]).assignments
    protected_item = next(
        item for item in _generated_items(db, result) if item.payload == {"text": protected["text"]}
    )
    assert assignments[protected_item.stable_key] == "test"
    assert {
        origin["dataset_version_id"]
        for origin in generated.provenance["items"][protected_item.stable_key]["origins"]
    } == {source["id"], labeled_source["id"]}
    snapshot = db.get(models.LabelSetVersion, version["label_set_version_ids"][0])
    assert snapshot.labels[protected_item.stable_key] == "no"


@pytest.mark.parametrize("omit_train_label", [True, False])
def test_review_round_retraining_reuses_parent_holdout_labels(client, db, omit_train_label):
    project, task = _context(client)
    records = _records("paper", 5)
    source = _source(client, project, "Reviewed source", records)
    first = _prepare(client, project, _request(task, source))
    parent = first["training_dataset_version"]
    parent_split = copy.deepcopy(db.get(models.SplitMap, parent["split_map_id"]).assignments)
    generated = db.get(models.DatasetVersion, parent["dataset_version_id"])
    protected_keys = {
        origin["stable_key"]
        for key, provenance in generated.provenance["items"].items()
        if parent_split[key] == "test"
        for origin in provenance["origins"]
    }
    omitted_key = next(key for key, partition in parent_split.items() if partition == "train")
    excluded_keys = set(protected_keys)
    if omit_train_label:
        excluded_keys.update(
            origin["stable_key"] for origin in generated.provenance["items"][omitted_key]["origins"]
        )
    round_id, _, decisions = _submit_subset(
        client, project, task, source, ["no"] * len(records), excluded_keys=excluded_keys
    )
    assert len(decisions) < len(records)
    payload = _request(
        task,
        training_dataset_id=first["training_dataset"]["id"],
        parent_version_id=parent["id"],
        idempotency_key="review-retrain",
    )
    payload["sources"] = [{"dataset_version_id": source["id"], "annotation_round_id": round_id}]
    preview = _preview(client, project, payload)
    assert preview["ready"], preview["issues"]
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    result = _prepare(client, project, payload)
    version = result["training_dataset_version"]
    assert result["preview"]["item_count"] == len(records) - int(omit_train_label)
    assignments = db.get(models.SplitMap, version["split_map_id"]).assignments
    assert assignments == {
        key: partition
        for key, partition in parent_split.items()
        if not omit_train_label or key != omitted_key
    }
    parent_labels = db.get(models.LabelSetVersion, parent["label_set_version_ids"][0]).labels
    current_labels = db.get(models.LabelSetVersion, version["label_set_version_ids"][0]).labels
    for key, partition in parent_split.items():
        if partition == "test":
            assert current_labels[key] == parent_labels[key]
        elif omit_train_label and key == omitted_key:
            assert key not in current_labels
        else:
            assert current_labels[key] == "no"


def test_explicit_split_conflict_is_actionable(client):
    project, task = _context(client)
    one = _source(
        client,
        project,
        "First",
        [
            {"key": "a", "text": "one", "pmid": "123", "label": "yes"},
            {"key": "b", "text": "two", "label": "yes"},
        ],
    )
    two = _source(
        client,
        project,
        "Second",
        [
            {"key": "c", "text": "three", "pmid": "123", "label": "yes"},
            {"key": "d", "text": "four", "label": "yes"},
        ],
    )
    maps = []
    for source, assignments in [
        (one, {"a": "test", "b": "train"}),
        (two, {"c": "train", "d": "validation"}),
    ]:
        response = client.post(
            "/api/datasets/split-maps",
            json={
                "project_id": project["id"],
                "dataset_version_id": source["id"],
                "name": "Existing",
                "strategy": "manual",
                "assignments": assignments,
            },
        )
        assert response.status_code == 201, response.text
        maps.append(response.json()["id"])
    payload = _request(task, one, two)
    for source, split_id in zip(payload["sources"], maps, strict=True):
        source["split_map_id"] = split_id
    preview = _preview(client, project, payload)
    assert not preview["ready"]
    assert "split_conflict" in {issue["code"] for issue in preview["issues"]}


def test_trainer_cannot_preview_unpublished_round_submissions(client, db):
    project, task = _context(client)
    source = _source(client, project, "Annotating", _records("source", 3))
    round_id, submission, _ = _submit_subset(client, project, task, source, ["yes", "no", "yes"])
    user = User(username="trainer-preparation", password_hash="unused", is_active=True)
    db.add(user)
    db.flush()
    db.add(WorkspaceMember(workspace_id=project["workspace_id"], user_id=user.id, role="trainer"))
    db.commit()
    headers = {"Authorization": f"Bearer {create_access_token(str(user.id))}"}
    payload = _request(task, source)
    external = client.post(
        f"/api/projects/{project['id']}/training-datasets/preview", json=payload, headers=headers
    )
    assert external.status_code == 200, external.text
    payload["sources"] = [
        {
            "dataset_version_id": source["id"],
            "annotation_round_id": round_id,
            "submission_ids": [submission["id"]],
        }
    ]
    for action in ("preview", "prepare"):
        response = client.post(
            f"/api/projects/{project['id']}/training-datasets/{action}",
            json=payload,
            headers=headers,
        )
        assert response.status_code == 403, response.text


def test_mapped_input_and_label_schema_errors_are_previewed(client):
    project, task = _context(client)
    source = _source(
        client,
        project,
        "Mapped",
        [
            {"key": "one", "abstract": "source text", "label": "unexpected"},
            {"key": "two", "abstract": "more text", "label": "yes"},
            {"key": "three", "abstract": "other text", "label": "yes"},
        ],
    )
    payload = _request(task, source)
    missing = _preview(client, project, payload)
    assert "missing_input" in {issue["code"] for issue in missing["issues"]}
    payload["sources"][0]["input_mapping"] = {"text": "abstract"}
    mismatch = _preview(client, project, payload)
    assert "incompatible_example" in {issue["code"] for issue in mismatch["issues"]}


def test_preview_detects_new_submissions_before_prepare(client, db):
    project, task = _context(client)
    source = _source(client, project, "Annotating", _records("source", 3))
    round_id, _, decisions = _submit_subset(client, project, task, source, ["yes", "yes", "yes"])
    payload = _request(task, source)
    payload["sources"] = [{"dataset_version_id": source["id"], "annotation_round_id": round_id}]
    preview = _preview(client, project, payload)
    revised = client.post(
        "/api/rounds/decisions",
        json={
            "project_id": project["id"],
            "round_item_id": decisions[0]["round_item_id"],
            "supersedes_decision_id": decisions[0]["id"],
            "output": "no",
        },
    ).json()
    submitted = client.post(
        "/api/rounds/submissions",
        json={
            "project_id": project["id"],
            "annotation_round_id": round_id,
            "decision_ids": [revised["id"]],
        },
    )
    assert submitted.status_code == 201, submitted.text
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 409, response.text
    assert db.query(models.TrainingDataset).count() == 0
