"""A later series cannot turn another series' learning articles into holdouts."""

import pytest
from test_training_preparation import _context, _prepare, _preview, _records, _request, _source

from al_medlit.workflow import models
from al_medlit.workflow.services.common import _canonical_hash


def _source_assignments(db, result, source):
    version = result["training_dataset_version"]
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    split = db.get(models.SplitMap, version["split_map_id"])
    return {
        origin["stable_key"]: split.assignments[key]
        for key, provenance in generated.provenance["items"].items()
        for origin in provenance["origins"]
        if origin["dataset_version_id"] == source["id"]
    }


def _prepare_previewed(client, project, payload):
    preview = _preview(client, project, payload)
    assert preview["ready"], preview["issues"]
    return _prepare(client, project, {**payload, "preview_manifest_hash": preview["manifest_hash"]})


def _update_first_series(client, project, task, source, first):
    payload = _request(
        task,
        source,
        training_dataset_id=first["training_dataset"]["id"],
        parent_version_id=first["training_dataset_version"]["id"],
        idempotency_key="first-series-update",
    )
    updated = _prepare_previewed(client, project, payload)
    assert updated["training_dataset_version"]["version_number"] == 2
    return updated


def _first_series_with_learning_records(client, db, *, pmids=False):
    project, task = _context(client)
    records = _records("paper", 12)
    if pmids:
        records = [{**row, "pmid": str(100_000 + index)} for index, row in enumerate(records)]
    source = _source(client, project, "Original source", records)
    first = _prepare_previewed(client, project, _request(task, source, seed=13))
    assignments = _source_assignments(db, first, source)
    learning = [row for row in records if assignments[row["key"]] in {"train", "validation"}]
    return project, task, source, first, learning


def test_new_series_without_clean_articles_cannot_reserve_another_series_learning_data(client, db):
    project, task = _context(client)
    records = _records("paper", 12)
    source = _source(client, project, "Original source", records)
    first = _prepare_previewed(client, project, _request(task, source, seed=13))
    original = _source_assignments(db, first, source)
    learning = [row for row in records if original[row["key"]] in {"train", "validation"}]
    assert {original[row["key"]] for row in learning} == {"train", "validation"}
    subset = _source(
        client,
        project,
        "Original source",
        learning,
        dataset_id=source["dataset_id"],
        revision="2",
    )
    payload = _request(task, subset, name="Second series", idempotency_key="second-series", seed=29)
    preview = _preview(client, project, payload)
    assert not preview["ready"]
    assert preview["split_counts"]["test"] == 0
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 422, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 1
    updated = _update_first_series(client, project, task, source, first)
    assert _source_assignments(db, updated, source) == original


@pytest.mark.parametrize("identity", ["source", "content", "pmid", "document"])
def test_new_series_uses_clean_holdouts_and_leaves_earlier_series_updatable(client, db, identity):
    project, task = _context(client)
    records = _records("paper", 12)
    for index, record in enumerate(records):
        if identity == "pmid":
            record["pmid"] = str(100_000 + index)
        elif identity == "document":
            record["document_id"] = index + 1
    source = _source(client, project, "Original source", records)
    first = _prepare_previewed(client, project, _request(task, source, seed=13))
    original = _source_assignments(db, first, source)
    learning = [row for row in records if original[row["key"]] in {"train", "validation"}]
    if identity != "source":
        learning = [
            {
                **row,
                "key": f"copied-{row['key']}",
                **({"text": f"Revised {row['text']}"} if identity != "content" else {}),
            }
            for row in learning
        ]
    new_records = [*learning, *_records("fresh", 3)]
    second_source = _source(
        client,
        project,
        "Second source",
        new_records,
        **({"dataset_id": source["dataset_id"], "revision": "2"} if identity == "source" else {}),
    )
    second = _prepare_previewed(
        client,
        project,
        _request(
            task,
            second_source,
            name="Second series",
            idempotency_key="second-series",
            seed=29,
        ),
    )
    second_assignments = _source_assignments(db, second, second_source)
    assert all(second_assignments[row["key"]] != "test" for row in learning)
    assert any(second_assignments[row["key"]] == "test" for row in _records("fresh", 3))
    updated = _update_first_series(client, project, task, source, first)
    assert _source_assignments(db, updated, source) == original


def test_other_task_can_reserve_articles_used_for_learning_by_first_task(client, db):
    project, task = _context(client)
    records = _records("paper", 12)
    source = _source(client, project, "Original source", records)
    first = _prepare_previewed(client, project, _request(task, source, seed=13))
    original = _source_assignments(db, first, source)
    learning = [row for row in records if original[row["key"]] in {"train", "validation"}]
    subset = _source(
        client,
        project,
        "Original source",
        learning,
        dataset_id=source["dataset_id"],
        revision="2",
    )
    definition = client.post(
        "/api/tasks",
        json={"project_id": project["id"], "key": "other-task", "name": "Other task"},
    )
    assert definition.status_code == 201, definition.text
    response = client.post(
        "/api/tasks/versions",
        json={
            "project_id": project["id"],
            "task_definition_id": definition.json()["id"],
            "task_kind": task["task_kind"],
            "input_schema": task["input_schema"],
            "output_schema": task["output_schema"],
        },
    )
    assert response.status_code == 201, response.text
    second = _prepare_previewed(
        client,
        project,
        _request(
            response.json(),
            subset,
            name="Other task series",
            idempotency_key="other-task-series",
            seed=29,
        ),
    )
    assert "test" in _source_assignments(db, second, subset).values()
    updated = _update_first_series(client, project, task, source, first)
    assert _source_assignments(db, updated, source) == original


def test_new_series_reserves_sole_clean_group_for_test(client, db):
    project, task, source, first, learning = _first_series_with_learning_records(client, db)
    subset = _source(
        client,
        project,
        "Learning and one clean article",
        [*learning[:2], *_records("fresh", 1)],
    )
    second = _prepare_previewed(
        client,
        project,
        _request(task, subset, name="Second series", idempotency_key="second-series", seed=0),
    )
    assignments = _source_assignments(db, second, subset)
    assert assignments["fresh-0"] == "test"
    assert {assignments[row["key"]] for row in learning[:2]} == {"train", "validation"}
    _update_first_series(client, project, task, source, first)


def test_explicit_source_test_assignment_cannot_override_prior_learning(client, db):
    project, task, source, first, learning = _first_series_with_learning_records(client, db)
    records = [*learning[:2], *_records("fresh", 1)]
    subset = _source(client, project, "Conflicting source", records)
    assignments = {
        learning[0]["key"]: "test",
        learning[1]["key"]: "train",
        "fresh-0": "validation",
    }
    # Existing source maps can predate the first prepared series' governance.
    split = models.SplitMap(
        project_id=project["id"],
        dataset_version_id=subset["id"],
        name="Pre-existing explicit test",
        strategy="manual",
        assignments=assignments,
        protected_splits=["test"],
        content_hash=_canonical_hash(assignments),
    )
    db.add(split)
    db.commit()
    payload = _request(task, subset, name="Second series", idempotency_key="second-series", seed=29)
    payload["sources"][0]["split_map_id"] = split.id
    preview = _preview(client, project, payload)
    assert not preview["ready"]
    assert "split_conflict" in {issue["code"] for issue in preview["issues"]}
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 422, response.text


def test_changed_labels_and_metadata_do_not_make_reused_mapped_inputs_clean(client, db):
    project, task, source, first, learning = _first_series_with_learning_records(client, db)
    copies = [
        {**row, "key": f"copy-{row['key']}", "label": "no", "metadata": {"revision": 2}}
        for row in learning[:2]
    ]
    subset = _source(client, project, "Relabeled copies", [*copies, *_records("fresh", 1)])
    second = _prepare_previewed(
        client,
        project,
        _request(task, subset, name="Second series", idempotency_key="second-series", seed=0),
    )
    assignments = _source_assignments(db, second, subset)
    assert assignments["fresh-0"] == "test"
    assert all(assignments[row["key"]] != "test" for row in copies)
    _update_first_series(client, project, task, source, first)


def test_unlabeled_current_group_peer_preserves_prior_learning_boundary(client, db):
    project, task, source, first, learning = _first_series_with_learning_records(
        client, db, pmids=True
    )
    bridge = {**learning[0], "key": "unlabeled-bridge", "group": "shared-article"}
    bridge.pop("label")
    subset = _source(
        client,
        project,
        "Unlabeled article bridge",
        [
            bridge,
            {"key": "peer", "group": "shared-article", "text": "new passage", "label": "yes"},
            learning[1],
            *_records("fresh", 1),
        ],
    )
    second = _prepare_previewed(
        client,
        project,
        _request(task, subset, name="Second series", idempotency_key="second-series", seed=0),
    )
    assignments = _source_assignments(db, second, subset)
    assert "unlabeled-bridge" not in assignments
    assert assignments["peer"] != "test"
    assert assignments[learning[1]["key"]] != "test"
    assert assignments["fresh-0"] == "test"
    _update_first_series(client, project, task, source, first)


def test_reordered_positional_csv_preserves_article_boundaries_across_series(client, db):
    project, task = _context(client)
    response = client.post(
        "/api/datasets",
        json={"project_id": project["id"], "name": "Positional CSV", "source_type": "upload"},
    )
    assert response.status_code == 201, response.text
    dataset_id = response.json()["id"]
    path = f"/api/projects/{project['id']}/datasets/{dataset_id}/versions/upload"
    original_rows = "".join(f"Paper {index},yes\n" for index in range(8))
    response = client.post(
        path,
        files={"file": ("papers.csv", f"text,label\n{original_rows}".encode(), "text/csv")},
    )
    assert response.status_code == 201, response.text
    source = response.json()
    first = _prepare_previewed(client, project, _request(task, source, seed=13))
    first_assignments = _source_assignments(db, first, source)
    first_by_text = {
        item.payload["text"]: first_assignments[item.stable_key]
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=source["id"])
    }
    assert "test" in first_by_text.values()

    response = client.post(
        path,
        files={
            "file": (
                "papers.csv",
                f"text,label\nNew paper,yes\n{original_rows}".encode(),
                "text/csv",
            )
        },
    )
    assert response.status_code == 201, response.text
    reordered = response.json()
    second = _prepare_previewed(
        client,
        project,
        _request(
            task,
            reordered,
            name="Second series",
            idempotency_key="second-series",
            seed=29,
        ),
    )
    second_assignments = _source_assignments(db, second, reordered)
    second_by_text = {
        item.payload["text"]: second_assignments[item.stable_key]
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=reordered["id"])
    }
    assert all(
        (second_by_text[text] == "test") == (split == "test")
        for text, split in first_by_text.items()
    )
