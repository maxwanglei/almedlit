"""Dataset ingestion and selection coverage for the canonical workflow."""

import hashlib
import json

import pytest


def _create_project(client, name: str) -> dict:
    response = client.post("/api/projects", json={"name": name})
    assert response.status_code == 200, response.text
    project = response.json()
    capabilities = client.patch(
        f"/api/workspaces/{project['workspace_id']}/capability",
        json={"preset": "full", "overrides": []},
    )
    assert capabilities.status_code == 200, capabilities.text
    return project


def _create_dataset(client, project_id: int, name: str, source_type: str) -> dict:
    response = client.post(
        "/api/datasets",
        json={
            "project_id": project_id,
            "name": name,
            "source_type": source_type,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _create_task_version(client, project_id: int) -> dict:
    task_response = client.post(
        "/api/tasks",
        json={
            "project_id": project_id,
            "key": "document-classification",
            "name": "Document classification",
        },
    )
    assert task_response.status_code == 201, task_response.text
    version_response = client.post(
        "/api/tasks/versions",
        json={
            "project_id": project_id,
            "task_definition_id": task_response.json()["id"],
            "task_kind": "classification",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "string"},
        },
    )
    assert version_response.status_code == 201, version_response.text
    return version_response.json()


def _upload_csv(client, project_id: int, dataset_id: int) -> tuple[dict, list[dict]]:
    content = b"id,patient,text\na,p1,alpha\nb,p2,beta\nc,p3,gamma\nd,p4,delta\n"
    response = client.post(
        f"/api/projects/{project_id}/datasets/{dataset_id}/versions/upload",
        data={
            "source_format": "auto",
            "stable_key_field": "id",
            "group_key_field": "patient",
        },
        files={"file": ("records.csv", content, "text/csv")},
    )
    assert response.status_code == 201, response.text
    version = response.json()
    items_response = client.get(
        "/api/datasets/items",
        params={
            "project_id": project_id,
            "dataset_version_id": version["id"],
        },
    )
    assert items_response.status_code == 200, items_response.text
    return version, items_response.json()


def _selection_context(client, name: str) -> dict:
    project = _create_project(client, name)
    task_version = _create_task_version(client, project["id"])
    dataset = _create_dataset(client, project["id"], "Selection pool", "upload")
    dataset_version, items = _upload_csv(client, project["id"], dataset["id"])
    item_by_key = {item["stable_key"]: item for item in items}
    split_response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": dataset_version["id"],
            "name": "fixed holdout",
            "strategy": "explicit",
            "assignments": {
                "a": "pool",
                "b": "pool",
                "c": "pool",
                "d": "test",
            },
            "protected_splits": ["test"],
        },
    )
    assert split_response.status_code == 201, split_response.text
    return {
        "project": project,
        "task_version": task_version,
        "dataset_version": dataset_version,
        "item_by_key": item_by_key,
        "split_map": split_response.json(),
    }


def _create_feedback_set(client, context: dict, candidates: list[dict]) -> dict:
    run_response = client.post(
        "/api/feedback-runs",
        json={
            "project_id": context["project"]["id"],
            "dataset_version_id": context["dataset_version"]["id"],
            "task_version_id": context["task_version"]["id"],
            "producer_type": "rule",
        },
    )
    assert run_response.status_code == 201, run_response.text
    set_response = client.post(
        "/api/feedback-runs/sets",
        json={
            "project_id": context["project"]["id"],
            "feedback_run_id": run_response.json()["id"],
            "candidates": candidates,
        },
    )
    assert set_response.status_code == 201, set_response.text
    return set_response.json()


def _materialize_strategy(
    client,
    context: dict,
    strategy: str,
    *,
    parameters: dict,
    feedback_set_id: int | None = None,
    seed: int = 42,
):
    payload = {
        "project_id": context["project"]["id"],
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "split_map_id": context["split_map"]["id"],
        "strategy": strategy,
        "parameters": parameters,
        "seed": seed,
    }
    if feedback_set_id is not None:
        payload["feedback_set_version_id"] = feedback_set_id
    run_response = client.post("/api/selection-runs", json=payload)
    assert run_response.status_code == 201, run_response.text
    return client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/"
        f"{run_response.json()['id']}/materialize"
    )


def test_project_corpus_snapshot_materializes_documents_idempotently(client):
    project = _create_project(client, "workflow-project-corpus")
    first_document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "external_id": "PMID-100",
            "title": "First article",
            "text": "First source document.",
            "source": "pubmed",
            "metadata_": {"group_id": "trial-1", "journal": "Example Medicine"},
        },
    )
    assert first_document.status_code == 200, first_document.text
    second_document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "external_id": "PMID-200",
            "title": "Second article",
            "text": "Second source document.",
            "source": "pubmed",
            "metadata_": {"group_id": "trial-1"},
        },
    )
    assert second_document.status_code == 200, second_document.text
    dataset = _create_dataset(
        client,
        project["id"],
        "Current project corpus",
        "project_corpus",
    )

    snapshot_path = (
        f"/api/projects/{project['id']}/datasets/{dataset['id']}/versions/project-corpus"
    )
    first_snapshot = client.post(snapshot_path)
    assert first_snapshot.status_code == 201, first_snapshot.text
    version_one = first_snapshot.json()
    assert version_one["version_number"] == 1
    assert version_one["item_count"] == 2
    assert version_one["source_uri"] == (f"project://projects/{project['id']}/documents")
    assert version_one["source_format"] == "project_corpus"
    assert len(version_one["source_revision"]) == 64
    assert len(version_one["content_hash"]) == 64
    assert version_one["provenance"] == {
        "ingestion": "project_corpus_snapshot",
        "source_project_id": project["id"],
        "source_document_ids": [
            first_document.json()["id"],
            second_document.json()["id"],
        ],
        "source_document_count": 2,
        "source_revision": version_one["source_revision"],
        "stable_identity": "project-document:{document_id}",
    }
    assert version_one["license_info"] == {
        "status": "inherited_from_project",
        "source_project_id": project["id"],
    }

    items_response = client.get(
        "/api/datasets/items",
        params={
            "project_id": project["id"],
            "dataset_version_id": version_one["id"],
        },
    )
    assert items_response.status_code == 200, items_response.text
    items = items_response.json()
    assert [item["stable_key"] for item in items] == [
        f"project-document:{first_document.json()['id']}",
        f"project-document:{second_document.json()['id']}",
    ]
    assert [item["group_key"] for item in items] == ["trial-1", "trial-1"]
    assert items[0]["payload"] == {
        "document_id": first_document.json()["id"],
        "external_id": "PMID-100",
        "title": "First article",
        "text": "First source document.",
        "source": "pubmed",
        "metadata": {"group_id": "trial-1", "journal": "Example Medicine"},
        "active_structure_version_id": first_document.json()["active_structure_version_id"],
    }
    assert all(len(item["content_hash"]) == 64 for item in items)

    unchanged_snapshot = client.post(snapshot_path)
    assert unchanged_snapshot.status_code == 201, unchanged_snapshot.text
    assert unchanged_snapshot.json()["id"] == version_one["id"]

    third_document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "external_id": "PMID-300",
            "title": "Third article",
            "text": "Third source document.",
            "source": "pubmed",
            "metadata_": {},
        },
    )
    assert third_document.status_code == 200, third_document.text
    changed_snapshot = client.post(snapshot_path)
    assert changed_snapshot.status_code == 201, changed_snapshot.text
    version_two = changed_snapshot.json()
    assert version_two["id"] != version_one["id"]
    assert version_two["version_number"] == 2
    assert version_two["item_count"] == 3
    assert version_two["source_revision"] != version_one["source_revision"]

    old_items = client.get(
        "/api/datasets/items",
        params={
            "project_id": project["id"],
            "dataset_version_id": version_one["id"],
        },
    )
    assert old_items.status_code == 200, old_items.text
    assert len(old_items.json()) == 2


def test_project_corpus_snapshot_rejects_empty_and_mismatched_sources(client):
    project = _create_project(client, "workflow-project-corpus-validation")
    project_dataset = _create_dataset(
        client,
        project["id"],
        "Empty project corpus",
        "project_corpus",
    )
    empty = client.post(
        f"/api/projects/{project['id']}/datasets/{project_dataset['id']}/versions/project-corpus"
    )
    assert empty.status_code == 422
    assert "no documents" in empty.json()["detail"]

    upload_dataset = _create_dataset(
        client,
        project["id"],
        "Not a project corpus",
        "upload",
    )
    wrong_source = client.post(
        f"/api/projects/{project['id']}/datasets/{upload_dataset['id']}/versions/project-corpus"
    )
    assert wrong_source.status_code == 422
    assert "project_corpus dataset" in wrong_source.json()["detail"]


def test_csv_and_jsonl_uploads_create_immutable_dataset_versions(client):
    project = _create_project(client, "workflow-upload-formats")
    csv_dataset = _create_dataset(client, project["id"], "CSV records", "upload")
    csv_content = b"id,patient,text,label\nr1,p1,Alpha,include\nr2,p1,Beta,exclude\n"
    csv_response = client.post(
        f"/api/projects/{project['id']}/datasets/{csv_dataset['id']}/versions/upload",
        data={
            "source_format": "csv",
            "stable_key_field": "id",
            "group_key_field": "patient",
        },
        files={"file": ("../unsafe-name.csv", csv_content, "text/csv")},
    )
    assert csv_response.status_code == 201, csv_response.text
    csv_version = csv_response.json()
    assert csv_version["source_revision"] == hashlib.sha256(csv_content).hexdigest()
    assert csv_version["source_uri"].startswith("upload://sha256/")
    assert csv_version["provenance"]["original_file_name"] == "unsafe-name.csv"
    assert csv_version["item_count"] == 2
    assert csv_version["artifact_package_id"] is not None
    assert csv_version["data_schema"]["required"] == [
        "id",
        "label",
        "patient",
        "text",
    ]

    csv_items = client.get(
        "/api/datasets/items",
        params={
            "project_id": project["id"],
            "dataset_version_id": csv_version["id"],
        },
    ).json()
    assert [(item["stable_key"], item["group_key"]) for item in csv_items] == [
        ("r1", "p1"),
        ("r2", "p1"),
    ]
    assert csv_items[0]["payload"]["text"] == "Alpha"
    task_version = _create_task_version(client, project["id"])
    label_response = client.post(
        "/api/datasets/label-sets/imported-field",
        json={
            "project_id": project["id"],
            "dataset_version_id": csv_version["id"],
            "task_version_id": task_version["id"],
            "name": "Imported source labels",
            "label_field": "label",
        },
    )
    assert label_response.status_code == 201, label_response.text
    assert label_response.json()["artifact_package_id"] is not None

    jsonl_dataset = _create_dataset(client, project["id"], "JSONL records", "upload")
    jsonl_content = b"\n".join(
        [
            json.dumps({"id": "j1", "text": "One", "label": 1}).encode(),
            json.dumps({"id": "j2", "text": "Two", "label": 0}).encode(),
        ]
    )
    jsonl_response = client.post(
        f"/api/projects/{project['id']}/datasets/{jsonl_dataset['id']}/versions/upload",
        data={"source_format": "jsonl", "stable_key_field": "id"},
        files={"file": ("records.jsonl", jsonl_content, "application/x-ndjson")},
    )
    assert jsonl_response.status_code == 201, jsonl_response.text
    jsonl_version = jsonl_response.json()
    assert jsonl_version["item_count"] == 2
    assert jsonl_version["artifact_package_id"] is not None
    jsonl_items = client.get(
        "/api/datasets/items",
        params={
            "project_id": project["id"],
            "dataset_version_id": jsonl_version["id"],
        },
    ).json()
    assert [item["stable_key"] for item in jsonl_items] == ["j1", "j2"]
    assert jsonl_items[0]["payload"]["label"] == 1


def test_upload_validation_is_project_scoped_and_parquet_is_lazy(client):
    first = _create_project(client, "workflow-upload-first")
    second = _create_project(client, "workflow-upload-second")
    dataset = _create_dataset(client, first["id"], "Scoped upload", "upload")

    cross_project = client.post(
        f"/api/projects/{second['id']}/datasets/{dataset['id']}/versions/upload",
        data={"source_format": "csv"},
        files={"file": ("records.csv", b"id,text\n1,one\n", "text/csv")},
    )
    assert cross_project.status_code == 404

    duplicate_key = client.post(
        f"/api/projects/{first['id']}/datasets/{dataset['id']}/versions/upload",
        data={"source_format": "jsonl", "stable_key_field": "id"},
        files={
            "file": (
                "records.jsonl",
                b'{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n',
                "application/x-ndjson",
            )
        },
    )
    assert duplicate_key.status_code == 422
    assert "duplicated" in duplicate_key.json()["detail"]

    parquet = client.post(
        f"/api/projects/{first['id']}/datasets/{dataset['id']}/versions/upload",
        data={"source_format": "parquet"},
        files={"file": ("records.parquet", b"PAR1", "application/vnd.apache.parquet")},
    )
    assert parquet.status_code == 422
    assert "optional 'pyarrow' package" in parquet.json()["detail"]


def test_public_registry_requires_an_immutable_revision_and_never_fetches(client):
    project = _create_project(client, "workflow-public-registry")
    task_version = _create_task_version(client, project["id"])
    dataset = _create_dataset(
        client,
        project["id"],
        "Pinned public dataset",
        "public_registry",
    )
    floating = client.post(
        f"/api/projects/{project['id']}/datasets/{dataset['id']}/versions/public-registry",
        json={
            "provider": "hugging_face",
            "registry_dataset_id": "owner/corpus",
            "exact_revision": "main",
        },
    )
    assert floating.status_code == 422

    revision = "a" * 40
    pinned = client.post(
        f"/api/projects/{project['id']}/datasets/{dataset['id']}/versions/public-registry",
        json={
            "provider": "hugging_face",
            "registry_dataset_id": "owner/corpus",
            "exact_revision": revision,
            "config_name": "default",
            "source_format": "parquet",
            "license_info": {"spdx": "Apache-2.0"},
            "expected_content_sha256": "b" * 64,
        },
    )
    assert pinned.status_code == 201, pinned.text
    version = pinned.json()
    assert version["source_uri"] == "hf://datasets/owner/corpus"
    assert version["source_revision"] == revision
    assert version["item_count"] == 0
    assert version["provenance"] == {
        "ingestion": "public_registry_reference",
        "provider": "hugging_face",
        "registry_dataset_id": "owner/corpus",
        "config_name": "default",
        "exact_revision": revision,
        "expected_content_sha256": "b" * 64,
        "fetch_performed": False,
    }

    snapshot = (
        b'{"id":"one","text":"Alpha","label":"yes"}\n{"id":"two","text":"Beta","label":"no"}\n'
    )
    materialized = client.post(
        (
            f"/api/projects/{project['id']}/datasets/{dataset['id']}"
            "/versions/public-registry-snapshot"
        ),
        data={
            "registry_dataset_id": "owner/corpus",
            "exact_revision": revision,
            "config_name": "default",
            "source_format": "jsonl",
            "stable_key_field": "id",
            "expected_content_sha256": hashlib.sha256(snapshot).hexdigest(),
            "license_identifier": "Apache-2.0",
        },
        files={
            "file": (
                "corpus.jsonl",
                snapshot,
                "application/x-ndjson",
            )
        },
    )
    assert materialized.status_code == 201, materialized.text
    materialized_version = materialized.json()
    assert materialized_version["item_count"] == 2
    assert materialized_version["artifact_package_id"] is not None
    assert materialized_version["source_revision"] == revision
    assert materialized_version["provenance"]["materialized"] is True
    assert (
        materialized_version["provenance"]["snapshot_sha256"]
        == hashlib.sha256(snapshot).hexdigest()
    )
    items = client.get(
        "/api/datasets/items",
        params={
            "project_id": project["id"],
            "dataset_version_id": materialized_version["id"],
        },
    )
    assert items.status_code == 200
    assert [item["stable_key"] for item in items.json()] == ["one", "two"]

    forged_import = client.post(
        "/api/datasets/label-sets",
        json={
            "project_id": project["id"],
            "dataset_version_id": materialized_version["id"],
            "task_version_id": task_version["id"],
            "name": "Posted predictions",
            "source_kind": "imported",
            "labels": {"one": "yes", "two": "yes"},
        },
    )
    assert forged_import.status_code == 422

    imported_labels = client.post(
        "/api/datasets/label-sets/imported-field",
        json={
            "project_id": project["id"],
            "dataset_version_id": materialized_version["id"],
            "task_version_id": task_version["id"],
            "name": "Snapshot labels",
            "label_field": "label",
        },
    )
    assert imported_labels.status_code == 201, imported_labels.text
    assert imported_labels.json()["labels"] == {"one": "yes", "two": "no"}
    assert imported_labels.json()["source_kind"] == "imported"

    checksum_mismatch = client.post(
        (
            f"/api/projects/{project['id']}/datasets/{dataset['id']}"
            "/versions/public-registry-snapshot"
        ),
        data={
            "registry_dataset_id": "owner/corpus",
            "exact_revision": revision,
            "source_format": "jsonl",
            "expected_content_sha256": "0" * 64,
        },
        files={"file": ("corpus.jsonl", snapshot, "application/x-ndjson")},
    )
    assert checksum_mismatch.status_code == 422
    assert "checksum" in checksum_mismatch.json()["detail"]


def test_materialized_selection_is_deterministic_and_excludes_protected_splits(client):
    project = _create_project(client, "workflow-selection-materialization")
    task_version = _create_task_version(client, project["id"])
    dataset = _create_dataset(client, project["id"], "Selection pool", "upload")
    dataset_version, items = _upload_csv(client, project["id"], dataset["id"])
    item_by_key = {item["stable_key"]: item for item in items}

    split_response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": dataset_version["id"],
            "name": "fixed holdout",
            "strategy": "explicit",
            "assignments": {
                "a": "pool",
                "b": "pool",
                "c": "pool",
                "d": "test",
            },
            "protected_splits": ["test"],
        },
    )
    assert split_response.status_code == 201, split_response.text
    split_map = split_response.json()

    missing_governance = client.post(
        "/api/selection-runs",
        json={
            "project_id": project["id"],
            "dataset_version_id": dataset_version["id"],
            "task_version_id": task_version["id"],
            "strategy": "random",
            "parameters": {"limit": 1},
        },
    )
    assert missing_governance.status_code == 422
    assert "governed split map" in missing_governance.json()["detail"]

    feedback_run_response = client.post(
        "/api/feedback-runs",
        json={
            "project_id": project["id"],
            "dataset_version_id": dataset_version["id"],
            "task_version_id": task_version["id"],
            "producer_type": "rule",
        },
    )
    assert feedback_run_response.status_code == 201, feedback_run_response.text
    feedback_set_response = client.post(
        "/api/feedback-runs/sets",
        json={
            "project_id": project["id"],
            "feedback_run_id": feedback_run_response.json()["id"],
            "candidates": [
                {
                    "dataset_item_id": item_by_key[key]["id"],
                    "candidate_key": "primary",
                    "output": "positive",
                    "score": score,
                }
                for key, score in {
                    "a": 0.3,
                    "b": 0.8,
                    "c": 0.6,
                    "d": 0.99,
                }.items()
            ],
        },
    )
    assert feedback_set_response.status_code == 201, feedback_set_response.text

    selection_run_response = client.post(
        "/api/selection-runs",
        json={
            "project_id": project["id"],
            "dataset_version_id": dataset_version["id"],
            "task_version_id": task_version["id"],
            "feedback_set_version_id": feedback_set_response.json()["id"],
            "split_map_id": split_map["id"],
            "strategy": "uncertainty",
            "parameters": {"limit": 2},
            "seed": 17,
        },
    )
    assert selection_run_response.status_code == 201, selection_run_response.text
    selection_run = selection_run_response.json()
    materialized = client.post(
        f"/api/projects/{project['id']}/selection-runs/{selection_run['id']}/materialize"
    )
    assert materialized.status_code == 201, materialized.text
    selection = materialized.json()
    assert [item["dataset_item_id"] for item in selection["items"]] == [
        item_by_key["b"]["id"],
        item_by_key["c"]["id"],
    ]
    assert [item["rank"] for item in selection["items"]] == [1, 2]
    assert all(item["dataset_item_id"] != item_by_key["d"]["id"] for item in selection["items"])
    assert [item["probability"] for item in selection["items"]] == [1.0, 1.0]
    assert selection["items"][0]["reason"] == {
        "strategy": "uncertainty",
        "selection_basis": "deterministic_rank",
        "probability_basis": "deterministic_rank",
        "signal": "uncertainty",
        "signal_value": 0.8,
        "signal_source": "score",
        "feedback_candidate_id": selection["items"][0]["reason"]["feedback_candidate_id"],
        "candidate_key": "primary",
    }

    repeated = client.post(
        f"/api/projects/{project['id']}/selection-runs/{selection_run['id']}/materialize"
    )
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["id"] == selection["id"]

    random_orders = []
    for _ in range(2):
        random_run_response = client.post(
            "/api/selection-runs",
            json={
                "project_id": project["id"],
                "dataset_version_id": dataset_version["id"],
                "task_version_id": task_version["id"],
                "split_map_id": split_map["id"],
                "strategy": "random",
                "parameters": {"limit": 2},
                "seed": 90210,
            },
        )
        assert random_run_response.status_code == 201, random_run_response.text
        random_selection = client.post(
            f"/api/projects/{project['id']}/selection-runs/"
            f"{random_run_response.json()['id']}/materialize"
        )
        assert random_selection.status_code == 201, random_selection.text
        random_orders.append([item["dataset_item_id"] for item in random_selection.json()["items"]])
        assert [item["probability"] for item in random_selection.json()["items"]] == [
            pytest.approx(2 / 3),
            pytest.approx(2 / 3),
        ]
        assert all(
            item["reason"]["probability_basis"] == "uniform_without_replacement"
            and item["reason"]["eligible_count"] == 3
            and item["reason"]["selection_limit"] == 2
            for item in random_selection.json()["items"]
        )
    assert random_orders[0] == random_orders[1]
    assert item_by_key["d"]["id"] not in random_orders[0]

    all_selection = _materialize_strategy(
        client,
        {
            "project": project,
            "task_version": task_version,
            "dataset_version": dataset_version,
            "item_by_key": item_by_key,
            "split_map": split_map,
        },
        "all",
        parameters={"limit": 2},
    )
    assert all_selection.status_code == 201, all_selection.text
    assert [item["dataset_item_id"] for item in all_selection.json()["items"]] == [
        item_by_key["a"]["id"],
        item_by_key["b"]["id"],
    ]
    assert all(
        item["probability"] == 1.0
        and item["reason"]["probability_basis"] == "deterministic_inclusion"
        and item["reason"]["selection_limit"] == 2
        for item in all_selection.json()["items"]
    )


def test_score_strategies_are_auditable_and_protect_holdouts(client):
    context = _selection_context(client, "workflow-score-selection-strategies")
    item_by_key = context["item_by_key"]
    scores = {
        "a": {
            "uncertainty": 0.9,
            "diversity_score": 0.1,
            "disagreement_score": 0.2,
            "error_score": 0.3,
        },
        "b": {
            "uncertainty": 0.6,
            "diversity_score": 0.8,
            "disagreement_score": 0.95,
            "error_score": 0.4,
        },
        "c": {
            "uncertainty": 0.2,
            "diversity_score": 0.7,
            "disagreement_score": 0.5,
            "error_score": 0.9,
        },
        "d": {
            "uncertainty": 1.0,
            "diversity_score": 1.0,
            "disagreement_score": 1.0,
            "error_score": 1.0,
        },
    }
    feedback_set = _create_feedback_set(
        client,
        context,
        [
            {
                "dataset_item_id": item_by_key[key]["id"],
                "candidate_key": "primary",
                "output": "candidate",
                "explanation": values,
            }
            for key, values in scores.items()
        ],
    )

    expected = {
        "diversity": ("diversity_score", ["b", "c"]),
        "disagreement": ("disagreement_score", ["b", "c"]),
        "error_based": ("error_score", ["c", "b"]),
    }
    for strategy, (signal_name, keys) in expected.items():
        response = _materialize_strategy(
            client,
            context,
            strategy,
            parameters={"limit": 2, "candidate_key": "primary"},
            feedback_set_id=feedback_set["id"],
        )
        assert response.status_code == 201, response.text
        selected = response.json()["items"]
        assert [item["dataset_item_id"] for item in selected] == [
            item_by_key[key]["id"] for key in keys
        ]
        assert all(
            item["probability"] == 1.0
            and item["reason"]["strategy"] == strategy
            and item["reason"]["signal"] == signal_name
            and item["reason"]["signal_source"] == f"explanation.{signal_name}"
            and item["reason"]["probability_basis"] == "deterministic_rank"
            and item["reason"]["feedback_candidate_id"] > 0
            for item in selected
        )
        assert item_by_key["d"]["id"] not in {item["dataset_item_id"] for item in selected}

    hybrid = _materialize_strategy(
        client,
        context,
        "hybrid_uncertainty_diversity",
        parameters={
            "limit": 2,
            "uncertainty_weight": 0.75,
            "diversity_weight": 0.25,
        },
        feedback_set_id=feedback_set["id"],
    )
    assert hybrid.status_code == 201, hybrid.text
    hybrid_items = hybrid.json()["items"]
    assert [item["dataset_item_id"] for item in hybrid_items] == [
        item_by_key["a"]["id"],
        item_by_key["b"]["id"],
    ]
    assert hybrid_items[0]["score"] == pytest.approx(0.75)
    assert hybrid_items[0]["probability"] == 1.0
    hybrid_reason = hybrid_items[0]["reason"]
    assert hybrid_reason["probability_basis"] == "deterministic_rank"
    assert hybrid_reason["score_formula"] == "weighted_min_max_normalized_sum"
    assert hybrid_reason["weights"] == {
        "uncertainty": 0.75,
        "diversity": 0.25,
        "normalized_uncertainty": 0.75,
        "normalized_diversity": 0.25,
    }
    assert hybrid_reason["uncertainty"]["signal_source"] == "explanation.uncertainty"
    assert hybrid_reason["diversity"]["signal_source"] == ("explanation.diversity_score")
    assert item_by_key["d"]["id"] not in {item["dataset_item_id"] for item in hybrid_items}


def test_embedding_diversity_uses_deterministic_greedy_max_min(client):
    context = _selection_context(client, "workflow-embedding-diversity")
    item_by_key = context["item_by_key"]
    embeddings = {
        "a": [0.0, 0.0],
        "b": [2.0, 0.0],
        "c": [1.0, 0.0],
        "d": [100.0, 100.0],
    }
    feedback_set = _create_feedback_set(
        client,
        context,
        [
            {
                "dataset_item_id": item_by_key[key]["id"],
                "candidate_key": "primary",
                "output": "candidate",
                "explanation": {"embedding": embedding},
            }
            for key, embedding in embeddings.items()
        ],
    )
    response = _materialize_strategy(
        client,
        context,
        "diversity",
        parameters={"limit": 2},
        feedback_set_id=feedback_set["id"],
    )
    assert response.status_code == 201, response.text
    selected = response.json()["items"]
    assert [item["dataset_item_id"] for item in selected] == [
        item_by_key["a"]["id"],
        item_by_key["b"]["id"],
    ]
    assert [item["score"] for item in selected] == [1.0, 2.0]
    assert selected[0]["reason"]["ranking_metric"] == "distance_to_centroid"
    assert selected[1]["reason"]["ranking_metric"] == "minimum_distance_to_selected"
    assert all(
        item["probability"] == 1.0
        and item["reason"]["algorithm"] == "greedy_max_min"
        and item["reason"]["embedding_dimension"] == 2
        and item["reason"]["signal_source"] == "explanation.embedding"
        for item in selected
    )


def test_selection_signals_fail_closed_and_hash_fallback_requires_opt_in(client):
    context = _selection_context(client, "workflow-selection-fail-closed")
    item_by_key = context["item_by_key"]

    no_signal_diversity = _materialize_strategy(
        client,
        context,
        "diversity",
        parameters={"limit": 2},
    )
    assert no_signal_diversity.status_code == 422
    assert "allow_stable_hash_fallback explicitly" in no_signal_diversity.json()["detail"]

    fallback_orders = []
    for _ in range(2):
        fallback = _materialize_strategy(
            client,
            context,
            "diversity",
            parameters={
                "limit": 2,
                "allow_stable_hash_fallback": True,
            },
            seed=73,
        )
        assert fallback.status_code == 201, fallback.text
        fallback_items = fallback.json()["items"]
        fallback_orders.append([item["dataset_item_id"] for item in fallback_items])
        assert all(
            item["probability"] == 1.0
            and item["reason"]["signal_source"] == "stable_hash_fallback"
            and item["reason"]["fallback_explicitly_enabled"] is True
            and item["reason"]["seed"] == 73
            for item in fallback_items
        )
    assert fallback_orders[0] == fallback_orders[1]
    assert item_by_key["d"]["id"] not in fallback_orders[0]

    generic_scores = _create_feedback_set(
        client,
        context,
        [
            {
                "dataset_item_id": item_by_key[key]["id"],
                "candidate_key": "primary",
                "output": "candidate",
                "score": score,
            }
            for key, score in {"a": 0.2, "b": 0.4, "c": 0.6}.items()
        ],
    )
    disagreement = _materialize_strategy(
        client,
        context,
        "disagreement",
        parameters={"limit": 1},
        feedback_set_id=generic_scores["id"],
    )
    assert disagreement.status_code == 422
    assert "disagreement_score" in disagreement.json()["detail"]

    error_based = _materialize_strategy(
        client,
        context,
        "error_based",
        parameters={"limit": 1},
        feedback_set_id=generic_scores["id"],
    )
    assert error_based.status_code == 422
    assert "error_score" in error_based.json()["detail"]

    hybrid = _materialize_strategy(
        client,
        context,
        "hybrid_uncertainty_diversity",
        parameters={"limit": 1},
        feedback_set_id=generic_scores["id"],
    )
    assert hybrid.status_code == 422
    assert "requires diversity_score" in hybrid.json()["detail"]
