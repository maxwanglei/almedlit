"""Requested group percentages survive protected source holdouts."""

from collections import Counter

import pytest
from test_training_preparation import _context, _prepare, _preview, _records, _request, _source

from al_medlit.workflow import models
from al_medlit.workflow import preparation_schemas as api
from al_medlit.workflow.services.preparation import _assign_splits, _Example


def _split_request(**extra):
    return api.TrainingPreparationRequest(
        name="Percentage regression",
        task_version_id=1,
        sources=[api.TrainingSource(dataset_version_id=1, label_field="label")],
        **extra,
    )


def _examples(size):
    return [
        _Example(
            stable_key=f"paper-{index}",
            payload={"text": f"Paper {index}"},
            label="yes",
            identity_tokens={f"paper:{index}"},
        )
        for index in range(size)
    ]


def _source_assignments(db, result, source):
    version = result["training_dataset_version"]
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    split = db.get(models.SplitMap, version["split_map_id"])
    return {
        origin["stable_key"]: split.assignments[key]
        for key, details in generated.provenance["items"].items()
        for origin in details["origins"]
        if origin["dataset_version_id"] == source["id"]
    }


@pytest.mark.parametrize(
    ("percentages", "expected"),
    [
        ({}, {"train": 161, "validation": 20, "test": 20}),
        (
            {"train_percent": 60, "validation_percent": 20},
            {"train": 121, "validation": 40, "test": 40},
        ),
    ],
)
def test_one_forced_test_does_not_freeze_new_series_test_percentage(percentages, expected):
    examples = _examples(201)
    examples[0].fixed_splits.add("test")
    examples[0].allowed_splits = {"test"}
    issues = []

    assignments, groups = _assign_splits(examples, _split_request(**percentages), issues)

    assert groups == 201
    assert not issues
    assert assignments["paper-0"] == "test"
    assert Counter(assignments.values()) == expected


@pytest.mark.parametrize(
    ("percentages", "expected"),
    [
        ({}, {"train": 161, "validation": 20, "test": 20, "pool": 0}),
        (
            {"train_percent": 60, "validation_percent": 20},
            {"train": 121, "validation": 40, "test": 40, "pool": 0},
        ),
    ],
)
def test_preview_and_prepare_top_up_single_protected_source_test_item(
    client, db, percentages, expected
):
    project, task = _context(client)
    protected = _source(client, project, "Reserved holdout", _records("protected", 1))
    response = client.post(
        "/api/datasets/split-maps",
        json={
            "project_id": project["id"],
            "dataset_version_id": protected["id"],
            "name": "Reserved source test",
            "strategy": "manual",
            "assignments": {"protected-0": "test"},
        },
    )
    assert response.status_code == 201, response.text
    fresh = _source(client, project, "Fresh papers", _records("fresh", 200))
    payload = _request(task, protected, fresh, **percentages)

    preview = _preview(client, project, payload)

    assert preview["ready"], preview["issues"]
    assert preview["group_count"] == 201
    assert preview["split_counts"] == expected
    payload["preview_manifest_hash"] = preview["manifest_hash"]
    prepared = _prepare(client, project, payload)
    assert prepared["preview"]["split_counts"] == expected
    version = prepared["training_dataset_version"]
    split = db.get(models.SplitMap, version["split_map_id"])
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    reserved_keys = {
        key
        for key, details in generated.provenance["items"].items()
        if any(origin["stable_key"] == "protected-0" for origin in details["origins"])
    }
    assert len(reserved_keys) == 1
    assert all(split.assignments[key] == "test" for key in reserved_keys)
    assert Counter(split.assignments.values()) == {
        split: count for split, count in expected.items() if count
    }


@pytest.mark.parametrize("seed", [0, 42, 71])
def test_requested_test_quota_reserves_scarce_clean_groups_before_learning_splits(seed):
    examples = _examples(101)
    learning = examples[:90]
    for example in learning:
        example.allowed_splits = {"train", "validation", "pool"}
    issues = []

    assignments, groups = _assign_splits(examples, _split_request(seed=seed), issues)

    assert groups == 101
    assert not issues
    assert Counter(assignments.values()) == {"train": 81, "validation": 10, "test": 10}
    assert all(assignments[example.stable_key] != "test" for example in learning)


def test_existing_test_and_sole_clean_group_both_count_toward_requested_test_quota():
    examples = _examples(20)
    examples[0].fixed_splits = {"test"}
    examples[0].allowed_splits = {"test"}
    for example in examples[1:19]:
        example.allowed_splits = {"train", "validation", "pool"}
    issues = []

    assignments, groups = _assign_splits(examples, _split_request(seed=0), issues)

    assert groups == 20
    assert not issues
    assert Counter(assignments.values()) == {"train": 16, "validation": 2, "test": 2}
    assert assignments["paper-0"] == assignments["paper-19"] == "test"
    assert all(assignments[example.stable_key] != "test" for example in examples[1:19])


def test_protected_article_with_many_rows_counts_as_one_group_for_percentages():
    examples = _examples(29)
    for example in examples[:10]:
        example.identity_tokens = {"article:protected"}
        example.fixed_splits = {"test"}
        example.allowed_splits = {"test"}
    issues = []

    assignments, groups = _assign_splits(
        examples, _split_request(train_percent=60, validation_percent=20), issues
    )

    assert groups == 20
    assert not issues
    by_group = {}
    for example in examples:
        assert by_group.setdefault(example.group_key, assignments[example.stable_key]) == (
            assignments[example.stable_key]
        )
    assert Counter(by_group.values()) == {"train": 12, "validation": 4, "test": 4}
    assert Counter(assignments.values()) == {"train": 12, "validation": 4, "test": 13}


@pytest.mark.parametrize("fresh_count", [1, 11])
def test_second_series_honors_feasible_quota_or_explains_eligibility_limit(
    client, db, fresh_count
):
    project, task = _context(client)
    records = _records("original", 100)
    source = _source(client, project, "First source", records)
    first = _prepare(client, project, _request(task, source))
    old_assignments = _source_assignments(db, first, source)
    learning = [row for row in records if old_assignments[row["key"]] != "test"]
    assert len(learning) == 90
    second_source = _source(
        client, project, "Second source", [*learning, *_records("fresh", fresh_count)]
    )
    payload = _request(
        task,
        second_source,
        name="Second series",
        idempotency_key="second-series",
        seed=0,
    )

    preview = _preview(client, project, payload)

    assert preview["ready"], preview["issues"]
    if fresh_count == 11:
        assert preview["split_counts"] == {
            "train": 81,
            "validation": 10,
            "test": 10,
            "pool": 0,
        }
        assert not preview["warnings"]
    else:
        assert preview["split_counts"]["test"] == 1
        warning = next(
            warning for warning in preview["warnings"]
            if warning["code"] == "split_percentage_adjusted"
        )
        assert "Requested group counts: train 73, validation 9, test 9" in warning["message"]
        actual_counts = ", ".join(
            f"{split} {preview['split_counts'][split]}" for split in ("train", "validation", "test")
        )
        assert f"actual group counts: {actual_counts}" in warning["message"]
    second = _prepare(
        client, project, {**payload, "preview_manifest_hash": preview["manifest_hash"]}
    )
    assert second["preview"]["warnings"] == preview["warnings"]
    second_assignments = _source_assignments(db, second, second_source)
    assert all(second_assignments[row["key"]] != "test" for row in learning)
    parent = first["training_dataset_version"]
    updated = _prepare(
        client,
        project,
        _request(
            task,
            source,
            training_dataset_id=first["training_dataset"]["id"],
            parent_version_id=parent["id"],
            idempotency_key="first-series-update",
        ),
    )
    assert _source_assignments(db, updated, source) == old_assignments


def test_parent_update_preserves_test_membership_and_explains_requested_percentage(client, db):
    project, task = _context(client)
    records = _records("parent", 20)
    source = _source(client, project, "Parent source", records)
    first = _prepare(client, project, _request(task, source))
    old = first["training_dataset_version"]
    old_assignments = dict(db.get(models.SplitMap, old["split_map_id"]).assignments)
    new_source = _source(
        client,
        project,
        "Parent source",
        [*records, *_records("new", 200)],
        dataset_id=source["dataset_id"],
        revision="2",
    )
    payload = _request(
        task,
        new_source,
        training_dataset_id=first["training_dataset"]["id"],
        parent_version_id=old["id"],
        train_percent=60,
        validation_percent=20,
        idempotency_key="parent-update",
    )

    preview = _preview(client, project, payload)

    assert preview["ready"], preview["issues"]
    assert preview["split_counts"]["test"] == 2
    assert "split_percentage_adjusted" in {warning["code"] for warning in preview["warnings"]}
    second = _prepare(
        client, project, {**payload, "preview_manifest_hash": preview["manifest_hash"]}
    )
    assignments = db.get(
        models.SplitMap, second["training_dataset_version"]["split_map_id"]
    ).assignments
    assert all(assignments[key] == split for key, split in old_assignments.items())
    assert {key for key, split in assignments.items() if split == "test"} == {
        key for key, split in old_assignments.items() if split == "test"
    }
    assert second["preview"]["warnings"] == preview["warnings"]
    assert db.get(models.SplitMap, old["split_map_id"]).assignments == old_assignments
