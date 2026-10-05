"""Selection distinguishes finalized work from abandoned or explicit repeat work."""

import pytest
from test_selection_eligibility import _round_for_items, _transition
from test_workflow_ingestion_selection import _materialize_strategy, _selection_context


def _round_decisions(client, context, annotation_round):
    project_id = context["project"]["id"]
    response = client.get(
        f"/api/rounds/{annotation_round['id']}/items",
        params={"project_id": project_id},
    )
    assert response.status_code == 200, response.text
    key_by_item_id = {item["id"]: key for key, item in context["item_by_key"].items()}
    decisions = {}
    for item in response.json():
        decision = client.post(
            "/api/rounds/decisions",
            json={
                "project_id": project_id,
                "round_item_id": item["id"],
                "output": "positive",
            },
        )
        assert decision.status_code == 201, decision.text
        decisions[key_by_item_id[item["dataset_item_id"]]] = decision.json()["id"]
    return decisions


def _submit(client, context, annotation_round, decision_ids):
    response = client.post(
        "/api/rounds/submissions",
        json={
            "project_id": context["project"]["id"],
            "annotation_round_id": annotation_round["id"],
            "decision_ids": decision_ids,
        },
    )
    assert response.status_code == 201, response.text


def _selected_item_ids(client, context):
    response = _materialize_strategy(client, context, "all", parameters={"limit": 3})
    assert response.status_code == 201, response.text
    return {item["dataset_item_id"] for item in response.json()["items"]}


@pytest.mark.parametrize("status", ["draft", "cancelled"])
def test_selection_keeps_unsubmitted_work_available(client, status):
    context = _selection_context(client, f"selection-unsubmitted-{status}")
    annotation_round = _round_for_items(client, context, ["a"])
    if status == "cancelled":
        _transition(client, context, annotation_round, "open")
        _round_decisions(client, context, annotation_round)
        _transition(client, context, annotation_round, "cancelled")

    assert _selected_item_ids(client, context) == {
        context["item_by_key"][key]["id"] for key in ("a", "b", "c")
    }


def test_cancelled_round_excludes_only_items_with_submitted_decisions(client):
    context = _selection_context(client, "selection-partially-submitted-cancelled")
    annotation_round = _round_for_items(client, context, ["a", "b"])
    _transition(client, context, annotation_round, "open")
    decisions = _round_decisions(client, context, annotation_round)
    _submit(client, context, annotation_round, [decisions["a"]])
    _transition(client, context, annotation_round, "cancelled")

    assert _selected_item_ids(client, context) == {
        context["item_by_key"][key]["id"] for key in ("b", "c")
    }


def test_existing_work_does_not_exclude_items_for_a_new_task_version(client):
    context = _selection_context(client, "selection-new-task-version")
    annotation_round = _round_for_items(client, context, ["a"])
    _transition(client, context, annotation_round, "open")
    decisions = _round_decisions(client, context, annotation_round)
    _submit(client, context, annotation_round, [decisions["a"]])
    prior_task = context["task_version"]
    version = client.post(
        "/api/tasks/versions",
        json={
            "project_id": context["project"]["id"],
            "task_definition_id": prior_task["task_definition_id"],
            "task_kind": "classification",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "string", "enum": ["positive", "negative"]},
        },
    )
    assert version.status_code == 201, version.text
    assert version.json()["id"] != prior_task["id"]
    updated_context = {**context, "task_version": version.json()}

    assert _selected_item_ids(client, updated_context) == {
        context["item_by_key"][key]["id"] for key in ("a", "b", "c")
    }


def test_manual_selection_can_create_an_explicit_reannotation_round(client):
    context = _selection_context(client, "selection-explicit-reannotation")
    annotation_round = _round_for_items(client, context, ["a"])
    _transition(client, context, annotation_round, "open")
    decisions = _round_decisions(client, context, annotation_round)
    _submit(client, context, annotation_round, [decisions["a"]])
    _transition(client, context, annotation_round, "closed")
    assert context["item_by_key"]["a"]["id"] not in _selected_item_ids(client, context)

    repeat_round = _round_for_items(client, context, ["a"])
    _transition(client, context, repeat_round, "open")
    response = client.get(
        f"/api/rounds/{repeat_round['id']}/items",
        params={"project_id": context["project"]["id"]},
    )
    assert response.status_code == 200, response.text
    assert [item["dataset_item_id"] for item in response.json()] == [
        context["item_by_key"]["a"]["id"]
    ]
