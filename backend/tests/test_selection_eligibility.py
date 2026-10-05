"""Active-learning rounds must advance through unannotated, unreserved items."""

import pytest
from test_workflow_ingestion_selection import (
    _create_feedback_set,
    _materialize_strategy,
    _selection_context,
)


def _round_for_items(client, context, keys):
    project_id = context["project"]["id"]
    run = client.post("/api/selection-runs", json={
        "project_id": project_id,
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "split_map_id": context["split_map"]["id"],
        "strategy": "all",
    })
    assert run.status_code == 201, run.text
    selection = client.post("/api/selection-runs/sets", json={
        "project_id": project_id,
        "selection_run_id": run.json()["id"],
        "items": [
            {"dataset_item_id": context["item_by_key"][key]["id"], "rank": rank}
            for rank, key in enumerate(keys, start=1)
        ],
    })
    assert selection.status_code == 201, selection.text
    annotation_round = client.post("/api/rounds", json={
        "project_id": project_id,
        "name": "Selected annotation work",
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "selection_set_version_id": selection.json()["id"],
        "assistance_policy": "blind",
        "open_to_all_annotators": True,
    })
    assert annotation_round.status_code == 201, annotation_round.text
    return annotation_round.json()


def _transition(client, context, annotation_round, status):
    response = client.post(
        f"/api/rounds/{annotation_round['id']}/transition",
        params={"project_id": context["project"]["id"]},
        json={"status": status},
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("strategy", ["all", "random", "uncertainty"])
@pytest.mark.parametrize("exclusion", ["open_round", "submitted_decision"])
def test_selection_excludes_existing_task_work(client, strategy, exclusion):
    context = _selection_context(client, f"selection-{strategy}-{exclusion}")
    annotation_round = _round_for_items(client, context, ["a"])
    _transition(client, context, annotation_round, "open")
    if exclusion == "submitted_decision":
        items = client.get(
            f"/api/rounds/{annotation_round['id']}/items",
            params={"project_id": context["project"]["id"]},
        )
        assert items.status_code == 200, items.text
        decision = client.post("/api/rounds/decisions", json={
            "project_id": context["project"]["id"],
            "round_item_id": items.json()[0]["id"],
            "output": "positive",
        })
        assert decision.status_code == 201, decision.text
        submission = client.post("/api/rounds/submissions", json={
            "project_id": context["project"]["id"],
            "annotation_round_id": annotation_round["id"],
            "decision_ids": [decision.json()["id"]],
        })
        assert submission.status_code == 201, submission.text
        _transition(client, context, annotation_round, "closed")

    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": context["item_by_key"][key]["id"],
            "candidate_key": "primary",
            "output": "positive",
            "score": score,
        }
        for key, score in {"a": 0.9, "b": 0.5, "c": 0.1}.items()
    ])
    selection = _materialize_strategy(
        client, context, strategy, parameters={"limit": 3},
        feedback_set_id=feedback_set["id"],
    )
    assert selection.status_code == 201, selection.text
    assert {item["dataset_item_id"] for item in selection.json()["items"]} == {
        context["item_by_key"][key]["id"] for key in ("b", "c")
    }


def test_selection_generates_and_persists_a_seed_per_run(client, monkeypatch):
    context = _selection_context(client, "selection-generated-seeds")
    seeds = iter([101, 202])
    monkeypatch.setattr("secrets.randbits", lambda _bits: next(seeds))
    runs = []
    for expected_seed in (101, 202):
        response = client.post("/api/selection-runs", json={
            "project_id": context["project"]["id"],
            "dataset_version_id": context["dataset_version"]["id"],
            "task_version_id": context["task_version"]["id"],
            "split_map_id": context["split_map"]["id"],
            "strategy": "random",
            "parameters": {"limit": 2},
        })
        assert response.status_code == 201, response.text
        run = response.json()
        assert run["seed"] == expected_seed
        runs.append(run)
        path = (
            f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize"
        )
        selection = client.post(path)
        assert selection.status_code == 201, selection.text
        assert all(item["reason"]["seed"] == expected_seed for item in selection.json()["items"])
        assert client.post(path).json() == selection.json()

    stored = client.get("/api/selection-runs", params={"project_id": context["project"]["id"]})
    assert stored.status_code == 200, stored.text
    assert {run["id"]: run["seed"] for run in stored.json()} == {
        run["id"]: run["seed"] for run in runs
    }


def test_repeated_random_rounds_advance_until_pool_is_exhausted(client):
    context = _selection_context(client, "selection-round-progression")
    key_by_id = {item["id"]: key for key, item in context["item_by_key"].items()}
    selected_ids = set()
    selections = []
    for eligible_count in (3, 2, 1):
        response = _materialize_strategy(client, context, "random", parameters={"limit": 1})
        assert response.status_code == 201, response.text
        selection = response.json()
        selections.append(selection)
        selected = selection["items"][0]
        assert selected["dataset_item_id"] not in selected_ids
        assert selected["probability"] == pytest.approx(1 / eligible_count)
        assert selected["reason"]["eligible_count"] == eligible_count
        selected_ids.add(selected["dataset_item_id"])
        annotation_round = _round_for_items(
            client, context, [key_by_id[selected["dataset_item_id"]]]
        )
        _transition(client, context, annotation_round, "open")

    exhausted = _materialize_strategy(client, context, "random", parameters={"limit": 1})
    assert exhausted.status_code == 422, exhausted.text
    assert "No eligible dataset items remain" in exhausted.json()["detail"]
    assert "open-round items" in exhausted.json()["detail"]
    for selection in selections:
        retry = client.post(
            f"/api/projects/{context['project']['id']}/selection-runs/"
            f"{selection['selection_run_id']}/materialize"
        )
        assert retry.status_code == 201, retry.text
        assert retry.json() == selection
