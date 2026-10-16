"""Uncertainty selection must rank confidence in the opposite direction."""

import pytest

from tests.test_workflow_ingestion_selection import (
    _create_feedback_set,
    _materialize_strategy,
    _selection_context,
)


@pytest.mark.parametrize("strategy", ["uncertainty", "hybrid_uncertainty_diversity"])
def test_confidence_fallback_selects_least_confident_items_first(client, strategy):
    context = _selection_context(client, f"confidence-fallback-{strategy}")
    item_by_key = context["item_by_key"]
    confidence_by_key = {"a": 0.95, "b": 0.2, "c": 0.55, "d": 0.0}
    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": item_by_key[key]["id"],
            "candidate_key": "primary",
            "output": "candidate",
            "score": confidence,
            "explanation": {"diversity_score": 0.5},
        }
        for key, confidence in confidence_by_key.items()
    ])
    parameters = {"limit": 2}
    if strategy == "hybrid_uncertainty_diversity":
        parameters.update(uncertainty_weight=1.0, diversity_weight=0.0)

    response = _materialize_strategy(
        client,
        context,
        strategy,
        parameters=parameters,
        feedback_set_id=feedback_set["id"],
    )

    assert response.status_code == 201, response.text
    selected = response.json()["items"]
    assert [item["dataset_item_id"] for item in selected] == [
        item_by_key["b"]["id"],
        item_by_key["c"]["id"],
    ]
    assert [item["rank"] for item in selected] == [1, 2]
    assert [item["probability"] for item in selected] == [1.0, 1.0]
    if strategy == "uncertainty":
        assert [item["score"] for item in selected] == pytest.approx([0.8, 0.45])
    else:
        assert selected[0]["reason"]["uncertainty"]["raw_score"] == pytest.approx(0.8)
        assert selected[1]["reason"]["uncertainty"]["raw_score"] == pytest.approx(0.45)
    for item, confidence in zip(selected, [0.2, 0.55], strict=True):
        reason = item["reason"]
        if strategy == "hybrid_uncertainty_diversity":
            reason = reason["uncertainty"]
        assert reason["signal_source"] == "score"
        assert reason["signal_transform"] == "1 - confidence"
        assert reason["confidence"] == confidence


@pytest.mark.parametrize("strategy", ["uncertainty", "hybrid_uncertainty_diversity"])
def test_explicit_uncertainty_precedes_confidence_and_preserves_its_scale(client, strategy):
    context = _selection_context(client, f"explicit-uncertainty-{strategy}")
    task_version = client.post("/api/tasks/versions", json={
        "project_id": context["project"]["id"],
        "task_definition_id": context["task_version"]["task_definition_id"],
        "task_kind": "classification",
        "input_schema": {"type": "object"},
        "output_schema": {},
    })
    assert task_version.status_code == 201, task_version.text
    context["task_version"] = task_version.json()
    item_by_key = context["item_by_key"]
    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": item_by_key["a"]["id"],
            "candidate_key": "primary",
            "score": 0.01,
            "output": {"uncertainty": 0.99},
            "explanation": {"uncertainty": 0.15, "diversity_score": 0.5},
        },
        {
            "dataset_item_id": item_by_key["b"]["id"],
            "candidate_key": "primary",
            "score": 0.99,
            "output": {"uncertainty": 0.35},
            "explanation": {"diversity_score": 0.5},
        },
        {
            "dataset_item_id": item_by_key["c"]["id"],
            "candidate_key": "primary",
            # Confidence is unused when an explicit uncertainty exists.
            "score": 1.5,
            "output": "candidate",
            "explanation": {"uncertainty": 2.0, "diversity_score": 0.5},
        },
    ])

    response = _materialize_strategy(
        client,
        context,
        strategy,
        parameters={"limit": 3},
        feedback_set_id=feedback_set["id"],
    )

    assert response.status_code == 201, response.text
    selected = response.json()["items"]
    assert [item["dataset_item_id"] for item in selected] == [
        item_by_key[key]["id"] for key in ["c", "b", "a"]
    ]
    for item, expected, source in zip(
        selected,
        [2.0, 0.35, 0.15],
        ["explanation.uncertainty", "output.uncertainty", "explanation.uncertainty"],
        strict=True,
    ):
        reason = item["reason"]
        if strategy == "hybrid_uncertainty_diversity":
            reason = reason["uncertainty"]
            assert reason["raw_score"] == expected
        else:
            assert item["score"] == expected
            assert reason["signal_value"] == expected
        assert reason["signal_source"] == source
        assert "signal_transform" not in reason
        assert "confidence" not in reason


@pytest.mark.parametrize("strategy", ["uncertainty", "hybrid_uncertainty_diversity"])
def test_uncertainty_compares_duplicate_candidates_after_confidence_conversion(client, strategy):
    context = _selection_context(client, f"duplicate-confidence-{strategy}")
    item_by_key = context["item_by_key"]
    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": item_by_key[key]["id"],
            "candidate_key": candidate_key,
            "score": confidence,
            "output": "candidate",
            "explanation": {"diversity_score": 0.5},
        }
        for key, candidate_key, confidence in [
            ("a", "high-confidence", 0.95),
            ("a", "low-confidence", 0.05),
            ("b", "primary", 0.2),
            ("c", "primary", 0.9),
        ]
    ])

    response = _materialize_strategy(
        client,
        context,
        strategy,
        parameters={"limit": 2},
        feedback_set_id=feedback_set["id"],
    )

    assert response.status_code == 201, response.text
    selected = response.json()["items"]
    assert [item["dataset_item_id"] for item in selected] == [
        item_by_key["a"]["id"],
        item_by_key["b"]["id"],
    ]
    reason = selected[0]["reason"]
    if strategy == "hybrid_uncertainty_diversity":
        reason = reason["uncertainty"]
        assert reason["raw_score"] == pytest.approx(0.95)
    else:
        assert selected[0]["score"] == pytest.approx(0.95)
    assert reason["candidate_key"] == "low-confidence"
    assert reason["confidence"] == 0.05


@pytest.mark.parametrize("strategy", ["uncertainty", "hybrid_uncertainty_diversity"])
def test_confidence_fallback_accepts_zero_and_one(client, strategy):
    context = _selection_context(client, f"confidence-boundaries-{strategy}")
    item_by_key = context["item_by_key"]
    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": item_by_key[key]["id"],
            "candidate_key": "primary",
            "score": confidence,
            "output": "candidate",
            "explanation": {"diversity_score": 0.5},
        }
        for key, confidence in [("a", 1.0), ("b", 0.0), ("c", 0.5)]
    ])

    response = _materialize_strategy(
        client,
        context,
        strategy,
        parameters={"limit": 3},
        feedback_set_id=feedback_set["id"],
    )

    assert response.status_code == 201, response.text
    selected = response.json()["items"]
    assert [item["dataset_item_id"] for item in selected] == [
        item_by_key[key]["id"] for key in ["b", "c", "a"]
    ]
    if strategy == "uncertainty":
        assert [item["score"] for item in selected] == [1.0, 0.5, 0.0]
    else:
        assert [item["reason"]["uncertainty"]["raw_score"] for item in selected] == [
            1.0, 0.5, 0.0,
        ]


@pytest.mark.parametrize("strategy", ["uncertainty", "hybrid_uncertainty_diversity"])
@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_uncertainty_rejects_out_of_range_confidence_fallback(client, strategy, confidence):
    context = _selection_context(client, f"invalid-confidence-{strategy}")
    feedback_set = _create_feedback_set(client, context, [{
        "dataset_item_id": context["item_by_key"]["a"]["id"],
        "candidate_key": "primary",
        "score": confidence,
        "output": "candidate",
        "explanation": {"diversity_score": 0.5},
    }])

    response = _materialize_strategy(
        client,
        context,
        strategy,
        parameters={"limit": 1},
        feedback_set_id=feedback_set["id"],
    )

    assert response.status_code == 422
    assert "confidence" in response.json()["detail"].lower()
