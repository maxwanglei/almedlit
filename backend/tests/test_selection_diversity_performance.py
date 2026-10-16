"""Greedy embedding selection must preserve its result without repeated distances."""

import math

import pytest

from al_medlit.workflow import models
from al_medlit.workflow.services import selection


def _selection_inputs(records):
    items = {}
    embeddings = {}
    for item_id, stable_key, vector in records:
        items[item_id] = models.DatasetItem(id=item_id, stable_key=stable_key)
        candidate = models.FeedbackCandidate(
            id=item_id + 1000,
            dataset_item_id=item_id,
            candidate_key=f"candidate-{item_id}",
        )
        source = "output.embedding" if item_id % 2 else "explanation.embedding"
        embeddings[item_id] = (vector, candidate, source)
    return items, embeddings


def _greedy_reference(embeddings, items, limit):
    """Straightforward max-min definition, independent of the optimized implementation."""
    dimension = len(next(iter(embeddings.values()))[0])
    centroid = tuple(
        sum(value[0][axis] for value in embeddings.values()) / len(embeddings)
        for axis in range(dimension)
    )
    remaining = dict(embeddings)
    selected = []
    result = []
    while remaining and len(result) < limit:
        scores = {
            item_id: (
                min(math.dist(vector, previous) for previous in selected)
                if selected
                else math.dist(vector, centroid)
            )
            for item_id, (vector, _candidate, _source) in remaining.items()
        }
        item_id = min(
            remaining,
            key=lambda key: (-scores[key], items[key].stable_key, items[key].id),
        )
        vector, candidate, source = remaining.pop(item_id)
        score = scores[item_id]
        result.append((
            item_id,
            score,
            1.0,
            {
                "strategy": "diversity",
                "selection_basis": "deterministic_rank",
                "probability_basis": "deterministic_rank",
                "signal": "embedding",
                "signal_value": score,
                "signal_source": source,
                "feedback_candidate_id": candidate.id,
                "candidate_key": candidate.candidate_key,
                "algorithm": "greedy_max_min",
                "distance_metric": "euclidean",
                "ranking_metric": (
                    "minimum_distance_to_selected" if selected else "distance_to_centroid"
                ),
                "embedding_dimension": dimension,
            },
        ))
        selected.append(vector)
    return result


@pytest.mark.parametrize("limit", [0, 1, 2, 4, 10])
@pytest.mark.parametrize("records", [
    pytest.param([
        (9, "z", (2.0, -1.0, 3.0)),
        (3, "b", (-4.0, 5.0, 0.0)),
        (6, "c", (0.5, -0.25, 1.5)),
        (2, "a", (1.0, 2.0, -3.0)),
    ], id="distinct-vectors"),
    pytest.param([
        (9, "z", (1.0, 1.0)),
        (7, "a", (-1.0, 1.0)),
        (2, "a", (-1.0, -1.0)),
        (5, "m", (1.0, -1.0)),
    ], id="distance-and-key-ties"),
    pytest.param([
        (9, "z", (2.0, -3.0)),
        (7, "a", (2.0, -3.0)),
        (2, "a", (2.0, -3.0)),
        (5, "m", (2.0, -3.0)),
    ], id="identical-vectors"),
    pytest.param([(4, "only", (1.25,))], id="single-item"),
])
def test_embedding_diversity_matches_greedy_reference(records, limit):
    items, embeddings = _selection_inputs(records)

    ranked = selection._rank_embedding_diversity(embeddings, items, limit)

    assert [
        (item.id, score, probability, reason)
        for item, score, probability, reason in ranked
    ] == _greedy_reference(embeddings, items, limit)


@pytest.mark.parametrize(("count", "limit"), [(40, 10), (80, 10), (80, 30), (80, 80)])
def test_embedding_diversity_distance_work_is_linear_in_pool_and_limit(monkeypatch, count, limit):
    items, embeddings = _selection_inputs([
        (index + 1, f"item-{index:03}", (float(index), float(index * index % 17)))
        for index in range(count)
    ])
    distance_calls = 0
    original_distance = selection._embedding_distance

    def counted_distance(left, right):
        nonlocal distance_calls
        distance_calls += 1
        return original_distance(left, right)

    monkeypatch.setattr(selection, "_embedding_distance", counted_distance)

    ranked = selection._rank_embedding_diversity(embeddings, items, limit)

    assert len(ranked) == limit
    # One centroid pass and at most one comparison per candidate per selected item.
    # Counting work avoids depending on machine speed or a flaky elapsed-time limit.
    assert distance_calls <= count * (limit + 1)
