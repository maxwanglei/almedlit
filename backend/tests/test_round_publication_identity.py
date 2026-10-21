"""Round publication retries retain the requested label-set lineage."""

import pytest
from test_round_adjudication_permissions import (
    _decision,
    _round,
    _submit,
)
from test_round_adjudication_permissions import paper_tasks as paper_tasks
from test_round_adjudication_permissions import scope as scope
from test_round_adjudication_permissions import workflow_scope as workflow_scope
from test_workflow import _headers


@pytest.fixture(params=["submitted_snapshot", "closed_round"])
def published_round(client, scope, request):
    annotation_round, items = _round(client, scope, assigned=[scope.annotators[0]])
    decision_ids = []
    for item in items:
        decision = _decision(client, scope, scope.annotators[0], item)
        assert decision.status_code == 201, decision.text
        decision_ids.append(decision.json()["id"])
    submitted = _submit(client, scope, scope.annotators[0], annotation_round, decision_ids)
    assert submitted.status_code == 201, submitted.text
    if request.param == "closed_round":
        closed = client.post(
            f"/api/rounds/{annotation_round['id']}/transition",
            params={"project_id": scope.project.id},
            json={"status": "closed"},
            headers=_headers(scope.manager),
        )
        assert closed.status_code == 200, closed.text
    publication = {
        "project_id": scope.project.id,
        "name": "Parent validation",
        "source_kind": "human",
        "submission_ids": [submitted.json()["id"]],
        "publication_mode": request.param,
    }
    url = f"/api/rounds/{annotation_round['id']}/label-sets"
    original = client.post(url, json=publication, headers=_headers(scope.manager))
    assert original.status_code == 201, original.text
    return url, publication, original.json()


def test_publication_rejects_nonexistent_parent_after_identical_snapshot(
    client, scope, published_round,
):
    url, publication, _ = published_round
    invalid = client.post(
        url,
        json={**publication, "parent_version_id": 999999},
        headers=_headers(scope.manager),
    )
    assert invalid.status_code == 404, invalid.text


def test_publication_preserves_changed_parent_and_reuses_exact_retry(
    client, scope, published_round,
):
    url, publication, original = published_round
    child_payload = {**publication, "parent_version_id": original["id"]}
    child = client.post(url, json=child_payload, headers=_headers(scope.manager))
    assert child.status_code == 201, child.text
    assert child.json()["id"] != original["id"]
    assert child.json()["parent_version_id"] == original["id"]
    repeated = client.post(url, json=child_payload, headers=_headers(scope.manager))
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["id"] == child.json()["id"]
    assert repeated.json()["parent_version_id"] == original["id"]
