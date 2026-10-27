import pytest
from test_round_adjudication_permissions import _decision, _round, _submit
from test_round_adjudication_permissions import paper_tasks as paper_tasks
from test_round_adjudication_permissions import scope as scope
from test_round_adjudication_permissions import workflow_scope as workflow_scope
from test_training_preparation import _source
from test_workflow import _headers

from al_medlit.workflow import models


@pytest.mark.parametrize("state", ["open", "closed"])
@pytest.mark.parametrize("explicit_submissions", [False, True], ids=["automatic", "explicit"])
def test_training_preparation_combines_consensus_with_finalized_adjudication(
    client, db, scope, state, explicit_submissions,
):
    client.headers.update(_headers(scope.manager))
    annotation_round, items = _round(client, scope)
    ordinary_decisions = []
    submission_ids = []
    for actor, disputed_output in zip(scope.annotators, ("relevant", "irrelevant"), strict=True):
        decision_ids = []
        for index, item in enumerate(items):
            response = _decision(
                client, scope, actor, item,
                kind="correction" if index else "annotation",
                output="relevant" if index else disputed_output,
            )
            assert response.status_code == 201, response.text
            decision_ids.append(response.json()["id"])
        ordinary_decisions.extend(decision_ids)
        response = _submit(client, scope, actor, annotation_round, decision_ids)
        assert response.status_code == 201, response.text
        submission_ids.append(response.json()["id"])
    if state == "closed":
        response = client.post(
            f"/api/rounds/{annotation_round['id']}/transition",
            params={"project_id": scope.project.id}, json={"status": state},
        )
        assert response.status_code == 200, response.text
    response = _decision(
        client, scope, scope.manager, items[0], kind="adjudication", output="irrelevant",
    )
    assert response.status_code == 201, response.text
    adjudication_id = response.json()["id"]
    external = _source(client, {"id": scope.project.id}, "External labels", [
        {"key": "external", "text": "Independent training article", "label": "relevant"},
    ])
    round_source = {
        "dataset_version_id": scope.source.id, "annotation_round_id": annotation_round["id"],
    }
    if explicit_submissions:
        round_source["submission_ids"] = list(submission_ids)
    payload = {
        "name": "Adjudicated training", "task_version_id": scope.task.id,
        "sources": [round_source, {"dataset_version_id": external["id"], "label_field": "label"}],
        "idempotency_key": "adjudicated-training",
    }
    path = f"/api/projects/{scope.project.id}/training-datasets"
    unresolved = client.post(f"{path}/preview", json=payload)
    assert unresolved.status_code == 200, unresolved.text
    assert not unresolved.json()["ready"]
    assert any(
        issue["code"] == "invalid_labels" and "disagree" in issue["message"]
        for issue in unresolved.json()["issues"]
    )
    rejected = client.post(f"{path}/prepare", json=payload)
    assert rejected.status_code == 422, rejected.text
    assert "disagree" in rejected.json()["detail"]
    finalized = _submit(client, scope, scope.manager, annotation_round, [adjudication_id])
    assert finalized.status_code == 201, finalized.text
    submission_ids.append(finalized.json()["id"])
    if explicit_submissions:
        excluded_adjudication = client.post(f"{path}/preview", json=payload)
        assert excluded_adjudication.status_code == 200, excluded_adjudication.text
        assert not excluded_adjudication.json()["ready"]
        assert any(
            issue["code"] == "invalid_labels" and "disagree" in issue["message"]
            for issue in excluded_adjudication.json()["issues"]
        )
        round_source["submission_ids"] = list(submission_ids)
    preview = client.post(f"{path}/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert preview.json()["ready"], preview.text
    assert preview.json()["item_count"] == 3
    assert preview.json()["excluded_unlabeled_count"] == 0
    payload["sources"] = preview.json()["resolved_sources"]
    payload["preview_manifest_hash"] = preview.json()["manifest_hash"]
    prepared = client.post(f"{path}/prepare", json=payload)
    assert prepared.status_code == 201, prepared.text
    version = prepared.json()["training_dataset_version"]
    round_ref = next(
        ref for ref in version["preparation_manifest"]["sources"]
        if ref.get("annotation_round_id") == annotation_round["id"]
    )
    expected_decisions = sorted([*ordinary_decisions, adjudication_id])
    assert round_ref["source_kind"] == "adjudicated"
    assert round_ref["submission_ids"] == sorted(submission_ids)
    assert round_ref["source_decision_ids"] == expected_decisions
    label_set = db.get(models.LabelSetVersion, round_ref["label_set_version_id"])
    expected_labels = {
        db.get(models.DatasetItem, item["dataset_item_id"]).stable_key:
        "relevant" if index else "irrelevant"
        for index, item in enumerate(items)
    }
    assert label_set.labels == expected_labels
    assert label_set.source_kind == "adjudicated"
    assert label_set.source_submission_ids == sorted(submission_ids)
    assert label_set.source_decision_ids == expected_decisions
    db.expire_all()
    assert db.get(models.AnnotationRound, annotation_round["id"]).status == state
