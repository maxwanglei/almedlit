"""Round creation retries retain the identity of the original user intent."""

from uuid import uuid4

import pytest
from sqlalchemy import update
from test_paper_entity_rounds import paper_tasks as paper_tasks
from test_workflow import _headers
from test_workflow import workflow_scope as workflow_scope

from al_medlit.workflow import models, schemas, service
from al_medlit.workspace.models import WorkspaceMember


@pytest.fixture
def round_request(workflow_scope, paper_tasks):
    return {
        "project_id": workflow_scope.project.id,
        "request_key": str(uuid4()),
        "name": "Retry this round",
        "dataset_version_id": paper_tasks.version.id,
        "task_version_id": paper_tasks.classification.id,
        "assistance_policy": "blind",
        "reannotation_mode": "full_dataset",
        "annotator_user_ids": [workflow_scope.annotator.id],
        "reason": "A reason entered by the manager",
    }


@pytest.mark.parametrize("status", ["draft", "open", "closed", "cancelled"])
def test_lost_creation_response_replays_one_persisted_round(
    client,
    db,
    workflow_scope,
    round_request,
    status,
):
    headers = _headers(workflow_scope.manager)
    original = client.post("/api/rounds", json=round_request, headers=headers)
    assert original.status_code == 201, original.text
    original_id = original.json()["id"]
    if status in {"open", "closed"}:
        opened = client.post(
            f"/api/rounds/{original_id}/transition",
            params={"project_id": workflow_scope.project.id},
            json={"status": "open"},
            headers=headers,
        )
        assert opened.status_code == 200, opened.text
    if status in {"closed", "cancelled"}:
        if status == "closed":
            decisions = [
                service.create_annotation_decision(
                    db,
                    schemas.AnnotationDecisionCreate(
                        project_id=workflow_scope.project.id,
                        round_item_id=item.id,
                        output="relevant",
                    ),
                    workflow_scope.annotator,
                )
                for item in service.list_round_items(
                    db,
                    workflow_scope.project.id,
                    original_id,
                )
            ]
            service.create_round_submission(
                db,
                schemas.RoundSubmissionCreate(
                    project_id=workflow_scope.project.id,
                    annotation_round_id=original_id,
                    decision_ids=[decision.id for decision in decisions],
                ),
                workflow_scope.annotator,
            )
        terminal = client.post(
            f"/api/rounds/{original_id}/transition",
            params={"project_id": workflow_scope.project.id},
            json={"status": status},
            headers=headers,
        )
        assert terminal.status_code == 200, terminal.text
    db.expire_all()
    retried = client.post("/api/rounds", json=round_request, headers=headers)
    assert retried.status_code == 201, retried.text
    assert retried.json()["id"] == original_id
    assert retried.json()["status"] == status
    assert retried.json()["reason"] == round_request["reason"]
    assert db.query(models.AnnotationRound).count() == 1
    assert db.query(models.RoundItem).count() == 2


@pytest.mark.parametrize(
    "change",
    [{"name": "Changed name"}, {"reason": "Changed reason"}, {"dataset_version_id": 999999}],
)
def test_same_creation_key_rejects_changed_request(
    client,
    db,
    workflow_scope,
    round_request,
    change,
):
    headers = _headers(workflow_scope.manager)
    first = client.post("/api/rounds", json=round_request, headers=headers)
    assert first.status_code == 201, first.text
    conflict = client.post("/api/rounds", json=round_request | change, headers=headers)
    assert conflict.status_code == 409, conflict.text
    assert "different inputs" in conflict.json()["detail"]
    assert db.query(models.AnnotationRound).count() == 1


def test_fresh_form_key_creates_another_matching_round(
    client,
    db,
    workflow_scope,
    round_request,
):
    headers = _headers(workflow_scope.manager)
    original = client.post("/api/rounds", json=round_request, headers=headers)
    fresh = client.post(
        "/api/rounds",
        json=round_request | {"request_key": str(uuid4())},
        headers=headers,
    )
    assert original.status_code == fresh.status_code == 201
    assert original.json()["id"] != fresh.json()["id"]
    assert db.query(models.AnnotationRound).count() == 2


def test_request_key_is_scoped_to_creating_actor(db, workflow_scope, round_request):
    payload = schemas.AnnotationRoundCreate(**round_request)
    first = service.create_annotation_round(db, payload, workflow_scope.manager)
    second = service.create_annotation_round(db, payload, workflow_scope.trainer)
    assert first.id != second.id
    assert second.created_by_user_id == workflow_scope.trainer.id


def test_unassigned_creation_retry_keeps_the_original_covered_items(
    client,
    db,
    workflow_scope,
    round_request,
):
    headers = _headers(workflow_scope.manager)
    payload = round_request | {"reannotation_mode": "unassigned"}
    first = client.post("/api/rounds", json=payload, headers=headers)
    second = client.post("/api/rounds", json=payload, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert db.query(models.RoundItem).count() == 2


def test_creation_request_hash_canonicalizes_assignee_order(
    client,
    db,
    workflow_scope,
    round_request,
):
    headers = _headers(workflow_scope.manager)
    payload = round_request | {
        "annotator_user_ids": [workflow_scope.annotator.id, workflow_scope.manager.id]
    }
    first = client.post("/api/rounds", json=payload, headers=headers)
    second = client.post(
        "/api/rounds",
        json=payload | {"annotator_user_ids": list(reversed(payload["annotator_user_ids"]))},
        headers=headers,
    )
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert db.query(models.AnnotationRound).count() == 1


def test_legacy_requests_without_keys_still_create_independent_rounds(
    client,
    workflow_scope,
    round_request,
):
    del round_request["request_key"]
    headers = _headers(workflow_scope.manager)
    first = client.post("/api/rounds", json=round_request, headers=headers)
    second = client.post("/api/rounds", json=round_request, headers=headers)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] != second.json()["id"]


@pytest.mark.parametrize("assignee_change", ["inactive", "removed_member"])
def test_creation_retry_survives_assignee_offboarding(
    client, db, workflow_scope, round_request, assignee_change
):
    headers = _headers(workflow_scope.manager)
    original = client.post("/api/rounds", json=round_request, headers=headers)
    assert original.status_code == 201, original.text
    if assignee_change == "inactive":
        workflow_scope.annotator.is_active = False
    else:
        db.delete(db.query(WorkspaceMember).filter_by(
            workspace_id=workflow_scope.workspace.id,
            user_id=workflow_scope.annotator.id,
        ).one())
    db.commit()

    replay = client.post("/api/rounds", json=round_request, headers=headers)
    assert replay.status_code == 201, replay.text
    assert replay.json() == original.json()

    conflict = client.post(
        "/api/rounds", json=round_request | {"name": "Changed intent"}, headers=headers
    )
    assert conflict.status_code == 409, conflict.text
    fresh = client.post(
        "/api/rounds", json=round_request | {"request_key": str(uuid4())}, headers=headers
    )
    assert fresh.status_code == 422, fresh.text
    assert db.query(models.AnnotationRound).count() == 1
    assert db.query(models.RoundItem).count() == 2


def test_creation_retry_refreshes_the_original_round(
    db, testing_session_factory, workflow_scope, round_request
):
    payload = schemas.AnnotationRoundCreate(**round_request)
    original = service.create_annotation_round(db, payload, workflow_scope.manager)
    original_id = original.id
    assert original.status == "draft"
    with testing_session_factory() as other_session:
        other_session.execute(
            update(models.AnnotationRound)
            .where(models.AnnotationRound.id == original_id)
            .values(status="cancelled")
        )
        other_session.commit()

    replay = service.create_annotation_round(db, payload, workflow_scope.manager)

    assert replay.id == original_id
    assert replay.status == "cancelled"
    assert db.query(models.AnnotationRound).count() == 1
