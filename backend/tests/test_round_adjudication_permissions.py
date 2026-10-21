"""Managers adjudicate assigned annotators' work without becoming annotators themselves."""

from types import SimpleNamespace

import pytest
from test_paper_entity_rounds import paper_tasks as paper_tasks
from test_workflow import _headers, _user
from test_workflow import workflow_scope as workflow_scope

from al_medlit.workflow import models
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.models import WorkspaceMember


@pytest.fixture
def scope(db, workflow_scope, paper_tasks):
    second = _user(db, "second-adjudication-annotator")
    workspace_service.add_member(db, workflow_scope.workspace.id, second.id, role="annotator")
    manager_member = db.query(WorkspaceMember).filter_by(
        workspace_id=workflow_scope.workspace.id, user_id=workflow_scope.manager.id,
    ).one()
    manager_member.role = "manager"
    db.commit()
    return SimpleNamespace(
        project=workflow_scope.project, manager=workflow_scope.manager,
        annotators=[workflow_scope.annotator, second], source=paper_tasks.version,
        task=paper_tasks.classification,
    )


def _round(client, scope, *, open_to_all=False, assigned=None):
    response = client.post("/api/rounds", json={
        "project_id": scope.project.id, "name": "Disagreement review",
        "dataset_version_id": scope.source.id, "task_version_id": scope.task.id,
        "assistance_policy": "blind", "reannotation_mode": "full_dataset",
        "open_to_all_annotators": open_to_all,
        "annotator_user_ids": [] if open_to_all else [
            actor.id for actor in (assigned if assigned is not None else scope.annotators)
        ],
    }, headers=_headers(scope.manager))
    assert response.status_code == 201, response.text
    annotation_round = response.json()
    opened = client.post(
        f"/api/rounds/{annotation_round['id']}/transition", params={"project_id": scope.project.id},
        json={"status": "open"}, headers=_headers(scope.manager),
    )
    assert opened.status_code == 200, opened.text
    items = client.get(
        f"/api/rounds/{annotation_round['id']}/items", params={"project_id": scope.project.id},
        headers=_headers(scope.manager),
    )
    assert items.status_code == 200, items.text
    return annotation_round, items.json()


def _decision(client, scope, actor, item, *, kind="annotation", output="relevant"):
    return client.post("/api/rounds/decisions", json={
        "project_id": scope.project.id, "round_item_id": item["id"],
        "output": output, "decision_kind": kind,
    }, headers=_headers(actor))


def _submit(client, scope, actor, annotation_round, decision_ids):
    return client.post("/api/rounds/submissions", json={
        "project_id": scope.project.id, "annotation_round_id": annotation_round["id"],
        "decision_ids": decision_ids,
    }, headers=_headers(actor))


@pytest.mark.parametrize("publication_mode", ["submitted_snapshot", "closed_round"])
def test_unassigned_manager_can_adjudicate_finalize_and_publish_actual_disagreement(
    client, db, scope, publication_mode,
):
    annotation_round, items = _round(client, scope)
    assert scope.manager.id not in annotation_round["annotator_user_ids"]
    submissions = []
    for actor, output in zip(scope.annotators, ("relevant", "irrelevant"), strict=True):
        decision_ids = []
        for index, item in enumerate(items):
            decision = _decision(
                client, scope, actor, item, output=output if index == 0 else "relevant",
            )
            assert decision.status_code == 201, decision.text
            decision_ids.append(decision.json()["id"])
        submission = _submit(client, scope, actor, annotation_round, decision_ids)
        assert submission.status_code == 201, submission.text
        submissions.append(submission.json()["id"])
    if publication_mode == "closed_round":
        closed = client.post(
            f"/api/rounds/{annotation_round['id']}/transition",
            params={"project_id": scope.project.id}, json={"status": "closed"},
            headers=_headers(scope.manager),
        )
        assert closed.status_code == 200, closed.text
    publication = {
        "project_id": scope.project.id, "name": "Resolved submitted disagreement",
        "source_kind": "human", "submission_ids": submissions,
        "publication_mode": publication_mode,
    }
    unresolved = client.post(
        f"/api/rounds/{annotation_round['id']}/label-sets", json=publication,
        headers=_headers(scope.manager),
    )
    assert unresolved.status_code == 409, unresolved.text
    assert "disagree" in unresolved.json()["detail"].lower()
    adjudication = _decision(client, scope, scope.manager, items[0], kind="adjudication")
    assert adjudication.status_code == 201, adjudication.text
    assert adjudication.json()["annotator_user_id"] == scope.manager.id
    finalized = _submit(
        client, scope, scope.manager, annotation_round, [adjudication.json()["id"]],
    )
    assert finalized.status_code == 201, finalized.text
    published = client.post(
        f"/api/rounds/{annotation_round['id']}/label-sets", json={
            **publication, "source_kind": "adjudicated",
            "submission_ids": [*submissions, finalized.json()["id"]],
        }, headers=_headers(scope.manager),
    )
    assert published.status_code == 201, published.text
    assert published.json()["label_count"] == 1
    assert list(published.json()["labels"].values()) == ["relevant"]
    assert published.json()["source_submission_ids"] == [finalized.json()["id"]]
    assert published.json()["source_decision_ids"] == [adjudication.json()["id"]]
    repeated = client.post(
        f"/api/rounds/{annotation_round['id']}/label-sets", json={
            **publication, "source_kind": "adjudicated",
            "submission_ids": [*submissions, finalized.json()["id"]],
        }, headers=_headers(scope.manager),
    )
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["id"] == published.json()["id"]
    revised_decision = _decision(
        client, scope, scope.manager, items[0], kind="adjudication", output="irrelevant",
    )
    assert revised_decision.status_code == 201, revised_decision.text
    revised_submission = _submit(
        client, scope, scope.manager, annotation_round, [revised_decision.json()["id"]],
    )
    assert revised_submission.status_code == 201, revised_submission.text
    revised = client.post(
        f"/api/rounds/{annotation_round['id']}/label-sets", json={
            **publication, "source_kind": "adjudicated",
            "submission_ids": [*submissions, revised_submission.json()["id"]],
        }, headers=_headers(scope.manager),
    )
    assert revised.status_code == 201, revised.text
    assert revised.json()["id"] != published.json()["id"]
    assert list(revised.json()["labels"].values()) == ["irrelevant"]
    assert db.query(models.RoundAnnotationDecision).count() == 6
    db.expire_all()
    assert db.get(models.AnnotationRound, annotation_round["id"]).status == (
        "closed" if publication_mode == "closed_round" else "open"
    )


@pytest.mark.parametrize("kind", ["annotation", "correction"])
@pytest.mark.parametrize("state", ["open", "closed"])
def test_unassigned_manager_still_cannot_create_ordinary_annotator_decisions(
    client, db, scope, kind, state,
):
    annotation_round, items = _round(client, scope)
    if state == "closed":
        db.get(models.AnnotationRound, annotation_round["id"]).status = state
        db.commit()
    response = _decision(client, scope, scope.manager, items[0], kind=kind)
    assert response.status_code == 409, response.text
    if state == "open":
        assert "assigned" in response.json()["detail"].lower()
    assert db.query(models.RoundAnnotationDecision).count() == 0


@pytest.mark.parametrize("open_to_all", [False, True], ids=["assigned", "open-to-all"])
def test_annotators_cannot_create_adjudication_even_with_round_access(
    client, db, scope, open_to_all,
):
    _annotation_round, items = _round(client, scope, open_to_all=open_to_all)
    response = _decision(client, scope, scope.annotators[0], items[0], kind="adjudication")
    assert response.status_code == 403, response.text
    assert "manager" in response.json()["detail"].lower()
    assert db.query(models.RoundAnnotationDecision).count() == 0


@pytest.mark.parametrize("kind", ["annotation", "correction"])
def test_unassigned_manager_cannot_finalize_historical_ordinary_annotator_decisions(
    client, db, scope, kind,
):
    annotation_round, items = _round(client, scope)
    historical = models.RoundAnnotationDecision(
        project_id=scope.project.id, round_item_id=items[0]["id"],
        annotator_user_id=scope.manager.id, decision_kind=kind, output="relevant",
        content_hash=_canonical_hash({"historical": kind}),
    )
    db.add(historical)
    db.commit()
    response = _submit(client, scope, scope.manager, annotation_round, [historical.id])
    assert response.status_code == 409, response.text
    assert "assigned" in response.json()["detail"].lower()
    assert db.query(models.RoundSubmission).count() == 0


@pytest.mark.parametrize("kind", ["annotation", "correction"])
def test_assigned_manager_can_still_annotate_and_finalize_normally(client, scope, kind):
    annotation_round, items = _round(client, scope, assigned=[scope.manager])
    decision = _decision(client, scope, scope.manager, items[0], kind=kind)
    assert decision.status_code == 201, decision.text
    finalized = _submit(client, scope, scope.manager, annotation_round, [decision.json()["id"]])
    assert finalized.status_code == 201, finalized.text


def _stored_decision(db, scope, actor, item, kind):
    decision = models.RoundAnnotationDecision(
        project_id=scope.project.id, round_item_id=item["id"], annotator_user_id=actor.id,
        decision_kind=kind, output="relevant",
        content_hash=_canonical_hash({"historical": kind, "actor": actor.id, "item": item["id"]}),
    )
    db.add(decision)
    db.flush()
    return decision


@pytest.mark.parametrize(
    ("state", "assigned"), [("open", False), ("closed", False), ("closed", True)],
)
def test_mixed_submissions_do_not_gain_adjudication_assignment_or_closed_round_bypass(
    client, db, scope, state, assigned,
):
    annotation_round, items = _round(
        client, scope, assigned=[scope.manager] if assigned else None,
    )
    adjudication = _stored_decision(db, scope, scope.manager, items[0], "adjudication")
    annotation = _stored_decision(db, scope, scope.manager, items[1], "annotation")
    db.get(models.AnnotationRound, annotation_round["id"]).status = state
    db.commit()
    response = _submit(
        client, scope, scope.manager, annotation_round, [adjudication.id, annotation.id],
    )
    assert response.status_code == 409, response.text
    assert db.query(models.RoundSubmission).count() == 0


@pytest.mark.parametrize("state", ["open", "closed"])
def test_adjudication_submission_rechecks_current_manager_role(client, db, scope, state):
    annotation_round, items = _round(client, scope)
    adjudication = _stored_decision(db, scope, scope.manager, items[0], "adjudication")
    db.query(WorkspaceMember).filter_by(
        workspace_id=scope.project.workspace_id, user_id=scope.manager.id,
    ).one().role = "trainer"
    db.get(models.AnnotationRound, annotation_round["id"]).status = state
    db.commit()
    response = _submit(client, scope, scope.manager, annotation_round, [adjudication.id])
    assert response.status_code == 403, response.text
    assert "manager" in response.json()["detail"].lower()
    assert db.query(models.RoundSubmission).count() == 0


@pytest.mark.parametrize("open_to_all", [False, True], ids=["assigned", "open-to-all"])
def test_annotators_cannot_finalize_historical_adjudication_even_with_round_access(
    client, db, scope, open_to_all,
):
    annotation_round, items = _round(client, scope, open_to_all=open_to_all)
    actor = scope.annotators[0]
    adjudication = _stored_decision(db, scope, actor, items[0], "adjudication")
    db.commit()
    response = _submit(client, scope, actor, annotation_round, [adjudication.id])
    assert response.status_code == 403, response.text
    assert "manager" in response.json()["detail"].lower()
    assert db.query(models.RoundSubmission).count() == 0


def test_manager_cannot_finalize_another_managers_adjudication(client, db, scope):
    annotation_round, items = _round(client, scope)
    other = _user(db, "other-adjudicating-manager")
    workspace_service.add_member(db, scope.project.workspace_id, other.id, role="manager")
    adjudication = _stored_decision(db, scope, other, items[0], "adjudication")
    db.commit()
    response = _submit(client, scope, scope.manager, annotation_round, [adjudication.id])
    assert response.status_code == 422, response.text
    assert "actor's decisions" in response.json()["detail"].lower()
    assert db.query(models.RoundSubmission).count() == 0


@pytest.mark.parametrize("state", ["draft", "cancelled"])
def test_manager_adjudication_does_not_bypass_round_lifecycle(client, db, scope, state):
    annotation_round, items = _round(client, scope)
    db.get(models.AnnotationRound, annotation_round["id"]).status = state
    db.commit()
    decision = _decision(client, scope, scope.manager, items[0], kind="adjudication")
    assert decision.status_code == 409, decision.text
    assert db.query(models.RoundAnnotationDecision).count() == 0
    historical = _stored_decision(db, scope, scope.manager, items[0], "adjudication")
    db.commit()
    finalized = _submit(client, scope, scope.manager, annotation_round, [historical.id])
    assert finalized.status_code == 409, finalized.text
    assert db.query(models.RoundSubmission).count() == 0


def test_manager_adjudication_cannot_finalize_a_decision_from_another_round(client, db, scope):
    first, _first_items = _round(client, scope)
    _second, second_items = _round(client, scope)
    adjudication = _stored_decision(db, scope, scope.manager, second_items[0], "adjudication")
    db.commit()
    response = _submit(client, scope, scope.manager, first, [adjudication.id])
    assert response.status_code == 422, response.text
    assert "from this round" in response.json()["detail"].lower()
    assert db.query(models.RoundSubmission).count() == 0


@pytest.mark.parametrize("kind", ["annotation", "correction"])
def test_closed_round_rejects_assigned_managers_ordinary_annotation_and_submission(
    client, db, scope, kind,
):
    annotation_round, items = _round(client, scope, assigned=[scope.manager])
    db.get(models.AnnotationRound, annotation_round["id"]).status = "closed"
    db.commit()
    decision = _decision(client, scope, scope.manager, items[0], kind=kind)
    assert decision.status_code == 409, decision.text
    assert db.query(models.RoundAnnotationDecision).count() == 0
    historical = _stored_decision(db, scope, scope.manager, items[0], kind)
    db.commit()
    finalized = _submit(client, scope, scope.manager, annotation_round, [historical.id])
    assert finalized.status_code == 409, finalized.text
    assert db.query(models.RoundSubmission).count() == 0
