import pytest
from test_round_adjudication_permissions import _decision, _round, _submit
from test_round_adjudication_permissions import paper_tasks as paper_tasks
from test_round_adjudication_permissions import scope as scope
from test_round_adjudication_permissions import workflow_scope as workflow_scope
from test_workflow import _headers

from al_medlit.workspace.models import WorkspaceMember


def _publish(client, scope, annotation_round, submission_ids, *, kind="adjudicated"):
    return client.post(
        f"/api/rounds/{annotation_round['id']}/label-sets", json={
            "project_id": scope.project.id, "name": f"Selected {kind} decisions",
            "source_kind": kind, "submission_ids": submission_ids,
            "publication_mode": "submitted_snapshot",
        }, headers=_headers(scope.manager),
    )


@pytest.mark.parametrize("ordinary_kind", ["annotation", "correction"])
def test_assigned_manager_keeps_ordinary_and_adjudication_provenance(
    client, scope, ordinary_kind,
):
    annotation_round, items = _round(client, scope, assigned=[scope.manager])
    ordinary = _decision(client, scope, scope.manager, items[0], kind=ordinary_kind)
    assert ordinary.status_code == 201, ordinary.text
    ordinary_submission = _submit(
        client, scope, scope.manager, annotation_round, [ordinary.json()["id"]],
    )
    assert ordinary_submission.status_code == 201, ordinary_submission.text
    adjudication = _decision(
        client, scope, scope.manager, items[0], kind="adjudication", output="irrelevant",
    )
    assert adjudication.status_code == 201, adjudication.text
    adjudication_submission = _submit(
        client, scope, scope.manager, annotation_round, [adjudication.json()["id"]],
    )
    assert adjudication_submission.status_code == 201, adjudication_submission.text
    draft = _decision(client, scope, scope.manager, items[0], kind="adjudication")
    assert draft.status_code == 201, draft.text
    submission_ids = [ordinary_submission.json()["id"], adjudication_submission.json()["id"]]
    human = _publish(client, scope, annotation_round, submission_ids, kind="human")
    assert human.status_code == 201, human.text
    assert list(human.json()["labels"].values()) == ["relevant"]
    assert human.json()["source_decision_ids"] == [ordinary.json()["id"]]
    published = _publish(client, scope, annotation_round, list(reversed(submission_ids)))
    assert published.status_code == 201, published.text
    assert list(published.json()["labels"].values()) == ["irrelevant"]
    assert published.json()["source_submission_ids"] == submission_ids
    assert published.json()["source_decision_ids"] == [
        ordinary.json()["id"], adjudication.json()["id"],
    ]


def test_latest_selected_manager_decision_wins_regardless_of_submission_order(
    client, db, scope,
):
    second_manager = scope.annotators[0]
    member = db.query(WorkspaceMember).filter_by(
        workspace_id=scope.project.workspace_id, user_id=second_manager.id,
    ).one()
    member.role = "manager"
    db.commit()
    annotation_round, items = _round(client, scope)
    older = _decision(
        client, scope, scope.manager, items[0], kind="adjudication", output="irrelevant",
    )
    assert older.status_code == 201, older.text
    newer = _decision(client, scope, second_manager, items[0], kind="adjudication")
    assert newer.status_code == 201, newer.text
    newer_submission = _submit(
        client, scope, second_manager, annotation_round, [newer.json()["id"]],
    )
    assert newer_submission.status_code == 201, newer_submission.text
    older_submission = _submit(
        client, scope, scope.manager, annotation_round, [older.json()["id"]],
    )
    assert older_submission.status_code == 201, older_submission.text
    unselected = _decision(
        client, scope, second_manager, items[0], kind="adjudication", output="irrelevant",
    )
    assert unselected.status_code == 201, unselected.text
    unselected_submission = _submit(
        client, scope, second_manager, annotation_round, [unselected.json()["id"]],
    )
    assert unselected_submission.status_code == 201, unselected_submission.text
    submission_ids = [newer_submission.json()["id"], older_submission.json()["id"]]
    published = _publish(client, scope, annotation_round, list(reversed(submission_ids)))
    assert published.status_code == 201, published.text
    assert list(published.json()["labels"].values()) == ["relevant"]
    assert published.json()["source_submission_ids"] == submission_ids
    assert published.json()["source_decision_ids"] == [older.json()["id"], newer.json()["id"]]
    repeated = _publish(client, scope, annotation_round, submission_ids)
    assert repeated.status_code == 201, repeated.text
    assert repeated.json()["id"] == published.json()["id"]
