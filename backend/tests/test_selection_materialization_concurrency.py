"""Rank without row locks, then validate the snapshot before publishing selection.

SQLite cannot exercise row-lock contention. These tests inspect requested ORM
locks and deterministically inject committed changes at the ranking boundary.
"""

import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_selection_eligibility import _round_for_items, _transition
from test_workflow_ingestion_selection import _create_feedback_set, _selection_context

from al_medlit.auth.models import User
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import selection
from al_medlit.workspace.models import WorkspaceMember


def _create_run(client, context, **overrides):
    response = client.post("/api/selection-runs", json={
        "project_id": context["project"]["id"],
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "split_map_id": context["split_map"]["id"],
        "strategy": "all",
        "parameters": {"limit": 1},
        "seed": 42,
        **overrides,
    })
    assert response.status_code == 201, response.text
    return response.json()


def _materialize(client, context, run):
    return client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize"
    )


def _change_after_ranking(monkeypatch, change):
    original = selection._materialized_selection_items

    def rank_then_change(db, run, **kwargs):
        result = original(db, run, **kwargs)
        change()
        return result

    monkeypatch.setattr(selection, "_materialized_selection_items", rank_then_change)


def _assert_unpublished(session_factory, run_id):
    with session_factory() as db:
        assert db.get(models.SelectionRun, run_id).status == "planned"
        assert db.query(models.SelectionSetVersion).filter_by(selection_run_id=run_id).count() == 0


def test_embedding_ranking_precedes_every_row_lock(client, monkeypatch):
    context = _selection_context(client, "selection-rank-before-locks")
    feedback_set = _create_feedback_set(client, context, [
        {
            "dataset_item_id": context["item_by_key"][key]["id"],
            "candidate_key": "primary",
            "output": "positive",
            "explanation": {"embedding": embedding},
        }
        for key, embedding in {"a": [0.0, 0.0], "b": [1.0, 0.0], "c": [4.0, 0.0]}.items()
    ])
    run = _create_run(
        client, context, strategy="diversity", feedback_set_version_id=feedback_set["id"]
    )
    requested_locks = []
    rankings = []
    eligibility_phases = []
    original = selection._rank_embedding_diversity
    original_eligible = selection._eligible_selection_items

    def record_locks(state):
        if getattr(state.statement, "_for_update_arg", None) is not None:
            requested_locks.append(state.statement)

    def assert_unlocked_ranking(*args, **kwargs):
        assert not requested_locks, "Ranking started while authorization/policy/run locks were held"
        rankings.append(True)
        return original(*args, **kwargs)

    def record_eligibility_phase(*args, **kwargs):
        eligibility_phases.append(bool(requested_locks))
        return original_eligible(*args, **kwargs)

    monkeypatch.setattr(selection, "_rank_embedding_diversity", assert_unlocked_ranking)
    monkeypatch.setattr(selection, "_eligible_selection_items", record_eligibility_phase)
    event.listen(Session, "do_orm_execute", record_locks)
    try:
        response = _materialize(client, context, run)
    finally:
        event.remove(Session, "do_orm_execute", record_locks)

    assert response.status_code == 201, response.text
    assert rankings == [True]
    assert eligibility_phases == [False, True]
    assert requested_locks, "Publication must still serialize authorization and selection writes"


def test_equal_size_eligibility_change_rejects_stale_ranking(
    client, testing_session_factory, monkeypatch
):
    context = _selection_context(client, "selection-eligibility-swap")
    old_reservation = _round_for_items(client, context, ["a"])
    new_reservation = _round_for_items(client, context, ["b"])
    _transition(client, context, old_reservation, "open")
    run = _create_run(client, context)

    def swap_reserved_items():
        with testing_session_factory() as db:
            db.get(models.AnnotationRound, old_reservation["id"]).status = "cancelled"
            db.get(models.AnnotationRound, new_reservation["id"]).status = "open"
            db.commit()

    _change_after_ranking(monkeypatch, swap_reserved_items)
    response = _materialize(client, context, run)

    assert response.status_code == 409, response.text
    _assert_unpublished(testing_session_factory, run["id"])


@pytest.mark.parametrize("field,value", [
    ("parameters", {"limit": 2}),
    ("eligibility_filter", {"include_stable_keys": ["a", "b", "c"]}),
    ("strategy", "random"),
    ("seed", 101),
    ("split_map_id", None),
])
def test_changed_run_input_rejects_stale_ranking(
    client, testing_session_factory, monkeypatch, field, value
):
    context = _selection_context(client, f"selection-input-race-{field}")
    run = _create_run(client, context)

    def change_run():
        with testing_session_factory() as db:
            setattr(db.get(models.SelectionRun, run["id"]), field, value)
            db.commit()

    _change_after_ranking(monkeypatch, change_run)
    response = _materialize(client, context, run)

    assert response.status_code == 409, response.text
    _assert_unpublished(testing_session_factory, run["id"])


@pytest.mark.parametrize("revoked_policy", ["trainer_role", "learning_module"])
def test_publication_rechecks_authorization_after_ranking(
    client, testing_session_factory, monkeypatch, revoked_policy
):
    context = _selection_context(client, f"selection-authorization-race-{revoked_policy}")
    run = _create_run(client, context)

    def revoke_access():
        with testing_session_factory() as db:
            if revoked_policy == "trainer_role":
                member = db.query(WorkspaceMember).filter_by(
                    workspace_id=context["project"]["workspace_id"],
                    user_id=run["created_by_user_id"],
                ).one()
                member.role = "annotator"
            else:
                db.get(Project, context["project"]["id"]).settings = {"modules": ["data"]}
            db.commit()

    _change_after_ranking(monkeypatch, revoke_access)
    response = _materialize(client, context, run)

    assert response.status_code == 403, response.text
    _assert_unpublished(testing_session_factory, run["id"])


def test_competing_completion_returns_existing_selection(
    client, testing_session_factory, monkeypatch
):
    context = _selection_context(client, "selection-competing-completion")
    run = _create_run(client, context)
    winner = {}

    def publish_competing_selection():
        with testing_session_factory() as db:
            existing = selection.create_selection_set(
                db,
                schemas.SelectionSetCreate(
                    project_id=context["project"]["id"],
                    selection_run_id=run["id"],
                    items=[schemas.SelectionItem(
                        dataset_item_id=context["item_by_key"]["c"]["id"], rank=1
                    )],
                ),
                db.get(User, run["created_by_user_id"]),
            )
            winner["id"] = existing.id

    _change_after_ranking(monkeypatch, publish_competing_selection)
    response = _materialize(client, context, run)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == winner["id"]
    assert response.json()["items"][0]["dataset_item_id"] == context["item_by_key"]["c"]["id"]
    with testing_session_factory() as db:
        snapshots = db.query(models.SelectionSetVersion).filter_by(selection_run_id=run["id"])
        assert snapshots.count() == 1


def test_completed_run_returns_existing_selection_after_eligibility_is_exhausted(
    client, monkeypatch
):
    context = _selection_context(client, "selection-completed-exhausted")
    run = _create_run(client, context)
    initial = _materialize(client, context, run)
    assert initial.status_code == 201, initial.text
    annotation_round = _round_for_items(client, context, ["a", "b", "c"])
    _transition(client, context, annotation_round, "open")

    def reject_reranking(*args, **kwargs):
        pytest.fail("A completed run must return its existing snapshot without ranking again")

    monkeypatch.setattr(selection, "_materialized_selection_items", reject_reranking)
    monkeypatch.setattr(selection, "_eligible_selection_items", reject_reranking)
    retry = _materialize(client, context, run)

    assert retry.status_code == 201, retry.text
    assert retry.json() == initial.json()
