"""Opt-in PostgreSQL races for selection ranking and atomic publication."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_holdout_policy_postgres_contention import _selection_scope
from test_system_administration_postgres_contention import _set_transaction_timeouts
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.auth.models import User
from al_medlit.project.models import Project
from al_medlit.workflow import access, models
from al_medlit.workflow.services import selection
from al_medlit.workspace.models import WorkspaceMember

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _materialize(session_factory, project_id, actor_id, run_id):
    with session_factory() as db:
        _set_transaction_timeouts(db)
        actor = db.get(User, actor_id)
        access.authorize_project_read(
            db,
            actor,
            project_id,
            min_role="trainer",
            module="learning",
        )
        return selection.materialize_selection_run(
            db,
            project_id=project_id,
            selection_run_id=run_id,
            actor=actor,
        ).id


def test_ranking_holds_no_project_run_or_actor_membership_locks(
    postgres_session_factory, monkeypatch
):
    with postgres_session_factory() as db:
        project_id, _, actor_ids, run_ids, _, expected_item_ids = _selection_scope(db)
    actor_id, run_id = actor_ids[0], run_ids[0]
    ranking = threading.Event()
    release = threading.Event()
    original_ranker = selection._materialized_selection_items

    def paused_ranker(db, run, **kwargs):
        ranking.set()
        assert release.wait(timeout=10)
        return original_ranker(db, run, **kwargs)

    monkeypatch.setattr(selection, "_materialized_selection_items", paused_ranker)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _materialize, postgres_session_factory, project_id, actor_id, run_id
        )
        try:
            assert ranking.wait(timeout=5)
            with postgres_session_factory() as observer:
                project = (
                    observer.query(Project)
                    .filter(Project.id == project_id)
                    .with_for_update(nowait=True)
                    .one()
                )
                assert project.id == project_id
                assert (
                    observer.query(models.SelectionRun.id)
                    .filter(models.SelectionRun.id == run_id)
                    .with_for_update(nowait=True)
                    .one()
                    .id
                ) == run_id
                assert (
                    observer.query(User.id)
                    .filter(User.id == actor_id)
                    .with_for_update(nowait=True)
                    .one()
                    .id
                ) == actor_id
                assert (
                    observer.query(WorkspaceMember.user_id)
                    .filter(
                        WorkspaceMember.workspace_id == project.workspace_id,
                        WorkspaceMember.user_id == actor_id,
                    )
                    .with_for_update(nowait=True)
                    .one()
                    .user_id
                ) == actor_id
                observer.rollback()
            release.set()
            selection_id = future.result(timeout=15)
        finally:
            release.set()

    with postgres_session_factory() as db:
        selected = db.get(models.SelectionSetVersion, selection_id)
        assert {item["dataset_item_id"] for item in selected.items} == expected_item_ids
        assert db.get(models.SelectionRun, run_id).status == "completed"


def test_simultaneous_materializations_publish_one_selection_set(
    postgres_session_factory, monkeypatch
):
    with postgres_session_factory() as db:
        project_id, _, actor_ids, run_ids, _, expected_item_ids = _selection_scope(db)
    run_id = run_ids[0]
    both_ranking = threading.Barrier(3)
    release = threading.Event()
    original_ranker = selection._materialized_selection_items

    def paused_ranker(db, run, **kwargs):
        both_ranking.wait(timeout=5)
        assert release.wait(timeout=10)
        return original_ranker(db, run, **kwargs)

    monkeypatch.setattr(selection, "_materialized_selection_items", paused_ranker)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                _materialize, postgres_session_factory, project_id, actor_id, run_id
            )
            for actor_id in actor_ids
        ]
        try:
            both_ranking.wait(timeout=5)
            release.set()
            selection_ids = [future.result(timeout=15) for future in futures]
        finally:
            release.set()

    assert selection_ids[0] == selection_ids[1]
    with postgres_session_factory() as db:
        selections = (
            db.query(models.SelectionSetVersion)
            .filter(models.SelectionSetVersion.selection_run_id == run_id)
            .all()
        )
        assert len(selections) == 1
        selected = selections[0]
        assert selected.id == selection_ids[0]
        assert selected.version_number == 1
        assert selected.created_by_user_id in actor_ids
        assert {item["dataset_item_id"] for item in selected.items} == expected_item_ids
        assert db.get(models.SelectionRun, run_id).status == "completed"
