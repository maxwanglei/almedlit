"""Opt-in PostgreSQL races for selection and protected holdout governance."""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy.exc import OperationalError
from test_system_administration_postgres_contention import (
    _new_user,
    _set_transaction_timeouts,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_preparation_postgres_contention import _scope as _source_scope

from al_medlit.auth.models import User
from al_medlit.project.models import Project
from al_medlit.workflow import access, models, schemas
from al_medlit.workflow.services import labels, selection
from al_medlit.workspace.models import Workspace, WorkspaceMember

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _selection_scope(db):
    project_id, creator_id, request = _source_scope(db)
    project = db.get(Project, project_id)
    db.get(Workspace, project.workspace_id).capability_preset = "full"
    source_id = request.sources[0].dataset_version_id
    assignments = {
        f"paper-{index}": "test" if index == 5 else "pool" for index in range(6)
    }
    labels.create_split_map(
        db,
        schemas.SplitMapCreate(
            project_id=project_id,
            dataset_version_id=source_id,
            name="Protected source governance",
            strategy="manual",
            assignments=assignments,
        ),
        db.get(User, creator_id),
    )
    actor_ids = []
    run_ids = []
    for index in range(2):
        actor = _new_user(db, f"holdout-selection-trainer-{index}")
        db.add(
            WorkspaceMember(
                workspace_id=project.workspace_id,
                user_id=actor.id,
                role="trainer",
            )
        )
        # Historical runs can omit a split map. Materialization still resolves
        # the project's protected source identities under the policy guard.
        run = models.SelectionRun(
            project_id=project_id,
            dataset_version_id=source_id,
            task_version_id=request.task_version_id,
            strategy="all",
            created_by_user_id=actor.id,
        )
        db.add(run)
        db.flush()
        actor_ids.append(actor.id)
        run_ids.append(run.id)
    db.commit()
    expected_item_ids = {
        item_id
        for item_id, stable_key in db.query(models.DatasetItem.id, models.DatasetItem.stable_key)
        .filter(models.DatasetItem.dataset_version_id == source_id)
        .all()
        if assignments[stable_key] != "test"
    }
    return project_id, source_id, actor_ids, run_ids, assignments, expected_item_ids


def _assert_governance_write_blocked(db, project_id):
    with pytest.raises(OperationalError) as blocked:
        (
            db.query(Project.id)
            .filter(Project.id == project_id)
            .with_for_update(nowait=True)
            .one()
        )
    assert blocked.value.orig.pgcode == "55P03"
    db.rollback()


def test_distinct_trainers_share_holdout_guard_and_exclude_governance_writes_until_commit(
    postgres_session_factory, monkeypatch
):
    with postgres_session_factory() as db:
        (
            project_id,
            source_id,
            actor_ids,
            run_ids,
            assignments,
            expected_item_ids,
        ) = _selection_scope(db)
    assert len(set(actor_ids)) == len(set(run_ids)) == 2

    both_selecting = threading.Barrier(3)
    release = {actor_id: threading.Event() for actor_id in actor_ids}
    original_resolver = selection.protected_dataset_item_ids
    original_policy_lock = selection.lock_holdout_policy

    def observed_policy_lock(db, guarded_project_id):
        original_policy_lock(db, guarded_project_id)
        db.info["holdout_policy_locked"] = True

    def paused_resolver(db, items, task_version_id):
        if db.info.get("holdout_policy_locked") and not db.info.get("holdout_guard_observed"):
            db.info["holdout_guard_observed"] = True
            # The actual service has acquired its project policy guard and its
            # distinct selection-run row lock. Both trainers must reach this
            # point concurrently before either may resolve or publish items.
            both_selecting.wait(timeout=5)
            assert release[db.info["holdout_actor_id"]].wait(timeout=10)
        return original_resolver(db, items, task_version_id)

    monkeypatch.setattr(selection, "lock_holdout_policy", observed_policy_lock)
    monkeypatch.setattr(selection, "protected_dataset_item_ids", paused_resolver)

    def materialize(actor_id, run_id):
        with postgres_session_factory() as db:
            _set_transaction_timeouts(db)
            actor = db.get(User, actor_id)
            access.authorize_project_write(
                db,
                actor,
                project_id,
                min_role="trainer",
                module="learning",
            )
            db.info["holdout_actor_id"] = actor_id
            return selection.materialize_selection_run(
                db,
                project_id=project_id,
                selection_run_id=run_id,
                actor=actor,
            ).id

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(materialize, actor_id, run_id)
            for actor_id, run_id in zip(actor_ids, run_ids, strict=True)
        ]
        try:
            both_selecting.wait(timeout=5)
            with postgres_session_factory() as governance_writer:
                # Split-map creation and prepared snapshot publication require
                # this exclusive project lock. Neither may change policy while
                # either selection is checking protected identities.
                _assert_governance_write_blocked(governance_writer, project_id)
                release[actor_ids[0]].set()
                left_id = futures[0].result(timeout=15)
                _assert_governance_write_blocked(governance_writer, project_id)
                release[actor_ids[1]].set()
                right_id = futures[1].result(timeout=15)
                assert left_id != right_id

                assert (
                    governance_writer.query(Project.id)
                    .filter(Project.id == project_id)
                    .with_for_update(nowait=True)
                    .one()
                    .id
                ) == project_id
                updated = labels.create_split_map(
                    governance_writer,
                    schemas.SplitMapCreate(
                        project_id=project_id,
                        dataset_version_id=source_id,
                        name="Governance after selection commits",
                        strategy="manual",
                        assignments={**assignments, "paper-4": "test"},
                    ),
                    governance_writer.get(User, actor_ids[0]),
                )
                assert updated.assignments["paper-4"] == "test"
        finally:
            for released in release.values():
                released.set()

    with postgres_session_factory() as db:
        selections = (
            db.query(models.SelectionSetVersion)
            .filter(models.SelectionSetVersion.selection_run_id.in_(run_ids))
            .all()
        )
        assert len(selections) == 2
        assert {selection.created_by_user_id for selection in selections} == set(actor_ids)
        for selected in selections:
            assert {item["dataset_item_id"] for item in selected.items} == expected_item_ids
        assert {
            status
            for (status,) in db.query(models.SelectionRun.status)
            .filter(models.SelectionRun.id.in_(run_ids))
            .all()
        } == {"completed"}
