"""Concurrent first attempts preserve one durable round request identity."""

import os
import time
from uuid import uuid4

import pytest
from test_system_administration_postgres_contention import (
    _run_concurrently,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as postgres_session_factory,
)
from test_workflow_stale_lock_postgres_contention import (
    _commit_after_concurrent_attempt,
    _scope,
)

from al_medlit.auth.models import User
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import rounds
from al_medlit.workspace import service as workspace_service

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def test_concurrent_round_creation_replays_the_first_committed_request(
    postgres_session_factory, monkeypatch
):
    with postgres_session_factory() as db:
        scope = _scope(db)
    payload = schemas.AnnotationRoundCreate(
        project_id=scope.project_id,
        request_key=str(uuid4()),
        name="Concurrent first request",
        dataset_version_id=scope.dataset_version_id,
        task_version_id=scope.task_version_id,
        annotator_user_ids=[scope.manager_id],
        assistance_policy="blind",
        reannotation_mode="full_dataset",
    )

    def create(db):
        actor = db.get(User, scope.manager_id)
        return rounds.create_annotation_round(db, payload, actor).id

    holder, waiter = _commit_after_concurrent_attempt(
        postgres_session_factory,
        monkeypatch,
        rounds,
        holder=create,
        waiter=create,
    )

    assert holder[0] == waiter[0] == "ok", (holder, waiter)
    assert holder[1] == waiter[1]
    with postgres_session_factory() as db:
        assert db.query(models.AnnotationRound).filter_by(project_id=scope.project_id).count() == 1
        assert db.query(models.RoundItem).filter_by(project_id=scope.project_id).count() == 1


def test_distinct_assisted_round_creators_do_not_upgrade_shared_project_locks(
    postgres_session_factory, monkeypatch,
):
    with postgres_session_factory() as db:
        scope = _scope(db)
        project = db.get(Project, scope.project_id)
        for user_id in scope.annotator_ids:
            workspace_service.add_member(db, project.workspace_id, user_id, role="manager")
        feedback = models.FeedbackRun(
            project_id=scope.project_id,
            dataset_version_id=scope.dataset_version_id,
            task_version_id=scope.task_version_id,
            producer_type="manual",
            status="completed",
        )
        db.add(feedback)
        db.flush()
        output = models.FeedbackSetVersion(
            project_id=scope.project_id,
            feedback_run_id=feedback.id,
            dataset_version_id=scope.dataset_version_id,
            task_version_id=scope.task_version_id,
            content_hash=uuid4().hex,
        )
        db.add(output)
        db.commit()
        output_id = output.id

    original_guard = rounds.lock_holdout_policy

    def delayed_guard(*args, **kwargs):
        result = original_guard(*args, **kwargs)
        time.sleep(0.1)
        return result

    monkeypatch.setattr(rounds, "lock_holdout_policy", delayed_guard)

    def create(db, actor_id):
        return rounds.create_annotation_round(
            db,
            schemas.AnnotationRoundCreate(
                project_id=scope.project_id,
                name=f"Assisted round {actor_id}",
                dataset_version_id=scope.dataset_version_id,
                task_version_id=scope.task_version_id,
                annotator_user_ids=[actor_id],
                feedback_set_version_id=output_id,
                assistance_policy="reveal_after_first_pass",
                reannotation_mode="full_dataset",
            ),
            db.get(User, actor_id),
        ).id

    first, second = _run_concurrently(
        postgres_session_factory,
        lambda db: create(db, scope.annotator_ids[0]),
        lambda db: create(db, scope.annotator_ids[1]),
    )
    assert first[0] == second[0] == "ok", (first, second)
    assert first[1] != second[1]
    with postgres_session_factory() as db:
        assert {row.sequence for row in db.query(models.AnnotationRound).filter_by(
            project_id=scope.project_id,
        )} == {1, 2}
