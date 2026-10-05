"""Opt-in PostgreSQL races for standalone prediction and review identity."""

import os
from uuid import uuid4

import pytest
from test_system_administration_postgres_contention import _run_concurrently
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_run_execution import (
    FakeProbabilityModel,
    MemoryObjectStorage,
    _completed_tfidf_model,
)

from al_medlit.auth.models import User
from al_medlit.workflow import models
from al_medlit.workflow import prediction_schemas as schemas
from al_medlit.workflow.services.feedback_scoring import (
    execute_feedback_scoring_run,
    request_feedback_run_materialization,
)
from al_medlit.workflow.services.predictions import (
    create_prediction_run,
    create_review_round,
    result_rows,
)

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _scope(db, storage):
    scope = _completed_tfidf_model(db, storage, suffix=f"prediction-race-{uuid4().hex[:10]}")
    draft = schemas.PredictionRunCreate(
        name="Concurrent prediction",
        dataset_version_id=scope.dataset.id,
        model_version_id=scope.model_version.id,
        request_key="same-prediction-request",
    )
    return scope.project.id, scope.actor.id, draft


def test_concurrent_prediction_requests_create_one_run(postgres_session_factory, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    with postgres_session_factory() as db:
        project_id, actor_id, draft = _scope(db, MemoryObjectStorage())

    def create(db):
        return create_prediction_run(db, project_id, draft, db.get(User, actor_id)).id

    left, right = _run_concurrently(postgres_session_factory, create, create)
    assert left[0] == right[0] == "ok", (left, right)
    assert left[1] == right[1]
    with postgres_session_factory() as db:
        assert db.query(models.FeedbackRun).filter_by(project_id=project_id).count() == 1
        assert db.query(models.FeedbackSetVersion).filter_by(project_id=project_id).count() == 0


def test_concurrent_review_requests_create_one_complete_original_item_round(
    postgres_session_factory,
    monkeypatch,
):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        project_id, actor_id, draft = _scope(db, storage)
        actor = db.get(User, actor_id)
        run = create_prediction_run(db, project_id, draft, actor)
        request_feedback_run_materialization(
            db,
            project_id=project_id,
            feedback_run_id=run.id,
            actor=actor,
        )
        run = execute_feedback_scoring_run(
            db,
            storage,
            feedback_run_id=run.id,
            model_loader=lambda _: FakeProbabilityModel(),
        )
        assert run.status == "completed", run.failure_reason
        run_id, output_id = run.id, run.output_feedback_set_version_id
        rows = result_rows(db, run)
        expected_ids = {row["dataset_item_id"] for row in rows if not row["protected"]}
        review = schemas.PredictionReviewCreate(
            name="Concurrent review",
            request_key="same-review-request",
            dataset_item_ids=[row["dataset_item_id"] for row in rows],
        )

    def create(db):
        return create_review_round(db, project_id, run_id, review, db.get(User, actor_id))

    left, right = _run_concurrently(postgres_session_factory, create, create)
    assert left[0] == right[0] == "ok", (left, right)
    assert left[1] == right[1]
    assert left[1]["item_count"] == len(expected_ids)
    with postgres_session_factory() as db:
        rounds = db.query(models.AnnotationRound).filter_by(project_id=project_id).all()
        assert len(rounds) == 1
        assert rounds[0].status == "open"
        assert rounds[0].feedback_set_version_id == output_id
        assert rounds[0].annotator_user_ids == [actor_id]
        assert rounds[0].assistance_policy == "immediate_suggestions"
        assert {
            item.dataset_item_id
            for item in db.query(models.RoundItem).filter_by(annotation_round_id=rounds[0].id)
        } == expected_ids
        assert db.query(models.SelectionRun).filter_by(project_id=project_id).count() == 0
        assert db.query(models.FeedbackSetVersion).filter_by(project_id=project_id).count() == 1
