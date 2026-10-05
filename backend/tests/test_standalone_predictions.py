"""Standalone inference authorization, lifecycle, downloads and review handoff."""

import copy
import json

import pytest
from test_training_run_execution import (
    FakeProbabilityModel,
    FakeTrainer,
    MemoryObjectStorage,
    _completed_tfidf_model,
)

from al_medlit.auth.security import create_access_token
from al_medlit.training.run_execution import execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.workflow import models, service
from al_medlit.workflow import prediction_schemas as schemas
from al_medlit.workflow import schemas as workflow_schemas
from al_medlit.workflow.routes import predictions as routes
from al_medlit.workflow.services import predictions
from al_medlit.workflow.services.feedback_scoring import (
    execute_feedback_scoring_run,
    request_feedback_run_materialization,
)
from al_medlit.workspace import capability_service
from al_medlit.workspace.models import WorkspaceMember


@pytest.fixture
def scope(db, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    value = _completed_tfidf_model(db, storage, suffix="standalone")
    value.storage = storage
    value.project.settings = {"modules": ["data", "models", "annotate"]}
    db.commit()
    return value


def draft(scope, **values):
    return schemas.PredictionRunCreate(
        **{
            "name": "Test predictions",
            "dataset_version_id": scope.dataset.id,
            "model_version_id": scope.model_version.id,
            "request_key": "prediction-request-1",
            **values,
        },
    )


def complete(db, scope):
    run = predictions.create_prediction_run(db, scope.project.id, draft(scope), scope.actor)
    request_feedback_run_materialization(
        db, project_id=scope.project.id, feedback_run_id=run.id, actor=scope.actor
    )
    return execute_feedback_scoring_run(
        db, scope.storage, feedback_run_id=run.id, model_loader=lambda _: FakeProbabilityModel()
    )


def headers(scope):
    return {"Authorization": f"Bearer {create_access_token(str(scope.actor.id))}"}


def test_standalone_create_is_idempotent_and_does_not_enable_active_learning(
    client, db, scope, monkeypatch
):
    queued = []
    monkeypatch.setattr(routes, "enqueue_feedback_scoring", queued.append)
    url = f"/api/projects/{scope.project.id}/prediction-runs"
    first = client.post(url, json=draft(scope).model_dump(), headers=headers(scope))
    second = client.post(url, json=draft(scope).model_dump(), headers=headers(scope))
    assert first.status_code == second.status_code == 202, first.text
    assert first.json()["id"] == second.json()["id"]
    assert queued == [first.json()["id"]]
    assert db.query(models.FeedbackRun).count() == 1
    assert client.get(url, headers=headers(scope)).json()[0]["name"] == "Test predictions"
    mismatch = client.post(
        url, json=draft(scope, name="Different").model_dump(), headers=headers(scope)
    )
    assert mismatch.status_code == 409
    learning = client.post(
        "/api/feedback-runs",
        json={
            "project_id": scope.project.id,
            "dataset_version_id": scope.dataset.id,
            "task_version_id": scope.task.id,
            "producer_type": "registered_model",
            "model_version_id": scope.model_version.id,
        },
        headers=headers(scope),
    )
    assert learning.status_code == 403


@pytest.mark.parametrize("restriction", ["inference", "data", "models", "role"])
def test_prediction_access_requires_capability_role_and_both_modules(
    client, db, scope, restriction
):
    if restriction == "inference":
        capability_service.set_capability(
            db,
            scope.project.workspace_id,
            preset="custom",
            overrides=["annotation", "training"],
            actor_user_id=scope.actor.id,
        )
    elif restriction == "role":
        member = (
            db.query(WorkspaceMember)
            .filter_by(workspace_id=scope.project.workspace_id, user_id=scope.actor.id)
            .one()
        )
        member.role = "annotator"
    else:
        scope.project.settings = {
            "modules": ["models"] if restriction == "data" else ["data", "annotate"]
        }
    db.commit()
    response = client.get(
        f"/api/projects/{scope.project.id}/prediction-runs", headers=headers(scope)
    )
    assert response.status_code == 403, response.text


def test_inference_only_workspace_can_predict_without_annotation_or_training(
    client, db, scope, monkeypatch
):
    capability_service.set_capability(
        db,
        scope.project.workspace_id,
        preset="custom",
        overrides=["inference"],
        actor_user_id=scope.actor.id,
    )
    scope.project.settings = {"modules": ["data", "models"]}
    db.commit()
    monkeypatch.setattr(routes, "enqueue_feedback_scoring", lambda _: None)
    response = client.post(
        f"/api/projects/{scope.project.id}/prediction-runs",
        json=draft(scope).model_dump(),
        headers=headers(scope),
    )
    assert response.status_code == 202, response.text
    review = client.post(
        f"/api/projects/{scope.project.id}/prediction-runs/{response.json()['id']}/review-round",
        json={"name": "Review", "request_key": "review-key", "dataset_item_ids": [1]},
        headers=headers(scope),
    )
    assert review.status_code == 403, review.text


def test_results_downloads_and_manual_review_pin_original_items(client, db, scope):
    run = complete(db, scope)
    url = f"/api/projects/{scope.project.id}/prediction-runs/{run.id}"
    page = client.get(url + "/results?offset=0&limit=2", headers=headers(scope))
    assert page.status_code == 200, page.text
    assert page.json()["total"] == 5
    assert len(page.json()["items"]) == 2
    download = client.get(url + "/download?format=jsonl", headers=headers(scope))
    assert download.status_code == 200
    rows = [json.loads(line) for line in download.text.splitlines()]
    assert len(rows) == 5
    assert sum(row["protected"] for row in rows) >= 1
    csv = client.get(url + "/download?format=csv", headers=headers(scope))
    assert "attachment" in csv.headers["content-disposition"]
    assert "prediction,confidence" in csv.text
    selected = [row["dataset_item_id"] for row in rows]
    payload = {
        "name": "Manual review",
        "request_key": "review-request-1",
        "dataset_item_ids": selected,
    }
    created = client.post(url + "/review-round", json=payload, headers=headers(scope))
    assert created.status_code == 201, created.text
    result = created.json()
    assert result["excluded_protected"] >= 1
    repeated = client.post(url + "/review-round", json=payload, headers=headers(scope))
    assert repeated.json() == result
    annotation_round = db.get(models.AnnotationRound, result["round_id"])
    assert annotation_round.status == "open"
    assert annotation_round.assistance_policy == "immediate_suggestions"
    assert annotation_round.annotator_user_ids == [scope.actor.id]
    assert annotation_round.feedback_set_version_id == run.output_feedback_set_version_id
    assert annotation_round.selection_set_version_id is None
    assert db.query(models.SelectionRun).count() == 0
    items = db.query(models.RoundItem).filter_by(annotation_round_id=annotation_round.id).all()
    assert {item.dataset_item_id for item in items} == {
        row["dataset_item_id"] for row in rows if not row["protected"]
    }


def test_prediction_review_retrains_with_previous_snapshot_holdout_labels(client, db, scope):
    scope.project.settings = {"modules": ["data", "models", "annotate", "train"]}
    db.commit()
    parent = scope.training_dataset
    parent_hash = parent.content_hash
    old_assignments = copy.deepcopy(scope.split_map.assignments)
    old_labels = {
        label_set_id: copy.deepcopy(db.get(models.LabelSetVersion, label_set_id).labels)
        for label_set_id in parent.label_set_version_ids
    }
    run = complete(db, scope)
    rows = predictions.result_rows(db, run)
    protected = {row["dataset_item_id"] for row in rows if row["protected"]}
    review = predictions.create_review_round(
        db,
        scope.project.id,
        run.id,
        schemas.PredictionReviewCreate(
            name="Review before retraining",
            request_key="retrain-review",
            dataset_item_ids=[row["dataset_item_id"] for row in rows],
        ),
        scope.actor,
    )
    assert review["excluded_protected"] == len(protected) == 1
    round_items = (
        db.query(models.RoundItem)
        .filter_by(annotation_round_id=review["round_id"])
        .order_by(models.RoundItem.selection_rank)
        .all()
    )
    assert {item.dataset_item_id for item in round_items}.isdisjoint(protected)
    decisions = []
    for item in round_items:
        response = client.post(
            "/api/rounds/decisions",
            json={
                "project_id": scope.project.id,
                "round_item_id": item.id,
                "output": {"label": "negative"},
            },
            headers=headers(scope),
        )
        assert response.status_code == 201, response.text
        decisions.append(response.json()["id"])
    submitted = client.post(
        "/api/rounds/submissions",
        json={
            "project_id": scope.project.id,
            "annotation_round_id": review["round_id"],
            "decision_ids": decisions,
        },
        headers=headers(scope),
    )
    assert submitted.status_code == 201, submitted.text
    payload = {
        "name": parent.name,
        "task_version_id": scope.task.id,
        "training_dataset_id": parent.training_dataset_id,
        "parent_version_id": parent.id,
        "sources": [
            {
                "dataset_version_id": scope.dataset.id,
                "annotation_round_id": review["round_id"],
                "submission_ids": [submitted.json()["id"]],
            }
        ],
        "idempotency_key": "reviewed-training-version",
    }
    url = f"/api/projects/{scope.project.id}/training-datasets"
    preview = client.post(url + "/preview", json=payload, headers=headers(scope))
    assert preview.status_code == 200, preview.text
    assert preview.json()["ready"], preview.json()["issues"]
    prepared = client.post(
        url + "/prepare",
        json={**payload, "preview_manifest_hash": preview.json()["manifest_hash"]},
        headers=headers(scope),
    )
    assert prepared.status_code == 201, prepared.text
    version = prepared.json()["training_dataset_version"]
    assert version["parent_version_id"] == parent.id
    assert version["version_number"] == 2
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    split = db.get(models.SplitMap, version["split_map_id"])
    labels = db.get(models.LabelSetVersion, version["label_set_version_ids"][0]).labels
    for key, provenance in generated.provenance["items"].items():
        origin_keys = {
            origin["stable_key"]
            for origin in provenance["origins"]
            if origin["dataset_version_id"] == scope.dataset.id
        }
        assert len(origin_keys) == 1
        origin_key = next(iter(origin_keys))
        assert split.assignments[key] == old_assignments[origin_key]
        assert labels[key] == {"label": "positive" if origin_key == "test-1" else "negative"}
    assert sum(value == "test" for value in split.assignments.values()) == 1
    retraining = service.create_training_run(
        db,
        workflow_schemas.TrainingRunCreate(
            project_id=scope.project.id,
            registered_model_id=scope.registered.id,
            task_version_id=scope.task.id,
            training_dataset_version_id=version["id"],
            recipe_version_id=scope.recipe_version.id,
            environment_id=scope.environment.id,
            storage_policy_id=scope.storage_policy.id,
            idempotency_key="retrain-reviewed-data",
            evaluation_plan={"splits": ["validation"], "metrics": ["accuracy"]},
            config=scope.run.config,
            seed=17,
            artifact_reservation_bytes=1024 * 1024,
        ),
        scope.actor,
    )
    trainer = FakeTrainer(
        key="sklearn_tfidf",
        recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu",
        output_name="model.skops",
    )
    registry = TrainerPluginRegistry()
    registry.register(trainer)
    completed = execute_training_run(
        db,
        scope.storage,
        training_run_id=retraining.id,
        trainer_registry=registry,
    )
    assert completed.status == "succeeded"
    assert {row["text"] for row in trainer.train_rows} == {"train corrected", "train excluded"}
    assert all(row["label"] == "negative" for row in trainer.train_rows)
    db.expire_all()
    assert db.get(models.TrainingDatasetVersion, parent.id).content_hash == parent_hash
    assert db.get(models.SplitMap, scope.split_map.id).assignments == old_assignments
    assert {
        label_set_id: db.get(models.LabelSetVersion, label_set_id).labels
        for label_set_id in parent.label_set_version_ids
    } == old_labels


def test_pending_results_and_non_prediction_runs_are_not_exposed(client, db, scope):
    run = predictions.create_prediction_run(db, scope.project.id, draft(scope), scope.actor)
    url = f"/api/projects/{scope.project.id}/prediction-runs/{run.id}"
    assert client.get(url + "/results", headers=headers(scope)).status_code == 409
    run.configuration = {}
    db.commit()
    assert client.get(url, headers=headers(scope)).status_code == 404


def test_review_rejects_foreign_ids_without_creating_partial_round(client, db, scope):
    run = complete(db, scope)
    response = client.post(
        f"/api/projects/{scope.project.id}/prediction-runs/{run.id}/review-round",
        json={
            "name": "Bad selection",
            "request_key": "foreign-request",
            "dataset_item_ids": [999999],
        },
        headers=headers(scope),
    )
    assert response.status_code == 422, response.text
    assert db.query(models.AnnotationRound).count() == 0


def test_dispatch_failure_remains_retryable(client, db, scope, monkeypatch):
    def fail(_):
        raise ConnectionError("worker unavailable")

    monkeypatch.setattr(routes, "enqueue_feedback_scoring", fail)
    url = f"/api/projects/{scope.project.id}/prediction-runs"
    response = client.post(url, json=draft(scope).model_dump(), headers=headers(scope))
    assert response.status_code == 202, response.text
    assert response.json()["status"] == "failed"
    queued = []
    monkeypatch.setattr(routes, "enqueue_feedback_scoring", queued.append)
    retry = client.post(f"{url}/{response.json()['id']}/retry", headers=headers(scope))
    assert retry.status_code == 202
    assert retry.json()["status"] == "queued"
    assert queued == [response.json()["id"]]


def test_csv_neutralizes_spreadsheet_formulas():
    assert routes._csv_cell("=HYPERLINK(1)").startswith("'")
    assert routes._csv_cell("  +formula").startswith("'")
    assert routes._csv_cell("ordinary text") == "ordinary text"


def test_review_excludes_only_submitted_decisions_and_can_explicitly_revisit(client, db, scope):
    run = complete(db, scope)
    eligible = [
        row["dataset_item_id"] for row in predictions.result_rows(db, run) if not row["protected"]
    ]
    initial = predictions.create_review_round(
        db,
        scope.project.id,
        run.id,
        schemas.PredictionReviewCreate(
            name="First review",
            request_key="first-review-key",
            dataset_item_ids=eligible[:2],
        ),
        scope.actor,
    )
    items = (
        db.query(models.RoundItem)
        .filter_by(annotation_round_id=initial["round_id"])
        .order_by(models.RoundItem.selection_rank)
        .all()
    )
    decisions = [
        models.RoundAnnotationDecision(
            project_id=scope.project.id,
            round_item_id=item.id,
            annotator_user_id=scope.actor.id,
            output="negative",
            content_hash=str(index) * 64,
        )
        for index, item in enumerate(items)
    ]
    db.add_all(decisions)
    db.flush()
    db.add(
        models.RoundSubmission(
            project_id=scope.project.id,
            annotation_round_id=initial["round_id"],
            annotator_user_id=scope.actor.id,
            sequence=1,
            decision_ids=[decisions[0].id],
            content_hash="s" * 64,
        )
    )
    db.commit()
    rows = predictions.result_rows(db, run)
    assert next(row for row in rows if row["dataset_item_id"] == eligible[0])["already_submitted"]
    assert not next(row for row in rows if row["dataset_item_id"] == eligible[1])[
        "already_submitted"
    ]
    result = predictions.create_review_round(
        db,
        scope.project.id,
        run.id,
        schemas.PredictionReviewCreate(
            name="Unsubmitted review",
            request_key="next-review-key",
            dataset_item_ids=eligible[:2],
        ),
        scope.actor,
    )
    assert result["excluded_submitted"] == 1
    assert result["item_count"] == 1
    revisited = predictions.create_review_round(
        db,
        scope.project.id,
        run.id,
        schemas.PredictionReviewCreate(
            name="Repeat submitted review",
            request_key="repeat-review-key",
            dataset_item_ids=eligible[:1],
            include_submitted=True,
        ),
        scope.actor,
    )
    assert revisited["item_count"] == 1
    assert db.query(models.RoundAnnotationDecision).count() == 2
