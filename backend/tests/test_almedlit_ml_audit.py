from types import SimpleNamespace

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError
from test_artifact_quotas import _headers, _package_files, _package_spec, _workspace_projects
from test_evidence_windowing import _sentences, _target
from test_inference_candidate_queries import _decoder_result, _inference_scope
from test_training_run_execution import (
    FakeEvaluator,
    FakeTrainer,
    MemoryObjectStorage,
    _training_run_fixture,
)

from al_medlit.inference import service as inference_service
from al_medlit.inference.models import InferenceWindow
from al_medlit.inference.scores import sentence_score_probabilities
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.models import ArtifactStorageReservation, WorkspaceArtifactQuota
from al_medlit.project.models import Project
from al_medlit.storage_reclaim.models import OrphanedStorageObject
from al_medlit.storage_reclaim.service import record_orphaned_object
from al_medlit.training.evaluation.evidence import blocks_from_bio
from al_medlit.training.evaluators.contracts import EvaluatorPluginRegistry
from al_medlit.training.registry import model_types
from al_medlit.training.run_execution import _declared_label_vocabulary, execute_training_run
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.training.windowing import (
    EvidenceBlockWindowBuilder,
    GoldBlockRange,
    ReviewedInterval,
    WindowBuilderConfig,
)
from al_medlit.workspace.models import Workspace, WorkspaceMember


@pytest.mark.parametrize("operation", ["upload", "delete", "list", "read"])
def test_minio_transport_failures_use_storage_error_contract(operation):
    from urllib3.exceptions import HTTPError

    from al_medlit.core.storage import MinioObjectStorage, ObjectStorageError

    def unavailable(*_a, **_k):
        raise HTTPError("temporary object-store outage")

    storage = MinioObjectStorage.__new__(MinioObjectStorage)
    storage.bucket = "test"
    storage.encryption_mode = "none"
    storage.client = SimpleNamespace(
        put_object=unavailable, remove_object=unavailable, list_objects=unavailable,
        get_object=lambda *_: SimpleNamespace(
            read=unavailable, close=lambda: None, release_conn=lambda: None,
        ),
    )
    operations = {
        "upload": lambda: storage.put_bytes("key", b"data"),
        "delete": lambda: storage.delete("key"),
        "list": lambda: list(storage.list_objects("prefix")),
        "read": lambda: storage.get_bytes("key"),
    }
    with pytest.raises(ObjectStorageError, match="temporary object-store outage"):
        operations[operation]()


def test_owner_reservation_renewal_rechecks_status_after_lock(db, monkeypatch):
    actor, _, _, _, project, _ = _workspace_projects(db)
    reservation = quota.reserve_artifact_bytes(
        db, project_id=project.id, owner_type="test", owner_id=1,
        idempotency_key="renew-race", requested_bytes=10_000, actor_user_id=actor.id,
    )
    locked = SimpleNamespace(id=reservation.id, status="committed")
    monkeypatch.setattr(quota, "_locked_reservation", lambda *_: (None, locked))
    monkeypatch.setattr(
        quota, "renew_artifact_reservation",
        lambda *_a, **_k: pytest.fail("A committed reservation must not be renewed"),
    )
    assert quota.renew_owner_artifact_reservation(db, owner_type="test", owner_id=1) is locked


def test_orphan_insert_failure_preserves_caller_transaction(db):
    _, _, _, _, project, _ = _workspace_projects(db)
    db.commit()
    project.description = "caller mutation"

    def duplicate(*_):
        raise IntegrityError("orphan insert", {}, RuntimeError("concurrent duplicate"))

    event.listen(OrphanedStorageObject, "before_insert", duplicate)
    try:
        result = record_orphaned_object(db, "unused", origin="audit", commit=False)
    finally:
        event.remove(OrphanedStorageObject, "before_insert", duplicate)
    assert result is None
    db.commit()
    db.expire_all()
    assert db.get(Project, project.id).description == "caller mutation"


@pytest.mark.parametrize("source", ["values", "enum", "items.enum"])
def test_declared_label_vocabulary_retains_unobserved_classes(source):
    label_rules = {"values": ["common", "rare"]} if source == "values" else {}
    output_schema = (
        {"enum": ["common", "rare"]} if source == "enum"
        else {"items": {"enum": ["common", "rare"]}} if source == "items.enum" else {}
    )
    task = SimpleNamespace(label_rules=label_rules, output_schema=output_schema)
    assert _declared_label_vocabulary(task, "label", ["common"]) == ("common", "rare")


def test_sentence_tagger_implementation_is_registered():
    assert model_types.get("evidence_block_sentence_tagger").key == "evidence_block_sentence_tagger"


def test_window_keeps_reviewed_gold_when_another_block_is_unreviewed():
    builder = EvidenceBlockWindowBuilder(
        lambda text: len(text.split()),
        WindowBuilderConfig(max_tokens=4096, overlap_tokens=0, target_conditioning=False),
    )
    result = builder.build(
        document_id=1, structure_version_id=2, target=_target(), sentences=_sentences(6),
        gold_blocks=[
            GoldBlockRange(id=1, start_ordinal=0, end_ordinal=0),
            GoldBlockRange(id=2, start_ordinal=3, end_ordinal=3),
        ],
        reviewed_intervals=[ReviewedInterval(start_ordinal=0, end_ordinal=1)],
    )
    assert result.windows
    assert any(window.start_sentence_ordinal == 0 for window in result.windows)
    assert all(3 not in [sentence.ordinal for sentence in w.sentences] for w in result.windows)


def test_bio_predictions_do_not_merge_across_unreviewed_sentence_gaps():
    blocks = blocks_from_bio(["B", "I"], sentence_ordinals=(0, 3))
    assert [(block.start, block.end) for block in blocks] == [(0, 0), (1, 1)]


def test_conventional_inference_retains_model_uncertainty():
    rows = sentence_score_probabilities([0.2, 0.8, 0.6], threshold=0.5)
    expected = [(0.8, 0.2, 0), (0.2, 0.8, 0), (0.4, 0, 0.6)]
    for actual, desired in zip(rows, expected, strict=True):
        assert actual == pytest.approx(desired)


def test_inference_window_materialization_can_roll_back(db):
    scope = _inference_scope(
        db, with_window=False,
        window_config={"max_tokens": 128, "overlap_tokens": 0},
    )
    windows = inference_service.materialize_inference_windows(
        db, run_id=scope.run.id, token_counter=lambda _: 1, commit=False,
    )
    assert windows
    db.rollback()
    assert db.query(InferenceWindow).filter_by(run_id=scope.run.id).count() == 0


def test_incomplete_inference_candidates_are_hidden(db):
    scope = _inference_scope(db, run_status="failed")
    candidates = inference_service.persist_decoder_result(
        db, run_id=scope.run.id, document_id=scope.window.document_id,
        structure_version_id=scope.window.structure_version_id,
        target_version_id=scope.target_version.id, result=_decoder_result(scope.sentences),
        source_window_ids=[scope.window.id],
    )
    assert candidates
    assert inference_service.list_candidates(db, run_id=scope.run.id, user=scope.user) == []


def test_manager_can_filter_reviews_made_by_another_annotator(db):
    scope = _inference_scope(db)
    candidates = inference_service.persist_decoder_result(
        db, run_id=scope.run.id, document_id=scope.window.document_id,
        structure_version_id=scope.window.structure_version_id,
        target_version_id=scope.target_version.id, result=_decoder_result(scope.sentences),
        source_window_ids=[scope.window.id],
    )
    candidates[0].status = "accepted"
    db.query(WorkspaceMember).filter_by(user_id=scope.user.id).update({"role": "manager"})
    db.commit()
    page = inference_service.list_candidates(
        db, run_id=scope.run.id, user=scope.user, status="accepted",
    )
    assert [candidate.id for candidate in page] == [candidates[0].id]
    assert page[0].review_status == "accepted"


def test_candidate_publication_keeps_quota_until_promotion(db, monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(
        db, storage=storage, suffix="quota-through-promotion",
        evaluation_plan={"splits": ["test"], "metrics": ["accuracy"]},
    )
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    quota_row = db.query(WorkspaceArtifactQuota).filter_by(workspace_id=scope.workspace.id).one()
    quota_row.limit_bytes = reservation.reserved_bytes
    db.commit()
    trainer = FakeTrainer(
        key="sklearn_tfidf", recipe_key="tfidf_logistic_regression",
        runtime_class="classical-cpu", output_name="model.skops",
    )
    trainers = TrainerPluginRegistry()
    trainers.register(trainer)

    class CheckingEvaluator(FakeEvaluator):
        def evaluate(self, **kwargs):
            reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
            assert reservation.status == "active"
            current_quota = db.query(WorkspaceArtifactQuota).filter_by(
                workspace_id=scope.workspace.id,
            ).one()
            assert current_quota.reserved_bytes == (
                reservation.reserved_bytes - reservation.committed_bytes
            )
            assert (
                current_quota.used_bytes + current_quota.reserved_bytes
                <= current_quota.limit_bytes
            )
            from al_medlit.core.exceptions import ConflictError

            with pytest.raises(ConflictError, match="quota exceeded"):
                quota.reserve_artifact_bytes(
                    db, project_id=scope.project.id, owner_type="other", owner_id=2,
                    idempotency_key="concurrent-admission", requested_bytes=1,
                    actor_user_id=scope.actor.id,
                )
            return super().evaluate(**kwargs)

    evaluators = EvaluatorPluginRegistry()
    evaluators.register(CheckingEvaluator(recipe_key="tfidf_logistic_regression", trainer=trainer))
    run = execute_training_run(
        db, storage, training_run_id=scope.run.id,
        trainer_registry=trainers, evaluator_registry=evaluators,
    )
    assert run.status == "succeeded"
    assert db.get(ArtifactStorageReservation, run.artifact_reservation_id).status == "committed"


def test_broker_dispatch_failure_releases_training_reservation(db, client, monkeypatch):
    from al_medlit.workflow.routes import training as training_routes

    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix="dispatch-recovery")

    def broker_unavailable(*_a, **_k):
        raise OSError("broker unavailable")

    monkeypatch.setattr(training_routes, "enqueue_training_run", broker_unavailable)
    payload = {name: getattr(scope.run, name) for name in (
        "project_id", "registered_model_id", "task_version_id", "training_dataset_version_id",
        "recipe_version_id", "environment_id", "storage_policy_id", "idempotency_key",
        "evaluation_plan", "config", "seed",
    )}
    payload["artifact_reservation_bytes"] = 1024 * 1024
    response = client.post("/api/training-runs", json=payload, headers=_headers(scope.actor))
    assert response.status_code == 201, response.text
    assert response.json()["status"] == "failed"
    assert response.json()["failure_code"] == "broker_dispatch_failed"
    db.expire_all()
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert reservation.status == "released"


def test_managers_can_provision_imported_inference_resources(db, client, object_storage):
    scope = _inference_scope(db)
    db.query(WorkspaceMember).filter_by(user_id=scope.user.id).update({"role": "manager"})
    db.get(Workspace, scope.project.workspace_id).capability_preset = "full"
    db.commit()
    package = artifact_service.publish_artifact_package(
        db, object_storage, project_id=scope.project.id,
        data=_package_spec("Imported evidence").model_copy(update={
            "model_type": "evidence_block_sentence_tagger",
            "task_contract": {"key": "evidence_blocks"},
            "metadata": {
                "trained_target_version_ids": [scope.target_version.id],
                "training_mode": "conditioned", "max_context_tokens": 512,
            },
        }),
        files=_package_files(), actor_user_id=scope.user.id,
    )
    db.commit()
    headers = _headers(scope.user)
    profile = client.post(
        f"/api/projects/{scope.project.id}/inference/compute-profiles",
        json={"name": "Imported local", "backend": "local", "config": {}}, headers=headers,
    )
    assert profile.status_code == 200, profile.text
    assert "config" not in profile.json()
    checkpoint = client.post(
        f"/api/projects/{scope.project.id}/inference/checkpoints",
        json={
            "artifact_package_id": package.id, "training_mode": "conditioned",
            "target_version_ids": [scope.target_version.id], "max_context_tokens": 512,
        }, headers=headers,
    )
    assert checkpoint.status_code == 200, checkpoint.text
    from al_medlit.training.models import ModelCheckpoint

    assert db.get(ModelCheckpoint, checkpoint.json()["id"]).training_job_id is None
    listing = client.get(
        f"/api/projects/{scope.project.id}/inference/checkpoints", headers=headers,
    )
    assert any(row["id"] == checkpoint.json()["id"] for row in listing.json())
