import pytest
from test_inference_candidate_queries import _inference_scope
from test_inference_execution import (
    DECODER_CONFIG,
    ONE_HOT,
    _candidates,
    _finalize,
    _runner_result,
)

from al_medlit.inference import service
from al_medlit.inference.models import InferenceWindow
from al_medlit.lineage.models import LineageArtifact


def _scope_and_result(db, *, labels=("B", "I", "O", "O")):
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_svm"},
        decoder_config=DECODER_CONFIG,
        with_window=False,
    )
    result = _runner_result(
        scope,
        {"window-0": {ordinal: ONE_HOT[label] for ordinal, label in enumerate(labels)}},
    )
    return scope, result


def _artifact_count(db, run, artifact_type):
    return (
        db.query(LineageArtifact)
        .filter(
            LineageArtifact.project_id == run.project_id,
            LineageArtifact.artifact_type == artifact_type,
        )
        .count()
    )


@pytest.mark.parametrize("failure_stage", ["candidates", "log"])
def test_remote_finalization_rolls_back_failed_outputs_and_retries(
    db, object_storage, tmp_path, monkeypatch, failure_stage
):
    scope, inference_result = _scope_and_result(db)
    original_put_file = object_storage.put_file

    def fail_candidate_storage(*args, **kwargs):
        raise RuntimeError("candidate storage unavailable")

    def fail_log_upload(key, *args, **kwargs):
        if key.endswith("/infer.log"):
            raise RuntimeError("log upload unavailable")
        return original_put_file(key, *args, **kwargs)

    with monkeypatch.context() as failure:
        if failure_stage == "candidates":
            failure.setattr(service, "persist_decoder_result", fail_candidate_storage)
        else:
            failure.setattr(object_storage, "put_file", fail_log_upload)
        with pytest.raises(RuntimeError, match="unavailable"):
            _finalize(db, object_storage, tmp_path, scope, inference_result)

    assert scope.run.status == "succeeded"
    assert scope.run.diagnostics_artifact_id is None
    assert _candidates(db, scope.run.id) == []
    assert db.query(InferenceWindow).filter(InferenceWindow.run_id == scope.run.id).count() == 0
    assert _artifact_count(db, scope.run, "inference_diagnostics") == 0
    assert _artifact_count(db, scope.run, "inference_log") == 0

    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    assert run.status == "succeeded"
    assert len(_candidates(db, run.id)) == 1
    assert run.metrics["candidate_count"] == 1
    assert run.metrics["inference_log_artifact_id"] is not None
    assert _artifact_count(db, run, "inference_diagnostics") == 1
    assert _artifact_count(db, run, "inference_log") == 1


def test_remote_finalization_recovers_preexisting_partial_diagnostics(
    db, object_storage, tmp_path
):
    scope, inference_result = _scope_and_result(db)
    diagnostics_id = service.store_inference_diagnostics(
        db,
        object_storage,
        run_id=scope.run.id,
        diagnostics=inference_result,
        actor_user_id=scope.user.id,
    )

    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    assert run.diagnostics_artifact_id == diagnostics_id
    assert len(_candidates(db, run.id)) == 1
    assert run.metrics["candidate_count"] == 1
    assert run.metrics["inference_log_artifact_id"] is not None
    assert _artifact_count(db, run, "inference_diagnostics") == 1
    assert _artifact_count(db, run, "inference_log") == 1


def test_remote_finalization_with_no_candidates_is_complete_on_retry(
    db, object_storage, tmp_path, monkeypatch
):
    scope, inference_result = _scope_and_result(db, labels=("O", "O", "O", "O"))
    run = _finalize(db, object_storage, tmp_path, scope, inference_result)
    artifact_ids = (run.diagnostics_artifact_id, run.metrics["inference_log_artifact_id"])

    def unavailable_storage(*args, **kwargs):
        raise RuntimeError("storage is offline after finalization")

    monkeypatch.setattr(object_storage, "put_file", unavailable_storage)
    monkeypatch.setattr(object_storage, "put_stream", unavailable_storage)
    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    assert run.status == "succeeded"
    assert _candidates(db, run.id) == []
    assert run.metrics["candidate_count"] == 0
    assert (run.diagnostics_artifact_id, run.metrics["inference_log_artifact_id"]) == artifact_ids
    assert _artifact_count(db, run, "inference_diagnostics") == 1
    assert _artifact_count(db, run, "inference_log") == 1
