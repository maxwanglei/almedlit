"""Inference execution must decode window scores as the kind each producer emits."""

import gzip
import json
import math
import zipfile
from types import SimpleNamespace

import pytest
from test_inference_candidate_queries import _inference_scope

from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ValidationError
from al_medlit.evidence.models import EvidenceTargetVersion
from al_medlit.inference import execution, service
from al_medlit.inference.models import EvidenceCandidatePrediction, InferenceWindow
from al_medlit.lineage.models import LineageArtifact
from al_medlit.training.compute.base import JobState
from al_medlit.training.model_types.catalog import builtin_model_descriptors
from al_medlit.training.model_types.evidence_conventional import model as conventional
from al_medlit.training.runner import RESULT_SCHEMA_VERSION
from al_medlit.workspace.models import Workspace, WorkspaceMember

DECODER_CONFIG = {
    "version": "evidence-block-decoder-v1",
    "block_threshold": 0.5,
    "allow_cross_section": False,
    "merge_adjacent": False,
}
WINDOW_CONFIG = {"max_tokens": 4096, "overlap_tokens": 512, "aggregation": "mean"}
ONE_HOT = {"O": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "I": [0.0, 0.0, 1.0]}
WINDOW_SCORE_KINDS = {
    "evidence_crf": "probabilities",
    "evidence_svm": "probabilities",
    "evidence_random_forest": "probabilities",
    "evidence_lora": "probabilities",
    "evidence_qlora": "probabilities",
    "evidence_bilstm": "probabilities",
    "evidence_cnn": "probabilities",
    "evidence_block_sentence_tagger": "logits",
}


class _RemoteResultBackend:
    """Serves a runner result the way SSHSlurmComputeBackend.collect_outputs does."""

    def __init__(self, run_id: int, inference_result: dict) -> None:
        self.run_id = run_id
        self.inference_result = inference_result

    def collect_scheduler_logs(self, *, job_key, output_root):
        pass

    def collect_outputs(self, *, job_key, output_root, limits):
        (output_root / "inference-result.json").write_text(
            json.dumps(self.inference_result),
            encoding="utf-8",
        )
        (output_root / "infer.log").write_text("ok\n", encoding="utf-8")
        (output_root / "artifact-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    "kind": "inference",
                    "job_key": f"inference:{self.run_id}",
                    "status": "succeeded",
                    "files": {},
                }
            ),
            encoding="utf-8",
        )


def _runner_result(scope, windows: dict[str, dict[int, list[float]]]) -> dict:
    """Return an untagged result shaped exactly like runner.run_inference_job writes it."""
    sentence = scope.sentences[0]
    rows = []
    for stable_key, scores in windows.items():
        ordinals = sorted(scores)
        rows.append(
            {
                "stable_key": stable_key,
                "document_id": sentence.structure_version.document_id,
                "structure_version_id": sentence.structure_version_id,
                "target_version_id": scope.target_version.id,
                "start_sentence_ordinal": ordinals[0],
                "end_sentence_ordinal": ordinals[-1],
                "token_count": 2 * len(ordinals),
                "logits": {str(ordinal): values for ordinal, values in scores.items()},
            }
        )
    return {
        "schema_version": "inference-window-logits-v1",
        "checkpoint_checksum_sha256": scope.run.checkpoint.artifact.content_hash,
        "synthetic_mode": False,
        "windows": rows,
    }


def _finalize(db, object_storage, tmp_path, scope, inference_result):
    return execution.finalize_remote_inference(
        db,
        object_storage,
        run_id=scope.run.id,
        backend=_RemoteResultBackend(scope.run.id, inference_result),
        work_root=tmp_path / "work",
    )


def _candidates(db, run_id: int) -> list[EvidenceCandidatePrediction]:
    return (
        db.query(EvidenceCandidatePrediction)
        .filter(EvidenceCandidatePrediction.run_id == run_id)
        .order_by(EvidenceCandidatePrediction.start_sentence_ordinal)
        .all()
    )


def _spans(candidates: list[EvidenceCandidatePrediction]) -> list[tuple[int, int]]:
    return [
        (candidate.start_sentence_ordinal, candidate.end_sentence_ordinal)
        for candidate in candidates
    ]


def test_every_builtin_model_type_declares_its_window_score_kind():
    assert {descriptor.key for descriptor in builtin_model_descriptors()} == set(WINDOW_SCORE_KINDS)
    for model_type, score_kind in WINDOW_SCORE_KINDS.items():
        assert execution._window_score_kind({"model_type": model_type}) == score_kind


@pytest.mark.parametrize(
    ("manifest", "score_kind"),
    [
        ({}, "logits"),  # The producers default to the sentence tagger.
        ({"synthetic_mode": True, "model_type": "evidence_svm"}, "logits"),
        ({"model_type": "community_plugin"}, "logits"),
    ],
)
def test_window_score_kind_follows_producer_dispatch(manifest, score_kind):
    assert execution._window_score_kind(manifest) == score_kind


def test_finalize_decodes_one_hot_labels_without_softmax(db, object_storage, tmp_path):
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_crf"},
        decoder_config={**DECODER_CONFIG, "block_threshold": 0.9},
        with_window=False,
    )
    labels = ("B", "I", "O", "O")
    inference_result = _runner_result(
        scope,
        {"window-0": {ordinal: ONE_HOT[label] for ordinal, label in enumerate(labels)}},
    )

    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    candidates = _candidates(db, run.id)
    assert _spans(candidates) == [(0, 1)]
    assert candidates[0].block_confidence == 1.0
    assert candidates[0].boundary_confidence == {"start": 1.0, "end": 1.0}
    assert candidates[0].uncertainty == 0.0
    assert candidates[0].decoder_version == "evidence-block-decoder-v2"
    assert run.metrics["score_kind"] == "probabilities"
    assert run.metrics["decoder_version"] == "evidence-block-decoder-v2"


def test_finalize_keeps_neural_probabilities_on_their_scale(db, object_storage, tmp_path):
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_bilstm"},
        decoder_config=DECODER_CONFIG,
        with_window=False,
    )
    probabilities = [
        [0.15, 0.70, 0.15],
        [0.10, 0.20, 0.70],
        [0.80, 0.10, 0.10],
        [0.90, 0.05, 0.05],
    ]
    inference_result = _runner_result(scope, {"window-0": dict(enumerate(probabilities))})

    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    candidates = _candidates(db, run.id)
    assert _spans(candidates) == [(0, 1)]
    assert candidates[0].block_confidence == pytest.approx(0.70)
    assert candidates[0].boundary_confidence["end"] == pytest.approx(0.90)


def test_finalize_still_softmaxes_transformer_logits(db, object_storage, tmp_path):
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_block_sentence_tagger"},
        decoder_config=DECODER_CONFIG,
        with_window=False,
    )
    logits = [[0.0, 6.0, 0.0], [0.0, 0.0, 6.0], [6.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    inference_result = _runner_result(
        scope,
        {
            "window-0": {ordinal: logits[ordinal] for ordinal in (0, 1, 2)},
            "window-1": {ordinal: logits[ordinal] for ordinal in (1, 2, 3)},
        },
    )

    run = _finalize(db, object_storage, tmp_path, scope, inference_result)

    candidates = _candidates(db, run.id)
    assert _spans(candidates) == [(0, 1)]
    assert candidates[0].block_confidence == pytest.approx(math.exp(6) / (math.exp(6) + 2))
    assert run.metrics["score_kind"] == "logits"


def test_local_inference_decodes_conventional_labels_as_probabilities(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
):
    labels = ["B", "I", "O", "O"]
    predictor = SimpleNamespace(produces_sentence_scores=False)
    monkeypatch.setattr(conventional, "load_conventional_model", lambda root: predictor)
    monkeypatch.setattr(conventional, "actual_token_count", lambda model, text: len(text.split()))
    monkeypatch.setattr(
        conventional,
        "predict_window",
        lambda model, *, target_text, sentences: labels[: len(sentences)],
    )
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("checkpoint/model.json", "{}")
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_crf"},
        checkpoint_object=object_storage.put_file(
            "checkpoints/crf.zip",
            archive,
            content_type="application/zip",
        ),
        window_config=WINDOW_CONFIG,
        decoder_config={**DECODER_CONFIG, "block_threshold": 0.9},
        run_status="queued",
        with_window=False,
    )

    run = execution.execute_local_inference(
        db,
        object_storage,
        run_id=scope.run.id,
        work_root=tmp_path / "work",
    )

    assert run.status == "succeeded"
    candidates = _candidates(db, run.id)
    assert _spans(candidates) == [(0, 1)]
    assert candidates[0].block_confidence == 1.0
    assert candidates[0].uncertainty == 0.0
    assert run.metrics["score_kind"] == "probabilities"
    assert run.metrics["decoder_version"] == "evidence-block-decoder-v2"
    artifact = db.get(LineageArtifact, run.diagnostics_artifact_id)
    diagnostics = json.loads(gzip.decompress(object_storage.get_bytes(artifact.storage_key)))
    assert diagnostics["score_kind"] == "probabilities"
    assert diagnostics["decoder_version"] == "evidence-block-decoder-v2"
    assert diagnostics["scopes"][0]["sentences"][0]["probabilities"] == ONE_HOT["B"]


@pytest.mark.parametrize("all_oversized", [False, True])
def test_local_inference_rejects_oversized_sentences_before_prediction(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
    all_oversized,
):
    long_sentence = " ".join(["Beta"] * 20) + " improved."
    document_text = (
        " ".join([long_sentence] * 4)
        if all_oversized
        else f"Alpha improved. {long_sentence} Gamma improved. Delta improved."
    )
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("checkpoint/model.json", "{}")
    scope = _inference_scope(
        db,
        checkpoint_object=object_storage.put_file(
            "checkpoints/synthetic.zip",
            archive,
            content_type="application/zip",
        ),
        window_config={**WINDOW_CONFIG, "max_tokens": 16, "overlap_tokens": 0},
        decoder_config=DECODER_CONFIG,
        run_status="queued",
        with_window=False,
        document_text=document_text,
    )
    prediction_calls = []
    original_predict = execution.synthetic_logits

    def predict(ordinal):
        prediction_calls.append(ordinal)
        return original_predict(ordinal)

    monkeypatch.setattr(execution, "synthetic_logits", predict)

    with pytest.raises(ValidationError, match="Oversized sentence") as error:
        execution.execute_local_inference(
            db,
            object_storage,
            run_id=scope.run.id,
            work_root=tmp_path / "work",
        )

    assert prediction_calls == []
    assert scope.run.windows == []
    assert scope.run.diagnostics_artifact_id is None
    db.expire(scope.run)
    assert scope.run.status == "failed"
    assert scope.run.failure_reason == str(error.value)
    assert scope.run.completed_at is not None
    assert scope.run.started_at is not None
    assert scope.run.external_job_id == f"local:inference-{scope.run.id}"


def test_local_inference_failure_after_claim_marks_run_failed(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
):
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("checkpoint/model.json", "{}")
    scope = _inference_scope(
        db,
        checkpoint_object=object_storage.put_file(
            "checkpoints/synthetic.zip",
            archive,
            content_type="application/zip",
        ),
        window_config=WINDOW_CONFIG,
        decoder_config=DECODER_CONFIG,
        run_status="queued",
        with_window=False,
    )

    def explode(*args, **kwargs):
        raise RuntimeError("diagnostics storage unavailable")

    monkeypatch.setattr(service, "store_inference_diagnostics", explode)

    with pytest.raises(RuntimeError, match="diagnostics storage unavailable"):
        execution.execute_local_inference(
            db,
            object_storage,
            run_id=scope.run.id,
            work_root=tmp_path / "work",
        )

    db.rollback()
    db.expire(scope.run)
    assert scope.run.status == "failed"
    assert scope.run.failure_reason == "diagnostics storage unavailable"
    assert scope.run.completed_at is not None


def test_cancel_route_releases_a_local_run_whose_executor_died_after_claim(
    client,
    db,
    object_storage,
    tmp_path,
    monkeypatch,
):
    archive = tmp_path / "checkpoint.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("checkpoint/model.json", "{}")
    scope = _inference_scope(
        db,
        checkpoint_object=object_storage.put_file(
            "checkpoints/synthetic.zip",
            archive,
            content_type="application/zip",
        ),
        window_config=WINDOW_CONFIG,
        decoder_config=DECODER_CONFIG,
        run_status="queued",
        with_window=False,
    )
    scope.run.compute_profile.config = {"runtime_root": str(tmp_path / "attempts")}
    db.get(Workspace, scope.project.workspace_id).capability_preset = "full"
    db.query(WorkspaceMember).filter(WorkspaceMember.user_id == scope.user.id).update(
        {"role": "trainer"}
    )
    db.commit()

    def executor_process_dies(*args, **kwargs):
        raise SystemExit("executor process terminated")

    monkeypatch.setattr(service, "store_inference_diagnostics", executor_process_dies)
    with pytest.raises(SystemExit):
        execution.execute_local_inference(
            db,
            object_storage,
            run_id=scope.run.id,
            work_root=tmp_path / "work",
        )
    db.rollback()
    db.expire(scope.run)
    assert scope.run.status == "running"

    headers = {"Authorization": f"Bearer {create_access_token(scope.user.id)}"}
    monkeypatch.setattr(
        service,
        "build_compute_backend",
        lambda _: pytest.fail("Local cancellation must not construct a scheduler backend"),
    )
    cancelled = client.post(f"/api/inference/runs/{scope.run.id}/cancel", headers=headers)
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["completed_at"] is not None

    retried = client.post(f"/api/inference/runs/{scope.run.id}/execute", headers=headers)
    assert retried.status_code == 409, retried.text
    db.expire(scope.run)
    assert scope.run.status == "cancelled"


def test_inference_window_validation_leaves_no_partial_windows_for_earlier_targets(db):
    scope = _inference_scope(
        db,
        window_config={**WINDOW_CONFIG, "max_tokens": 16, "overlap_tokens": 0},
        run_status="queued",
        with_window=False,
    )
    long_target = EvidenceTargetVersion(
        target_id=scope.target_version.target_id,
        version_number=2,
        text="one two three four five six seven eight",
        created_by_user_id=scope.user.id,
    )
    db.add(long_target)
    db.flush()
    # Every sentence fits the first target's budget, but fails the second's.
    scope.run.target_version_ids = [scope.target_version.id, long_target.id]
    db.commit()

    with pytest.raises(ValidationError, match=f"target version {long_target.id}"):
        service.materialize_inference_windows(
            db,
            run_id=scope.run.id,
            token_counter=lambda text: len(text.split()),
        )

    db.commit()
    assert db.query(InferenceWindow).filter(InferenceWindow.run_id == scope.run.id).count() == 0


def test_remote_cancellation_uses_the_scheduler_backend(db, monkeypatch):
    scope = _inference_scope(db, run_status="running")
    scope.run.compute_profile.backend = "ssh_slurm"
    scope.run.external_job_id = "slurm:123"
    db.commit()
    cancelled_jobs = []

    def cancel(job_id):
        cancelled_jobs.append(job_id)
        return JobState(status="cancelled", raw_state="CANCELLED")

    backend = SimpleNamespace(cancel=cancel)
    monkeypatch.setattr(service, "build_compute_backend", lambda _: backend)

    run = service.cancel_inference_run(db, scope.run.id)

    assert cancelled_jobs == ["slurm:123"]
    assert run.status == "cancelled"
    assert run.completed_at is not None


def _sentence_scorer_checkpoint(tmp_path, storage, *, model_type, threshold):
    archive = tmp_path / "sentence-scorer.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "checkpoint/al-medlit-conventional.json",
            json.dumps(
                {
                    "schema_version": conventional.MODEL_SCHEMA_VERSION,
                    "model_type": model_type,
                    "config": {"sentence_score_threshold": threshold},
                }
            ),
        )
    return storage.put_file("checkpoints/sentence-scorer.zip", archive)


@pytest.mark.parametrize("model_type", ["evidence_svm", "evidence_random_forest"])
@pytest.mark.parametrize(
    ("score", "threshold", "positive"),
    [
        (0.7, 0.8, False),
        (0.4, 0.3, True),
        (0.5, 0.5, True),
    ],
)
def test_local_sentence_scorer_respects_verified_predictor_threshold(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
    model_type,
    score,
    threshold,
    positive,
):
    predictor = SimpleNamespace(
        produces_sentence_scores=True,
        config=SimpleNamespace(sentence_score_threshold=threshold),
    )
    monkeypatch.setattr(conventional, "load_conventional_model", lambda root: predictor)
    monkeypatch.setattr(conventional, "actual_token_count", lambda model, text: len(text.split()))
    monkeypatch.setattr(conventional, "predict_window", lambda model, **kwargs: [score, 0, 0, 0])
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": model_type},
        checkpoint_object=_sentence_scorer_checkpoint(
            tmp_path,
            object_storage,
            model_type=model_type,
            threshold=threshold,
        ),
        window_config=WINDOW_CONFIG,
        decoder_config={**DECODER_CONFIG, "block_threshold": 0},
        run_status="queued",
        with_window=False,
    )
    run = execution.execute_local_inference(
        db,
        object_storage,
        run_id=scope.run.id,
        work_root=tmp_path / "work",
    )
    candidates = _candidates(db, run.id)
    assert _spans(candidates) == ([(0, 0)] if positive else [])
    if positive:
        assert candidates[0].block_confidence == pytest.approx(score)
        assert candidates[0].uncertainty > 0


@pytest.mark.parametrize("model_type", ["evidence_svm", "evidence_random_forest"])
def test_remote_sentence_scorer_pins_metadata_and_preserves_probability(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
    model_type,
):
    from al_medlit.training.runner import run_inference_job

    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": model_type},
        checkpoint_object=_sentence_scorer_checkpoint(
            tmp_path,
            object_storage,
            model_type=model_type,
            threshold=0.3,
        ),
        window_config=WINDOW_CONFIG,
        decoder_config={**DECODER_CONFIG, "block_threshold": 0},
        with_window=False,
    )
    bundle = execution.build_inference_bundle(
        db,
        object_storage,
        run_id=scope.run.id,
        backend=SimpleNamespace(render_sbatch_script=lambda bundle: "#!/bin/sh\n"),
        work_root=tmp_path / "work",
    )
    job_path = bundle.local_bundle_path / "job.json"
    assert json.loads(job_path.read_text())["checkpoint"]["sentence_score_threshold"] == 0.3
    predictor = SimpleNamespace(
        produces_sentence_scores=True,
        config=SimpleNamespace(sentence_score_threshold=0.3),
    )
    monkeypatch.setattr(conventional, "load_conventional_model", lambda root: predictor)
    monkeypatch.setattr(conventional, "actual_token_count", lambda model, text: len(text.split()))
    monkeypatch.setattr(
        conventional, "predict_window", lambda model, **kwargs: [0.4, 0.2, 0.1, 0.1]
    )
    run_inference_job(job_path)
    inference_result = json.loads(
        (bundle.local_bundle_path / "outputs/inference-result.json").read_text(),
    )
    assert inference_result["sentence_score_threshold"] == 0.3
    run = _finalize(db, object_storage, tmp_path, scope, inference_result)
    candidates = _candidates(db, run.id)
    assert _spans(candidates) == [(0, 0)]
    assert candidates[0].block_confidence == pytest.approx(0.4)
    assert candidates[0].boundary_confidence["end"] == pytest.approx(0.8)
    assert candidates[0].uncertainty > 0


@pytest.mark.parametrize("reported_threshold", [None, 0.5])
def test_remote_sentence_scorer_rejects_unverified_threshold(
    db,
    object_storage,
    tmp_path,
    reported_threshold,
):
    from al_medlit.core.exceptions import ConflictError

    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_svm"},
        checkpoint_object=_sentence_scorer_checkpoint(
            tmp_path,
            object_storage,
            model_type="evidence_svm",
            threshold=0.8,
        ),
        decoder_config={**DECODER_CONFIG, "block_threshold": 0},
        with_window=False,
    )
    result = _runner_result(
        scope,
        {
            "window-0": {
                0: [0.3, 0.7, 0],
                1: [0.9, 0.1, 0],
                2: [1, 0, 0],
                3: [1, 0, 0],
            }
        },
    )
    result["sentence_score_threshold"] = reported_threshold
    with pytest.raises(ConflictError, match="threshold does not match"):
        _finalize(db, object_storage, tmp_path, scope, result)
    assert not _candidates(db, scope.run.id)
