"""Inference execution must decode window scores as the kind each producer emits."""

import gzip
import json
import math
import zipfile
from types import SimpleNamespace

import pytest
from test_inference_candidate_queries import _inference_scope

from al_medlit.core.exceptions import ValidationError
from al_medlit.evidence.models import EvidenceTargetVersion
from al_medlit.inference import execution, service
from al_medlit.inference.models import EvidenceCandidatePrediction, InferenceWindow
from al_medlit.lineage.models import LineageArtifact
from al_medlit.training.model_types.catalog import builtin_model_descriptors
from al_medlit.training.model_types.evidence_conventional import model as conventional
from al_medlit.training.runner import RESULT_SCHEMA_VERSION

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
        checkpoint_manifest={"model_type": "evidence_svm"},
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
    original_predict = execution._synthetic_logits

    def predict(ordinal):
        prediction_calls.append(ordinal)
        return original_predict(ordinal)

    monkeypatch.setattr(execution, "_synthetic_logits", predict)

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
    assert scope.run.started_at is None
    assert scope.run.external_job_id is None


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
