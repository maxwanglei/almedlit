import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from al_medlit.training import runner


def _inference_bundle(tmp_path: Path, sentence_texts: list[str]) -> Path:
    checkpoint_manifest = {
        "model_type": "evidence_block_sentence_tagger",
        "synthetic_mode": True,
    }
    checkpoint_path = tmp_path / "checkpoint.zip"
    with ZipFile(checkpoint_path, "w") as archive:
        archive.writestr("checkpoint/manifest.json", json.dumps(checkpoint_manifest))
    sentences = []
    offset = 0
    for ordinal, text in enumerate(sentence_texts):
        sentences.append(
            {
                "id": 100 + ordinal,
                "ordinal": ordinal,
                "paragraph_ordinal": ordinal,
                "section_path": ["Results"],
                "text": text,
                "start_char": offset,
                "end_char": offset + len(text),
            }
        )
        offset += len(text) + 1
    corpus_path = tmp_path / "inference-input.json"
    corpus_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"document_id": 11, "structure_version_id": 22, "sentences": sentences}
                ],
                "targets": [{"id": 7, "key": "pk", "name": "PK", "text": "evidence"}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "job.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": runner.RUNNER_SCHEMA_VERSION,
                "kind": "inference",
                "job_key": "inference:1",
                "checkpoint": {
                    "path": checkpoint_path.name,
                    "checksum_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    "training_mode": "conditioned",
                    "manifest": checkpoint_manifest,
                },
                "corpus": {
                    "path": corpus_path.name,
                    "checksum_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                },
                "window_config": {"max_tokens": 12, "overlap_tokens": 0},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize(
    "sentence_texts",
    [["Fits.", "one two three four five", "Also fits."], ["one two three four five"]],
    ids=["mixed", "all-oversized"],
)
def test_remote_inference_rejects_oversized_sentences_before_prediction(
    tmp_path, monkeypatch, sentence_texts
):
    manifest_path = _inference_bundle(tmp_path, sentence_texts)
    prediction_calls = []

    def record_prediction(ordinal):
        prediction_calls.append(ordinal)
        return (6.0, 0.0, 0.0)

    monkeypatch.setattr(runner, "_synthetic_inference_logits", record_prediction)

    with pytest.raises(runner.RunnerError, match="Oversized sentence"):
        runner.run_inference_job(manifest_path)

    assert prediction_calls == []
    assert not (tmp_path / "outputs" / "inference-result.json").exists()
    assert not (tmp_path / "outputs" / "artifact-manifest.json").exists()


def test_remote_inference_preserves_every_sentence_when_longest_exactly_fits(tmp_path):
    # Three target tokens and four reserved tokens leave five tokens per window,
    # including the sentence marker: four words fit exactly.
    manifest_path = _inference_bundle(tmp_path, ["one two three four", "Fits."])

    result = runner.run_inference_job(manifest_path)

    assert result["status"] == "succeeded"
    output = json.loads((tmp_path / "outputs" / "inference-result.json").read_text())
    assert {ordinal for window in output["windows"] for ordinal in window["logits"]} == {
        "0",
        "1",
    }
    assert all(window["token_count"] <= 12 for window in output["windows"])
