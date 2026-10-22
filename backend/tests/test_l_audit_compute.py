import json

import pytest


def test_skops_loader_rejects_unexpected_library_type_before_deserialization(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from al_medlit.training.model_types.evidence_conventional import model

    (tmp_path / "model.skops").touch()
    monkeypatch.setattr(model, "_load_metadata", lambda _: {"model_type": "evidence_svm"})
    monkeypatch.setattr(
        model,
        "_require_sklearn_and_skops",
        lambda: {
            "skops_io": SimpleNamespace(
                get_untrusted_types=lambda **_: ["sklearn.unexpected.Executable"],
                load=lambda *_, **__: pytest.fail("An unapproved type reached deserialization"),
            ),
        },
    )
    with pytest.raises(ValueError, match="allowlist"):
        model.load_conventional_model(tmp_path)


def test_neural_documents_can_be_bounded_without_dropping_sentence_ordinals():
    from al_medlit.training.model_types.evidence_neural.data import documents_from_window_rows

    rows = [
        {
            "document_id": "long",
            "target": {"id": 1, "text": "Outcome"},
            "sentences": [
                {"ordinal": ordinal, "text": f"Sentence {ordinal}", "label": "O"}
                for ordinal in range(513)
            ],
        }
    ]
    documents = documents_from_window_rows(rows, max_sentences=512)
    assert [len(item.sentences) for item in documents] == [512, 1]
    assert [ordinal for item in documents for ordinal in item.sentence_ordinals] == list(range(513))


def test_neural_cnn_prediction_does_not_depend_on_padding():
    torch = pytest.importorskip("torch")
    from al_medlit.training.model_types.evidence_neural.config import EvidenceCNNConfig
    from al_medlit.training.model_types.evidence_neural.model import _model_class

    config = EvidenceCNNConfig(
        embedding_dimension=4,
        convolution_channels=4,
        token_kernel_sizes=(2, 3),
        dropout=0,
    )
    model = _model_class(torch, config)(10).eval()
    short = torch.tensor([[[1, 2, 3]]])
    padded = torch.tensor([[[1, 2, 3, 7, 8], [9, 9, 9, 9, 9]]])
    with torch.no_grad():
        expected = model(
            input_ids=short,
            token_mask=torch.ones_like(short, dtype=torch.bool),
            sentence_mask=torch.tensor([[True]]),
        )["logits"][0, 0]
        actual = model(
            input_ids=padded,
            token_mask=torch.tensor([[[True, True, True, False, False], [False] * 5]]),
            sentence_mask=torch.tensor([[True, False]]),
        )["logits"][0, 0]
    assert torch.allclose(actual, expected, atol=1e-6)


def test_ignored_neural_batch_has_finite_zero_loss():
    torch = pytest.importorskip("torch")
    from al_medlit.training.model_types.evidence_neural.model import _output_with_loss

    logits = torch.randn(1, 2, 3, requires_grad=True)
    loss = _output_with_loss(torch, logits, torch.full((1, 2), -100))["loss"]
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_building_neural_bundle_preserves_rng_and_determinism():
    import random

    torch = pytest.importorskip("torch")
    from al_medlit.training.model_types.evidence_neural import (
        EvidenceBiLSTMConfig,
        EvidenceTextDocument,
        build_vocabulary,
    )
    from al_medlit.training.model_types.evidence_neural.model import build_neural_bundle

    config = EvidenceBiLSTMConfig(embedding_dimension=4, hidden_dimension=4)
    vocabulary = build_vocabulary(
        (
            EvidenceTextDocument(
                document_id="one",
                target_id=1,
                target_text="A",
                sentences=("Text",),
            ),
        ),
        config,
    )
    before_python = random.getstate()
    before_torch = torch.random.get_rng_state().clone()
    before_determinism = torch.are_deterministic_algorithms_enabled()
    build_neural_bundle(config, vocabulary)
    assert random.getstate() == before_python
    assert torch.equal(torch.random.get_rng_state(), before_torch)
    assert torch.are_deterministic_algorithms_enabled() == before_determinism


def test_remote_cleanup_retains_logs_then_reclaims_only_expired_terminal_job(
    db,
    object_storage,
    monkeypatch,
):
    from datetime import UTC, datetime, timedelta

    from test_inference_candidate_queries import _inference_scope

    from al_medlit.lineage.models import LineageArtifact
    from al_medlit.training import remote_cleanup
    from al_medlit.training.compute.slurm import SSHSlurmComputeBackend

    scope = _inference_scope(db, run_status="failed")
    scope.run.compute_profile.backend = "ssh_slurm"
    scope.run.completed_at = datetime.now(UTC) - timedelta(days=8)
    db.commit()
    effects = []
    backend = object.__new__(SSHSlurmComputeBackend)

    def logs(*, job_key, output_root):
        (output_root / "scheduler.stderr").write_text("worker failed")
        effects.append("logs")

    monkeypatch.setattr(backend, "collect_scheduler_logs", logs)
    monkeypatch.setattr(backend, "cleanup_job_directory", lambda key: effects.append(key))
    monkeypatch.setattr(backend, "cleanup_expired_base_models", lambda: effects.append("cache"))
    monkeypatch.setattr(remote_cleanup, "build_compute_backend", lambda _: backend)
    assert remote_cleanup.cleanup_remote_inference_jobs(db, object_storage) == 1
    assert effects == ["logs", f"inference-{scope.run.id}", "cache"]
    artifact = db.get(LineageArtifact, scope.run.metrics["scheduler_log_artifact_id"])
    assert b"worker failed" in object_storage.get_bytes(artifact.storage_key)
    assert remote_cleanup.cleanup_remote_inference_jobs(db, object_storage) == 0


def test_remote_cleanup_keeps_uncollected_successful_outputs(db, object_storage, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from test_inference_candidate_queries import _inference_scope

    from al_medlit.training import remote_cleanup
    from al_medlit.training.compute.slurm import SSHSlurmComputeBackend

    scope = _inference_scope(db, run_status="succeeded")
    scope.run.compute_profile.backend = "ssh_slurm"
    scope.run.completed_at = datetime.now(UTC) - timedelta(days=8)
    db.commit()
    backend = object.__new__(SSHSlurmComputeBackend)
    monkeypatch.setattr(backend, "cleanup_job_directory", lambda _: pytest.fail("Live output"))
    monkeypatch.setattr(backend, "cleanup_expired_base_models", lambda: None)
    monkeypatch.setattr(remote_cleanup, "build_compute_backend", lambda _: backend)
    assert remote_cleanup.cleanup_remote_inference_jobs(db, object_storage) == 0


def test_marker_window_counts_match_separate_bpe_encoding():
    from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
        encode_sentence_markers,
    )
    from al_medlit.training.windowing import (
        EvidenceBlockWindowBuilder,
        TargetCondition,
        WindowBuilderConfig,
        WindowSentenceInput,
    )

    class Tokenizer:
        cls_token_id = 1
        sep_token_id = 2

        def convert_tokens_to_ids(self, token):
            return {"[SENT]": 3, "[TARGET]": 4}[token]

        def encode(self, text, *, add_special_tokens=False):
            return list(range(len(text) + (3 if text.startswith(" ") else 0)))

    tokenizer = Tokenizer()
    builder = EvidenceBlockWindowBuilder(
        lambda _: pytest.fail("Already counted with the encoder's tokenizer"),
        WindowBuilderConfig(
            max_tokens=8, overlap_tokens=0, reserved_special_tokens=3, require_reviewed_gold=False
        ),
    )
    result = builder.build(
        document_id=1,
        structure_version_id=1,
        target=TargetCondition(id=1, key="a", name="A", text="A"),
        target_token_count=2,
        sentences=[
            WindowSentenceInput(
                id=1,
                ordinal=0,
                paragraph_ordinal=0,
                section_path=(),
                text="Hi",
                start_char=0,
                end_char=2,
                token_count=3,
            )
        ],
    )
    encoded = encode_sentence_markers(tokenizer, target_text="A", sentences=["Hi"], max_length=8)
    assert result.windows[0].token_count == len(encoded.input_ids) == 8


def test_neural_evaluation_keeps_every_long_document_chunk(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from al_medlit.training import runner
    from al_medlit.training.model_types import evidence_neural

    sentences = [
        {"ordinal": ordinal, "text": f"Sentence {ordinal}", "label": "B" if ordinal == 512 else "O"}
        for ordinal in range(513)
    ]
    row = {"document_id": "long", "target": {"id": 1, "text": "Outcome"}, "sentences": sentences}
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps({**row, "split": split}) for split in ("train", "validation"))
    )
    monkeypatch.setattr(
        evidence_neural,
        "load_neural_bundle",
        lambda _: SimpleNamespace(
            config=SimpleNamespace(max_sentences_per_document=512),
        ),
    )
    observed_lengths = []

    def predict(_bundle, documents, **_kwargs):
        observed_lengths.extend(len(document.sentences) for document in documents)
        assert all(len(document.sentences) <= 512 for document in documents)
        return tuple(SimpleNamespace(labels=document.labels) for document in documents)

    monkeypatch.setattr(evidence_neural, "predict_neural_model", predict)
    report = runner._evaluate_checkpoint_splits(
        tmp_path,
        dataset_path=dataset_path,
        target_version_id=1,
        config={"bootstrap_samples": 0},
        synthetic_mode=False,
        model_type="evidence_cnn",
        splits=("validation",),
    )["validation"]
    assert observed_lengths == [512, 1]
    assert report["metrics"]["macro_exact_block_f1"] == 1
    assert report["report"]["overall"]["sentence_count"] == 513


def test_large_scheduler_logs_are_collected_as_bounded_tails(tmp_path, monkeypatch):
    import base64
    from pathlib import PurePosixPath

    from al_medlit.training.compute.base import CommandResult
    from al_medlit.training.compute.slurm import SSHSlurmComputeBackend

    backend = object.__new__(SSHSlurmComputeBackend)
    monkeypatch.setattr(
        backend, "remote_job_directory", lambda _: PurePosixPath("/jobs/inference-1")
    )
    monkeypatch.setattr(backend, "ssh_argv", lambda command: command)
    monkeypatch.setattr(backend, "remote_file_size", lambda **_: 10 * 1024 * 1024)
    commands = []

    def run(command):
        commands.append(command)
        if command[0] == "test":
            return CommandResult(command, int(command[-1].endswith("stderr")), "", "")
        assert command[:4] == ("bash", "-o", "pipefail", "-c")
        assert "tail -c 4194304" in command[-1]
        return CommandResult(command, 0, base64.b64encode(b"last log line\n").decode(), "")

    monkeypatch.setattr(backend, "_run", run)
    backend.collect_scheduler_logs(job_key="inference-1", output_root=tmp_path)
    payload = (tmp_path / "scheduler.stdout").read_bytes()
    assert payload.endswith(b"last log line\n")
    assert b"truncated" in payload
    assert not (tmp_path / "scheduler.stderr").exists()
