from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from al_medlit.training.model_types.evidence_neural import (
    EvidenceBiLSTMConfig,
    EvidenceBiLSTMPlugin,
    EvidenceCNNConfig,
    EvidenceCNNPlugin,
    EvidenceTextDocument,
    EvidenceVocabulary,
    NeuralCheckpointScore,
    build_vocabulary,
    canonical_checkpoint_is_better,
    documents_from_window_rows,
    fit_neural_model,
    load_neural_bundle,
    predict_neural_model,
    register_builtin_neural_model_types,
)
from al_medlit.training.model_types.evidence_neural.device import (
    preflight_neural_device,
)
from al_medlit.training.registry import ModelTypeRegistry


def _documents():
    return (
        EvidenceTextDocument(
            document_id="doc-1",
            target_id=10,
            target_text="Blood pressure",
            sentences=("The pressure was high.", "Treatment reduced pressure."),
            labels=("B", "I"),
        ),
        EvidenceTextDocument(
            document_id="doc-2",
            target_id=10,
            target_text="Blood pressure",
            sentences=("No result was reported.", "Pressure remained stable."),
            labels=("O", "B"),
        ),
    )


def test_neural_configs_persist_deterministic_training_and_tokenizer_options():
    bilstm = EvidenceBiLSTMConfig()
    assert bilstm.seed == 42
    assert bilstm.deterministic_algorithms is True
    assert bilstm.gradient_accumulation_steps == 1
    assert bilstm.gradient_clip_norm == 1.0
    assert bilstm.reload_best_checkpoint is True
    assert bilstm.tokenizer_pattern

    with pytest.raises(ValidationError, match="must be unique"):
        EvidenceCNNConfig(token_kernel_sizes=(2, 2))
    with pytest.raises(ValidationError, match="must be odd"):
        EvidenceCNNConfig(sentence_context_kernel_size=2)


def test_vocabulary_is_deterministic_actual_preparation_and_safe_json(tmp_path):
    config = EvidenceBiLSTMConfig(
        lowercase=True,
        max_vocabulary_size=16,
        max_tokens_per_sentence=12,
    )
    first = build_vocabulary(_documents(), config)
    second = build_vocabulary(reversed(_documents()), config)
    assert first.tokens == second.tokens
    assert first.tokens[:4] == ("<PAD>", "<UNK>", "<TARGET>", "<SEP>")
    encoded = first.encode_sentence(
        target_text="Blood pressure",
        sentence="Pressure is high",
        max_tokens=config.max_tokens_per_sentence,
    )
    assert encoded[0] == first.token_to_id()["<TARGET>"]
    assert first.token_to_id()["<SEP>"] in encoded

    path = first.save(tmp_path / "vocabulary.json")
    loaded = EvidenceVocabulary.load(path)
    assert loaded.payload == first.payload


def test_prepared_overlapping_windows_collapse_without_duplicate_sentences():
    rows = [
        {
            "document_id": "doc-1",
            "target": {"id": 10, "text": "Outcome"},
            "sentences": [
                {"ordinal": 0, "text": "First", "label": "B", "reviewed": True},
                {"ordinal": 1, "text": "Second", "label": "I", "reviewed": True},
            ],
        },
        {
            "document_id": "doc-1",
            "target": {"id": 10, "text": "Outcome"},
            "sentences": [
                {"ordinal": 1, "text": "Second", "label": "I", "reviewed": True},
                {"ordinal": 2, "text": "Third", "label": "O", "reviewed": False},
            ],
        },
    ]
    documents = documents_from_window_rows(rows)
    assert len(documents) == 1
    assert documents[0].sentences == ("First", "Second", "Third")
    assert documents[0].sentence_ordinals == (0, 1, 2)
    assert documents[0].labels == ("B", "I", "IGNORE")


def test_device_preflight_resolves_cpu_and_fails_closed_for_missing_ascend():
    fake_torch = SimpleNamespace(
        __version__="2.5.0",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        version=SimpleNamespace(cuda=None),
    )

    def loader(name):
        if name == "torch":
            return fake_torch
        raise ModuleNotFoundError(name)

    automatic = preflight_neural_device("auto", module_loader=loader)
    assert automatic.available is True
    assert automatic.resolved_device == "cpu"
    ascend = preflight_neural_device("ascend", module_loader=loader)
    assert ascend.available is False
    assert ascend.missing_dependencies == ("torch_npu",)


def test_canonical_neural_checkpoint_selection_uses_exact_block_tiebreaker():
    incumbent = NeuralCheckpointScore(
        macro_block_iou_f1_0_50=0.7,
        macro_exact_block_f1=0.4,
    )
    assert canonical_checkpoint_is_better(
        NeuralCheckpointScore(
            macro_block_iou_f1_0_50=0.71,
            macro_exact_block_f1=0.1,
        ),
        incumbent,
    )
    assert canonical_checkpoint_is_better(
        NeuralCheckpointScore(
            macro_block_iou_f1_0_50=0.7,
            macro_exact_block_f1=0.5,
        ),
        incumbent,
    )
    assert not canonical_checkpoint_is_better(incumbent, incumbent)


def test_neural_plugins_register_for_durable_runner_dispatch():
    registry = ModelTypeRegistry()
    registry.register_builtin_descriptors()
    register_builtin_neural_model_types(registry)

    assert isinstance(registry.get("evidence_bilstm"), EvidenceBiLSTMPlugin)
    assert isinstance(registry.get("evidence_cnn"), EvidenceCNNPlugin)
    assert registry.get_descriptor("evidence_bilstm").implementation_status == "implemented"


@pytest.mark.parametrize(
    "config",
    [
        EvidenceBiLSTMConfig(
            embedding_dimension=8,
            hidden_dimension=8,
            epochs=2,
            batch_size=2,
            dropout=0,
            device="cpu",
            max_vocabulary_size=64,
            max_tokens_per_sentence=16,
        ),
        EvidenceCNNConfig(
            embedding_dimension=8,
            convolution_channels=8,
            token_kernel_sizes=(2, 3),
            epochs=2,
            batch_size=2,
            dropout=0,
            device="cpu",
            max_vocabulary_size=64,
            max_tokens_per_sentence=16,
        ),
    ],
)
def test_optional_torch_fit_predict_and_safetensors_round_trip(config, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    bundle, summary = fit_neural_model(
        config,
        _documents(),
        validation_documents=_documents(),
        validation_selector=lambda _bundle, epoch: NeuralCheckpointScore(
            macro_block_iou_f1_0_50=0.5,
            macro_exact_block_f1=epoch / 10,
        ),
    )
    assert summary.best_epoch == 2
    assert summary.selection_metric == "macro_block_iou_f1_0_50"
    assert summary.selection_tiebreaker_exact_block_f1 == 0.2
    predictions = predict_neural_model(bundle, _documents(), device="cpu")
    assert len(predictions) == 2
    assert all(len(item.labels) == 2 for item in predictions)

    checkpoint = tmp_path / config.model_kind
    manifest = bundle.save_pretrained(checkpoint)
    assert (checkpoint / "model.safetensors").is_file()
    assert not tuple(checkpoint.glob("*.pt"))
    assert not tuple(checkpoint.glob("*.pkl"))
    assert manifest["safe_serialization"] is True
    reloaded = load_neural_bundle(checkpoint)
    loaded_predictions = predict_neural_model(reloaded, _documents(), device="cpu")
    assert [item.labels for item in loaded_predictions] == [item.labels for item in predictions]
