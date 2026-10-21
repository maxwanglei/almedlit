"""Sequence boundaries, generation budgets, and repeated inference regressions."""

import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_evidence_windowing import _sentences, _target
from test_inference_window_limits import _inference_bundle

from al_medlit.inference.decoder import (
    SentenceDecodingInput,
    aggregate_window_logits,
    decode_evidence_blocks,
)
from al_medlit.training import runner
from al_medlit.training.model_types.evidence_conventional import model as conventional
from al_medlit.training.model_types.evidence_conventional.config import EvidenceCRFConfig
from al_medlit.training.model_types.evidence_neural import model as neural
from al_medlit.training.model_types.evidence_neural.config import EvidenceCNNConfig
from al_medlit.training.model_types.evidence_neural.data import EvidenceTextDocument
from al_medlit.training.model_types.evidence_neural.vocabulary import build_vocabulary
from al_medlit.training.model_types.evidence_peft import training as peft
from al_medlit.training.model_types.evidence_peft.config import EvidenceLoRAConfig
from al_medlit.training.windowing import EvidenceBlockWindowBuilder, WindowBuilderConfig


def test_long_paragraph_respects_token_stride():
    builder = EvidenceBlockWindowBuilder(
        lambda _: 1,
        WindowBuilderConfig(
            max_tokens=12, overlap_tokens=2, target_conditioning=False, reserved_special_tokens=0
        ),
    )
    result = builder.build(
        document_id=1,
        structure_version_id=1,
        target=_target(),
        sentences=[
            replace(s, paragraph_ordinal=0, section_path=()) for s in _sentences(100, tokens=1)
        ],
    )
    assert [w.start_sentence_ordinal for w in result.windows] == list(range(0, 90, 10)) + [90]
    assert {s.ordinal for w in result.windows for s in w.sentences} == set(range(100))


@pytest.mark.parametrize("model_type", ["evidence_svm", "evidence_random_forest"])
@pytest.mark.parametrize("overlap", [0, 2])
def test_binary_inference_preserves_block_across_windows(
    tmp_path, monkeypatch, model_type, overlap
):
    path = _inference_bundle(tmp_path, ["Evidence."] * 8)
    job = json.loads(path.read_text())
    job["checkpoint"]["manifest"] = {"model_type": model_type}
    job["window_config"]["overlap_tokens"] = overlap
    path.write_text(json.dumps(job))
    predictor = SimpleNamespace(
        produces_sentence_scores=True, config=SimpleNamespace(sentence_score_threshold=0.5)
    )
    monkeypatch.setattr(conventional, "load_conventional_model", lambda _: predictor)
    monkeypatch.setattr(conventional, "actual_token_count", lambda *_: 1)
    monkeypatch.setattr(
        conventional, "predict_window", lambda _, **kw: [0.9] * len(kw["sentences"])
    )
    runner.run_inference_job(path)
    windows = json.loads((tmp_path / "outputs/inference-result.json").read_text())["windows"]
    assert len(windows) > 1
    assert {int(k) for w in windows for k, v in w["logits"].items() if v == [0, 1, 0]} == {0}
    aggregated = aggregate_window_logits(
        [{int(key): value for key, value in window["logits"].items()} for window in windows],
        score_kind="probabilities",
    )
    decoded = decode_evidence_blocks(
        [
            SentenceDecodingInput(id=i, ordinal=i, start_char=i * 10, end_char=i * 10 + 9)
            for i in range(8)
        ],
        aggregated,
    )
    assert [(block.start_ordinal, block.end_ordinal) for block in decoded.blocks] == [(0, 7)]


def test_crf_inference_matches_training_features_across_windows(tmp_path, monkeypatch):
    seen = []
    config = EvidenceCRFConfig()
    texts = [f"Sentence {i}." for i in range(8)]
    records = [
        conventional.SentenceRecord("d", "t", "evidence", i, text, "O")
        for i, text in enumerate(texts)
    ]
    expected = [conventional._crf_sentence_features(records, i, config) for i in range(8)]
    estimator = SimpleNamespace(
        predict_single=lambda features: seen.append(features) or ["O"] * len(features)
    )
    model = conventional.LoadedConventionalModel("evidence_crf", config, estimator)
    path = _inference_bundle(tmp_path, texts)
    job = json.loads(path.read_text())
    job["checkpoint"]["manifest"] = {"model_type": "evidence_crf"}
    path.write_text(json.dumps(job))
    monkeypatch.setattr(conventional, "load_conventional_model", lambda _: model)
    monkeypatch.setattr(conventional, "actual_token_count", lambda *_: 1)
    runner.run_inference_job(path)
    assert seen == [expected]


class CharacterTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **_kwargs):
        return {"input_ids": list(range(len(text)))}

    def decode(self, *_args, **_kwargs):
        return '{"labels":['


@pytest.mark.parametrize("supervised", [True, False])
@pytest.mark.parametrize("max_new_tokens", [32, 512])
def test_peft_windows_fit_context_and_complete_answer(supervised, max_new_tokens):
    tokenizer = CharacterTokenizer()
    config = EvidenceLoRAConfig(
        base_model_asset_id=7, max_sequence_length=400, max_new_tokens=max_new_tokens
    )
    rows = [
        {
            "document_id": "d",
            "target": {"id": 1, "text": "t"},
            "sentences": [
                {"ordinal": i, "text": "x", "label": "O", "reviewed": True} for i in range(24)
            ],
        }
    ]
    examples = peft.prepare_peft_examples(
        rows, tokenizer, config, supervised=supervised, include_unreviewed=True
    )
    assert [i for e in examples for i in e.sentence_ordinals] == list(range(24))
    for example in examples:
        answer = len(peft.render_evidence_completion(["O"] * len(example.sentences))) + 1
        assert answer <= config.max_new_tokens
        assert len(example.prompt) + answer <= config.max_sequence_length


def test_peft_invalid_output_raises_instead_of_inventing_negative_labels(monkeypatch):
    class Generated:
        def __getitem__(self, _):
            return self

        def detach(self):
            return self

        def to(self, _):
            return self

        def tolist(self):
            return [1]

    model = SimpleNamespace(
        eval=lambda: None,
        parameters=lambda: iter([SimpleNamespace(device="cpu")]),
        generate=lambda **_: Generated(),
    )
    torch = SimpleNamespace(
        no_grad=nullcontext,
        tensor=lambda *a, **kw: object(),
        long="long",
        ones_like=lambda _: object(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    example = peft.PeftEvidenceExample(
        document_id="d",
        target_id="t",
        target_text="t",
        sentence_ordinals=(0,),
        sentences=("text",),
        gold_labels=None,
        prompt="prompt",
    )
    with pytest.raises(ValueError, match="(?i)(invalid|malformed).*output"):
        peft.predict_peft_examples(
            SimpleNamespace(model=model),
            CharacterTokenizer(),
            (example,),
            config=EvidenceLoRAConfig(base_model_asset_id=7),
        )


def test_vocabulary_lookup_allocated_once_and_immutable():
    documents = (
        EvidenceTextDocument(
            document_id="d", target_id=1, target_text="target", sentences=("alpha beta",)
        ),
    )
    vocabulary = build_vocabulary(documents, EvidenceCNNConfig())
    mappings = [vocabulary.token_to_id() for _ in range(5)]
    assert len({id(mapping) for mapping in mappings}) == 1
    with pytest.raises(TypeError):
        mappings[0]["alpha"] = 123
    assert vocabulary.encode_sentence(
        target_text="target", sentence="alpha beta", max_tokens=10
    ) == (2, 6, 3, 4, 5)


def test_repeated_neural_prediction_moves_model_once(monkeypatch):
    moves = []

    class Device(str):
        @property
        def type(self):
            return str(self)

        index = None

    parameter = SimpleNamespace(device=Device("cpu"))

    class Scores:
        def detach(self):
            return self

        def to(self, _):
            return self

        def __getitem__(self, _):
            return SimpleNamespace(tolist=lambda: [0.1, 0.8, 0.1])

    class Model:
        def parameters(self):
            return iter([parameter])

        def to(self, device):
            moves.append(device)
            parameter.device = device

        def eval(self):
            pass

        def __call__(self, **_):
            return {"logits": Scores()}

    torch = SimpleNamespace(device=Device, no_grad=nullcontext, softmax=lambda values, **_: values)
    monkeypatch.setattr(neural, "_import_torch", lambda: torch)
    monkeypatch.setattr(
        neural, "require_neural_device", lambda _: SimpleNamespace(torch_device="cuda")
    )
    monkeypatch.setattr(neural, "_collate", lambda *a, **kw: {})
    documents = (
        EvidenceTextDocument(
            document_id="d", target_id=1, target_text="target", sentences=("alpha",)
        ),
    )
    config = EvidenceCNNConfig()
    bundle = neural.EvidenceNeuralBundle(Model(), build_vocabulary(documents, config), config)
    results = [neural.predict_neural_model(bundle, documents, device="cuda") for _ in range(3)]
    assert [result[0].labels for result in results] == [("B",)] * 3
    assert moves == ["cuda"]


def test_crf_evaluation_keeps_absolute_ordinals_and_review_gaps(tmp_path, monkeypatch):
    seen = []
    config = EvidenceCRFConfig()
    records = [
        conventional.SentenceRecord("d", "1", "target", ordinal, f"Sentence {ordinal}.", "O")
        for ordinal in (12, 13, 16)
    ]
    expected = [
        [
            conventional._crf_sentence_features(sequence, index, config)
            for index in range(len(sequence))
        ]
        for sequence in (records[:2], records[2:])
    ]
    estimator = SimpleNamespace(
        predict_single=lambda features: seen.append(features) or ["O"] * len(features)
    )
    model = conventional.LoadedConventionalModel("evidence_crf", config, estimator)
    monkeypatch.setattr(conventional, "load_conventional_model", lambda _: model)
    rows = [
        {
            "document_id": "d",
            "split": "validation",
            "target": {"id": 1, "text": "target"},
            "sentences": [
                {"ordinal": item.ordinal, "text": item.text, "label": "O"} for item in sequence
            ],
        }
        for sequence in (records[:2], records[1:])
    ]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps(row)
            for row in [{**rows[0], "document_id": "training", "split": "train"}, *rows]
        )
    )
    report = runner._evaluate_checkpoint_splits(
        tmp_path,
        dataset_path=dataset,
        target_version_id=None,
        config=config.model_dump(),
        synthetic_mode=False,
        model_type="evidence_crf",
        splits=("validation",),
    )
    assert "validation" in report
    assert seen == expected


def test_peft_generation_caps_context_and_accepts_valid_all_o(monkeypatch):
    calls = []

    class Generated:
        def __getitem__(self, _):
            return self

        def detach(self):
            return self

        def to(self, _):
            return self

        def tolist(self):
            return [1]

    def generate(**kwargs):
        calls.append(kwargs["max_new_tokens"])
        return Generated()

    model = SimpleNamespace(
        eval=lambda: None,
        parameters=lambda: iter([SimpleNamespace(device="cpu")]),
        generate=generate,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            no_grad=nullcontext,
            tensor=lambda *a, **kw: object(),
            long="long",
            ones_like=lambda _: object(),
        ),
    )
    tokenizer = CharacterTokenizer()
    tokenizer.decode = lambda *a, **kw: '{"labels":["O"]}'
    example = peft.PeftEvidenceExample(
        document_id="d",
        target_id="t",
        target_text="t",
        sentence_ordinals=(0,),
        sentences=("text",),
        gold_labels=None,
        prompt="x" * 100,
    )
    config = EvidenceLoRAConfig(base_model_asset_id=7, max_sequence_length=128, max_new_tokens=512)
    assert peft.predict_peft_examples(
        SimpleNamespace(model=model), tokenizer, (example,), config=config
    ) == ((("O",),), 1)
    assert calls == [28]
    with pytest.raises(ValueError, match="insufficient context or JSON answer budget"):
        peft.predict_peft_examples(
            SimpleNamespace(model=model),
            tokenizer,
            (replace(example, prompt="x" * 125),),
            config=config,
        )
    assert calls == [28]
