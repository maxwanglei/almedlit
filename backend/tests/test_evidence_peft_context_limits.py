"""PEFT context must fit both immutable base metadata and the recipe budget."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from al_medlit.training.model_types.evidence_peft import training
from al_medlit.training.model_types.evidence_peft.config import EvidenceLoRAConfig


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, limit=256):
        self.model_max_length = limit

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text)))}

    def decode(self, *args, **kwargs):
        return '{"labels":["O"]}'


def config():
    return EvidenceLoRAConfig(base_model_asset_id=1, max_sequence_length=1024)


def test_preparation_rejects_sentence_exceeding_tokenizer_context():
    rows = [
        {
            "document_id": "d",
            "target": {"id": 1, "text": "target"},
            "sentences": [{"ordinal": 0, "text": "x" * 300, "label": "O"}],
        }
    ]
    with pytest.raises(ValueError, match="context"):
        training.prepare_peft_examples(rows, Tokenizer(), config())


def test_supervised_encoding_enforces_same_tokenizer_context():
    example = training.PeftEvidenceExample(
        "d", "t", "target", (0,), ("sentence",), ("O",), "x" * 250
    )
    with pytest.raises(ValueError, match="context"):
        training._encode_supervised_example(Tokenizer(), example, config())


@pytest.mark.parametrize("position_field", ["max_position_embeddings", "n_positions"])
@pytest.mark.parametrize("tokenizer_limit,expected", [(10**30, 256), (128, 128), (512, 256)])
def test_loaded_model_propagates_actual_context_to_preparation(
    monkeypatch,
    tmp_path,
    position_field,
    tokenizer_limit,
    expected,
):
    tokenizer = Tokenizer(tokenizer_limit)
    model = SimpleNamespace(config=SimpleNamespace(**{position_field: 256}), to=lambda device: None)
    monkeypatch.setattr(training, "require_peft_preflight", lambda config: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(device=lambda device: device))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model),
        ),
    )
    _, loaded, _, _ = training._load_base_model(config(), tmp_path)
    assert loaded.model_max_length == expected


@pytest.mark.parametrize(
    "tokenizer_limit,model_limit,expected",
    [
        (256, 512, 56),
        (10**30, 256, 56),
        (512, 256, 56),
    ],
)
def test_generation_uses_remaining_actual_context(
    monkeypatch,
    tokenizer_limit,
    model_limit,
    expected,
):
    calls = []

    class Generated:
        def __getitem__(self, index):
            return self

        def detach(self):
            return self

        def to(self, device):
            return self

        def tolist(self):
            return [1]

    def generate(**kwargs):
        calls.append(kwargs["max_new_tokens"])
        return Generated()

    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=model_limit),
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
            ones_like=lambda tensor: object(),
        ),
    )
    example = training.PeftEvidenceExample("d", "t", "target", (0,), ("sentence",), None, "x" * 200)
    result = training.predict_peft_examples(
        SimpleNamespace(model=model), Tokenizer(tokenizer_limit), [example], config=config()
    )
    assert result == ((("O",),), 1)
    assert calls == [expected]


def test_unspecified_tokenizer_limit_does_not_replace_recipe_limit():
    tokenizer = Tokenizer(10**30)
    rows = [
        {
            "document_id": "d",
            "target": {"id": 1, "text": "target"},
            "sentences": [{"ordinal": 0, "text": "x" * 300, "label": "O"}],
        }
    ]
    assert len(training.prepare_peft_examples(rows, tokenizer, config())) == 1
