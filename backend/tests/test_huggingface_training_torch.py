"""Numerical checks for workers with the optional Torch/HF runtime installed."""

from types import SimpleNamespace

import pytest

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.trainers.huggingface_common import execute_torch_training
from al_medlit.training.trainers.huggingface_sft import HuggingFaceCausalSftTrainer
from al_medlit.training.trainers.huggingface_tasks import HuggingFaceTokenTrainer


@pytest.mark.parametrize("precision", ["none", "fp16", "bf16"])
def test_real_torch_fractional_steps_keep_fp32_optimizer_state(monkeypatch, precision):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    if getattr(torch, "npu", None):
        monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    optimizers = []
    adamw = torch.optim.AdamW

    def capture(parameters, **kwargs):
        optimizer = adamw(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", capture)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            prediction = x @ self.weight
            return SimpleNamespace(loss=(prediction.float() - 0.75).square().mean())

    model = Model()
    execute_torch_training(
        torch=torch,
        model=model,
        train_rows=[{"x": 1.0}] * 4,
        validation_rows=[],
        collator=lambda rows: {"x": torch.tensor([[row["x"]] for row in rows])},
        config={
            "epochs": 1.25,
            "batch_size": 1,
            "learning_rate": 0.01,
            "gradient_accumulation_steps": 2,
            "mixed_precision": precision,
        },
        seed=42,
    )
    assert model.calls == 5
    assert model.weight.dtype == torch.float32
    assert torch.isfinite(model.weight).all()
    assert model.weight.item() > 0.5
    state = optimizers[0].state[model.weight]
    assert state["step"].item() == 3
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_real_fast_tokenizer_preserves_whole_supervision_or_rejects():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordPiece(
            vocab={
                "[UNK]": 0,
                "[CLS]": 1,
                "[SEP]": 2,
                "the": 3,
                "asp": 4,
                "##irin": 5,
                "helps": 6,
                "question": 7,
                "answer": 8,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[SEP]"
    )
    plan = SimpleNamespace(
        manifest={
            "task": {"row_mapping": {"input": {"field": "tokens"}, "target": {"field": "tags"}}},
            "output_contract": {"label_to_id": {"O": 0, "B-drug": 1}},
        }
    )
    source = [{"tokens": ["the", "aspirin", "helps"], "tags": ["O", "B-drug", "O"]}]
    rows = HuggingFaceTokenTrainer._encode_rows(tokenizer, source, {"max_sequence_length": 6}, plan)
    assert rows[0]["input_ids"] == [1, 3, 4, 5, 6, 2]
    assert rows[0]["labels"] == [-100, 0, 1, -100, 0, -100]
    with pytest.raises(ValidationError, match="length|truncat"):
        HuggingFaceTokenTrainer._encode_rows(tokenizer, source, {"max_sequence_length": 4}, plan)
    with pytest.raises(ValidationError, match="length|truncat"):
        HuggingFaceCausalSftTrainer._causal_rows(tokenizer, [("question", "answer " * 5)], 4)


def test_real_fp16_scaler_skips_overflow_then_recovers(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    if getattr(torch, "npu", None):
        monkeypatch.setattr(torch.npu, "is_available", lambda: False)
    optimizers = []
    adamw = torch.optim.AdamW

    def capture(parameters, **kwargs):
        optimizer = adamw(parameters, **kwargs)
        optimizers.append(optimizer)
        return optimizer

    monkeypatch.setattr(torch.optim, "AdamW", capture)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32))
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            prediction = x @ self.weight
            factor = float("inf") if self.calls == 1 else 1.0
            return SimpleNamespace(loss=(prediction.float() - 0.75).square().mean() * factor)

    model = Model()
    execute_torch_training(
        torch=torch,
        model=model,
        train_rows=[{"x": 1.0}] * 4,
        validation_rows=[],
        collator=lambda rows: {"x": torch.tensor([[row["x"]] for row in rows])},
        config={
            "epochs": 0.5,
            "batch_size": 1,
            "learning_rate": 0.01,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "fp16",
        },
        seed=42,
    )
    assert model.calls == 2
    assert torch.isfinite(model.weight).all()
    assert model.weight.item() > 0.5
    assert optimizers[0].state[model.weight]["step"].item() == 1


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_real_full_sft_loads_low_precision_checkpoint_as_float32(tmp_path, precision):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(vocab_size=16, n_positions=16, n_embd=8, n_layer=1, n_head=2)
    )
    model.to(torch.bfloat16).save_pretrained(tmp_path, safe_serialization=True)
    loaded, dispatched = HuggingFaceCausalSftTrainer()._load_model(
        torch=torch,
        transformers=transformers,
        base_model_path=tmp_path,
        plan=SimpleNamespace(
            normalized_config={"mixed_precision": precision},
            manifest={"output_contract": {"parameterization": "full"}},
        ),
    )
    assert dispatched is False
    assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())


@pytest.mark.parametrize("task", ["sequence", "token", "span", "causal", "seq2seq"])
def test_real_torch_model_initialization_uses_execution_seed(task, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from al_medlit.training.trainers.huggingface_sft import HuggingFaceSeq2SeqSftTrainer
    from al_medlit.training.trainers.huggingface_tasks import (
        HuggingFaceSequenceTrainer,
        HuggingFaceSpanTrainer,
    )

    trainers = {
        "sequence": (HuggingFaceSequenceTrainer(), transformers.AutoModelForSequenceClassification),
        "token": (HuggingFaceTokenTrainer(), transformers.AutoModelForTokenClassification),
        "span": (HuggingFaceSpanTrainer(), transformers.AutoModelForQuestionAnswering),
        "causal": (HuggingFaceCausalSftTrainer(), transformers.AutoModelForCausalLM),
        "seq2seq": (HuggingFaceSeq2SeqSftTrainer(), transformers.AutoModelForSeq2SeqLM),
    }
    trainer, factory = trainers[task]
    weights = []

    class Loaded(Exception):
        pass

    def initialize(*args, **kwargs):
        weights.append(torch.nn.Linear(4, 2).weight.detach().clone())
        raise Loaded

    monkeypatch.setattr(factory, "from_pretrained", initialize)
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace()
    )
    plan = SimpleNamespace(
        normalized_config={"mixed_precision": "fp16"},
        manifest={
            "output_contract": {
                "labels": ["a", "b"],
                "label_to_id": {"a": 0, "b": 1},
                "type": "single_label_classification",
                "parameterization": "full",
            }
        },
    )
    for previous_seed in [1, 2]:
        torch.manual_seed(previous_seed)
        with pytest.raises(Loaded):
            trainer._execute(
                plan=plan,
                training_input=None,
                base_model_path=tmp_path,
                destination=tmp_path,
                seed=42,
            )
    assert torch.equal(weights[0], weights[1])
