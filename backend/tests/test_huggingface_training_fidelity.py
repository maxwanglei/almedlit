"""Trainer boundary regressions runnable without optional worker dependencies."""

import random
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.trainers.huggingface_common import execute_torch_training
from al_medlit.training.trainers.huggingface_sft import (
    HuggingFaceCausalSftTrainer,
    HuggingFaceSeq2SeqSftTrainer,
)
from al_medlit.training.trainers.huggingface_tasks import (
    HuggingFaceSequenceTrainer,
    HuggingFaceSpanTrainer,
    HuggingFaceTokenTrainer,
)


class Encoding(dict):
    def __init__(self, ids, words=None):
        super().__init__(input_ids=ids, attention_mask=[1] * len(ids))
        self.words = words

    def word_ids(self):
        return self.words


class Tokenizer:
    eos_token_id = 2

    def __call__(self, text=None, *, text_target=None, **kwargs):
        text = text if text_target is None else text_target
        words = None
        if isinstance(text, list):
            ids = [1] + [10 + i for i in range(len(text))] + [2]
            words = [None] + list(range(len(text))) + [None]
        else:
            ids = list(range(10, 10 + len(text.split())))
            if kwargs.get("add_special_tokens", True):
                ids = [1, *ids]
        if kwargs.get("truncation"):
            ids = ids[: kwargs["max_length"]]
            words = words[: len(ids)] if words else None
        return Encoding(ids, words)


def token_plan():
    return SimpleNamespace(
        manifest={
            "task": {"row_mapping": {"input": {"field": "tokens"}, "target": {"field": "tags"}}},
            "output_contract": {"label_to_id": {"O": 0, "B-drug": 1}},
        }
    )


def test_ner_rejects_lost_supervised_words():
    with pytest.raises(ValidationError, match="row 0.*(length|truncat)"):
        HuggingFaceTokenTrainer._encode_rows(
            Tokenizer(),
            [{"tokens": ["the", "drug", "aspirin"], "tags": ["O", "O", "B-drug"]}],
            {"max_sequence_length": 3},
            token_plan(),
        )


def test_ner_preserves_labels_when_every_word_fits():
    rows = HuggingFaceTokenTrainer._encode_rows(
        Tokenizer(),
        [{"tokens": ["the", "aspirin"], "tags": ["O", "B-drug"]}],
        {"max_sequence_length": 4},
        token_plan(),
    )
    assert rows[0]["labels"] == [-100, 0, 1, -100]


@pytest.mark.parametrize(
    "prompt,completion", [("question", "one two three four"), ("one two three four", "answer")]
)
def test_causal_sft_rejects_loss_of_prompt_or_completion(prompt, completion):
    with pytest.raises(ValidationError, match="row 0.*(length|truncat)"):
        HuggingFaceCausalSftTrainer._causal_rows(Tokenizer(), [(prompt, completion)], 4)


def test_causal_sft_preserves_prompt_and_targets_at_limit():
    (row,) = HuggingFaceCausalSftTrainer._causal_rows(Tokenizer(), [("question", "answer")], 4)
    assert row["input_ids"] == [1, 10, 10, 2]
    assert row["labels"] == [-100, -100, 10, 2]


@pytest.mark.parametrize(
    "prompt,completion", [("question", "one two three four"), ("one two three four", "answer")]
)
def test_seq2seq_rejects_loss_of_prompt_or_targets(prompt, completion):
    with pytest.raises(ValidationError, match="row 0.*(length|truncat)"):
        HuggingFaceSeq2SeqSftTrainer._seq2seq_rows(Tokenizer(), [(prompt, completion)], 3)


@pytest.mark.parametrize(
    "trainer",
    [
        HuggingFaceSequenceTrainer(),
        HuggingFaceTokenTrainer(),
        HuggingFaceSpanTrainer(),
        HuggingFaceCausalSftTrainer(),
        HuggingFaceSeq2SeqSftTrainer(),
    ],
)
def test_seed_applies_before_loading_tokenizer_or_model(trainer, monkeypatch, tmp_path):
    class Loaded(Exception):
        pass

    samples = []

    def load(*args, **kwargs):
        samples.append(random.random())
        raise Loaded

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(set_seed=random.seed, AutoTokenizer=SimpleNamespace(from_pretrained=load)),
    )
    plan = SimpleNamespace(
        normalized_config={},
        manifest={"output_contract": {"labels": ["a"], "label_to_id": {"a": 0}}},
    )
    for previous_seed in [1, 2]:
        random.seed(previous_seed)
        with pytest.raises(Loaded):
            trainer._execute(
                plan=plan,
                training_input=None,
                base_model_path=tmp_path,
                destination=tmp_path,
                seed=42,
            )
    assert samples[0] == samples[1]


@pytest.mark.parametrize("trainer", [HuggingFaceCausalSftTrainer(), HuggingFaceSeq2SeqSftTrainer()])
@pytest.mark.parametrize("precision", ["none", "fp16", "bf16"])
def test_full_sft_loads_optimizer_weights_in_float32(trainer, precision, tmp_path):
    captured = {}

    def load(path, **kwargs):
        captured.update(kwargs)
        return object()

    factory = SimpleNamespace(from_pretrained=load)
    trainer._load_model(
        torch=SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32"),
        transformers=SimpleNamespace(AutoModelForCausalLM=factory, AutoModelForSeq2SeqLM=factory),
        base_model_path=tmp_path,
        plan=SimpleNamespace(
            normalized_config={"mixed_precision": precision},
            manifest={"output_contract": {"parameterization": "full"}},
        ),
    )
    assert captured["torch_dtype"] == "fp32"


class Loss(float):
    def __new__(cls, value, events, scaled=False):
        instance = super().__new__(cls, value)
        instance.events = events
        instance.scaled = scaled
        return instance

    def __truediv__(self, divisor):
        return Loss(float(self) / divisor, self.events, self.scaled)

    def backward(self):
        self.events.append(("backward", float(self), self.scaled))

    def detach(self):
        return self

    def cpu(self):
        return self


def run_loop(*, epochs, accumulation=1, precision="none"):
    events = []

    class Loader:
        def __init__(self, dataset, *, batch_size, **kwargs):
            self.rows = [dataset[i] for i in range(len(dataset))]

        def __len__(self):
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    class Optimizer:
        def __init__(self, *args, **kwargs):
            pass

        def step(self):
            events.append(("optimizer_step",))

        def zero_grad(self, **kwargs):
            pass

    class Scaler:
        def __init__(self, device, **kwargs):
            pass

        def scale(self, loss):
            return Loss(loss, events, scaled=True)

        def step(self, optimizer):
            events.append(("scaler_step",))
            optimizer.step()

        def update(self):
            events.append(("scaler_update",))

    class Model:
        def to(self, device):
            pass

        def parameters(self):
            return iter([SimpleNamespace(requires_grad=True)])

        def train(self):
            pass

        def __call__(self, **batch):
            events.append(("forward",))
            return SimpleNamespace(loss=Loss(1.0, events))

    torch = SimpleNamespace(
        manual_seed=lambda seed: None,
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(),
        device=lambda kind: SimpleNamespace(type=kind),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: None),
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=Loader)),
        optim=SimpleNamespace(AdamW=Optimizer),
        float16="fp16",
        bfloat16="bf16",
        autocast=lambda **kwargs: nullcontext(),
        amp=SimpleNamespace(GradScaler=Scaler),
    )
    execute_torch_training(
        torch=torch,
        model=Model(),
        train_rows=[{}] * 4,
        validation_rows=[],
        collator=lambda rows: rows,
        config={
            "batch_size": 1,
            "learning_rate": 0.01,
            "epochs": epochs,
            "gradient_accumulation_steps": accumulation,
            "mixed_precision": precision,
        },
        seed=42,
    )
    return events


@pytest.mark.parametrize(
    "epochs,batches,steps", [(0.25, 1, 1), (0.5, 2, 1), (1.25, 5, 3), (1.5, 6, 3), (2.0, 8, 4)]
)
def test_fractional_epochs_use_the_requested_batch_budget(epochs, batches, steps):
    events = run_loop(epochs=epochs, accumulation=2)
    assert sum(event[0] == "forward" for event in events) == batches
    assert sum(event[0] == "optimizer_step" for event in events) == steps


def test_partial_accumulation_uses_actual_group_size():
    events = run_loop(epochs=1.25, accumulation=2)
    assert [event[1] for event in events if event[0] == "backward"] == [0.5] * 4 + [1.0]


def test_fp16_scales_backward_and_updates_once_per_optimizer_step():
    events = run_loop(epochs=1.0, accumulation=2, precision="fp16")
    assert all(event[2] for event in events if event[0] == "backward")
    assert [event[0] for event in events if "step" in event[0] or "update" in event[0]] == [
        "scaler_step",
        "optimizer_step",
        "scaler_update",
        "scaler_step",
        "optimizer_step",
        "scaler_update",
    ]


@pytest.mark.parametrize("trainer", [HuggingFaceCausalSftTrainer(), HuggingFaceSeq2SeqSftTrainer()])
@pytest.mark.parametrize("parameterization", ["lora", "qlora"])
@pytest.mark.parametrize(
    "precision,expected_dtype", [("fp16", "fp16"), ("bf16", "bf16"), ("none", "fp32")]
)
def test_adapters_keep_fp32_trainables_and_frozen_quantized_storage(
    trainer,
    parameterization,
    precision,
    expected_dtype,
    monkeypatch,
    tmp_path,
):
    class Tensor:
        def __init__(self, dtype):
            self.dtype = dtype

        def float(self):
            return Tensor("fp32")

    frozen = SimpleNamespace(requires_grad=False, data=Tensor("int4"))
    adapter = SimpleNamespace(requires_grad=True, data=Tensor("fp16"))
    model = SimpleNamespace(parameters=lambda: iter([frozen, adapter]))
    captured = {}

    def load(path, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            prepare_model_for_kbit_training=lambda model: model,
            TaskType=SimpleNamespace(CAUSAL_LM="causal", SEQ_2_SEQ_LM="seq2seq"),
            LoraConfig=lambda **kwargs: kwargs,
            get_peft_model=lambda model, config: model,
        ),
    )
    factory = SimpleNamespace(from_pretrained=load)
    trainer._load_model(
        torch=SimpleNamespace(
            float16="fp16",
            bfloat16="bf16",
            float32="fp32",
            cuda=SimpleNamespace(current_device=lambda: 0),
        ),
        transformers=SimpleNamespace(
            AutoModelForCausalLM=factory,
            AutoModelForSeq2SeqLM=factory,
            BitsAndBytesConfig=lambda **kwargs: kwargs,
        ),
        base_model_path=tmp_path,
        plan=SimpleNamespace(
            normalized_config={
                "mixed_precision": precision,
                "lora_rank": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": ["q_proj"],
            },
            manifest={"output_contract": {"parameterization": parameterization}},
        ),
    )
    assert adapter.data.dtype == "fp32"
    assert frozen.data.dtype == "int4"
    assert captured["torch_dtype"] == expected_dtype
    if parameterization == "qlora":
        assert captured["quantization_config"]["bnb_4bit_compute_dtype"] == expected_dtype
