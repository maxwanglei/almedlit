import importlib.util
import json
import sys

import pytest

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.trainers import (
    HuggingFaceCausalSftTrainer,
    HuggingFaceSeq2SeqSftTrainer,
    HuggingFaceSequenceTrainer,
    HuggingFaceSpanTrainer,
    HuggingFaceTokenTrainer,
    SklearnTfidfTrainer,
    trainer_plugins,
)
from al_medlit.training.trainers.contracts import (
    TrainerPlugin,
    TrainerPluginRegistry,
    TrainingInput,
)
from al_medlit.training.trainers.huggingface_common import artifact_inventory


def test_classical_trainer_is_registered_without_importing_optional_packages():
    plugin = trainer_plugins.require_recipe(
        "sklearn_tfidf",
        "tfidf_linear_regression",
    )
    report = plugin.preflight()

    assert plugin.key == "sklearn_tfidf"
    assert report.runtime_class == "classical-cpu"
    assert len(report.checks) == 2


def test_trainer_registry_rejects_recipe_mismatch():
    registry = TrainerPluginRegistry()
    registry.register(SklearnTfidfTrainer())

    with pytest.raises(ValidationError, match="does not support"):
        registry.require_recipe("sklearn_tfidf", "causal_lm_sft_lora")


def test_classical_trainer_fails_closed_when_runtime_is_missing(tmp_path):
    plugin = SklearnTfidfTrainer()
    report = plugin.preflight()
    if report.ready:
        pytest.skip("Optional classical runtime is installed")

    with pytest.raises(ValidationError, match="preflight failed"):
        plugin.train(
            recipe_key="tfidf_linear_regression",
            config={"fields": {"input_field": "text", "target_field": "score"}},
            training_input=TrainingInput(
                rows=({"text": "example", "score": 1.0},),
                dataset_fingerprint="dataset-1",
                split_fingerprint="split-1",
            ),
            destination=tmp_path / "output",
            seed=42,
        )


def _multilabel_training_input() -> TrainingInput:
    return TrainingInput(
        rows=(
            {"text": "Improved outcomes", "labels": ["efficacy"]},
            {"text": "Adverse reaction", "labels": ["safety"]},
            {"text": "Effective and affordable", "labels": ["efficacy", "cost"]},
            {"text": "Safe and affordable", "labels": ["safety", "cost"]},
        ),
        validation_rows=(
            {"text": "Effective with adverse effects", "labels": ["efficacy", "safety"]},
            {"text": "No coded finding", "labels": []},
        ),
        dataset_fingerprint="multilabel-dataset",
        split_fingerprint="multilabel-splits",
        task_kind="multilabel_classification",
        task_schema={"output": {"type": "array", "items": "label"}},
        label_vocabulary=("safety", "efficacy", "cost"),
    )


def _multilabel_config() -> dict:
    return {
        "fields": {"input_field": "text", "target_field": "labels"},
        "max_features": 100,
        "min_document_frequency": 1,
        "max_iterations": 100,
    }


def test_tfidf_logistic_plan_pins_multilabel_encoding_without_optional_imports():
    trainer = SklearnTfidfTrainer()

    plan = trainer.plan(
        recipe_key="tfidf_logistic_regression",
        config=_multilabel_config(),
        training_input=_multilabel_training_input(),
        seed=19,
    )

    assert plan.manifest["task"] == {
        "kind": "multilabel_classification",
        "schema": {"output": {"type": "array", "items": "label"}},
    }
    assert plan.manifest["output_contract"] == {
        "type": "multilabel_classification",
        "encoding": "binary_indicator",
        "labels": ["safety", "efficacy", "cost"],
        "label_to_index": {"safety": 0, "efficacy": 1, "cost": 2},
        "prediction_threshold": 0.5,
    }
    assert "sklearn" not in sys.modules
    assert "skops" not in sys.modules


def test_tfidf_logistic_multilabel_plan_rejects_invalid_or_unknown_labels():
    trainer = SklearnTfidfTrainer()
    base = _multilabel_training_input()

    with pytest.raises(ValidationError, match="must be a label list"):
        trainer.plan(
            recipe_key="tfidf_logistic_regression",
            config=_multilabel_config(),
            training_input=base.model_copy(
                update={"rows": ({"text": "Invalid", "labels": "efficacy"},)}
            ),
            seed=19,
        )

    with pytest.raises(ValidationError, match="Validation labels are absent"):
        trainer.plan(
            recipe_key="tfidf_logistic_regression",
            config=_multilabel_config(),
            training_input=base.model_copy(
                update={
                    "validation_rows": (
                        {"text": "New category", "labels": ["unseen"]},
                    )
                }
            ),
            seed=19,
        )


def test_tfidf_logistic_trains_real_multilabel_pipeline_when_runtime_is_installed(
    tmp_path,
):
    if any(importlib.util.find_spec(name) is None for name in ("sklearn", "skops")):
        pytest.skip("classical-CPU optional dependencies are not installed")

    trainer = SklearnTfidfTrainer()
    destination = tmp_path / "multilabel-output"
    output = trainer.train(
        recipe_key="tfidf_logistic_regression",
        config=_multilabel_config(),
        training_input=_multilabel_training_input(),
        destination=destination,
        seed=19,
    )

    from sklearn.multiclass import OneVsRestClassifier
    from skops.io import get_untrusted_types, load

    model_path = destination / "model.skops"
    pipeline = load(
        model_path,
        trusted=get_untrusted_types(file=model_path),
    )
    predictions = pipeline.predict(["Effective affordable option"])
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert isinstance(pipeline.named_steps["estimator"], OneVsRestClassifier)
    assert predictions.shape == (1, 3)
    assert set(output.validation_metrics) == {
        "subset_accuracy",
        "micro_f1",
        "macro_f1",
        "hamming_loss",
    }
    assert manifest["output_contract"]["labels"] == ["safety", "efficacy", "cost"]
    assert manifest["task"]["kind"] == "multilabel_classification"
    assert output.artifact_paths == ("model.skops", "recipe.json", "manifest.json")


def _transformer_config() -> dict:
    return {
        "base_model_asset_id": 17,
        "input_field": "text",
        "target_field": "label",
    }


def _sft_config() -> dict:
    return {
        "base_model_asset_id": 23,
        "prompt_field": "prompt",
        "completion_field": "answer",
        "prompt_template_version": "clinical-prompt-v1",
        "input_template": "Question: {prompt}\nAnswer: ",
        "completion_template": "{completion}",
    }


def test_all_builtin_recipes_resolve_to_registered_worker_plugins():
    expected = {
        "huggingface_sequence",
        "huggingface_token",
        "huggingface_span",
        "huggingface_causal_sft",
        "huggingface_seq2seq_sft",
        "sklearn_tfidf",
    }
    assert {plugin.key for plugin in trainer_plugins.list()} == expected
    for recipe in training_recipes.list():
        plugin = trainer_plugins.require_recipe(recipe.trainer_key, recipe.key)
        assert isinstance(plugin, TrainerPlugin)

    optional_roots = {
        "accelerate",
        "bitsandbytes",
        "peft",
        "safetensors",
        "torch",
        "transformers",
    }
    assert optional_roots.isdisjoint(sys.modules)


def test_transformer_task_plans_are_deterministic_and_schema_specific():
    sequence = HuggingFaceSequenceTrainer()
    sequence_input = TrainingInput(
        rows=(
            {"text": "Clear benefit", "label": "positive"},
            {"text": "No benefit", "label": "negative"},
        ),
        validation_rows=({"text": "Mixed", "label": "negative"},),
        dataset_fingerprint="dataset-sha",
        split_fingerprint="split-sha",
        task_kind="classification",
        task_schema={"input": "text", "output": "label"},
        base_model_fingerprint="base-model-sha",
    )
    first = sequence.plan(
        recipe_key="transformer_sequence_classification",
        config=_transformer_config(),
        training_input=sequence_input,
        seed=11,
    )
    second = sequence.plan(
        recipe_key="transformer_sequence_classification",
        config=_transformer_config(),
        training_input=sequence_input,
        seed=11,
    )
    assert first == second
    assert first.manifest["output_contract"]["labels"] == ["negative", "positive"]
    assert first.manifest["base_model"] == {
        "asset_id": 17,
        "fingerprint": "base-model-sha",
        "local_only": True,
    }
    assert ".bin" in first.manifest["artifact_policy"]["forbidden_extensions"]

    token = HuggingFaceTokenTrainer().plan(
        recipe_key="transformer_token_classification",
        config={
            **_transformer_config(),
            "input_field": "tokens",
            "target_field": "tags",
        },
        training_input=TrainingInput(
            rows=(
                {
                    "tokens": ["Aspirin", "helps"],
                    "tags": ["B-DRUG", "O"],
                },
            ),
            dataset_fingerprint="ner-dataset",
            split_fingerprint="ner-split",
            task_kind="token_labeling",
            base_model_fingerprint="ner-base",
        ),
        seed=12,
    )
    assert token.manifest["task"]["row_mapping"]["target"]["subtoken_policy"] == (
        "first_subtoken"
    )

    span = HuggingFaceSpanTrainer().plan(
        recipe_key="transformer_span_extraction",
        config={
            **_transformer_config(),
            "target_field": "evidence",
        },
        training_input=TrainingInput(
            rows=(
                {
                    "text": "Aspirin reduced pain.",
                    "evidence": {"start_char": 0, "end_char": 7},
                },
            ),
            dataset_fingerprint="span-dataset",
            split_fingerprint="span-split",
            task_kind="span_extraction",
            base_model_fingerprint="span-base",
        ),
        seed=13,
    )
    assert span.manifest["output_contract"]["type"] == "token_span"


def test_task_plans_reject_misaligned_labels_and_spans():
    with pytest.raises(ValidationError, match="counts must match"):
        HuggingFaceTokenTrainer().plan(
            recipe_key="transformer_token_classification",
            config={
                **_transformer_config(),
                "input_field": "tokens",
                "target_field": "tags",
            },
            training_input=TrainingInput(
                rows=({"tokens": ["one", "two"], "tags": ["O"]},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="token_labeling",
                base_model_fingerprint="base",
            ),
            seed=1,
        )
    with pytest.raises(ValidationError, match="invalid character offsets"):
        HuggingFaceSpanTrainer().plan(
            recipe_key="transformer_span_extraction",
            config={
                **_transformer_config(),
                "target_field": "span",
            },
            training_input=TrainingInput(
                rows=({"text": "short", "span": {"start_char": 2, "end_char": 99}},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="span_extraction",
                base_model_fingerprint="base",
            ),
            seed=1,
        )


@pytest.mark.parametrize(
    ("trainer", "recipe_key", "expected_architecture", "expected_parameterization"),
    [
        (
            HuggingFaceCausalSftTrainer(),
            "causal_lm_sft_full",
            "causal_lm",
            "full",
        ),
        (
            HuggingFaceCausalSftTrainer(),
            "causal_lm_sft_lora",
            "causal_lm",
            "lora",
        ),
        (
            HuggingFaceCausalSftTrainer(),
            "causal_lm_sft_qlora",
            "causal_lm",
            "qlora",
        ),
        (
            HuggingFaceSeq2SeqSftTrainer(),
            "seq2seq_lm_sft_full",
            "seq2seq_lm",
            "full",
        ),
        (
            HuggingFaceSeq2SeqSftTrainer(),
            "seq2seq_lm_sft_lora",
            "seq2seq_lm",
            "lora",
        ),
        (
            HuggingFaceSeq2SeqSftTrainer(),
            "seq2seq_lm_sft_qlora",
            "seq2seq_lm",
            "qlora",
        ),
    ],
)
def test_sft_plans_cover_full_lora_and_qlora(
    trainer,
    recipe_key,
    expected_architecture,
    expected_parameterization,
):
    plan = trainer.plan(
        recipe_key=recipe_key,
        config=_sft_config(),
        training_input=TrainingInput(
            rows=({"prompt": "What is PICO?", "answer": "A study framework."},),
            dataset_fingerprint="sft-dataset",
            split_fingerprint="sft-split",
            task_kind="instruction_tuning",
            base_model_fingerprint="sft-base",
        ),
        seed=42,
    )
    contract = plan.manifest["output_contract"]
    assert contract["architecture"] == expected_architecture
    assert contract["parameterization"] == expected_parameterization
    assert contract["prompt_tokens_in_loss"] is False
    assert len(plan.manifest["task"]["row_mapping"]["input"]["template_sha256"]) == 64


@pytest.mark.parametrize(
    ("trainer", "recipe_key", "config", "training_input"),
    [
        (
            HuggingFaceSequenceTrainer(),
            "transformer_sequence_classification",
            _transformer_config(),
            TrainingInput(
                rows=({"text": "Example", "label": "yes"},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="classification",
                base_model_fingerprint="base",
            ),
        ),
        (
            HuggingFaceTokenTrainer(),
            "transformer_token_classification",
            {
                **_transformer_config(),
                "input_field": "tokens",
                "target_field": "tags",
            },
            TrainingInput(
                rows=({"tokens": ["Example"], "tags": ["O"]},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="token_labeling",
                base_model_fingerprint="base",
            ),
        ),
        (
            HuggingFaceSpanTrainer(),
            "transformer_span_extraction",
            {**_transformer_config(), "target_field": "span"},
            TrainingInput(
                rows=({"text": "Example", "span": {"start_char": 0, "end_char": 7}},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="span_extraction",
                base_model_fingerprint="base",
            ),
        ),
        (
            HuggingFaceCausalSftTrainer(),
            "causal_lm_sft_qlora",
            _sft_config(),
            TrainingInput(
                rows=({"prompt": "Question", "answer": "Answer"},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="instruction_tuning",
                base_model_fingerprint="base",
            ),
        ),
        (
            HuggingFaceSeq2SeqSftTrainer(),
            "seq2seq_lm_sft_lora",
            _sft_config(),
            TrainingInput(
                rows=({"prompt": "Question", "answer": "Answer"},),
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                task_kind="instruction_tuning",
                base_model_fingerprint="base",
            ),
        ),
    ],
)
def test_optional_trainers_fail_before_writing_when_runtime_is_missing(
    monkeypatch,
    tmp_path,
    trainer,
    recipe_key,
    config,
    training_input,
):
    from al_medlit.training.trainers import huggingface_common

    monkeypatch.setattr(
        huggingface_common.importlib.util,
        "find_spec",
        lambda _module: None,
    )
    destination = tmp_path / recipe_key
    with pytest.raises(ValidationError, match="Worker preflight failed"):
        trainer.train(
            recipe_key=recipe_key,
            config=config,
            training_input=training_input,
            destination=destination,
            seed=42,
        )
    assert not destination.exists()


def test_artifact_inventory_rejects_executable_weight_serialization(tmp_path):
    unsafe = tmp_path / "pytorch_model.bin"
    unsafe.write_bytes(b"not-safe")
    with pytest.raises(ValidationError, match="Unsafe training artifact"):
        artifact_inventory(tmp_path)
