"""Trusted, task-neutral training recipe catalog.

The API publishes requirements and validates configuration, while package
installation and execution stay inside separately verified worker runtimes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from al_medlit.core.exceptions import NotFoundError, ValidationError
from al_medlit.training.contracts import ModelFamily
from al_medlit.training.recipe_contracts import (
    EnvironmentRequirement,
    LanguageModelSftConfig,
    Parameterization,
    RecipeConfigurationValidation,
    RuntimeClass,
    TaskKind,
    TfidfLinearRegressionConfig,
    TfidfLogisticRegressionConfig,
    TrainingRecipeDescriptor,
    TransformerTaskConfig,
)

ConfigT = TypeVar("ConfigT", bound=BaseModel)


class TrainingRecipeRegistry:
    def __init__(self) -> None:
        self._descriptors: dict[str, TrainingRecipeDescriptor] = {}
        self._config_models: dict[str, type[BaseModel]] = {}

    def register(
        self,
        descriptor: TrainingRecipeDescriptor,
        config_model: type[ConfigT],
        *,
        replace: bool = False,
    ) -> None:
        if descriptor.key in self._descriptors and not replace:
            raise ValidationError(
                f"Training recipe '{descriptor.key}' is already registered"
            )
        if descriptor.config_schema != config_model.model_json_schema():
            raise ValidationError(
                f"Training recipe '{descriptor.key}' has a mismatched configuration schema"
            )
        self._descriptors[descriptor.key] = descriptor
        self._config_models[descriptor.key] = config_model

    def get(self, key: str) -> TrainingRecipeDescriptor:
        try:
            return self._descriptors[key]
        except KeyError as exc:
            raise NotFoundError(f"Training recipe '{key}' was not found") from exc

    def list(self) -> tuple[TrainingRecipeDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def validate(
        self,
        key: str,
        config: dict,
    ) -> RecipeConfigurationValidation:
        descriptor = self.get(key)
        config_model = self._config_models[key]
        try:
            normalized = config_model.model_validate(config).model_dump(mode="json")
        except PydanticValidationError as exc:
            return RecipeConfigurationValidation(
                recipe_key=descriptor.key,
                recipe_version=descriptor.version,
                valid=False,
                errors=[
                    ".".join(str(part) for part in error["loc"]) + ": " + error["msg"]
                    for error in exc.errors()
                ],
            )
        return RecipeConfigurationValidation(
            recipe_key=descriptor.key,
            recipe_version=descriptor.version,
            valid=True,
            normalized_config=normalized,
        )


def _environment(
    runtime_class: RuntimeClass,
    *,
    packages: tuple[str, ...],
    devices: tuple[str, ...],
    memory: float,
) -> EnvironmentRequirement:
    return EnvironmentRequirement(
        runtime_class=runtime_class,
        packages=packages,
        devices=devices,
        minimum_memory_gb=memory,
        setup_hint=(
            f"Ask a workspace administrator to enable and verify the "
            f"{runtime_class.value} runtime."
        ),
    )


def _descriptor(
    *,
    key: str,
    label: str,
    description: str,
    family: ModelFamily,
    architecture: str,
    parameterization: Parameterization,
    tasks: Iterable[TaskKind],
    trainer_key: str,
    environment: EnvironmentRequirement,
    config_model: type[BaseModel],
    formats: tuple[str, ...],
    experimental: bool = False,
) -> tuple[TrainingRecipeDescriptor, type[BaseModel]]:
    return (
        TrainingRecipeDescriptor(
            key=key,
            version="1",
            label=label,
            description=description,
            model_family=family,
            architecture_family=architecture,
            parameterization=parameterization,
            supported_task_kinds=tuple(tasks),
            trainer_key=trainer_key,
            implementation_status="experimental" if experimental else "implemented",
            environment=environment,
            config_schema=config_model.model_json_schema(),
            artifact_formats=formats,
        ),
        config_model,
    )


def builtin_training_recipes() -> tuple[
    tuple[TrainingRecipeDescriptor, type[BaseModel]], ...
]:
    classical = _environment(
        RuntimeClass.CLASSICAL_CPU,
        packages=("scikit-learn", "skops"),
        devices=("cpu",),
        memory=2,
    )
    transformer = _environment(
        RuntimeClass.TRANSFORMER_CPU,
        packages=("torch", "transformers", "safetensors"),
        devices=("cpu", "mps", "cuda", "ascend"),
        memory=8,
    )
    accelerator = _environment(
        RuntimeClass.PEFT_ACCELERATOR,
        packages=("torch", "transformers", "peft", "safetensors"),
        devices=("mps", "cuda", "ascend"),
        memory=16,
    )
    qlora = _environment(
        RuntimeClass.QLORA_CUDA,
        packages=("torch", "transformers", "peft", "bitsandbytes", "safetensors"),
        devices=("cuda",),
        memory=16,
    )

    recipes = (
        _descriptor(
            key="tfidf_linear_regression",
            label="TF-IDF linear regression",
            description="Sparse text features with a linear numeric predictor.",
            family=ModelFamily.CONVENTIONAL_ML,
            architecture="linear_regression",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.REGRESSION,),
            trainer_key="sklearn_tfidf",
            environment=classical,
            config_model=TfidfLinearRegressionConfig,
            formats=("skops", "json"),
        ),
        _descriptor(
            key="tfidf_logistic_regression",
            label="TF-IDF logistic regression",
            description="Sparse text baseline for single-label or multilabel classification.",
            family=ModelFamily.CONVENTIONAL_ML,
            architecture="logistic_regression",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.CLASSIFICATION, TaskKind.MULTILABEL_CLASSIFICATION),
            trainer_key="sklearn_tfidf",
            environment=classical,
            config_model=TfidfLogisticRegressionConfig,
            formats=("skops", "json"),
        ),
        _descriptor(
            key="transformer_sequence_classification",
            label="Transformer sequence classification",
            description="Encoder fine-tuning for document or sequence classification.",
            family=ModelFamily.DEEP_LEARNING,
            architecture="transformer_encoder",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.CLASSIFICATION, TaskKind.MULTILABEL_CLASSIFICATION),
            trainer_key="huggingface_sequence",
            environment=transformer,
            config_model=TransformerTaskConfig,
            formats=("safetensors", "huggingface_json", "tokenizer"),
        ),
        _descriptor(
            key="transformer_token_classification",
            label="Transformer token classification",
            description="Encoder fine-tuning for NER and other token-labeling tasks.",
            family=ModelFamily.DEEP_LEARNING,
            architecture="transformer_encoder",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.TOKEN_LABELING,),
            trainer_key="huggingface_token",
            environment=transformer,
            config_model=TransformerTaskConfig,
            formats=("safetensors", "huggingface_json", "tokenizer"),
        ),
        _descriptor(
            key="transformer_span_extraction",
            label="Transformer span extraction",
            description="Encoder fine-tuning for extractive answers and labeled spans.",
            family=ModelFamily.DEEP_LEARNING,
            architecture="transformer_encoder",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.SPAN_EXTRACTION,),
            trainer_key="huggingface_span",
            environment=transformer,
            config_model=TransformerTaskConfig,
            formats=("safetensors", "huggingface_json", "tokenizer"),
        ),
        _descriptor(
            key="causal_lm_sft_full",
            label="Causal LLM full fine-tuning",
            description="Full-parameter supervised fine-tuning for instruction data.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="causal_lm",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_causal_sft",
            environment=accelerator,
            config_model=LanguageModelSftConfig,
            formats=("safetensors", "huggingface_json", "tokenizer"),
            experimental=True,
        ),
        _descriptor(
            key="causal_lm_sft_lora",
            label="Causal LLM LoRA",
            description="Parameter-efficient supervised fine-tuning with LoRA adapters.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="causal_lm",
            parameterization=Parameterization.LORA,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_causal_sft",
            environment=accelerator,
            config_model=LanguageModelSftConfig,
            formats=("peft_adapter_safetensors", "json", "tokenizer"),
        ),
        _descriptor(
            key="causal_lm_sft_qlora",
            label="Causal LLM QLoRA",
            description="Four-bit parameter-efficient supervised fine-tuning.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="causal_lm",
            parameterization=Parameterization.QLORA,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_causal_sft",
            environment=qlora,
            config_model=LanguageModelSftConfig,
            formats=("peft_adapter_safetensors", "json", "tokenizer"),
            experimental=True,
        ),
        _descriptor(
            key="seq2seq_lm_sft_full",
            label="Seq2seq full fine-tuning",
            description="Full-parameter supervised fine-tuning for text-to-text models.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="seq2seq_lm",
            parameterization=Parameterization.FULL,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_seq2seq_sft",
            environment=accelerator,
            config_model=LanguageModelSftConfig,
            formats=("safetensors", "huggingface_json", "tokenizer"),
            experimental=True,
        ),
        _descriptor(
            key="seq2seq_lm_sft_lora",
            label="Seq2seq LoRA",
            description="Parameter-efficient text-to-text fine-tuning with LoRA adapters.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="seq2seq_lm",
            parameterization=Parameterization.LORA,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_seq2seq_sft",
            environment=accelerator,
            config_model=LanguageModelSftConfig,
            formats=("peft_adapter_safetensors", "json", "tokenizer"),
        ),
        _descriptor(
            key="seq2seq_lm_sft_qlora",
            label="Seq2seq QLoRA",
            description="Four-bit text-to-text fine-tuning with LoRA adapters.",
            family=ModelFamily.LLM_FINETUNE,
            architecture="seq2seq_lm",
            parameterization=Parameterization.QLORA,
            tasks=(TaskKind.GENERATION, TaskKind.INSTRUCTION_TUNING),
            trainer_key="huggingface_seq2seq_sft",
            environment=qlora,
            config_model=LanguageModelSftConfig,
            formats=("peft_adapter_safetensors", "json", "tokenizer"),
            experimental=True,
        ),
    )
    return recipes


training_recipes = TrainingRecipeRegistry()
for _recipe_descriptor, _recipe_config in builtin_training_recipes():
    training_recipes.register(_recipe_descriptor, _recipe_config)
