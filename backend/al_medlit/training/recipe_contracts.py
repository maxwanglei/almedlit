"""Task-neutral contracts for trusted training recipes and worker runtimes."""

from __future__ import annotations

from enum import StrEnum
from string import Formatter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from al_medlit.training.contracts import ModelFamily


class TaskKind(StrEnum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    TOKEN_LABELING = "token_labeling"
    SPAN_EXTRACTION = "span_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    RANKING = "ranking"
    GENERATION = "generation"
    INSTRUCTION_TUNING = "instruction_tuning"


class RuntimeClass(StrEnum):
    CLASSICAL_CPU = "classical-cpu"
    TORCH_CPU = "torch-cpu"
    TRANSFORMER_CPU = "transformer-cpu"
    PEFT_ACCELERATOR = "peft-accelerator"
    QLORA_CUDA = "qlora-cuda"


class Parameterization(StrEnum):
    FULL = "full"
    HEAD_ONLY = "head_only"
    LORA = "lora"
    QLORA = "qlora"


class RecipeContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EnvironmentRequirement(RecipeContract):
    runtime_class: RuntimeClass
    packages: tuple[str, ...]
    devices: tuple[str, ...]
    minimum_memory_gb: float = Field(ge=0)
    requires_verified_environment: bool = True
    setup_hint: str


class TrainingRecipeDescriptor(RecipeContract):
    schema_version: Literal["training-recipe-descriptor-v1"] = (
        "training-recipe-descriptor-v1"
    )
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=120)
    version: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=160)
    description: str
    model_family: ModelFamily
    architecture_family: str = Field(min_length=1, max_length=80)
    parameterization: Parameterization
    supported_task_kinds: tuple[TaskKind, ...]
    trainer_key: str = Field(pattern=r"^[a-z][a-z0-9_]*$", max_length=120)
    implementation_status: Literal["implemented", "experimental"]
    environment: EnvironmentRequirement
    config_schema: dict
    artifact_formats: tuple[str, ...]
    supports_resume: bool = False


class DatasetFieldBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_field: str = Field(min_length=1, max_length=255)
    target_field: str = Field(min_length=1, max_length=255)


class TfidfLinearRegressionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: DatasetFieldBinding
    max_features: int = Field(default=50_000, ge=100, le=2_000_000)
    ngram_min: int = Field(default=1, ge=1, le=5)
    ngram_max: int = Field(default=2, ge=1, le=5)
    min_document_frequency: int = Field(default=2, ge=1)
    fit_intercept: bool = True

    @model_validator(mode="after")
    def validate_ngrams(self):
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be greater than or equal to ngram_min")
        return self


class TfidfLogisticRegressionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: DatasetFieldBinding
    max_features: int = Field(default=50_000, ge=100, le=2_000_000)
    ngram_min: int = Field(default=1, ge=1, le=5)
    ngram_max: int = Field(default=2, ge=1, le=5)
    min_document_frequency: int = Field(default=2, ge=1)
    regularization_c: float = Field(default=1.0, gt=0)
    class_weight: Literal["balanced"] | None = None
    max_iterations: int = Field(default=1_000, ge=10, le=100_000)

    @model_validator(mode="after")
    def validate_ngrams(self):
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be greater than or equal to ngram_min")
        return self


class TransformerTaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_asset_id: int = Field(gt=0)
    input_field: str = Field(min_length=1, max_length=255)
    target_field: str = Field(min_length=1, max_length=255)
    max_sequence_length: int = Field(default=512, ge=32, le=32_768)
    learning_rate: float = Field(default=2e-5, gt=0)
    epochs: float = Field(default=3.0, gt=0)
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    mixed_precision: Literal["none", "fp16", "bf16"] = "none"


class LanguageModelSftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_asset_id: int = Field(gt=0)
    prompt_field: str = Field(min_length=1, max_length=255)
    completion_field: str = Field(min_length=1, max_length=255)
    prompt_template_version: str = Field(min_length=1, max_length=120)
    input_template: str = Field(default="{prompt}", min_length=1, max_length=8_192)
    completion_template: str = Field(
        default="{completion}",
        min_length=1,
        max_length=8_192,
    )
    max_sequence_length: int = Field(default=2_048, ge=128, le=131_072)
    learning_rate: float = Field(default=2e-5, gt=0)
    epochs: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=8, ge=1)
    mixed_precision: Literal["fp16", "bf16"] = "bf16"
    lora_rank: int | None = Field(default=16, ge=1, le=1024)
    lora_alpha: int | None = Field(default=32, ge=1, le=4096)
    lora_dropout: float | None = Field(default=0.05, ge=0, lt=1)
    lora_target_modules: list[str] | None = None

    @model_validator(mode="after")
    def validate_templates(self):
        templates = (
            ("input_template", self.input_template, "prompt"),
            ("completion_template", self.completion_template, "completion"),
        )
        for name, template, expected in templates:
            parsed = tuple(Formatter().parse(template))
            fields = [field for _, field, _, _ in parsed if field is not None]
            if fields != [expected]:
                raise ValueError(
                    f"{name} must contain exactly one '{{{expected}}}' placeholder"
                )
            if any(format_spec or conversion for _, _, format_spec, conversion in parsed):
                raise ValueError(f"{name} cannot use conversions or format specifications")
        if self.lora_target_modules is not None:
            cleaned = [module.strip() for module in self.lora_target_modules]
            if not cleaned or any(not module for module in cleaned):
                raise ValueError("lora_target_modules must contain non-empty module names")
            if len(cleaned) != len(set(cleaned)):
                raise ValueError("lora_target_modules must be unique")
            self.lora_target_modules = cleaned
        return self


class RecipeConfigurationValidation(BaseModel):
    recipe_key: str
    recipe_version: str
    valid: bool
    normalized_config: dict | None = None
    errors: list[str] = Field(default_factory=list)
