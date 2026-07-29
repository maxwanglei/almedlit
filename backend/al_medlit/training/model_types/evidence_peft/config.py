"""Typed LoRA and QLoRA training configuration."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidencePeftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["evidence_block_sentence_tagging"] = "evidence_block_sentence_tagging"
    base_model_asset_id: int = Field(gt=0)
    target_conditioning: bool = True
    peft_task_type: Literal["CAUSAL_LM", "TOKEN_CLS"] = "CAUSAL_LM"
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    rank: int = Field(default=16, ge=1, le=4_096)
    alpha: float = Field(default=32.0, gt=0)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    bias: Literal["none", "all", "lora_only"] = "none"

    seed: int = 42
    deterministic_algorithms: bool = True
    epochs: int = Field(default=3, ge=1, le=100)
    batch_size: int = Field(default=1, ge=1, le=1_024)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=4_096)
    learning_rate: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    optimizer: Literal["adamw_torch", "paged_adamw_8bit"] = "adamw_torch"
    scheduler: Literal["linear", "cosine", "constant"] = "linear"
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    early_stopping_patience: int = Field(default=2, ge=0, le=100)
    reload_best_checkpoint: Literal[True] = True
    max_sequence_length: int = Field(default=2_048, ge=128, le=131_072)
    max_new_tokens: int = Field(default=512, ge=16, le=8_192)
    tokenizer_truncation: Literal["longest_first"] = "longest_first"
    prompt_template_version: Literal["evidence-json-bio-v1"] = "evidence-json-bio-v1"
    bootstrap_samples: int = Field(default=200, ge=0, le=10_000)

    @field_validator("target_modules")
    @classmethod
    def safe_target_modules(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("At least one LoRA target module is required")
        if len(value) != len(set(value)):
            raise ValueError("LoRA target modules must be unique")
        for module in value:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", module):
                raise ValueError("LoRA target modules must be safe module names")
        return value


class EvidenceLoRAConfig(EvidencePeftConfig):
    model_kind: Literal["lora"] = "lora"
    device: Literal["cuda", "ascend"] = "cuda"


class EvidenceQLoRAConfig(EvidencePeftConfig):
    model_kind: Literal["qlora"] = "qlora"
    device: Literal["cuda"] = "cuda"
    quantization_bits: Literal[4] = 4
    quantization_type: Literal["nf4", "fp4"] = "nf4"
    double_quantization: bool = True
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    optimizer: Literal["paged_adamw_8bit"] = "paged_adamw_8bit"


class ImmutableBaseModelReference(BaseModel):
    """The package identity captured in every adapter package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_id: int = Field(gt=0)
    package_id: int = Field(gt=0)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    exact_revision: str = Field(min_length=1, max_length=255)

    @field_validator("exact_revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.lower() in {"head", "latest", "main", "master", "stable"}:
            raise ValueError("A PEFT base requires an exact immutable revision")
        return normalized


def validate_base_reference(
    config: EvidencePeftConfig,
    reference: ImmutableBaseModelReference,
) -> None:
    if config.base_model_asset_id != reference.asset_id:
        raise ValueError("Resolved base-model reference does not match base_model_asset_id")
