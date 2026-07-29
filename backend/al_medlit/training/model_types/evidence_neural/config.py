"""Typed, persisted configuration for compact neural Evidence models."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

EvidenceDevice = Literal["auto", "cpu", "mps", "cuda", "ascend"]


class EvidenceNeuralConfig(BaseModel):
    """Configuration shared by the BiLSTM and sentence-CNN plugins.

    Token preparation is deliberately part of the persisted configuration.
    Neither plugin uses transformer token estimates: both build and package the
    exact word vocabulary used for training.
    """

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["evidence_block_sentence_tagging"] = "evidence_block_sentence_tagging"
    labels: tuple[Literal["O", "B", "I"], ...] = ("O", "B", "I")
    target_conditioning: bool = True
    tokenizer_pattern: str = r"(?u)\b\w+\b"
    lowercase: bool = True
    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    min_token_frequency: int = Field(default=1, ge=1)
    max_vocabulary_size: int = Field(default=50_000, ge=8, le=2_000_000)
    max_tokens_per_sentence: int = Field(default=256, ge=4, le=16_384)
    max_sentences_per_document: int = Field(default=512, ge=1, le=16_384)

    seed: int = 42
    deterministic_algorithms: bool = True
    epochs: int = Field(default=20, ge=1, le=1_000)
    batch_size: int = Field(default=8, ge=1, le=4_096)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=4_096)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["none", "linear"] = "linear"
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    early_stopping_patience: int = Field(default=5, ge=0, le=1_000)
    early_stopping_min_delta: float = Field(default=0.0, ge=0)
    reload_best_checkpoint: Literal[True] = True
    device: EvidenceDevice = "auto"

    embedding_dimension: int = Field(default=128, ge=4, le=8_192)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    bootstrap_samples: int = Field(default=200, ge=0, le=10_000)

    @model_validator(mode="after")
    def validate_tokenizer_and_labels(self):
        try:
            re.compile(self.tokenizer_pattern)
        except re.error as exc:
            raise ValueError("tokenizer_pattern must be a valid regular expression") from exc
        if self.labels != ("O", "B", "I"):
            raise ValueError("Evidence-block v1 labels must be ordered as O, B, I")
        return self


class EvidenceBiLSTMConfig(EvidenceNeuralConfig):
    model_kind: Literal["bilstm"] = "bilstm"
    hidden_dimension: int = Field(default=128, ge=4, le=8_192)
    recurrent_layers: int = Field(default=1, ge=1, le=16)
    bidirectional: Literal[True] = True


class EvidenceCNNConfig(EvidenceNeuralConfig):
    model_kind: Literal["cnn"] = "cnn"
    convolution_channels: int = Field(default=128, ge=4, le=8_192)
    token_kernel_sizes: tuple[int, ...] = (2, 3, 4)
    sentence_context_kernel_size: int = Field(default=3, ge=1, le=31)

    @model_validator(mode="after")
    def validate_kernels(self):
        if not self.token_kernel_sizes:
            raise ValueError("At least one token convolution kernel is required")
        if len(set(self.token_kernel_sizes)) != len(self.token_kernel_sizes):
            raise ValueError("token_kernel_sizes must be unique")
        if any(
            kernel < 1 or kernel > self.max_tokens_per_sentence
            for kernel in self.token_kernel_sizes
        ):
            raise ValueError(
                "token_kernel_sizes must be positive and no larger than max_tokens_per_sentence"
            )
        if self.sentence_context_kernel_size % 2 == 0:
            raise ValueError("sentence_context_kernel_size must be odd")
        return self
