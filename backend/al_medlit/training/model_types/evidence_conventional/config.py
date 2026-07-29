"""Typed, persisted configuration for conventional Evidence models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceConventionalConfig(BaseModel):
    """Options shared by sentence-scoring and sequence baselines.

    Window creation happens when the immutable dataset is prepared.  These
    models therefore persist the decoder threshold and their *actual* feature
    preparation settings, rather than estimating transformer tokens.
    """

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["evidence_block_sentence_tagging"] = (
        "evidence_block_sentence_tagging"
    )
    target_conditioning: bool = True
    seed: int = 42
    sentence_score_threshold: float = Field(default=0.5, ge=0, le=1)
    bootstrap_samples: int = Field(default=200, ge=0, le=10_000)
    lowercase: bool = True
    strip_accents: Literal["ascii", "unicode"] | None = "unicode"
    min_df: int = Field(default=1, ge=1)
    max_df: float = Field(default=1.0, gt=0, le=1)
    max_features: int | None = Field(default=50_000, ge=100)
    word_ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    class_weighting: Literal["none", "balanced"] = "balanced"

    @model_validator(mode="after")
    def validate_feature_range(self):
        minimum, maximum = self.word_ngram_range
        if minimum < 1 or maximum < minimum or maximum > 5:
            raise ValueError("word_ngram_range must be ordered between 1 and 5")
        return self


class EvidenceSVMConfig(EvidenceConventionalConfig):
    model_kind: Literal["svm"] = "svm"
    c: float = Field(default=1.0, gt=0)
    loss: Literal["hinge", "squared_hinge"] = "squared_hinge"
    tolerance: float = Field(default=1e-4, gt=0)
    max_iterations: int = Field(default=5_000, ge=1, le=1_000_000)


class EvidenceRandomForestConfig(EvidenceConventionalConfig):
    model_kind: Literal["random_forest"] = "random_forest"
    n_estimators: int = Field(default=300, ge=1, le=10_000)
    max_depth: int | None = Field(default=None, ge=1)
    min_samples_leaf: int = Field(default=1, ge=1)
    max_features_per_split: Literal["sqrt", "log2"] | float | None = "sqrt"
    n_jobs: int = Field(default=1, ge=1, le=256)

    @model_validator(mode="after")
    def validate_split_features(self):
        value = self.max_features_per_split
        if isinstance(value, float) and not 0 < value <= 1:
            raise ValueError("Floating max_features_per_split must be in (0, 1]")
        return self


class EvidenceCRFConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_type: Literal["evidence_block_sentence_tagging"] = (
        "evidence_block_sentence_tagging"
    )
    model_kind: Literal["crf"] = "crf"
    target_conditioning: bool = True
    seed: int = 42
    bootstrap_samples: int = Field(default=200, ge=0, le=10_000)
    token_pattern: str = r"(?u)\b\w\w+\b"
    lowercase: bool = True
    c1: float = Field(default=0.1, ge=0)
    c2: float = Field(default=0.1, ge=0)
    max_iterations: int = Field(default=100, ge=1, le=100_000)
    all_possible_transitions: bool = True

    @model_validator(mode="after")
    def validate_token_pattern(self):
        import re

        try:
            re.compile(self.token_pattern)
        except re.error as exc:
            raise ValueError("token_pattern must be a valid regular expression") from exc
        return self
