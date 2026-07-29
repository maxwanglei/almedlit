from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceBlockSentenceTaggerConfig(BaseModel):
    task_type: Literal["evidence_block_sentence_tagging"] = (
        "evidence_block_sentence_tagging"
    )
    model_id: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=7)
    base_model_asset_id: int | None = Field(default=None, gt=0)
    staged_base_model_artifact_id: int | None = None
    local_model_path: str | None = None
    max_length: int = Field(default=4096, ge=128)
    encoder_max_length: int | None = Field(default=None, ge=128)
    target_conditioning: bool = True
    target_marker_token: str = "[TARGET]"
    sentence_marker_token: str = "[SENT]"
    window_overlap_tokens: int = Field(default=512, ge=0)
    labels: tuple[Literal["O", "B", "I"], ...] = ("O", "B", "I")
    class_weighting: Literal["none", "balanced"] = "balanced"
    loss: Literal["cross_entropy"] = "cross_entropy"
    use_crf: Literal[False] = False
    local_files_only: bool = True
    # Deterministic, dependency-free execution used by CI and smoke tests. It
    # exercises the complete orchestration/artifact/lineage path without
    # pretending to be a scientific model checkpoint.
    synthetic_mode: bool = False
    synthetic_seed: int = 42
    seed: int = Field(default=42, ge=0, le=2**31 - 1)
    deterministic: bool = True
    epochs: int = Field(default=1, ge=1, le=100)
    batch_size: int = Field(default=2, ge=1, le=256)
    gradient_accumulation_steps: int = Field(default=1, ge=1, le=1024)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["none", "linear", "cosine"] = "linear"
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    gradient_clip_norm: float = Field(default=1.0, gt=0)
    early_stopping_patience: int = Field(default=3, ge=0, le=100)
    early_stopping_min_delta: float = Field(default=0.0, ge=0)
    bootstrap_samples: int = Field(default=200, ge=0, le=10_000)
    device: Literal["auto", "cpu", "mps", "cuda", "ascend"] = "auto"

    @model_validator(mode="after")
    def validate_model_source_and_context(self):
        remote_source = self.model_id is not None
        staged_source = self.staged_base_model_artifact_id is not None
        catalog_source = self.base_model_asset_id is not None
        local_source = self.local_model_path is not None
        source_count = sum((remote_source, staged_source, catalog_source, local_source))
        if not self.synthetic_mode and source_count != 1:
            raise ValueError(
                "Exactly one immutable model source is required"
            )
        if self.synthetic_mode and source_count > 1:
            raise ValueError("Synthetic mode accepts at most one model source")
        if remote_source and not self.revision:
            raise ValueError("An immutable revision is required with model_id")
        if self.encoder_max_length is not None and self.max_length > self.encoder_max_length:
            raise ValueError("max_length exceeds the selected encoder capacity")
        if self.labels != ("O", "B", "I"):
            raise ValueError("Evidence-block v1 labels must be ordered as O, B, I")
        if self.target_marker_token == self.sentence_marker_token:
            raise ValueError("Target and sentence marker tokens must differ")
        if self.window_overlap_tokens >= self.max_length:
            raise ValueError("window_overlap_tokens must be smaller than max_length")
        return self
