from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class InferenceWindowConfig(BaseModel):
    max_tokens: int = Field(default=4096, ge=128)
    overlap_tokens: int = Field(default=512, ge=0)
    aggregation: Literal["mean"] = "mean"

    @model_validator(mode="after")
    def valid_overlap(self):
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class InferenceDecoderConfig(BaseModel):
    version: Literal["evidence-block-decoder-v1"] = "evidence-block-decoder-v1"
    block_threshold: float = Field(default=0.5, ge=0, le=1)
    allow_cross_section: bool = False
    merge_adjacent: bool = False


class InferenceRunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    corpus_snapshot_id: int
    checkpoint_id: int
    compute_profile_id: int
    target_version_ids: list[int] = Field(min_length=1)
    window_config: InferenceWindowConfig = Field(default_factory=InferenceWindowConfig)
    decoder_config: InferenceDecoderConfig = Field(default_factory=InferenceDecoderConfig)
    idempotency_key: str = Field(min_length=8, max_length=120)

    @model_validator(mode="after")
    def unique_targets(self):
        if len(self.target_version_ids) != len(set(self.target_version_ids)):
            raise ValueError("target_version_ids must be unique")
        return self


class InferenceRunRead(BaseModel):
    id: int
    project_id: int
    corpus_snapshot_id: int
    checkpoint_id: int
    compute_profile_id: int
    name: str
    target_version_ids: list[int]
    window_config: dict
    decoder_config: dict
    status: str
    idempotency_key: str
    external_job_id: str | None
    diagnostics_artifact_id: int | None
    started_at: datetime | None
    completed_at: datetime | None
    failure_reason: str | None
    metrics: dict

    model_config = {"from_attributes": True}


class InferenceRunSummaryRead(BaseModel):
    """Non-operational run fields needed for annotator candidate discovery."""

    id: int
    project_id: int
    name: str
    target_version_ids: list[int]
    status: str

    model_config = {"from_attributes": True}


class InferenceWindowRead(BaseModel):
    id: int
    run_id: int
    document_id: int
    structure_version_id: int
    target_version_id: int
    stable_key: str
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    token_count: int
    status: str
    sentence_contribution_counts: dict
    diagnostics_artifact_id: int | None

    model_config = {"from_attributes": True}


class PredictionReviewCreate(BaseModel):
    action: Literal["accept", "modify", "reject"]
    start_sentence_id: int | None = None
    end_sentence_id: int | None = None
    labels: list[str] = Field(default_factory=list)
    note: str | None = None
    metadata_: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_boundaries(self):
        if self.action == "modify" and (
            self.start_sentence_id is None or self.end_sentence_id is None
        ):
            raise ValueError("Modified predictions require start_sentence_id and end_sentence_id")
        if self.action != "modify" and (
            self.start_sentence_id is not None or self.end_sentence_id is not None
        ):
            raise ValueError("Only a modify action accepts replacement boundaries")
        return self


class PredictionReviewRead(BaseModel):
    id: int
    prediction_id: int
    assignment_id: int | None
    guideline_version_id: int | None
    reviewer_user_id: int
    action: str
    revision: int
    resulting_annotation_id: int | None
    selected_boundaries: dict | None
    note: str | None
    metadata_: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class EvidenceCandidatePredictionRead(BaseModel):
    id: int
    project_id: int
    run_id: int
    checkpoint_id: int
    document_id: int
    structure_version_id: int
    target_version_id: int
    start_sentence_id: int
    end_sentence_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    start_char: int
    end_char: int
    block_confidence: float
    boundary_confidence: dict
    uncertainty: float
    decoder_version: str
    source_window_ids: list[int]
    status: str
    review_status: Literal["pending", "accepted", "modified", "rejected"] = "pending"
    diagnostics_artifact_id: int | None
    metadata_: dict
    reviews: list[PredictionReviewRead] = Field(default_factory=list)

    model_config = {"from_attributes": True}
