from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class EvidenceTargetVersionCreate(BaseModel):
    text: str = Field(min_length=1)
    guidance: str | None = None
    inclusion_guidance: str | None = None
    exclusion_guidance: str | None = None
    metadata_: dict = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("target text is required")
        return cleaned


class EvidenceTargetCreate(BaseModel):
    task_id: int
    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    initial_version: EvidenceTargetVersionCreate


class EvidenceTargetVersionRead(EvidenceTargetVersionCreate):
    id: int
    target_id: int
    version_number: int
    created_by_user_id: int | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EvidenceTargetRead(BaseModel):
    id: int
    project_id: int
    task_id: int
    key: str
    name: str
    description: str | None
    active_version_id: int | None
    is_active: bool
    created_by_user_id: int | None
    versions: list[EvidenceTargetVersionRead] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EvidenceTargetActivate(BaseModel):
    version_id: int


class EvidenceBlockPayloadV1(BaseModel):
    structure_version_id: int
    target_version_id: int
    start_sentence_id: int
    end_sentence_id: int
    labels: list[str] = Field(default_factory=list)
    note: str | None = None
    boundary_policy: Literal["sentence"] = "sentence"


class EvidenceBlockRead(EvidenceBlockPayloadV1):
    annotation_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    start_offset: int
    end_offset: int
    revision: int
    locked: bool
    last_command_group_key: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EvidenceMergeRequest(BaseModel):
    annotation_ids: list[int] = Field(min_length=2)
    expected_revisions: dict[int, int]
    labels: list[str] | None = None
    note: str | None = None
    boundary_policy: Literal["sentence"] = "sentence"


class EvidenceSplitRequest(BaseModel):
    expected_revision: int
    split_before_sentence_id: int


class EvidenceReviewIntervalRequest(BaseModel):
    target_version_id: int
    structure_version_id: int
    guideline_version_id: int | None = None
    start_sentence_id: int
    end_sentence_id: int
    reason: str | None = None


class EvidenceReviewIntervalRead(BaseModel):
    id: int
    start_sentence_id: int
    end_sentence_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EvidenceReviewEventRead(BaseModel):
    id: int
    action: str
    start_sentence_id: int
    end_sentence_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    reason: str | None
    metadata_: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class EvidenceReviewCoverageRead(BaseModel):
    project_id: int
    document_id: int
    structure_version_id: int
    target_version_id: int
    guideline_version_id: int | None
    reviewer_user_id: int
    intervals: list[EvidenceReviewIntervalRead]
    events: list[EvidenceReviewEventRead]
    fully_reviewed: bool


class EvidenceAdjudicationCreate(BaseModel):
    target_version_id: int
    structure_version_id: int
    guideline_version_id: int
    strategy: Literal["a", "b", "union", "intersection", "custom"]
    source_annotation_ids: list[int] = Field(default_factory=list)
    start_sentence_id: int | None = None
    end_sentence_id: int | None = None
    labels: list[str] = Field(default_factory=list)
    note: str | None = None
    solo_gold: bool = False


class EvidenceComparisonBlock(BaseModel):
    annotation_id: int
    annotator_user_id: int | None
    annotator_id: str | None
    status: str
    start_sentence_id: int
    end_sentence_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    labels: list[str]
    note: str | None


class EvidenceAdjudicationRead(BaseModel):
    project_id: int
    document_id: int
    target_version_id: int
    structure_version_id: int
    guideline_version_id: int
    blocks: list[EvidenceComparisonBlock]
