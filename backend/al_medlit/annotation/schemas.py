from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from al_medlit.annotation.types import (
    AnnotationSource,
    AnnotationStatus,
    AnnotationType,
    CorrectionSeverity,
    CorrectionSource,
)
from al_medlit.evidence.schemas import EvidenceBlockPayloadV1, EvidenceBlockRead


class AnnotationCreate(BaseModel):
    project_id: int
    document_id: int
    annotation_type: AnnotationType
    label: str
    start_offset: int | None = None
    end_offset: int | None = None
    text_span: str | None = None
    source: AnnotationSource = "human"
    status: AnnotationStatus = "draft"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    annotator_user_id: int | None = None
    annotator_id: str | None = None
    model_checkpoint_id: str | None = None
    guideline_version_id: int | None = None
    structure_version_id: int | None = None
    head_annotation_id: int | None = None
    tail_annotation_id: int | None = None
    evidence: dict = Field(default_factory=dict)
    attributes: dict = Field(default_factory=dict)
    evidence_block: EvidenceBlockPayloadV1 | None = None


class AnnotationUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, min_length=1, max_length=120)
    start_offset: int | None = None
    end_offset: int | None = None
    text_span: str | None = None
    status: AnnotationStatus | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    head_annotation_id: int | None = None
    tail_annotation_id: int | None = None
    evidence: dict | None = None
    attributes: dict | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    evidence_block: EvidenceBlockPayloadV1 | None = None

    @model_validator(mode="after")
    def reject_null_for_non_nullable_fields(self) -> "AnnotationUpdate":
        for field in ("label", "status", "evidence", "attributes"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class AnnotationRead(AnnotationCreate):
    id: int
    created_at: datetime
    updated_at: datetime
    evidence_block: EvidenceBlockRead | None = None

    model_config = {"from_attributes": True}


class EvidenceCommandResult(BaseModel):
    command_group_key: str
    status: Literal["applied", "undone"]
    annotations: list[AnnotationRead]


class EvidenceCommandSummary(BaseModel):
    command_group_key: str
    operation: str
    status: Literal["applied", "undone"]
    project_id: int
    document_id: int
    target_version_id: int
    structure_version_id: int
    guideline_version_id: int | None
    actor_user_id: int | None
    created_at: datetime


class AnnotationCorrectionCreate(BaseModel):
    project_id: int
    document_id: int
    original_annotation_id: int | None = None
    corrected_annotation_id: int | None = None
    correction_source: CorrectionSource = "human"
    correction_note: str | None = None
    error_type: str | None = None
    severity: CorrectionSeverity = "medium"
    metadata_: dict = Field(default_factory=dict)


class AnnotationCorrectionRead(AnnotationCorrectionCreate):
    id: int
    created_by_user_id: int | None = None

    model_config = {"from_attributes": True}
