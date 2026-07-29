from typing import Literal

from pydantic import BaseModel, Field


class EvidencePairIaa(BaseModel):
    left_annotator_id: str
    right_annotator_id: str
    reviewed_sentence_count: int
    left_block_count: int
    right_block_count: int
    sentence_precision: float
    sentence_recall: float
    sentence_f1: float
    exact_f1: float
    iou_f1: dict[str, float]
    coverage: float
    overreach: float
    mean_start_boundary_deviation: float | None
    mean_end_boundary_deviation: float | None
    document_presence_agreement: bool


class EvidenceIaaMetrics(BaseModel):
    target_version_id: int
    structure_version_id: int | None
    guideline_version_id: int | None
    pairs: list[EvidencePairIaa] = Field(default_factory=list)
    aggregate: dict = Field(default_factory=dict)


class IaaReport(BaseModel):
    project_id: int
    annotation_type: str
    document_id: int | None
    status: Literal["ok", "insufficient_annotators", "no_items"]
    annotator_ids: list[str]
    item_count: int
    percent_agreement: float | None
    cohens_kappa: float | None
    fleiss_kappa: float | None
    span_detection_f1: float | None
    evidence_metrics: EvidenceIaaMetrics | None = None
