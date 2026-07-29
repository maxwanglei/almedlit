from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CorpusSnapshotDocumentCreate(BaseModel):
    document_id: int
    structure_version_id: int
    split: Literal["train", "validation", "test", "pool"] = "train"
    group_key: str
    source_hash: str = Field(min_length=64, max_length=64)
    metadata_: dict = Field(default_factory=dict)


class CorpusSnapshotCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    split_strategy: str = Field(
        default="document_grouped_80_10_10",
        min_length=1,
        max_length=80,
    )
    split_seed: int = 42
    documents: list[CorpusSnapshotDocumentCreate] = Field(min_length=1)
    metadata_: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_documents(self):
        ids = [item.document_id for item in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("A corpus snapshot may contain each document only once")
        return self


class CorpusSnapshotRead(BaseModel):
    id: int
    project_id: int
    artifact_id: int
    name: str
    split_strategy: str
    split_seed: int
    document_count: int
    metadata_: dict

    model_config = {"from_attributes": True}


class AnnotationSetItemCreate(BaseModel):
    source_annotation_id: int | None = None
    document_id: int
    structure_version_id: int
    target_version_id: int
    guideline_version_id: int | None = None
    start_sentence_id: int
    end_sentence_id: int
    start_sentence_ordinal: int = Field(ge=0)
    end_sentence_ordinal: int = Field(ge=0)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    block_text: str
    section_paths: list[list[str]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    source: Literal["adjudicated", "solo_gold"] = "adjudicated"
    metadata_: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_boundaries(self):
        if self.end_sentence_ordinal < self.start_sentence_ordinal:
            raise ValueError("end_sentence_ordinal must be at least start_sentence_ordinal")
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        return self


class AnnotationSetReviewRegionCreate(BaseModel):
    document_id: int
    structure_version_id: int
    target_version_id: int
    start_sentence_ordinal: int = Field(ge=0)
    end_sentence_ordinal: int = Field(ge=0)
    metadata_: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_boundaries(self):
        if self.end_sentence_ordinal < self.start_sentence_ordinal:
            raise ValueError("end_sentence_ordinal must be at least start_sentence_ordinal")
        return self


class AnnotationSetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    corpus_snapshot_id: int
    target_version_ids: list[int] = Field(min_length=1)
    guideline_version_ids: list[int] = Field(default_factory=list)
    items: list[AnnotationSetItemCreate] = Field(default_factory=list)
    reviewed_regions: list[AnnotationSetReviewRegionCreate] = Field(default_factory=list)
    metadata_: dict = Field(default_factory=dict)


class AnnotationSetRead(BaseModel):
    id: int
    project_id: int
    artifact_id: int
    corpus_snapshot_id: int
    name: str
    target_version_ids: list[int]
    guideline_version_ids: list[int]
    block_count: int
    reviewed_region_count: int
    metadata_: dict

    model_config = {"from_attributes": True}


class LineageArtifactRead(BaseModel):
    id: int
    project_id: int
    artifact_type: str
    schema_version: str
    content_hash: str
    content_type: str
    size_bytes: int
    manifest: dict

    model_config = {"from_attributes": True}


class LineageGraphRead(BaseModel):
    artifacts: list[LineageArtifactRead]
    edges: list[dict]


class ExportArtifactRead(BaseModel):
    id: int
    project_id: int
    artifact_id: int
    corpus_snapshot_id: int | None
    annotation_set_id: int | None
    format_key: str
    file_name: str
    row_count: int
    metadata_: dict

    model_config = {"from_attributes": True}
