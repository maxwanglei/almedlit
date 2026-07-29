from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    project_id: int
    external_id: str | None = None
    title: str | None = None
    text: str
    source: str | None = None
    metadata_: dict = Field(default_factory=dict)


class DocumentRead(DocumentCreate):
    id: int
    active_structure_version_id: int | None = None
    sentences: list[list[int]] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class StructureVersionRead(BaseModel):
    id: int
    document_id: int
    version: int
    segmenter_name: str
    segmenter_version: str
    source_hash: str
    text_length: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentSectionRead(BaseModel):
    id: int
    ordinal: int
    title: str | None = None
    path: list[str] = Field(default_factory=list)
    kind: str
    start_offset: int
    end_offset: int
    locator: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class DocumentParagraphRead(BaseModel):
    id: int
    section_id: int
    ordinal: int
    section_ordinal: int
    start_offset: int
    end_offset: int
    locator: dict = Field(default_factory=dict)

    model_config = {"from_attributes": True}


class DocumentSentenceRead(BaseModel):
    id: int
    section_id: int
    paragraph_id: int
    ordinal: int
    paragraph_ordinal: int
    start_offset: int
    end_offset: int
    text: str


class StructureRangeRead(BaseModel):
    start_ordinal: int
    end_ordinal: int
    total_sentences: int
    has_more: bool


class DocumentStructureRead(BaseModel):
    document_id: int
    active_structure_version_id: int | None
    structure_version: StructureVersionRead
    range: StructureRangeRead
    sections: list[DocumentSectionRead] = Field(default_factory=list)
    paragraphs: list[DocumentParagraphRead] = Field(default_factory=list)
    sentences: list[DocumentSentenceRead] = Field(default_factory=list)


class StructureRebuildRequest(BaseModel):
    activate: bool = True
