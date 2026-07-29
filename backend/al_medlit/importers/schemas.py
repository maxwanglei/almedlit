from typing import Literal

from pydantic import BaseModel, Field

ImportStatus = Literal["full_text", "abstract_only", "not_found", "error"]


class ImportPreviewRequest(BaseModel):
    pmids: list[str] = Field(default_factory=list)


class ImportPreviewItem(BaseModel):
    pmid: str
    title: str = ""
    journal: str = ""
    year: str = ""
    pmcid: str | None = None
    status: ImportStatus
    has_full_text: bool = False
    has_abstract: bool = False


class ImportPreviewResponse(BaseModel):
    items: list[ImportPreviewItem] = Field(default_factory=list)


class ImportRequest(BaseModel):
    pmids: list[str] = Field(default_factory=list)
    include_abstract_only: bool = False


class ImportOutcome(BaseModel):
    pmid: str
    status: ImportStatus
    title: str = ""
    document_id: int | None = None
    reason: str | None = None


class ImportResponse(BaseModel):
    created: list[ImportOutcome] = Field(default_factory=list)
    skipped: list[ImportOutcome] = Field(default_factory=list)
