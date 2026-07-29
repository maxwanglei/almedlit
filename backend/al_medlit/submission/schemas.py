from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

SubmissionKind = Literal["submission", "re_export"]


class SubmissionCreate(BaseModel):
    assignment_id: int | None = None
    annotator_user_id: int | None = None
    annotator_id: str | None = None
    kind: SubmissionKind = "submission"
    metadata_: dict = Field(default_factory=dict)


class SubmissionRead(BaseModel):
    id: int
    project_id: int
    document_id: int
    assignment_id: int | None
    annotator_user_id: int | None
    annotator_id: str | None
    kind: SubmissionKind
    storage_key: str
    file_name: str
    content_type: str
    size_bytes: int
    checksum_sha256: str
    annotation_count: int
    metadata_: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
