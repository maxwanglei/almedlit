from pathlib import PureWindowsPath
from unicodedata import category

from pydantic import BaseModel, Field, field_validator, model_validator


class ExportCreate(BaseModel):
    format_key: str = Field(min_length=1, max_length=100)
    annotation_set_id: int | None = None
    corpus_snapshot_id: int | None = None
    file_name: str | None = Field(
        default=None,
        max_length=255,
        description="A leaf file name. Paths and control characters are not allowed.",
    )
    options: dict = Field(default_factory=dict)
    metadata_: dict = Field(default_factory=dict)

    @field_validator("file_name")
    @classmethod
    def valid_leaf_file_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if (
            not value.strip()
            or value in {".", ".."}
            or "/" in value
            or "\\" in value
            or PureWindowsPath(value).drive
            or any(category(char) in {"Cc", "Cs"} for char in value)
        ):
            raise ValueError("file_name must be a leaf name without paths or control characters")
        return value

    @model_validator(mode="after")
    def has_source(self):
        if self.annotation_set_id is None and self.corpus_snapshot_id is None:
            raise ValueError("annotation_set_id or corpus_snapshot_id is required")
        return self


class ExportFormatRead(BaseModel):
    key: str
    content_type: str
    extension: str
