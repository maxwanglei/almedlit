from pydantic import BaseModel, Field, model_validator


class ExportCreate(BaseModel):
    format_key: str = Field(min_length=1, max_length=100)
    annotation_set_id: int | None = None
    corpus_snapshot_id: int | None = None
    file_name: str | None = Field(default=None, max_length=255)
    options: dict = Field(default_factory=dict)
    metadata_: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def has_source(self):
        if self.annotation_set_id is None and self.corpus_snapshot_id is None:
            raise ValueError("annotation_set_id or corpus_snapshot_id is required")
        return self


class ExportFormatRead(BaseModel):
    key: str
    content_type: str
    extension: str
