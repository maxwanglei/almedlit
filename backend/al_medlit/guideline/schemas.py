from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GuidelineVersionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: int
    version_label: str = Field(default="v1", min_length=1, max_length=50)
    markdown: str = ""
    author_id: str | None = None
    status: Literal["active", "draft"] = "active"


class GuidelineVersionRead(GuidelineVersionCreate):
    status: Literal["active", "draft", "superseded"]
    id: int

    model_config = {"from_attributes": True}
