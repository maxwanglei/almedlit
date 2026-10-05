"""Standalone prediction contracts, independent of active-learning selection."""

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from .schemas import InputModel, ReadModel


class PredictionRunCreate(InputModel):
    name: str = Field(min_length=1, max_length=255)
    dataset_version_id: int = Field(gt=0)
    model_version_id: int = Field(gt=0)
    request_key: str = Field(min_length=8, max_length=120)

    @field_validator("name", "request_key")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value.strip()


class PredictionRunRead(ReadModel):
    project_id: int
    name: str
    dataset_version_id: int
    task_version_id: int
    model_version_id: int
    status: Literal["planned", "queued", "running", "completed", "failed"]
    result_count: int
    failure_reason: str | None
    completed_at: datetime | None


class PredictionResult(InputModel):
    dataset_item_id: int
    stable_key: str
    title: str
    text: str
    prediction: Any
    confidence: float | None
    uncertainty: float | None
    already_submitted: bool
    protected: bool


class PredictionResultPage(InputModel):
    items: list[PredictionResult]
    total: int
    offset: int
    limit: int


class PredictionReviewCreate(InputModel):
    name: str = Field(min_length=1, max_length=255)
    request_key: str = Field(min_length=8, max_length=120)
    dataset_item_ids: list[int] = Field(min_length=1, max_length=5000)
    include_submitted: bool = False

    @field_validator("name", "request_key")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank")
        return value.strip()

    @field_validator("dataset_item_ids")
    @classmethod
    def unique_positive(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)) or any(item <= 0 for item in value):
            raise ValueError("Select unique positive dataset item IDs")
        return value


class PredictionReviewRead(InputModel):
    round_id: int
    item_count: int
    excluded_submitted: int
    excluded_protected: int
