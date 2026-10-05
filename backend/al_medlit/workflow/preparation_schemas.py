"""Contracts for previewing and freezing training data from reusable sources."""

from typing import Literal

from pydantic import Field, field_validator, model_validator

from al_medlit.workflow.schemas import InputModel, ReadModel, TrainingDatasetVersionRead


class TrainingSource(InputModel):
    dataset_version_id: int = Field(ge=1)
    input_mapping: dict[str, str] = Field(default_factory=lambda: {"text": "text"})
    label_set_version_id: int | None = Field(default=None, ge=1)
    label_field: str | None = Field(default=None, min_length=1, max_length=255)
    annotation_round_id: int | None = Field(default=None, ge=1)
    submission_ids: list[int] | None = None
    split_map_id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_source(self):
        choices = (self.label_set_version_id, self.label_field, self.annotation_round_id)
        if sum(value is not None for value in choices) != 1:
            raise ValueError("Choose one label field, label set, or submitted annotation round")
        if not self.input_mapping or any(
            not target.strip() or not source.strip()
            for target, source in self.input_mapping.items()
        ):
            raise ValueError("Input mappings must have non-empty source and target fields")
        if self.submission_ids is not None:
            if self.annotation_round_id is None:
                raise ValueError("submission_ids require annotation_round_id")
            if not self.submission_ids or any(value < 1 for value in self.submission_ids):
                raise ValueError("Choose at least one finalized submission")
            if len(set(self.submission_ids)) != len(self.submission_ids):
                raise ValueError("submission_ids must be unique")
        return self


class TrainingPreparationRequest(InputModel):
    name: str = Field(min_length=1, max_length=255)
    task_version_id: int = Field(ge=1)
    sources: list[TrainingSource] = Field(min_length=1, max_length=30)
    training_dataset_id: int | None = Field(default=None, ge=1)
    parent_version_id: int | None = Field(default=None, ge=1)
    train_percent: float = Field(default=80, gt=0, lt=100)
    validation_percent: float = Field(default=10, gt=0, lt=100)
    seed: int = 42
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=160)
    preview_manifest_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Name cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_preparation(self):
        if (self.training_dataset_id is None) != (self.parent_version_id is None):
            raise ValueError("An update requires training_dataset_id and parent_version_id")
        if self.train_percent + self.validation_percent >= 100:
            raise ValueError("Leave a positive percentage for protected test examples")
        version_ids = [source.dataset_version_id for source in self.sources]
        if len(version_ids) != len(set(version_ids)):
            raise ValueError("Choose each source dataset version only once")
        return self


class PreparationIssue(InputModel):
    code: str
    message: str
    source_index: int | None = None
    item_key: str | None = None


class PreparationSourceCounts(InputModel):
    dataset_version_id: int
    total_count: int
    labeled_count: int
    excluded_unlabeled_count: int


class TrainingPreparationPreview(InputModel):
    ready: bool
    issues: list[PreparationIssue]
    source_counts: list[PreparationSourceCounts]
    input_count: int
    labeled_count: int
    excluded_unlabeled_count: int
    duplicate_count: int
    item_count: int
    group_count: int
    split_counts: dict[Literal["train", "validation", "test", "pool"], int]
    manifest_hash: str
    resolved_sources: list[dict]


class PreparedTrainingVersionRead(TrainingDatasetVersionRead):
    training_dataset_id: int | None
    version_number: int
    parent_version_id: int | None
    preparation_manifest: dict


class TrainingDatasetRead(ReadModel):
    project_id: int
    name: str
    task_version_id: int


class TrainingDatasetSeriesRead(TrainingDatasetRead):
    versions: list[PreparedTrainingVersionRead]
    latest_version_id: int | None


class TrainingPreparationResult(InputModel):
    training_dataset: TrainingDatasetRead
    training_dataset_version: PreparedTrainingVersionRead
    preview: TrainingPreparationPreview
