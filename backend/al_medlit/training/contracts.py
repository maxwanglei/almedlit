"""Versioned, UI-safe contracts shared by training model types and tasks.

The objects in this module deliberately contain only data.  A model plugin may
declare which of the supported panel primitives it can populate, but it cannot
send executable frontend code or HTML through a descriptor.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


class ModelFamily(StrEnum):
    CONVENTIONAL_ML = "conventional_ml"
    DEEP_LEARNING = "deep_learning"
    LLM_FINETUNE = "llm_finetune"


class AvailabilityStatus(StrEnum):
    AVAILABLE = "available"
    EXPERIMENTAL = "experimental"
    UNAVAILABLE = "unavailable"


class PanelKind(StrEnum):
    METRIC_CARDS = "metric_cards"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    CONFUSION_MATRIX = "confusion_matrix"
    DISTRIBUTION = "distribution"
    TABLE = "table"


class ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AvailabilityDescriptor(ContractModel):
    status: AvailabilityStatus
    reason: str | None = None
    missing_dependencies: tuple[str, ...] = ()
    synthetic_available: bool = False

    @computed_field
    @property
    def available(self) -> bool:
        return self.status in {
            AvailabilityStatus.AVAILABLE,
            AvailabilityStatus.EXPERIMENTAL,
        }


class ModelCapabilities(ContractModel):
    train: bool = True
    predict: bool = True
    resume: bool = False
    target_conditioning: bool = True
    supported_devices: tuple[str, ...] = ("cpu",)
    supported_compute_backends: tuple[str, ...] = ("local", "ssh_slurm")
    artifact_formats: tuple[str, ...]
    required_worker_dependencies: tuple[str, ...] = ()
    requires_device_preflight: bool = False
    produces_sentence_scores: bool = False
    produces_sequence_labels: bool = True


class MetricDescriptor(ContractModel):
    key: str = Field(min_length=1, max_length=160)
    label: str = Field(min_length=1, max_length=160)
    direction: Literal["maximize", "minimize", "none"] = "maximize"
    value_type: Literal["ratio", "count", "seconds", "bytes", "number"] = "ratio"
    comparable: bool = True
    nullable: bool = True
    description: str = ""


class PanelDescriptor(ContractModel):
    key: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=160)
    kind: PanelKind
    data_keys: tuple[str, ...] = ()
    optional: bool = False

    @field_validator("data_keys")
    @classmethod
    def reject_executable_panel_payloads(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            lowered = value.lower()
            if "<script" in lowered or "javascript:" in lowered or "data:text/html" in lowered:
                raise ValueError("Panel descriptors may reference data keys only")
        return values


class ModelTypeDescriptor(ContractModel):
    schema_version: Literal["model-type-descriptor-v1"] = "model-type-descriptor-v1"
    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1, max_length=160)
    family: ModelFamily
    model_kind: str = Field(min_length=1, max_length=80)
    task_type: str = Field(min_length=1, max_length=120)
    description: str = ""
    implementation_status: Literal["implemented", "experimental", "planned"]
    availability: AvailabilityDescriptor
    capabilities: ModelCapabilities
    config_schema: dict
    metrics: tuple[MetricDescriptor, ...] = ()
    panels: tuple[PanelDescriptor, ...] = ()


class TaskContractDescriptor(ContractModel):
    schema_version: Literal["task-contract-descriptor-v1"] = (
        "task-contract-descriptor-v1"
    )
    key: str = Field(min_length=1, max_length=120, pattern=r"^[a-z][a-z0-9_]*$")
    contract_version: str = Field(min_length=1, max_length=40)
    label: str = Field(min_length=1, max_length=160)
    task_type: str = Field(min_length=1, max_length=120)
    prediction_schema_version: str = Field(min_length=1, max_length=120)
    evaluator_version: str = Field(min_length=1, max_length=120)
    metric_suite_version: str = Field(min_length=1, max_length=120)
    selection_metric: str = Field(min_length=1, max_length=160)
    selection_tiebreakers: tuple[str, ...] = ()
    metrics: tuple[MetricDescriptor, ...]


@runtime_checkable
class ModelTypePlugin(Protocol):
    key: str
    task_type: str

    @property
    def descriptor(self) -> ModelTypeDescriptor: ...

    def validate_config(self, config: dict): ...

    def build_model(self, config: dict): ...

    def build_dataset(self, *args, **kwargs): ...

    def train(self, *args, **kwargs): ...

    def fit(self, *args, **kwargs): ...

    def evaluate(self, *args, **kwargs): ...

    def predict(self, *args, **kwargs): ...

    def package(self, *args, **kwargs): ...

    def load(self, *args, **kwargs): ...


@runtime_checkable
class TaskEvaluator(Protocol):
    descriptor: TaskContractDescriptor

    def evaluate(self, *args, **kwargs): ...
