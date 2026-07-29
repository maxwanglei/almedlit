"""Declarative, image-bound runtime profiles for optional training stacks."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

RuntimeProfileKey = Literal[
    "auto",
    "classical-cpu",
    "torch-cpu",
    "transformer-cpu",
    "peft-accelerator",
    "qlora-cuda",
]


@dataclass(frozen=True, slots=True)
class RuntimeProfileDescriptor:
    key: str
    package_extra: str
    queue: str
    required_imports: tuple[str, ...]
    distributions: dict[str, str]
    required_device: str
    minimum_scratch_bytes: int = 1024 * 1024 * 1024


RUNTIME_PROFILES: dict[str, RuntimeProfileDescriptor] = {
    profile.key: profile
    for profile in (
        RuntimeProfileDescriptor(
            key="classical-cpu",
            package_extra="runtime-classical-cpu",
            queue="classical-cpu",
            required_imports=("sklearn", "sklearn_crfsuite", "skops"),
            distributions={
                "sklearn": "scikit-learn",
                "sklearn_crfsuite": "sklearn-crfsuite",
                "skops": "skops",
            },
            required_device="cpu",
        ),
        RuntimeProfileDescriptor(
            key="torch-cpu",
            package_extra="runtime-torch-cpu",
            queue="torch-cpu",
            required_imports=("torch", "safetensors"),
            distributions={"torch": "torch", "safetensors": "safetensors"},
            required_device="cpu",
        ),
        RuntimeProfileDescriptor(
            key="transformer-cpu",
            package_extra="runtime-transformer-cpu",
            queue="transformer-cpu",
            required_imports=("torch", "transformers", "safetensors"),
            distributions={
                "torch": "torch",
                "transformers": "transformers",
                "safetensors": "safetensors",
            },
            required_device="cpu",
        ),
        RuntimeProfileDescriptor(
            key="peft-accelerator",
            package_extra="runtime-peft-accelerator",
            queue="peft-accelerator",
            required_imports=(
                "torch",
                "transformers",
                "safetensors",
                "accelerate",
                "peft",
            ),
            distributions={
                "torch": "torch",
                "transformers": "transformers",
                "safetensors": "safetensors",
                "accelerate": "accelerate",
                "peft": "peft",
            },
            required_device="accelerator",
        ),
        RuntimeProfileDescriptor(
            key="qlora-cuda",
            package_extra="runtime-qlora-cuda",
            queue="qlora-cuda",
            required_imports=(
                "torch",
                "transformers",
                "safetensors",
                "accelerate",
                "peft",
                "bitsandbytes",
            ),
            distributions={
                "torch": "torch",
                "transformers": "transformers",
                "safetensors": "safetensors",
                "accelerate": "accelerate",
                "peft": "peft",
                "bitsandbytes": "bitsandbytes",
            },
            required_device="cuda",
        ),
    )
}

MODEL_RUNTIME_PROFILES: dict[str, str] = {
    "tfidf_linear_regression": "classical-cpu",
    "tfidf_logistic_regression": "classical-cpu",
    "transformer_sequence_classification": "transformer-cpu",
    "transformer_token_classification": "transformer-cpu",
    "transformer_span_extraction": "transformer-cpu",
    "causal_lm_sft_full": "peft-accelerator",
    "causal_lm_sft_lora": "peft-accelerator",
    "causal_lm_sft_qlora": "qlora-cuda",
    "seq2seq_lm_sft_full": "peft-accelerator",
    "seq2seq_lm_sft_lora": "peft-accelerator",
    "seq2seq_lm_sft_qlora": "qlora-cuda",
    "evidence_crf": "classical-cpu",
    "evidence_svm": "classical-cpu",
    "evidence_random_forest": "classical-cpu",
    "evidence_bilstm": "torch-cpu",
    "evidence_cnn": "torch-cpu",
    "evidence_block_sentence_tagger": "transformer-cpu",
    "evidence_lora": "peft-accelerator",
    "evidence_qlora": "qlora-cuda",
}


class RuntimeReadinessReport(BaseModel):
    schema_version: Literal["runtime-readiness-v1"] = "runtime-readiness-v1"
    runtime_profile: RuntimeProfileKey
    generated_at: datetime
    python_version: str = Field(min_length=1)
    worker_image_digest: str | None = Field(
        default=None,
        pattern=r"^(?:sha256:)?[0-9a-fA-F]{64}$",
    )
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    missing_dependencies: list[str] = Field(default_factory=list)
    device: str
    device_available: bool
    device_memory_bytes: int | None = Field(default=None, ge=0)
    scratch_path: str = Field(min_length=1)
    scratch_available_bytes: int = Field(ge=0)
    storage_access_verified: bool
    ready: bool

    model_config = {"extra": "forbid"}


def validate_ready_runtime_report(report: RuntimeReadinessReport) -> None:
    """Reject a claimed-ready report unless every profile invariant is attested."""

    if report.runtime_profile == "auto":
        raise ValueError("Auto runtimes cannot be marked as verified")
    descriptor = RUNTIME_PROFILES[report.runtime_profile]
    if not report.ready:
        raise ValueError("Runtime readiness report must be ready")
    if report.missing_dependencies:
        raise ValueError("Runtime readiness report contains missing dependencies")
    if set(report.dependency_versions) != set(descriptor.required_imports):
        raise ValueError(
            "Runtime readiness dependencies do not match the selected profile"
        )
    if any(not str(version).strip() for version in report.dependency_versions.values()):
        raise ValueError("Runtime readiness dependency versions cannot be empty")
    if not report.storage_access_verified:
        raise ValueError("Runtime readiness report did not verify object storage")
    if report.scratch_available_bytes < descriptor.minimum_scratch_bytes:
        raise ValueError("Runtime readiness scratch capacity is below the profile minimum")
    if not report.device_available:
        raise ValueError("Runtime readiness report did not verify its required device")
    allowed_devices = {
        "cpu": {"cpu"},
        "cuda": {"cuda"},
        "accelerator": {"cuda", "ascend"},
    }[descriptor.required_device]
    if report.device not in allowed_devices:
        raise ValueError("Runtime readiness device does not match the selected profile")
    if not report.device_memory_bytes:
        raise ValueError("Runtime readiness report must measure available device memory")
    if report.worker_image_digest is None or re.fullmatch(
        r"(?:sha256:)?[0-9a-fA-F]{64}",
        report.worker_image_digest,
    ) is None:
        raise ValueError("Runtime readiness report must be bound to an image digest")


def runtime_profile_for_model_type(model_type: str) -> RuntimeProfileDescriptor:
    profile_key = MODEL_RUNTIME_PROFILES.get(model_type)
    if profile_key is None:
        raise KeyError(f"Model type '{model_type}' has no runtime profile")
    return RUNTIME_PROFILES[profile_key]


def runtime_profile_for_environment_class(
    environment_class: str,
) -> RuntimeProfileDescriptor:
    try:
        return RUNTIME_PROFILES[environment_class]
    except KeyError as exc:
        raise KeyError(
            f"Execution environment class '{environment_class}' has no worker queue"
        ) from exc


def runtime_profile_report_sha256(report: RuntimeReadinessReport | dict) -> str:
    normalized = (
        report.model_dump(mode="json")
        if isinstance(report, RuntimeReadinessReport)
        else RuntimeReadinessReport.model_validate(report).model_dump(mode="json")
    )
    payload = json.dumps(
        normalized,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def runtime_queue_for_model_type(model_type: str) -> str:
    return runtime_profile_for_model_type(model_type).queue


def runtime_queue_for_environment_class(environment_class: str) -> str:
    return runtime_profile_for_environment_class(environment_class).queue
