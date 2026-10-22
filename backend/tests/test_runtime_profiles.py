from datetime import UTC, datetime

import pytest

from al_medlit.training.compute.config import SSHSlurmProfileConfig
from al_medlit.training.runtime_profiles import (
    RUNTIME_PROFILES,
    RuntimeReadinessReport,
    runtime_profile_report_sha256,
    runtime_queue_for_model_type,
)


def _ready_fragment(profile_key: str) -> dict:
    descriptor = RUNTIME_PROFILES[profile_key]
    device = {
        "cpu": "cpu",
        "accelerator": "cuda",
        "cuda": "cuda",
    }[descriptor.required_device]
    capabilities = ["cuda"] if device == "cuda" else []
    if profile_key == "qlora-cuda":
        capabilities.append("qlora_4bit")
    report = RuntimeReadinessReport(
        runtime_profile=profile_key,
        generated_at=datetime(2026, 7, 27, tzinfo=UTC),
        python_version="3.12.11",
        worker_image_digest="a" * 64,
        dependency_versions={dependency: "1.0.0" for dependency in descriptor.required_imports},
        device=device,
        device_available=True,
        device_memory_bytes=16 * 1024**3,
        scratch_path="/var/lib/al-medlit/attempts",
        scratch_available_bytes=descriptor.minimum_scratch_bytes,
        storage_access_verified=True,
        ready=True,
    )
    return {
        "host": "cluster.example.org",
        "username": "alice",
        "known_hosts_file": "/tmp/known_hosts",
        "remote_root": "/remote/jobs",
        "resources": {"account": "test"},
        "apptainer_image": "/images/runtime.sif",
        "apptainer_module": "apptainer",
        "runtime_profile": profile_key,
        "verified_dependencies": list(descriptor.required_imports),
        "verified_capabilities": capabilities,
        "worker_image_digest": report.worker_image_digest,
        "capability_report_sha256": runtime_profile_report_sha256(report),
        "readiness_report": report.model_dump(mode="json"),
    }


def test_named_runtime_requires_an_untampered_image_bound_report():
    fragment = _ready_fragment("classical-cpu")
    normalized = SSHSlurmProfileConfig.model_validate(fragment).model_dump(mode="json")

    assert normalized["runtime_profile"] == "classical-cpu"
    assert normalized["readiness_report"]["storage_access_verified"] is True

    fragment["readiness_report"]["scratch_available_bytes"] += 1
    with pytest.raises(ValueError, match="does not match readiness_report"):
        SSHSlurmProfileConfig.model_validate(fragment)


def test_named_runtime_rejects_unverified_storage():
    fragment = _ready_fragment("transformer-cpu")
    fragment["readiness_report"]["storage_access_verified"] = False
    fragment["readiness_report"]["ready"] = False
    fragment["capability_report_sha256"] = runtime_profile_report_sha256(
        fragment["readiness_report"]
    )

    with pytest.raises(ValueError, match="must be ready"):
        SSHSlurmProfileConfig.model_validate(fragment)


def test_named_runtime_rejects_ready_claim_without_measured_memory():
    fragment = _ready_fragment("classical-cpu")
    fragment["readiness_report"]["device_memory_bytes"] = None
    fragment["capability_report_sha256"] = runtime_profile_report_sha256(
        fragment["readiness_report"]
    )

    with pytest.raises(ValueError, match="measure available device memory"):
        SSHSlurmProfileConfig.model_validate(fragment)


def test_model_types_route_to_isolated_runtime_queues():
    assert runtime_queue_for_model_type("evidence_svm") == "classical-cpu"
    assert runtime_queue_for_model_type("evidence_bilstm") == "torch-cpu"
    assert runtime_queue_for_model_type("evidence_block_sentence_tagger") == "transformer-cpu"
    assert runtime_queue_for_model_type("evidence_lora") == "peft-accelerator"
    assert runtime_queue_for_model_type("evidence_qlora") == "qlora-cuda"
    assert runtime_queue_for_model_type("transformer_token_classification") == "transformer-cpu"
    assert runtime_queue_for_model_type("causal_lm_sft_lora") == "peft-accelerator"
    assert runtime_queue_for_model_type("seq2seq_lm_sft_qlora") == "qlora-cuda"
