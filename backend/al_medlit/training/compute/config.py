import re
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from al_medlit.core.config import settings
from al_medlit.training.compute.slurm import SlurmResources, SSHSlurmConfig
from al_medlit.training.runtime_profiles import (
    RUNTIME_PROFILES,
    RuntimeProfileKey,
    RuntimeReadinessReport,
    runtime_profile_report_sha256,
    validate_ready_runtime_report,
)


class LocalComputeConfig(BaseModel):
    backend: Literal["local"] = "local"
    runtime_root: Path = Field(
        default_factory=lambda: Path(settings.local_attempt_root)
    )
    synchronous: bool = False
    stale_heartbeat_seconds: int = Field(default=180, ge=30, le=3600)
    verified_capabilities: tuple[Literal["cuda", "ascend", "qlora_4bit"], ...] = ()
    worker_image_digest: str | None = None
    capability_report_sha256: str | None = None
    runtime_profile: RuntimeProfileKey = "auto"
    verified_dependencies: tuple[str, ...] = ()
    readiness_report: RuntimeReadinessReport | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_capability_attestation(self):
        _validate_capability_attestation(self)
        return self


class SlurmResourcesConfig(BaseModel):
    account: str = Field(min_length=1)
    partition: str | None = None
    qos: str | None = None
    time_limit: str = "01:00:00"
    nodes: int = Field(default=1, gt=0)
    tasks_per_node: int = Field(default=1, gt=0)
    cpus_per_task: int = Field(default=4, gt=0)
    memory: str = "32G"
    gres: str | None = "gpu:1"

    model_config = {"extra": "forbid"}


class SSHSlurmProfileConfig(BaseModel):
    backend: Literal["ssh_slurm"] = "ssh_slurm"
    site_profile: Literal["generic", "osc_ascend"] = "generic"
    host: str
    username: str
    known_hosts_file: Path
    remote_root: PurePosixPath
    resources: SlurmResourcesConfig
    apptainer_image: PurePosixPath
    apptainer_module: str
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_identity_file: Path | None = None
    apptainer_args: tuple[str, ...] = ("--nv",)
    module_prelude: tuple[str, ...] = ()
    verified_capabilities: tuple[Literal["cuda", "ascend", "qlora_4bit"], ...] = ()
    worker_image_digest: str | None = None
    capability_report_sha256: str | None = None
    runtime_profile: RuntimeProfileKey = "auto"
    verified_dependencies: tuple[str, ...] = ()
    readiness_report: RuntimeReadinessReport | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_ascend_host(self):
        if self.site_profile == "osc_ascend" and self.host != "ascend.osc.edu":
            raise ValueError("The osc_ascend profile must use ascend.osc.edu")
        _validate_capability_attestation(self)
        return self

    def to_runtime_config(self) -> SSHSlurmConfig:
        resources = SlurmResources(**self.resources.model_dump())
        return SSHSlurmConfig(
            host=self.host,
            username=self.username,
            known_hosts_file=self.known_hosts_file,
            remote_root=self.remote_root,
            resources=resources,
            apptainer_image=self.apptainer_image,
            apptainer_module=self.apptainer_module,
            ssh_port=self.ssh_port,
            ssh_identity_file=self.ssh_identity_file,
            apptainer_args=self.apptainer_args,
            module_prelude=self.module_prelude,
        )


def validate_compute_config(backend: str, config: dict) -> dict:
    payload = {**config, "backend": backend}
    if backend == "local":
        validated = LocalComputeConfig.model_validate(payload)
    elif backend == "ssh_slurm":
        validated = SSHSlurmProfileConfig.model_validate(payload)
        validated.to_runtime_config()
    else:
        raise ValueError(f"Unknown compute backend '{backend}'")
    return validated.model_dump(mode="json", exclude={"backend"})


def _validate_capability_attestation(config) -> None:
    capabilities = tuple(config.verified_capabilities)
    if len(capabilities) != len(set(capabilities)):
        raise ValueError("verified_capabilities must be unique")
    if "qlora_4bit" in capabilities and "cuda" not in capabilities:
        raise ValueError("qlora_4bit attestation also requires verified CUDA")
    if capabilities:
        for value, label in (
            (config.worker_image_digest, "worker_image_digest"),
            (config.capability_report_sha256, "capability_report_sha256"),
        ):
            if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 for verified capabilities")

    dependencies = tuple(config.verified_dependencies)
    if len(dependencies) != len(set(dependencies)):
        raise ValueError("verified_dependencies must be unique")
    if config.runtime_profile == "auto":
        if config.readiness_report is not None:
            raise ValueError("auto runtime profiles cannot carry an image readiness report")
        return

    descriptor = RUNTIME_PROFILES[config.runtime_profile]
    for value, label in (
        (config.worker_image_digest, "worker_image_digest"),
        (config.capability_report_sha256, "capability_report_sha256"),
    ):
        if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} must be a lowercase SHA-256 for named runtimes")
    report = config.readiness_report
    if report is None:
        raise ValueError("named runtime profiles require a readiness_report")
    validate_ready_runtime_report(report)
    if report.runtime_profile != config.runtime_profile:
        raise ValueError("readiness_report runtime_profile does not match the compute profile")
    if not report.ready:
        raise ValueError("readiness_report must be ready")
    if report.missing_dependencies:
        raise ValueError("readiness_report contains missing dependencies")
    if not report.storage_access_verified:
        raise ValueError("readiness_report must verify object storage access")
    if report.scratch_available_bytes < descriptor.minimum_scratch_bytes:
        raise ValueError(
            f"{config.runtime_profile} requires at least "
            f"{descriptor.minimum_scratch_bytes} scratch bytes"
        )
    if set(report.dependency_versions) != set(descriptor.required_imports):
        raise ValueError("readiness_report dependencies do not match the runtime profile")
    if set(dependencies) != set(descriptor.required_imports):
        raise ValueError("verified_dependencies do not match the runtime profile")
    if not report.device_available:
        raise ValueError("readiness_report must verify its required device")
    if descriptor.required_device == "cpu" and report.device != "cpu":
        raise ValueError("CPU runtime profiles require a CPU readiness report")
    if descriptor.required_device == "cuda" and report.device != "cuda":
        raise ValueError("QLoRA runtime profiles require a CUDA readiness report")
    if descriptor.required_device == "accelerator" and report.device not in {
        "cuda",
        "ascend",
    }:
        raise ValueError("accelerator runtime profiles require CUDA or Ascend")
    if report.worker_image_digest != config.worker_image_digest:
        raise ValueError("readiness_report is not bound to worker_image_digest")
    if runtime_profile_report_sha256(report) != config.capability_report_sha256:
        raise ValueError("capability_report_sha256 does not match readiness_report")
