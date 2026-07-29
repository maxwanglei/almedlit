from typing import Protocol

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.compute.base import CommandRunner, ComputeBackend
from al_medlit.training.compute.config import (
    LocalComputeConfig,
    SSHSlurmProfileConfig,
)
from al_medlit.training.compute.local import LocalComputeBackend
from al_medlit.training.compute.slurm import SSHSlurmComputeBackend


class ComputeBackendPlugin(Protocol):
    key: str

    def validate_config(self, config: dict) -> dict: ...

    def build(self, config: dict, runner: CommandRunner | None = None) -> ComputeBackend: ...


class LocalComputePlugin:
    key = "local"

    def validate_config(self, config: dict) -> dict:
        return LocalComputeConfig.model_validate({**config, "backend": self.key}).model_dump(
            mode="json",
            exclude={"backend"},
        )

    def build(self, config: dict, runner: CommandRunner | None = None):
        validated = LocalComputeConfig.model_validate({**config, "backend": self.key})
        return LocalComputeBackend(
            runner,
            runtime_root=validated.runtime_root,
            synchronous=validated.synchronous,
            stale_heartbeat_seconds=validated.stale_heartbeat_seconds,
        )


class SSHSlurmComputePlugin:
    key = "ssh_slurm"

    def validate_config(self, config: dict) -> dict:
        validated = SSHSlurmProfileConfig.model_validate({**config, "backend": self.key})
        validated.to_runtime_config()
        return validated.model_dump(mode="json", exclude={"backend"})

    def build(self, config: dict, runner: CommandRunner | None = None):
        validated = SSHSlurmProfileConfig.model_validate({**config, "backend": self.key})
        return SSHSlurmComputeBackend(validated.to_runtime_config(), runner)


class ComputeBackendRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ComputeBackendPlugin] = {}

    def register(self, plugin: ComputeBackendPlugin, *, replace: bool = False) -> None:
        if plugin.key in self._plugins and not replace:
            raise ValidationError(f"Compute backend '{plugin.key}' is already registered")
        self._plugins[plugin.key] = plugin

    def get(self, key: str) -> ComputeBackendPlugin:
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise ValidationError(f"Unknown compute backend '{key}'") from exc

    def list(self) -> tuple[ComputeBackendPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))


compute_backends = ComputeBackendRegistry()


def register_builtin_compute_backends() -> None:
    for plugin in (LocalComputePlugin(), SSHSlurmComputePlugin()):
        try:
            compute_backends.register(plugin)
        except ValidationError as exc:
            if "already registered" not in exc.message:
                raise
