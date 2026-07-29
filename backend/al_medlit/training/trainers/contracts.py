"""Worker-facing plugin boundary for versioned training recipes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.core.exceptions import ValidationError


class TrainerValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrainerPreflight(TrainerValue):
    ready: bool
    runtime_class: str
    checks: tuple[dict, ...]


class TrainingInput(TrainerValue):
    rows: tuple[dict, ...]
    validation_rows: tuple[dict, ...] = ()
    dataset_fingerprint: str = Field(min_length=1, max_length=128)
    split_fingerprint: str = Field(min_length=1, max_length=128)
    task_kind: str | None = Field(default=None, min_length=1, max_length=80)
    task_schema: dict = Field(default_factory=dict)
    label_vocabulary: tuple[str, ...] = ()
    base_model_path: str | None = Field(default=None, min_length=1, max_length=4096)
    base_model_fingerprint: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )


class TrainingPlan(TrainerValue):
    manifest: dict
    normalized_config: dict


class TrainingOutput(TrainerValue):
    manifest: dict
    validation_metrics: dict[str, float | None]
    artifact_paths: tuple[str, ...]


@runtime_checkable
class TrainerPlugin(Protocol):
    key: str
    recipe_keys: tuple[str, ...]

    def preflight(self, recipe_key: str | None = None) -> TrainerPreflight: ...

    def plan(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan: ...

    def train(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        destination: Path,
        seed: int,
    ) -> TrainingOutput: ...


class TrainerPluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, TrainerPlugin] = {}

    def register(self, plugin: TrainerPlugin, *, replace: bool = False) -> None:
        if not plugin.key:
            raise ValidationError("Trainer plugin key cannot be empty")
        if not plugin.recipe_keys:
            raise ValidationError("Trainer plugin must support at least one recipe")
        if plugin.key in self._plugins and not replace:
            raise ValidationError(f"Trainer plugin '{plugin.key}' is already registered")
        self._plugins[plugin.key] = plugin

    def get(self, key: str) -> TrainerPlugin:
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise ValidationError(f"Trainer plugin '{key}' is not installed") from exc

    def list(self) -> tuple[TrainerPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    def require_recipe(self, trainer_key: str, recipe_key: str) -> TrainerPlugin:
        plugin = self.get(trainer_key)
        if recipe_key not in plugin.recipe_keys:
            raise ValidationError(
                f"Trainer plugin '{trainer_key}' does not support recipe '{recipe_key}'"
            )
        return plugin


trainer_plugins = TrainerPluginRegistry()
