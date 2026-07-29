"""Worker-only contracts for trusted post-training evaluators."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.core.exceptions import ValidationError


class EvaluatorValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EvaluationInput(EvaluatorValue):
    rows: tuple[dict, ...]
    split_name: str = Field(min_length=1, max_length=40)
    protected_split: bool
    requested_metrics: tuple[str, ...] = ()
    dataset_fingerprint: str = Field(min_length=1, max_length=128)
    training_dataset_fingerprint: str = Field(min_length=1, max_length=128)
    split_fingerprint: str = Field(min_length=1, max_length=128)
    model_fingerprint: str = Field(min_length=1, max_length=128)
    task_kind: str = Field(min_length=1, max_length=80)
    task_schema: dict = Field(default_factory=dict)
    label_vocabulary: tuple[str, ...] = ()


class EvaluationOutput(EvaluatorValue):
    metrics: dict[str, float | None]
    prediction_count: int = Field(ge=0)
    report: dict = Field(default_factory=dict)


@runtime_checkable
class EvaluatorPlugin(Protocol):
    key: str
    evaluator_version: str
    recipe_keys: tuple[str, ...]

    def evaluate(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        evaluation_input: EvaluationInput,
        model_directory: Path,
    ) -> EvaluationOutput: ...


class EvaluatorPluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, EvaluatorPlugin] = {}
        self._recipe_owners: dict[str, str] = {}

    def register(
        self,
        plugin: EvaluatorPlugin,
        *,
        replace: bool = False,
    ) -> None:
        if not plugin.key:
            raise ValidationError("Evaluator plugin key cannot be empty")
        if not plugin.evaluator_version:
            raise ValidationError("Evaluator plugin version cannot be empty")
        if not plugin.recipe_keys:
            raise ValidationError("Evaluator plugin must support at least one recipe")
        if plugin.key in self._plugins and not replace:
            raise ValidationError(
                f"Evaluator plugin '{plugin.key}' is already registered"
            )
        for recipe_key in plugin.recipe_keys:
            owner = self._recipe_owners.get(recipe_key)
            if owner is not None and owner != plugin.key and not replace:
                raise ValidationError(
                    f"Recipe '{recipe_key}' already has evaluator '{owner}'"
                )
        prior = self._plugins.get(plugin.key)
        if prior is not None:
            for recipe_key in prior.recipe_keys:
                if self._recipe_owners.get(recipe_key) == plugin.key:
                    self._recipe_owners.pop(recipe_key)
        self._plugins[plugin.key] = plugin
        for recipe_key in plugin.recipe_keys:
            self._recipe_owners[recipe_key] = plugin.key

    def find_for_recipe(self, recipe_key: str) -> EvaluatorPlugin | None:
        plugin_key = self._recipe_owners.get(recipe_key)
        return self._plugins.get(plugin_key) if plugin_key is not None else None

    def list(self) -> tuple[EvaluatorPlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))


evaluator_plugins = EvaluatorPluginRegistry()
