"""Stable facade for canonical workflow domain services."""

from types import ModuleType
from typing import Any

from al_medlit.workflow.services import (
    common,
    composition,
    cycles,
    datasets,
    feedback,
    feedback_scoring,
    guidelines,
    labels,
    models,
    rounds,
    selection,
    tasks,
    training_runs,
    training_setup,
    workspace_training,
)

_DOMAIN_MODULES = (
    common,
    tasks,
    datasets,
    labels,
    composition,
    models,
    cycles,
    selection,
    training_setup,
    training_runs,
    workspace_training,
    rounds,
    feedback,
    feedback_scoring,
    guidelines,
)


def _domain_exports(module: ModuleType) -> dict[str, Any]:
    return {
        name: value
        for name, value in vars(module).items()
        if callable(value) and getattr(value, "__module__", None) == module.__name__
    }


for _domain_module in _DOMAIN_MODULES:
    globals().update(_domain_exports(_domain_module))

__all__ = sorted(
    name
    for name, value in globals().items()
    if callable(value)
    and name not in {"Any", "ModuleType", "_domain_exports"}
    and not name.startswith("__")
)
