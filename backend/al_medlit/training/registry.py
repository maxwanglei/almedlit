from al_medlit.core.exceptions import ValidationError
from al_medlit.training.contracts import ModelTypeDescriptor, ModelTypePlugin
from al_medlit.training.model_types.evidence_conventional.plugin import (
    register_builtin_conventional_model_types,
)
from al_medlit.training.model_types.evidence_neural.plugin import (
    register_builtin_neural_model_types,
)
from al_medlit.training.model_types.evidence_peft.plugin import (
    register_builtin_peft_model_types,
)


class ModelTypeRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, ModelTypePlugin] = {}
        self._descriptors: dict[str, ModelTypeDescriptor] = {}

    def register(self, plugin: ModelTypePlugin, *, replace: bool = False) -> None:
        if not plugin.key:
            raise ValidationError("Model type key cannot be empty")
        if plugin.key in self._plugins and not replace:
            raise ValidationError(f"Model type '{plugin.key}' is already registered")
        descriptor = getattr(plugin, "descriptor", None)
        if descriptor is not None:
            if descriptor.key != plugin.key or descriptor.task_type != plugin.task_type:
                raise ValidationError(
                    "Model plugin descriptor key and task type must match the plugin"
                )
        self._plugins[plugin.key] = plugin
        if descriptor is not None:
            self._descriptors[plugin.key] = descriptor

    def register_descriptor(
        self,
        descriptor: ModelTypeDescriptor,
        *,
        replace: bool = False,
    ) -> None:
        if descriptor.key in self._descriptors and not replace:
            raise ValidationError(
                f"Model type descriptor '{descriptor.key}' is already registered"
            )
        self._descriptors[descriptor.key] = descriptor

    def get(self, key: str) -> ModelTypePlugin:
        try:
            return self._plugins[key]
        except KeyError as exc:
            raise ValidationError(f"Unknown model type '{key}'") from exc

    def list(self) -> tuple[ModelTypePlugin, ...]:
        return tuple(self._plugins[key] for key in sorted(self._plugins))

    def get_descriptor(self, key: str) -> ModelTypeDescriptor:
        try:
            return self._descriptors[key]
        except KeyError as exc:
            raise ValidationError(f"Unknown model type descriptor '{key}'") from exc

    def list_descriptors(self) -> tuple[ModelTypeDescriptor, ...]:
        return tuple(self._descriptors[key] for key in sorted(self._descriptors))

    def register_builtin_descriptors(self) -> None:
        from al_medlit.training.model_types.catalog import builtin_model_descriptors

        for descriptor in builtin_model_descriptors():
            self.register_descriptor(descriptor, replace=True)


model_types = ModelTypeRegistry()
model_types.register_builtin_descriptors()

# These adapters import their optional ML dependencies only when a worker fits
# or loads a model, so registering them is safe in the API process.
register_builtin_conventional_model_types(model_types)
register_builtin_neural_model_types(model_types)
register_builtin_peft_model_types(model_types)
