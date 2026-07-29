"""LoRA/QLoRA plugin scaffolding with mandatory compute preflight."""

from __future__ import annotations

from pathlib import Path

from al_medlit.training.model_types.evidence_peft.adapter import (
    PeftAdapterBundle,
    create_peft_adapter,
    load_peft_adapter,
)
from al_medlit.training.model_types.evidence_peft.config import (
    EvidenceLoRAConfig,
    EvidenceQLoRAConfig,
    ImmutableBaseModelReference,
)
from al_medlit.training.model_types.evidence_peft.preflight import (
    preflight_peft,
    require_peft_preflight,
)


class _EvidencePeftPlugin:
    task_type = "evidence_block_sentence_tagging"
    task_contract_key = "evidence_blocks"
    task_contract_version = "1"
    config_class = EvidenceLoRAConfig

    @property
    def descriptor(self):
        from al_medlit.training.model_types.catalog import get_builtin_model_descriptor

        return get_builtin_model_descriptor(self.key)

    @property
    def family(self) -> str:
        return self.descriptor.family.value

    def validate_config(self, config: dict):
        return self.config_class.model_validate(config)

    def preflight(self, config: dict):
        return preflight_peft(self.validate_config(config))

    def build_model(
        self,
        config: dict,
        *,
        base_model: object,
        base_reference: ImmutableBaseModelReference,
    ) -> PeftAdapterBundle:
        return create_peft_adapter(
            base_model,
            self.validate_config(config),
            base_reference,
        )

    def build_dataset(self, preparer, *args, **kwargs):
        """Use the exact tokenizer supplied by the immutable base package."""

        return preparer.prepare(*args, **kwargs)

    def train(self, trainer, *, config: dict, **kwargs):
        validated = self.validate_config(config)
        require_peft_preflight(validated)
        return trainer.train(**kwargs)

    fit = train

    def evaluate(self, evaluator, *args, **kwargs):
        return evaluator.evaluate(*args, **kwargs)

    def predict(self, predictor, *args, **kwargs):
        return predictor.predict(*args, **kwargs)

    def package(self, bundle: PeftAdapterBundle, destination: str | Path):
        return bundle.save_pretrained(destination)

    def load(
        self,
        checkpoint_directory: str | Path,
        *,
        base_model: object,
        base_reference: ImmutableBaseModelReference,
        trainable: bool = False,
    ) -> PeftAdapterBundle:
        return load_peft_adapter(
            checkpoint_directory,
            base_model=base_model,
            base_reference=base_reference,
            trainable=trainable,
        )


class EvidenceLoRAPlugin(_EvidencePeftPlugin):
    key = "evidence_lora"
    config_class = EvidenceLoRAConfig


class EvidenceQLoRAPlugin(_EvidencePeftPlugin):
    key = "evidence_qlora"
    config_class = EvidenceQLoRAConfig


def register_builtin_peft_model_types(registry=None) -> None:
    """Opt-in registration hook; call only after runner support is enabled."""

    from al_medlit.core.exceptions import ValidationError
    from al_medlit.training.registry import model_types

    selected_registry = registry or model_types
    for plugin in (EvidenceLoRAPlugin(), EvidenceQLoRAPlugin()):
        try:
            selected_registry.register(plugin)
        except ValidationError as exc:
            if "already registered" not in exc.message:
                raise
