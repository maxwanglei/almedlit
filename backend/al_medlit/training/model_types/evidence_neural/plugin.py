"""Model-plugin adapters for the BiLSTM and CNN implementations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from al_medlit.training.model_types.evidence_neural.config import (
    EvidenceBiLSTMConfig,
    EvidenceCNNConfig,
)
from al_medlit.training.model_types.evidence_neural.data import EvidenceTextDocument
from al_medlit.training.model_types.evidence_neural.device import preflight_neural_device
from al_medlit.training.model_types.evidence_neural.model import (
    EvidenceNeuralBundle,
    build_neural_bundle,
    fit_neural_model,
    load_neural_bundle,
    predict_neural_model,
)
from al_medlit.training.model_types.evidence_neural.vocabulary import (
    EvidenceVocabulary,
    build_vocabulary,
)


class _EvidenceNeuralPlugin:
    task_type = "evidence_block_sentence_tagging"
    task_contract_key = "evidence_blocks"
    task_contract_version = "1"
    config_class = EvidenceBiLSTMConfig

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
        validated = self.validate_config(config)
        return preflight_neural_device(validated.device)

    def build_dataset(
        self,
        documents: Sequence[EvidenceTextDocument],
        config: dict,
    ) -> EvidenceVocabulary:
        return build_vocabulary(documents, self.validate_config(config))

    def build_model(
        self,
        config: dict,
        *,
        vocabulary: EvidenceVocabulary,
    ) -> EvidenceNeuralBundle:
        return build_neural_bundle(self.validate_config(config), vocabulary)

    def fit(
        self,
        training_documents: Sequence[EvidenceTextDocument],
        *,
        config: dict,
        validation_documents: Sequence[EvidenceTextDocument] = (),
        vocabulary: EvidenceVocabulary | None = None,
        validation_selector=None,
    ):
        return fit_neural_model(
            self.validate_config(config),
            training_documents,
            validation_documents=validation_documents,
            vocabulary=vocabulary,
            validation_selector=validation_selector,
        )

    train = fit

    def evaluate(self, evaluator, *args, **kwargs):
        return evaluator.evaluate(*args, **kwargs)

    def predict(
        self,
        bundle: EvidenceNeuralBundle,
        documents: Sequence[EvidenceTextDocument],
        *,
        device: str | None = None,
    ):
        return predict_neural_model(bundle, documents, device=device)

    def package(self, bundle: EvidenceNeuralBundle, destination: str | Path):
        return bundle.save_pretrained(destination)

    def load(self, checkpoint_directory: str | Path) -> EvidenceNeuralBundle:
        return load_neural_bundle(checkpoint_directory)


class EvidenceBiLSTMPlugin(_EvidenceNeuralPlugin):
    key = "evidence_bilstm"
    config_class = EvidenceBiLSTMConfig


class EvidenceCNNPlugin(_EvidenceNeuralPlugin):
    key = "evidence_cnn"
    config_class = EvidenceCNNConfig


def register_builtin_neural_model_types(registry=None) -> None:
    """Opt-in registration hook used once runner dispatch is enabled."""

    from al_medlit.core.exceptions import ValidationError
    from al_medlit.training.registry import model_types

    selected_registry = registry or model_types
    for plugin in (EvidenceBiLSTMPlugin(), EvidenceCNNPlugin()):
        try:
            selected_registry.register(plugin)
        except ValidationError as exc:
            if "already registered" not in exc.message:
                raise
