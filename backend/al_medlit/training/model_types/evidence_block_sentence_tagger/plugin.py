from al_medlit.training.model_types.evidence_block_sentence_tagger.config import (
    EvidenceBlockSentenceTaggerConfig,
)
from al_medlit.training.model_types.evidence_block_sentence_tagger.model import (
    build_sentence_tagger,
)


class EvidenceBlockSentenceTaggerPlugin:
    key = "evidence_block_sentence_tagger"
    task_type = "evidence_block_sentence_tagging"
    task_contract_key = "evidence_blocks"
    task_contract_version = "1"

    @property
    def family(self) -> str:
        return self.descriptor.family.value

    @property
    def descriptor(self):
        from al_medlit.training.model_types.catalog import get_builtin_model_descriptor

        return get_builtin_model_descriptor(self.key)

    def validate_config(self, config: dict) -> EvidenceBlockSentenceTaggerConfig:
        return EvidenceBlockSentenceTaggerConfig.model_validate(config)

    def build_model(self, config: dict):
        return build_sentence_tagger(self.validate_config(config))

    def build_dataset(self, *args, **kwargs):
        from al_medlit.training.windowing import EvidenceBlockWindowBuilder

        return EvidenceBlockWindowBuilder(*args, **kwargs)

    def train(self, trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)

    def fit(self, trainer, *args, **kwargs):
        return trainer.train(*args, **kwargs)

    def evaluate(self, evaluator, *args, **kwargs):
        return evaluator.evaluate(*args, **kwargs)

    def predict(self, predictor, *args, **kwargs):
        return predictor.predict(*args, **kwargs)

    def package(self, packager, *args, **kwargs):
        return packager.package(*args, **kwargs)

    def load(self, loader, *args, **kwargs):
        return loader.load(*args, **kwargs)


def register_builtin_model_type(registry=None) -> None:
    from al_medlit.core.exceptions import ValidationError
    from al_medlit.training.registry import model_types

    selected_registry = registry or model_types
    try:
        selected_registry.register(EvidenceBlockSentenceTaggerPlugin())
    except ValidationError as exc:
        if "already registered" not in exc.message:
            raise
