"""ModelPlugin adapters for conventional Evidence baselines."""

from __future__ import annotations

from pathlib import Path

from al_medlit.training.model_types.evidence_conventional.config import (
    EvidenceCRFConfig,
    EvidenceRandomForestConfig,
    EvidenceSVMConfig,
)
from al_medlit.training.model_types.evidence_conventional.model import (
    dependency_preflight,
    load_conventional_model,
    predict_window,
    train_crf,
    train_sentence_scorer,
)


class _EvidenceConventionalPlugin:
    task_type = "evidence_block_sentence_tagging"
    task_contract_key = "evidence_blocks"
    task_contract_version = "1"
    config_class = None

    @property
    def family(self) -> str:
        return "conventional_ml"

    @property
    def descriptor(self):
        from al_medlit.training.model_types.catalog import get_builtin_model_descriptor

        return get_builtin_model_descriptor(self.key)

    def validate_config(self, config: dict):
        return self.config_class.model_validate(config)

    def preflight(self) -> dict:
        return dependency_preflight(self.key)

    def build_model(self, config: dict):
        """Validate configuration before a worker constructs the fitted model."""

        status = self.preflight()
        if not status["available"]:
            from al_medlit.training.model_types.evidence_conventional.model import (
                MissingConventionalDependencyError,
            )

            raise MissingConventionalDependencyError(status["reason"])
        return self.validate_config(config)

    def build_dataset(self, *args, **kwargs):
        from al_medlit.training.windowing import EvidenceBlockWindowBuilder

        return EvidenceBlockWindowBuilder(*args, **kwargs)

    def evaluate(self, evaluator, *args, **kwargs):
        return evaluator.evaluate(*args, **kwargs)

    def predict(self, model, *, target_text: str, sentences: list[str]):
        return predict_window(model, target_text=target_text, sentences=sentences)

    def package(self, packager, *args, **kwargs):
        return packager.package(*args, **kwargs)

    def load(self, checkpoint_directory: str | Path):
        return load_conventional_model(checkpoint_directory)


class EvidenceSVMPlugin(_EvidenceConventionalPlugin):
    key = "evidence_svm"
    config_class = EvidenceSVMConfig

    def train(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        return self.fit(rows, config, destination)

    def fit(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        validated = self.validate_config(config)
        return train_sentence_scorer(self.key, rows, validated, destination)


class EvidenceRandomForestPlugin(_EvidenceConventionalPlugin):
    key = "evidence_random_forest"
    config_class = EvidenceRandomForestConfig

    def train(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        return self.fit(rows, config, destination)

    def fit(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        validated = self.validate_config(config)
        return train_sentence_scorer(self.key, rows, validated, destination)


class EvidenceCRFPlugin(_EvidenceConventionalPlugin):
    key = "evidence_crf"
    config_class = EvidenceCRFConfig

    def train(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        return self.fit(rows, config, destination)

    def fit(self, rows: list[dict], config: dict, destination: str | Path) -> dict:
        validated = self.validate_config(config)
        return train_crf(rows, validated, destination)


def register_builtin_conventional_model_types(registry=None) -> None:
    from al_medlit.core.exceptions import ValidationError

    if registry is None:
        from al_medlit.training.registry import model_types

        registry = model_types
    for plugin in (
        EvidenceCRFPlugin(),
        EvidenceRandomForestPlugin(),
        EvidenceSVMPlugin(),
    ):
        try:
            registry.register(plugin)
        except ValidationError as exc:
            if "already registered" not in exc.message:
                raise
