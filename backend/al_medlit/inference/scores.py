"""Portable score rows shared by local and remote evidence inference."""

import json
from pathlib import Path

SENTENCE_SCORER_MODEL_TYPES = frozenset({"evidence_svm", "evidence_random_forest"})

_LABEL_PROBABILITIES = {
    "O": (1.0, 0.0, 0.0),
    "B": (0.0, 1.0, 0.0),
    "I": (0.0, 0.0, 1.0),
}


def label_probabilities(label: str) -> tuple[float, float, float]:
    return _LABEL_PROBABILITIES[label]


def checkpoint_sentence_score_threshold(checkpoint_root: Path, *, model_type: str) -> float:
    from al_medlit.training.model_types.evidence_conventional.config import (
        EvidenceRandomForestConfig,
        EvidenceSVMConfig,
    )
    from al_medlit.training.model_types.evidence_conventional.model import MODEL_SCHEMA_VERSION

    metadata = json.loads(
        (checkpoint_root / "al-medlit-conventional.json").read_text(encoding="utf-8")
    )
    if (
        metadata.get("schema_version") != MODEL_SCHEMA_VERSION
        or metadata.get("model_type") != model_type
    ):
        raise ValueError("Sentence scorer metadata does not match the selected checkpoint")
    config_class = EvidenceSVMConfig if model_type == "evidence_svm" else EvidenceRandomForestConfig
    return config_class.model_validate(metadata["config"]).sentence_score_threshold


def sentence_score_probabilities(
    scores: list[float], *, threshold: float,
) -> list[tuple[float, float, float]]:
    rows = []
    inside = False
    for score in scores:
        probability = float(score)
        positive = probability >= threshold
        rows.append((1 - probability, 0.0, probability) if inside else (
            1 - probability, probability, 0.0,
        ))
        inside = positive
    return rows


def synthetic_logits(ordinal: int) -> tuple[float, float, float]:
    if ordinal == 0:
        return (0.0, 6.0, 0.0)
    if ordinal == 1:
        return (0.0, 0.0, 6.0)
    return (6.0, 0.0, 0.0)
