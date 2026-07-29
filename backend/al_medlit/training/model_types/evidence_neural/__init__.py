"""Compact neural models for sentence-aligned Evidence Blocks."""

from al_medlit.training.model_types.evidence_neural.config import (
    EvidenceBiLSTMConfig,
    EvidenceCNNConfig,
)
from al_medlit.training.model_types.evidence_neural.data import (
    EvidenceSequencePrediction,
    EvidenceTextDocument,
    NeuralCheckpointScore,
    NeuralTrainingSummary,
    documents_from_window_rows,
)
from al_medlit.training.model_types.evidence_neural.model import (
    EvidenceNeuralBundle,
    canonical_checkpoint_is_better,
    fit_neural_model,
    load_neural_bundle,
    predict_neural_model,
)
from al_medlit.training.model_types.evidence_neural.plugin import (
    EvidenceBiLSTMPlugin,
    EvidenceCNNPlugin,
    register_builtin_neural_model_types,
)
from al_medlit.training.model_types.evidence_neural.vocabulary import (
    EvidenceVocabulary,
    build_vocabulary,
)

__all__ = [
    "EvidenceBiLSTMConfig",
    "EvidenceBiLSTMPlugin",
    "EvidenceCNNConfig",
    "EvidenceCNNPlugin",
    "EvidenceNeuralBundle",
    "EvidenceSequencePrediction",
    "EvidenceTextDocument",
    "EvidenceVocabulary",
    "NeuralCheckpointScore",
    "NeuralTrainingSummary",
    "build_vocabulary",
    "canonical_checkpoint_is_better",
    "documents_from_window_rows",
    "fit_neural_model",
    "load_neural_bundle",
    "predict_neural_model",
    "register_builtin_neural_model_types",
]
