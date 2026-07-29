from al_medlit.training.evaluation.evidence import (
    EVIDENCE_TASK_CONTRACT,
    EvidenceBlockEvaluator,
    EvidenceEvaluationContext,
    EvidenceEvaluationExample,
    EvidenceEvaluationReport,
    checkpoint_selection_key,
    comparison_tier,
    configuration_fingerprint,
    select_best_checkpoint,
)
from al_medlit.training.evaluation.registry import task_evaluators

task_evaluators.register(EvidenceBlockEvaluator(), replace=True)

__all__ = [
    "EVIDENCE_TASK_CONTRACT",
    "EvidenceBlockEvaluator",
    "EvidenceEvaluationContext",
    "EvidenceEvaluationExample",
    "EvidenceEvaluationReport",
    "checkpoint_selection_key",
    "comparison_tier",
    "configuration_fingerprint",
    "select_best_checkpoint",
    "task_evaluators",
]
