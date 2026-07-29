"""Trusted post-training evaluators registered inside worker processes."""

from al_medlit.training.evaluators.contracts import (
    EvaluationInput,
    EvaluationOutput,
    EvaluatorPlugin,
    EvaluatorPluginRegistry,
    evaluator_plugins,
)
from al_medlit.training.evaluators.sklearn_tfidf import (
    SklearnTfidfEvaluator,
    register_sklearn_tfidf_evaluator,
)

register_sklearn_tfidf_evaluator()

__all__ = [
    "EvaluationInput",
    "EvaluationOutput",
    "EvaluatorPlugin",
    "EvaluatorPluginRegistry",
    "SklearnTfidfEvaluator",
    "evaluator_plugins",
    "register_sklearn_tfidf_evaluator",
]
