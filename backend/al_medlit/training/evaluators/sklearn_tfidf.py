"""Trusted holdout evaluation for TF-IDF sklearn model packages."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.artifact_loading import load_safe_skops_model
from al_medlit.training.evaluators.contracts import (
    EvaluationInput,
    EvaluationOutput,
)

_CLASSIFICATION_ALIASES = {"f1": "macro_f1"}
_REGRESSION_ALIASES = {"mae": "mean_absolute_error", "rmse": "root_mean_squared_error"}


def _python_value(value: Any) -> Any:
    converter = getattr(value, "tolist", None)
    return converter() if callable(converter) else value


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else (2.0 * tp) / denominator


def _macro_f1(expected: Sequence[Any], predicted: Sequence[Any]) -> float:
    labels = sorted(set(expected) | set(predicted), key=str)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(
            expected_value == label and predicted_value == label
            for expected_value, predicted_value in zip(
                expected,
                predicted,
                strict=True,
            )
        )
        fp = sum(
            expected_value != label and predicted_value == label
            for expected_value, predicted_value in zip(
                expected,
                predicted,
                strict=True,
            )
        )
        fn = sum(
            expected_value == label and predicted_value != label
            for expected_value, predicted_value in zip(
                expected,
                predicted,
                strict=True,
            )
        )
        scores.append(_f1(tp, fp, fn))
    return sum(scores) / len(scores)


def _selected_metrics(
    available: dict[str, float | None],
    requested: Sequence[str],
    aliases: Mapping[str, str],
) -> dict[str, float | None]:
    if not requested:
        return available
    selected: dict[str, float | None] = {}
    for raw_name in requested:
        name = aliases.get(raw_name, raw_name)
        if name not in available:
            raise ValidationError(
                f"Metric '{raw_name}' is not supported by this evaluator"
            )
        selected[raw_name] = available[name]
    return selected


class SklearnTfidfEvaluator:
    key = "sklearn_tfidf_holdout"
    evaluator_version = "1"
    recipe_keys = (
        "tfidf_linear_regression",
        "tfidf_logistic_regression",
    )

    def __init__(
        self,
        *,
        model_loader: Callable[[Path], Any] | None = None,
    ) -> None:
        self._model_loader = model_loader

    def _load_model(self, model_path: Path):
        return load_safe_skops_model(model_path, model_loader=self._model_loader)

    @staticmethod
    def _field_mapping(config: Mapping) -> tuple[str, str]:
        fields = config.get("fields")
        if not isinstance(fields, Mapping):
            raise ValidationError("TF-IDF evaluation requires a fields mapping")
        input_field = fields.get("input_field")
        target_field = fields.get("target_field")
        if not isinstance(input_field, str) or not isinstance(target_field, str):
            raise ValidationError(
                "TF-IDF evaluation requires input_field and target_field"
            )
        return input_field, target_field

    @staticmethod
    def _extract(
        rows: Sequence[Mapping],
        input_field: str,
        target_field: str,
    ) -> tuple[list[str], list[Any]]:
        texts: list[str] = []
        targets: list[Any] = []
        for index, row in enumerate(rows):
            text = row.get(input_field)
            if not isinstance(text, str) or not text.strip():
                raise ValidationError(
                    f"Evaluation row {index} field '{input_field}' must be non-empty text"
                )
            if target_field not in row:
                raise ValidationError(
                    f"Evaluation row {index} is missing target field '{target_field}'"
                )
            texts.append(text)
            targets.append(row[target_field])
        if not texts:
            raise ValidationError("Protected test evaluation requires at least one row")
        return texts, targets

    @staticmethod
    def _regression_metrics(
        expected: Sequence[Any],
        predicted: Sequence[Any],
    ) -> dict[str, float | None]:
        values = [*expected, *predicted]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise ValidationError("Regression evaluation values must be numeric")
        absolute_errors = [
            abs(float(actual) - float(estimate))
            for actual, estimate in zip(expected, predicted, strict=True)
        ]
        squared_errors = [
            (float(actual) - float(estimate)) ** 2
            for actual, estimate in zip(expected, predicted, strict=True)
        ]
        mean_expected = sum(float(value) for value in expected) / len(expected)
        total_variance = sum(
            (float(value) - mean_expected) ** 2 for value in expected
        )
        residual = sum(squared_errors)
        return {
            "mean_absolute_error": sum(absolute_errors) / len(absolute_errors),
            "root_mean_squared_error": math.sqrt(
                sum(squared_errors) / len(squared_errors)
            ),
            "r2": None if total_variance == 0 else 1.0 - residual / total_variance,
        }

    @staticmethod
    def _classification_metrics(
        expected: Sequence[Any],
        predicted: Sequence[Any],
    ) -> dict[str, float | None]:
        if any(
            isinstance(value, (list, tuple, dict, set))
            for value in [*expected, *predicted]
        ):
            raise ValidationError(
                "Classification evaluation values must be scalar labels"
            )
        accuracy = sum(
            actual == estimate
            for actual, estimate in zip(expected, predicted, strict=True)
        ) / len(expected)
        return {
            "accuracy": accuracy,
            "macro_f1": _macro_f1(expected, predicted),
        }

    @staticmethod
    def _label_sets(
        values: Sequence[Any],
        vocabulary: Sequence[str],
        *,
        predicted: bool,
    ) -> list[set[str]]:
        normalized: list[set[str]] = []
        for index, raw_value in enumerate(values):
            value = _python_value(raw_value)
            if not isinstance(value, (list, tuple, set)):
                raise ValidationError(
                    f"Multilabel evaluation row {index} must contain a label sequence"
                )
            sequence = list(value)
            if predicted and len(sequence) == len(vocabulary) and all(
                isinstance(item, (bool, int, float)) for item in sequence
            ):
                normalized.append(
                    {
                        vocabulary[position]
                        for position, item in enumerate(sequence)
                        if bool(item)
                    }
                )
                continue
            if any(not isinstance(item, str) for item in sequence):
                raise ValidationError(
                    f"Multilabel evaluation row {index} contains an invalid label"
                )
            normalized.append(set(sequence))
        return normalized

    @classmethod
    def _multilabel_metrics(
        cls,
        expected: Sequence[Any],
        predicted: Sequence[Any],
        vocabulary: Sequence[str],
    ) -> dict[str, float | None]:
        if not vocabulary:
            raise ValidationError(
                "Multilabel evaluation requires a pinned label vocabulary"
            )
        expected_sets = cls._label_sets(expected, vocabulary, predicted=False)
        predicted_sets = cls._label_sets(predicted, vocabulary, predicted=True)
        known = set(vocabulary)
        if any((labels - known) for labels in [*expected_sets, *predicted_sets]):
            raise ValidationError(
                "Multilabel evaluation contains labels outside the pinned vocabulary"
            )
        subset_accuracy = sum(
            actual == estimate
            for actual, estimate in zip(expected_sets, predicted_sets, strict=True)
        ) / len(expected_sets)
        tp_total = fp_total = fn_total = mismatches = 0
        label_scores = []
        for label in vocabulary:
            tp = sum(
                label in actual and label in estimate
                for actual, estimate in zip(
                    expected_sets,
                    predicted_sets,
                    strict=True,
                )
            )
            fp = sum(
                label not in actual and label in estimate
                for actual, estimate in zip(
                    expected_sets,
                    predicted_sets,
                    strict=True,
                )
            )
            fn = sum(
                label in actual and label not in estimate
                for actual, estimate in zip(
                    expected_sets,
                    predicted_sets,
                    strict=True,
                )
            )
            tp_total += tp
            fp_total += fp
            fn_total += fn
            mismatches += fp + fn
            label_scores.append(_f1(tp, fp, fn))
        return {
            "subset_accuracy": subset_accuracy,
            "micro_f1": _f1(tp_total, fp_total, fn_total),
            "macro_f1": sum(label_scores) / len(label_scores),
            "hamming_loss": mismatches
            / (len(expected_sets) * len(vocabulary)),
        }

    def evaluate(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        evaluation_input: EvaluationInput,
        model_directory: Path,
    ) -> EvaluationOutput:
        if recipe_key not in self.recipe_keys:
            raise ValidationError(f"Unsupported sklearn recipe '{recipe_key}'")
        if not evaluation_input.protected_split:
            raise ValidationError(
                "The holdout evaluator requires a protected split binding"
            )
        model_path = model_directory / "model.skops"
        if not model_path.is_file() or model_path.is_symlink():
            raise ValidationError("TF-IDF evaluation requires model.skops")
        model = self._load_model(model_path)
        predict = getattr(model, "predict", None)
        if not callable(predict):
            raise ValidationError("The generated TF-IDF artifact cannot make predictions")
        input_field, target_field = self._field_mapping(config)
        texts, expected = self._extract(
            evaluation_input.rows,
            input_field,
            target_field,
        )
        raw_predictions = _python_value(predict(texts))
        if not isinstance(raw_predictions, (list, tuple)):
            raise ValidationError("The TF-IDF evaluator returned invalid predictions")
        predicted = [_python_value(value) for value in raw_predictions]
        if len(predicted) != len(expected):
            raise ValidationError(
                "The TF-IDF evaluator returned the wrong prediction count"
            )

        if evaluation_input.task_kind == "regression":
            available = self._regression_metrics(expected, predicted)
            aliases = _REGRESSION_ALIASES
        elif evaluation_input.task_kind == "multilabel_classification":
            available = self._multilabel_metrics(
                expected,
                predicted,
                evaluation_input.label_vocabulary,
            )
            aliases = _CLASSIFICATION_ALIASES
        elif evaluation_input.task_kind == "classification":
            available = self._classification_metrics(expected, predicted)
            aliases = _CLASSIFICATION_ALIASES
        else:
            raise ValidationError(
                f"TF-IDF evaluation does not support task kind "
                f"'{evaluation_input.task_kind}'"
            )
        metrics = _selected_metrics(
            available,
            evaluation_input.requested_metrics,
            aliases,
        )
        return EvaluationOutput(
            metrics=metrics,
            prediction_count=len(predicted),
            report={
                "task_kind": evaluation_input.task_kind,
                "split_name": evaluation_input.split_name,
                "protected_split": True,
                "prediction_count": len(predicted),
                "available_metrics": sorted(available),
            },
        )


def register_sklearn_tfidf_evaluator(registry=None) -> None:
    from al_medlit.training.evaluators.contracts import evaluator_plugins

    selected = registry or evaluator_plugins
    selected.register(SklearnTfidfEvaluator(), replace=True)
