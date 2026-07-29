"""Lazy-loaded, safely packaged TF-IDF linear-model trainer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.trainers.contracts import (
    TrainerPreflight,
    TrainingInput,
    TrainingOutput,
    TrainingPlan,
)


class SklearnTfidfTrainer:
    key = "sklearn_tfidf"
    recipe_keys = (
        "tfidf_linear_regression",
        "tfidf_logistic_regression",
    )
    trainer_version = "1"

    def preflight(self, recipe_key: str | None = None) -> TrainerPreflight:
        if recipe_key is not None and recipe_key not in self.recipe_keys:
            raise ValidationError(f"Unsupported sklearn recipe '{recipe_key}'")
        checks = []
        for module, package in (("sklearn", "scikit-learn"), ("skops", "skops")):
            present = importlib.util.find_spec(module) is not None
            checks.append(
                {
                    "key": f"package:{package}",
                    "status": "pass" if present else "fail",
                    "message": (
                        f"{package} is installed"
                        if present
                        else f"{package} is required in the classical-cpu worker"
                    ),
                }
            )
        return TrainerPreflight(
            ready=all(check["status"] == "pass" for check in checks),
            runtime_class="classical-cpu",
            checks=tuple(checks),
        )

    def plan(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan:
        if recipe_key not in self.recipe_keys:
            raise ValidationError(f"Unsupported sklearn recipe '{recipe_key}'")
        validated = training_recipes.validate(recipe_key, dict(config))
        if not validated.valid or validated.normalized_config is None:
            raise ValidationError("; ".join(validated.errors))
        fields = validated.normalized_config["fields"]
        train_texts, train_targets = self._extract(
            training_input.rows,
            fields["input_field"],
            fields["target_field"],
        )
        validation_targets = []
        if training_input.validation_rows:
            _, validation_targets = self._extract(
                training_input.validation_rows,
                fields["input_field"],
                fields["target_field"],
            )
        task_kind = self._task_kind(recipe_key, training_input.task_kind)
        output_contract = self._output_contract(
            task_kind,
            train_targets,
            validation_targets,
            training_input.label_vocabulary,
        )
        return TrainingPlan(
            normalized_config=validated.normalized_config,
            manifest={
                "schema_version": "al-medlit-training-plan-v1",
                "recipe": {"key": recipe_key, "version": validated.recipe_version},
                "trainer": {"key": self.key, "version": self.trainer_version},
                "runtime_class": "classical-cpu",
                "task": {
                    "kind": task_kind,
                    "schema": training_input.task_schema,
                },
                "dataset_fingerprint": training_input.dataset_fingerprint,
                "split_fingerprint": training_input.split_fingerprint,
                "seed": seed,
                "row_counts": {
                    "train": len(train_texts),
                    "validation": len(training_input.validation_rows),
                },
                "field_mapping": fields,
                "output_contract": output_contract,
                "artifact_policy": {
                    "formats": ["skops", "json"],
                    "forbidden_extensions": [".pkl", ".pickle", ".joblib"],
                },
            },
        )

    @staticmethod
    def _extract(
        rows: Sequence[Mapping],
        input_field: str,
        target_field: str,
    ) -> tuple[list[str], list]:
        texts: list[str] = []
        targets: list = []
        for index, row in enumerate(rows):
            if input_field not in row or target_field not in row:
                raise ValidationError(
                    f"Training row {index} must contain '{input_field}' and '{target_field}'"
                )
            text = row[input_field]
            if not isinstance(text, str) or not text.strip():
                raise ValidationError(
                    f"Training row {index} field '{input_field}' must be non-empty text"
                )
            texts.append(text)
            targets.append(row[target_field])
        if not texts:
            raise ValidationError("Training input must contain at least one row")
        return texts, targets

    @staticmethod
    def _task_kind(recipe_key: str, task_kind: str | None) -> str:
        default = (
            "regression"
            if recipe_key == "tfidf_linear_regression"
            else "classification"
        )
        resolved = task_kind or default
        allowed = (
            {"regression"}
            if recipe_key == "tfidf_linear_regression"
            else {"classification", "multilabel_classification"}
        )
        if resolved not in allowed:
            raise ValidationError(
                f"Recipe '{recipe_key}' does not support task kind '{resolved}'"
            )
        return resolved

    @staticmethod
    def _multilabel_rows(
        targets: Sequence[Any],
        *,
        split: str,
    ) -> list[tuple[str, ...]]:
        normalized: list[tuple[str, ...]] = []
        for index, target in enumerate(targets):
            if not isinstance(target, (list, tuple)):
                raise ValidationError(
                    f"{split.capitalize()} row {index} target must be a label list "
                    "for multilabel classification"
                )
            labels = tuple(target)
            if any(not isinstance(label, str) or not label.strip() for label in labels):
                raise ValidationError(
                    f"{split.capitalize()} row {index} contains an invalid multilabel value"
                )
            if len(labels) != len(set(labels)):
                raise ValidationError(
                    f"{split.capitalize()} row {index} contains duplicate multilabel values"
                )
            normalized.append(labels)
        return normalized

    @classmethod
    def _multilabel_vocabulary(
        cls,
        train_targets: Sequence[Any],
        validation_targets: Sequence[Any],
        declared_vocabulary: Sequence[str],
    ) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]], tuple[str, ...]]:
        train_rows = cls._multilabel_rows(train_targets, split="training")
        validation_rows = cls._multilabel_rows(
            validation_targets,
            split="validation",
        )
        if declared_vocabulary:
            vocabulary = tuple(declared_vocabulary)
            if (
                any(not label.strip() for label in vocabulary)
                or len(vocabulary) != len(set(vocabulary))
            ):
                raise ValidationError(
                    "Multilabel label_vocabulary must contain unique non-empty labels"
                )
        else:
            vocabulary = tuple(
                sorted({label for labels in train_rows for label in labels})
            )
        if not vocabulary:
            raise ValidationError(
                "Multilabel classification requires at least one training label"
            )
        known = set(vocabulary)
        unknown_training = sorted(
            {label for labels in train_rows for label in labels} - known
        )
        if unknown_training:
            raise ValidationError(
                "Training labels are absent from label_vocabulary: "
                + ", ".join(unknown_training)
            )
        unknown_validation = sorted(
            {label for labels in validation_rows for label in labels} - known
        )
        if unknown_validation:
            raise ValidationError(
                "Validation labels are absent from the training label vocabulary: "
                + ", ".join(unknown_validation)
            )
        for label in vocabulary:
            positive_count = sum(label in labels for labels in train_rows)
            if positive_count == 0:
                raise ValidationError(
                    f"Multilabel class '{label}' has no positive training example"
                )
            if positive_count == len(train_rows):
                raise ValidationError(
                    f"Multilabel class '{label}' has no negative training example"
                )
        return train_rows, validation_rows, vocabulary

    @classmethod
    def _output_contract(
        cls,
        task_kind: str,
        train_targets: Sequence[Any],
        validation_targets: Sequence[Any],
        declared_vocabulary: Sequence[str],
    ) -> dict:
        all_targets = (*train_targets, *validation_targets)
        if task_kind == "multilabel_classification":
            _, _, vocabulary = cls._multilabel_vocabulary(
                train_targets,
                validation_targets,
                declared_vocabulary,
            )
            return {
                "type": "multilabel_classification",
                "encoding": "binary_indicator",
                "labels": list(vocabulary),
                "label_to_index": {
                    label: index for index, label in enumerate(vocabulary)
                },
                "prediction_threshold": 0.5,
            }
        if task_kind == "regression":
            if any(
                isinstance(target, bool) or not isinstance(target, (int, float))
                for target in all_targets
            ):
                raise ValidationError("Regression targets must be numeric")
            return {"type": "regression", "value": "number"}
        if any(
            isinstance(target, (list, tuple, dict, set))
            for target in all_targets
        ):
            raise ValidationError(
                "Single-label classification targets must be scalar values"
            )
        return {"type": "classification", "encoding": "class_label"}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def train(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        destination: Path,
        seed: int,
    ) -> TrainingOutput:
        if recipe_key not in self.recipe_keys:
            raise ValidationError(f"Unsupported sklearn recipe '{recipe_key}'")
        plan = self.plan(
            recipe_key=recipe_key,
            config=config,
            training_input=training_input,
            seed=seed,
        )
        report = self.preflight(recipe_key)
        if not report.ready:
            failures = [
                check["message"] for check in report.checks if check["status"] == "fail"
            ]
            raise ValidationError("Classical worker preflight failed: " + "; ".join(failures))

        normalized = plan.normalized_config
        validated = training_recipes.validate(recipe_key, dict(config))
        fields = normalized["fields"]
        task_kind = plan.manifest["task"]["kind"]
        train_texts, train_targets = self._extract(
            training_input.rows,
            fields["input_field"],
            fields["target_field"],
        )

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            hamming_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
        from skops.io import dump

        vectorizer = TfidfVectorizer(
            max_features=normalized["max_features"],
            ngram_range=(normalized["ngram_min"], normalized["ngram_max"]),
            min_df=normalized["min_document_frequency"],
        )
        if recipe_key == "tfidf_linear_regression":
            estimator = LinearRegression(fit_intercept=normalized["fit_intercept"])
        else:
            logistic = LogisticRegression(
                C=normalized["regularization_c"],
                class_weight=normalized["class_weight"],
                max_iter=normalized["max_iterations"],
                random_state=seed,
            )
            estimator = (
                OneVsRestClassifier(logistic)
                if task_kind == "multilabel_classification"
                else logistic
            )
        pipeline = Pipeline([("vectorizer", vectorizer), ("estimator", estimator)])
        label_binarizer = None
        fit_targets = train_targets
        if task_kind == "multilabel_classification":
            label_binarizer = MultiLabelBinarizer(
                classes=plan.manifest["output_contract"]["labels"]
            )
            fit_targets = label_binarizer.fit_transform(train_targets)
        pipeline.fit(train_texts, fit_targets)

        metrics: dict[str, float | None] = {}
        if training_input.validation_rows:
            validation_texts, validation_targets = self._extract(
                training_input.validation_rows,
                fields["input_field"],
                fields["target_field"],
            )
            predictions = pipeline.predict(validation_texts)
            if task_kind == "regression":
                mse = float(mean_squared_error(validation_targets, predictions))
                metrics = {
                    "mean_absolute_error": float(
                        mean_absolute_error(validation_targets, predictions)
                    ),
                    "root_mean_squared_error": math.sqrt(mse),
                    "r2": float(r2_score(validation_targets, predictions)),
                }
            elif task_kind == "multilabel_classification":
                validation_indicators = label_binarizer.transform(validation_targets)
                metrics = {
                    "subset_accuracy": float(
                        accuracy_score(validation_indicators, predictions)
                    ),
                    "micro_f1": float(
                        f1_score(
                            validation_indicators,
                            predictions,
                            average="micro",
                            zero_division=0,
                        )
                    ),
                    "macro_f1": float(
                        f1_score(
                            validation_indicators,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "hamming_loss": float(
                        hamming_loss(validation_indicators, predictions)
                    ),
                }
            else:
                metrics = {
                    "accuracy": float(accuracy_score(validation_targets, predictions)),
                    "macro_f1": float(
                        f1_score(
                            validation_targets,
                            predictions,
                            average="macro",
                            zero_division=0,
                        )
                    ),
                }

        destination.mkdir(parents=True, exist_ok=False)
        model_path = destination / "model.skops"
        config_path = destination / "recipe.json"
        manifest_path = destination / "manifest.json"
        dump(pipeline, model_path)
        config_path.write_text(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "al-medlit-training-package-v1",
            "recipe": {"key": recipe_key, "version": validated.recipe_version},
            "trainer": {"key": self.key, "version": self.trainer_version},
            "task": plan.manifest["task"],
            "output_contract": plan.manifest["output_contract"],
            "dataset_fingerprint": training_input.dataset_fingerprint,
            "split_fingerprint": training_input.split_fingerprint,
            "seed": seed,
            "row_counts": {
                "train": len(training_input.rows),
                "validation": len(training_input.validation_rows),
            },
            "validation_metrics": metrics,
            "files": {
                "model.skops": self._sha256(model_path),
                "recipe.json": self._sha256(config_path),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return TrainingOutput(
            manifest=manifest,
            validation_metrics=metrics,
            artifact_paths=("model.skops", "recipe.json", "manifest.json"),
        )


def register_sklearn_tfidf_trainer(registry=None) -> None:
    from al_medlit.training.trainers.contracts import trainer_plugins

    selected = registry or trainer_plugins
    selected.register(SklearnTfidfTrainer(), replace=True)
