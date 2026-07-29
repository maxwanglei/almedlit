"""Lazy Hugging Face trainers for sequence, token, and span tasks."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.trainers.contracts import (
    TrainerPreflight,
    TrainingInput,
    TrainingOutput,
    TrainingPlan,
)
from al_medlit.training.trainers.huggingface_common import (
    base_plan,
    execute_torch_training,
    finalize_output_manifest,
    preflight_for_recipe,
    relative_artifact_paths,
    require_local_base_model,
    require_ready,
    validate_recipe,
    validate_task_kind,
    write_json,
)


def _nonempty_text(row: Mapping, field: str, row_index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"Training row {row_index} field '{field}' must be non-empty text"
        )
    return value


def _label_key(value, *, row_index: int, field: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValidationError(
            f"Training row {row_index} field '{field}' must contain string or integer labels"
        )
    return str(value)


def _label_vocabulary(
    discovered: set[str],
    configured: Sequence[str],
) -> tuple[str, ...]:
    if configured:
        if len(configured) != len(set(configured)):
            raise ValidationError("label_vocabulary values must be unique")
        vocabulary = tuple(configured)
    else:
        vocabulary = tuple(sorted(discovered))
    if not vocabulary:
        raise ValidationError("Training data must define at least one label")
    missing = discovered - set(vocabulary)
    if missing:
        raise ValidationError(
            "Training labels are absent from label_vocabulary: "
            + ", ".join(sorted(missing))
        )
    return vocabulary


class _HuggingFaceTaskTrainer:
    key: str
    recipe_keys: tuple[str, ...]
    trainer_version = "1"

    def preflight(self, recipe_key: str | None = None) -> TrainerPreflight:
        return preflight_for_recipe(self.recipe_keys, recipe_key)

    def _execute(
        self,
        *,
        plan: TrainingPlan,
        training_input: TrainingInput,
        base_model_path: Path,
        destination: Path,
        seed: int,
    ) -> dict[str, float | None]:
        raise NotImplementedError

    def train(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        destination: Path,
        seed: int,
    ) -> TrainingOutput:
        plan = self.plan(
            recipe_key=recipe_key,
            config=config,
            training_input=training_input,
            seed=seed,
        )
        if destination.exists():
            raise ValidationError("Training destination must not already exist")
        require_ready(self.preflight(recipe_key))
        base_model_path = require_local_base_model(training_input)
        with tempfile.TemporaryDirectory(prefix="al-medlit-hf-task-") as temporary:
            package_root = Path(temporary) / "package"
            package_root.mkdir()
            metrics = self._execute(
                plan=plan,
                training_input=training_input,
                base_model_path=base_model_path,
                destination=package_root,
                seed=seed,
            )
            write_json(package_root / "recipe.json", plan.normalized_config)
            manifest = finalize_output_manifest(
                package_root,
                plan=plan,
                validation_metrics=metrics,
            )
            shutil.copytree(package_root, destination)
        return TrainingOutput(
            manifest=manifest,
            validation_metrics=metrics,
            artifact_paths=relative_artifact_paths(destination),
        )


class HuggingFaceSequenceTrainer(_HuggingFaceTaskTrainer):
    key = "huggingface_sequence"
    recipe_keys = ("transformer_sequence_classification",)

    def plan(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan:
        normalized, recipe_version = validate_recipe(
            self.recipe_keys, recipe_key, config
        )
        task_kind = validate_task_kind(recipe_key, training_input)
        input_field = normalized["input_field"]
        target_field = normalized["target_field"]
        discovered: set[str] = set()
        if not training_input.rows:
            raise ValidationError("Training input must contain at least one row")

        def validate_rows(rows) -> set[str]:
            labels: set[str] = set()
            for index, row in enumerate(rows):
                _nonempty_text(row, input_field, index)
                target = row.get(target_field)
                if task_kind == "multilabel_classification":
                    if not isinstance(target, (list, tuple)):
                        raise ValidationError(
                            f"Training row {index} field '{target_field}' must be a label list"
                        )
                    labels.update(
                        _label_key(value, row_index=index, field=target_field)
                        for value in target
                    )
                else:
                    labels.add(
                        _label_key(target, row_index=index, field=target_field)
                    )
            return labels

        discovered = validate_rows(training_input.rows)
        vocabulary = _label_vocabulary(
            discovered,
            training_input.label_vocabulary,
        )
        validation_labels = validate_rows(training_input.validation_rows)
        unknown_validation = validation_labels - set(vocabulary)
        if unknown_validation:
            raise ValidationError(
                "Validation labels are absent from the training label vocabulary: "
                + ", ".join(sorted(unknown_validation))
            )
        return base_plan(
            trainer_key=self.key,
            trainer_version=self.trainer_version,
            recipe_key=recipe_key,
            recipe_version=recipe_version,
            normalized_config=normalized,
            training_input=training_input,
            task_kind=task_kind,
            seed=seed,
            row_mapping={
                "input": {"field": input_field, "type": "text"},
                "target": {
                    "field": target_field,
                    "type": (
                        "label_set"
                        if task_kind == "multilabel_classification"
                        else "label"
                    ),
                },
            },
            output_contract={
                "type": task_kind,
                "labels": list(vocabulary),
                "label_to_id": {
                    label: index for index, label in enumerate(vocabulary)
                },
            },
        )

    @staticmethod
    def _encode_rows(tokenizer, rows, config, plan):
        mapping = plan.manifest["task"]["row_mapping"]
        input_field = mapping["input"]["field"]
        target_field = mapping["target"]["field"]
        contract = plan.manifest["output_contract"]
        label_to_id = contract["label_to_id"]
        multilabel = contract["type"] == "multilabel_classification"
        encoded_rows = []
        for row in rows:
            encoded = tokenizer(
                row[input_field],
                truncation=True,
                max_length=config["max_sequence_length"],
            )
            if multilabel:
                labels = [0.0] * len(label_to_id)
                for label in row[target_field]:
                    labels[label_to_id[str(label)]] = 1.0
                encoded["labels"] = labels
            else:
                encoded["labels"] = label_to_id[str(row[target_field])]
            encoded_rows.append(dict(encoded))
        return encoded_rows

    def _execute(
        self,
        *,
        plan: TrainingPlan,
        training_input: TrainingInput,
        base_model_path: Path,
        destination: Path,
        seed: int,
    ) -> dict[str, float | None]:
        import torch
        import transformers

        config = plan.normalized_config
        output_contract = plan.manifest["output_contract"]
        labels = output_contract["labels"]
        label_to_id = output_contract["label_to_id"]
        id_to_label = {index: label for label, index in label_to_id.items()}
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model_config = transformers.AutoConfig.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            num_labels=len(labels),
            label2id=label_to_id,
            id2label=id_to_label,
        )
        if output_contract["type"] == "multilabel_classification":
            model_config.problem_type = "multi_label_classification"
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            config=model_config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        train_rows = self._encode_rows(tokenizer, training_input.rows, config, plan)
        validation_rows = self._encode_rows(
            tokenizer,
            training_input.validation_rows,
            config,
            plan,
        )
        collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        metrics = execute_torch_training(
            torch=torch,
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            collator=collator,
            config=config,
            seed=seed,
        )
        model_dir = destination / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        write_json(destination / "label_mapping.json", output_contract)
        return metrics


class HuggingFaceTokenTrainer(_HuggingFaceTaskTrainer):
    key = "huggingface_token"
    recipe_keys = ("transformer_token_classification",)

    def plan(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan:
        normalized, recipe_version = validate_recipe(
            self.recipe_keys, recipe_key, config
        )
        task_kind = validate_task_kind(recipe_key, training_input)
        input_field = normalized["input_field"]
        target_field = normalized["target_field"]
        if not training_input.rows:
            raise ValidationError("Training input must contain at least one row")

        def validate_rows(rows) -> set[str]:
            discovered_labels: set[str] = set()
            for index, row in enumerate(rows):
                tokens = row.get(input_field)
                labels = row.get(target_field)
                if (
                    not isinstance(tokens, (list, tuple))
                    or not tokens
                    or any(not isinstance(token, str) or not token for token in tokens)
                ):
                    raise ValidationError(
                        f"Training row {index} field '{input_field}' must be a token list"
                    )
                if not isinstance(labels, (list, tuple)) or len(labels) != len(tokens):
                    raise ValidationError(
                        f"Training row {index} token and label counts must match"
                    )
                discovered_labels.update(
                    _label_key(value, row_index=index, field=target_field)
                    for value in labels
                )
            return discovered_labels

        discovered = validate_rows(training_input.rows)
        vocabulary = _label_vocabulary(
            discovered,
            training_input.label_vocabulary,
        )
        validation_labels = validate_rows(training_input.validation_rows)
        unknown_validation = validation_labels - set(vocabulary)
        if unknown_validation:
            raise ValidationError(
                "Validation labels are absent from the training label vocabulary: "
                + ", ".join(sorted(unknown_validation))
            )
        return base_plan(
            trainer_key=self.key,
            trainer_version=self.trainer_version,
            recipe_key=recipe_key,
            recipe_version=recipe_version,
            normalized_config=normalized,
            training_input=training_input,
            task_kind=task_kind,
            seed=seed,
            row_mapping={
                "input": {"field": input_field, "type": "tokens"},
                "target": {
                    "field": target_field,
                    "type": "word_labels",
                    "subtoken_policy": "first_subtoken",
                    "ignored_label_id": -100,
                },
            },
            output_contract={
                "type": "token_labels",
                "labels": list(vocabulary),
                "label_to_id": {
                    label: index for index, label in enumerate(vocabulary)
                },
            },
        )

    @staticmethod
    def _encode_rows(tokenizer, rows, config, plan):
        mapping = plan.manifest["task"]["row_mapping"]
        input_field = mapping["input"]["field"]
        target_field = mapping["target"]["field"]
        label_to_id = plan.manifest["output_contract"]["label_to_id"]
        encoded_rows = []
        for row_index, row in enumerate(rows):
            encoded = tokenizer(
                list(row[input_field]),
                is_split_into_words=True,
                truncation=True,
                max_length=config["max_sequence_length"],
            )
            word_ids = encoded.word_ids()
            if word_ids is None:
                raise ValidationError("Token classification requires a fast tokenizer")
            aligned: list[int] = []
            prior_word = None
            for word_id in word_ids:
                if word_id is None or word_id == prior_word:
                    aligned.append(-100)
                elif word_id >= len(row[target_field]):
                    raise ValidationError(
                        f"Tokenizer alignment failed for training row {row_index}"
                    )
                else:
                    aligned.append(label_to_id[str(row[target_field][word_id])])
                prior_word = word_id
            item = dict(encoded)
            item["labels"] = aligned
            encoded_rows.append(item)
        return encoded_rows

    def _execute(
        self,
        *,
        plan: TrainingPlan,
        training_input: TrainingInput,
        base_model_path: Path,
        destination: Path,
        seed: int,
    ) -> dict[str, float | None]:
        import torch
        import transformers

        config = plan.normalized_config
        output_contract = plan.manifest["output_contract"]
        labels = output_contract["labels"]
        label_to_id = output_contract["label_to_id"]
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model_config = transformers.AutoConfig.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            num_labels=len(labels),
            label2id=label_to_id,
            id2label={index: label for label, index in label_to_id.items()},
        )
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            base_model_path,
            config=model_config,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        train_rows = self._encode_rows(tokenizer, training_input.rows, config, plan)
        validation_rows = self._encode_rows(
            tokenizer,
            training_input.validation_rows,
            config,
            plan,
        )
        collator = transformers.DataCollatorForTokenClassification(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
        )
        metrics = execute_torch_training(
            torch=torch,
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            collator=collator,
            config=config,
            seed=seed,
        )
        model_dir = destination / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        write_json(destination / "label_mapping.json", output_contract)
        return metrics


class HuggingFaceSpanTrainer(_HuggingFaceTaskTrainer):
    key = "huggingface_span"
    recipe_keys = ("transformer_span_extraction",)

    @staticmethod
    def _span(row: Mapping, field: str, text: str, row_index: int) -> tuple[int, int]:
        target = row.get(field)
        if not isinstance(target, Mapping):
            raise ValidationError(
                f"Training row {row_index} field '{field}' must be a span object"
            )
        start = target.get("start_char", target.get("start"))
        end = target.get("end_char", target.get("end"))
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(text)
        ):
            raise ValidationError(
                f"Training row {row_index} field '{field}' has invalid character offsets"
            )
        return start, end

    def plan(
        self,
        *,
        recipe_key: str,
        config: Mapping,
        training_input: TrainingInput,
        seed: int,
    ) -> TrainingPlan:
        normalized, recipe_version = validate_recipe(
            self.recipe_keys, recipe_key, config
        )
        task_kind = validate_task_kind(recipe_key, training_input)
        input_field = normalized["input_field"]
        target_field = normalized["target_field"]
        rows = (*training_input.rows, *training_input.validation_rows)
        if not training_input.rows:
            raise ValidationError("Training input must contain at least one row")
        for index, row in enumerate(rows):
            text = _nonempty_text(row, input_field, index)
            self._span(row, target_field, text, index)
        return base_plan(
            trainer_key=self.key,
            trainer_version=self.trainer_version,
            recipe_key=recipe_key,
            recipe_version=recipe_version,
            normalized_config=normalized,
            training_input=training_input,
            task_kind=task_kind,
            seed=seed,
            row_mapping={
                "input": {"field": input_field, "type": "text"},
                "target": {
                    "field": target_field,
                    "type": "character_span",
                    "end_offset": "exclusive",
                },
            },
            output_contract={
                "type": "token_span",
                "start": "start_logits",
                "end": "end_logits",
            },
        )

    def _encode_rows(self, tokenizer, rows, config, plan):
        mapping = plan.manifest["task"]["row_mapping"]
        input_field = mapping["input"]["field"]
        target_field = mapping["target"]["field"]
        encoded_rows = []
        for row_index, row in enumerate(rows):
            text = row[input_field]
            start_char, end_char = self._span(
                row,
                target_field,
                text,
                row_index,
            )
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=config["max_sequence_length"],
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            start_token = None
            end_token = None
            for token_index, (token_start, token_end) in enumerate(offsets):
                if token_start == token_end:
                    continue
                if start_token is None and token_start <= start_char < token_end:
                    start_token = token_index
                if token_start < end_char <= token_end:
                    end_token = token_index
                    break
            if start_token is None or end_token is None:
                raise ValidationError(
                    f"Training row {row_index} span was truncated or cannot be token-aligned"
                )
            item = dict(encoded)
            item["start_positions"] = start_token
            item["end_positions"] = end_token
            encoded_rows.append(item)
        return encoded_rows

    def _execute(
        self,
        *,
        plan: TrainingPlan,
        training_input: TrainingInput,
        base_model_path: Path,
        destination: Path,
        seed: int,
    ) -> dict[str, float | None]:
        import torch
        import transformers

        config = plan.normalized_config
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        model = transformers.AutoModelForQuestionAnswering.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        train_rows = self._encode_rows(tokenizer, training_input.rows, config, plan)
        validation_rows = self._encode_rows(
            tokenizer,
            training_input.validation_rows,
            config,
            plan,
        )
        collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        metrics = execute_torch_training(
            torch=torch,
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            collator=collator,
            config=config,
            seed=seed,
        )
        model_dir = destination / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        return metrics


def register_huggingface_task_trainers(registry=None) -> None:
    from al_medlit.training.trainers.contracts import trainer_plugins

    selected = registry or trainer_plugins
    selected.register(HuggingFaceSequenceTrainer(), replace=True)
    selected.register(HuggingFaceTokenTrainer(), replace=True)
    selected.register(HuggingFaceSpanTrainer(), replace=True)
