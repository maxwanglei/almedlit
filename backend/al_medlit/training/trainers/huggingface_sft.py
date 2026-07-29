"""Lazy causal and seq2seq supervised fine-tuning worker plugins."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from al_medlit.core.exceptions import ValidationError
from al_medlit.training.recipe_contracts import Parameterization
from al_medlit.training.recipe_registry import training_recipes
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


def _template_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _render_rows(
    rows: Sequence[Mapping],
    *,
    prompt_field: str,
    completion_field: str,
    input_template: str,
    completion_template: str,
) -> list[tuple[str, str]]:
    rendered: list[tuple[str, str]] = []
    for index, row in enumerate(rows):
        prompt = row.get(prompt_field)
        completion = row.get(completion_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValidationError(
                f"Training row {index} field '{prompt_field}' must be non-empty text"
            )
        if not isinstance(completion, str) or not completion.strip():
            raise ValidationError(
                f"Training row {index} field '{completion_field}' must be non-empty text"
            )
        rendered.append(
            (
                input_template.format(prompt=prompt),
                completion_template.format(completion=completion),
            )
        )
    return rendered


class _HuggingFaceSftTrainer:
    key: str
    recipe_keys: tuple[str, ...]
    architecture: str
    trainer_version = "1"

    def preflight(self, recipe_key: str | None = None) -> TrainerPreflight:
        return preflight_for_recipe(self.recipe_keys, recipe_key)

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
        if not training_input.rows:
            raise ValidationError("Training input must contain at least one row")
        _render_rows(
            (*training_input.rows, *training_input.validation_rows),
            prompt_field=normalized["prompt_field"],
            completion_field=normalized["completion_field"],
            input_template=normalized["input_template"],
            completion_template=normalized["completion_template"],
        )
        descriptor = training_recipes.get(recipe_key)
        parameterization = descriptor.parameterization.value
        if parameterization in {
            Parameterization.LORA.value,
            Parameterization.QLORA.value,
        } and (
            normalized["lora_rank"] is None
            or normalized["lora_alpha"] is None
            or normalized["lora_dropout"] is None
        ):
            raise ValidationError("LoRA and QLoRA recipes require complete adapter settings")
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
                "input": {
                    "field": normalized["prompt_field"],
                    "type": "text",
                    "template_version": normalized["prompt_template_version"],
                    "template_sha256": _template_hash(normalized["input_template"]),
                },
                "target": {
                    "field": normalized["completion_field"],
                    "type": "text",
                    "template_version": normalized["prompt_template_version"],
                    "template_sha256": _template_hash(
                        normalized["completion_template"]
                    ),
                },
            },
            output_contract={
                "type": "language_model",
                "architecture": self.architecture,
                "parameterization": parameterization,
                "objective": (
                    "completion_only_causal_lm"
                    if self.architecture == "causal_lm"
                    else "sequence_to_sequence"
                ),
                "prompt_tokens_in_loss": False,
            },
        )

    @staticmethod
    def _load_tokenizer(transformers, base_model_path: Path):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValidationError(
                    "SFT tokenizer must define either a pad token or an EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(
        self,
        *,
        torch,
        transformers,
        base_model_path: Path,
        plan: TrainingPlan,
    ):
        parameterization = plan.manifest["output_contract"]["parameterization"]
        precision = plan.normalized_config["mixed_precision"]
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        model_class = (
            transformers.AutoModelForCausalLM
            if self.architecture == "causal_lm"
            else transformers.AutoModelForSeq2SeqLM
        )
        load_kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
            "torch_dtype": dtype,
        }
        dispatched = False
        if parameterization == Parameterization.QLORA.value:
            quantization = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            load_kwargs.update(
                {
                    "quantization_config": quantization,
                    "device_map": {"": torch.cuda.current_device()},
                }
            )
            dispatched = True
        model = model_class.from_pretrained(base_model_path, **load_kwargs)

        if parameterization in {
            Parameterization.LORA.value,
            Parameterization.QLORA.value,
        }:
            import peft

            if parameterization == Parameterization.QLORA.value:
                model = peft.prepare_model_for_kbit_training(model)
            task_type = (
                peft.TaskType.CAUSAL_LM
                if self.architecture == "causal_lm"
                else peft.TaskType.SEQ_2_SEQ_LM
            )
            adapter_config = peft.LoraConfig(
                r=plan.normalized_config["lora_rank"],
                lora_alpha=plan.normalized_config["lora_alpha"],
                lora_dropout=plan.normalized_config["lora_dropout"],
                target_modules=plan.normalized_config["lora_target_modules"],
                bias="none",
                task_type=task_type,
            )
            model = peft.get_peft_model(model, adapter_config)
        return model, dispatched

    @staticmethod
    def _causal_rows(tokenizer, rendered_rows, max_length: int) -> list[dict]:
        encoded_rows = []
        for index, (prompt, completion) in enumerate(rendered_rows):
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            completion_ids = tokenizer(
                completion,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]
            if tokenizer.eos_token_id is not None:
                completion_ids = [*completion_ids, tokenizer.eos_token_id]
            if not completion_ids:
                raise ValidationError(
                    f"Training row {index} completion produced no target tokens"
                )
            if len(completion_ids) >= max_length:
                completion_ids = completion_ids[:max_length]
                prompt_ids = []
            else:
                prompt_ids = prompt_ids[-(max_length - len(completion_ids)) :]
            input_ids = [*prompt_ids, *completion_ids]
            encoded_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": [-100] * len(prompt_ids) + completion_ids,
                }
            )
        return encoded_rows

    @staticmethod
    def _seq2seq_rows(tokenizer, rendered_rows, max_length: int) -> list[dict]:
        encoded_rows = []
        for prompt, completion in rendered_rows:
            encoded = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
            )
            target = tokenizer(
                text_target=completion,
                truncation=True,
                max_length=max_length,
            )
            item = dict(encoded)
            item["labels"] = target["input_ids"]
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

        tokenizer = self._load_tokenizer(transformers, base_model_path)
        model, dispatched = self._load_model(
            torch=torch,
            transformers=transformers,
            base_model_path=base_model_path,
            plan=plan,
        )
        config = plan.normalized_config
        rendering = {
            "prompt_field": config["prompt_field"],
            "completion_field": config["completion_field"],
            "input_template": config["input_template"],
            "completion_template": config["completion_template"],
        }
        rendered_train = _render_rows(training_input.rows, **rendering)
        rendered_validation = _render_rows(
            training_input.validation_rows,
            **rendering,
        )
        encoder = (
            self._causal_rows
            if self.architecture == "causal_lm"
            else self._seq2seq_rows
        )
        train_rows = encoder(
            tokenizer,
            rendered_train,
            config["max_sequence_length"],
        )
        validation_rows = encoder(
            tokenizer,
            rendered_validation,
            config["max_sequence_length"],
        )
        collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            padding=True,
        )
        metrics = execute_torch_training(
            torch=torch,
            model=model,
            train_rows=train_rows,
            validation_rows=validation_rows,
            collator=collator,
            config=config,
            seed=seed,
            dispatched=dispatched,
        )
        parameterization = plan.manifest["output_contract"]["parameterization"]
        output_dir = (
            destination / "model"
            if parameterization == Parameterization.FULL.value
            else destination / "adapter"
        )
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(destination / "tokenizer")
        write_json(
            destination / "sft_contract.json",
            {
                "architecture": self.architecture,
                "parameterization": parameterization,
                "prompt_template_version": config["prompt_template_version"],
                "input_template_sha256": _template_hash(config["input_template"]),
                "completion_template_sha256": _template_hash(
                    config["completion_template"]
                ),
                "prompt_tokens_in_loss": False,
            },
        )
        return metrics

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
        with tempfile.TemporaryDirectory(prefix="al-medlit-hf-sft-") as temporary:
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


class HuggingFaceCausalSftTrainer(_HuggingFaceSftTrainer):
    key = "huggingface_causal_sft"
    architecture = "causal_lm"
    recipe_keys = (
        "causal_lm_sft_full",
        "causal_lm_sft_lora",
        "causal_lm_sft_qlora",
    )


class HuggingFaceSeq2SeqSftTrainer(_HuggingFaceSftTrainer):
    key = "huggingface_seq2seq_sft"
    architecture = "seq2seq_lm"
    recipe_keys = (
        "seq2seq_lm_sft_full",
        "seq2seq_lm_sft_lora",
        "seq2seq_lm_sft_qlora",
    )


def register_huggingface_sft_trainers(registry=None) -> None:
    from al_medlit.training.trainers.contracts import trainer_plugins

    selected = registry or trainer_plugins
    selected.register(HuggingFaceCausalSftTrainer(), replace=True)
    selected.register(HuggingFaceSeq2SeqSftTrainer(), replace=True)
