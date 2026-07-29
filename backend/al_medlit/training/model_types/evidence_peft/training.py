"""Durable LoRA/QLoRA training for structured Evidence BIO predictions.

The worker consumes only a locally materialized, checksum-verified base-model
package.  It never resolves a mutable model identifier or downloads code.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from al_medlit.training.evaluation.evidence import (
    EvidenceBlockEvaluator,
    EvidenceEvaluationContext,
    EvidenceEvaluationExample,
    blocks_from_bio,
    canonical_evaluation_content_fingerprint,
    canonical_fingerprint,
)
from al_medlit.training.model_types.evidence_neural import documents_from_window_rows
from al_medlit.training.model_types.evidence_peft.adapter import (
    PeftAdapterBundle,
    create_peft_adapter,
    load_peft_adapter,
)
from al_medlit.training.model_types.evidence_peft.config import (
    EvidenceLoRAConfig,
    EvidencePeftConfig,
    EvidenceQLoRAConfig,
    ImmutableBaseModelReference,
)
from al_medlit.training.model_types.evidence_peft.preflight import (
    build_qlora_quantization_config,
    require_peft_preflight,
)

VALID_LABELS = {"O", "B", "I"}


@dataclass(frozen=True, slots=True)
class PeftEvidenceExample:
    document_id: str
    target_id: str
    target_text: str
    sentence_ordinals: tuple[int, ...]
    sentences: tuple[str, ...]
    gold_labels: tuple[str, ...] | None
    prompt: str


@dataclass(frozen=True, slots=True)
class PeftTrainingResult:
    bundle: PeftAdapterBundle
    tokenizer: object
    history: tuple[dict, ...]
    best_epoch: int
    selection_value: float | None
    selection_tiebreaker: float | None
    evaluations: dict[str, dict]
    device: str
    adapter_parameter_count: int
    trainable_parameter_count: int


def train_peft_adapter(
    config: EvidenceLoRAConfig | EvidenceQLoRAConfig,
    *,
    base_model_root: str | Path,
    base_reference: ImmutableBaseModelReference,
    training_rows: Sequence[dict],
    validation_rows: Sequence[dict],
    test_rows: Sequence[dict],
    training_dataset_hash: str,
    validation_dataset_hash: str,
) -> PeftTrainingResult:
    """Fit adapter weights and evaluate the held-out test split once.

    Checkpoint choice uses validation macro block-IoU F1@0.50, exact-block F1,
    then the earlier epoch.  Test predictions are generated only after the
    selected adapter state has been restored.
    """

    preflight = require_peft_preflight(config)
    torch, tokenizer, base_model, runtime_device = _load_base_model(
        config,
        Path(base_model_root),
    )
    _seed_everything(torch, config)
    bundle = create_peft_adapter(base_model, config, base_reference)
    if not isinstance(config, EvidenceQLoRAConfig):
        bundle.model.to(torch.device(runtime_device))

    train_examples = prepare_peft_examples(training_rows, tokenizer, config)
    validation_examples = prepare_peft_examples(validation_rows, tokenizer, config)
    test_examples = prepare_peft_examples(test_rows, tokenizer, config)
    if not train_examples:
        raise ValueError("PEFT training contains no reviewed Evidence labels")
    if not validation_examples or not test_examples:
        raise ValueError("PEFT training requires non-empty validation and test splits")

    encoded_training = tuple(
        _encode_supervised_example(tokenizer, example, config) for example in train_examples
    )
    parameters = [parameter for parameter in bundle.model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("PEFT did not expose any trainable adapter parameters")
    optimizer = _optimizer(torch, config, parameters)
    batch_count = math.ceil(len(encoded_training) / config.batch_size)
    updates_per_epoch = math.ceil(batch_count / config.gradient_accumulation_steps)
    total_updates = max(1, updates_per_epoch * config.epochs)
    warmup_updates = int(total_updates * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _schedule_multiplier(
            config.scheduler,
            step,
            total_updates,
            warmup_updates,
        ),
    )

    try:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("PEFT dependency became unavailable after preflight") from exc

    best_state = None
    best_epoch = 0
    best_primary = -math.inf
    best_exact = -math.inf
    best_validation: dict | None = None
    stale_epochs = 0
    history: list[dict] = []
    optimizer_steps = 0
    for epoch in range(1, config.epochs + 1):
        started = time.monotonic()
        bundle.model.train()
        order = list(range(len(encoded_training)))
        random.Random(config.seed + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        supervised_tokens = 0
        for batch_number, offset in enumerate(
            range(0, len(order), config.batch_size),
            start=1,
        ):
            batch = [
                encoded_training[index] for index in order[offset : offset + config.batch_size]
            ]
            tensors, token_count = _collate_supervised(
                torch,
                tokenizer,
                batch,
                device=runtime_device,
            )
            output = bundle.model(**tensors)
            loss = output.loss if hasattr(output, "loss") else output["loss"]
            if loss is None or not bool(torch.isfinite(loss)):
                raise RuntimeError("PEFT training produced a non-finite loss")
            (loss / config.gradient_accumulation_steps).backward()
            epoch_loss += float(loss.detach().to("cpu"))
            supervised_tokens += token_count
            should_update = (
                batch_number % config.gradient_accumulation_steps == 0
                or offset + config.batch_size >= len(order)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config.gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        validation, validity = evaluate_peft_examples(
            bundle,
            tokenizer,
            validation_examples,
            config=config,
            split="validation",
            training_dataset_hash=training_dataset_hash,
            validation_dataset_hash=validation_dataset_hash,
        )
        primary_value = validation["metrics"].get("macro_block_iou_f1_0_50")
        exact_value = validation["metrics"].get("macro_exact_block_f1")
        primary = float(primary_value) if primary_value is not None else -math.inf
        exact = float(exact_value) if exact_value is not None else -math.inf
        improved = (
            best_state is None
            or primary > best_primary
            or (
                math.isclose(primary, best_primary, rel_tol=0.0, abs_tol=1e-12)
                and exact > best_exact
            )
        )
        if improved:
            state = get_peft_model_state_dict(bundle.model)
            best_state = {name: tensor.detach().to("cpu").clone() for name, tensor in state.items()}
            best_epoch = epoch
            best_primary = primary
            best_exact = exact
            best_validation = validation
            stale_epochs = 0
        else:
            stale_epochs += 1
        elapsed = max(time.monotonic() - started, 1e-9)
        history.append(
            {
                "phase": "train",
                "split": "train",
                "epoch": epoch,
                "step": optimizer_steps,
                "values": {
                    "token_loss": epoch_loss / batch_count,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "tokens_per_second": supervised_tokens / elapsed,
                    "structured_output_validity": validity,
                    "validation_macro_block_iou_f1_0_50": primary_value,
                    "validation_macro_exact_block_f1": exact_value,
                },
            }
        )
        if config.early_stopping_patience > 0 and stale_epochs >= config.early_stopping_patience:
            break

    if best_state is None or best_validation is None:
        raise RuntimeError("PEFT training did not produce a validation-selected adapter")
    set_peft_model_state_dict(bundle.model, best_state)
    bundle.model.eval()
    # The test split is intentionally generated exactly once, after selection.
    test_evaluation, _test_validity = evaluate_peft_examples(
        bundle,
        tokenizer,
        test_examples,
        config=config,
        split="test",
        training_dataset_hash=training_dataset_hash,
        validation_dataset_hash=validation_dataset_hash,
    )
    adapter_parameter_count = sum(parameter.numel() for parameter in parameters)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in bundle.model.parameters() if parameter.requires_grad
    )
    return PeftTrainingResult(
        bundle=bundle,
        tokenizer=tokenizer,
        history=tuple(history),
        best_epoch=best_epoch,
        selection_value=(None if best_primary == -math.inf else best_primary),
        selection_tiebreaker=(None if best_exact == -math.inf else best_exact),
        evaluations={"validation": best_validation, "test": test_evaluation},
        device=str(preflight.requested_device),
        adapter_parameter_count=adapter_parameter_count,
        trainable_parameter_count=trainable_parameter_count,
    )


def prepare_peft_examples(
    rows: Iterable[dict],
    tokenizer,
    config: EvidencePeftConfig,
    *,
    include_unreviewed: bool = False,
    supervised: bool = True,
) -> tuple[PeftEvidenceExample, ...]:
    """Create sentence-boundary windows using the base model's exact tokenizer."""

    documents = documents_from_window_rows(rows)
    examples: list[PeftEvidenceExample] = []
    for document in documents:
        ordinals = document.sentence_ordinals or tuple(range(len(document.sentences)))
        labels = document.labels
        items = []
        for index, (ordinal, sentence) in enumerate(zip(ordinals, document.sentences, strict=True)):
            label = labels[index] if supervised and labels is not None else None
            if not include_unreviewed and label == "IGNORE":
                continue
            items.append((ordinal, sentence, label))
        start = 0
        while start < len(items):
            best_end = start
            for end in range(start + 1, len(items) + 1):
                selected = items[start:end]
                prompt = render_evidence_prompt(
                    document.target_text,
                    tuple(item[1] for item in selected),
                    target_conditioning=config.target_conditioning,
                )
                completion = (
                    render_evidence_completion(tuple(str(item[2]) for item in selected))
                    if selected[0][2] is not None
                    else None
                )
                if _encoded_length(tokenizer, prompt, completion) > config.max_sequence_length:
                    break
                best_end = end
            if best_end == start:
                raise ValueError(
                    "A PEFT Evidence sentence cannot fit the immutable base tokenizer context"
                )
            selected = items[start:best_end]
            sentences = tuple(item[1] for item in selected)
            gold = tuple(str(item[2]) for item in selected) if selected[0][2] is not None else None
            examples.append(
                PeftEvidenceExample(
                    document_id=document.document_id,
                    target_id=str(document.target_id),
                    target_text=document.target_text,
                    sentence_ordinals=tuple(item[0] for item in selected),
                    sentences=sentences,
                    gold_labels=gold,
                    prompt=render_evidence_prompt(
                        document.target_text,
                        sentences,
                        target_conditioning=config.target_conditioning,
                    ),
                )
            )
            start = best_end
    return tuple(examples)


def evaluate_peft_examples(
    bundle: PeftAdapterBundle,
    tokenizer,
    examples: Sequence[PeftEvidenceExample],
    *,
    config: EvidencePeftConfig,
    split: str,
    training_dataset_hash: str,
    validation_dataset_hash: str | None,
) -> tuple[dict, float]:
    predictions, valid_count = predict_peft_examples(bundle, tokenizer, examples, config=config)
    groups: dict[tuple[str, str], dict] = {}
    for example, predicted in zip(examples, predictions, strict=True):
        if example.gold_labels is None:
            raise ValueError("Evaluation examples require reviewed labels")
        group = groups.setdefault(
            (example.document_id, example.target_id),
            {
                "gold": {},
                "votes": {},
                "sentences": {},
                "target_text": example.target_text,
            },
        )
        if group["target_text"] != example.target_text:
            raise ValueError("Overlapping PEFT windows disagree on target text")
        for ordinal, gold, predicted_label in zip(
            example.sentence_ordinals,
            example.gold_labels,
            predicted,
            strict=True,
        ):
            existing = group["gold"].get(ordinal)
            if existing is not None and existing != gold:
                raise ValueError("Overlapping PEFT windows disagree on gold Evidence labels")
            group["gold"][ordinal] = gold
            group["votes"].setdefault(ordinal, Counter())[predicted_label] += 1
        for ordinal, sentence_text in zip(
            example.sentence_ordinals,
            example.sentences,
            strict=True,
        ):
            existing_text = group["sentences"].get(ordinal)
            if existing_text is not None and existing_text != sentence_text:
                raise ValueError("Overlapping PEFT windows disagree on sentence text")
            group["sentences"][ordinal] = sentence_text

    evaluation_examples = []
    label_order = {"O": 0, "B": 1, "I": 2}
    for (document_id, target_id), group in sorted(groups.items()):
        ordinals = sorted(group["gold"])
        gold = tuple(group["gold"][ordinal] for ordinal in ordinals)
        predicted = tuple(
            max(
                group["votes"][ordinal],
                key=lambda label: (group["votes"][ordinal][label], -label_order[label]),
            )
            for ordinal in ordinals
        )
        evaluation_examples.append(
            EvidenceEvaluationExample(
                document_id=document_id,
                target_version_id=target_id,
                sentence_count=len(ordinals),
                sentence_ordinals=tuple(ordinals),
                sentence_texts=tuple(group["sentences"][ordinal] for ordinal in ordinals),
                target_text=group["target_text"],
                reference_blocks=blocks_from_bio(gold),
                predicted_labels=predicted,
            )
        )
    if not evaluation_examples:
        raise ValueError(f"PEFT {split} evaluation contains no reviewed labels")
    dataset_fingerprint = canonical_evaluation_content_fingerprint(evaluation_examples)
    context = EvidenceEvaluationContext(
        evaluation_dataset_hash=dataset_fingerprint,
        target_version_ids=tuple(
            sorted({example.target_version_id for example in evaluation_examples})
        ),
        training_dataset_hash=training_dataset_hash,
        validation_dataset_hash=validation_dataset_hash,
        preprocessing_version="evidence-peft-json-v1:"
        + canonical_fingerprint(
            {
                "base_model_asset_id": config.base_model_asset_id,
                "max_sequence_length": config.max_sequence_length,
                "prompt_template_version": config.prompt_template_version,
            }
        ),
        decoder_config={
            "kind": "structured_json_bio",
            "overlap": "majority_vote",
            "invalid_output": "all_o",
        },
    )
    report = EvidenceBlockEvaluator().evaluate(
        evaluation_examples,
        context=context,
        split=split,
        bootstrap_samples=200,
        bootstrap_seed=config.seed,
    )
    payload = {
        **report.to_record_payload(),
        "report": report.model_dump(mode="json"),
    }
    validity = valid_count / len(examples) if examples else 0.0
    payload["diagnostics"] = {
        **payload.get("diagnostics", {}),
        "structured_output_validity": validity,
        "structured_output_valid_count": valid_count,
        "structured_output_count": len(examples),
    }
    return payload, validity


def predict_peft_examples(
    bundle: PeftAdapterBundle,
    tokenizer,
    examples: Sequence[PeftEvidenceExample],
    *,
    config: EvidencePeftConfig,
) -> tuple[tuple[tuple[str, ...], ...], int]:
    import torch

    model = bundle.model
    model.eval()
    predictions: list[tuple[str, ...]] = []
    valid_count = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for example in examples:
            prompt_ids = _token_ids(tokenizer, example.prompt, add_special_tokens=True)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_ids = generated[0, len(prompt_ids) :].detach().to("cpu").tolist()
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            labels = parse_evidence_completion(decoded, expected_count=len(example.sentences))
            if labels is None:
                labels = ("O",) * len(example.sentences)
            else:
                valid_count += 1
            predictions.append(labels)
    return tuple(predictions), valid_count


def load_peft_runtime(
    adapter_root: str | Path,
    *,
    base_model_root: str | Path,
    base_reference: ImmutableBaseModelReference,
) -> tuple[PeftAdapterBundle, object]:
    config_path = Path(adapter_root) / "al-medlit-training-config.json"
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    config: EvidenceLoRAConfig | EvidenceQLoRAConfig
    if raw.get("model_kind") == "lora":
        config = EvidenceLoRAConfig.model_validate(raw)
    elif raw.get("model_kind") == "qlora":
        config = EvidenceQLoRAConfig.model_validate(raw)
    else:
        raise ValueError("Unsupported PEFT adapter model_kind")
    _torch, tokenizer, base_model, _runtime_device = _load_base_model(
        config,
        Path(base_model_root),
    )
    return (
        load_peft_adapter(
            adapter_root,
            base_model=base_model,
            base_reference=base_reference,
            trainable=False,
        ),
        tokenizer,
    )


def render_evidence_prompt(
    target_text: str,
    sentences: Sequence[str],
    *,
    target_conditioning: bool = True,
) -> str:
    normalized_sentences = [" ".join(sentence.split()) for sentence in sentences]
    numbered = "\n".join(
        f"{index}: {sentence}" for index, sentence in enumerate(normalized_sentences)
    )
    target_line = (
        f"Target: {' '.join(target_text.split())}\n" if target_conditioning else ""
    )
    return (
        "Label each sentence for the requested medical-literature Evidence Block.\n"
        "Use exactly one BIO label per sentence: O, B, or I.\n"
        'Return only a JSON object with this shape: {"labels":["O","B"]}.\n'
        f"{target_line}"
        f"Sentences:\n{numbered}\nJSON:\n"
    )


def render_evidence_completion(labels: Sequence[str]) -> str:
    if any(label not in VALID_LABELS for label in labels):
        raise ValueError("PEFT completion labels must be O, B, or I")
    return json.dumps({"labels": list(labels)}, separators=(",", ":"))


def parse_evidence_completion(text: str, *, expected_count: int) -> tuple[str, ...] | None:
    try:
        payload = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"labels"}:
        return None
    labels = payload["labels"]
    if (
        not isinstance(labels, list)
        or len(labels) != expected_count
        or any(not isinstance(label, str) or label not in VALID_LABELS for label in labels)
    ):
        return None
    return tuple(labels)


def _load_base_model(config: EvidencePeftConfig, root: Path):
    require_peft_preflight(config)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - guarded by preflight
        raise RuntimeError("Transformers dependency became unavailable after preflight") from exc
    if root.is_symlink() or not root.is_dir():
        raise ValueError("The immutable base-model package is not a regular directory")
    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("The immutable causal base model requires an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    runtime_device = "npu" if config.device == "ascend" else config.device
    if isinstance(config, EvidenceQLoRAConfig):
        load_kwargs["quantization_config"] = build_qlora_quantization_config(config)
        load_kwargs["device_map"] = {"": torch.cuda.current_device()}
    base_model = AutoModelForCausalLM.from_pretrained(root, **load_kwargs)
    if not isinstance(config, EvidenceQLoRAConfig):
        base_model.to(torch.device(runtime_device))
    return torch, tokenizer, base_model, runtime_device


def _encode_supervised_example(tokenizer, example, config: EvidencePeftConfig) -> dict:
    if example.gold_labels is None:
        raise ValueError("Supervised PEFT examples require gold labels")
    prompt_ids = _token_ids(tokenizer, example.prompt, add_special_tokens=True)
    completion = render_evidence_completion(example.gold_labels)
    completion_ids = _token_ids(tokenizer, completion, add_special_tokens=False)
    completion_ids.append(int(tokenizer.eos_token_id))
    input_ids = prompt_ids + completion_ids
    if len(input_ids) > config.max_sequence_length:
        raise ValueError("Prepared PEFT example exceeds max_sequence_length")
    return {
        "input_ids": input_ids,
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def _collate_supervised(torch, tokenizer, batch, *, device: str):
    maximum = max(len(item["input_ids"]) for item in batch)
    input_rows = []
    attention_rows = []
    label_rows = []
    supervised_tokens = 0
    for item in batch:
        padding = maximum - len(item["input_ids"])
        input_rows.append(item["input_ids"] + [int(tokenizer.pad_token_id)] * padding)
        attention_rows.append([1] * len(item["input_ids"]) + [0] * padding)
        label_rows.append(item["labels"] + [-100] * padding)
        supervised_tokens += sum(label != -100 for label in item["labels"])
    tensors = {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_rows, dtype=torch.long, device=device),
        "labels": torch.tensor(label_rows, dtype=torch.long, device=device),
    }
    return tensors, supervised_tokens


def _optimizer(torch, config: EvidencePeftConfig, parameters):
    if config.optimizer == "paged_adamw_8bit":
        try:
            from bitsandbytes.optim import PagedAdamW8bit
        except ImportError as exc:  # pragma: no cover - guarded by QLoRA preflight
            raise RuntimeError("bitsandbytes optimizer became unavailable") from exc
        return PagedAdamW8bit(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _schedule_multiplier(
    scheduler: str,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-12, (step + 1) / warmup_steps)
    progress = min(
        1.0,
        max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)),
    )
    if scheduler == "constant":
        return 1.0
    if scheduler == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0 - progress


def _seed_everything(torch, config: EvidencePeftConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available() and hasattr(npu, "manual_seed_all"):
        npu.manual_seed_all(config.seed)
    if config.deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _encoded_length(tokenizer, prompt: str, completion: str | None) -> int:
    length = len(_token_ids(tokenizer, prompt, add_special_tokens=True))
    if completion is not None:
        length += len(_token_ids(tokenizer, completion, add_special_tokens=False)) + 1
    return length


def _token_ids(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, truncation=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]
