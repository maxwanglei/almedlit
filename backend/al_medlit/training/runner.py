"""Portable ML job-bundle runner.

The runner deliberately has no database or object-storage dependency.  A
bundle contains immutable inputs and a JSON job contract; the runner verifies
those inputs and writes a checksummed result manifest for either the local
or SSH/Slurm orchestrator to collect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from al_medlit.core.archive import (
    ArchiveExtractionError,
    ArchiveExtractionLimits,
    extract_zip_bounded,
)

RUNNER_SCHEMA_VERSION = "al-medlit-job-v1"
RESULT_SCHEMA_VERSION = "al-medlit-job-result-v1"
NEURAL_MODEL_TYPES = {"evidence_bilstm", "evidence_cnn"}
PEFT_MODEL_TYPES = {"evidence_lora", "evidence_qlora"}


class RunnerError(RuntimeError):
    pass


def _resolve_torch_device(torch, requested: str) -> tuple[str, str]:
    """Return the persisted logical device and the PyTorch runtime device."""

    selected = requested
    if selected == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            selected = "mps"
        else:
            try:
                import torch_npu  # noqa: F401

                selected = "ascend"
            except ImportError:
                selected = "cpu"
    if selected == "cuda" and not torch.cuda.is_available():
        raise RunnerError("CUDA was requested but is not available")
    if selected == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RunnerError("MPS was requested but is not available")
    if selected == "ascend":
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RunnerError("Ascend was requested but torch_npu is unavailable") from exc
        return selected, "npu"
    return selected, selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _contained_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise RunnerError(f"Bundle path escapes its root: {relative!r}")
    return candidate


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def _append_json_line(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as target:
        target.write(canonical_json(value) + b"\n")


def _training_statistics(dataset_path: Path, target_version_id: int | None) -> dict:
    rows = 0
    sentences = 0
    labels: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    target_ids: set[int] = set()
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"Invalid dataset JSON on line {line_number}") from exc
            if row.get("schema_version") != "training-windows-v1":
                raise RunnerError(f"Unsupported dataset row on line {line_number}")
            row_target_id = int(row["target"]["id"])
            if target_version_id is not None and row_target_id != target_version_id:
                continue
            rows += 1
            target_ids.add(row_target_id)
            splits[str(row.get("split", "unknown"))] += 1
            for sentence in row.get("sentences", []):
                sentences += 1
                labels[str(sentence.get("label", "IGNORE"))] += 1
    if rows == 0:
        raise RunnerError("The immutable training export contains no rows for this job")
    return {
        "window_count": rows,
        "sentence_count": sentences,
        "label_counts": dict(sorted(labels.items())),
        "split_window_counts": dict(sorted(splits.items())),
        "target_version_ids_seen": sorted(target_ids),
    }


def _selected_training_rows(dataset_path: Path, target_version_id: int | None) -> list[dict]:
    rows = []
    with dataset_path.open(encoding="utf-8") as source:
        for raw_line in source:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if target_version_id is not None and int(row["target"]["id"]) != target_version_id:
                continue
            if row.get("split") == "train":
                rows.append(row)
    if not rows:
        raise RunnerError("The immutable export contains no training-split rows for this job")
    return rows


def _evaluation_rows(dataset_path: Path, target_version_id: int | None, split: str) -> list[dict]:
    rows = []
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if target_version_id is not None and int(row["target"]["id"]) != target_version_id:
                continue
            if row.get("split") == split:
                rows.append({**row, "_line_number": line_number})
    return rows


def _canonical_cohort_fingerprint(rows: list[dict]) -> str:
    """Fingerprint the exact labeled document/target cohort used by a run.

    Window overlap and JSONL ordering do not affect the result.  Exact target
    text, sentence ordinal/text, and gold label do, which prevents controlled
    experiment badges across silently different train/validation cohorts.
    """

    from al_medlit.training.evaluation.evidence import canonical_fingerprint

    groups: dict[tuple[str, str], dict] = {}
    for row_index, row in enumerate(rows):
        document_id = str(row.get("document_id", f"legacy-row-{row_index}"))
        target = row.get("target") or {}
        target_id = str(target.get("id", "unknown-target"))
        target_text = str(target.get("text", ""))
        group = groups.setdefault(
            (document_id, target_id),
            {"target_text": target_text, "sentences": {}},
        )
        if group["target_text"] != target_text:
            raise RunnerError("Overlapping cohort windows disagree on target text")
        for sentence_index, sentence in enumerate(row.get("sentences") or []):
            label = str(sentence.get("label", "IGNORE"))
            if label == "IGNORE" or sentence.get("reviewed") is False:
                continue
            ordinal = int(sentence.get("ordinal", sentence_index))
            value = {"text": str(sentence.get("text", "")), "label": label}
            existing = group["sentences"].get(ordinal)
            if existing is not None and existing != value:
                raise RunnerError(
                    "Overlapping cohort windows disagree on sentence content or gold label"
                )
            group["sentences"][ordinal] = value
    canonical_units = []
    for (document_id, target_id), group in sorted(groups.items()):
        canonical_units.append(
            {
                "document_id": document_id,
                "target_version_id": target_id,
                "target_text": group["target_text"],
                "sentences": [
                    {"ordinal": ordinal, **group["sentences"][ordinal]}
                    for ordinal in sorted(group["sentences"])
                ],
            }
        )
    return canonical_fingerprint(canonical_units)


def _rewindow_transformer_rows(rows: list[dict], tokenizer, config) -> list[dict]:
    """Rebuild windows with the plugin's exact pinned tokenizer.

    The frozen export carries complete sentence text and deterministic split
    assignments.  Its export-time token counts are only estimates; fitting and
    evaluation always reconstruct windows here so no sentence is truncated or
    silently dropped by a model-specific tokenizer.
    """

    from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
        encode_sentence_markers,
    )

    groups: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("document_id")), str(row["target"]["id"]))
        group = groups.setdefault(
            key,
            {
                "prototype": row,
                "sentences": {},
            },
        )
        for sentence in row.get("sentences") or []:
            ordinal = int(sentence["ordinal"])
            existing = group["sentences"].get(ordinal)
            if existing is not None and existing != sentence:
                raise RunnerError("Overlapping frozen windows disagree on sentence text or labels")
            group["sentences"][ordinal] = sentence

    prepared: list[dict] = []
    for (document_id, target_id), group in sorted(groups.items()):
        prototype = group["prototype"]
        sentences = [group["sentences"][key] for key in sorted(group["sentences"])]
        if not sentences:
            continue
        start = 0
        window_index = 0
        while start < len(sentences):
            best_end = start
            best_encoded = None
            for end in range(start + 1, len(sentences) + 1):
                try:
                    encoded = encode_sentence_markers(
                        tokenizer,
                        target_text=str(prototype["target"].get("text", "")),
                        sentences=[str(item.get("text", "")) for item in sentences[start:end]],
                        target_marker_token=config.target_marker_token,
                        sentence_marker_token=config.sentence_marker_token,
                        target_conditioning=config.target_conditioning,
                        max_length=config.max_length,
                    )
                except ValueError:
                    break
                best_end = end
                best_encoded = encoded
            if best_encoded is None:
                ordinal = sentences[start].get("ordinal", start)
                raise RunnerError(f"Sentence {ordinal} cannot fit the selected model context")
            selected = sentences[start:best_end]
            prepared.append(
                {
                    **prototype,
                    "window": {
                        "id": f"tokenized:{document_id}:{target_id}:{window_index}",
                        "start_sentence_ordinal": int(selected[0]["ordinal"]),
                        "end_sentence_ordinal": int(selected[-1]["ordinal"]),
                        "token_count": len(best_encoded.input_ids),
                        "tokenizer_exact": True,
                    },
                    "sentences": selected,
                }
            )
            window_index += 1
            if best_end >= len(sentences):
                break
            overlap_tokens = 0
            next_start = best_end
            while next_start > start + 1 and overlap_tokens < config.window_overlap_tokens:
                next_start -= 1
                overlap_tokens += 1 + len(
                    tokenizer.encode(
                        str(sentences[next_start].get("text", "")),
                        add_special_tokens=False,
                    )
                )
            start = max(start + 1, next_start)
    return prepared


def _validate_dataset_splits(
    dataset_path: Path,
    target_version_id: int | None,
    *,
    synthetic_mode: bool,
) -> None:
    """Reject document leakage before any feature fitting or model loading."""

    required_splits = {"train", "validation", "test"}
    documents_by_split: dict[str, set[str]] = {split: set() for split in required_splits}
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"Invalid dataset JSON on line {line_number}") from exc
            row_target_id = int(row["target"]["id"])
            if target_version_id is not None and row_target_id != target_version_id:
                continue
            split = str(row.get("split", ""))
            if split not in required_splits:
                if synthetic_mode:
                    continue
                raise RunnerError(
                    f"Production training row {line_number} has no valid split assignment"
                )
            document_id = row.get("document_id")
            if document_id is None:
                if not synthetic_mode:
                    raise RunnerError(
                        "Production training rows require immutable document identifiers"
                    )
                document_id = f"synthetic-row-{line_number}"
            documents_by_split[split].add(str(document_id))
    for left_index, left in enumerate(sorted(required_splits)):
        for right in sorted(required_splits)[left_index + 1 :]:
            overlap = documents_by_split[left] & documents_by_split[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:5])
                raise RunnerError(
                    "Document-level train/validation/test leakage detected between "
                    f"{left} and {right}: {sample}"
                )
    if not synthetic_mode:
        missing = sorted(split for split, documents in documents_by_split.items() if not documents)
        if missing:
            raise RunnerError(
                "Production training requires non-empty train, validation, and test "
                f"document splits; missing: {', '.join(missing)}"
            )


def _empty_prediction_group(
    groups: dict[tuple[str, str], dict],
    document_id: str,
    target_id: str,
) -> dict:
    return groups.setdefault(
        (document_id, target_id),
        {
            "gold": {},
            "sentences": {},
            "target_text": None,
            "logit_sums": {},
            "logit_counts": Counter(),
            "score_sums": Counter(),
            "score_counts": Counter(),
            "label_votes": {},
        },
    )


def _evaluate_checkpoint_splits(
    checkpoint_root: Path,
    *,
    dataset_path: Path,
    target_version_id: int | None,
    config: dict,
    synthetic_mode: bool,
    model_type: str,
    splits: tuple[str, ...] = ("validation", "test"),
    device_override: str | None = None,
) -> dict:
    """Evaluate canonical document/target predictions on validation and test.

    Overlapping window logits are averaged at sentence ordinals before BIO
    decoding, so duplicated windows never inflate support.
    """

    from al_medlit.training.evaluation.evidence import (
        EvidenceBlockEvaluator,
        EvidenceEvaluationContext,
        EvidenceEvaluationExample,
        blocks_from_bio,
        canonical_evaluation_content_fingerprint,
        canonical_fingerprint,
    )
    from al_medlit.training.model_types.evidence_conventional.model import (
        CONVENTIONAL_MODEL_TYPES,
        MissingConventionalDependencyError,
        load_conventional_model,
        predict_window,
    )
    from al_medlit.training.model_types.evidence_neural import (
        documents_from_window_rows,
        load_neural_bundle,
        predict_neural_model,
    )
    from al_medlit.training.model_types.evidence_neural.model import (
        MissingNeuralDependencyError,
    )

    predictor = None
    torch = None
    encode_sentence_markers = None
    conventional = model_type in CONVENTIONAL_MODEL_TYPES
    neural = model_type in NEURAL_MODEL_TYPES
    if conventional:
        try:
            predictor = load_conventional_model(checkpoint_root)
        except (MissingConventionalDependencyError, ValueError) as exc:
            raise RunnerError(str(exc)) from exc
    elif neural:
        try:
            predictor = load_neural_bundle(checkpoint_root)
        except (MissingNeuralDependencyError, ValueError) as exc:
            raise RunnerError(str(exc)) from exc
    elif not synthetic_mode:
        try:
            import torch as torch_module

            from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
                encode_sentence_markers as encoder,
            )
            from al_medlit.training.model_types.evidence_block_sentence_tagger.model import (
                load_sentence_tagger,
            )
        except ImportError as exc:  # pragma: no cover - optional ML environment
            raise RunnerError("Install the optional 'ml' dependencies for evaluation") from exc
        predictor = load_sentence_tagger(checkpoint_root)
        torch = torch_module
        encode_sentence_markers = encoder
        _logical_device, runtime_device = _resolve_torch_device(
            torch, device_override or predictor.config.device
        )
        predictor.model.to(torch.device(runtime_device))
        predictor.model.eval()

    evaluator = EvidenceBlockEvaluator()
    reports = {}
    training_cohort_fingerprint = _canonical_cohort_fingerprint(
        _selected_training_rows(dataset_path, target_version_id)
    )
    validation_rows = _evaluation_rows(dataset_path, target_version_id, "validation")
    validation_cohort_fingerprint = _canonical_cohort_fingerprint(validation_rows)
    for split in splits:
        rows = (
            validation_rows
            if split == "validation"
            else _evaluation_rows(dataset_path, target_version_id, split)
        )
        if not rows:
            continue
        if not conventional and not neural and not synthetic_mode:
            rows = _rewindow_transformer_rows(rows, predictor.tokenizer, predictor.config)
        groups: dict[tuple[str, str], dict] = {}
        if neural:
            documents = documents_from_window_rows(rows)
            predictions = predict_neural_model(
                predictor,
                documents,
                device=device_override or str(config.get("device", "auto")),
            )
            prediction_by_scope = {
                (item.document_id, str(item.target_id)): item for item in predictions
            }
            for document in documents:
                target_id = str(document.target_id)
                prediction = prediction_by_scope[(document.document_id, target_id)]
                group = _empty_prediction_group(groups, document.document_id, target_id)
                group["target_text"] = document.target_text
                ordinals = document.sentence_ordinals or tuple(range(len(document.sentences)))
                for ordinal, sentence_text, label, predicted_label in zip(
                    ordinals,
                    document.sentences,
                    document.labels or (),
                    prediction.labels,
                    strict=True,
                ):
                    if label == "IGNORE":
                        continue
                    group["gold"][ordinal] = label
                    group["sentences"][ordinal] = sentence_text
                    votes = group["label_votes"].setdefault(ordinal, Counter())
                    votes[predicted_label] += 1
        else:
            for row in rows:
                document_id = str(row.get("document_id", f"row-{row['_line_number']}"))
                target_id = str(row["target"]["id"])
                group = _empty_prediction_group(groups, document_id, target_id)
                target_text = str(row["target"].get("text", ""))
                if group["target_text"] not in (None, target_text):
                    raise RunnerError("Overlapping evaluation windows disagree on target text")
                group["target_text"] = target_text
                sentences = row.get("sentences") or []
                if conventional:
                    row_predictions = predict_window(
                        predictor,
                        target_text=str(row["target"].get("text", "")),
                        sentences=[str(sentence.get("text", "")) for sentence in sentences],
                    )
                    row_logits = None
                elif synthetic_mode:
                    row_logits = [
                        _synthetic_inference_logits(int(sentence.get("ordinal", index)))
                        for index, sentence in enumerate(sentences)
                    ]
                    row_predictions = None
                else:
                    encoded = encode_sentence_markers(
                        predictor.tokenizer,
                        target_text=str(row["target"].get("text", "")),
                        sentences=[str(sentence.get("text", "")) for sentence in sentences],
                        target_marker_token=predictor.config.target_marker_token,
                        sentence_marker_token=predictor.config.sentence_marker_token,
                        target_conditioning=predictor.config.target_conditioning,
                        max_length=predictor.config.max_length,
                    )
                    device = next(predictor.model.parameters()).device
                    with torch.no_grad():
                        output = predictor.model(
                            input_ids=torch.tensor(
                                [encoded.input_ids], dtype=torch.long, device=device
                            ),
                            attention_mask=torch.tensor(
                                [encoded.attention_mask], dtype=torch.long, device=device
                            ),
                            sentence_marker_positions=torch.tensor(
                                [encoded.sentence_marker_positions],
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                    row_logits = output["logits"][0].detach().cpu().tolist()
                    row_predictions = None
                for index, sentence in enumerate(sentences):
                    label = str(sentence.get("label", "IGNORE"))
                    if label == "IGNORE" or sentence.get("reviewed") is False:
                        continue
                    ordinal = int(sentence.get("ordinal", index))
                    existing_label = group["gold"].get(ordinal)
                    if existing_label is not None and existing_label != label:
                        raise RunnerError(
                            "Overlapping evaluation windows contain inconsistent gold labels"
                        )
                    group["gold"][ordinal] = label
                    sentence_text = str(sentence.get("text", ""))
                    existing_text = group["sentences"].get(ordinal)
                    if existing_text is not None and existing_text != sentence_text:
                        raise RunnerError(
                            "Overlapping evaluation windows contain inconsistent sentence text"
                        )
                    group["sentences"][ordinal] = sentence_text
                    if conventional and predictor.produces_sentence_scores:
                        group["score_sums"][ordinal] += float(row_predictions[index])
                        group["score_counts"][ordinal] += 1
                    elif conventional:
                        votes = group["label_votes"].setdefault(ordinal, Counter())
                        votes[str(row_predictions[index])] += 1
                    else:
                        totals = group["logit_sums"].setdefault(ordinal, [0.0, 0.0, 0.0])
                        for label_index, value in enumerate(row_logits[index]):
                            totals[label_index] += float(value)
                        group["logit_counts"][ordinal] += 1

        examples = []
        for (document_id, target_id), group in sorted(groups.items()):
            ordinals = sorted(group["gold"])
            if not ordinals:
                continue
            gold_labels = tuple(group["gold"][ordinal] for ordinal in ordinals)
            example_data = {
                "document_id": document_id,
                "target_version_id": target_id,
                "sentence_count": len(ordinals),
                "sentence_ordinals": tuple(ordinals),
                "sentence_texts": tuple(group["sentences"][ordinal] for ordinal in ordinals),
                "target_text": group["target_text"] or "",
                "reference_blocks": blocks_from_bio(gold_labels),
            }
            if conventional and predictor.produces_sentence_scores:
                example_data["predicted_scores"] = tuple(
                    group["score_sums"][ordinal] / group["score_counts"][ordinal]
                    for ordinal in ordinals
                )
            elif conventional or neural:
                label_order = {"O": 0, "B": 1, "I": 2}
                example_data["predicted_labels"] = tuple(
                    max(
                        group["label_votes"][ordinal],
                        key=lambda value: (
                            group["label_votes"][ordinal][value],
                            -label_order[value],
                        ),
                    )
                    for ordinal in ordinals
                )
            else:
                predicted_labels = []
                for ordinal in ordinals:
                    count = group["logit_counts"][ordinal]
                    averaged = [value / count for value in group["logit_sums"][ordinal]]
                    predicted_labels.append(
                        ("O", "B", "I")[max(range(3), key=averaged.__getitem__)]
                    )
                example_data["predicted_labels"] = tuple(predicted_labels)
            examples.append(EvidenceEvaluationExample(**example_data))
        if not examples:
            continue
        # Use the same evaluator-owned held-out fingerprint across every model
        # family.  Training/validation cohort hashing remains runner-owned,
        # while evaluation compatibility is deliberately family-neutral.
        dataset_fingerprint = canonical_evaluation_content_fingerprint(examples)
        decoder_config = (
            {
                "kind": "sentence_score_threshold",
                "threshold": float(config.get("sentence_score_threshold", 0.5)),
                "overlap": "mean_scores",
            }
            if conventional and predictor.produces_sentence_scores
            else {
                "kind": ("bio_majority_vote" if conventional or neural else "argmax_bio"),
                "overlap": ("majority_vote" if conventional or neural else "mean_logits"),
            }
        )
        if conventional:
            preprocessing_kind = (
                "sentence-crf-features-v1"
                if model_type == "evidence_crf"
                else "tfidf-sentence-features-v1"
            )
            preprocessing_keys = (
                (
                    "token_pattern",
                    "lowercase",
                    "target_conditioning",
                )
                if model_type == "evidence_crf"
                else (
                    "lowercase",
                    "strip_accents",
                    "min_df",
                    "max_df",
                    "max_features",
                    "word_ngram_range",
                    "sublinear_tf",
                    "target_conditioning",
                )
            )
        elif neural:
            preprocessing_kind = "neural-vocabulary-v1"
            preprocessing_keys = (
                "tokenizer_pattern",
                "lowercase",
                "unicode_normalization",
                "min_token_frequency",
                "max_vocabulary_size",
                "max_tokens_per_sentence",
                "max_sentences_per_document",
                "target_conditioning",
            )
        else:
            preprocessing_kind = "sentence-marker-tokenization-v1"
            preprocessing_keys = (
                "max_length",
                "target_conditioning",
                "target_marker_token",
                "sentence_marker_token",
            )
        preprocessing_version = "evidence-prep-v1:" + canonical_fingerprint(
            {
                "kind": preprocessing_kind,
                **{key: config.get(key) for key in preprocessing_keys},
            }
        )
        context = EvidenceEvaluationContext(
            evaluation_dataset_hash=dataset_fingerprint,
            target_version_ids=tuple(sorted({example.target_version_id for example in examples})),
            training_dataset_hash=training_cohort_fingerprint,
            validation_dataset_hash=validation_cohort_fingerprint,
            preprocessing_version=preprocessing_version,
            decoder_config=decoder_config,
        )
        report = evaluator.evaluate(
            examples,
            context=context,
            split=split,
            sentence_score_threshold=float(config.get("sentence_score_threshold", 0.5)),
            bootstrap_samples=int(config.get("bootstrap_samples", 200)),
            bootstrap_seed=int(config.get("seed", config.get("synthetic_seed", 42))),
        )
        reports[split] = {
            **report.to_record_payload(),
            "report": report.model_dump(mode="json"),
        }
    return reports


def _write_synthetic_checkpoint(
    checkpoint_root: Path,
    *,
    config: dict,
) -> dict:
    tokenizer_root = checkpoint_root / "tokenizer"
    tokenizer_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        tokenizer_root / "tokenizer.json",
        {
            "schema_version": "synthetic-tokenizer-v1",
            "special_tokens": [
                config.get("target_marker_token", "[TARGET]"),
                config.get("sentence_marker_token", "[SENT]"),
            ],
        },
    )
    _write_json(
        checkpoint_root / "sentence-tagger.json",
        {
            "schema_version": "synthetic-sentence-tagger-v1",
            "labels": config.get("labels", ["O", "B", "I"]),
            "seed": int(config.get("synthetic_seed", 42)),
        },
    )
    return {"synthetic_mode": True, "train_loss": 0.0}


def _train_real_checkpoint(
    checkpoint_root: Path,
    *,
    dataset_path: Path,
    target_version_id: int | None,
    raw_config: dict,
    staged_model_archive: Path | None = None,
) -> dict:
    try:
        import torch

        from al_medlit.training.model_types.evidence_block_sentence_tagger.config import (
            EvidenceBlockSentenceTaggerConfig,
        )
        from al_medlit.training.model_types.evidence_block_sentence_tagger.dataset import (
            balanced_class_weights,
            encode_labels,
            encode_sentence_markers,
        )
        from al_medlit.training.model_types.evidence_block_sentence_tagger.model import (
            MissingMLDependencyError,
            build_sentence_tagger,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional worker image
        raise RunnerError("Install the optional 'ml' dependencies for real training") from exc

    resolved_config = dict(raw_config)
    if staged_model_archive is not None:
        model_root = _resolve_hf_model_root(
            staged_model_archive,
            label="staged base model",
        )
        resolved_config.pop("staged_base_model_artifact_id", None)
        resolved_config.pop("base_model_asset_id", None)
        resolved_config["local_model_path"] = str(model_root)
    config = EvidenceBlockSentenceTaggerConfig.model_validate(resolved_config)
    try:
        bundle = build_sentence_tagger(config)
    except MissingMLDependencyError as exc:  # pragma: no cover - optional worker image
        raise RunnerError(str(exc)) from exc
    rows = _rewindow_transformer_rows(
        _selected_training_rows(dataset_path, target_version_id),
        bundle.tokenizer,
        config,
    )
    encoded_rows = []
    label_sequences = []
    for row in rows:
        labels = encode_labels([sentence["label"] for sentence in row["sentences"]])
        encoded = encode_sentence_markers(
            bundle.tokenizer,
            target_text=row["target"]["text"],
            sentences=[sentence["text"] for sentence in row["sentences"]],
            target_marker_token=config.target_marker_token,
            sentence_marker_token=config.sentence_marker_token,
            target_conditioning=config.target_conditioning,
            max_length=config.max_length,
        )
        encoded_rows.append((encoded, labels))
        label_sequences.append(labels)

    class_weights = None
    if config.class_weighting == "balanced":
        try:
            class_weights = balanced_class_weights(label_sequences)
        except ValueError as exc:
            raise RunnerError(str(exc)) from exc
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    device_name, runtime_device = _resolve_torch_device(torch, config.device)
    device = torch.device(runtime_device)
    model = bundle.model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    batches_per_epoch = math.ceil(len(encoded_rows) / config.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / config.gradient_accumulation_steps)
    total_optimizer_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)

    def lr_multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-12, (step + 1) / warmup_steps)
        if config.scheduler == "none":
            return 1.0
        denominator = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        if config.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0 - progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    pad_token_id = bundle.tokenizer.pad_token_id if bundle.tokenizer.pad_token_id is not None else 0

    def collate(batch):
        max_tokens = max(len(encoded.input_ids) for encoded, _labels in batch)
        max_sentences = max(len(labels) for _encoded, labels in batch)
        input_ids = []
        attention_masks = []
        marker_positions = []
        label_rows = []
        for encoded, labels in batch:
            token_padding = max_tokens - len(encoded.input_ids)
            sentence_padding = max_sentences - len(labels)
            input_ids.append(list(encoded.input_ids) + [pad_token_id] * token_padding)
            attention_masks.append(list(encoded.attention_mask) + [0] * token_padding)
            marker_positions.append(
                list(encoded.sentence_marker_positions) + [-1] * sentence_padding
            )
            label_rows.append(list(labels) + [-100] * sentence_padding)
        return (
            torch.tensor(input_ids, dtype=torch.long, device=device),
            torch.tensor(attention_masks, dtype=torch.long, device=device),
            torch.tensor(marker_positions, dtype=torch.long, device=device),
            torch.tensor(label_rows, dtype=torch.long, device=device),
        )

    losses: list[float] = []
    history: list[dict] = []
    progress_path = checkpoint_root.parent / "metric-points.jsonl"
    progress_path.unlink(missing_ok=True)
    best_primary = float("-inf")
    best_exact = float("-inf")
    best_epoch = None
    best_step = None
    epochs_without_improvement = 0
    optimizer_steps = 0
    best_root = checkpoint_root.parent / ".best-checkpoint"
    if best_root.exists():
        shutil.rmtree(best_root)
    for epoch_index in range(config.epochs):
        epoch_started = time.monotonic()
        order = list(range(len(encoded_rows)))
        random.Random(config.seed + epoch_index).shuffle(order)
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for batch_index in range(0, len(order), config.batch_size):
            batch = [
                encoded_rows[index]
                for index in order[batch_index : batch_index + config.batch_size]
            ]
            input_ids, attention_mask, sentence_positions, labels = collate(batch)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentence_marker_positions=sentence_positions,
                labels=labels,
                class_weights=class_weights,
            )
            loss = output["loss"]
            if loss is None or not torch.isfinite(loss):
                raise RunnerError("Training produced a non-finite loss")
            (loss / config.gradient_accumulation_steps).backward()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            epoch_losses.append(loss_value)
            batch_number = batch_index // config.batch_size + 1
            if (
                batch_number % config.gradient_accumulation_steps == 0
                or batch_index + config.batch_size >= len(order)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        model.to("cpu")
        bundle.save_pretrained(checkpoint_root)
        validation = _evaluate_checkpoint_splits(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            config=config.model_dump(mode="json"),
            synthetic_mode=False,
            model_type="evidence_block_sentence_tagger",
            splits=("validation",),
            device_override="cpu",
        ).get("validation")
        if validation is None:
            raise RunnerError("Validation split produced no comparable Evidence metrics")
        primary_value = validation["metrics"].get("macro_block_iou_f1_0_50")
        exact_value = validation["metrics"].get("macro_exact_block_f1")
        primary = float(primary_value) if primary_value is not None else float("-inf")
        exact = float(exact_value) if exact_value is not None else float("-inf")
        canonical_improved = (
            best_epoch is None
            or primary > best_primary
            or (
                math.isclose(primary, best_primary, rel_tol=0.0, abs_tol=1e-12)
                and exact > best_exact
            )
        )
        patience_improved = (
            best_epoch is None
            or primary > best_primary + config.early_stopping_min_delta
            or (
                math.isclose(primary, best_primary, rel_tol=0.0, abs_tol=1e-12)
                and exact > best_exact
            )
        )
        if canonical_improved:
            best_primary = primary
            best_exact = exact
            best_epoch = epoch_index + 1
            best_step = optimizer_steps
            if best_root.exists():
                shutil.rmtree(best_root)
            shutil.copytree(checkpoint_root, best_root)
        if patience_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed = max(time.monotonic() - epoch_started, 1e-9)
        metric_point = {
            "phase": "train",
            "split": "train",
            "epoch": epoch_index + 1,
            "step": optimizer_steps,
            "values": {
                "loss": sum(epoch_losses) / len(epoch_losses),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "windows_per_second": len(order) / elapsed,
            },
        }
        history.append(metric_point)
        _append_json_line(progress_path, metric_point)
        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            break
        model.to(device)
        model.train()

    if best_epoch is None or not best_root.is_dir():
        raise RunnerError("Training did not produce a validation-selected checkpoint")
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    shutil.move(str(best_root), checkpoint_root)
    return {
        "synthetic_mode": False,
        "train_loss": sum(losses) / len(losses),
        "epochs": len(history),
        "optimizer_steps": optimizer_steps,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "selection_metric_key": "macro_block_iou_f1_0_50",
        "selection_metric_value": (None if best_primary == float("-inf") else best_primary),
        "selection_tiebreaker_exact_block_f1": (
            None if best_exact == float("-inf") else best_exact
        ),
        "device": device_name,
        "history": history,
        "resolved_config": config.model_dump(mode="json"),
        "runtime": {
            "torch_version": str(torch.__version__),
            "deterministic": config.deterministic,
            "scheduler": config.scheduler,
        },
    }


def _train_conventional_checkpoint(
    checkpoint_root: Path,
    *,
    dataset_path: Path,
    target_version_id: int | None,
    model_type: str,
    raw_config: dict,
) -> dict:
    from pydantic import ValidationError as PydanticValidationError

    from al_medlit.training.model_types.evidence_conventional.model import (
        MissingConventionalDependencyError,
    )
    from al_medlit.training.model_types.evidence_conventional.plugin import (
        EvidenceCRFPlugin,
        EvidenceRandomForestPlugin,
        EvidenceSVMPlugin,
    )

    plugins = {
        "evidence_crf": EvidenceCRFPlugin(),
        "evidence_svm": EvidenceSVMPlugin(),
        "evidence_random_forest": EvidenceRandomForestPlugin(),
    }
    try:
        plugin = plugins[model_type]
        rows = _selected_training_rows(dataset_path, target_version_id)
        validated_config = plugin.validate_config(raw_config)
        metrics = plugin.fit(
            rows,
            validated_config.model_dump(mode="json"),
            checkpoint_root,
        )
        return {
            **metrics,
            "resolved_config": validated_config.model_dump(mode="json"),
        }
    except KeyError as exc:
        raise RunnerError(f"Unsupported conventional model type {model_type!r}") from exc
    except (MissingConventionalDependencyError, PydanticValidationError, ValueError) as exc:
        raise RunnerError(str(exc)) from exc


def _train_neural_checkpoint(
    checkpoint_root: Path,
    *,
    dataset_path: Path,
    target_version_id: int | None,
    model_type: str,
    raw_config: dict,
) -> dict:
    """Fit a compact neural plugin with canonical validation-only selection."""

    from pydantic import ValidationError as PydanticValidationError

    from al_medlit.training.model_types.evidence_neural import (
        EvidenceBiLSTMPlugin,
        EvidenceCNNPlugin,
        NeuralCheckpointScore,
        documents_from_window_rows,
    )
    from al_medlit.training.model_types.evidence_neural.model import (
        MissingNeuralDependencyError,
    )

    plugins = {
        "evidence_bilstm": EvidenceBiLSTMPlugin(),
        "evidence_cnn": EvidenceCNNPlugin(),
    }
    try:
        plugin = plugins[model_type]
        config = plugin.validate_config(raw_config)
        training_documents = documents_from_window_rows(
            _selected_training_rows(dataset_path, target_version_id)
        )
        validation_documents = documents_from_window_rows(
            _evaluation_rows(dataset_path, target_version_id, "validation")
        )

        def validation_selector(bundle, _epoch: int) -> NeuralCheckpointScore:
            plugin.package(bundle, checkpoint_root)
            report = _evaluate_checkpoint_splits(
                checkpoint_root,
                dataset_path=dataset_path,
                target_version_id=target_version_id,
                config=config.model_dump(mode="json"),
                synthetic_mode=False,
                model_type=model_type,
                splits=("validation",),
                device_override="cpu",
            ).get("validation")
            if report is None:
                raise RunnerError("Validation split produced no comparable Evidence metrics")
            return NeuralCheckpointScore(
                macro_block_iou_f1_0_50=float(report["metrics"]["macro_block_iou_f1_0_50"]),
                macro_exact_block_f1=float(report["metrics"]["macro_exact_block_f1"]),
            )

        bundle, summary = plugin.fit(
            training_documents,
            config=config.model_dump(mode="json"),
            validation_documents=validation_documents,
            validation_selector=validation_selector,
        )
        package_manifest = plugin.package(bundle, checkpoint_root)
        history = [
            {
                "phase": "train",
                "split": "train",
                "epoch": item.epoch,
                "step": item.epoch,
                "values": {
                    key: value
                    for key, value in {
                        "loss": item.train_loss,
                        "validation_loss": item.validation_loss,
                        "learning_rate": item.learning_rate,
                        "validation_macro_block_iou_f1_0_50": (
                            item.validation_macro_block_iou_f1_0_50
                        ),
                        "validation_macro_exact_block_f1": (item.validation_macro_exact_block_f1),
                    }.items()
                    if value is not None
                },
            }
            for item in summary.history
        ]
        progress_path = checkpoint_root.parent / "metric-points.jsonl"
        progress_path.unlink(missing_ok=True)
        for point in history:
            _append_json_line(progress_path, point)
        return {
            "synthetic_mode": False,
            "model_family": "deep_learning",
            "package_format": "safetensors",
            "train_loss": summary.history[-1].train_loss,
            "epochs": len(summary.history),
            "best_epoch": summary.best_epoch,
            "best_step": summary.best_epoch,
            "selection_metric_key": summary.selection_metric,
            "selection_metric_value": summary.selection_value,
            "selection_tiebreaker_exact_block_f1": (summary.selection_tiebreaker_exact_block_f1),
            "device": summary.device,
            "history": history,
            "resolved_config": config.model_dump(mode="json"),
            "runtime": {
                "safe_serialization": bool(package_manifest["safe_serialization"]),
                "deterministic": config.deterministic_algorithms,
                "scheduler": config.scheduler,
            },
        }
    except KeyError as exc:
        raise RunnerError(f"Unsupported neural model type {model_type!r}") from exc
    except (
        MissingNeuralDependencyError,
        PydanticValidationError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, RunnerError):
            raise
        raise RunnerError(str(exc)) from exc


def _train_peft_checkpoint(
    checkpoint_root: Path,
    *,
    dataset_path: Path,
    target_version_id: int | None,
    model_type: str,
    raw_config: dict,
    staged_model_archive: Path,
    staged_model_metadata: dict,
) -> dict:
    """Fine-tune an adapter against an exact, locally materialized base package."""

    from pydantic import ValidationError as PydanticValidationError

    from al_medlit.training.model_types.evidence_peft import (
        EvidenceLoRAPlugin,
        EvidenceQLoRAPlugin,
        ImmutableBaseModelReference,
    )
    from al_medlit.training.model_types.evidence_peft.training import train_peft_adapter

    plugins = {
        "evidence_lora": EvidenceLoRAPlugin(),
        "evidence_qlora": EvidenceQLoRAPlugin(),
    }
    try:
        plugin = plugins[model_type]
        config = plugin.validate_config(raw_config)
        base_reference = ImmutableBaseModelReference(
            asset_id=int(staged_model_metadata["base_model_asset_id"]),
            package_id=int(staged_model_metadata["base_model_package_id"]),
            manifest_digest=str(staged_model_metadata["base_model_manifest_sha256"]),
            exact_revision=str(staged_model_metadata["base_model_exact_revision"]),
        )
        base_model_root = _resolve_hf_model_root(
            staged_model_archive,
            label="PEFT base",
        )
        result = train_peft_adapter(
            config,
            base_model_root=base_model_root,
            base_reference=base_reference,
            training_rows=_selected_training_rows(dataset_path, target_version_id),
            validation_rows=_evaluation_rows(dataset_path, target_version_id, "validation"),
            test_rows=_evaluation_rows(dataset_path, target_version_id, "test"),
            training_dataset_hash=_canonical_cohort_fingerprint(
                _selected_training_rows(dataset_path, target_version_id)
            ),
            validation_dataset_hash=_canonical_cohort_fingerprint(
                _evaluation_rows(dataset_path, target_version_id, "validation")
            ),
        )
        package_manifest = plugin.package(result.bundle, checkpoint_root)
        progress_path = checkpoint_root.parent / "metric-points.jsonl"
        progress_path.unlink(missing_ok=True)
        for point in result.history:
            _append_json_line(progress_path, point)
        final_loss = result.history[-1]["values"]["token_loss"]
        return {
            "synthetic_mode": False,
            "model_family": "llm_finetune",
            "package_format": "peft_adapter_safetensors",
            "train_loss": final_loss,
            "epochs": len(result.history),
            "best_epoch": result.best_epoch,
            "best_step": result.history[result.best_epoch - 1]["step"],
            "selection_metric_key": "macro_block_iou_f1_0_50",
            "selection_metric_value": result.selection_value,
            "selection_tiebreaker_exact_block_f1": result.selection_tiebreaker,
            "device": result.device,
            "history": list(result.history),
            "resolved_config": config.model_dump(mode="json"),
            "_evaluations": result.evaluations,
            "runtime": {
                "safe_serialization": True,
                "adapter_parameter_count": result.adapter_parameter_count,
                "trainable_parameter_count": result.trainable_parameter_count,
                "base_model_asset_id": base_reference.asset_id,
                "base_model_package_id": base_reference.package_id,
                "base_model_manifest_sha256": base_reference.manifest_digest,
                "base_model_exact_revision": base_reference.exact_revision,
                "adapter_manifest": package_manifest.model_dump(mode="json"),
            },
        }
    except KeyError as exc:
        raise RunnerError(
            "PEFT training requires complete immutable base-model package metadata"
        ) from exc
    except (
        PydanticValidationError,
        RuntimeError,
        ValueError,
    ) as exc:
        if isinstance(exc, RunnerError):
            raise
        raise RunnerError(str(exc)) from exc


def _extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    limits: ArchiveExtractionLimits | None = None,
) -> Path:
    try:
        return extract_zip_bounded(archive_path, destination, limits=limits)
    except ArchiveExtractionError as exc:
        raise RunnerError(f"Unsafe checkpoint archive: {exc}") from exc


def _resolve_hf_model_root(source: Path, *, label: str) -> Path:
    """Locate an HF model in either a verified package directory or legacy ZIP."""

    if source.is_dir():
        unpacked = source
    elif source.is_file():
        unpacked = _extract_zip(source, source.parent / f"{label}-extracted")
    else:
        raise RunnerError(f"The immutable {label} model source is missing")
    candidates = [unpacked / "base-model", unpacked / "model", unpacked]
    candidates.extend(path for path in unpacked.iterdir() if path.is_dir())
    model_root = next(
        (candidate for candidate in candidates if (candidate / "config.json").is_file()),
        None,
    )
    if model_root is None:
        raise RunnerError(
            f"The immutable {label} package must contain a Hugging Face config.json"
        )
    return model_root


def _resolve_materialized_package(
    bundle_root: Path,
    package: dict,
    *,
    label: str,
) -> Path:
    """Verify a manifest-declared package directory without rebuilding a ZIP."""

    package_root = _contained_path(bundle_root, str(package.get("path", "")))
    files = package.get("files")
    if not package_root.is_dir() or not isinstance(files, list) or not files:
        raise RunnerError(f"The immutable {label} package is incomplete")
    for item in files:
        if not isinstance(item, dict):
            raise RunnerError(f"The immutable {label} package manifest is invalid")
        relative = str(item.get("relative_path", ""))
        declared_path = _contained_path(bundle_root, str(item.get("path", "")))
        expected_path = _contained_path(package_root, relative)
        if declared_path != expected_path or not declared_path.is_file():
            raise RunnerError(f"The immutable {label} package path is invalid")
        try:
            expected_size = int(item["size_bytes"])
            expected_checksum = str(item["checksum_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError(f"The immutable {label} package manifest is invalid") from exc
        if (
            declared_path.stat().st_size != expected_size
            or sha256_file(declared_path) != expected_checksum
        ):
            raise RunnerError(f"The immutable {label} package checksum does not match")
    return package_root


def _synthetic_inference_logits(ordinal: int) -> tuple[float, float, float]:
    if ordinal == 0:
        return (0.0, 6.0, 0.0)
    if ordinal == 1:
        return (0.0, 0.0, 6.0)
    return (6.0, 0.0, 0.0)


def run_inference_job(manifest_path: str | Path) -> dict:
    manifest_file = Path(manifest_path).resolve()
    bundle_root = manifest_file.parent
    try:
        job = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("Unable to read the job manifest") from exc
    if job.get("schema_version") != RUNNER_SCHEMA_VERSION or job.get("kind") != "inference":
        raise RunnerError("Unsupported inference job manifest")
    checkpoint = job.get("checkpoint") or {}
    checkpoint_path = None
    checkpoint_root = None
    checkpoint_package = checkpoint.get("package")
    if isinstance(checkpoint_package, dict):
        checkpoint_root = _resolve_materialized_package(
            bundle_root,
            checkpoint_package,
            label="checkpoint",
        )
    else:
        checkpoint_path = _contained_path(bundle_root, str(checkpoint.get("path", "")))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint.get(
            "checksum_sha256"
        ):
            raise RunnerError("The immutable checkpoint checksum does not match")
    corpus = job.get("corpus") or {}
    corpus_path = _contained_path(bundle_root, str(corpus.get("path", "")))
    if sha256_file(corpus_path) != corpus.get("checksum_sha256"):
        raise RunnerError("The frozen inference input checksum does not match")
    corpus_input = json.loads(corpus_path.read_text(encoding="utf-8"))
    synthetic_mode = bool(checkpoint.get("manifest", {}).get("synthetic_mode"))
    model_type = str(
        checkpoint.get("manifest", {}).get("model_type", "evidence_block_sentence_tagger")
    )

    from al_medlit.training.model_types.evidence_conventional.model import (
        CONVENTIONAL_MODEL_TYPES,
        MissingConventionalDependencyError,
        actual_token_count,
        load_conventional_model,
    )

    conventional = model_type in CONVENTIONAL_MODEL_TYPES
    neural = model_type in NEURAL_MODEL_TYPES

    predictor = None
    peft_tokenizer = None
    torch = None
    if synthetic_mode:

        def token_counter(text: str) -> int:
            return max(1, len(text.split()))
    elif conventional:
        checkpoint_root = checkpoint_root or (
            _extract_zip(checkpoint_path, bundle_root / "checkpoint-extracted") / "checkpoint"
        )
        try:
            predictor = load_conventional_model(checkpoint_root)
        except (MissingConventionalDependencyError, ValueError) as exc:
            raise RunnerError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return actual_token_count(predictor, text)
    elif neural:
        from al_medlit.training.model_types.evidence_neural import load_neural_bundle
        from al_medlit.training.model_types.evidence_neural.model import (
            MissingNeuralDependencyError,
        )

        checkpoint_root = checkpoint_root or (
            _extract_zip(checkpoint_path, bundle_root / "checkpoint-extracted") / "checkpoint"
        )
        try:
            predictor = load_neural_bundle(checkpoint_root)
        except (MissingNeuralDependencyError, ValueError) as exc:
            raise RunnerError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return max(1, len(predictor.vocabulary.tokenize(text)))
    elif model_type in PEFT_MODEL_TYPES:
        from al_medlit.training.model_types.evidence_peft import (
            ImmutableBaseModelReference,
        )
        from al_medlit.training.model_types.evidence_peft.training import (
            load_peft_runtime,
        )

        base_metadata = job.get("base_model") or {}
        base_package = base_metadata.get("package")
        if isinstance(base_package, dict):
            base_root = _resolve_materialized_package(
                bundle_root,
                base_package,
                label="PEFT base-model",
            )
        else:
            base_archive = _contained_path(
                bundle_root,
                str(base_metadata.get("path", "")),
            )
            if not base_archive.is_file() or sha256_file(base_archive) != base_metadata.get(
                "checksum_sha256"
            ):
                raise RunnerError("The immutable PEFT base-model checksum does not match")
            base_root = _extract_zip(base_archive, bundle_root / "base-model-extracted")
        checkpoint_root = checkpoint_root or (
            _extract_zip(checkpoint_path, bundle_root / "checkpoint-extracted") / "checkpoint"
        )
        try:
            reference = ImmutableBaseModelReference(
                asset_id=int(base_metadata["base_model_asset_id"]),
                package_id=int(base_metadata["base_model_package_id"]),
                manifest_digest=str(base_metadata["base_model_manifest_sha256"]),
                exact_revision=str(base_metadata["base_model_exact_revision"]),
            )
            predictor, peft_tokenizer = load_peft_runtime(
                checkpoint_root,
                base_model_root=base_root,
                base_reference=reference,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RunnerError(str(exc)) from exc

        def token_counter(text: str) -> int:
            return max(
                1,
                len(peft_tokenizer.encode(text, add_special_tokens=False)),
            )
    else:
        try:
            import torch as torch_module

            from al_medlit.training.model_types.evidence_block_sentence_tagger.model import (
                MissingMLDependencyError,
                load_sentence_tagger,
            )
        except ImportError as exc:  # pragma: no cover - optional worker image
            raise RunnerError("Install the optional 'ml' dependencies for inference") from exc
        checkpoint_root = checkpoint_root or (
            _extract_zip(checkpoint_path, bundle_root / "checkpoint-extracted") / "checkpoint"
        )
        try:
            predictor = load_sentence_tagger(checkpoint_root)
        except MissingMLDependencyError as exc:  # pragma: no cover
            raise RunnerError(str(exc)) from exc
        torch = torch_module
        _device_name, runtime_device = _resolve_torch_device(torch, predictor.config.device)
        predictor.model.to(torch.device(runtime_device))
        predictor.model.eval()

        def token_counter(text: str) -> int:
            return max(1, len(predictor.tokenizer.encode(text, add_special_tokens=False)))

    from al_medlit.training.windowing import (
        EvidenceBlockWindowBuilder,
        TargetCondition,
        WindowBuilderConfig,
        WindowSentenceInput,
    )

    window_config = job["window_config"]
    builder = EvidenceBlockWindowBuilder(
        token_counter,
        WindowBuilderConfig(
            max_tokens=int(window_config["max_tokens"]),
            overlap_tokens=int(window_config["overlap_tokens"]),
            target_conditioning=checkpoint["training_mode"] == "conditioned",
            require_reviewed_gold=False,
        ),
    )
    output_windows = []
    for document in corpus_input["documents"]:
        sentence_inputs = [
            WindowSentenceInput(
                id=sentence["id"],
                ordinal=sentence["ordinal"],
                paragraph_ordinal=sentence["paragraph_ordinal"],
                section_path=tuple(sentence["section_path"]),
                text=sentence["text"],
                start_char=sentence["start_char"],
                end_char=sentence["end_char"],
            )
            for sentence in document["sentences"]
        ]
        for target in corpus_input["targets"]:
            result = builder.build(
                document_id=document["document_id"],
                structure_version_id=document["structure_version_id"],
                target=TargetCondition(
                    id=target["id"],
                    key=target["key"],
                    name=target["name"],
                    text=target["text"],
                ),
                sentences=sentence_inputs,
            )
            for window in result.windows:
                if synthetic_mode:
                    logits = {
                        str(sentence.ordinal): list(_synthetic_inference_logits(sentence.ordinal))
                        for sentence in window.sentences
                    }
                elif conventional:
                    from al_medlit.training.model_types.evidence_conventional.model import (
                        predict_window,
                    )

                    predictions = predict_window(
                        predictor,
                        target_text=target["text"],
                        sentences=[sentence.text for sentence in window.sentences],
                    )
                    if predictor.produces_sentence_scores:
                        threshold = predictor.config.sentence_score_threshold
                        labels = []
                        inside = False
                        for score in predictions:
                            positive = float(score) >= threshold
                            labels.append(
                                "B" if positive and not inside else ("I" if positive else "O")
                            )
                            inside = positive
                    else:
                        labels = [str(value) for value in predictions]
                    label_logits = {
                        "O": [1.0, 0.0, 0.0],
                        "B": [0.0, 1.0, 0.0],
                        "I": [0.0, 0.0, 1.0],
                    }
                    logits = {
                        str(sentence.ordinal): label_logits[labels[index]]
                        for index, sentence in enumerate(window.sentences)
                    }
                elif neural:
                    from al_medlit.training.model_types.evidence_neural import (
                        EvidenceTextDocument,
                        predict_neural_model,
                    )

                    prediction = predict_neural_model(
                        predictor,
                        (
                            EvidenceTextDocument(
                                document_id=str(document["document_id"]),
                                target_id=target["id"],
                                target_text=target["text"],
                                sentences=tuple(sentence.text for sentence in window.sentences),
                                sentence_ordinals=tuple(
                                    sentence.ordinal for sentence in window.sentences
                                ),
                            ),
                        ),
                        device=predictor.config.device,
                    )[0]
                    logits = {
                        str(ordinal): list(prediction.probabilities[index])
                        for index, ordinal in enumerate(prediction.sentence_ordinals)
                    }
                elif model_type in PEFT_MODEL_TYPES:
                    from al_medlit.training.model_types.evidence_peft.training import (
                        predict_peft_examples,
                        prepare_peft_examples,
                    )

                    row = {
                        "document_id": str(document["document_id"]),
                        "target": {"id": target["id"], "text": target["text"]},
                        "sentences": [
                            {"ordinal": sentence.ordinal, "text": sentence.text}
                            for sentence in window.sentences
                        ],
                    }
                    examples = prepare_peft_examples(
                        (row,),
                        peft_tokenizer,
                        predictor.config,
                        include_unreviewed=True,
                        supervised=False,
                    )
                    predicted, _valid_count = predict_peft_examples(
                        predictor,
                        peft_tokenizer,
                        examples,
                        config=predictor.config,
                    )
                    one_hot = {
                        "O": [1.0, 0.0, 0.0],
                        "B": [0.0, 1.0, 0.0],
                        "I": [0.0, 0.0, 1.0],
                    }
                    labels_by_ordinal = {
                        ordinal: label
                        for example, labels in zip(examples, predicted, strict=True)
                        for ordinal, label in zip(
                            example.sentence_ordinals,
                            labels,
                            strict=True,
                        )
                    }
                    logits = {
                        str(sentence.ordinal): one_hot[labels_by_ordinal[sentence.ordinal]]
                        for sentence in window.sentences
                    }
                else:
                    from al_medlit.training.model_types.evidence_block_sentence_tagger import (
                        dataset as tagger_dataset,
                    )

                    encoded = tagger_dataset.encode_sentence_markers(
                        predictor.tokenizer,
                        target_text=target["text"],
                        sentences=[sentence.text for sentence in window.sentences],
                        target_marker_token=predictor.config.target_marker_token,
                        sentence_marker_token=predictor.config.sentence_marker_token,
                        target_conditioning=predictor.config.target_conditioning,
                        max_length=predictor.config.max_length,
                    )
                    device = next(predictor.model.parameters()).device
                    with torch.no_grad():
                        model_output = predictor.model(
                            input_ids=torch.tensor(
                                [encoded.input_ids], dtype=torch.long, device=device
                            ),
                            attention_mask=torch.tensor(
                                [encoded.attention_mask], dtype=torch.long, device=device
                            ),
                            sentence_marker_positions=torch.tensor(
                                [encoded.sentence_marker_positions],
                                dtype=torch.long,
                                device=device,
                            ),
                        )
                    values = model_output["logits"][0].detach().cpu().tolist()
                    logits = {
                        str(sentence.ordinal): values[index]
                        for index, sentence in enumerate(window.sentences)
                    }
                output_windows.append(
                    {
                        "stable_key": window.id,
                        "document_id": document["document_id"],
                        "structure_version_id": document["structure_version_id"],
                        "target_version_id": target["id"],
                        "start_sentence_ordinal": window.start_sentence_ordinal,
                        "end_sentence_ordinal": window.end_sentence_ordinal,
                        "token_count": window.token_count,
                        "logits": logits,
                    }
                )
    output_root = _contained_path(bundle_root, str(job.get("output_directory", "outputs")))
    output_root.mkdir(parents=True, exist_ok=True)
    inference_result = {
        "schema_version": "inference-window-logits-v1",
        "checkpoint_checksum_sha256": checkpoint["checksum_sha256"],
        "synthetic_mode": synthetic_mode,
        "windows": output_windows,
    }
    _write_json(output_root / "inference-result.json", inference_result)
    (output_root / "infer.log").write_text(
        "AL-MedLit inference completed successfully\n"
        f"job_key={job['job_key']}\n"
        f"checkpoint_sha256={checkpoint['checksum_sha256']}\n",
        encoding="utf-8",
    )
    files = {}
    for relative in ("inference-result.json", "infer.log"):
        path = output_root / relative
        files[relative] = {
            "checksum_sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    artifact_manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "inference",
        "job_key": job["job_key"],
        "status": "succeeded",
        "files": files,
    }
    _write_json(output_root / "artifact-manifest.json", artifact_manifest)
    return artifact_manifest


def run_training_job(manifest_path: str | Path) -> dict:
    manifest_file = Path(manifest_path).resolve()
    bundle_root = manifest_file.parent
    try:
        job = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError("Unable to read the job manifest") from exc
    if job.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise RunnerError("Unsupported job manifest schema")
    if job.get("kind") != "training":
        raise RunnerError("The supplied bundle is not a training job")

    dataset = job.get("dataset") or {}
    dataset_path = _contained_path(bundle_root, str(dataset.get("path", "")))
    if not dataset_path.is_file():
        raise RunnerError("The immutable training dataset is missing")
    actual_dataset_hash = sha256_file(dataset_path)
    if actual_dataset_hash != dataset.get("checksum_sha256"):
        raise RunnerError("The immutable training dataset checksum does not match")

    config = job.get("config") or {}
    target_version_id = job.get("target_version_id")
    model_type = str(
        job.get("model_type")
        or (job.get("checkpoint_manifest") or {}).get("model_type")
        or "evidence_block_sentence_tagger"
    )
    synthetic_mode = bool(config.get("synthetic_mode"))
    _validate_dataset_splits(
        dataset_path,
        target_version_id,
        synthetic_mode=synthetic_mode,
    )
    statistics = _training_statistics(dataset_path, target_version_id)
    output_root = _contained_path(bundle_root, str(job.get("output_directory", "outputs")))
    checkpoint_root = output_root / "checkpoint"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    checkpoint_manifest = {
        **(job.get("checkpoint_manifest") or {}),
        "schema_version": "model-checkpoint-v1",
        "dataset_checksum_sha256": actual_dataset_hash,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "statistics": statistics,
    }
    staged_model_archive = None
    staged_model = job.get("staged_base_model")
    if staged_model is not None:
        staged_package = staged_model.get("package")
        if isinstance(staged_package, dict):
            staged_model_archive = _resolve_materialized_package(
                bundle_root,
                staged_package,
                label="base model",
            )
        else:
            staged_model_archive = _contained_path(
                bundle_root,
                str(staged_model["path"]),
            )
            if not staged_model_archive.is_file():
                raise RunnerError("The staged base-model artifact is missing")
            if sha256_file(staged_model_archive) != staged_model["checksum_sha256"]:
                raise RunnerError("The staged base-model checksum does not match")
    from al_medlit.training.model_types.evidence_conventional.model import (
        CONVENTIONAL_MODEL_TYPES,
    )

    if model_type in CONVENTIONAL_MODEL_TYPES:
        if synthetic_mode:
            raise RunnerError("Conventional model plugins do not publish synthetic models")
        if staged_model_archive is not None:
            raise RunnerError("Conventional model plugins do not accept a base-model archive")
        training_metrics = _train_conventional_checkpoint(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            model_type=model_type,
            raw_config=config,
        )
    elif model_type in NEURAL_MODEL_TYPES:
        if synthetic_mode:
            raise RunnerError("BiLSTM/CNN plugins do not publish synthetic models")
        if staged_model_archive is not None:
            raise RunnerError("BiLSTM/CNN plugins do not accept a base-model archive")
        training_metrics = _train_neural_checkpoint(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            model_type=model_type,
            raw_config=config,
        )
    elif synthetic_mode:
        training_metrics = _write_synthetic_checkpoint(checkpoint_root, config=config)
    elif model_type == "evidence_block_sentence_tagger":
        training_metrics = _train_real_checkpoint(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            raw_config=config,
            staged_model_archive=staged_model_archive,
        )
    elif model_type in PEFT_MODEL_TYPES:
        if synthetic_mode:
            raise RunnerError("PEFT plugins do not publish synthetic adapters")
        if staged_model_archive is None or staged_model is None:
            raise RunnerError("PEFT training requires an immutable base-model package")
        training_metrics = _train_peft_checkpoint(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            model_type=model_type,
            raw_config=config,
            staged_model_archive=staged_model_archive,
            staged_model_metadata=staged_model,
        )
    else:
        raise RunnerError(f"Unsupported model type {model_type!r}")
    evaluations = training_metrics.pop("_evaluations", None)
    if evaluations is None:
        evaluations = _evaluate_checkpoint_splits(
            checkpoint_root,
            dataset_path=dataset_path,
            target_version_id=target_version_id,
            config=training_metrics.get("resolved_config", config),
            synthetic_mode=synthetic_mode,
            model_type=model_type,
        )
    validation_report = evaluations.get("validation")
    if validation_report is not None:
        training_metrics.update(
            {
                "selection_metric_key": "macro_block_iou_f1_0_50",
                "selection_metric_value": validation_report["metrics"].get(
                    "macro_block_iou_f1_0_50"
                ),
                "selection_tiebreaker_exact_block_f1": validation_report["metrics"].get(
                    "macro_exact_block_f1"
                ),
                "checkpoint_ordinal": training_metrics.get("best_epoch", 1),
                "best_step": training_metrics.get(
                    "best_step", training_metrics.get("optimizer_steps", 1)
                ),
            }
        )
    checkpoint_manifest.update(
        {
            "model_type": model_type,
            "model_family": training_metrics.get("model_family", "deep_learning"),
            "package_format": training_metrics.get(
                "package_format", "synthetic" if synthetic_mode else "hf_pretrained"
            ),
            "synthetic_mode": synthetic_mode,
            "synthetic_seed": int(config.get("synthetic_seed", 42)),
            "selection_metric_key": training_metrics.get("selection_metric_key"),
            "selection_metric_value": training_metrics.get("selection_metric_value"),
            "selection_tiebreaker_exact_block_f1": training_metrics.get(
                "selection_tiebreaker_exact_block_f1"
            ),
            "checkpoint_ordinal": training_metrics.get("checkpoint_ordinal"),
            "best_step": training_metrics.get("best_step"),
            "runtime": training_metrics.get("runtime") or {},
            # The checkpoint's discovery metadata is bound to the held-out test
            # cohort. Validation remains the only checkpoint-selection split.
            "evaluation_fingerprint": (
                evaluations.get("test", {}).get("evaluation_fingerprint")
            ),
        }
    )
    _write_json(checkpoint_root / "manifest.json", checkpoint_manifest)
    metrics = {
        "schema_version": "training-metrics-v1",
        **training_metrics,
        **statistics,
        "evaluations": evaluations,
    }
    _write_json(output_root / "metrics.json", metrics)
    (output_root / "train.log").write_text(
        "AL-MedLit training completed successfully\n"
        f"job_key={job['job_key']}\n"
        f"dataset_sha256={actual_dataset_hash}\n",
        encoding="utf-8",
    )

    relative_files = [
        path.relative_to(output_root).as_posix()
        for path in checkpoint_root.rglob("*")
        if path.is_file()
    ] + ["metrics.json", "train.log"]
    if (output_root / "metric-points.jsonl").is_file():
        relative_files.append("metric-points.jsonl")
    files = {}
    for relative in relative_files:
        path = output_root / relative
        files[relative] = {
            "checksum_sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": "training",
        "job_key": job["job_key"],
        "status": "succeeded",
        "checkpoint_manifest": checkpoint_manifest,
        "files": files,
    }
    _write_json(output_root / "artifact-manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an immutable AL-MedLit job bundle")
    parser.add_argument("command", choices=("train", "infer"))
    parser.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            run_training_job(args.manifest)
        else:
            run_inference_job(args.manifest)
    except RunnerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
