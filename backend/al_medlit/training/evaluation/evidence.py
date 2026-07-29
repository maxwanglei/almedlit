"""Canonical, model-family-neutral Evidence Block evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field, field_validator, model_validator

from al_medlit.iaa.evidence_metrics import maximum_weight_matching
from al_medlit.training.contracts import ContractModel, TaskContractDescriptor
from al_medlit.training.model_types.catalog import COMMON_EVIDENCE_METRICS, TASK_TYPE

IOU_THRESHOLDS = (0.25, 0.50, 0.75)
EVIDENCE_TASK_CONTRACT = TaskContractDescriptor(
    key="evidence_blocks",
    contract_version="1",
    label="Evidence Block detection",
    task_type=TASK_TYPE,
    prediction_schema_version="evidence-block-prediction-v1",
    evaluator_version="evidence-block-evaluator-v1",
    metric_suite_version="evidence-block-metrics-v1",
    selection_metric="macro.block_iou_f1.0.50",
    selection_tiebreakers=("macro.exact_block_f1", "checkpoint_ordinal_ascending"),
    metrics=COMMON_EVIDENCE_METRICS,
)


def canonical_fingerprint(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def configuration_fingerprint(config: dict) -> str:
    """Fingerprint replicate-equivalent config, excluding random seed fields."""

    excluded = {"seed", "random_seed", "synthetic_seed"}

    def normalized(value):
        if isinstance(value, dict):
            return {
                key: normalized(item)
                for key, item in sorted(value.items())
                if key not in excluded
            }
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value

    return canonical_fingerprint(normalized(config))


class EvidenceSpan(ContractModel):
    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Evidence span end must not precede its start")
        return self

    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end


class EvidenceEvaluationExample(ContractModel):
    document_id: str = Field(min_length=1, max_length=255)
    target_version_id: str = Field(min_length=1, max_length=255)
    sentence_count: int = Field(ge=0)
    # These fields are optional only so legacy evaluation reports can still be
    # loaded.  Every new runner supplies them.  Keeping the source text in the
    # transient evaluator input lets the fingerprint bind the exact held-out
    # content without persisting that text in the metric summary.
    sentence_ordinals: tuple[int, ...] | None = None
    sentence_texts: tuple[str, ...] | None = None
    target_text: str | None = None
    reference_blocks: tuple[EvidenceSpan, ...] = ()
    predicted_blocks: tuple[EvidenceSpan, ...] | None = None
    predicted_labels: tuple[Literal["O", "B", "I"], ...] | None = None
    predicted_scores: tuple[float, ...] | None = None

    @field_validator("document_id", "target_version_id", mode="before")
    @classmethod
    def normalize_identifier(cls, value) -> str:
        return str(value)

    @model_validator(mode="after")
    def validate_prediction_and_spans(self):
        sources = (
            self.predicted_blocks is not None,
            self.predicted_labels is not None,
            self.predicted_scores is not None,
        )
        if sum(sources) != 1:
            raise ValueError(
                "Exactly one of predicted_blocks, predicted_labels, or predicted_scores "
                "is required"
            )
        if self.predicted_labels is not None and len(self.predicted_labels) != self.sentence_count:
            raise ValueError("predicted_labels must contain one label per sentence")
        if self.predicted_scores is not None:
            if len(self.predicted_scores) != self.sentence_count:
                raise ValueError("predicted_scores must contain one score per sentence")
            if any(
                not math.isfinite(score) or score < 0 or score > 1
                for score in self.predicted_scores
            ):
                raise ValueError("predicted_scores must be finite probabilities from 0 to 1")
        if self.sentence_ordinals is not None:
            if len(self.sentence_ordinals) != self.sentence_count:
                raise ValueError("sentence_ordinals must contain one ordinal per sentence")
            if any(
                current <= previous
                for previous, current in zip(
                    self.sentence_ordinals,
                    self.sentence_ordinals[1:],
                    strict=False,
                )
            ):
                raise ValueError("sentence_ordinals must be strictly increasing")
        if self.sentence_texts is not None and len(self.sentence_texts) != self.sentence_count:
            raise ValueError("sentence_texts must contain one value per sentence")
        self._validate_block_set(self.reference_blocks, "reference_blocks")
        if self.predicted_blocks is not None:
            self._validate_block_set(self.predicted_blocks, "predicted_blocks")
        return self

    def _validate_block_set(self, spans: tuple[EvidenceSpan, ...], field_name: str) -> None:
        previous_end = -1
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if span.end >= self.sentence_count:
                raise ValueError(f"{field_name} contains a span outside the document")
            if span.start <= previous_end:
                raise ValueError(f"{field_name} must contain non-overlapping spans")
            previous_end = span.end


def canonical_evaluation_content_fingerprint(
    examples: list[EvidenceEvaluationExample],
) -> str:
    """Hash the canonical held-out inputs and references, never predictions.

    Document/target identity, exact sentence ordinals and text, target text, and
    reference blocks are all part of compatibility.  Missing content fields
    use explicit legacy sentinels so an old report cannot accidentally compare
    as identical to a content-bound report.
    """

    units = []
    for example in sorted(
        examples,
        key=lambda item: (item.document_id, item.target_version_id),
    ):
        ordinals: tuple[int | str, ...]
        if example.sentence_ordinals is None:
            ordinals = tuple(f"legacy-index:{index}" for index in range(example.sentence_count))
        else:
            ordinals = example.sentence_ordinals
        texts: tuple[str, ...]
        if example.sentence_texts is None:
            texts = tuple("legacy-text-unavailable" for _ in range(example.sentence_count))
        else:
            texts = example.sentence_texts
        units.append(
            {
                "document_id": example.document_id,
                "target_version_id": example.target_version_id,
                "target_text": (
                    example.target_text
                    if example.target_text is not None
                    else "legacy-target-text-unavailable"
                ),
                "sentences": [
                    {"ordinal": ordinal, "text": text}
                    for ordinal, text in zip(ordinals, texts, strict=True)
                ],
                "reference_blocks": [
                    {"start": span.start, "end": span.end}
                    for span in example.reference_blocks
                ],
            }
        )
    return canonical_fingerprint(units)


class EvidenceEvaluationContext(ContractModel):
    evaluation_dataset_hash: str = Field(min_length=1, max_length=255)
    target_version_ids: tuple[str, ...] = Field(min_length=1)
    training_dataset_hash: str | None = Field(default=None, max_length=255)
    validation_dataset_hash: str | None = Field(default=None, max_length=255)
    preprocessing_version: str = Field(default="unknown", min_length=1, max_length=120)
    decoder_config: dict = Field(default_factory=dict)

    @field_validator("target_version_ids", mode="before")
    @classmethod
    def normalize_target_ids(cls, values) -> tuple[str, ...]:
        normalized = tuple(sorted({str(value) for value in values}))
        return normalized

    @property
    def target_scope_hash(self) -> str:
        return canonical_fingerprint(list(self.target_version_ids))

    @property
    def evaluation_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "task_contract": EVIDENCE_TASK_CONTRACT.key,
                "prediction_schema_version": EVIDENCE_TASK_CONTRACT.prediction_schema_version,
                "evaluator_version": EVIDENCE_TASK_CONTRACT.evaluator_version,
                "metric_suite_version": EVIDENCE_TASK_CONTRACT.metric_suite_version,
                "target_scope_hash": self.target_scope_hash,
                "evaluation_dataset_hash": self.evaluation_dataset_hash,
            }
        )

    @property
    def controlled_cohort_fingerprint(self) -> str | None:
        if self.training_dataset_hash is None or self.validation_dataset_hash is None:
            return None
        return canonical_fingerprint(
            {
                "evaluation_fingerprint": self.evaluation_fingerprint,
                "training_dataset_hash": self.training_dataset_hash,
                "validation_dataset_hash": self.validation_dataset_hash,
                "preprocessing_version": self.preprocessing_version,
                "decoder_config": self.decoder_config,
            }
        )

    @property
    def preprocessing_fingerprint(self) -> str:
        return canonical_fingerprint(self.preprocessing_version)

    @property
    def decoder_fingerprint(self) -> str:
        return canonical_fingerprint(self.decoder_config)


class MetricValue(ContractModel):
    value: float | None
    support: int = Field(ge=0)
    reason: str | None = None


class ConfusionMatrix(ContractModel):
    true_positive: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)
    true_negative: int | None = Field(default=None, ge=0)


class BinaryMetricReport(ContractModel):
    confusion_matrix: ConfusionMatrix
    precision: MetricValue
    recall: MetricValue
    f1: MetricValue
    support: int = Field(ge=0)


class BoundaryMetricReport(ContractModel):
    mean_start_deviation: MetricValue
    mean_end_deviation: MetricValue


class BlockMetricReport(ContractModel):
    reference_count: int = Field(ge=0)
    predicted_count: int = Field(ge=0)
    exact: BinaryMetricReport
    iou: dict[str, BinaryMetricReport]
    boundary: BoundaryMetricReport


class EvidenceMetricBundle(ContractModel):
    document_count: int = Field(ge=0)
    evaluation_unit_count: int = Field(ge=0)
    sentence_count: int = Field(ge=0)
    sentence: BinaryMetricReport
    blocks: BlockMetricReport
    document_presence: BinaryMetricReport
    document_presence_accuracy: MetricValue


class EvidenceMacroMetrics(ContractModel):
    sentence_f1: MetricValue
    exact_block_f1: MetricValue
    block_iou_f1: dict[str, MetricValue]
    document_presence_f1: MetricValue


class BootstrapInterval(ContractModel):
    confidence_level: float = Field(gt=0, lt=1)
    lower: float | None
    upper: float | None
    samples: int = Field(ge=0)
    reason: str | None = None


class EvidenceEvaluationReport(ContractModel):
    schema_version: Literal["evidence-evaluation-report-v1"] = (
        "evidence-evaluation-report-v1"
    )
    task_contract: TaskContractDescriptor
    split: Literal["validation", "test"]
    evaluation_fingerprint: str
    controlled_cohort_fingerprint: str | None
    target_scope_hash: str
    evaluation_dataset_hash: str
    preprocessing_fingerprint: str
    decoder_fingerprint: str
    overall: EvidenceMetricBundle
    macro: EvidenceMacroMetrics
    per_target: dict[str, EvidenceMetricBundle]
    bootstrap: dict[str, BootstrapInterval]

    @property
    def selection_metric_value(self) -> float | None:
        return self.macro.block_iou_f1["0.50"].value

    def to_record_payload(self) -> dict:
        """Return the normalized summary consumed by ``EvaluationResult``.

        Full per-target details remain available on this immutable report and
        can be written to object storage; the database receives searchable
        scalar summaries, supports, confusion matrices, and confidence bounds.
        """

        values = {
            "sentence_precision": self.overall.sentence.precision,
            "sentence_recall": self.overall.sentence.recall,
            "sentence_f1": self.overall.sentence.f1,
            "exact_block_f1": self.overall.blocks.exact.f1,
            "block_iou_f1_0_25": self.overall.blocks.iou["0.25"].f1,
            "block_iou_f1_0_50": self.overall.blocks.iou["0.50"].f1,
            "block_iou_f1_0_75": self.overall.blocks.iou["0.75"].f1,
            "mean_start_boundary_deviation": (
                self.overall.blocks.boundary.mean_start_deviation
            ),
            "mean_end_boundary_deviation": (
                self.overall.blocks.boundary.mean_end_deviation
            ),
            "document_presence_f1": self.overall.document_presence.f1,
            "document_presence_accuracy": self.overall.document_presence_accuracy,
            "macro_sentence_f1": self.macro.sentence_f1,
            "macro_exact_block_f1": self.macro.exact_block_f1,
            "macro_block_iou_f1_0_25": self.macro.block_iou_f1["0.25"],
            "macro_block_iou_f1_0_50": self.macro.block_iou_f1["0.50"],
            "macro_block_iou_f1_0_75": self.macro.block_iou_f1["0.75"],
            "macro_document_presence_f1": self.macro.document_presence_f1,
        }
        confidence_interval_keys = {
            "sentence.f1": "sentence_f1",
            "blocks.exact.f1": "exact_block_f1",
            "blocks.iou.0.50.f1": "block_iou_f1_0_50",
            "document_presence.f1": "document_presence_f1",
            "macro.block_iou_f1.0.50": "macro_block_iou_f1_0_50",
        }
        return {
            "evaluator_key": self.task_contract.key,
            "evaluator_version": self.task_contract.evaluator_version,
            "metric_schema_version": self.task_contract.metric_suite_version,
            "prediction_schema_version": self.task_contract.prediction_schema_version,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "controlled_cohort_fingerprint": self.controlled_cohort_fingerprint,
            "dataset_fingerprint": self.evaluation_dataset_hash,
            "task_contract_key": self.task_contract.key,
            "task_contract_version": self.task_contract.contract_version,
            "target_scope_hash": self.target_scope_hash,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "decoder_fingerprint": self.decoder_fingerprint,
            "metrics": {key: metric.value for key, metric in values.items()},
            "supports": {key: metric.support for key, metric in values.items()},
            "confusion_matrix": {
                "sentence": self.overall.sentence.confusion_matrix.model_dump(),
                "exact_blocks": self.overall.blocks.exact.confusion_matrix.model_dump(),
                "block_iou": {
                    threshold: metric.confusion_matrix.model_dump()
                    for threshold, metric in self.overall.blocks.iou.items()
                },
                "document_presence": (
                    self.overall.document_presence.confusion_matrix.model_dump()
                ),
            },
            "confidence_intervals": {
                confidence_interval_keys.get(key, key): interval.model_dump()
                for key, interval in self.bootstrap.items()
            },
            "diagnostics": {
                "document_count": self.overall.document_count,
                "evaluation_unit_count": self.overall.evaluation_unit_count,
                "sentence_count": self.overall.sentence_count,
                "reference_block_count": self.overall.blocks.reference_count,
                "predicted_block_count": self.overall.blocks.predicted_count,
                "per_target": {
                    key: value.model_dump(mode="json")
                    for key, value in self.per_target.items()
                },
            },
            "selection_metric_key": "macro_block_iou_f1_0_50",
            "selection_metric_value": self.selection_metric_value,
        }


def blocks_from_bio(labels: tuple[str, ...] | list[str]) -> tuple[EvidenceSpan, ...]:
    blocks: list[EvidenceSpan] = []
    start: int | None = None
    for ordinal, label in enumerate(labels):
        if label not in {"O", "B", "I"}:
            raise ValueError(f"Unknown BIO label {label!r}")
        if label == "B":
            if start is not None:
                blocks.append(EvidenceSpan(start=start, end=ordinal - 1))
            start = ordinal
        elif label == "I":
            if start is None:
                start = ordinal
        elif start is not None:
            blocks.append(EvidenceSpan(start=start, end=ordinal - 1))
            start = None
    if start is not None:
        blocks.append(EvidenceSpan(start=start, end=len(labels) - 1))
    return tuple(blocks)


def blocks_from_scores(
    scores: tuple[float, ...] | list[float],
    *,
    threshold: float,
) -> tuple[EvidenceSpan, ...]:
    if not 0 <= threshold <= 1:
        raise ValueError("Sentence-score threshold must be from 0 to 1")
    labels = tuple("I" if score >= threshold else "O" for score in scores)
    return blocks_from_bio(labels)


def canonical_prediction_blocks(
    example: EvidenceEvaluationExample,
    *,
    sentence_score_threshold: float,
) -> tuple[EvidenceSpan, ...]:
    if example.predicted_blocks is not None:
        return tuple(sorted(example.predicted_blocks, key=lambda span: (span.start, span.end)))
    if example.predicted_labels is not None:
        return blocks_from_bio(example.predicted_labels)
    assert example.predicted_scores is not None
    return blocks_from_scores(example.predicted_scores, threshold=sentence_score_threshold)


def _maximum_cardinality_matches(
    references: tuple[tuple[int, int], ...],
    predictions: tuple[tuple[int, int], ...],
    threshold: float,
) -> int:
    """Return maximum one-to-one match count at a threshold.

    Maximizing total IoU and then applying a threshold can under-count valid
    matches.  This threshold-specific augmenting-path matching maximizes the
    actual metric numerator.
    """

    adjacency: list[list[int]] = []
    for reference in references:
        neighbours = []
        for prediction_index, prediction in enumerate(predictions):
            left = max(reference[0], prediction[0])
            right = min(reference[1], prediction[1])
            intersection = max(0, right - left + 1)
            union = max(reference[1], prediction[1]) - min(reference[0], prediction[0]) + 1
            if intersection / union >= threshold:
                neighbours.append(prediction_index)
        adjacency.append(neighbours)

    prediction_owner: dict[int, int] = {}

    def assign(reference_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[reference_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            owner = prediction_owner.get(prediction_index)
            if owner is None or assign(owner, visited):
                prediction_owner[prediction_index] = reference_index
                return True
        return False

    return sum(assign(index, set()) for index in range(len(references)))


@dataclass
class _Accumulator:
    evaluation_unit_count: int = 0
    document_ids: set[str] = field(default_factory=set)
    sentence_count: int = 0
    sentence_tp: int = 0
    sentence_fp: int = 0
    sentence_fn: int = 0
    sentence_tn: int = 0
    reference_blocks: int = 0
    predicted_blocks: int = 0
    exact_matches: int = 0
    iou_matches_025: int = 0
    iou_matches_050: int = 0
    iou_matches_075: int = 0
    boundary_pairs: int = 0
    start_deviation_total: float = 0
    end_deviation_total: float = 0
    presence_tp: int = 0
    presence_fp: int = 0
    presence_fn: int = 0
    presence_tn: int = 0

    def add(
        self,
        example: EvidenceEvaluationExample,
        predictions: tuple[EvidenceSpan, ...],
    ) -> None:
        references = tuple(span.as_tuple() for span in example.reference_blocks)
        predicted = tuple(span.as_tuple() for span in predictions)
        reference_sentences = {
            ordinal for start, end in references for ordinal in range(start, end + 1)
        }
        predicted_sentences = {
            ordinal for start, end in predicted for ordinal in range(start, end + 1)
        }
        self.evaluation_unit_count += 1
        self.document_ids.add(example.document_id)
        self.sentence_count += example.sentence_count
        self.sentence_tp += len(reference_sentences & predicted_sentences)
        self.sentence_fp += len(predicted_sentences - reference_sentences)
        self.sentence_fn += len(reference_sentences - predicted_sentences)
        self.sentence_tn += example.sentence_count - len(reference_sentences | predicted_sentences)
        self.reference_blocks += len(references)
        self.predicted_blocks += len(predicted)
        self.exact_matches += len(set(references) & set(predicted))
        self.iou_matches_025 += _maximum_cardinality_matches(references, predicted, 0.25)
        self.iou_matches_050 += _maximum_cardinality_matches(references, predicted, 0.50)
        self.iou_matches_075 += _maximum_cardinality_matches(references, predicted, 0.75)
        for reference_index, predicted_index, iou in maximum_weight_matching(
            list(references), list(predicted)
        ):
            if iou <= 0:
                continue
            self.boundary_pairs += 1
            self.start_deviation_total += abs(
                references[reference_index][0] - predicted[predicted_index][0]
            )
            self.end_deviation_total += abs(
                references[reference_index][1] - predicted[predicted_index][1]
            )
        reference_present = bool(references)
        predicted_present = bool(predicted)
        if reference_present and predicted_present:
            self.presence_tp += 1
        elif predicted_present:
            self.presence_fp += 1
        elif reference_present:
            self.presence_fn += 1
        else:
            self.presence_tn += 1


def _ratio(numerator: int, denominator: int, support: int, reason: str) -> MetricValue:
    if denominator == 0:
        return MetricValue(value=None, support=support, reason=reason)
    return MetricValue(value=numerator / denominator, support=support)


def _binary_report(
    *,
    true_positive: int,
    false_positive: int,
    false_negative: int,
    true_negative: int | None,
) -> BinaryMetricReport:
    reference = true_positive + false_negative
    predicted = true_positive + false_positive
    precision = _ratio(
        true_positive,
        predicted,
        predicted,
        "No positive predictions",
    )
    recall = _ratio(
        true_positive,
        reference,
        reference,
        "No positive references",
    )
    if reference + predicted == 0:
        f1 = MetricValue(
            value=None,
            support=0,
            reason="No positive references or predictions",
        )
    else:
        f1 = MetricValue(
            value=(2 * true_positive / (reference + predicted)),
            support=reference,
        )
    return BinaryMetricReport(
        confusion_matrix=ConfusionMatrix(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            true_negative=true_negative,
        ),
        precision=precision,
        recall=recall,
        f1=f1,
        support=reference,
    )


def _bundle(accumulator: _Accumulator) -> EvidenceMetricBundle:
    iou_matches = {
        "0.25": accumulator.iou_matches_025,
        "0.50": accumulator.iou_matches_050,
        "0.75": accumulator.iou_matches_075,
    }
    if accumulator.boundary_pairs:
        start_deviation = MetricValue(
            value=accumulator.start_deviation_total / accumulator.boundary_pairs,
            support=accumulator.boundary_pairs,
        )
        end_deviation = MetricValue(
            value=accumulator.end_deviation_total / accumulator.boundary_pairs,
            support=accumulator.boundary_pairs,
        )
    else:
        start_deviation = MetricValue(
            value=None,
            support=0,
            reason="No overlapping reference/prediction block pairs",
        )
        end_deviation = MetricValue(
            value=None,
            support=0,
            reason="No overlapping reference/prediction block pairs",
        )
    presence = _binary_report(
        true_positive=accumulator.presence_tp,
        false_positive=accumulator.presence_fp,
        false_negative=accumulator.presence_fn,
        true_negative=accumulator.presence_tn,
    )
    return EvidenceMetricBundle(
        document_count=len(accumulator.document_ids),
        evaluation_unit_count=accumulator.evaluation_unit_count,
        sentence_count=accumulator.sentence_count,
        sentence=_binary_report(
            true_positive=accumulator.sentence_tp,
            false_positive=accumulator.sentence_fp,
            false_negative=accumulator.sentence_fn,
            true_negative=accumulator.sentence_tn,
        ),
        blocks=BlockMetricReport(
            reference_count=accumulator.reference_blocks,
            predicted_count=accumulator.predicted_blocks,
            exact=_binary_report(
                true_positive=accumulator.exact_matches,
                false_positive=accumulator.predicted_blocks - accumulator.exact_matches,
                false_negative=accumulator.reference_blocks - accumulator.exact_matches,
                true_negative=None,
            ),
            iou={
                threshold: _binary_report(
                    true_positive=matches,
                    false_positive=accumulator.predicted_blocks - matches,
                    false_negative=accumulator.reference_blocks - matches,
                    true_negative=None,
                )
                for threshold, matches in iou_matches.items()
            },
            boundary=BoundaryMetricReport(
                mean_start_deviation=start_deviation,
                mean_end_deviation=end_deviation,
            ),
        ),
        document_presence=presence,
        document_presence_accuracy=MetricValue(
            value=(
                (accumulator.presence_tp + accumulator.presence_tn)
                / accumulator.evaluation_unit_count
                if accumulator.evaluation_unit_count
                else None
            ),
            support=accumulator.evaluation_unit_count,
            reason=None if accumulator.evaluation_unit_count else "No evaluation documents",
        ),
    )


def _macro_metric(values: list[MetricValue]) -> MetricValue:
    defined = [value.value for value in values if value.value is not None]
    if not defined:
        return MetricValue(value=None, support=0, reason="No targets have a defined metric")
    return MetricValue(value=sum(defined) / len(defined), support=len(defined))


def _macro(per_target: dict[str, EvidenceMetricBundle]) -> EvidenceMacroMetrics:
    bundles = list(per_target.values())
    return EvidenceMacroMetrics(
        sentence_f1=_macro_metric([bundle.sentence.f1 for bundle in bundles]),
        exact_block_f1=_macro_metric([bundle.blocks.exact.f1 for bundle in bundles]),
        block_iou_f1={
            threshold: _macro_metric(
                [bundle.blocks.iou[threshold].f1 for bundle in bundles]
            )
            for threshold in ("0.25", "0.50", "0.75")
        },
        document_presence_f1=_macro_metric(
            [bundle.document_presence.f1 for bundle in bundles]
        ),
    )


def _calculate_metrics(
    examples: list[EvidenceEvaluationExample],
    *,
    sentence_score_threshold: float,
) -> tuple[EvidenceMetricBundle, EvidenceMacroMetrics, dict[str, EvidenceMetricBundle]]:
    overall_accumulator = _Accumulator()
    target_accumulators: defaultdict[str, _Accumulator] = defaultdict(_Accumulator)
    for example in examples:
        predictions = canonical_prediction_blocks(
            example,
            sentence_score_threshold=sentence_score_threshold,
        )
        overall_accumulator.add(example, predictions)
        target_accumulators[example.target_version_id].add(example, predictions)
    per_target = {
        target_id: _bundle(target_accumulators[target_id])
        for target_id in sorted(target_accumulators)
    }
    return _bundle(overall_accumulator), _macro(per_target), per_target


def _percentile(values: list[float], probability: float) -> float:
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def _bootstrap(
    examples: list[EvidenceEvaluationExample],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    sentence_score_threshold: float,
) -> dict[str, BootstrapInterval]:
    metric_values: defaultdict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    examples_by_document: defaultdict[str, list[EvidenceEvaluationExample]] = defaultdict(list)
    for example in examples:
        examples_by_document[example.document_id].append(example)
    document_ids = sorted(examples_by_document)
    for _sample in range(samples):
        sampled_document_ids = [
            document_ids[rng.randrange(len(document_ids))] for _ in document_ids
        ]
        resampled = [
            example
            for document_id in sampled_document_ids
            for example in examples_by_document[document_id]
        ]
        overall, macro, _per_target = _calculate_metrics(
            resampled,
            sentence_score_threshold=sentence_score_threshold,
        )
        candidates = {
            "sentence.f1": overall.sentence.f1.value,
            "blocks.exact.f1": overall.blocks.exact.f1.value,
            "blocks.iou.0.50.f1": overall.blocks.iou["0.50"].f1.value,
            "document_presence.f1": overall.document_presence.f1.value,
            "macro.block_iou_f1.0.50": macro.block_iou_f1["0.50"].value,
        }
        for key, value in candidates.items():
            if value is not None:
                metric_values[key].append(value)

    alpha = (1 - confidence_level) / 2
    intervals = {}
    for key in (
        "sentence.f1",
        "blocks.exact.f1",
        "blocks.iou.0.50.f1",
        "document_presence.f1",
        "macro.block_iou_f1.0.50",
    ):
        values = metric_values[key]
        intervals[key] = BootstrapInterval(
            confidence_level=confidence_level,
            lower=_percentile(values, alpha) if values else None,
            upper=_percentile(values, 1 - alpha) if values else None,
            samples=len(values),
            reason=None if values else "Metric was undefined in every bootstrap sample",
        )
    return intervals


class EvidenceBlockEvaluator:
    descriptor = EVIDENCE_TASK_CONTRACT

    def evaluate(
        self,
        examples: list[EvidenceEvaluationExample | dict],
        *,
        context: EvidenceEvaluationContext | dict,
        split: Literal["validation", "test"],
        sentence_score_threshold: float = 0.5,
        bootstrap_samples: int = 1000,
        bootstrap_seed: int = 42,
        confidence_level: float = 0.95,
    ) -> EvidenceEvaluationReport:
        if not examples:
            raise ValueError("Evidence evaluation requires at least one document-target example")
        if not 0 <= sentence_score_threshold <= 1:
            raise ValueError("Sentence-score threshold must be from 0 to 1")
        if not 0 <= bootstrap_samples <= 10_000:
            raise ValueError("bootstrap_samples must be from 0 to 10000")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        normalized_examples = [
            example
            if isinstance(example, EvidenceEvaluationExample)
            else EvidenceEvaluationExample.model_validate(example)
            for example in examples
        ]
        normalized_context = (
            context
            if isinstance(context, EvidenceEvaluationContext)
            else EvidenceEvaluationContext.model_validate(context)
        )
        # A caller-provided frozen-dataset hash is useful lineage, but is not
        # sufficient for scientific compatibility: two exports can retain the
        # same identifiers and labels while their evaluated text changes.
        # Combine it with a canonical hash of the actual held-out content.
        normalized_context = normalized_context.model_copy(
            update={
                "evaluation_dataset_hash": canonical_fingerprint(
                    {
                        "declared_dataset_hash": normalized_context.evaluation_dataset_hash,
                        "canonical_content_hash": canonical_evaluation_content_fingerprint(
                            normalized_examples
                        ),
                    }
                )
            }
        )
        example_keys = [
            (example.document_id, example.target_version_id)
            for example in normalized_examples
        ]
        if len(example_keys) != len(set(example_keys)):
            raise ValueError(
                "Evidence evaluation accepts one example per document and target"
            )
        example_targets = {example.target_version_id for example in normalized_examples}
        if example_targets != set(normalized_context.target_version_ids):
            raise ValueError(
                "Evaluation examples must cover the complete frozen target scope"
            )
        overall, macro, per_target = _calculate_metrics(
            normalized_examples,
            sentence_score_threshold=sentence_score_threshold,
        )
        bootstrap = (
            _bootstrap(
                normalized_examples,
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
                sentence_score_threshold=sentence_score_threshold,
            )
            if bootstrap_samples
            else {}
        )
        return EvidenceEvaluationReport(
            task_contract=self.descriptor,
            split=split,
            evaluation_fingerprint=normalized_context.evaluation_fingerprint,
            controlled_cohort_fingerprint=(
                normalized_context.controlled_cohort_fingerprint
            ),
            target_scope_hash=normalized_context.target_scope_hash,
            evaluation_dataset_hash=normalized_context.evaluation_dataset_hash,
            preprocessing_fingerprint=normalized_context.preprocessing_fingerprint,
            decoder_fingerprint=normalized_context.decoder_fingerprint,
            overall=overall,
            macro=macro,
            per_target=per_target,
            bootstrap=bootstrap,
        )


def checkpoint_selection_key(
    report: EvidenceEvaluationReport,
    *,
    checkpoint_ordinal: int,
) -> tuple[float, float, int]:
    """Higher sorts first: validation IoU@.50, exact F1, then earlier checkpoint."""

    if report.split != "validation":
        raise ValueError("Checkpoint selection may use validation results only")
    primary = report.macro.block_iou_f1["0.50"].value
    tiebreaker = report.macro.exact_block_f1.value
    return (
        primary if primary is not None else -math.inf,
        tiebreaker if tiebreaker is not None else -math.inf,
        -checkpoint_ordinal,
    )


def select_best_checkpoint(
    candidates: list[tuple[int, EvidenceEvaluationReport]],
) -> tuple[int, EvidenceEvaluationReport]:
    if not candidates:
        raise ValueError("At least one validation checkpoint is required")
    return max(
        candidates,
        key=lambda candidate: checkpoint_selection_key(
            candidate[1], checkpoint_ordinal=candidate[0]
        ),
    )


def comparison_tier(
    reports: list[EvidenceEvaluationReport],
) -> Literal["controlled_cohort", "evaluation_compatible", "non_compatible"]:
    if len(reports) < 2:
        raise ValueError("Comparison requires at least two evaluation reports")
    if len({report.split for report in reports}) != 1:
        return "non_compatible"
    evaluation_fingerprints = {report.evaluation_fingerprint for report in reports}
    if len(evaluation_fingerprints) != 1:
        return "non_compatible"
    controlled_fingerprints = {
        report.controlled_cohort_fingerprint
        for report in reports
        if report.controlled_cohort_fingerprint is not None
    }
    if (
        len(controlled_fingerprints) == 1
        and all(report.controlled_cohort_fingerprint is not None for report in reports)
    ):
        return "controlled_cohort"
    return "evaluation_compatible"
