import pytest
from pydantic import ValidationError

from al_medlit.training.evaluation import task_evaluators
from al_medlit.training.evaluation.evidence import (
    EvidenceBlockEvaluator,
    EvidenceEvaluationContext,
    EvidenceEvaluationExample,
    blocks_from_bio,
    blocks_from_scores,
    checkpoint_selection_key,
    comparison_tier,
    configuration_fingerprint,
    select_best_checkpoint,
)


def _context(**overrides):
    values = {
        "evaluation_dataset_hash": "test-dataset-sha256",
        "target_version_ids": ["benefit"],
        "training_dataset_hash": "train-sha256",
        "validation_dataset_hash": "validation-sha256",
        "preprocessing_version": "evidence-windows-v2",
        "decoder_config": {"threshold": 0.5},
    }
    values.update(overrides)
    return EvidenceEvaluationContext.model_validate(values)


def _example(
    *,
    reference,
    predicted,
    target="benefit",
    sentence_count=5,
    sentence_texts=None,
):
    texts = sentence_texts or [f"Sentence {index}" for index in range(sentence_count)]
    return {
        "document_id": "doc-1",
        "target_version_id": target,
        "sentence_count": sentence_count,
        "sentence_ordinals": list(range(sentence_count)),
        "sentence_texts": texts,
        "target_text": f"Target {target}",
        "reference_blocks": [
            {"start": start, "end": end} for start, end in reference
        ],
        "predicted_blocks": [
            {"start": start, "end": end} for start, end in predicted
        ],
    }


def test_evaluator_registry_exposes_versioned_evidence_contract():
    evaluator = task_evaluators.get("evidence_blocks")

    assert evaluator.descriptor.prediction_schema_version == "evidence-block-prediction-v1"
    assert evaluator.descriptor.contract_version == "1"
    assert evaluator.descriptor.selection_metric == "macro.block_iou_f1.0.50"


def test_perfect_evaluation_reports_supports_confusion_and_all_block_thresholds():
    report = EvidenceBlockEvaluator().evaluate(
        [_example(reference=[(1, 2)], predicted=[(1, 2)])],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )

    assert report.overall.sentence.confusion_matrix.model_dump() == {
        "true_positive": 2,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 3,
    }
    assert report.overall.sentence.f1.value == 1
    assert report.overall.document_count == 1
    assert report.overall.evaluation_unit_count == 1
    assert report.overall.sentence.f1.support == 2
    assert report.overall.blocks.exact.f1.value == 1
    assert {
        threshold: metric.f1.value
        for threshold, metric in report.overall.blocks.iou.items()
    } == {"0.25": 1, "0.50": 1, "0.75": 1}
    assert report.overall.document_presence.confusion_matrix.true_positive == 1
    assert report.overall.document_presence_accuracy.value == 1
    assert report.macro.block_iou_f1["0.50"].value == 1
    payload = report.to_record_payload()
    assert payload["metrics"]["macro_block_iou_f1_0_50"] == 1
    assert payload["supports"]["sentence_f1"] == 2
    assert payload["selection_metric_key"] == "macro_block_iou_f1_0_50"
    assert payload["task_contract_key"] == "evidence_blocks"


def test_partial_boundaries_report_iou_levels_and_boundary_deviation():
    report = EvidenceBlockEvaluator().evaluate(
        [_example(reference=[(1, 2)], predicted=[(1, 3)])],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )

    assert report.overall.blocks.exact.f1.value == 0
    assert report.overall.blocks.iou["0.50"].f1.value == 1
    assert report.overall.blocks.iou["0.75"].f1.value == 0
    assert report.overall.blocks.boundary.mean_start_deviation.value == 0
    assert report.overall.blocks.boundary.mean_end_deviation.value == 1


def test_bio_and_sentence_score_predictions_share_canonical_block_evaluation():
    assert [(span.start, span.end) for span in blocks_from_bio(["I", "I", "O", "B"])] == [
        (0, 1),
        (3, 3),
    ]
    assert [
        (span.start, span.end)
        for span in blocks_from_scores([0.8, 0.7, 0.2, 0.9], threshold=0.5)
    ] == [(0, 1), (3, 3)]

    evaluator = EvidenceBlockEvaluator()
    shared = {
        "document_id": "doc-1",
        "target_version_id": "benefit",
        "sentence_count": 4,
        "reference_blocks": [{"start": 0, "end": 1}, {"start": 3, "end": 3}],
    }
    bio = evaluator.evaluate(
        [{**shared, "predicted_labels": ["I", "I", "O", "B"]}],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )
    scores = evaluator.evaluate(
        [{**shared, "predicted_scores": [0.8, 0.7, 0.2, 0.9]}],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )
    assert bio.overall == scores.overall


def test_empty_positive_target_is_na_not_zero_and_macro_skips_it():
    examples = [
        _example(reference=[(0, 0)], predicted=[(0, 0)], target="benefit"),
        {
            **_example(reference=[], predicted=[], target="harm"),
            "document_id": "doc-2",
        },
    ]
    report = EvidenceBlockEvaluator().evaluate(
        examples,
        context=_context(target_version_ids=["benefit", "harm"]),
        split="validation",
        bootstrap_samples=0,
    )

    empty = report.per_target["harm"]
    assert empty.sentence.f1.value is None
    assert empty.sentence.f1.reason == "No positive references or predictions"
    assert empty.document_presence_accuracy.value == 1
    assert report.macro.sentence_f1.value == 1
    assert report.macro.sentence_f1.support == 1


def test_example_contract_rejects_ambiguous_or_invalid_predictions():
    with pytest.raises(ValidationError, match="Exactly one"):
        EvidenceEvaluationExample.model_validate(
            {
                **_example(reference=[], predicted=[]),
                "predicted_labels": ["O"] * 5,
            }
        )
    with pytest.raises(ValidationError, match="outside the document"):
        EvidenceEvaluationExample.model_validate(
            _example(reference=[(0, 5)], predicted=[])
        )
    with pytest.raises(ValidationError, match="non-overlapping"):
        EvidenceEvaluationExample.model_validate(
            _example(reference=[(0, 2), (2, 3)], predicted=[])
        )


def test_fingerprints_define_evaluation_and_controlled_comparison_cohorts():
    first = _context(target_version_ids=[2, 1], decoder_config={"threshold": 0.4})
    reordered = _context(target_version_ids=["1", "2"], decoder_config={"threshold": 0.9})

    assert first.target_scope_hash == reordered.target_scope_hash
    assert first.evaluation_fingerprint == reordered.evaluation_fingerprint
    assert first.controlled_cohort_fingerprint != reordered.controlled_cohort_fingerprint
    assert configuration_fingerprint({"seed": 1, "model": {"width": 8}}) == (
        configuration_fingerprint({"seed": 999, "model": {"width": 8}})
    )
    assert configuration_fingerprint({"seed": 1, "model": {"width": 8}}) != (
        configuration_fingerprint({"seed": 1, "model": {"width": 16}})
    )


def test_comparison_tier_is_server_derived_from_evaluation_and_cohort_fingerprints():
    evaluator = EvidenceBlockEvaluator()
    example = [_example(reference=[(0, 0)], predicted=[(0, 0)])]
    controlled = evaluator.evaluate(
        example,
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )
    same_evaluation = evaluator.evaluate(
        example,
        context=_context(decoder_config={"threshold": 0.8}),
        split="validation",
        bootstrap_samples=0,
    )
    different_evaluation = evaluator.evaluate(
        example,
        context=_context(evaluation_dataset_hash="other-test-set"),
        split="validation",
        bootstrap_samples=0,
    )

    assert comparison_tier([controlled, controlled]) == "controlled_cohort"
    assert comparison_tier([controlled, same_evaluation]) == "evaluation_compatible"
    assert comparison_tier([controlled, different_evaluation]) == "non_compatible"
    assert comparison_tier(
        [controlled, controlled.model_copy(update={"split": "test"})]
    ) == "non_compatible"


def test_evaluation_fingerprint_binds_exact_held_out_sentence_content():
    evaluator = EvidenceBlockEvaluator()
    first = evaluator.evaluate(
        [_example(reference=[(0, 0)], predicted=[(0, 0)])],
        context=_context(),
        split="test",
        bootstrap_samples=0,
    )
    changed_text = evaluator.evaluate(
        [
            _example(
                reference=[(0, 0)],
                predicted=[(0, 0)],
                sentence_texts=["Changed source sentence", *[f"Sentence {i}" for i in range(1, 5)]],
            )
        ],
        context=_context(),
        split="test",
        bootstrap_samples=0,
    )

    assert first.evaluation_dataset_hash != changed_text.evaluation_dataset_hash
    assert first.evaluation_fingerprint != changed_text.evaluation_fingerprint
    assert comparison_tier([first, changed_text]) == "non_compatible"


def test_same_held_out_data_with_different_train_validation_is_not_controlled():
    evaluator = EvidenceBlockEvaluator()
    example = [_example(reference=[(0, 0)], predicted=[(0, 0)])]
    first = evaluator.evaluate(
        example,
        context=_context(
            training_dataset_hash="train-cohort-a",
            validation_dataset_hash="validation-cohort-a",
        ),
        split="test",
        bootstrap_samples=0,
    )
    second = evaluator.evaluate(
        example,
        context=_context(
            training_dataset_hash="train-cohort-b",
            validation_dataset_hash="validation-cohort-b",
        ),
        split="test",
        bootstrap_samples=0,
    )

    assert first.evaluation_fingerprint == second.evaluation_fingerprint
    assert first.controlled_cohort_fingerprint != second.controlled_cohort_fingerprint
    assert comparison_tier([first, second]) == "evaluation_compatible"


def test_bootstrap_is_document_deterministic_and_checkpoint_selection_never_uses_test():
    evaluator = EvidenceBlockEvaluator()
    examples = [
        _example(reference=[(0, 1)], predicted=[(0, 1)]),
        {
            **_example(reference=[(2, 2)], predicted=[]),
            "document_id": "doc-2",
        },
    ]
    first = evaluator.evaluate(
        examples,
        context=_context(),
        split="validation",
        bootstrap_samples=50,
        bootstrap_seed=7,
    )
    second = evaluator.evaluate(
        examples,
        context=_context(),
        split="validation",
        bootstrap_samples=50,
        bootstrap_seed=7,
    )
    assert first.bootstrap == second.bootstrap
    interval = first.bootstrap["blocks.iou.0.50.f1"]
    assert interval.lower is not None
    assert interval.upper is not None
    assert interval.lower <= interval.upper
    assert "macro_block_iou_f1_0_50" in first.to_record_payload()[
        "confidence_intervals"
    ]

    exact = evaluator.evaluate(
        [_example(reference=[(1, 2)], predicted=[(1, 2)])],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )
    partial = evaluator.evaluate(
        [_example(reference=[(1, 2)], predicted=[(1, 3)])],
        context=_context(),
        split="validation",
        bootstrap_samples=0,
    )
    ordinal, selected = select_best_checkpoint([(2, partial), (3, exact)])
    assert (ordinal, selected) == (3, exact)
    assert checkpoint_selection_key(exact, checkpoint_ordinal=3) > (
        checkpoint_selection_key(partial, checkpoint_ordinal=2)
    )

    test_report = exact.model_copy(update={"split": "test"})
    with pytest.raises(ValueError, match="validation results only"):
        checkpoint_selection_key(test_report, checkpoint_ordinal=1)
