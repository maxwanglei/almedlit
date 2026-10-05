import math

import pytest

from al_medlit.inference.decoder import (
    DecoderConfig,
    DecoderError,
    SentenceDecodingInput,
    aggregate_window_logits,
    decode_evidence_blocks,
)


def _sentences(count: int, *, section_break: int | None = None):
    return [
        SentenceDecodingInput(
            id=100 + ordinal,
            ordinal=ordinal,
            start_char=ordinal * 10,
            end_char=ordinal * 10 + 8,
            section_path=("A",) if section_break is None or ordinal < section_break else ("B",),
        )
        for ordinal in range(count)
    ]


def test_overlap_logits_are_averaged_with_contribution_counts():
    aggregated = aggregate_window_logits(
        [
            {0: [2, 0, 0], 1: [0, 2, 0]},
            {1: [0, 4, 0], 2: [0, 0, 3]},
        ],
        score_kind="logits",
    )

    assert aggregated[1].logits == (0.0, 3.0, 0.0)
    assert aggregated[1].contribution_count == 2
    assert aggregated[0].contribution_count == 1


def test_constrained_decoder_converts_initial_i_and_splits_on_new_b():
    aggregated = aggregate_window_logits(
        [
            {
                0: [0, 0, 8],  # I at document start becomes B
                1: [0, 0, 8],
                2: [0, 8, 0],  # New B closes the first block
                3: [8, 0, 0],
            }
        ],
        score_kind="logits",
    )
    result = decode_evidence_blocks(
        _sentences(4),
        aggregated,
        DecoderConfig(block_threshold=0.1),
    )

    assert [sentence.decoded_label for sentence in result.sentences] == ["B", "I", "B", "O"]
    assert [(block.start_ordinal, block.end_ordinal) for block in result.blocks] == [
        (0, 1),
        (2, 2),
    ]
    assert result.blocks[0].start_sentence_id == 100
    assert result.blocks[0].end_sentence_id == 101
    assert result.blocks[0].end_confidence > 0.99


def test_constrained_i_to_b_keeps_probability_mass_at_default_threshold():
    aggregated = aggregate_window_logits([{0: [0, 0, 8]}], score_kind="logits")

    result = decode_evidence_blocks(_sentences(1), aggregated)

    assert len(result.blocks) == 1
    assert result.blocks[0].confidence > 0.99
    assert result.blocks[0].start_confidence > 0.99


def test_section_boundary_forces_i_to_begin_a_new_block():
    aggregated = aggregate_window_logits(
        [{index: [0, 0, 8] for index in range(4)}],
        score_kind="logits",
    )
    result = decode_evidence_blocks(
        _sentences(4, section_break=2),
        aggregated,
        DecoderConfig(block_threshold=0, allow_cross_section=False),
    )
    assert [(block.start_ordinal, block.end_ordinal) for block in result.blocks] == [
        (0, 1),
        (2, 3),
    ]


def test_low_confidence_blocks_remain_in_diagnostics_but_are_not_emitted():
    aggregated = aggregate_window_logits(
        [{0: [0, 0.1, 0], 1: [0, 0, 0.1]}],
        score_kind="logits",
    )
    result = decode_evidence_blocks(
        _sentences(2),
        aggregated,
        DecoderConfig(block_threshold=0.9),
    )
    assert result.blocks == ()
    assert len(result.suppressed_blocks) == 1


def test_invalid_logits_are_rejected():
    with pytest.raises(DecoderError, match="exactly O/B/I"):
        aggregate_window_logits([{0: [1, 2]}], score_kind="logits")


ONE_HOT = {"O": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 0.0], "I": [0.0, 0.0, 1.0]}


def test_logits_are_softmaxed_after_averaging():
    aggregated = aggregate_window_logits([{0: [0, 6, 0]}, {0: [0, 2, 0]}], score_kind="logits")

    assert aggregated[0].logits == (0.0, 4.0, 0.0)
    assert aggregated[0].probabilities[1] == pytest.approx(math.exp(4) / (math.exp(4) + 2))


def test_probabilities_are_averaged_without_softmax():
    aggregated = aggregate_window_logits(
        [
            {0: [0.15, 0.70, 0.15], 1: ONE_HOT["B"]},
            {1: ONE_HOT["I"]},
            {1: ONE_HOT["B"]},
        ],
        score_kind="probabilities",
    )

    assert aggregated[0].logits is None
    assert aggregated[0].probabilities == pytest.approx((0.15, 0.70, 0.15))
    assert aggregated[1].probabilities == pytest.approx((0.0, 2 / 3, 1 / 3))
    assert aggregated[1].contribution_count == 3


def test_unanimous_one_hot_labels_keep_full_confidence():
    aggregated = aggregate_window_logits(
        [{0: ONE_HOT["B"], 1: ONE_HOT["I"], 2: ONE_HOT["O"]}],
        score_kind="probabilities",
    )
    result = decode_evidence_blocks(
        _sentences(3),
        aggregated,
        DecoderConfig(block_threshold=0.9),
    )

    assert [(block.start_ordinal, block.end_ordinal) for block in result.blocks] == [(0, 1)]
    block = result.blocks[0]
    assert (block.confidence, block.start_confidence, block.end_confidence) == (1.0, 1.0, 1.0)
    assert block.uncertainty == 0.0


def test_neural_probabilities_keep_their_scale_at_default_threshold():
    aggregated = aggregate_window_logits(
        [{0: [0.15, 0.70, 0.15], 1: [0.10, 0.20, 0.70], 2: [0.80, 0.10, 0.10]}],
        score_kind="probabilities",
    )
    result = decode_evidence_blocks(_sentences(3), aggregated)

    assert [(block.start_ordinal, block.end_ordinal) for block in result.blocks] == [(0, 1)]
    assert result.blocks[0].confidence == pytest.approx(0.70)
    assert result.blocks[0].end_confidence == pytest.approx(0.90)
    assert result.sentences[0].entropy == pytest.approx(
        -sum(value * math.log(value) for value in (0.15, 0.70, 0.15))
    )


def test_float32_rounded_probabilities_are_renormalized():
    aggregated = aggregate_window_logits(
        [{0: [0.2, 0.3, 0.5000001]}],
        score_kind="probabilities",
    )

    assert abs(sum(aggregated[0].probabilities) - 1.0) <= 1e-15


@pytest.mark.parametrize(
    "scores",
    [
        [0, 6, 0],  # Logits declared as probabilities.
        [-0.1, 0.6, 0.5],
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.3, 0.3, 0.401],  # Off by 1e-3, far beyond float32 softmax rounding.
    ],
)
def test_invalid_probabilities_are_rejected(scores):
    with pytest.raises(DecoderError, match="non-negative and sum to one"):
        aggregate_window_logits([{0: scores}], score_kind="probabilities")


def test_unknown_score_kind_is_rejected():
    with pytest.raises(DecoderError, match="Unsupported sentence score kind"):
        aggregate_window_logits([{0: [0, 1, 0]}], score_kind="log_probabilities")
