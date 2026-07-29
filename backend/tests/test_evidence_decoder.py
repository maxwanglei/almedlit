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
        ]
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
        ]
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
    aggregated = aggregate_window_logits([{0: [0, 0, 8]}])

    result = decode_evidence_blocks(_sentences(1), aggregated)

    assert len(result.blocks) == 1
    assert result.blocks[0].confidence > 0.99
    assert result.blocks[0].start_confidence > 0.99


def test_section_boundary_forces_i_to_begin_a_new_block():
    aggregated = aggregate_window_logits([{index: [0, 0, 8] for index in range(4)}])
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
    aggregated = aggregate_window_logits([{0: [0, 0.1, 0], 1: [0, 0, 0.1]}])
    result = decode_evidence_blocks(
        _sentences(2),
        aggregated,
        DecoderConfig(block_threshold=0.9),
    )
    assert result.blocks == ()
    assert len(result.suppressed_blocks) == 1


def test_invalid_logits_are_rejected():
    with pytest.raises(DecoderError, match="exactly O/B/I"):
        aggregate_window_logits([{0: [1, 2]}])
