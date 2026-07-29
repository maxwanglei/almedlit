from dataclasses import replace

import pytest

from al_medlit.training.windowing import (
    EvidenceBlockWindowBuilder,
    GoldBlockRange,
    ReviewedInterval,
    TargetCondition,
    WindowBuilderConfig,
    WindowBuildError,
    WindowSentenceInput,
    assign_grouped_splits,
)


def _sentences(count: int, *, tokens: int = 2) -> list[WindowSentenceInput]:
    return [
        WindowSentenceInput(
            id=index + 100,
            ordinal=index,
            paragraph_ordinal=index // 2,
            section_path=("Results",) if index < 4 else ("Discussion",),
            text=f"Sentence {index}.",
            start_char=index * 20,
            end_char=index * 20 + 11,
            token_count=tokens,
        )
        for index in range(count)
    ]


def _target() -> TargetCondition:
    return TargetCondition(id=7, key="pk", name="PK evidence", text="Find PK evidence")


def test_builds_complete_overlapping_windows_and_contiguous_bio_labels():
    builder = EvidenceBlockWindowBuilder(
        lambda text: len(text.split()),
        WindowBuilderConfig(
            max_tokens=8,
            overlap_tokens=2,
            reserved_special_tokens=0,
            target_conditioning=False,
        ),
    )
    result = builder.build(
        document_id=1,
        structure_version_id=2,
        target=_target(),
        sentences=_sentences(7),
        gold_blocks=[GoldBlockRange(id=9, start_ordinal=2, end_ordinal=4)],
        reviewed_intervals=[ReviewedInterval(start_ordinal=0, end_ordinal=6)],
    )

    assert all(window.token_count <= 8 for window in result.windows)
    assert any(
        window.start_sentence_ordinal <= 2 and window.end_sentence_ordinal >= 4
        for window in result.windows
    )
    labels = {
        sentence.ordinal: sentence.label
        for window in result.windows
        for sentence in window.sentences
    }
    assert [labels[index] for index in range(2, 5)] == ["B", "I", "I"]
    assert result.report.total_reviewed_sentences == 7
    assert result.report.positive_window_ratio > 0


def test_unreviewed_gold_never_becomes_a_training_sequence():
    builder = EvidenceBlockWindowBuilder(
        lambda _text: 1,
        WindowBuilderConfig(
            max_tokens=4,
            overlap_tokens=1,
            reserved_special_tokens=0,
            target_conditioning=False,
        ),
    )
    result = builder.build(
        document_id=1,
        structure_version_id=2,
        target=_target(),
        sentences=_sentences(8, tokens=1),
        gold_blocks=[GoldBlockRange(id=22, start_ordinal=3, end_ordinal=5)],
        reviewed_intervals=[ReviewedInterval(start_ordinal=0, end_ordinal=3)],
    )

    assert result.report.unreviewed_gold_blocks_excluded == [22]
    assert all(
        not (window.start_sentence_ordinal <= 5 and window.end_sentence_ordinal >= 3)
        for window in result.windows
    )
    assert all(
        sentence.label not in {"B", "I"}
        for window in result.windows
        for sentence in window.sentences
    )


def test_oversized_sentence_and_gold_block_are_reported_without_truncation():
    sentences = _sentences(3, tokens=2)
    sentences[1] = replace(sentences[1], token_count=9)
    builder = EvidenceBlockWindowBuilder(
        lambda _text: 1,
        WindowBuilderConfig(
            max_tokens=6,
            overlap_tokens=1,
            reserved_special_tokens=0,
            target_conditioning=False,
        ),
    )
    result = builder.build(
        document_id=1,
        structure_version_id=2,
        target=_target(),
        sentences=sentences,
        gold_blocks=[GoldBlockRange(id=31, start_ordinal=0, end_ordinal=1)],
        reviewed_intervals=[ReviewedInterval(start_ordinal=0, end_ordinal=2)],
    )

    assert [item.id for item in result.report.oversized_sentences] == [101]
    assert [item.id for item in result.report.oversized_gold_blocks] == [31]
    assert all(
        1 not in [sentence.ordinal for sentence in window.sentences]
        for window in result.windows
    )


def test_rejects_overlapping_gold_blocks():
    builder = EvidenceBlockWindowBuilder(lambda _text: 1)
    with pytest.raises(WindowBuildError, match="may not overlap"):
        builder.build(
            document_id=1,
            structure_version_id=2,
            target=_target(),
            sentences=_sentences(5),
            gold_blocks=[
                GoldBlockRange(id=1, start_ordinal=1, end_ordinal=3),
                GoldBlockRange(id=2, start_ordinal=3, end_ordinal=4),
            ],
        )


def test_grouped_split_is_deterministic_and_never_splits_a_group():
    groups = {1: "family-a", 2: "family-a", 3: "family-b", 4: "family-c", 5: "family-d"}
    first = assign_grouped_splits(groups, seed=42)
    second = assign_grouped_splits(groups, seed=42)

    assert first == second
    assert first[1] == first[2]
    assert set(first.values()) == {"train", "validation", "test"}
