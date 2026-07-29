import hashlib
import json
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

SentenceLabel = Literal["O", "B", "I", "IGNORE"]


class WindowBuildError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TargetCondition:
    id: int
    key: str
    name: str
    text: str


@dataclass(frozen=True, slots=True)
class WindowSentenceInput:
    id: int
    ordinal: int
    paragraph_ordinal: int
    section_path: tuple[str, ...]
    text: str
    start_char: int
    end_char: int
    token_count: int | None = None


@dataclass(frozen=True, slots=True)
class GoldBlockRange:
    id: int
    start_ordinal: int
    end_ordinal: int


@dataclass(frozen=True, slots=True)
class ReviewedInterval:
    start_ordinal: int
    end_ordinal: int


@dataclass(frozen=True, slots=True)
class WindowSentence:
    id: int
    ordinal: int
    paragraph_ordinal: int
    section_path: tuple[str, ...]
    text: str
    start_char: int
    end_char: int
    token_count: int
    label: SentenceLabel
    reviewed: bool


@dataclass(frozen=True, slots=True)
class EvidenceWindow:
    id: str
    document_id: int
    structure_version_id: int
    target_version_id: int
    start_sentence_ordinal: int
    end_sentence_ordinal: int
    token_count: int
    sentences: tuple[WindowSentence, ...]
    supplemental_for_block_id: int | None = None


@dataclass(frozen=True, slots=True)
class OversizedItem:
    kind: Literal["sentence", "gold_block"]
    id: int
    start_ordinal: int
    end_ordinal: int
    token_count: int


@dataclass(slots=True)
class WindowValidationReport:
    total_documents: int = 1
    total_reviewed_sentences: int = 0
    total_gold_blocks: int = 0
    block_length_distribution: dict[str, int] = field(default_factory=dict)
    blocks_fitting_512: int = 0
    blocks_fitting_1024: int = 0
    blocks_fitting_2048: int = 0
    blocks_fitting_4096: int = 0
    oversized_gold_blocks: list[OversizedItem] = field(default_factory=list)
    oversized_sentences: list[OversizedItem] = field(default_factory=list)
    unreviewed_gold_blocks_excluded: list[int] = field(default_factory=list)
    unreviewed_regions_excluded: int = 0
    windows_generated: int = 0
    positive_window_ratio: float = 0.0
    label_frequencies: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WindowBuildResult:
    windows: tuple[EvidenceWindow, ...]
    report: WindowValidationReport


@dataclass(frozen=True, slots=True)
class WindowBuilderConfig:
    max_tokens: int = 4096
    overlap_tokens: int = 512
    reserved_special_tokens: int = 4
    target_conditioning: bool = True
    prefer_boundary_min_fill: float = 0.5
    require_reviewed_gold: bool = True

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise WindowBuildError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise WindowBuildError("overlap_tokens cannot be negative")
        if self.reserved_special_tokens < 0:
            raise WindowBuildError("reserved_special_tokens cannot be negative")
        if not 0 <= self.prefer_boundary_min_fill <= 1:
            raise WindowBuildError("prefer_boundary_min_fill must be between zero and one")


class EvidenceBlockWindowBuilder:
    """Build complete-sentence windows for both training and inference.

    The tokenizer-specific token counter is injected so this module remains
    dependency-free and ordinary CI never downloads a public model.
    """

    def __init__(
        self,
        token_counter: Callable[[str], int],
        config: WindowBuilderConfig | None = None,
    ) -> None:
        self.token_counter = token_counter
        self.config = config or WindowBuilderConfig()

    def build(
        self,
        *,
        document_id: int,
        structure_version_id: int,
        target: TargetCondition,
        sentences: Sequence[WindowSentenceInput],
        gold_blocks: Sequence[GoldBlockRange] = (),
        reviewed_intervals: Sequence[ReviewedInterval] = (),
    ) -> WindowBuildResult:
        ordered = sorted(sentences, key=lambda item: item.ordinal)
        self._validate_inputs(ordered, gold_blocks, reviewed_intervals)
        target_tokens = (
            self.token_counter(f"[TARGET] {target.name} {target.text}")
            if self.config.target_conditioning
            else 0
        )
        usable_tokens = (
            self.config.max_tokens - target_tokens - self.config.reserved_special_tokens
        )
        if usable_tokens <= 0:
            raise WindowBuildError("Target conditioning leaves no room for document sentences")

        token_counts = [
            sentence.token_count
            if sentence.token_count is not None
            else self.token_counter(f"[SENT] {sentence.text}")
            for sentence in ordered
        ]
        if any(count <= 0 for count in token_counts):
            raise WindowBuildError("Every sentence must have a positive token count")

        reviewed_ordinals = self._covered_ordinals(reviewed_intervals)
        block_by_ordinal: dict[int, tuple[int, SentenceLabel]] = {}
        valid_blocks: list[GoldBlockRange] = []
        excluded_block_ids: set[int] = set()
        report = WindowValidationReport(total_gold_blocks=len(gold_blocks))
        for block in gold_blocks:
            block_ordinals = range(block.start_ordinal, block.end_ordinal + 1)
            if self.config.require_reviewed_gold and any(
                ordinal not in reviewed_ordinals for ordinal in block_ordinals
            ):
                report.unreviewed_gold_blocks_excluded.append(block.id)
                excluded_block_ids.add(block.id)
                continue
            valid_blocks.append(block)
            for ordinal in block_ordinals:
                block_by_ordinal[ordinal] = (
                    block.id,
                    "B" if ordinal == block.start_ordinal else "I",
                )

        materialized: list[WindowSentence] = []
        for sentence, token_count in zip(ordered, token_counts, strict=True):
            block_label = block_by_ordinal.get(sentence.ordinal)
            reviewed = sentence.ordinal in reviewed_ordinals
            label: SentenceLabel
            if block_label is not None:
                label = block_label[1]
            elif reviewed:
                label = "O"
            else:
                label = "IGNORE"
            materialized.append(
                WindowSentence(
                    id=sentence.id,
                    ordinal=sentence.ordinal,
                    paragraph_ordinal=sentence.paragraph_ordinal,
                    section_path=sentence.section_path,
                    text=sentence.text,
                    start_char=sentence.start_char,
                    end_char=sentence.end_char,
                    token_count=token_count,
                    label=label,
                    reviewed=reviewed,
                )
            )

        report.total_reviewed_sentences = sum(item.reviewed for item in materialized)
        report.unreviewed_regions_excluded = sum(not item.reviewed for item in materialized)
        report.label_frequencies = dict(Counter(item.label for item in materialized))
        report.oversized_sentences = [
            OversizedItem(
                kind="sentence",
                id=item.id,
                start_ordinal=item.ordinal,
                end_ordinal=item.ordinal,
                token_count=item.token_count,
            )
            for item in materialized
            if item.token_count > usable_tokens
        ]

        ordinal_to_index = {item.ordinal: index for index, item in enumerate(materialized)}
        fitting_blocks: list[GoldBlockRange] = []
        block_lengths: Counter[str] = Counter()
        for block in valid_blocks:
            start_index = ordinal_to_index[block.start_ordinal]
            end_index = ordinal_to_index[block.end_ordinal]
            token_count = sum(
                item.token_count for item in materialized[start_index : end_index + 1]
            )
            sentence_count = end_index - start_index + 1
            block_lengths[self._length_bucket(sentence_count)] += 1
            report.blocks_fitting_512 += token_count <= 512
            report.blocks_fitting_1024 += token_count <= 1024
            report.blocks_fitting_2048 += token_count <= 2048
            report.blocks_fitting_4096 += token_count <= 4096
            if token_count > usable_tokens:
                report.oversized_gold_blocks.append(
                    OversizedItem(
                        kind="gold_block",
                        id=block.id,
                        start_ordinal=block.start_ordinal,
                        end_ordinal=block.end_ordinal,
                        token_count=token_count,
                    )
                )
            else:
                fitting_blocks.append(block)
        report.block_length_distribution = dict(block_lengths)

        ranges = self._base_ranges(materialized, usable_tokens)
        range_records: list[tuple[int, int, int | None]] = [
            (start, end, None) for start, end in ranges
        ]
        for block in fitting_blocks:
            if any(
                materialized[start].ordinal <= block.start_ordinal
                and materialized[end - 1].ordinal >= block.end_ordinal
                for start, end, _ in range_records
            ):
                continue
            range_records.append(
                (*self._supplemental_range(materialized, block, usable_tokens), block.id)
            )

        deduplicated: dict[tuple[int, int], int | None] = {}
        for start, end, supplemental_id in range_records:
            deduplicated.setdefault((start, end), supplemental_id)

        windows: list[EvidenceWindow] = []
        for (start, end), supplemental_id in sorted(deduplicated.items()):
            window_sentences = tuple(materialized[start:end])
            if not window_sentences:
                continue
            # A partially reviewed gold block must not leak a false sequence into
            # training. Exclude every context window touching such a block.
            if excluded_block_ids and any(
                block.start_ordinal <= window_sentences[-1].ordinal
                and block.end_ordinal >= window_sentences[0].ordinal
                for block in gold_blocks
                if block.id in excluded_block_ids
            ):
                continue
            stable_id = self._stable_window_id(
                document_id=document_id,
                structure_version_id=structure_version_id,
                target_version_id=target.id,
                start_ordinal=window_sentences[0].ordinal,
                end_ordinal=window_sentences[-1].ordinal,
                max_tokens=self.config.max_tokens,
                overlap_tokens=self.config.overlap_tokens,
            )
            windows.append(
                EvidenceWindow(
                    id=stable_id,
                    document_id=document_id,
                    structure_version_id=structure_version_id,
                    target_version_id=target.id,
                    start_sentence_ordinal=window_sentences[0].ordinal,
                    end_sentence_ordinal=window_sentences[-1].ordinal,
                    token_count=target_tokens
                    + self.config.reserved_special_tokens
                    + sum(item.token_count for item in window_sentences),
                    sentences=window_sentences,
                    supplemental_for_block_id=supplemental_id,
                )
            )

        report.windows_generated = len(windows)
        positive_windows = sum(
            any(sentence.label in {"B", "I"} for sentence in window.sentences)
            for window in windows
        )
        report.positive_window_ratio = positive_windows / len(windows) if windows else 0.0
        return WindowBuildResult(tuple(windows), report)

    @staticmethod
    def _validate_inputs(
        sentences: Sequence[WindowSentenceInput],
        gold_blocks: Sequence[GoldBlockRange],
        reviewed_intervals: Sequence[ReviewedInterval],
    ) -> None:
        ordinals = [item.ordinal for item in sentences]
        if len(ordinals) != len(set(ordinals)):
            raise WindowBuildError("Sentence ordinals must be unique")
        if any(item.end_char <= item.start_char for item in sentences):
            raise WindowBuildError("Sentence character ranges must be non-empty")
        ordinal_set = set(ordinals)
        previous_end = -1
        for block in sorted(gold_blocks, key=lambda item: item.start_ordinal):
            if block.end_ordinal < block.start_ordinal:
                raise WindowBuildError("Gold block boundaries are reversed")
            if block.start_ordinal <= previous_end:
                raise WindowBuildError("Gold blocks may not overlap")
            if any(
                ordinal not in ordinal_set
                for ordinal in range(block.start_ordinal, block.end_ordinal + 1)
            ):
                raise WindowBuildError("Gold block references a missing sentence ordinal")
            previous_end = block.end_ordinal
        for interval in reviewed_intervals:
            if interval.end_ordinal < interval.start_ordinal:
                raise WindowBuildError("Reviewed interval boundaries are reversed")

    @staticmethod
    def _covered_ordinals(intervals: Sequence[ReviewedInterval]) -> set[int]:
        covered: set[int] = set()
        for interval in intervals:
            covered.update(range(interval.start_ordinal, interval.end_ordinal + 1))
        return covered

    def _base_ranges(
        self,
        sentences: Sequence[WindowSentence],
        usable_tokens: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < len(sentences):
            if sentences[start].token_count > usable_tokens:
                start += 1
                continue
            end = start
            used = 0
            while end < len(sentences) and used + sentences[end].token_count <= usable_tokens:
                used += sentences[end].token_count
                end += 1
            preferred_end = self._preferred_end(sentences, start, end, used, usable_tokens)
            end = max(start + 1, preferred_end)
            ranges.append((start, end))
            if end >= len(sentences):
                break
            next_start = end
            overlap = 0
            while next_start > start + 1 and overlap < self.config.overlap_tokens:
                next_start -= 1
                overlap += sentences[next_start].token_count
            next_start = self._prefer_paragraph_start(sentences, start, next_start, end)
            start = max(start + 1, next_start)
        return ranges

    def _preferred_end(
        self,
        sentences: Sequence[WindowSentence],
        start: int,
        end: int,
        used: int,
        usable_tokens: int,
    ) -> int:
        if end >= len(sentences) or used < usable_tokens * self.config.prefer_boundary_min_fill:
            return end
        minimum_tokens = usable_tokens * self.config.prefer_boundary_min_fill
        running = 0
        candidates: list[tuple[int, bool, int]] = []
        for index in range(start, end):
            running += sentences[index].token_count
            if running < minimum_tokens or index + 1 >= end:
                continue
            section_break = sentences[index].section_path != sentences[index + 1].section_path
            paragraph_break = (
                sentences[index].paragraph_ordinal
                != sentences[index + 1].paragraph_ordinal
            )
            if section_break or paragraph_break:
                candidates.append((index + 1, section_break, running))
        section_candidates = [item for item in candidates if item[1]]
        selected = (
            section_candidates[-1]
            if section_candidates
            else (candidates[-1] if candidates else None)
        )
        return selected[0] if selected else end

    @staticmethod
    def _prefer_paragraph_start(
        sentences: Sequence[WindowSentence],
        previous_start: int,
        candidate: int,
        end: int,
    ) -> int:
        if candidate <= previous_start or candidate >= end:
            return candidate
        paragraph = sentences[candidate].paragraph_ordinal
        while (
            candidate > previous_start + 1
            and sentences[candidate - 1].paragraph_ordinal == paragraph
        ):
            candidate -= 1
        return candidate

    @staticmethod
    def _supplemental_range(
        sentences: Sequence[WindowSentence],
        block: GoldBlockRange,
        usable_tokens: int,
    ) -> tuple[int, int]:
        index_by_ordinal = {item.ordinal: index for index, item in enumerate(sentences)}
        start = index_by_ordinal[block.start_ordinal]
        end = index_by_ordinal[block.end_ordinal] + 1
        used = sum(item.token_count for item in sentences[start:end])
        left = start - 1
        right = end
        while True:
            options: list[tuple[int, str]] = []
            if left >= 0 and used + sentences[left].token_count <= usable_tokens:
                options.append((sentences[left].token_count, "left"))
            if right < len(sentences) and used + sentences[right].token_count <= usable_tokens:
                options.append((sentences[right].token_count, "right"))
            if not options:
                break
            _, side = min(options, key=lambda item: (item[0], item[1]))
            if side == "left":
                used += sentences[left].token_count
                start = left
                left -= 1
            else:
                used += sentences[right].token_count
                right += 1
                end = right
        return start, end

    @staticmethod
    def _length_bucket(length: int) -> str:
        if length == 1:
            return "1"
        if length <= 3:
            return "2-3"
        if length <= 7:
            return "4-7"
        if length <= 15:
            return "8-15"
        return "16+"

    @staticmethod
    def _stable_window_id(**payload: int) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return f"ebw-{hashlib.sha256(encoded).hexdigest()[:24]}"


def assign_grouped_splits(
    document_groups: dict[int, str],
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[int, Literal["train", "validation", "test"]]:
    """Assign entire groups to one split using deterministic seeded shuffling."""
    if train_fraction <= 0 or validation_fraction < 0:
        raise WindowBuildError("Split fractions must be non-negative with a positive train split")
    if train_fraction + validation_fraction >= 1:
        raise WindowBuildError("Train and validation fractions must leave room for test")
    groups = sorted(set(document_groups.values()))
    random.Random(seed).shuffle(groups)
    count = len(groups)
    if count == 0:
        return {}
    validation_count = round(count * validation_fraction)
    test_count = round(count * (1 - train_fraction - validation_fraction))
    if count >= 3:
        validation_count = max(1, validation_count)
        test_count = max(1, test_count)
    while validation_count + test_count >= count:
        if test_count > validation_count and test_count > 0:
            test_count -= 1
        elif validation_count > 0:
            validation_count -= 1
        else:
            break
    validation_groups = set(groups[:validation_count])
    test_groups = set(groups[validation_count : validation_count + test_count])
    return {
        document_id: (
            "validation"
            if group_key in validation_groups
            else "test"
            if group_key in test_groups
            else "train"
        )
        for document_id, group_key in document_groups.items()
    }
