import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

LABELS = ("O", "B", "I")


class DecoderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AggregatedSentenceLogits:
    ordinal: int
    logits: tuple[float, float, float]
    probabilities: tuple[float, float, float]
    contribution_count: int


@dataclass(frozen=True, slots=True)
class SentenceDecodingInput:
    id: int
    ordinal: int
    start_char: int
    end_char: int
    section_path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DecodedSentence:
    id: int
    ordinal: int
    raw_label: Literal["O", "B", "I"]
    decoded_label: Literal["O", "B", "I"]
    probabilities: tuple[float, float, float]
    contribution_count: int
    entropy: float


@dataclass(frozen=True, slots=True)
class DecodedBlock:
    start_sentence_id: int
    end_sentence_id: int
    start_ordinal: int
    end_ordinal: int
    start_char: int
    end_char: int
    confidence: float
    start_confidence: float
    end_confidence: float
    uncertainty: float
    sentence_ordinals: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DecoderResult:
    blocks: tuple[DecodedBlock, ...]
    suppressed_blocks: tuple[DecodedBlock, ...]
    sentences: tuple[DecodedSentence, ...]
    decoder_version: str = "evidence-block-decoder-v1"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    block_threshold: float = 0.5
    allow_cross_section: bool = False
    merge_adjacent: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.block_threshold <= 1:
            raise DecoderError("block_threshold must be between zero and one")


def aggregate_window_logits(
    windows: Iterable[Mapping[int, Sequence[float]]],
    *,
    method: Literal["mean"] = "mean",
) -> dict[int, AggregatedSentenceLogits]:
    if method != "mean":
        raise DecoderError(f"Unsupported overlap aggregation method '{method}'")
    values: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for window in windows:
        for ordinal, logits in window.items():
            if len(logits) != 3:
                raise DecoderError("Each sentence must have exactly O/B/I logits")
            converted = tuple(float(value) for value in logits)
            if not all(math.isfinite(value) for value in converted):
                raise DecoderError("Sentence logits must be finite")
            values[int(ordinal)].append(converted)
    aggregated: dict[int, AggregatedSentenceLogits] = {}
    for ordinal, contributions in values.items():
        mean = tuple(
            sum(item[index] for item in contributions) / len(contributions)
            for index in range(3)
        )
        aggregated[ordinal] = AggregatedSentenceLogits(
            ordinal=ordinal,
            logits=mean,
            probabilities=_softmax(mean),
            contribution_count=len(contributions),
        )
    return aggregated


def decode_evidence_blocks(
    sentences: Sequence[SentenceDecodingInput],
    aggregated_logits: Mapping[int, AggregatedSentenceLogits],
    config: DecoderConfig | None = None,
) -> DecoderResult:
    selected_config = config or DecoderConfig()
    ordered = sorted(sentences, key=lambda item: item.ordinal)
    if len({item.ordinal for item in ordered}) != len(ordered):
        raise DecoderError("Sentence ordinals must be unique")

    decoded_sentences: list[DecodedSentence] = []
    previous_label: Literal["O", "B", "I"] = "O"
    previous_section: tuple[str, ...] | None = None
    for sentence in ordered:
        aggregated = aggregated_logits.get(sentence.ordinal)
        if aggregated is None:
            raise DecoderError(f"Missing aggregated logits for sentence {sentence.ordinal}")
        raw_index = max(range(3), key=lambda index: aggregated.probabilities[index])
        raw_label = LABELS[raw_index]
        section_break = (
            previous_section is not None and sentence.section_path != previous_section
        )
        effective_previous = (
            "O" if section_break and not selected_config.allow_cross_section else previous_label
        )
        decoded_label = "B" if raw_label == "I" and effective_previous == "O" else raw_label
        decoded_sentences.append(
            DecodedSentence(
                id=sentence.id,
                ordinal=sentence.ordinal,
                raw_label=raw_label,
                decoded_label=decoded_label,
                probabilities=aggregated.probabilities,
                contribution_count=aggregated.contribution_count,
                entropy=_entropy(aggregated.probabilities),
            )
        )
        previous_label = decoded_label
        previous_section = sentence.section_path

    index_by_ordinal = {item.ordinal: index for index, item in enumerate(ordered)}
    decoded_by_ordinal = {item.ordinal: item for item in decoded_sentences}
    candidates: list[DecodedBlock] = []
    active: list[DecodedSentence] = []
    for decoded in decoded_sentences:
        if decoded.decoded_label == "O":
            if active:
                candidates.append(
                    _make_block(active, ordered, index_by_ordinal, decoded_by_ordinal)
                )
                active = []
        elif decoded.decoded_label == "B":
            if active:
                candidates.append(
                    _make_block(active, ordered, index_by_ordinal, decoded_by_ordinal)
                )
            active = [decoded]
        else:
            if not active:  # Defensive: the normalization above should prevent this.
                active = [decoded]
            else:
                active.append(decoded)
    if active:
        candidates.append(_make_block(active, ordered, index_by_ordinal, decoded_by_ordinal))

    if selected_config.merge_adjacent:
        candidates = _merge_adjacent_blocks(candidates)
    emitted = tuple(
        block for block in candidates if block.confidence >= selected_config.block_threshold
    )
    suppressed = tuple(
        block for block in candidates if block.confidence < selected_config.block_threshold
    )
    return DecoderResult(
        blocks=emitted,
        suppressed_blocks=suppressed,
        sentences=tuple(decoded_sentences),
    )


def _make_block(
    active: Sequence[DecodedSentence],
    source_sentences: Sequence[SentenceDecodingInput],
    index_by_ordinal: Mapping[int, int],
    decoded_by_ordinal: Mapping[int, DecodedSentence],
) -> DecodedBlock:
    first = active[0]
    last = active[-1]
    first_source = source_sentences[index_by_ordinal[first.ordinal]]
    last_source = source_sentences[index_by_ordinal[last.ordinal]]
    class_probabilities = [
        _decoded_probability(sentence, "B" if index == 0 else "I")
        for index, sentence in enumerate(active)
    ]
    next_index = index_by_ordinal[last.ordinal] + 1
    if next_index < len(source_sentences):
        next_ordinal = source_sentences[next_index].ordinal
        end_confidence = 1.0 - decoded_by_ordinal[next_ordinal].probabilities[2]
    else:
        end_confidence = 1.0
    return DecodedBlock(
        start_sentence_id=first_source.id,
        end_sentence_id=last_source.id,
        start_ordinal=first.ordinal,
        end_ordinal=last.ordinal,
        start_char=first_source.start_char,
        end_char=last_source.end_char,
        confidence=sum(class_probabilities) / len(class_probabilities),
        start_confidence=_decoded_probability(first, "B"),
        end_confidence=end_confidence,
        uncertainty=sum(item.entropy for item in active) / len(active),
        sentence_ordinals=tuple(item.ordinal for item in active),
    )


def _decoded_probability(
    sentence: DecodedSentence,
    label: Literal["B", "I"],
) -> float:
    if label == "B" and sentence.raw_label == "I" and sentence.decoded_label == "B":
        # The constrained decoder collapses an illegal I transition into B;
        # preserve the probability mass of both valid onset outcomes.
        return sentence.probabilities[1] + sentence.probabilities[2]
    return sentence.probabilities[1 if label == "B" else 2]


def _merge_adjacent_blocks(blocks: Sequence[DecodedBlock]) -> list[DecodedBlock]:
    if not blocks:
        return []
    merged = [blocks[0]]
    for block in blocks[1:]:
        previous = merged[-1]
        if previous.end_ordinal + 1 != block.start_ordinal:
            merged.append(block)
            continue
        previous_length = len(previous.sentence_ordinals)
        block_length = len(block.sentence_ordinals)
        total = previous_length + block_length
        merged[-1] = DecodedBlock(
            start_sentence_id=previous.start_sentence_id,
            end_sentence_id=block.end_sentence_id,
            start_ordinal=previous.start_ordinal,
            end_ordinal=block.end_ordinal,
            start_char=previous.start_char,
            end_char=block.end_char,
            confidence=(
                previous.confidence * previous_length + block.confidence * block_length
            )
            / total,
            start_confidence=previous.start_confidence,
            end_confidence=block.end_confidence,
            uncertainty=(
                previous.uncertainty * previous_length + block.uncertainty * block_length
            )
            / total,
            sentence_ordinals=previous.sentence_ordinals + block.sentence_ordinals,
        )
    return merged


def _softmax(logits: Sequence[float]) -> tuple[float, float, float]:
    maximum = max(logits)
    exponents = [math.exp(value - maximum) for value in logits]
    total = sum(exponents)
    return tuple(value / total for value in exponents)


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)
