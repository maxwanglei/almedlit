from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from al_medlit.training.windowing import SentenceLabel

IGNORE_INDEX = -100
LABEL_TO_ID = {"O": 0, "B": 1, "I": 2, "IGNORE": IGNORE_INDEX}


class TokenizerLike(Protocol):
    cls_token_id: int
    sep_token_id: int

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


@dataclass(frozen=True, slots=True)
class MarkerEncodedInput:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    sentence_marker_positions: tuple[int, ...]


def encode_sentence_markers(
    tokenizer: TokenizerLike,
    *,
    target_text: str,
    sentences: Sequence[str],
    target_marker_token: str = "[TARGET]",
    sentence_marker_token: str = "[SENT]",
    target_conditioning: bool = True,
    max_length: int = 4096,
) -> MarkerEncodedInput:
    target_marker_id = tokenizer.convert_tokens_to_ids(target_marker_token)
    sentence_marker_id = tokenizer.convert_tokens_to_ids(sentence_marker_token)
    input_ids = [tokenizer.cls_token_id]
    if target_conditioning:
        input_ids.append(target_marker_id)
        input_ids.extend(tokenizer.encode(target_text, add_special_tokens=False))
        input_ids.append(tokenizer.sep_token_id)
    marker_positions: list[int] = []
    for sentence in sentences:
        marker_positions.append(len(input_ids))
        input_ids.append(sentence_marker_id)
        input_ids.extend(tokenizer.encode(sentence, add_special_tokens=False))
    input_ids.append(tokenizer.sep_token_id)
    if len(input_ids) > max_length:
        raise ValueError(
            f"Encoded window has {len(input_ids)} tokens, exceeding max_length={max_length}; "
            "complete sentences are never silently truncated"
        )
    return MarkerEncodedInput(
        input_ids=tuple(input_ids),
        attention_mask=(1,) * len(input_ids),
        sentence_marker_positions=tuple(marker_positions),
    )


def encode_labels(labels: Sequence[SentenceLabel]) -> tuple[int, ...]:
    return tuple(LABEL_TO_ID[label] for label in labels)


def balanced_class_weights(label_sequences: Sequence[Sequence[int]]) -> tuple[float, float, float]:
    counts = Counter(
        label
        for sequence in label_sequences
        for label in sequence
        if label != IGNORE_INDEX
    )
    if any(counts[label] == 0 for label in range(3)):
        missing = [label for label in range(3) if counts[label] == 0]
        raise ValueError(f"Cannot balance classes absent from the dataset: {missing}")
    total = sum(counts.values())
    return tuple(total / (3 * counts[label]) for label in range(3))
