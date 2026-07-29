"""Deterministic word preparation and safe JSON vocabularies."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from al_medlit.training.model_types.evidence_neural.config import EvidenceNeuralConfig
from al_medlit.training.model_types.evidence_neural.data import EvidenceTextDocument

PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"
TARGET_TOKEN = "<TARGET>"
SEPARATOR_TOKEN = "<SEP>"
RESERVED_TOKENS = (PAD_TOKEN, UNKNOWN_TOKEN, TARGET_TOKEN, SEPARATOR_TOKEN)


class VocabularyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["evidence-neural-vocabulary-v1"] = "evidence-neural-vocabulary-v1"
    tokens: tuple[str, ...] = Field(min_length=len(RESERVED_TOKENS), max_length=2_000_000)
    tokenizer_pattern: str
    lowercase: bool
    unicode_normalization: Literal["NFC", "NFKC"]
    target_conditioning: bool

    @model_validator(mode="after")
    def safe_ordered_tokens(self):
        if self.tokens[: len(RESERVED_TOKENS)] != RESERVED_TOKENS:
            raise ValueError("Vocabulary reserved tokens are missing or reordered")
        if len(self.tokens) != len(set(self.tokens)):
            raise ValueError("Vocabulary tokens must be unique")
        if any(not token or "\x00" in token for token in self.tokens):
            raise ValueError("Vocabulary contains an unsafe token")
        try:
            re.compile(self.tokenizer_pattern)
        except re.error as exc:
            raise ValueError("Vocabulary tokenizer_pattern is invalid") from exc
        return self


@dataclass(frozen=True, slots=True)
class EvidenceVocabulary:
    payload: VocabularyPayload

    @property
    def tokens(self) -> tuple[str, ...]:
        return self.payload.tokens

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unknown_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.payload.tokens)

    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.payload.tokens)}

    def tokenize(self, text: str) -> tuple[str, ...]:
        normalized = unicodedata.normalize(self.payload.unicode_normalization, text)
        if self.payload.lowercase:
            normalized = normalized.lower()
        return tuple(re.findall(self.payload.tokenizer_pattern, normalized))

    def encode_sentence(
        self,
        *,
        target_text: str,
        sentence: str,
        max_tokens: int,
    ) -> tuple[int, ...]:
        token_ids = self.token_to_id()
        sentence_tokens = list(self.tokenize(sentence))
        if self.payload.target_conditioning:
            target_tokens = list(self.tokenize(target_text))
            # Always reserve a position for the sentence when it is non-empty.
            target_budget = max(0, max_tokens - 3)
            tokens = [TARGET_TOKEN, *target_tokens[:target_budget], SEPARATOR_TOKEN]
            tokens.extend(sentence_tokens[: max(0, max_tokens - len(tokens))])
        else:
            tokens = sentence_tokens[:max_tokens]
        if not tokens:
            tokens = [UNKNOWN_TOKEN]
        return tuple(token_ids.get(token, self.unknown_id) for token in tokens[:max_tokens])

    def save(self, destination: str | Path) -> Path:
        path = Path(destination)
        path.write_text(
            json.dumps(self.payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, source: str | Path) -> EvidenceVocabulary:
        path = Path(source)
        if path.is_symlink() or not path.is_file():
            raise ValueError("Vocabulary must be a regular, non-symlink JSON file")
        if path.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Vocabulary JSON exceeds the 128 MiB safety limit")
        return cls(VocabularyPayload.model_validate_json(path.read_text(encoding="utf-8")))


def build_vocabulary(
    documents: Iterable[EvidenceTextDocument],
    config: EvidenceNeuralConfig,
) -> EvidenceVocabulary:
    tokenizer_payload = VocabularyPayload(
        tokens=RESERVED_TOKENS,
        tokenizer_pattern=config.tokenizer_pattern,
        lowercase=config.lowercase,
        unicode_normalization=config.unicode_normalization,
        target_conditioning=config.target_conditioning,
    )
    tokenizer = EvidenceVocabulary(tokenizer_payload)
    counts: Counter[str] = Counter()
    for document in documents:
        if len(document.sentences) > config.max_sentences_per_document:
            raise ValueError(
                "Prepared neural inputs must be windowed; sentences are never silently truncated"
            )
        if config.target_conditioning:
            counts.update(tokenizer.tokenize(document.target_text))
        for sentence in document.sentences:
            counts.update(tokenizer.tokenize(sentence))
    candidates = (token for token, count in counts.items() if count >= config.min_token_frequency)
    ordered = sorted(candidates, key=lambda token: (-counts[token], token))
    capacity = config.max_vocabulary_size - len(RESERVED_TOKENS)
    return EvidenceVocabulary(
        tokenizer_payload.model_copy(update={"tokens": RESERVED_TOKENS + tuple(ordered[:capacity])})
    )
