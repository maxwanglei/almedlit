"""Deterministic document structure and sentence segmentation.

Stored offsets are Python string offsets: zero-based Unicode code-point indices
with an inclusive start and exclusive end.  The implementation deliberately
does not depend on an external NLP model so a structure can be reproduced on a
worker, in a migration backfill, or in an offline test environment.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

SEGMENTER_NAME = "builtin"
SEGMENTER_VERSION = "1.0.0"

_BLANK_LINES = re.compile(r"(?:\r?\n[ \t]*){2,}")
_CLOSING_PUNCTUATION = frozenset('"\'\u2019\u201d)]}')
_NON_TERMINAL_ABBREVIATIONS = frozenset(
    {
        "al.",
        "approx.",
        "dr.",
        "e.g.",
        "fig.",
        "i.e.",
        "mr.",
        "mrs.",
        "ms.",
        "no.",
        "prof.",
        "ref.",
        "st.",
        "vs.",
    }
)
_INITIALISM = re.compile(r"(?:[A-Za-z]\.){2,}$")
_SINGLE_INITIAL = re.compile(r"(?:^|\s)[A-Z]\.$")


@dataclass(frozen=True, slots=True)
class SegmentedSentence:
    ordinal: int
    paragraph_ordinal: int
    start_offset: int
    end_offset: int
    text_hash: str


@dataclass(frozen=True, slots=True)
class SegmentedParagraph:
    ordinal: int
    section_ordinal: int
    start_offset: int
    end_offset: int
    locator: dict[str, Any]
    sentences: tuple[SegmentedSentence, ...]


@dataclass(frozen=True, slots=True)
class SegmentedSection:
    ordinal: int
    title: str | None
    path: tuple[str, ...]
    kind: str
    start_offset: int
    end_offset: int
    locator: dict[str, Any]
    paragraphs: tuple[SegmentedParagraph, ...]


@dataclass(frozen=True, slots=True)
class SegmentedDocument:
    sections: tuple[SegmentedSection, ...]
    paragraph_count: int
    sentence_count: int


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Split on blank lines while retaining offsets into the original text."""
    if not text:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BLANK_LINES.finditer(text):
        trimmed = _trim_span(text, start, match.start())
        if trimmed[0] < trimmed[1]:
            spans.append(trimmed)
        start = match.end()
    trimmed = _trim_span(text, start, len(text))
    if trimmed[0] < trimmed[1]:
        spans.append(trimmed)
    return spans


def _period_continues_sentence(text: str, index: int, paragraph_start: int) -> bool:
    if index > paragraph_start and index + 1 < len(text):
        if text[index - 1].isdigit() and text[index + 1].isdigit():
            return True

    token_start = index
    while token_start > paragraph_start and not text[token_start - 1].isspace():
        token_start -= 1
    token = text[token_start : index + 1].lower().strip('"\'([{')
    if token in _NON_TERMINAL_ABBREVIATIONS:
        return True

    original_token = text[token_start : index + 1].strip('"\'([{')
    return bool(
        _INITIALISM.search(original_token) or _SINGLE_INITIAL.search(original_token)
    )


def _sentence_spans_in_range(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    sentence_start = start
    index = start

    while index < end:
        character = text[index]
        if character not in ".!?":
            index += 1
            continue

        punctuation_end = index + 1
        while punctuation_end < end and text[punctuation_end] in ".!?":
            punctuation_end += 1
        while (
            punctuation_end < end
            and text[punctuation_end] in _CLOSING_PUNCTUATION
        ):
            punctuation_end += 1

        followed_by_boundary = punctuation_end == end or text[punctuation_end].isspace()
        if not followed_by_boundary:
            index = punctuation_end
            continue
        if character == "." and _period_continues_sentence(text, index, start):
            index = punctuation_end
            continue

        sentence_end = punctuation_end
        candidate = _trim_span(text, sentence_start, sentence_end)
        if candidate[0] < candidate[1]:
            spans.append(candidate)
        sentence_start = punctuation_end
        while sentence_start < end and text[sentence_start].isspace():
            sentence_start += 1
        index = sentence_start

    candidate = _trim_span(text, sentence_start, end)
    if candidate[0] < candidate[1]:
        spans.append(candidate)
    return spans


def segment_sentences(text: str) -> list[tuple[int, int]]:
    """Return sentence spans, respecting blank-line paragraph boundaries."""
    spans: list[tuple[int, int]] = []
    for start, end in paragraph_spans(text):
        spans.extend(_sentence_spans_in_range(text, start, end))
    return spans


def _valid_jats_blocks(
    text: str,
    source_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not source_metadata or source_metadata.get("format") != "jats-v1":
        return []
    raw_blocks = source_metadata.get("blocks")
    if not isinstance(raw_blocks, list):
        return []

    blocks: list[dict[str, Any]] = []
    previous_end = -1
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            return []
        start = raw.get("start_offset")
        end = raw.get("end_offset")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or start >= end
            or end > len(text)
            or start < previous_end
        ):
            return []
        expected_hash = raw.get("text_hash")
        if expected_hash and expected_hash != text_sha256(text[start:end]):
            return []
        blocks.append(raw)
        previous_end = end
    return blocks


def _plain_paragraph_specs(text: str) -> list[dict[str, Any]]:
    return [
        {
            "start_offset": start,
            "end_offset": end,
            "section_path": [],
            "section_title": None,
            "section_kind": "unknown",
            "locator": {"format": "plain", "paragraph": ordinal},
        }
        for ordinal, (start, end) in enumerate(paragraph_spans(text))
    ]


def segment_document(
    text: str,
    *,
    source_metadata: dict[str, Any] | None = None,
) -> SegmentedDocument:
    """Create deterministic section/paragraph/sentence plans for ``text``.

    Valid ``jats-v1`` source metadata preserves imported section paths and
    locators. Invalid or absent metadata safely falls back to one unknown
    section, rather than inventing section headings from prose.
    """
    paragraph_specs = _valid_jats_blocks(text, source_metadata)
    is_jats = bool(paragraph_specs)
    if not paragraph_specs:
        paragraph_specs = _plain_paragraph_specs(text)

    grouped_specs: list[list[dict[str, Any]]] = []
    for spec in paragraph_specs:
        raw_path = spec.get("section_path", [])
        path = tuple(str(item) for item in raw_path if str(item).strip())
        spec = {**spec, "_normalized_path": path}
        if not grouped_specs:
            grouped_specs.append([spec])
            continue
        previous = grouped_specs[-1][-1]
        if (
            previous["_normalized_path"] == path
            and previous.get("section_kind", "jats" if is_jats else "unknown")
            == spec.get("section_kind", "jats" if is_jats else "unknown")
        ):
            grouped_specs[-1].append(spec)
        else:
            grouped_specs.append([spec])

    # Empty documents still have a stable unknown section, which makes the API
    # shape predictable without fabricating any paragraph or sentence offsets.
    if not grouped_specs:
        return SegmentedDocument(
            sections=(
                SegmentedSection(
                    ordinal=0,
                    title=None,
                    path=(),
                    kind="unknown",
                    start_offset=0,
                    end_offset=0,
                    locator={"format": "plain"},
                    paragraphs=(),
                ),
            ),
            paragraph_count=0,
            sentence_count=0,
        )

    section_results: list[SegmentedSection] = []
    paragraph_ordinal = 0
    sentence_ordinal = 0
    for section_ordinal, specs in enumerate(grouped_specs):
        paragraphs: list[SegmentedParagraph] = []
        for within_section_ordinal, spec in enumerate(specs):
            start = int(spec["start_offset"])
            end = int(spec["end_offset"])
            sentences: list[SegmentedSentence] = []
            for within_paragraph_ordinal, (sentence_start, sentence_end) in enumerate(
                _sentence_spans_in_range(text, start, end)
            ):
                sentences.append(
                    SegmentedSentence(
                        ordinal=sentence_ordinal,
                        paragraph_ordinal=within_paragraph_ordinal,
                        start_offset=sentence_start,
                        end_offset=sentence_end,
                        text_hash=text_sha256(text[sentence_start:sentence_end]),
                    )
                )
                sentence_ordinal += 1

            locator = spec.get("locator")
            paragraphs.append(
                SegmentedParagraph(
                    ordinal=paragraph_ordinal,
                    section_ordinal=within_section_ordinal,
                    start_offset=start,
                    end_offset=end,
                    locator=dict(locator) if isinstance(locator, dict) else {},
                    sentences=tuple(sentences),
                )
            )
            paragraph_ordinal += 1

        first = specs[0]
        path = first["_normalized_path"]
        raw_title = first.get("section_title")
        title = str(raw_title).strip() if raw_title else (path[-1] if path else None)
        first_locator = first.get("locator")
        section_results.append(
            SegmentedSection(
                ordinal=section_ordinal,
                title=title or None,
                path=path,
                kind=str(first.get("section_kind") or ("jats" if is_jats else "unknown")),
                start_offset=paragraphs[0].start_offset,
                end_offset=paragraphs[-1].end_offset,
                locator={
                    "format": "jats" if is_jats else "plain",
                    "first_block": (
                        dict(first_locator) if isinstance(first_locator, dict) else {}
                    ),
                },
                paragraphs=tuple(paragraphs),
            )
        )

    return SegmentedDocument(
        sections=tuple(section_results),
        paragraph_count=paragraph_ordinal,
        sentence_count=sentence_ordinal,
    )
