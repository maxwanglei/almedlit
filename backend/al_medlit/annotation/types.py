"""Registry of supported annotation structures."""

from dataclasses import dataclass
from typing import Literal

AnnotationType = Literal[
    "entity",
    "relation",
    "doc_label",
    "sentence_label",
    "passage_label",
    "evidence_block",
]

AnnotationSource = Literal["human", "model", "llm"]

AnnotationStatus = Literal["draft", "accepted", "rejected", "gold"]

CorrectionSource = Literal["human", "adjudication", "model"]

CorrectionSeverity = Literal["low", "medium", "high", "critical"]


@dataclass(frozen=True)
class AnnotationTypeSpec:
    name: AnnotationType
    requires_span: bool
    requires_head_tail: bool
    description: str
    selection_mode: str = "none"
    renderer_key: str = "legacy"
    relation_endpoint_allowed: bool = False
    handler_key: str = "generic"


REGISTRY: dict[str, AnnotationTypeSpec] = {
    "entity": AnnotationTypeSpec(
        name="entity",
        requires_span=True,
        requires_head_tail=False,
        description="Free-form character span with a single label.",
        selection_mode="character_span",
        relation_endpoint_allowed=True,
    ),
    "relation": AnnotationTypeSpec(
        name="relation",
        requires_span=False,
        requires_head_tail=True,
        description="Directed link from a head annotation to a tail annotation.",
        selection_mode="relation",
    ),
    "doc_label": AnnotationTypeSpec(
        name="doc_label",
        requires_span=False,
        requires_head_tail=False,
        description="A label on the whole document.",
    ),
    "sentence_label": AnnotationTypeSpec(
        name="sentence_label",
        requires_span=True,
        requires_head_tail=False,
        description="A label on a sentence span.",
        selection_mode="character_span",
        relation_endpoint_allowed=True,
    ),
    "passage_label": AnnotationTypeSpec(
        name="passage_label",
        requires_span=True,
        requires_head_tail=False,
        description="A label on an arbitrary multi-sentence span.",
        selection_mode="character_span",
        relation_endpoint_allowed=True,
    ),
    "evidence_block": AnnotationTypeSpec(
        name="evidence_block",
        requires_span=False,
        requires_head_tail=False,
        description="A target-conditioned, sentence-aligned evidence block.",
        selection_mode="sentence_range",
        renderer_key="evidence_block_v1",
        relation_endpoint_allowed=False,
        handler_key="evidence_block_v1",
    ),
}


class AnnotationValidationError(ValueError):
    pass


def validate_create(values: dict) -> None:
    annotation_type = values.get("annotation_type")
    spec = REGISTRY.get(annotation_type)
    if spec is None:
        raise AnnotationValidationError(
            f"Unknown annotation_type '{annotation_type}'. Known: {sorted(REGISTRY)}"
        )

    start_offset = values.get("start_offset")
    end_offset = values.get("end_offset")
    if spec.requires_span and (start_offset is None or end_offset is None):
        raise AnnotationValidationError(
            f"{spec.name} requires start_offset and end_offset"
        )

    head_annotation_id = values.get("head_annotation_id")
    tail_annotation_id = values.get("tail_annotation_id")
    if spec.requires_head_tail:
        if head_annotation_id is None or tail_annotation_id is None:
            raise AnnotationValidationError(
                f"{spec.name} requires head_annotation_id and tail_annotation_id"
            )
    elif head_annotation_id is not None or tail_annotation_id is not None:
        raise AnnotationValidationError(
            f"{spec.name} must not set head_annotation_id or tail_annotation_id"
        )
