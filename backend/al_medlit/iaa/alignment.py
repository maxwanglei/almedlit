"""Turn raw annotations into the aligned rating matrix the metrics consume.

The output is intentionally framework-free so it can be unit-tested with simple
objects: any object exposing ``annotator_id``, ``document_id``, ``start_offset``,
``end_offset``, and ``label`` attributes works. Relations additionally expose
``annotation_type``, ``head``, and ``tail``, with span attributes on both endpoints.
"""

NONE_LABEL = "__none__"
MULTI_LABEL_PREFIX = "__multi_label__:"

type SpanKey = tuple[int, int | None, int | None]
type RelationKey = tuple[str, SpanKey, SpanKey]
type ItemKey = SpanKey | RelationKey


def _rating_label(labels: set[str]) -> str:
    if len(labels) == 1:
        return next(iter(labels))
    return f"{MULTI_LABEL_PREFIX}{'|'.join(sorted(labels))}"


def _item_key(annotation) -> ItemKey:
    if getattr(annotation, "annotation_type", None) != "relation":
        return (annotation.document_id, annotation.start_offset, annotation.end_offset)

    endpoints = []
    for side in ("head", "tail"):
        endpoint = getattr(annotation, side, None)
        if endpoint is None:
            raise ValueError("IAA requires each relation endpoint to have a valid span")
        start = endpoint.start_offset
        end = endpoint.end_offset
        if (
            endpoint.document_id != annotation.document_id
            or start is None
            or end is None
            or start < 0
            or end <= start
        ):
            raise ValueError(
                "IAA requires each relation endpoint to have a valid span in the same document"
            )
        endpoints.append((endpoint.document_id, start, end))
    return ("relation", endpoints[0], endpoints[1])


def build_rating_matrix(annotations):
    """Build the aligned rating matrix for one annotation type in one scope.

    Returns a tuple ``(annotators, items, spans_by_annotator)`` where:

    - ``annotators`` is the sorted list of distinct non-empty annotator ids.
    - ``items`` is a list of dicts, one per document/span or directed relation
      endpoint pair, mapping every annotator id to its label or ``NONE_LABEL``.
    - ``spans_by_annotator`` maps annotator id to the set of item keys it marked.

    Relation endpoints align by document and offsets rather than annotation ids.
    Raises ``ValueError`` if a relation endpoint cannot identify a valid span.
    """
    annotators = sorted(
        {annotation.annotator_id for annotation in annotations if annotation.annotator_id}
    )
    by_item: dict[ItemKey, dict[str, set[str]]] = {}
    spans_by_annotator: dict[str, set[ItemKey]] = {annotator: set() for annotator in annotators}

    for annotation in annotations:
        annotator = annotation.annotator_id
        if not annotator:
            continue

        key = _item_key(annotation)
        slot = by_item.setdefault(key, {})
        slot.setdefault(annotator, set()).add(annotation.label)
        spans_by_annotator[annotator].add(key)

    items = [
        {
            annotator: _rating_label(slot[annotator])
            if annotator in slot
            else NONE_LABEL
            for annotator in annotators
        }
        for slot in by_item.values()
    ]
    return annotators, items, spans_by_annotator
