"""Turn raw annotations into the aligned rating matrix the metrics consume.

The output is intentionally framework-free so it can be unit-tested with simple
objects: any object exposing ``annotator_id``, ``document_id``, ``start_offset``,
``end_offset``, and ``label`` attributes works.
"""

NONE_LABEL = "__none__"
MULTI_LABEL_PREFIX = "__multi_label__:"

type SpanKey = tuple[int, int | None, int | None]


def _rating_label(labels: set[str]) -> str:
    if len(labels) == 1:
        return next(iter(labels))
    return f"{MULTI_LABEL_PREFIX}{'|'.join(sorted(labels))}"


def build_rating_matrix(annotations):
    """Build the aligned rating matrix for one annotation type in one scope.

    Returns a tuple ``(annotators, items, spans_by_annotator)`` where:

    - ``annotators`` is the sorted list of distinct non-empty annotator ids.
    - ``items`` is a list of dicts, one per ``(document_id, start, end)`` key,
      each mapping every annotator id to its label or ``NONE_LABEL`` if absent.
    - ``spans_by_annotator`` maps annotator id to the set of span keys it marked.
    """
    annotators = sorted(
        {annotation.annotator_id for annotation in annotations if annotation.annotator_id}
    )
    by_item: dict[SpanKey, dict[str, set[str]]] = {}
    spans_by_annotator: dict[str, set[SpanKey]] = {annotator: set() for annotator in annotators}

    for annotation in annotations:
        annotator = annotation.annotator_id
        if not annotator:
            continue

        key = (annotation.document_id, annotation.start_offset, annotation.end_offset)
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
