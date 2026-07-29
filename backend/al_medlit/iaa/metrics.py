"""Pure inter-annotator agreement metric functions.

Each function operates on plain Python structures so it can be unit-tested
without a database or web framework. An "item" is a dict mapping every
annotator id to the category (label) they assigned; absence is represented by
the sentinel category produced in alignment.py.
"""

import itertools


def percent_agreement(items, annotators):
    """Average pairwise observed agreement across all items and annotator pairs."""
    pairs = list(itertools.combinations(annotators, 2))
    if not items or not pairs:
        return None

    total = 0
    agree = 0
    for row in items:
        for first, second in pairs:
            total += 1
            if row[first] == row[second]:
                agree += 1
    return agree / total if total else None


def cohens_kappa(items, annotator_a, annotator_b):
    """Cohen's kappa between exactly two annotators over the given items."""
    count = len(items)
    if count == 0:
        return None

    labels_a = [row[annotator_a] for row in items]
    labels_b = [row[annotator_b] for row in items]
    categories = set(labels_a) | set(labels_b)
    observed = sum(1 for x, y in zip(labels_a, labels_b, strict=True) if x == y) / count
    expected = 0.0
    for category in categories:
        prop_a = labels_a.count(category) / count
        prop_b = labels_b.count(category) / count
        expected += prop_a * prop_b

    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def fleiss_kappa(items, annotators):
    """Fleiss' kappa over items rated by a fixed set of annotators (n >= 2).

    Every item must contain a category for every annotator (the alignment step
    guarantees this by filling absences with the sentinel category).
    """
    n_raters = len(annotators)
    n_items = len(items)
    if n_items == 0 or n_raters < 2:
        return None

    categories = sorted({label for row in items for label in row.values()})
    category_totals = dict.fromkeys(categories, 0)
    agreement_per_item = []
    for row in items:
        counts = dict.fromkeys(categories, 0)
        for annotator in annotators:
            counts[row[annotator]] += 1
        for category in categories:
            category_totals[category] += counts[category]
        sum_squares = sum(value * value for value in counts.values())
        agreement_per_item.append((sum_squares - n_raters) / (n_raters * (n_raters - 1)))

    mean_agreement = sum(agreement_per_item) / n_items
    category_props = {
        category: category_totals[category] / (n_items * n_raters) for category in categories
    }
    expected = sum(value * value for value in category_props.values())
    if expected >= 1.0:
        return 1.0 if mean_agreement >= 1.0 else 0.0
    return (mean_agreement - expected) / (1 - expected)


def pairwise_span_f1(spans_by_annotator, annotators):
    """Average label-agnostic exact-match F1 over all annotator pairs.

    ``spans_by_annotator`` maps annotator id to a set of ``(document_id, start, end)`` keys.
    """
    pairs = list(itertools.combinations(annotators, 2))
    if not pairs:
        return None

    scores = []
    for first, second in pairs:
        spans_first = spans_by_annotator.get(first, set())
        spans_second = spans_by_annotator.get(second, set())
        denom = len(spans_first) + len(spans_second)
        if denom == 0:
            scores.append(1.0)
            continue
        matched = len(spans_first & spans_second)
        scores.append(2 * matched / denom)
    return sum(scores) / len(scores)
