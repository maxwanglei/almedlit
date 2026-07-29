from __future__ import annotations

type Span = tuple[int, int]


def span_iou(left: Span, right: Span) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]) + 1)
    if intersection == 0:
        return 0.0
    union = max(left[1], right[1]) - min(left[0], right[0]) + 1
    return intersection / union


def maximum_weight_matching(
    left: list[Span], right: list[Span]
) -> list[tuple[int, int, float]]:
    """Return a maximum-total-IoU one-to-one matching via Hungarian assignment."""
    if not left or not right:
        return []
    size = max(len(left), len(right))
    weights = [
        [span_iou(left[i], right[j]) if i < len(left) and j < len(right) else 0.0
         for j in range(size)]
        for i in range(size)
    ]
    max_weight = max(max(row) for row in weights)
    costs = [[max_weight - weight for weight in row] for row in weights]

    # Classic O(n^3) Hungarian algorithm for a square, one-indexed cost matrix.
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        min_value = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                if current < min_value[j]:
                    min_value[j] = current
                    way[j] = j0
                if min_value[j] < delta:
                    delta = min_value[j]
                    j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    min_value[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = []
    for right_index in range(1, size + 1):
        left_index = p[right_index]
        if left_index == 0:
            continue
        i = left_index - 1
        j = right_index - 1
        if i < len(left) and j < len(right):
            result.append((i, j, weights[i][j]))
    return result


def _prf(true_positive: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else (1.0 if reference == 0 else 0.0)
    recall = true_positive / reference if reference else (1.0 if predicted == 0 else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def pair_metrics(
    left: list[Span],
    right: list[Span],
    *,
    reviewed_sentence_count: int,
) -> dict:
    left_sentences = {
        ordinal for start, end in left for ordinal in range(start, end + 1)
    }
    right_sentences = {
        ordinal for start, end in right for ordinal in range(start, end + 1)
    }
    sentence_tp = len(left_sentences & right_sentences)
    sentence_precision, sentence_recall, sentence_f1 = _prf(
        sentence_tp, len(right_sentences), len(left_sentences)
    )
    matching = maximum_weight_matching(left, right)
    exact_matches = sum(1 for _i, _j, iou in matching if iou == 1.0)
    _exact_p, _exact_r, exact_f1 = _prf(exact_matches, len(right), len(left))
    iou_f1 = {}
    for threshold in (0.25, 0.5, 0.75):
        matches = sum(1 for _i, _j, iou in matching if iou >= threshold)
        _p, _r, score = _prf(matches, len(right), len(left))
        iou_f1[f"{threshold:.2f}"] = score
    positive_union = left_sentences | right_sentences
    positive_intersection = left_sentences & right_sentences
    coverage = (
        len(positive_intersection) / len(positive_union) if positive_union else 1.0
    )
    overreach = (
        len(right_sentences - left_sentences) / len(right_sentences)
        if right_sentences
        else 0.0
    )
    positive_matches = [(i, j) for i, j, iou in matching if iou > 0]
    if positive_matches:
        start_deviation = sum(abs(left[i][0] - right[j][0]) for i, j in positive_matches) / len(
            positive_matches
        )
        end_deviation = sum(abs(left[i][1] - right[j][1]) for i, j in positive_matches) / len(
            positive_matches
        )
    else:
        start_deviation = None
        end_deviation = None
    return {
        "reviewed_sentence_count": reviewed_sentence_count,
        "left_block_count": len(left),
        "right_block_count": len(right),
        "sentence_precision": sentence_precision,
        "sentence_recall": sentence_recall,
        "sentence_f1": sentence_f1,
        "exact_f1": exact_f1,
        "iou_f1": iou_f1,
        "coverage": coverage,
        "overreach": overreach,
        "mean_start_boundary_deviation": start_deviation,
        "mean_end_boundary_deviation": end_deviation,
        "document_presence_agreement": bool(left) == bool(right),
    }


def average_pair_metrics(pair_reports: list[dict]) -> dict:
    if not pair_reports:
        return {}
    numeric_keys = (
        "sentence_precision",
        "sentence_recall",
        "sentence_f1",
        "exact_f1",
        "coverage",
        "overreach",
    )
    result = {
        key: sum(report[key] for report in pair_reports) / len(pair_reports)
        for key in numeric_keys
    }
    result["iou_f1"] = {
        threshold: sum(report["iou_f1"][threshold] for report in pair_reports)
        / len(pair_reports)
        for threshold in ("0.25", "0.50", "0.75")
    }
    for key in ("mean_start_boundary_deviation", "mean_end_boundary_deviation"):
        values = [report[key] for report in pair_reports if report[key] is not None]
        result[key] = sum(values) / len(values) if values else None
    result["document_presence_agreement"] = sum(
        bool(report["document_presence_agreement"]) for report in pair_reports
    ) / len(pair_reports)
    return result


def intersect_intervals(left: list[Span], right: list[Span]) -> list[Span]:
    intersections: list[Span] = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if start <= end:
                intersections.append((start, end))
    intersections.sort()
    merged: list[Span] = []
    for start, end in intersections:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clip_spans(spans: list[Span], intervals: list[Span]) -> list[Span]:
    clipped = []
    for span_start, span_end in spans:
        for interval_start, interval_end in intervals:
            start = max(span_start, interval_start)
            end = min(span_end, interval_end)
            if start <= end:
                clipped.append((start, end))
    return clipped
