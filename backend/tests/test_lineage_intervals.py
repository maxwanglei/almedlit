from itertools import permutations

import pytest

from al_medlit.lineage.service import _intersect_interval_sets


@pytest.mark.parametrize(
    ("interval_sets", "expected"),
    [
        ([], []),
        ([[]], []),
        ([[(0, 4)], []], []),
        ([[], [(0, 4)]], []),
        ([[(0, 4), (2, 6), (8, 8)]], [(0, 6), (8, 8)]),
        ([[(0, 2), (1, 5)], [(0, 4)]], [(0, 4)]),
        ([[(0, 1), (2, 3)], [(1, 2)]], [(1, 2)]),
        (
            [[(8, 10), (0, 4)], [(2, 9)], [(3, 3), (8, 8)]],
            [(3, 3), (8, 8)],
        ),
    ],
    ids=[
        "no-reviewers",
        "empty-reviewer",
        "last-reviewer-empty",
        "first-reviewer-empty",
        "single-reviewer-overlaps",
        "merge-overlaps-before-intersecting",
        "merge-adjacent-sentence-intervals",
        "sort-before-intersecting-all-reviewers",
    ],
)
def test_review_coverage_intersection_is_canonical(interval_sets, expected):
    assert _intersect_interval_sets(interval_sets) == expected


@pytest.mark.parametrize("intervals", list(permutations([(0, 1), (0, 5), (2, 3)])))
def test_equal_start_coverage_rows_have_order_independent_intersection(intervals):
    # The database may return equal-start rows in either end-ordinal order.
    # A canonical result keeps the frozen manifest independent of that order.
    assert _intersect_interval_sets([list(intervals), [(0, 4)]]) == [(0, 4)]
    assert _intersect_interval_sets([[(0, 4)], list(intervals)]) == [(0, 4)]
