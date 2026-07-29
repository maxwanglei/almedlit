import { describe, expect, it } from "vitest";

import {
  clampSentenceRange,
  contractRangeEnd,
  contractRangeStart,
  expandRangeEnd,
  expandRangeStart,
  normalizeSentenceRange,
  rangeContainsOrdinal,
  rangesAreAdjacent,
  rangesOverlap,
} from "./evidenceBlockSelection";

describe("evidence block sentence range selection", () => {
  it("normalizes forward and backward selections", () => {
    expect(normalizeSentenceRange(3, 8)).toEqual({ startOrdinal: 3, endOrdinal: 8 });
    expect(normalizeSentenceRange(8, 3)).toEqual({ startOrdinal: 3, endOrdinal: 8 });
  });

  it("distinguishes overlap from adjacency", () => {
    const first = { startOrdinal: 2, endOrdinal: 4 };
    expect(rangesOverlap(first, { startOrdinal: 4, endOrdinal: 7 })).toBe(true);
    expect(rangesOverlap(first, { startOrdinal: 5, endOrdinal: 7 })).toBe(false);
    expect(rangesAreAdjacent(first, { startOrdinal: 5, endOrdinal: 7 })).toBe(true);
  });

  it("expands and contracts without producing an invalid range", () => {
    const range = { startOrdinal: 4, endOrdinal: 6 };
    expect(expandRangeStart(range, 0)).toEqual({ startOrdinal: 3, endOrdinal: 6 });
    expect(expandRangeEnd(range, 6)).toEqual(range);
    expect(contractRangeStart({ startOrdinal: 4, endOrdinal: 4 })).toEqual({
      startOrdinal: 4,
      endOrdinal: 4,
    });
    expect(contractRangeEnd({ startOrdinal: 4, endOrdinal: 4 })).toEqual({
      startOrdinal: 4,
      endOrdinal: 4,
    });
  });

  it("clamps ranges and tests inclusive membership", () => {
    const range = clampSentenceRange({ startOrdinal: -4, endOrdinal: 20 }, 0, 10);
    expect(range).toEqual({ startOrdinal: 0, endOrdinal: 10 });
    expect(rangeContainsOrdinal(range, 10)).toBe(true);
    expect(rangeContainsOrdinal(range, 11)).toBe(false);
  });
});
