export interface SentenceOrdinalRange {
  startOrdinal: number;
  endOrdinal: number;
}

export function normalizeSentenceRange(
  anchorOrdinal: number,
  headOrdinal: number,
): SentenceOrdinalRange {
  return anchorOrdinal <= headOrdinal
    ? { startOrdinal: anchorOrdinal, endOrdinal: headOrdinal }
    : { startOrdinal: headOrdinal, endOrdinal: anchorOrdinal };
}

export function rangesOverlap(
  first: SentenceOrdinalRange,
  second: SentenceOrdinalRange,
): boolean {
  return first.startOrdinal <= second.endOrdinal && second.startOrdinal <= first.endOrdinal;
}

export function rangesAreAdjacent(
  first: SentenceOrdinalRange,
  second: SentenceOrdinalRange,
): boolean {
  return first.endOrdinal + 1 === second.startOrdinal || second.endOrdinal + 1 === first.startOrdinal;
}

export function rangeContainsOrdinal(range: SentenceOrdinalRange, ordinal: number): boolean {
  return ordinal >= range.startOrdinal && ordinal <= range.endOrdinal;
}

export function clampSentenceRange(
  range: SentenceOrdinalRange,
  minimumOrdinal: number,
  maximumOrdinal: number,
): SentenceOrdinalRange {
  const startOrdinal = Math.max(minimumOrdinal, Math.min(range.startOrdinal, maximumOrdinal));
  const endOrdinal = Math.max(startOrdinal, Math.min(range.endOrdinal, maximumOrdinal));
  return { startOrdinal, endOrdinal };
}

export function expandRangeStart(
  range: SentenceOrdinalRange,
  minimumOrdinal: number,
): SentenceOrdinalRange {
  return { ...range, startOrdinal: Math.max(minimumOrdinal, range.startOrdinal - 1) };
}

export function expandRangeEnd(
  range: SentenceOrdinalRange,
  maximumOrdinal: number,
): SentenceOrdinalRange {
  return { ...range, endOrdinal: Math.min(maximumOrdinal, range.endOrdinal + 1) };
}

export function contractRangeStart(range: SentenceOrdinalRange): SentenceOrdinalRange {
  return {
    ...range,
    startOrdinal: Math.min(range.endOrdinal, range.startOrdinal + 1),
  };
}

export function contractRangeEnd(range: SentenceOrdinalRange): SentenceOrdinalRange {
  return {
    ...range,
    endOrdinal: Math.max(range.startOrdinal, range.endOrdinal - 1),
  };
}
