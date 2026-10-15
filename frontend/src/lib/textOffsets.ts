export interface TextOffsetMap {
  codePointLength: number;
  toUtf16: (codePointOffset: number) => number;
  toCodePoint: (utf16Offset: number) => number | null;
}

/** Map API code-point boundaries to JavaScript/DOM UTF-16 boundaries once per text. */
export function createTextOffsetMap(text: string): TextOffsetMap {
  const boundaries = [0];
  let utf16Offset = 0;
  for (const character of text) {
    utf16Offset += character.length;
    boundaries.push(utf16Offset);
  }

  return {
    codePointLength: boundaries.length - 1,
    toUtf16: (offset) => boundaries[offset],
    toCodePoint: (offset) => {
      let low = 0;
      let high = boundaries.length - 1;
      while (low <= high) {
        const middle = Math.floor((low + high) / 2);
        const boundary = boundaries[middle];
        if (boundary === offset) return middle;
        if (boundary < offset) low = middle + 1;
        else high = middle - 1;
      }
      // A DOM range can bisect a surrogate pair; that is not an API boundary.
      return null;
    },
  };
}
