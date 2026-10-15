import { describe, expect, it } from "vitest";

import { createTextOffsetMap } from "./textOffsets";

describe("text offset boundaries", () => {
  it.each(["", "ASCII text", "é 中 e\u0301", "𝑃 = 𝑛 😀.", "👩‍🔬 disease"])(
    "round-trips every code-point boundary in %j",
    (text) => {
      const offsets = createTextOffsetMap(text);
      const characters = Array.from(text);
      expect(offsets.codePointLength).toBe(characters.length);
      for (let index = 0; index <= characters.length; index += 1) {
        const utf16Offset = offsets.toUtf16(index);
        expect(utf16Offset).toBe(characters.slice(0, index).join("").length);
        expect(offsets.toCodePoint(utf16Offset)).toBe(index);
      }
    },
  );

  it("rejects a DOM offset inside a surrogate pair", () => {
    const offsets = createTextOffsetMap("𝑃 😀");
    expect(offsets.toCodePoint(1)).toBeNull();
    expect(offsets.toCodePoint(4)).toBeNull();
  });
});
