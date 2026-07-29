import { describe, expect, it } from "vitest";

import { canTrain, hasCapability } from "./capabilities";

describe("capability helpers", () => {
  it("hasCapability checks membership", () => {
    expect(hasCapability(["annotation", "training"], "training")).toBe(true);
    expect(hasCapability(["annotation"], "training")).toBe(false);
  });

  it("canTrain is true only when training is present", () => {
    expect(canTrain(["annotation", "training", "inference"])).toBe(true);
    expect(canTrain(["annotation"])).toBe(false);
  });
});
