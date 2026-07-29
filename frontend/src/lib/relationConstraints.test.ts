import { describe, expect, it } from "vitest";

import {
  filterRelationLabels,
  relationConstraintsOf,
  relationLabelAllows,
} from "@/lib/relationConstraints";
import type { LabelDef } from "@/types/api";

const LABELS: LabelDef[] = [
  { name: "causes", color: "#8e3a3a", description: null },
  { name: "treats", color: "#16a34a", description: null },
];

describe("relationConstraintsOf", () => {
  it("returns empty map when task or settings are missing", () => {
    expect(relationConstraintsOf(null)).toEqual({});
    expect(relationConstraintsOf({ settings: {} })).toEqual({});
  });

  it("parses well-formed constraints", () => {
    const task = {
      settings: {
        relation_constraints: {
          causes: { head: ["Drug"], tail: ["AdverseEvent"] },
        },
      },
    };
    expect(relationConstraintsOf(task)).toEqual({
      causes: { head: ["Drug"], tail: ["AdverseEvent"] },
    });
  });

  it("normalizes malformed entries instead of crashing", () => {
    const task = {
      settings: {
        relation_constraints: {
          causes: { head: "Drug", tail: [7, "AdverseEvent"] },
          treats: null,
          broken: "x",
        },
      },
    };
    expect(relationConstraintsOf(task)).toEqual({
      causes: { head: [], tail: ["AdverseEvent"] },
    });
  });
});

describe("relationLabelAllows", () => {
  it("allows everything when there is no constraint", () => {
    expect(relationLabelAllows(undefined, "Drug", "Disease")).toBe(true);
  });

  it("treats an empty side as unconstrained", () => {
    const constraint = { head: ["Drug"], tail: [] };
    expect(relationLabelAllows(constraint, "Drug", "Anything")).toBe(true);
    expect(relationLabelAllows(constraint, "Disease", "Anything")).toBe(false);
  });

  it("requires both sides to match", () => {
    const constraint = { head: ["Drug"], tail: ["AdverseEvent"] };
    expect(relationLabelAllows(constraint, "Drug", "AdverseEvent")).toBe(true);
    expect(relationLabelAllows(constraint, "Drug", "Disease")).toBe(false);
    expect(relationLabelAllows(constraint, "AdverseEvent", "Drug")).toBe(false);
  });
});

describe("filterRelationLabels", () => {
  it("keeps labels whose constraints pass and unconstrained labels", () => {
    const constraints = {
      causes: { head: ["Drug"], tail: ["AdverseEvent"] },
    };
    expect(
      filterRelationLabels(LABELS, constraints, "Drug", "AdverseEvent").map(
        (label) => label.name,
      ),
    ).toEqual(["causes", "treats"]);
    expect(
      filterRelationLabels(LABELS, constraints, "Disease", "Drug").map(
        (label) => label.name,
      ),
    ).toEqual(["treats"]);
  });
});
