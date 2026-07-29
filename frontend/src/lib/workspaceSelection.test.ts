import { describe, expect, it } from "vitest";

import type { MeMembership } from "@/api/client";

import { preferredMembership } from "./workspaceSelection";

const memberships: MeMembership[] = [
  {
    workspace_id: 10,
    workspace_name: "Personal",
    workspace_kind: "individual",
    role: "admin",
  },
  {
    workspace_id: 20,
    workspace_name: "Research team",
    workspace_kind: "team",
    role: "annotator",
  },
];

describe("preferredMembership", () => {
  it("restores an accessible preferred workspace", () => {
    expect(preferredMembership(memberships, 20)).toEqual(memberships[1]);
  });

  it("falls back to the first membership when a saved workspace is inaccessible", () => {
    expect(preferredMembership(memberships, 999)).toEqual(memberships[0]);
  });
});
