import { describe, expect, it } from "vitest";

import {
  canPerform,
  createAccessSnapshot,
  effectiveRoleLabel,
  expandEffectiveRoles,
} from "./AccessContext";

describe("AccessContext policy", () => {
  it("expands the v4.3 role hierarchy without persona switching", () => {
    expect(expandEffectiveRoles("annotator", "team")).toEqual(["annotator"]);
    expect(expandEffectiveRoles("trainer", "team")).toEqual([
      "annotator",
      "trainer",
    ]);
    expect(expandEffectiveRoles("manager", "team")).toEqual([
      "annotator",
      "trainer",
      "manager",
    ]);
    expect(expandEffectiveRoles("admin", "team")).toEqual([
      "annotator",
      "trainer",
      "manager",
      "admin",
    ]);
  });

  it("gives an individual owner all role authority but not disabled capabilities", () => {
    const access = createAccessSnapshot({
      workspaceId: 1,
      workspaceKind: "individual",
      membershipRole: "admin",
      effectiveCapabilities: ["training"],
    });
    expect(effectiveRoleLabel(access)).toBe("Owner");
    expect(canPerform(access, "projects:create")).toBe(true);
    expect(canPerform(access, "training:launch")).toBe(true);
    expect(canPerform(access, "annotation:work")).toBe(false);
  });

  it("treats blocked infrastructure as unavailable", () => {
    const access = createAccessSnapshot({
      workspaceId: 1,
      workspaceKind: "team",
      membershipRole: "trainer",
      effectiveCapabilities: ["training"],
      blockedCapabilities: { training: "Runtime unavailable" },
    });
    expect(canPerform(access, "training:read")).toBe(false);
  });

  it("reserves system administration for deployment superusers", () => {
    const workspaceAdmin = createAccessSnapshot({
      workspaceId: 1,
      workspaceKind: "team",
      membershipRole: "admin",
      effectiveCapabilities: [],
    });
    expect(canPerform(workspaceAdmin, "workspace:manage")).toBe(true);
    expect(canPerform(workspaceAdmin, "system:admin")).toBe(false);
    expect(
      canPerform({ ...workspaceAdmin, isSuperuser: true }, "system:admin"),
    ).toBe(true);
  });
});
