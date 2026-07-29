import { describe, expect, it } from "vitest";

import {
  parseProjectPlatformRoute,
  projectBackendModule,
  projectPlatformPath,
  projectSupportsSection,
  PROJECT_ROUTE_REGISTRY,
  projectSwitchPath,
  withSearchAndHash,
} from "./navigation";

describe("project platform navigation", () => {
  it("parses canonical project tabs with navigation metadata", () => {
    expect(parseProjectPlatformRoute("/projects/42/rounds")).toEqual({
      projectId: 42,
      tab: "rounds",
      requestedTab: "rounds",
      canonicalPath: "/projects/42/rounds",
      redirect: null,
    });
    expect(parseProjectPlatformRoute("/projects/42")).toEqual({
      projectId: 42,
      tab: "overview",
      requestedTab: null,
      canonicalPath: "/projects/42/overview",
      redirect: { kind: "canonical", notice: null },
    });
  });

  it.each([
    ["annotate", "rounds"],
    ["documents", "data"],
    ["export", "data"],
    ["progress", "activity"],
  ] as const)("maps legacy %s routes to %s", (alias, canonicalTab) => {
    const route = parseProjectPlatformRoute(`/projects/4/${alias}`);
    expect(route).toMatchObject({
      tab: canonicalTab,
      canonicalPath: `/projects/4/${canonicalTab}`,
      redirect: { kind: "legacy", notice: null },
    });
  });

  it.each([
    ["learning", "Learning Loop"],
    ["loop", "Learning Loop"],
    ["guidelines", "Guideline Learning"],
  ] as const)(
    "redirects planned %s routes to Overview with an announcement",
    (plannedTab, featureName) => {
      const route = parseProjectPlatformRoute(`/projects/4/${plannedTab}`);
      expect(route?.tab).toBe("overview");
      expect(route?.canonicalPath).toBe("/projects/4/overview");
      expect(route?.redirect?.kind).toBe("planned");
      expect(route?.redirect?.notice).toContain(featureName);
    },
  );

  it("maps the canonical rounds tab to the annotate backend module", () => {
    expect(projectBackendModule("rounds")).toBe("annotate");
    expect(
      PROJECT_ROUTE_REGISTRY.find((route) => route.id === "rounds"),
    ).toMatchObject({
      label: "Team & Rounds",
      requiredCapability: "annotation",
      backendModule: "annotate",
      releaseState: "released",
    });
  });

  it("falls back to Overview when a section is unavailable after project switch", () => {
    expect(
      projectSwitchPath(9, "quality", ["overview", "data", "rounds"]),
    ).toBe("/projects/9/overview");
    expect(
      projectSwitchPath(9, "rounds", ["overview", "data", "rounds"]),
    ).toBe("/projects/9/rounds");
  });

  it("derives training-only project sections from configured modules", () => {
    const project = {
      id: 9,
      name: "Public benchmark",
      description: null,
      annotation_schema: { labels: {} },
      annotation_validation_mode: "strict" as const,
      tasks: [],
      settings: {
        modules: ["data", "models", "train", "activity"],
      },
      workspace_id: 4,
    };

    expect(projectSupportsSection(project, "data")).toBe(true);
    expect(projectSupportsSection(project, "activity")).toBe(true);
    expect(projectSupportsSection(project, "tasks")).toBe(false);
    expect(projectSupportsSection(project, "rounds")).toBe(false);
  });

  it("rejects unknown tabs and unsafe project ids", () => {
    expect(parseProjectPlatformRoute("/projects/4/nope")).toBeNull();
    expect(
      parseProjectPlatformRoute("/projects/9007199254740992/data"),
    ).toBeNull();
  });

  it("builds canonical paths and preserves the current query and hash", () => {
    const path = projectPlatformPath(9, "settings");
    expect(path).toBe("/projects/9/settings");
    expect(withSearchAndHash(path, "?view=versions", "#latest")).toBe(
      "/projects/9/settings?view=versions#latest",
    );
    expect(withSearchAndHash(path, "view=versions", "latest")).toBe(
      "/projects/9/settings?view=versions#latest",
    );
  });
});
