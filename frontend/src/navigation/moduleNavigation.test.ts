import { describe, expect, it } from "vitest";

import { createAccessSnapshot } from "./AccessContext";
import {
  authorizedRedirectPath,
  availableModules,
  canAccessProjectSection,
  compatibilityRedirectNotice,
  currentModuleId,
  defaultModulePath,
  GLOBAL_ROUTE_REGISTRY,
  preserveLocation,
  resolveWorkspaceRoute,
  workspaceSettingsDestination,
  type ModuleNavigationContext,
} from "./moduleNavigation";

interface ContextOptions {
  workspaceKind?: string | null;
  role?: string | null;
  capabilities?: string[];
  blocked?: Record<string, string>;
  isSuperuser?: boolean;
  selectedProjectId?: number | null;
}

function context(
  options: ContextOptions = {},
): ModuleNavigationContext {
  return {
    ...createAccessSnapshot({
      workspaceId: 1,
      workspaceKind: options.workspaceKind ?? "team",
      membershipRole: options.role ?? "annotator",
      effectiveCapabilities: options.capabilities ?? ["annotation"],
      blockedCapabilities: options.blocked,
      isSuperuser: options.isSuperuser,
    }),
    selectedProjectId: options.selectedProjectId ?? null,
  };
}

describe("module navigation model", () => {
  it.each([
    {
      name: "individual owner with annotation",
      input: context({ workspaceKind: "individual", role: "admin" }),
      ids: ["my-work", "projects"],
    },
    {
      name: "individual owner with training",
      input: context({
        workspaceKind: "individual",
        role: "admin",
        capabilities: ["training"],
      }),
      ids: ["projects", "training", "models"],
    },
    {
      name: "team annotator",
      input: context(),
      ids: ["my-work"],
    },
    {
      name: "team trainer with annotation and training",
      input: context({
        role: "trainer",
        capabilities: ["annotation", "training"],
      }),
      ids: ["my-work", "projects", "training", "models"],
    },
    {
      name: "team manager without feature capabilities",
      input: context({ role: "manager", capabilities: [] }),
      ids: ["projects"],
    },
    {
      name: "workspace admin is not automatically a deployment administrator",
      input: context({
        role: "admin",
        capabilities: ["annotation", "training"],
      }),
      ids: ["my-work", "projects", "training", "models"],
    },
    {
      name: "deployment superuser",
      input: context({
        role: "admin",
        capabilities: ["annotation", "training"],
        isSuperuser: true,
      }),
      ids: ["my-work", "projects", "training", "models", "administration"],
    },
  ])("returns module-first navigation for a $name", ({ input, ids }) => {
    expect(availableModules(input).map((module) => module.id)).toEqual(ids);
  });

  it("publishes role, capability, command, and backend metadata", () => {
    expect(GLOBAL_ROUTE_REGISTRY).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "my-work",
          requiredCapability: "annotation",
          requiredCommand: "annotation:work",
          backendModule: "annotate",
        }),
        expect.objectContaining({
          id: "training",
          path: "/training",
          requiredCommand: "training:read",
          backendModule: "train",
        }),
        expect.objectContaining({
          id: "models",
          path: "/models",
          requiredCommand: "models:read",
        }),
        expect.objectContaining({
          id: "administration",
          requiredCommand: "system:admin",
        }),
      ]),
    );
  });

  it("uses the first available module and keeps Projects stable", () => {
    expect(defaultModulePath(context())).toBe("/my-work");
    expect(
      defaultModulePath(context({ role: "trainer", capabilities: ["training"] })),
    ).toBe("/projects");
    expect(
      availableModules(
        context({
          workspaceKind: "individual",
          role: "admin",
          capabilities: ["training"],
          selectedProjectId: 41,
        }),
      ).find((module) => module.id === "projects")?.path,
    ).toBe("/projects");
    expect(
      defaultModulePath(context({ workspaceKind: null, role: null, capabilities: [] })),
    ).toBe("/no-access");
    expect(
      defaultModulePath(
        context({
          workspaceKind: null,
          role: null,
          capabilities: [],
          isSuperuser: true,
        }),
      ),
    ).toBe("/admin/users");
  });

  it("recognizes every canonical route family and compatibility alias", () => {
    expect(resolveWorkspaceRoute("/my-work/rounds/31")).toBe("/my-work");
    expect(resolveWorkspaceRoute("/projects/3/quality/")).toBe("/projects");
    expect(resolveWorkspaceRoute("/training/new")).toBe("/training");
    expect(resolveWorkspaceRoute("/training/runs/17")).toBe("/training");
    expect(resolveWorkspaceRoute("/models/9/versions/3")).toBe("/models");
    expect(resolveWorkspaceRoute("/projects/3/train")).toBe("/training");
    expect(resolveWorkspaceRoute("/projects/3/models")).toBe("/models");
    expect(resolveWorkspaceRoute("/admin/users")).toBe("/admin");
    expect(resolveWorkspaceRoute("/no-access")).toBe("/no-access");
    expect(resolveWorkspaceRoute("/missing")).toBe("/not-found");
  });

  it("maps all route families to the correct global module", () => {
    const trainer = context({
      role: "trainer",
      capabilities: ["annotation", "training"],
    });
    expect(currentModuleId("/projects/3/tasks", trainer)).toBe("projects");
    expect(currentModuleId("/training/new", trainer)).toBe("training");
    expect(currentModuleId("/projects/3/train", trainer)).toBe("training");
    expect(currentModuleId("/models/2", trainer)).toBe("models");
    expect(currentModuleId("/missing", trainer)).toBeNull();
  });

  it("uses the same inherited-role policy for project sections", () => {
    const trainer = context({
      role: "trainer",
      capabilities: ["annotation", "training"],
    });
    expect(canAccessProjectSection(trainer, "overview")).toBe(true);
    expect(canAccessProjectSection(trainer, "data")).toBe(true);
    expect(canAccessProjectSection(trainer, "tasks")).toBe(false);
    expect(canAccessProjectSection(trainer, "settings")).toBe(false);

    const manager = context({
      role: "manager",
      capabilities: ["annotation"],
    });
    expect(canAccessProjectSection(manager, "tasks")).toBe(true);
    expect(canAccessProjectSection(manager, "quality")).toBe(true);
    expect(canAccessProjectSection(manager, "settings")).toBe(true);
  });

  it("canonicalizes aliases and project deep links", () => {
    const trainer = context({
      role: "trainer",
      capabilities: ["annotation", "training"],
    });
    expect(authorizedRedirectPath("/annotator/workbench", trainer)).toBe(
      "/my-work",
    );
    expect(authorizedRedirectPath("/manager/projects", trainer)).toBe(
      "/projects",
    );
    expect(authorizedRedirectPath("/trainer/training", trainer)).toBe(
      "/training",
    );
    expect(authorizedRedirectPath("/projects/12/train", trainer)).toBe(
      "/training?projectId=12",
    );
    expect(authorizedRedirectPath("/projects/12/models", trainer)).toBe(
      "/models?projectId=12",
    );
    expect(compatibilityRedirectNotice("/projects/12/train")).toContain(
      "independent workspace",
    );
    expect(compatibilityRedirectNotice("/admin/users")).toBeNull();
  });

  it("redirects known unauthorized routes to the role default", () => {
    const annotator = context();
    expect(authorizedRedirectPath("/training", annotator)).toBe("/my-work");
    expect(authorizedRedirectPath("/models", annotator)).toBe("/my-work");
    expect(authorizedRedirectPath("/projects", annotator)).toBe("/my-work");
    expect(authorizedRedirectPath("/admin/health", annotator)).toBe("/my-work");
    expect(authorizedRedirectPath("/does-not-exist", annotator)).toBeNull();
  });

  it("uses a stable no-access destination until a module becomes available", () => {
    const noModules = context({
      workspaceKind: null,
      role: null,
      capabilities: [],
    });
    expect(authorizedRedirectPath("/no-access", noModules)).toBeNull();
    expect(authorizedRedirectPath("/training", noModules)).toBe("/no-access");
    expect(authorizedRedirectPath("/no-access", context())).toBe("/my-work");
  });

  it("separates workspace settings from deployment administration", () => {
    const workspaceAdmin = context({ role: "admin", capabilities: [] });
    expect(workspaceSettingsDestination(workspaceAdmin)?.path).toBe(
      "/workspace-settings",
    );
    expect(currentModuleId("/admin/users", workspaceAdmin)).toBe("administration");
    expect(authorizedRedirectPath("/admin/users", context())).toBe("/my-work");
    expect(authorizedRedirectPath("/admin/users", workspaceAdmin)).toBe("/projects");
    expect(
      authorizedRedirectPath(
        "/admin/users",
        context({ isSuperuser: true }),
      ),
    ).toBeNull();
  });

  it("preserves legacy query and hash state while alias parameters win", () => {
    expect(
      preserveLocation(
        "/training?projectId=12",
        "?status=running&projectId=3",
        "#runs",
      ),
    ).toBe("/training?status=running&projectId=12#runs");
  });

});
