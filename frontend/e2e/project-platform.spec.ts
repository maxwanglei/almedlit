import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test.setTimeout(90_000);

const viewports = [
  { label: "mobile-375", width: 375, height: 812 },
  { label: "tablet-768", width: 768, height: 1024 },
  { label: "compact-1024", width: 1024, height: 768 },
  { label: "desktop-1440", width: 1440, height: 900 },
] as const;

async function expectNoDocumentOverflow(page: Page): Promise<void> {
  await expect
    .poll(() =>
      page.evaluate(() => ({
        body:
          document.body.scrollWidth <=
          document.documentElement.clientWidth + 1,
        document:
          document.documentElement.scrollWidth <=
          document.documentElement.clientWidth + 1,
      })),
    )
    .toEqual({ body: true, document: true });
}

async function expectNoAxeViolations(page: Page): Promise<void> {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    results.violations,
    results.violations
      .map(
        (violation) =>
          `${violation.id}: ${violation.nodes
            .map((node) => node.target.join(" "))
            .join(", ")}`,
      )
      .join("\n"),
  ).toEqual([]);
}

async function expectRouteIdentity(
  page: Page,
  expectedLocation: string,
  heading: string,
  options: {
    title?: string;
    focused?: boolean;
  } = {},
): Promise<void> {
  const destinationHeading = page.getByRole("heading", {
    level: 1,
    name: heading,
    exact: true,
  });
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          `${window.location.pathname}${window.location.search}${window.location.hash}`,
      ),
    )
    .toBe(expectedLocation);
  await expect(destinationHeading).toBeVisible();
  await expect(page).toHaveTitle(`${options.title ?? heading} | AL-MedLit`);
  if (options.focused) {
    await expect(destinationHeading).toBeFocused();
  }
}

async function auditCurrentRoute(page: Page): Promise<void> {
  await expectNoDocumentOverflow(page);
  await expectNoAxeViolations(page);
}

for (const viewport of viewports) {
  test.describe(`${viewport.label} module workspaces`, () => {
    test.use({ viewport });

    test("canonical Projects, Training, New Training, and Models remain accessible", async ({
      page,
    }) => {
      await installPlatformApiMock(page, { role: "trainer" });
      await page.emulateMedia({ reducedMotion: "reduce" });

      await page.goto("/projects");
      await expectRouteIdentity(page, "/projects", "Projects");
      await expect(
        page.getByRole("region", { name: "Projects table" }),
      ).toContainText("Evidence Review Demo");
      await auditCurrentRoute(page);

      await page.goto("/training");
      await expectRouteIdentity(page, "/training", "Training");
      await expect(
        page.getByRole("region", { name: "Workspace training runs" }),
      ).toContainText("Relevance baseline");
      await expect(page.getByRole("link", { name: "Run 151" })).toBeVisible();
      await auditCurrentRoute(page);

      await page.goto("/training/new");
      await expectRouteIdentity(page, "/training/new", "New training");
      await expect(
        page.getByRole("radiogroup", { name: "Training context type" }),
      ).toBeVisible();
      await expect(
        page.getByText("Training-only project", { exact: true }),
      ).toBeVisible();
      await expect(
        page.getByRole("button", { name: "Use selected project" }),
      ).toBeVisible();
      await auditCurrentRoute(page);

      await page.goto("/models");
      await expectRouteIdentity(page, "/models", "Models");
      await expect(
        page.getByRole("region", { name: "Workspace model registry" }),
      ).toContainText("Relevance baseline");
      await expect(
        page.getByRole("link", { name: "Relevance baseline" }),
      ).toBeVisible();
      await auditCurrentRoute(page);
    });
  });
}

test("Trainer receives cumulative global modules while project navigation stays configuration-focused", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "trainer" });
  await page.goto("/projects");

  const primaryNavigation = page.getByRole("navigation", { name: "Primary" });
  for (const moduleName of ["My Work", "Projects", "Training", "Models"]) {
    await expect(
      primaryNavigation.getByRole("link", {
        name: moduleName,
        exact: true,
      }),
    ).toBeVisible();
  }
  await expect(
    primaryNavigation.getByRole("link", {
      name: "Administration",
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Workspace Settings" })).toHaveCount(
    0,
  );
  await expect(
    primaryNavigation.getByRole("link", { name: "Projects", exact: true }),
  ).toHaveAttribute("aria-current", "page");

  await page.getByRole("link", { name: "Open", exact: true }).click();
  await expectRouteIdentity(page, "/projects/1/overview", "Project overview", {
    focused: true,
  });

  const projectNavigation = page.getByRole("navigation", { name: "Project" });
  await expect(
    projectNavigation.getByRole("link", { name: "Overview", exact: true }),
  ).toBeVisible();
  await expect(
    projectNavigation.getByRole("link", { name: "Train", exact: true }),
  ).toHaveCount(0);
  await expect(
    projectNavigation.getByRole("link", { name: "Models", exact: true }),
  ).toHaveCount(0);

  const modelDevelopment = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Model development" }),
  });
  await modelDevelopment.getByRole("link", { name: /Training/ }).click();
  await expectRouteIdentity(
    page,
    "/training?projectId=1",
    "Training",
    { focused: true },
  );

  await page.goBack();
  await expectRouteIdentity(page, "/projects/1/overview", "Project overview", {
    focused: true,
  });
  await page
    .locator("section")
    .filter({
      has: page.getByRole("heading", { name: "Model development" }),
    })
    .getByRole("link", { name: /Models/ })
    .click();
  await expectRouteIdentity(page, "/models?projectId=1", "Models", {
    focused: true,
  });
});

test("individual owner receives every released workspace module", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "personal" });
  await page.goto("/projects");

  const primaryNavigation = page.getByRole("navigation", { name: "Primary" });
  for (const moduleName of ["My Work", "Projects", "Training", "Models"]) {
    await expect(
      primaryNavigation.getByRole("link", {
        name: moduleName,
        exact: true,
      }),
    ).toBeVisible();
  }
  await expect(
    primaryNavigation.getByRole("link", {
      name: "Administration",
      exact: true,
    }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("link", { name: "Workspace Settings", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Workspace" }),
  ).toContainText("Alice's Workspace · Owner");
});

test("New Training chooses project context before entering the five-step wizard", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "trainer" });
  await page.goto("/training/new");

  const contextPicker = page.getByRole("combobox", {
    name: "Existing project",
  });
  await contextPicker.selectOption("1");
  await page.getByRole("button", { name: "Use selected project" }).click();

  await expect(
    page.getByRole("button", {
      name: "Step 1 of 5: Task & Data",
      exact: true,
    }),
  ).toHaveAttribute("aria-current", "step");
  await expect(
    page.getByRole("combobox", { name: "Training dataset" }),
  ).toHaveValue("51");
  await expect(page.getByText("Evidence Review Demo", { exact: true })).toBeVisible();
  await expectNoDocumentOverflow(page);
  await expectNoAxeViolations(page);
});

test("legacy Training and Models URLs preserve query and hash context", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "trainer" });

  await page.goto("/trainer/training?source=legacy#runs");
  await expectRouteIdentity(
    page,
    "/training?source=legacy#runs",
    "Training",
  );
  await expect(
    page.getByRole("status").filter({
      hasText: "Training is now an independent workspace. Training loaded.",
    }),
  ).toHaveText("Training is now an independent workspace. Training loaded.");

  await page.goto("/projects/1/train?source=project#recipe");
  await expectRouteIdentity(
    page,
    "/training?source=project&projectId=1#recipe",
    "Training",
  );
  await expect(
    page.getByRole("status").filter({
      hasText: "Training is now an independent workspace. Training loaded.",
    }),
  ).toHaveText("Training is now an independent workspace. Training loaded.");

  await page.goto("/projects/1/models?source=project#version");
  await expectRouteIdentity(
    page,
    "/models?source=project&projectId=1#version",
    "Models",
  );
  await expect(
    page.getByRole("status").filter({
      hasText: "Models is now an independent workspace. Models loaded.",
    }),
  ).toHaveText("Models is now an independent workspace. Models loaded.");
});

test("workspace switching reloads effective access without a not-found transition", async ({
  page,
}) => {
  await installPlatformApiMock(page, {
    role: "trainer",
    includeSecondWorkspace: true,
  });
  await page.goto("/training");

  const visitedPaths: string[] = [];
  await page.exposeFunction("recordWorkspacePath", (path: string) => {
    visitedPaths.push(path);
  });
  await page.evaluate(() => {
    const record = (): void => {
      void (
        window as typeof window & {
          recordWorkspacePath: (path: string) => Promise<void>;
        }
      ).recordWorkspacePath(window.location.pathname);
    };
    const observer = new MutationObserver(record);
    observer.observe(document.body, { childList: true, subtree: true });
    record();
  });

  const workspacePicker = page.getByRole("combobox", { name: "Workspace" });
  await expect(workspacePicker).toHaveValue("1");
  await expect(
    workspacePicker.getByRole("option", {
      name: "Evidence Team · Model Trainer",
    }),
  ).toHaveCount(1);
  await expect(
    workspacePicker.getByRole("option", {
      name: "Screening Team · Annotator",
    }),
  ).toHaveCount(1);

  await workspacePicker.selectOption("2");
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          `${window.location.pathname}${window.location.search}${window.location.hash}`,
      ),
    )
    .toBe("/my-work");
  await expect(page).toHaveTitle("My Work | AL-MedLit");
  await expect(page.locator("main#main-content")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Page not found" })).toHaveCount(
    0,
  );
  expect(visitedPaths).not.toContain("/not-found");

  const primaryNavigation = page.getByRole("navigation", { name: "Primary" });
  await expect(
    primaryNavigation.getByRole("link", { name: "My Work", exact: true }),
  ).toBeVisible();
  for (const unavailable of ["Projects", "Training", "Models"]) {
    await expect(
      primaryNavigation.getByRole("link", {
        name: unavailable,
        exact: true,
      }),
    ).toHaveCount(0);
  }
});

test.describe("mobile navigation", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("drawer separates destinations from workspace switching and restores focus", async ({
    page,
  }) => {
    await installPlatformApiMock(page, {
      role: "trainer",
      includeSecondWorkspace: true,
    });
    await page.goto("/training");

    const menuButton = page.getByRole("button", {
      name: "Open navigation menu",
    });
    await menuButton.click();
    const drawer = page.getByRole("dialog", { name: "Navigation" });
    await expect(
      drawer.getByRole("heading", { name: "Go to", exact: true }),
    ).toBeVisible();
    await expect(
      drawer.getByRole("heading", { name: "Switch workspace", exact: true }),
    ).toBeVisible();
    for (const moduleName of ["My Work", "Projects", "Training", "Models"]) {
      await expect(
        drawer.getByRole("link", { name: moduleName, exact: true }),
      ).toBeVisible();
    }
    await expect(
      drawer.getByRole("combobox", { name: "Switch workspace" }),
    ).toHaveValue("1");
    await expect(
      drawer.getByRole("link", { name: "My Work", exact: true }),
    ).toBeFocused();
    await expectNoAxeViolations(page);

    await page.keyboard.press("Escape");
    await expect(drawer).toHaveCount(0);
    await expect(menuButton).toBeFocused();
  });
});

test.describe("200 percent zoom equivalent", () => {
  test.use({ viewport: { width: 720, height: 450 } });

  test("New Training reflows without page overflow", async ({ page }) => {
    await installPlatformApiMock(page, { role: "trainer" });
    await page.goto("/training/new");
    await expectRouteIdentity(page, "/training/new", "New training");
    await expect(
      page.getByRole("radiogroup", { name: "Training context type" }),
    ).toBeVisible();
    await expectNoDocumentOverflow(page);
    await expectNoAxeViolations(page);
  });
});
