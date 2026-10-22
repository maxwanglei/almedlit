import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "./fixtures/test";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test("OS dark mode retains the light theme and readable settings actions", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await installPlatformApiMock(page, { role: "personal" });
  await page.goto("/projects/1/settings");
  await expect(page.getByRole("heading", { name: "Project settings", exact: true })).toBeVisible();
  await expect(page.locator('meta[name="color-scheme"]')).toHaveAttribute("content", "light");
  const save = page.getByRole("button", { name: "Save project", exact: true });
  await expect(save).toBeVisible();
  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations.map(({ id, nodes }) => ({ id, targets: nodes.map(({ target }) => target) }))).toEqual([]);
});

for (const width of [390, 820, 1440]) {
  test(`metric details and controls keep their type size at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 1000 });
    await installPlatformApiMock(page, { role: "manager" });
    await page.goto("/projects/1/overview");
    await expect(page.getByRole("heading", { name: "Project overview" })).toBeVisible();
    const details = page.locator(".platform-stat-detail");
    await expect(details.first()).toBeVisible();
    expect(await details.evaluateAll((elements) => elements.every((element) => {
      const size = Number.parseFloat(getComputedStyle(element).fontSize);
      return size >= 13 && size < 20;
    }))).toBe(true);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    await page.goto("/projects/1/settings");
    await expect(page.getByRole("button", { name: "Save project", exact: true })).toBeVisible();
    expect(await page.locator("input, select, textarea").evaluateAll((elements) => elements.every((element) => {
      if (getComputedStyle(element).display === "none") return true;
      return Number.parseFloat(getComputedStyle(element).fontSize) >= 13;
    }))).toBe(true);
  });
}

test("query history and loading refreshes preserve the active tab", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal" });
  await page.goto("/training/data?projectId=1&tab=training");
  const sources = page.getByRole("tab", { name: "Source datasets", exact: true });
  await expect(sources).toBeVisible();
  await sources.click();
  await expect(page).toHaveURL(/tab=source/);
  await expect(sources).toBeFocused();
  const announcement = page.locator(".unified-app-shell > [role='status']");
  await expect(announcement).toHaveText("Training data loaded.");
  await page.goBack();
  await expect(sources).toBeFocused();
  await expect(announcement).toHaveText("Training data loaded.");
});

test("failed projects do not show an empty-state create prompt", async ({ page }) => {
  await installPlatformApiMock(page, { role: "manager" });
  await page.route("**/api/projects?**", (route) => route.fulfill({ status: 503, json: { detail: "Projects temporarily unavailable" } }));
  await page.goto("/projects");
  await expect(page.getByRole("alert")).toContainText("Projects temporarily unavailable");
  await expect(page.getByText("No projects yet", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Retry/i })).toBeVisible();
});

test("failed rounds offer retry without claiming the round is missing", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal" });
  let attempts = 0;
  await page.route("**/api/rounds/71/work-context**", async (route) => {
    attempts += 1;
    if (attempts === 1) await route.fulfill({ status: 503, json: { detail: "Round service unavailable" } });
    else await route.fallback();
  });
  await page.goto("/my-work/rounds/71");
  await expect(page.getByRole("heading", { name: "Round could not be loaded" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Round not found" })).toHaveCount(0);
  await page.getByRole("button", { name: "Retry loading round" }).click();
  await expect(page.getByRole("heading", { name: "Round could not be loaded" })).toHaveCount(0);
  expect(attempts).toBe(2);
});

test("invalid settings fields keep their names and expose a single associated error", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal" });
  await page.goto("/projects/1/settings");
  const name = page.getByRole("textbox", { name: "Project name", exact: true });
  await name.fill("");
  await name.press("Tab");
  await expect(name).toHaveAttribute("aria-invalid", "true");
  await expect(name).toHaveAccessibleName("Project name");
  await expect(name).toHaveAccessibleDescription("Enter a project name.");
  await expect(page.getByText("Enter a project name.", { exact: true })).toHaveCount(1);
  await name.fill("Revised project");
  await expect(name).not.toHaveAttribute("aria-invalid", "true");
});

test("switching project sections shows loading before any empty-state action", async ({ page }) => {
  await installPlatformApiMock(page, { role: "manager" });
  await page.goto("/projects/1/overview");
  await expect(page.getByRole("heading", { name: "Project overview" })).toBeVisible();
  let finish: () => void = () => {};
  const pending = new Promise<void>((resolve) => { finish = resolve; });
  await page.route("**/api/rounds?**", async (route) => {
    await pending;
    await route.fallback();
  });
  await page.getByRole("link", { name: "Team & rounds", exact: true }).click();
  await expect(page.getByRole("status", { name: "Loading project…", exact: true })).toBeVisible();
  await expect(page.getByText("No rounds yet", { exact: true })).toHaveCount(0);
  finish();
  await expect(page.getByRole("heading", { name: "Team & rounds", exact: true })).toBeVisible();
});
