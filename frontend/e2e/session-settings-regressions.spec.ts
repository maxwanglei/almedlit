import { expect, test, type Page, type Route } from "@playwright/test";

import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test.use({ baseURL: process.env.AL_MEDLIT_E2E_BASE_URL ?? "http://127.0.0.1:4173" });

async function respond(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

test("cold login returns to the complete deep link", async ({ page }) => {
  await installPlatformApiMock(page);
  let authenticated = false;
  await page.route("**/api/auth/me", async (route) => {
    if (authenticated) await route.fallback();
    else await respond(route, { detail: "Session expired" }, 401);
  });
  await page.route("**/api/auth/login", async (route) => {
    authenticated = true;
    await respond(route, { authenticated: true });
  });
  const destination = "/projects?folder=screening#evidence";
  await page.goto(destination);
  await page.getByRole("textbox", { name: /^Username/ }).fill("alice");
  await page.getByLabel(/^Password/).fill("correct-password");
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
  await expect(page).toHaveURL(new RegExp("/projects\\?folder=screening#evidence$"));
});

test("a rejected invite password does not log out another tab", async ({ page, context }) => {
  await installPlatformApiMock(page);
  await page.route("**/api/auth/me", (route) => respond(route, { detail: "Sign in required" }, 401));
  await page.route("**/api/invites/browser-invite", (route) => respond(route, {
    workspace_name: "Invited team", role: "annotator", expires_at: "2099-01-01T00:00:00Z",
  }));
  await page.route("**/api/invites/browser-invite/accept", (route) =>
    respond(route, { detail: "Invalid username or password" }, 401));
  await page.goto("/invites/browser-invite");
  await page.getByRole("button", { name: "Use an existing account" }).click();
  const other = await context.newPage();
  await installPlatformApiMock(other);
  await other.goto("/projects");
  await expect(other.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
  await page.getByRole("textbox", { name: /^Username/ }).fill("alice");
  await page.getByLabel(/^Password/).fill("incorrect-password");
  await page.getByRole("button", { name: "Accept invite", exact: true }).click();
  await expect(page.getByText("Invalid username or password")).toBeVisible();
  await expect(other.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
  await expect(other.getByRole("button", { name: "Sign in", exact: true })).toHaveCount(0);
});

async function installSettings(page: Page, fail: boolean): Promise<void> {
  await installPlatformApiMock(page);
  await page.route("**/api/workspaces/1/members", (route) => respond(route, []));
  let reads = 0;
  await page.route("**/api/workspaces/1/capabilities", (route) => {
    reads += 1;
    if (fail && reads > 1) return respond(route, { detail: "Capabilities unavailable" }, 503);
    return respond(route, {
      preset: "custom", overrides: [], effective: ["annotation", "training", "lineage"], blocked: {},
    });
  });
}

test("failed settings load cannot submit a default capability preset", async ({ page }) => {
  await installSettings(page, true);
  await page.goto("/workspace-settings");
  await expect(page.getByText("Capabilities unavailable", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Save capabilities" })).toBeDisabled();
  await expect(page.getByRole("combobox", { name: /^Capability preset/ })).toBeDisabled();
});

test("settings displays the loaded custom policy", async ({ page }) => {
  await installSettings(page, false);
  await page.goto("/workspace-settings");
  const preset = page.getByRole("combobox", { name: /^Capability preset/ });
  await expect(preset).toHaveValue("custom");
  await expect(preset.locator("option:checked")).toHaveText("Custom");
});
