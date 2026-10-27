import { expect, test } from "./fixtures/test";
import { installEvidenceApiMock } from "./fixtures/mockEvidenceApi";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test("invite acceptance selects the invited workspace for an existing account", async ({ page }) => {
  await installPlatformApiMock(page, { includeSecondWorkspace: true });
  await page.route("**/api/invites/preview", (route) => route.fulfill({
    json: { workspace_name: "Screening Team", role: "annotator", expires_at: null },
  }));
  await page.route("**/api/invites/accept", (route) => route.fulfill({
    json: { authenticated: true, workspace_id: 2 },
  }));
  await page.goto("/invites/audit-invite");
  await page.getByRole("button", { name: "Accept invite", exact: true }).click();
  await expect(page).not.toHaveURL(/\/invites\//);
  await expect(page.getByRole("combobox", { name: "Workspace", exact: true })).toHaveValue("2");
});

test("switching an invite account preserves the invite and shows authentication", async ({ page }) => {
  await installPlatformApiMock(page);
  await page.route("**/api/invites/preview", (route) => route.fulfill({
    json: { workspace_name: "Screening Team", role: "annotator", expires_at: null },
  }));
  await page.route("**/api/auth/logout", (route) => route.fulfill({ status: 204 }));
  await page.goto("/invites/audit-invite");
  await page.getByRole("button", { name: "Switch account", exact: true }).click();
  await expect(page).toHaveURL(/\/invites\/audit-invite$/);
  await expect(page.getByRole("textbox", { name: /^Username/ })).toBeVisible();
  await expect(page.getByRole("button", { name: "Use an existing account" })).toBeVisible();
});

test("creating a block releases its edit state before the next selection", async ({ page }) => {
  const api = await installEvidenceApiMock(page, "annotator");
  await page.goto("/my-work");
  await page.getByRole("button", { name: "Resume annotating" }).click();
  const canvas = page.getByRole("region", { name: "Structure-aware evidence annotation canvas" });
  await expect(canvas.locator("[data-sentence-id]")).toHaveCount(5);
  await canvas.locator('[data-sentence-id="1001"]').click();
  await page.getByLabel("Note").fill("First block only");
  await page.getByRole("button", { name: "Create block", exact: true }).click();
  await expect(page.getByText("1 block", { exact: true })).toBeVisible();
  await canvas.locator('[data-sentence-id="1005"]').click();
  await expect(page.getByRole("button", { name: "Create block", exact: true })).toBeEnabled();
  await expect(page.getByLabel("Note")).toHaveValue("");
  await page.getByRole("button", { name: "Create block", exact: true }).click();
  await expect(page.getByText("2 blocks", { exact: true })).toBeVisible();
  const creates = api.requests.filter((request) => request.method === "POST" && request.pathname === "/api/annotations");
  expect(creates).toHaveLength(2);
  expect(creates[1]?.body).toMatchObject({ evidence_block: { start_sentence_id: 1005, end_sentence_id: 1005, note: null } });
});
