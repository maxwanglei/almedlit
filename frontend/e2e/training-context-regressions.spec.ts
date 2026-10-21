import { expect, test } from "@playwright/test";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test("training preparation keeps its context and nested-dialog draft on SPA navigation", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal" });
  let contextReads = 0;
  page.on("request", (request) => {
    if (new URL(request.url()).pathname.endsWith("/training-contexts")) contextReads += 1;
  });
  await page.goto("/training/data");
  await expect(page.getByRole("combobox", { name: "Project context", exact: true })).toHaveValue("1");
  const initialContextReads = contextReads;
  await page.getByRole("button", { name: "Create training dataset", exact: true }).first().click();
  await expect(page).toHaveURL(/projectId=1&flow=prepare/);
  const name = page.getByLabel("Training dataset name", { exact: true });
  await name.fill("Preserved preparation");
  const importButton = page.getByRole("button", { name: "Import external data", exact: true });
  await importButton.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Close", exact: true }).click();
  await expect(name).toHaveValue("Preserved preparation");
  await expect(importButton).toBeFocused();
  expect(contextReads).toBe(initialContextReads);
});
