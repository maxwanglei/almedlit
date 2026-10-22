import { expect, test } from "./fixtures/test";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test("invalid runtime JSON opens its disclosure and focuses the described field", async ({ page }) => {
  const api = await installPlatformApiMock(page, { role: "personal" });
  await page.goto("/training/runtimes?projectId=1");
  await page.getByRole("textbox", { name: "Runtime name", exact: true }).fill("CPU runtime");
  await page.getByRole("textbox", { name: "Immutable image digest" }).fill(`sha256:${"a".repeat(64)}`);
  await page.getByRole("textbox", { name: "Policy name" }).fill("Artifacts");
  await page.getByRole("combobox", { name: "Runtime", exact: true }).selectOption("131");

  for (const { name, action, disclosure, invalid, message } of [
    { name: "Package manifest JSON", action: "Register runtime", disclosure: "Runtime manifest", invalid: "{", message: "Package manifest must be valid JSON." },
    { name: "Hardware constraints JSON", action: "Register runtime", disclosure: "Runtime manifest", invalid: "[]", message: "Hardware constraints must be a JSON object." },
    { name: "Preflight report JSON", action: "Record preflight", disclosure: null, invalid: "{", message: "Preflight report must be valid JSON." },
    { name: "Encryption settings JSON", action: "Add storage policy", disclosure: "Storage controls", invalid: "[]", message: "Encryption settings must be a JSON object." },
    { name: "Cache policy JSON", action: "Add storage policy", disclosure: "Storage controls", invalid: "{", message: "Cache policy must be valid JSON." },
  ]) {
    const field = page.getByRole("textbox", { name, exact: true });
    if (disclosure) await page.getByText(disclosure, { exact: true }).click();
    await field.fill(invalid);
    if (disclosure) await page.getByText(disclosure, { exact: true }).click();
    await page.getByRole("button", { name: action, exact: true }).click();
    await expect(field).toBeVisible();
    await expect(field).toBeFocused();
    await expect(field).toHaveAttribute("aria-invalid", "true");
    await expect(field).toHaveAccessibleName(name);
    await expect(field).toHaveAccessibleDescription(message);
    await field.fill("{}");
    await expect(field).not.toHaveAttribute("aria-invalid", "true");
    if (disclosure) await page.getByText(disclosure, { exact: true }).click();
  }
  expect(api.requests.filter(({ method }) => method === "POST")).toEqual([]);
});
