import { expect, test, type Locator, type Page } from "@playwright/test";

import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

async function expectMenuSeparatedFromTrigger(page: Page, trigger: Locator): Promise<void> {
  await trigger.click();
  const listbox = page.getByRole("listbox");
  await expect(listbox).toBeVisible();

  const [triggerBox, listboxBox] = await Promise.all([
    trigger.boundingBox(),
    listbox.boundingBox(),
  ]);
  expect(triggerBox).not.toBeNull();
  expect(listboxBox).not.toBeNull();
  if (!triggerBox || !listboxBox) {
    return;
  }

  const opensAbove = listboxBox.y + listboxBox.height <= triggerBox.y + 1;
  const opensBelow =
    listboxBox.y >= triggerBox.y + triggerBox.height - 1;
  expect(opensAbove || opensBelow).toBe(true);

  const layerVisuals = await listbox.evaluate((element) => {
    const layer = element.closest<HTMLElement>("[popover]");
    const dialog = element.closest<HTMLElement>('[role="dialog"]');
    return {
      layerOpacity: layer ? Number.parseFloat(getComputedStyle(layer).opacity) : 0,
      dialogBackground: dialog
        ? getComputedStyle(dialog).backgroundColor
        : "transparent",
    };
  });
  expect(layerVisuals.layerOpacity).toBe(1);
  expect(layerVisuals.dialogBackground).not.toBe("rgba(0, 0, 0, 0)");
  expect(layerVisuals.dialogBackground).not.toBe("transparent");

  const listboxOwnsTopLayerPoint = await page.evaluate(
    ({ x, y }) =>
      document.elementFromPoint(x, y)?.closest('[role="listbox"]') !== null,
    {
      x: listboxBox.x + Math.min(24, listboxBox.width / 2),
      y: listboxBox.y + Math.min(16, listboxBox.height / 2),
    },
  );
  expect(listboxOwnsTopLayerPoint).toBe(true);
}

test("task creation opens above the task list in the current project dialog", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installPlatformApiMock(page, { role: "manager" });
  await page.goto("/projects/1/tasks");

  await expect(page.getByRole("heading", { name: "Tasks", exact: true })).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Task contracts table" }),
  ).toContainText("Abstract relevance");

  const addTaskButton = page.getByRole("button", {
    name: "New task",
    exact: true,
  });
  const dialog = page.getByRole("dialog", { name: "Define NLP task" });
  await expect(dialog).toHaveCount(0);

  await addTaskButton.click();
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("textbox", { name: "Name" })).toBeFocused();
  await expect(
    dialog.getByRole("combobox", { name: "Task kind" }),
  ).toHaveValue("classification");

  const dialogVisuals = await dialog.evaluate((element) => ({
    background: getComputedStyle(element).backgroundColor,
    opacity: Number.parseFloat(getComputedStyle(element).opacity),
  }));
  expect(dialogVisuals.opacity).toBe(1);
  expect(dialogVisuals.background).not.toBe("rgba(0, 0, 0, 0)");
  expect(dialogVisuals.background).not.toBe("transparent");

  const dialogBox = await dialog.boundingBox();
  expect(dialogBox).not.toBeNull();
  if (dialogBox) {
    const dialogOwnsTopLayerPoint = await page.evaluate(
      ({ x, y }) =>
        document.elementFromPoint(x, y)?.closest('[role="dialog"]') !== null,
      {
        x: dialogBox.x + Math.min(32, dialogBox.width / 2),
        y: dialogBox.y + Math.min(32, dialogBox.height / 2),
      },
    );
    expect(dialogOwnsTopLayerPoint).toBe(true);
  }

  await expect(
    page.getByRole("region", { name: "Task contracts table" }),
  ).toBeVisible();

  for (const viewport of [
    { width: 390, height: 844, expectedColumns: 1 },
    { width: 844, height: 390, expectedColumns: 2 },
  ]) {
    await page.setViewportSize({
      width: viewport.width,
      height: viewport.height,
    });
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            document.documentElement.scrollWidth <=
            document.documentElement.clientWidth + 1,
        ),
      )
      .toBe(true);
    await expect
      .poll(() =>
        dialog.locator(".platform-form-grid").evaluate(
          (element) =>
            getComputedStyle(element).gridTemplateColumns.split(" ").length,
        ),
      )
      .toBe(viewport.expectedColumns);
  }

  await dialog.getByRole("button", { name: "Close" }).click();
  await expect(dialog).toHaveCount(0);
  await expect(addTaskButton).toBeFocused();
});

test("registration workspace selector opens below its field", async ({ page }) => {
  await page.goto("/login");
  await page.getByRole("button", { name: "Create an account" }).click();

  await expectMenuSeparatedFromTrigger(
    page,
    page.getByRole("combobox", { name: "Workspace" }),
  );
});
