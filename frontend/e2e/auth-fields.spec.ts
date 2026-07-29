import { expect, test } from "@playwright/test";

test("filled login fields stay inside one rounded control shell", async ({
  page,
}) => {
  await page.goto("/login");
  await page.getByLabel("Username").fill("testteam");
  await page.getByLabel("Password").fill("correct-horse-battery-staple");

  const cdp = await page.context().newCDPSession(page);
  await Promise.all([cdp.send("DOM.enable"), cdp.send("CSS.enable")]);
  const { root } = await cdp.send("DOM.getDocument");
  for (const selector of [
    'input[name="username"]',
    'input[name="password"]',
  ]) {
    const { nodeId } = await cdp.send("DOM.querySelector", {
      nodeId: root.nodeId,
      selector,
    });
    await cdp.send("CSS.forcePseudoState", {
      nodeId,
      forcedPseudoClasses: ["autofill"],
    });
  }

  for (const fieldName of ["Username", "Password"]) {
    const input = page.getByLabel(fieldName);
    const control = input.locator("..");
    const label = page.locator(`label[for="${await input.getAttribute("id")}"]`);

    await expect(control).toHaveClass(/auth-field-control/);
    await expect(label).toContainText("Required");
    await expect
      .poll(() => label.evaluate((element) => getComputedStyle(element).display))
      .toBe("flex");
    await expect
      .poll(() => control.evaluate((element) => getComputedStyle(element).overflow))
      .toBe("hidden");
    await expect
      .poll(() => input.evaluate((element) => getComputedStyle(element).boxShadow))
      .not.toBe("none");

    const [inputBox, controlBox] = await Promise.all([
      input.boundingBox(),
      control.boundingBox(),
    ]);
    expect(inputBox).not.toBeNull();
    expect(controlBox).not.toBeNull();
    if (!inputBox || !controlBox) {
      continue;
    }
    expect(inputBox.y).toBeGreaterThanOrEqual(controlBox.y);
    expect(inputBox.y + inputBox.height).toBeLessThanOrEqual(
      controlBox.y + controlBox.height,
    );
  }
});
