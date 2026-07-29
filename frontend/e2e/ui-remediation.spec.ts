import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { installEvidenceApiMock } from "./fixtures/mockEvidenceApi";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test.setTimeout(60_000);

const viewports = [
  { label: "desktop", width: 1440, height: 900 },
  { label: "tablet", width: 820, height: 1180 },
  { label: "mobile", width: 390, height: 844 },
] as const;

async function disableMotion(page: Page): Promise<void> {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.addStyleTag({
    content: "*, *::before, *::after { animation: none !important; transition: none !important; caret-color: transparent !important; }",
  });
}

async function expectNoDocumentOverflow(page: Page): Promise<void> {
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.documentElement.scrollWidth <=
          document.documentElement.clientWidth + 1,
      ),
    )
    .toBe(true);
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

for (const viewport of viewports) {
  test.describe(`${viewport.label} UI remediation`, () => {
    test.use({ viewport });

    test("login, My Work, manager, and annotator remain contained and accessible", async ({
      browser,
      page,
    }) => {
      await page.goto("/login");
      await disableMotion(page);
      await expect(page.getByRole("heading", { name: "AL-MedLit" })).toBeVisible();
      await page.keyboard.press("Tab");
      await expect(page.getByRole("link", { name: "Skip to main content" })).toBeFocused();
      await expectNoDocumentOverflow(page);
      await expectNoAxeViolations(page);

      const personalContext = await browser.newContext({ viewport });
      const personalPage = await personalContext.newPage();
      await installEvidenceApiMock(personalPage, "personal");
      await personalPage.goto("/my-work");
      await disableMotion(personalPage);
      await expect(personalPage.getByRole("heading", { name: "My Work" })).toBeVisible();
      await expect(
        personalPage.getByRole("button", {
          name: "Continue",
          exact: true,
        }),
      ).toHaveCount(1);
      if (viewport.width <= 760) {
        const menuButton = personalPage.getByRole("button", {
          name: "Open navigation menu",
        });
        await menuButton.click();
        await expect(
          personalPage.getByRole("link", { name: "My Work", exact: true }),
        ).toBeFocused();
        await personalPage.keyboard.press("Escape");
        await expect(menuButton).toBeFocused();
        await menuButton.evaluate((element) => (element as HTMLElement).blur());
      }
      await expectNoDocumentOverflow(personalPage);
      await expectNoAxeViolations(personalPage);

      const managerContext = await browser.newContext({ viewport });
      const managerPage = await managerContext.newPage();
      await installPlatformApiMock(managerPage, { role: "manager" });
      await managerPage.goto("/projects/1/overview");
      await disableMotion(managerPage);
      await expect(
        managerPage.getByRole("heading", { name: "Project overview" }),
      ).toBeVisible();
      if (viewport.width <= 760) {
        await expect(
          managerPage.getByRole("combobox", { name: "Project section" }),
        ).toHaveValue("overview");
      } else {
        await expect(
          managerPage.getByRole("link", { name: "Overview" }),
        ).toBeVisible();
      }
      await expectNoDocumentOverflow(managerPage);
      await expectNoAxeViolations(managerPage);

      const annotatorContext = await browser.newContext({ viewport });
      const annotatorPage = await annotatorContext.newPage();
      await installEvidenceApiMock(annotatorPage, "annotator");
      await annotatorPage.goto(
        "/annotator/workbench?view=annotate&project=1&document=41&assignment=51&tool=evidence&pane=document",
      );
      await disableMotion(annotatorPage);
      await expect(
        annotatorPage.getByRole("heading", {
          name: "Three-paragraph clinical trial",
        }),
      ).toBeVisible();
      const firstSentence = annotatorPage.getByText(
        "Trial enrolled adults at three centers.",
      );
      await expect(firstSentence).toBeVisible();
      if (viewport.width <= 760) {
        const sentenceBox = await firstSentence.boundingBox();
        expect(sentenceBox).not.toBeNull();
        expect((sentenceBox?.y ?? viewport.height) + (sentenceBox?.height ?? 0)).toBeLessThan(
          viewport.height,
        );
      }
      if (viewport.width <= 760) {
        const workspaceTabs = annotatorPage.getByRole("tablist", {
          name: "Workspace panes",
        });
        await expect(workspaceTabs.getByRole("tab", { name: "Document" })).toHaveAttribute(
          "aria-selected",
          "true",
        );
        await workspaceTabs.getByRole("tab", { name: "Queue" }).click();
        await expect(annotatorPage.getByRole("heading", { name: /Queue/ })).toBeVisible();
        await workspaceTabs.getByRole("tab", { name: "Document" }).click();
      }
      await expectNoDocumentOverflow(annotatorPage);
      await expectNoAxeViolations(annotatorPage);

      await managerPage.goto("/projects/1/overview");
      await disableMotion(managerPage);
      await expect(
        managerPage.getByRole("heading", { name: "Project overview" }),
      ).toBeVisible();
      await expectNoDocumentOverflow(managerPage);
      await expectNoAxeViolations(managerPage);

      await Promise.all([
        personalContext.close(),
        managerContext.close(),
        annotatorContext.close(),
      ]);
    });
  });
}
