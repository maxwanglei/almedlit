import { expect, test, type Page } from "@playwright/test";

import { installEvidenceApiMock } from "./fixtures/mockEvidenceApi";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

async function openAssignedEvidenceWorkbench(page: Page): Promise<void> {
  await page.goto("/my-work");
  await page.getByRole("button", { name: "Resume annotating" }).click();
  await expect
    .poll(() => {
      const url = new URL(page.url());
      return {
        pathname: url.pathname,
        view: url.searchParams.get("view"),
        project: url.searchParams.get("project"),
        document: url.searchParams.get("document"),
        assignment: url.searchParams.get("assignment"),
      };
    })
    .toEqual({
      pathname: "/my-work",
      view: "annotate",
      project: "1",
      document: "41",
      assignment: "51",
    });
}

test("individual user moves between My Work, task annotation, and project sections", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "personal" });

  await page.goto("/my-work");
  const primaryNavigation = page.getByRole("navigation", { name: "Primary" });
  await expect(primaryNavigation.getByRole("link", { name: "My Work" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  await expect(primaryNavigation.getByRole("link")).toHaveText([
    "My Work",
    "Projects",
    "Training",
    "Models",
  ]);

  await page
    .getByRole("button", { name: /Three-paragraph clinical trial.*Continue/ })
    .click();
  await expect(page.getByLabel("Task")).toHaveValue("51");
  await expect(
    page.getByRole("button", { name: "Submit task for this paper" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Submit assignment" }),
  ).toHaveCount(0);
  await page
    .getByRole("button", { name: "Submit task for this paper" })
    .click();
  await expect(
    page.getByRole("dialog", { name: "Submit this paper task?" }),
  ).toContainText("This paper task has no annotations.");
  await expect(
    page.getByRole("button", { name: "Submit with no annotations" }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "Keep editing" }).click();

  await primaryNavigation.getByRole("link", { name: "Projects" }).click();
  await expect(page).toHaveURL(/\/projects$/);
  await expect(page.getByRole("heading", { name: "Projects" })).toBeVisible();
  await expect(
    page.getByRole("button", { name: "New project", exact: true }),
  ).toBeVisible();

  await page.getByRole("link", { name: "Open", exact: true }).click();
  await expect(page).toHaveURL(/\/projects\/1\/overview$/);
  await expect(
    page.getByRole("heading", { name: "Project overview" }),
  ).toBeVisible();

  await page
    .getByRole("navigation", { name: "Project" })
    .getByRole("link", { name: "Data", exact: true })
    .click();
  await expect(page).toHaveURL(/\/projects\/1\/data$/);
  await expect(page.getByRole("heading", { name: "Data", exact: true })).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Dataset registry table" }),
  ).toContainText("PubMed screening corpus");
  await expect(
    page.getByRole("region", { name: "Label layers table" }),
  ).toContainText("Imported screening decisions");
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          document.documentElement.scrollWidth <=
          document.documentElement.clientWidth + 1,
      ),
    )
    .toBe(true);
});

test("annotator header keeps navigation, actions, and document context separated", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  await installEvidenceApiMock(page, "annotator");

  await openAssignedEvidenceWorkbench(page);

  await expect(
    page.getByRole("navigation", { name: "Primary" }).getByRole("link", {
      name: "My Work",
    }),
  ).toHaveAttribute("aria-current", "page");
  await expect(
    page.locator(".aw-context").getByText("Three-paragraph clinical trial", {
      exact: true,
    }),
  ).toBeVisible();

  const tabs = page.locator(".aw-tabs");
  const actions = page.locator(".aw-primary-actions");
  const context = page.locator(".aw-context");
  const [tabsBox, actionsBox, contextBox] = await Promise.all([
    tabs.boundingBox(),
    actions.boundingBox(),
    context.boundingBox(),
  ]);

  expect(tabsBox).not.toBeNull();
  expect(actionsBox).not.toBeNull();
  expect(contextBox).not.toBeNull();
  if (!tabsBox || !actionsBox || !contextBox) {
    return;
  }

  expect(tabsBox.y + tabsBox.height).toBeLessThanOrEqual(contextBox.y);
  expect(actionsBox.y + actionsBox.height).toBeLessThanOrEqual(contextBox.y);
});

test("annotator creates a cross-paragraph block, reviews coverage, and submits its target assignment", async ({
  page,
}) => {
  const api = await installEvidenceApiMock(page, "annotator");

  await openAssignedEvidenceWorkbench(page);
  await expect(
    page.getByRole("navigation", { name: "Primary" }).getByRole("link", {
      name: "My Work",
    }),
  ).toHaveAttribute("aria-current", "page");
  await page.getByRole("button", { name: "Evidence blocks" }).click();

  const canvas = page.getByRole("region", {
    name: "Structure-aware evidence annotation canvas",
  });
  await expect(canvas.locator("[data-sentence-id]")).toHaveCount(5);

  await canvas.locator('[data-sentence-id="1001"]').click();
  await canvas.locator('[data-sentence-id="1004"]').click({ modifiers: ["Shift"] });
  await expect(canvas.locator('[data-sentence-id="1002"]')).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await page.getByLabel("Note").fill("Three-paragraph evidence block");
  await page.getByRole("button", { name: "Create block" }).click();

  await expect(page.getByText("1 blocks", { exact: true })).toBeVisible();
  await expect(page.getByText("Sentences 1–4", { exact: true })).toBeVisible();
  const createRequest = api.requests.find(
    (request) => request.method === "POST" && request.pathname === "/api/annotations",
  );
  expect(createRequest?.body).toMatchObject({
    project_id: 1,
    document_id: 41,
    annotation_type: "evidence_block",
    label: "evidence_block",
    annotator_id: "alice",
    evidence_block: {
      structure_version_id: 201,
      target_version_id: 101,
      start_sentence_id: 1001,
      end_sentence_id: 1004,
      note: "Three-paragraph evidence block",
    },
  });
  expect(createRequest?.body).not.toHaveProperty("start_offset");
  expect(createRequest?.body).not.toHaveProperty("end_offset");

  await page.getByRole("button", { name: "Review region" }).click();
  await canvas.locator('[data-sentence-id="1001"]').click();
  await canvas.locator('[data-sentence-id="1005"]').click({ modifiers: ["Shift"] });
  await page.getByRole("button", { name: "Mark reviewed" }).click();
  await expect(page.getByText("Fully reviewed", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Submit assignment" }).click();
  await expect(
    page.getByRole("dialog", { name: "Submit this assignment?" }),
  ).toContainText("1");
  await page.getByRole("button", { name: "Confirm submission" }).click();
  await expect
    .poll(() =>
      api.requests.some(
        (request) =>
          request.method === "POST" &&
          request.pathname === "/api/projects/1/documents/41/submissions",
      ),
    )
    .toBe(true);
  const submissionRequest = api.requests.find(
    (request) =>
      request.method === "POST" &&
      request.pathname === "/api/projects/1/documents/41/submissions",
  );
  expect(submissionRequest?.body).toMatchObject({
    annotator_id: "alice",
    assignment_id: 51,
  });
  expect(api.unhandledRequests).toEqual([]);
});

test("manager reviews quality scope and explicitly creates union gold", async ({ page }) => {
  const platformApi = await installPlatformApiMock(page, { role: "manager" });
  const api = platformApi.evidence;

  await page.goto("/projects/1/quality");
  await expect(
    page.getByRole("heading", { name: "Quality & Review" }),
  ).toBeVisible();
  await expect(
    page.getByRole("region", { name: "Round quality table" }),
  ).toContainText("False-negative review");

  await expect(page.getByLabel("Review scope")).toContainText(
    "Three-paragraph clinical trial",
  );
  await page.getByRole("button", { name: "Load comparison" }).click();

  const comparison = page.getByRole("region", { name: "Evidence comparison" });
  await expect(comparison).toContainText("alice");
  await expect(comparison).toContainText("Alice boundary");
  await expect(comparison).toContainText("bob");
  await expect(comparison).toContainText("Bob boundary");
  await page.getByRole("checkbox", { name: "Use evidence block 701" }).check();
  await page.getByRole("checkbox", { name: "Use evidence block 702" }).check();
  await page.getByLabel("Gold boundary").selectOption("union");
  await page
    .getByLabel("Gold note")
    .fill("Consensus union after adjudication");

  const adjudicationPayload = {
    target_version_id: 101,
    structure_version_id: 201,
    guideline_version_id: 301,
    strategy: "union",
    source_annotation_ids: [701, 702],
    start_sentence_id: null,
    end_sentence_id: null,
    note: "Consensus union after adjudication",
    solo_gold: false,
  };
  await page.getByRole("button", { name: "Create gold block" }).click();
  await expect(
    page
      .getByRole("status")
      .filter({ hasText: "Gold evidence block #999 created" }),
  ).toHaveText("Gold evidence block #999 created");
  await expect(comparison).toContainText("manager");
  await expect(comparison).toContainText("Consensus union after adjudication");
  await expect(
    page.getByRole("checkbox", { name: "Use evidence block 999" }),
  ).toBeDisabled();

  const adjudicationRequest = api.requests.find(
    (request) =>
      request.method === "POST" &&
      request.pathname === "/api/projects/1/documents/41/evidence-adjudication",
  );
  expect(adjudicationRequest?.body).toEqual(adjudicationPayload);
  expect(api.unhandledRequests).toEqual([]);
});
