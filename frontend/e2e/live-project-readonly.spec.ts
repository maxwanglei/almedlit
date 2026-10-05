import fs from "node:fs";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

// Opt-in replay of actual GET responses. No credentials are used, and every
// non-GET request is blocked: navigation checks cannot change the user's work.
const capturePath = process.env.AL_MEDLIT_READONLY_REPLAY;
const baseURL = process.env.AL_MEDLIT_REPLAY_BASE_URL ?? "http://127.0.0.1";
test.setTimeout(90_000);
test.skip(!capturePath, "Requires a local read-only API capture");

type CaptureResponse = { path: string; status: number; body: unknown };
type PaperItem = { dataset_item: { payload: { document_id: number; title: string } } };

async function setupReplay(page: Page) {
  const capture = JSON.parse(fs.readFileSync(capturePath!, "utf8")) as { responses: CaptureResponse[] };
  const key = (value: string) => {
    const url = new URL(value, "http://localhost");
    url.searchParams.sort();
    return `${url.pathname}${url.search}`;
  };
  const responses = new Map(capture.responses.map((response) => [key(response.path), response]));
  const unexpected: string[] = [];
  const errors: string[] = [];
  const navigation: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("framenavigated", (frame) => { if (frame === page.mainFrame()) navigation.push(frame.url()); });
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const request = route.request();
    const path = key(request.url());
    const response = responses.get(path);
    if (request.method() !== "GET" || !response) {
      unexpected.push(`${request.method()} ${path}`);
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Request blocked by read-only diagnostic replay" }) });
      return;
    }
    await route.fulfill({ status: response.status, contentType: "application/json", body: JSON.stringify(response.body) });
  });
  const papers = responses.get(key("/api/rounds/1/work-items?project_id=1"))!.body as PaperItem[];
  return { papers, unexpected, errors, navigation };
}

async function saveDiagnostics(page: Page, testInfo: TestInfo, replay: Awaited<ReturnType<typeof setupReplay>>, name: string) {
  await page.screenshot({ path: testInfo.outputPath(`${name}.png`), fullPage: false });
  await testInfo.attach(`${name}-diagnostics.json`, {
    contentType: "application/json",
    body: JSON.stringify({ unexpected: replay.unexpected, errors: replay.errors, navigation: replay.navigation,
      url: page.url(), activeQueueItem: await page.locator(".aw-queue-row.active").allTextContents(),
      body: await page.locator("body").innerText() }, null, 2),
  });
}

async function expectPaper(page: Page, documentId: number, title: string) {
  await expect.poll(() => new URL(page.url()).searchParams.get("document"), { timeout: 10_000 }).toBe(String(documentId));
  await expect(page.locator(".aw-queue-row.active")).toContainText(title, { timeout: 30_000 });
  // A stale editor with a changed highlight is still a failed document switch.
  await expect(page.locator(".aw-workspace-heading")).toContainText(title);
  const assignment = await page.getByRole("combobox", { name: "Task", exact: true }).inputValue();
  await expect.poll(() => new URL(page.url()).searchParams.get("assignment")).toBe(assignment);
}

for (const control of ["Next document", "queue"] as const) {
  test(`actual submitted paper changes using ${control}, reload, and browser history`, async ({ page }, testInfo) => {
    const replay = await setupReplay(page);
    const [first, second, third] = replay.papers.map((item) => item.dataset_item.payload);
    await page.goto(`${baseURL}/my-work/rounds/1?view=annotate&document=${first.document_id}`);
    await expect(page.locator(".aw-queue-row")).toHaveCount(replay.papers.length, { timeout: 30_000 });
    await expect(page.getByRole("combobox", { name: "Task", exact: true })).toContainText("Paper relevance");
    await expect(page.getByRole("combobox", { name: "Task", exact: true })).toContainText("drug name annoation");
    await saveDiagnostics(page, testInfo, replay, "before-switch");
    try {
      if (control === "Next document") await page.getByRole("button", { name: "Next document", exact: true }).click();
      else await page.locator(".aw-queue-row").nth(1).click();
      await expectPaper(page, second.document_id, second.title);
      expect(await page.getByRole("combobox", { name: "Task", exact: true }).inputValue()).not.toBe("1");
      await saveDiagnostics(page, testInfo, replay, "switched-to-paper-2");
      await page.locator(".aw-queue-row").nth(2).click();
      await expectPaper(page, third.document_id, third.title);
      await page.goBack();
      await expectPaper(page, second.document_id, second.title);
      await page.goForward();
      await expectPaper(page, third.document_id, third.title);
      await page.reload();
      await expectPaper(page, third.document_id, third.title);
      await page.locator(".aw-queue-row").first().click();
      await expectPaper(page, first.document_id, first.title);
      await expect(page.getByRole("button", { name: "Edit this paper task", exact: true })).toBeVisible();
      await expect(page.locator("#aw-pane-review")).toContainText("Tucatinib");
      await expect(page.locator("#aw-pane-review")).toContainText("HER2-positive metastatic breast cancer");
      expect(replay.unexpected).toEqual([]);
      expect(replay.errors).toEqual([]);
    } finally {
      await saveDiagnostics(page, testInfo, replay, "after-switch");
    }
  });
}

test("actual My Work has one header and per-task paper status; mobile queue supports keyboard navigation", async ({ page }, testInfo) => {
  const replay = await setupReplay(page);
  const [first, second] = replay.papers.map((item) => item.dataset_item.payload);
  await page.goto(`${baseURL}/my-work?project=1`);
  await expect(page.getByRole("heading", { level: 1, name: "My Work", exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1);
  await expect(page.getByRole("combobox", { name: "Project", exact: true })).toHaveCount(1);
  await expect(page.getByRole("link", { name: "Manage tasks", exact: true })).toHaveCount(1);
  const inventory = page.getByRole("region", { name: /Annotation tasks\s*2/ });
  await expect(inventory.getByRole("article")).toHaveCount(2);
  await expect(inventory).toContainText("Paper relevance");
  await expect(inventory).toContainText("drug name annoation");
  await expect(page.locator(".aw-progress-doc").first()).toContainText("Paper relevance: Submitted");
  await expect(page.locator(".aw-progress-doc").first()).toContainText("drug name annoation: Submitted");
  await expect(page.locator(".aw-progress-doc").nth(1)).toContainText("Paper relevance: To do");
  await expect(page.locator(".aw-progress-doc").nth(1)).toContainText("drug name annoation: To do");
  await saveDiagnostics(page, testInfo, replay, "my-work-desktop");
  await page.locator(".aw-progress-doc").nth(1).click();
  await expectPaper(page, second.document_id, second.title);
  const backToWork = page.getByRole("button", { name: /^Back to My Work/ });
  await expect(backToWork).toHaveCount(1);
  await backToWork.click();
  await expect(page.getByRole("heading", { level: 1, name: "My Work", exact: true })).toBeVisible();
  await page.setViewportSize({ width: 375, height: 812 });
  await saveDiagnostics(page, testInfo, replay, "my-work-mobile");
  const heading = await page.getByRole("heading", { level: 1, name: "My Work", exact: true }).boundingBox();
  const summary = await page.locator(".paper-progress-summary").boundingBox();
  const tasks = await inventory.boundingBox();
  expect(heading!.y).toBeLessThan(summary!.y);
  expect(heading!.y).toBeLessThan(tasks!.y);
  await page.goto(`${baseURL}/my-work/rounds/1?view=annotate&document=${first.document_id}`);
  await page.getByRole("tab", { name: "Queue", exact: true }).click();
  await expect(page.locator(".aw-queue-row")).toHaveCount(replay.papers.length);
  await page.locator(".aw-queue-row").nth(1).focus();
  await page.keyboard.press("Enter");
  await expect.poll(() => new URL(page.url()).searchParams.get("document")).toBe(String(second.document_id));
  await expect(page.locator(".aw-workspace-heading")).toContainText(second.title);
  await expect(page.getByRole("tab", { name: "Document", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await saveDiagnostics(page, testInfo, replay, "mobile-switched-paper");
  expect(replay.unexpected).toEqual([]);
  expect(replay.errors).toEqual([]);
});
