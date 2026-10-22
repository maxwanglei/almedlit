import type { Page, Route } from "@playwright/test";
import type { Project } from "../src/types/api";
import { expect, test } from "./fixtures/test";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

async function respond(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function project(id: number, name: string): Project {
  return {
    id, name, description: null, workspace_id: 1, tasks: [],
    annotation_schema: { labels: {} }, annotation_validation_mode: "strict",
    settings: { modules: ["data"] }, workflow_task_count: 0, workflow_round_count: 0,
  };
}

async function installProjectSwitchFixture(page: Page): Promise<string[]> {
  await installPlatformApiMock(page);
  const projects = [project(1, "First collection"), project(2, "Second collection")];
  const reads: string[] = [];
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() !== "GET") return route.fallback();
    if (["/api/projects", "/api/projects/my-work"].includes(url.pathname)) return respond(route, projects);
    if (url.pathname === "/api/workspaces/1/my-work/rounds") return respond(route, []);
    const moduleMatch = url.pathname.match(/^\/api\/projects\/([12])\/modules$/);
    if (moduleMatch) {
      reads.push(url.pathname);
      return respond(route, { project_id: Number(moduleMatch[1]), selected: ["data"], effective: ["data"], workspace_capabilities: ["data"] });
    }
    if (url.pathname === "/api/documents") {
      reads.push(`${url.pathname}?project_id=${url.searchParams.get("project_id")}`);
      return respond(route, []);
    }
    const progressMatch = url.pathname.match(/^\/api\/projects\/([12])\/progress$/);
    if (progressMatch) return respond(route, { project_id: Number(progressMatch[1]), total: 0, by_status: {}, by_task: [], by_document: [], by_annotator: [] });
    if (/^\/api\/projects\/[12]\/assignments$/.test(url.pathname) || ["/api/datasets", "/api/datasets/split-maps"].includes(url.pathname)) return respond(route, []);
    return route.fallback();
  });
  await page.addInitScript(() => {
    const record = (): void => {
      const paths: string[] = JSON.parse(sessionStorage.getItem("test:project-paths") ?? "[]");
      paths.push(`${location.pathname}${location.search}`);
      sessionStorage.setItem("test:project-paths", JSON.stringify(paths));
    };
    for (const method of ["pushState", "replaceState"] as const) {
      const navigate = history[method].bind(history);
      history[method] = (data: unknown, unused: string, url?: string | URL | null): void => {
        navigate(data, unused, url);
        record();
      };
    }
  });
  return reads;
}

test("switching projects navigates once and loads the selected project once", async ({ page }) => {
  const reads = await installProjectSwitchFixture(page);
  await page.goto("/projects/1/overview");
  await expect(page.getByRole("combobox", { name: "Project", exact: true })).toHaveValue("1");
  await expect(page.getByRole("heading", { name: "Project overview", exact: true })).toBeVisible();
  reads.length = 0;
  await page.evaluate(() => sessionStorage.removeItem("test:project-paths"));

  await page.getByRole("combobox", { name: "Project", exact: true }).selectOption("2");
  await expect(page).toHaveURL(/\/projects\/2\/overview$/);
  await expect(page.getByRole("combobox", { name: "Project", exact: true })).toHaveValue("2");
  await expect(page.locator(".platform-context-bar strong")).toHaveText("Second collection");
  await expect.poll(() => reads.filter((path) => path === "/api/documents?project_id=2").length).toBe(1);
  expect([...reads].sort()).toEqual(["/api/documents?project_id=2", "/api/projects/2/modules"]);
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem("test:project-paths") ?? "[]"))).toEqual(["/projects/2/overview"]);
});

test("the My Work project selector changes URL context without loading either project twice", async ({ page }) => {
  const reads = await installProjectSwitchFixture(page);
  await page.goto("/my-work?project=1");
  const projectPicker = page.locator(".aw-project-select");
  await expect(projectPicker).toHaveValue("1");
  await expect(projectPicker).toBeEnabled();
  await expect(page.getByRole("heading", { name: "My Work", exact: true })).toBeVisible();
  reads.length = 0;
  await page.evaluate(() => sessionStorage.removeItem("test:project-paths"));

  await projectPicker.selectOption("2");
  await expect(page).toHaveURL(/\/my-work\?project=2$/);
  await expect(projectPicker).toHaveValue("2");
  await expect(projectPicker).toBeEnabled();
  expect(reads).toEqual(["/api/documents?project_id=2"]);
  expect(await page.evaluate(() => JSON.parse(sessionStorage.getItem("test:project-paths") ?? "[]"))).toEqual(["/my-work?project=2"]);
});

test("opening and closing assigned round work keeps the loaded round queue", async ({ page }) => {
  const mock = await installPlatformApiMock(page);
  await page.route(/\/api\/rounds\/71\/(?:work-items|decisions|submissions)\?/, (route) => respond(route, []));
  await page.goto("/my-work");
  const continueRound = page.getByRole("link", { name: "Continue Abstract relevance", exact: true });
  await expect(continueRound).toBeVisible();
  await continueRound.click();
  await expect(page).toHaveURL(/\/my-work\/rounds\/71\?view=annotate$/);
  await expect(page.getByRole("heading", { name: "False-negative review", exact: true })).toBeVisible();
  await expect(page.getByText("This round has no materialized items.", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Close workbench", exact: true }).click();
  await expect(page).toHaveURL(/\/my-work\?project=1&datasetVersionId=22(?:&|$)/);
  await expect(continueRound).toBeVisible();
  expect(mock.requests.filter((request) => request.pathname === "/api/workspaces/1/my-work/rounds")).toHaveLength(1);
});

test("accepting an invite while signed in refreshes another tab without discarding its draft", async ({ page, context }) => {
  const other = await context.newPage();
  const mock = await installPlatformApiMock(other);
  await other.goto("/projects/1/data?flow=import");
  const collectionName = other.getByRole("textbox", { name: "Dataset name" });
  await expect(collectionName).toBeVisible();
  await collectionName.fill("Unfinished review collection");
  await other.evaluate(() => sessionStorage.setItem("prediction-draft:1:22", "unsaved prediction"));
  const initialSessionReads = mock.requests.filter((request) => request.pathname === "/api/auth/me").length;
  const initialProjectReads = mock.requests.filter((request) => request.pathname === "/api/projects").length;
  await installPlatformApiMock(page);
  await page.route("**/api/invites/preview", (route) => respond(route, {
    workspace_name: "Invited team", role: "annotator", expires_at: "2099-01-01T00:00:00Z",
  }));
  await page.route("**/api/invites/accept", (route) => respond(route, { authenticated: true }));
  await page.goto("/invites/browser-invite");
  await page.getByRole("button", { name: "Accept invite", exact: true }).click();
  await expect(page).not.toHaveURL(/\/invites\//);
  await expect.poll(() => mock.requests.filter((request) => request.pathname === "/api/auth/me").length).toBe(initialSessionReads + 1);
  await expect(collectionName).toHaveValue("Unfinished review collection");
  expect(await other.evaluate(() => sessionStorage.getItem("prediction-draft:1:22"))).toBe("unsaved prediction");
  expect(mock.requests.filter((request) => request.pathname === "/api/projects")).toHaveLength(initialProjectReads);
  expect(await other.evaluate(() => localStorage.getItem("al_medlit_session_event"))).toBeNull();
});

test("signing out clears account drafts in this tab and another tab", async ({ page, context }) => {
  await installPlatformApiMock(page);
  await page.route("**/api/auth/logout", (route) => respond(route, null, 204));
  const other = await context.newPage();
  await installPlatformApiMock(other);
  await Promise.all([page.goto("/projects"), other.goto("/projects")]);
  for (const tab of [page, other]) {
    await expect(tab.getByRole("heading", { name: "Projects", exact: true })).toBeVisible();
    await tab.evaluate(() => {
      sessionStorage.setItem("al-medlit:source-setup:1:new", "source draft");
      sessionStorage.setItem("prediction-draft:1:22", "prediction draft");
      sessionStorage.setItem("prediction-review:1:22", "prediction review");
      sessionStorage.setItem("other-app:draft", "keep this");
    });
  }
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  for (const tab of [page, other]) {
    await expect(tab.getByRole("button", { name: "Sign in", exact: true })).toBeVisible();
    expect(await tab.evaluate(() => ({
      source: sessionStorage.getItem("al-medlit:source-setup:1:new"),
      prediction: sessionStorage.getItem("prediction-draft:1:22"),
      review: sessionStorage.getItem("prediction-review:1:22"),
      other: sessionStorage.getItem("other-app:draft"),
    }))).toEqual({ source: null, prediction: null, review: null, other: "keep this" });
  }
});

async function openPubmedImport(page: Page): Promise<void> {
  await installPlatformApiMock(page);
  await page.goto("/projects/1/data?flow=import");
  await expect(page.getByRole("heading", { name: "Add papers from PubMed", exact: true })).toBeVisible();
}

test("PubMed accepts a differently named CSV column and blocks import during a new preview", async ({ page }) => {
  await openPubmedImport(page);
  await page.getByLabel("Upload PMID file").setInputFiles({
    name: "publications.csv", mimeType: "text/csv", buffer: Buffer.from('Article identifier,Title\r\n29860986,"A title, with a comma"\r\n29717446,Another title'),
  });
  await expect(page.getByRole("textbox", { name: "PMIDs", exact: true })).toHaveValue("29860986\n29717446");
  await page.getByRole("button", { name: "Preview documents", exact: true }).click();
  const importButton = page.getByRole("button", { name: "Import 2 selected documents", exact: true });
  await expect(importButton).toBeEnabled();
  let finishPreview: (() => void) | undefined;
  const previewResponse = new Promise<void>((resolve) => { finishPreview = resolve; });
  await page.route("**/api/projects/1/import/pubmed/preview", async (route) => {
    await previewResponse;
    await route.fallback();
  });
  await page.getByRole("button", { name: "Preview documents", exact: true }).click();
  try {
    await expect(page.getByRole("status").filter({ hasText: "Checking documents, 0 of 2 PMIDs." })).toBeVisible();
    await expect(importButton).toBeDisabled();
    await expect(page.getByRole("textbox", { name: "PMIDs", exact: true })).toBeDisabled();
  } finally {
    finishPreview?.();
  }
  await expect(importButton).toBeEnabled();
});
