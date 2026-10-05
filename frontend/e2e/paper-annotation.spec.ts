import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";

import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

test.setTimeout(90_000);

const timestamp = "2026-09-18T12:00:00Z";
const hash = "b".repeat(64);
const base = { project_id: 1, created_at: timestamp, updated_at: timestamp, created_by_user_id: 2 };
type Resource = Record<string, unknown> & { id: number };

async function installPaperAnnotationMock(page: Page) {
  await installPlatformApiMock(page, { role: "personal" });
  const project = {
    id: 1, name: "Tucatinib screening project", description: "Saved import with two tasks",
    workspace_id: 1, tasks: [], workflow_task_count: 2, workflow_round_count: 1,
    annotation_schema: { labels: {} }, annotation_validation_mode: "relaxed",
    settings: { modules: ["data", "annotate", "train", "models", "activity"] },
  };
  const documents = [
    { id: 41, title: "Tucatinib study", text: "Tucatinib improved treatment outcomes.", external_id: "31825569" },
    { id: 42, title: "Second study", text: "No eligible drug treatment was reported.", external_id: "33226752" },
  ].map((document) => ({ ...document, project_id: 1, source: "pubmed_abstract", metadata_: {}, sentences: [[0, document.text.length]], active_structure_version_id: null }));
  const definitions = [
    { ...base, id: 1, name: "Paper relevance", key: "paper-relevance", description: "Screen papers" },
    { ...base, id: 2, name: "Drug entities", key: "drug-entities", description: "Mark drug names" },
  ];
  const versions: Resource[] = [
    { ...base, id: 1, task_definition_id: 1, version_number: 1, task_kind: "classification",
      input_schema: { type: "object", required: ["text"], properties: { text: { type: "string" } } },
      output_schema: { type: "string", enum: ["Relevant", "Not relevant"] },
      label_rules: { values: ["Relevant", "Not relevant"], closed_set: true },
      annotation_ui: { preset: "classification" }, trainer_compatibility: ["tfidf_logistic_regression"], metrics: ["f1"], content_hash: hash },
    { ...base, id: 2, task_definition_id: 2, version_number: 1, task_kind: "token_labeling",
      input_schema: { type: "object", required: ["tokens"], properties: { tokens: { type: "array", items: { type: "string" } } } },
      output_schema: { type: "array", items: { type: "string" } },
      label_rules: { values: ["Drug"], closed_set: true }, annotation_ui: { preset: "token_labeling" },
      trainer_compatibility: ["transformer_token_classification"], metrics: ["entity_f1"], content_hash: hash },
  ];
  const initialTokenVersion = JSON.stringify(versions[1]);
  const dataset = { ...base, id: 1, name: "Tucatinib papers", source_type: "project_corpus", purposes: ["annotation"] };
  const source = { ...base, id: 1, dataset_id: 1, version_number: 1, item_count: 2,
    source_format: "other", source_revision: hash, source_uri: null, content_hash: hash,
    data_schema: {}, provenance: { source_document_ids: [41, 42], ingestion: "project_corpus" }, license_info: {} };
  const items = documents.map((document, index) => ({ ...base, id: 101 + index, dataset_version_id: 1,
    stable_key: `project-document:${document.id}`, group_key: document.external_id, content_hash: hash,
    payload: { document_id: document.id, title: document.title, text: document.text, external_id: document.external_id, metadata: {} } }));
  const rounds: Resource[] = [{ ...base, id: 1, name: "Tucatinib papers · Paper relevance", sequence: 1,
    dataset_version_id: 1, task_version_id: 1, assistance_policy: "blind", reannotation_mode: "full_dataset",
    status: "open", annotator_user_ids: [2], open_to_all_annotators: false,
    cycle_id: null, parent_round_id: null, selection_set_version_id: null, guideline_revision_id: null,
    feedback_set_version_id: null, opened_at: timestamp, closed_at: null }];
  const decisions: Resource[] = [];
  const submissions: Resource[] = [];
  const mutations: Array<{ path: string; body: Record<string, unknown> }> = [];
  const context = (round: Resource) => {
    const taskVersion = versions.find((version) => version.id === round.task_version_id)!;
    return { project: { id: 1, name: project.name }, round: { ...round, feedback_available: false },
      task: definitions.find((definition) => definition.id === taskVersion.task_definition_id),
      task_version: taskVersion, cycle: null, guideline: null };
  };
  await page.route(/\/api\//, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const json = (value: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });
    const body = request.method() === "POST" ? request.postDataJSON() as Record<string, unknown> : {};
    if (request.method() === "POST") mutations.push({ path, body });
    if (request.method() === "GET") {
      if (["/api/projects", "/api/projects/my-work"].includes(path)) return json([{ ...project, workflow_round_count: rounds.length }]);
      if (path === "/api/projects/1") return json({ ...project, workflow_round_count: rounds.length });
      if (path === "/api/documents") return json(documents);
      if (path === "/api/tasks") return json(definitions);
      if (path === "/api/tasks/versions") return json(versions.filter((version) => !url.searchParams.has("task_definition_id") || version.task_definition_id === Number(url.searchParams.get("task_definition_id"))));
      if (path === "/api/datasets") return json([dataset]);
      if (path === "/api/datasets/versions") {
        if (!url.searchParams.has("project_id") || !url.searchParams.has("dataset_id")) {
          return json({ detail: "project_id and dataset_id are required" }, 422);
        }
        return json(Number(url.searchParams.get("project_id")) === 1 && Number(url.searchParams.get("dataset_id")) === source.dataset_id ? [source] : []);
      }
      if (path === "/api/datasets/items") return json(items);
      if (["/api/datasets/training-versions", "/api/datasets/label-sets", "/api/datasets/split-maps", "/api/projects/1/assignments", "/api/projects/1/tasks", "/api/feedback-runs", "/api/feedback-sets", "/api/models", "/api/training-runs"].includes(path)) return json([]);
      if (path === "/api/projects/1/progress") return json({ project_id: 1, total: 0, by_status: {}, by_task: [], by_document: [], by_annotator: [], by_target: [] });
      if (path === "/api/projects/1/datasets/annotation-progress") return json([{ dataset_id: 1, dataset_version_id: 1, total: 2, submitted: submissions.length ? 1 : 0 }]);
      if (path === "/api/workspaces/1/my-work/rounds") return json(rounds.filter((round) => round.status === "open").map(context));
      if (path === "/api/rounds") return json(rounds);
      const roundPath = path.match(/^\/api\/rounds\/(\d+)\/(work-context|work-items|decisions|submissions)$/);
      if (roundPath) {
        const round = rounds.find((item) => item.id === Number(roundPath[1]))!;
        if (!round) return json({ detail: "Round not found" }, 404);
        if (roundPath[2] === "work-context") return json(context(round));
        if (roundPath[2] === "work-items") return json(items.map((item, index) => ({ dataset_item: item, round_item: {
          ...base, id: round.id * 1000 + index + 1, annotation_round_id: round.id, dataset_item_id: item.id,
          selection_rank: null, selection_reason: {}, metadata_: {},
        } })));
        if (roundPath[2] === "decisions") return json(decisions.filter((decision) => Math.floor(Number(decision.round_item_id) / 1000) === round.id));
        return json(submissions.filter((submission) => submission.annotation_round_id === round.id));
      }
    }
    if (path === "/api/tasks/versions" && request.method() === "POST") {
      const version = { ...base, ...body, id: 3, version_number: 2, content_hash: "c".repeat(64) };
      versions.push(version); return json(version, 201);
    }
    if (path === "/api/rounds" && request.method() === "POST") {
      const round = { ...base, ...body, id: 2, sequence: 2, status: "draft", opened_at: null, closed_at: null };
      rounds.push(round); return json(round, 201);
    }
    if (path === "/api/rounds/2/transition") {
      const round = rounds.find((item) => item.id === 2)!;
      round.status = body.status; round.opened_at = timestamp; return json(round);
    }
    if (path === "/api/rounds/decisions") {
      const decision = { ...base, ...body, id: 200 + decisions.length, annotator_user_id: 2, content_hash: hash };
      decisions.push(decision); return json(decision, 201);
    }
    if (path === "/api/rounds/submissions") {
      const submission = { ...base, ...body, id: 300 + submissions.length, annotator_user_id: 2, sequence: 1, content_hash: hash, submitted_at: timestamp };
      submissions.push(submission); return json(submission, 201);
    }
    // A projected editor record must never be persisted in the document API.
    if (path === "/api/annotations" && request.method() === "POST") return json({ detail: "Unexpected legacy annotation write" }, 500);
    return route.fallback();
  });
  return { project, versions, rounds, decisions, submissions, mutations, initialTokenVersion };
}

async function selectDrugText(page: Page) {
  const sentence = page.locator('[data-sentence-index="0"]').filter({ hasText: "Tucatinib improved" }).first();
  await sentence.evaluate((element) => {
    const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node && !node.textContent?.includes("Tucatinib")) node = walker.nextNode();
    if (!node) throw new Error("Paper text was not rendered");
    const start = node.textContent!.indexOf("Tucatinib");
    const range = document.createRange();
    range.setStart(node, start); range.setEnd(node, start + "Tucatinib".length);
    const selection = window.getSelection()!;
    selection.removeAllRanges(); selection.addRange(range);
    element.dispatchEvent(new MouseEvent("mouseup", { bubbles: true }));
  });
}

async function expectUnifiedMyWork(page: Page) {
  await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1);
  await expect(page.getByRole("heading", { level: 1, name: "My Work", exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Project", exact: true })).toHaveCount(1);
  await expect(page.getByRole("link", { name: "Manage tasks", exact: true })).toHaveCount(1);
  await expect(page.getByRole("button", { name: /^(Start|Continue) annotating$/ })).toHaveCount(1);
  await expect(page.getByRole("heading", { name: "Annotate papers", exact: true })).toHaveCount(0);
}

test("saved classification and token NER setup share the original paper editor without losing decisions", async ({ page }, testInfo) => {
  const state = await installPaperAnnotationMock(page);
  await page.goto("/projects/1/overview");
  await expect(page.getByRole("heading", { name: "Annotation tasks", exact: true })).toBeVisible();
  await expect(page.locator(".platform-stats > div").filter({ has: page.getByText("Annotation tasks", { exact: true }) }).locator("dd").first()).toHaveText("2");
  await expect(page.getByRole("row").filter({ hasText: "Paper relevance" })).toContainText("1 open round");
  await expect(page.getByRole("row").filter({ hasText: "Drug entities" })).toContainText("No annotation round yet");

  // Reproduce the saved project before any repair: two tasks, only one round.
  // My Work must list NER even though the user has not created its round yet.
  await page.goto("/my-work?project=1");
  await expectUnifiedMyWork(page);
  const taskInventory = page.locator(".paper-task-inventory");
  await expect(taskInventory.getByRole("heading", { level: 2 })).toHaveText(/Annotation tasks\s*2/);
  await expect(taskInventory.getByText("Paper relevance", { exact: true })).toBeVisible();
  await expect(taskInventory.getByText("Drug entities", { exact: true })).toBeVisible();
  await expect(taskInventory).toContainText("Needs setup");
  await expect(taskInventory).toContainText("Set up this task for paper annotation.");
  await taskInventory.getByRole("link", { name: "Continue Paper relevance", exact: true }).click();
  await expect(page).toHaveURL(/\/my-work\/rounds\/1\?view=annotate/);
  await expect(page.getByRole("tabpanel", { name: "Document", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Relevant/ })).toBeVisible();
  await expect(taskInventory).toHaveCount(0);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Tucatinib study");
  // Direct round links also default to the paper editor without a view query.
  await page.goto("/my-work/rounds/1");
  await expect(page.getByRole("tabpanel", { name: "Document", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Relevant/ })).toBeVisible();
  await expect(taskInventory).toHaveCount(0);
  await page.getByRole("button", { name: /^Back to My Work/ }).click();
  await expectUnifiedMyWork(page);
  await expect(taskInventory.getByRole("heading", { level: 2 })).toHaveText(/Annotation tasks\s*2/);
  await page.getByRole("link", { name: "Set up Drug entities", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Annotation task", exact: true })).toHaveValue("2");
  await expect(page.getByRole("button", { name: "Start annotating", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Enable NER for these papers", exact: true }).click();
  await expect(page.getByRole("combobox", { name: "Annotation task", exact: true })).toHaveValue("3");
  expect(JSON.stringify(state.versions.find((version) => version.id === 2))).toBe(state.initialTokenVersion);
  expect(state.mutations.filter((request) => request.path === "/api/tasks/versions")).toHaveLength(1);
  expect(state.versions.find((version) => version.id === 3)).toMatchObject({ task_definition_id: 2, version_number: 2, annotation_ui: { preset: "document_entities", offset_unit: "utf16_code_unit" } });
  await page.getByRole("button", { name: "Start annotating", exact: true }).click();
  await expect(page).toHaveURL(/\/my-work\/rounds\/2/);
  expect(state.rounds.find((round) => round.id === 2)).toMatchObject({ dataset_version_id: 1, task_version_id: 3, assistance_policy: "blind", annotator_user_ids: [2] });

  // Both entry routes must render the established paper editor and its two tasks.
  await page.goto("/my-work?project=1");
  await expectUnifiedMyWork(page);
  await expect(taskInventory.getByText("Ready for annotation", { exact: true })).toHaveCount(2);
  const firstPaper = page.getByRole("button", { name: /Tucatinib study 31825569/ });
  await expect(firstPaper).toContainText("Paper relevance: To do");
  await expect(firstPaper).toContainText("Drug entities: To do");
  await expect(firstPaper).toBeInViewport();
  await page.screenshot({ path: testInfo.outputPath("my-work-desktop.png"), fullPage: false });
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: /Tucatinib study 31825569/ }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Paper relevance", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Drug entities", exact: true })).toBeVisible();
  await page.getByRole("button", { name: /^Relevant/ }).click();
  await expect.poll(() => state.decisions.filter((decision) => decision.round_item_id === 1001).length).toBe(1);
  await selectDrugText(page);
  await page.getByRole("button", { name: /^Drug/ }).last().click();
  await expect.poll(() => state.decisions.filter((decision) => decision.round_item_id === 2001).length).toBe(1);
  expect(state.decisions.find((decision) => decision.round_item_id === 1001)?.output).toBe("Relevant");
  expect(state.decisions.find((decision) => decision.round_item_id === 2001)?.output).toEqual({ entities: [{ start: 0, end: 9, label: "Drug" }] });

  await page.getByRole("button", { name: /^Back to My Work/ }).click();
  await expectUnifiedMyWork(page);
  await expect(firstPaper).toContainText("Paper relevance: Draft");
  await expect(firstPaper).toContainText("Drug entities: Draft");
  await expect(page.locator(".paper-progress-summary")).toContainText("0 / 2");
  await firstPaper.click();
  await page.reload();
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Tucatinib study");
  await expect(page.getByRole("button", { name: /^Relevant/ })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".aw-entity").filter({ hasText: "Tucatinib" })).toBeVisible();
  let expectedSubmissions = 0;
  for (const assignmentId of ["1001", "2001"]) {
    await page.getByRole("combobox", { name: "Task", exact: true }).selectOption(assignmentId);
    await page.getByRole("button", { name: "Submit task for this paper", exact: true }).click();
    await page.getByRole("dialog").getByRole("button", { name: "Submit this paper task", exact: true }).click();
    expectedSubmissions += 1;
    await expect.poll(() => state.submissions.length).toBe(expectedSubmissions);
    if (assignmentId === "1001") {
      // A submitted classification stays selected while NER remains editable.
      await expect(page.getByRole("combobox", { name: "Task", exact: true })).toHaveValue("1001");
      await expect(page.getByRole("button", { name: /^Drug/ }).first()).toBeEnabled();
      await page.getByRole("button", { name: "Edit this paper task", exact: true }).click();
      await page.getByRole("dialog").getByRole("button", { name: "Reopen and edit", exact: true }).click();
      await expect(page.getByRole("button", { name: "Submit task for this paper", exact: true })).toBeEnabled();
      const classificationHistory = state.decisions.filter((decision) => decision.round_item_id === 1001);
      expect(classificationHistory).toHaveLength(2);
      expect(classificationHistory[1]).toMatchObject({ output: "Relevant", supersedes_decision_id: classificationHistory[0].id });
      expect(state.submissions[0].decision_ids).toEqual([classificationHistory[0].id]);
      await page.getByRole("button", { name: "Submit task for this paper", exact: true }).click();
      await page.getByRole("dialog").getByRole("button", { name: "Submit this paper task", exact: true }).click();
      expectedSubmissions += 1;
      await expect.poll(() => state.submissions.length).toBe(expectedSubmissions);
    }
  }
  expect(state.submissions.map((submission) => submission.annotation_round_id).sort()).toEqual([1, 1, 2]);
  expect(state.mutations.some((request) => request.path === "/api/annotations")).toBe(false);
  expect(state.mutations.some((request) => request.path.includes("/versions/project-corpus"))).toBe(false);

  await page.goto("/my-work?project=1");
  await expectUnifiedMyWork(page);
  await expect(firstPaper).toContainText("Paper relevance: Submitted");
  await expect(firstPaper).toContainText("Drug entities: Submitted");
  await expect(page.locator(".paper-progress-summary")).toContainText("1 / 2");
  await page.goto("/my-work/rounds/1?view=annotate&document=41");
  await expect(page.getByRole("heading", { name: "Drug entities", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Edit this paper task", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("classification-and-ner-saved.png"), fullPage: true });
  const accessibility = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(accessibility.violations).toEqual([]);

  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("combobox", { name: "Task", exact: true }).selectOption("2001");
  await expect(page.getByRole("combobox", { name: "Task", exact: true })).toHaveValue("2001");
  await page.getByRole("tab", { name: "Tools", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "Paper relevance", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Drug entities", exact: true })).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze()).violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("classification-and-ner-mobile-tools.png"), fullPage: true });
  await page.getByRole("tab", { name: "Review", exact: true }).click();
  await expect(page.getByRole("tabpanel", { name: "Review", exact: true })).toContainText("Tucatinib");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze()).violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("classification-and-ner-mobile-review.png"), fullPage: true });
  await page.getByRole("button", { name: /^Back to My Work/ }).click();
  await expectUnifiedMyWork(page);
  await expect(firstPaper).toContainText("Paper relevance: Submitted");
  await expect(firstPaper).toContainText("Drug entities: Submitted");
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze()).violations).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("my-work-mobile.png"), fullPage: true });
  await page.goto("/projects/1/overview");
  await expect(page.getByRole("row").filter({ hasText: "Drug entities" })).toContainText("1 open round");
  await expect(page.locator(".platform-stats > div").filter({ has: page.getByText("Annotation tasks", { exact: true }) }).locator("dd").first()).toHaveText("2");
});

test("paper deep links retain the second paper through loading and reload", async ({ page }) => {
  const state = await installPaperAnnotationMock(page);
  await page.goto("/my-work?project=1&view=annotate&document=42");
  await expect(page.getByRole("heading", { level: 1, name: "Second study", exact: true })).toBeVisible();
  await expect(page.getByRole("tabpanel", { name: "Document", exact: true })).toContainText("No eligible drug treatment was reported.");
  await page.getByRole("button", { name: /^Relevant/ }).click();
  await expect.poll(() => state.decisions.length).toBe(1);
  expect(state.decisions[0]).toMatchObject({ round_item_id: 1002, output: "Relevant" });
  await page.reload();
  await expect(page.getByRole("heading", { level: 1, name: "Second study", exact: true })).toBeVisible();
  await expect(page).toHaveURL(/document=42/);
  await expect(page.getByRole("button", { name: /^Relevant/ })).toHaveAttribute("aria-pressed", "true");
  await page.goto("/my-work/rounds/1?view=annotate&document=42");
  await expect(page.getByRole("heading", { level: 1, name: "Second study", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /^Relevant/ })).toHaveAttribute("aria-pressed", "true");
  expect(state.decisions).toHaveLength(1);
});
