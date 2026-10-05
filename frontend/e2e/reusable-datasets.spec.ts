import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";

const timestamp = "2026-09-14T12:00:00Z";
const hash = "a".repeat(64);
const base = { project_id: 1, created_at: timestamp, updated_at: timestamp, created_by_user_id: 2 };

async function installReusableMock(page: Page, populated = false) {
  const existing = await installPlatformApiMock(page, { role: "personal" });
  const datasets: Record<string, unknown>[] = populated ? [{ ...base, id: 21, name: "Tucatinib papers", source_type: "project_corpus", purposes: ["annotation", "inference"] }] : [];
  const versions: Record<string, unknown>[] = populated ? [{ ...base, id: 22, dataset_id: 21, version_number: 1, item_count: 3, content_hash: hash, data_schema: {}, provenance: { source_document_ids: [1, 2, 3] }, source_format: "other", source_revision: hash, license_info: {} }] : [];
  const training: Record<string, unknown>[] = [];
  const mutations: Array<{ path: string; body: Record<string, unknown> }> = [];
  const runs: Record<string, unknown>[] = [];
  const rounds: Record<string, unknown>[] = [];
  await page.route(/\/api\//, async (route) => {
    const req = route.request(); const url = new URL(req.url()); const path = url.pathname;
    const body = req.method() === "POST" ? (req.postDataJSON() ?? {}) as Record<string, unknown> : {};
    const json = (value: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(value) });
    if (req.method() === "POST") mutations.push({ path, body });
    if (path === "/api/workspaces/1/capabilities") return json({ preset: "annotate_train", overrides: [], effective: ["annotation", "training", "inference", "export", "lineage"], blocked: {} });
    if (path === "/api/projects/1/modules") return json({ project_id: 1, selected: ["data", "annotate", "train", "models", "activity"], effective: ["data", "annotate", "train", "models", "activity"], workspace_capabilities: ["annotation", "training", "inference", "export", "lineage"] });
    if (req.method() === "GET") {
      if (path === "/api/datasets") return json(datasets);
      if (path === "/api/datasets/versions") {
        if (!url.searchParams.has("project_id") || !url.searchParams.has("dataset_id")) {
          return json({ detail: "project_id and dataset_id are required" }, 422);
        }
        return json(versions.filter((version) => version.project_id === Number(url.searchParams.get("project_id")) && version.dataset_id === Number(url.searchParams.get("dataset_id"))));
      }
      if (path === "/api/datasets/training-versions") return json(training);
      if (path === "/api/datasets/label-sets" || path === "/api/datasets/split-maps") return json([]);
      if (path === "/api/rounds") return json(rounds);
      if (path === "/api/projects/1/training-datasets") return json([]);
      if (path === "/api/projects/1/datasets/annotation-progress") return json(versions.map((version) => ({ dataset_id: version.dataset_id, dataset_version_id: version.id, total: version.item_count, submitted: 0 })));
    }
    if (path === "/api/projects/1/import/pubmed/preview") return json({ items: [
      { pmid: "12345678", title: "Tucatinib full text", status: "full_text", has_full_text: true, has_abstract: true, pmcid: "PMC123" },
      { pmid: "23456789", title: "Tucatinib abstract", status: "abstract_only", has_full_text: false, has_abstract: true },
      { pmid: "34567890", title: "Unavailable article", status: "error", has_full_text: false, has_abstract: false, error: "No abstract or full text available" },
    ] });
    if (path === "/api/projects/1/import/pubmed") return json({ created: [{ pmid: "12345678", document_id: 1, title: "Tucatinib full text", source: "pmc" }], skipped: [{ pmid: "23456789", document_id: 2, title: "Tucatinib abstract", source: "pubmed_abstract", reason: "Duplicate: already in this project; available to reuse" }], failed: [] });
    if (path === "/api/datasets" && req.method() === "POST") {
      const dataset = { ...base, ...body, id: 21 }; datasets.push(dataset); return json(dataset, 201);
    }
    if (path === "/api/projects/1/datasets/21/versions/project-corpus") {
      const version = { ...base, id: 22, dataset_id: 21, version_number: 1, item_count: (body.document_ids as number[]).length, content_hash: hash, data_schema: {}, provenance: { source_document_ids: body.document_ids }, source_format: "other", source_revision: hash, license_info: {} };
      versions.push(version); return json(version, 201);
    }
    if (path === "/api/rounds" && req.method() === "POST") rounds.push({ ...base, ...body, id: 78, status: "open", sequence: 1 });
    const preview = { ready: true, issues: [], source_counts: [{ dataset_version_id: 22, total_count: 3, labeled_count: 3, excluded_unlabeled_count: 0 }], input_count: 3, labeled_count: 3, excluded_unlabeled_count: 0, duplicate_count: 0, item_count: 3, group_count: 3, split_counts: { train: 1, validation: 1, test: 1, pool: 0 }, manifest_hash: hash, resolved_sources: body.sources };
    if (path === "/api/projects/1/training-datasets/preview") return json(preview);
    if (path === "/api/projects/1/training-datasets/prepare") {
      const version = { ...base, id: 51, training_dataset_id: 50, version_number: 1, parent_version_id: null, name: body.name, task_version_id: body.task_version_id, dataset_version_id: 22, label_set_version_ids: [31], split_map_id: 41, composition: [], preprocessing: {}, content_hash: hash, preparation_manifest: { sources: body.sources, preview } };
      training.push(version); return json({ training_dataset: { ...base, id: 50, name: body.name, task_version_id: body.task_version_id }, training_dataset_version: version, preview }, 201);
    }
    if (path === "/api/projects/1/prediction-runs") {
      if (req.method() === "GET") return json(runs);
      const run = { ...base, ...body, id: 91, task_version_id: 12, status: "completed", result_count: 3, failure_reason: null, completed_at: timestamp }; runs.push(run); return json(run, 201);
    }
    if (path === "/api/projects/1/prediction-runs/91/results") return json({ total: 3, offset: 0, limit: 50, items: [
      { dataset_item_id: 221, stable_key: "pmid:12345678", title: "Tucatinib full text", text: "Treatment results", prediction: "include", confidence: 0.91, uncertainty: 0.09, already_submitted: false, protected: false },
      { dataset_item_id: 222, stable_key: "pmid:23456789", title: "Tucatinib abstract", text: "Abstract results", prediction: "exclude", confidence: null, uncertainty: null, already_submitted: true, protected: false },
      { dataset_item_id: 223, stable_key: "pmid:34567890", title: "Protected evaluation", text: "Evaluation results", prediction: "include", confidence: 0.8, uncertainty: 0.2, already_submitted: false, protected: true },
    ] });
    if (path === "/api/projects/1/prediction-runs/91/download") return route.fulfill({ contentType: "text/csv", headers: { "Content-Disposition": 'attachment; filename="predictions-91.csv"' }, body: "run_id,dataset_item_id,model_version_id,prediction\n91,221,82,include\n" });
    if (path === "/api/projects/1/prediction-runs/91/review-round") return json({ round_id: 71, item_count: 1, excluded_submitted: 0, excluded_protected: 0 }, 201);
    return route.fallback();
  });
  return { ...existing, mutations, datasets, versions, training };
}

async function audit(page: Page) {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(result.violations).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
}

test("PMID import includes abstracts and reused papers, then pins annotation to the named source", async ({ page }, testInfo) => {
  const state = await installReusableMock(page);
  await page.goto("/projects/1/data?tab=source");
  await page.getByRole("button", { name: "Import PMIDs", exact: true }).first().click();
  await page.getByLabel("Collection name").fill("Tucatinib annotation papers");
  await page.getByLabel("PMIDs", { exact: true }).fill("12345678 23456789 34567890");
  await page.getByRole("button", { name: "Preview documents" }).focus();
  await page.keyboard.press("Enter");
  await expect(page.getByLabel("Select PMID 12345678")).toBeChecked();
  await expect(page.getByLabel("Select PMID 23456789")).toBeChecked();
  await expect(page.getByLabel("Select PMID 34567890")).toBeDisabled();
  await page.getByRole("button", { name: "Import 2 selected documents" }).click();
  await expect(page.getByRole("button", { name: "Create source dataset" })).toBeEnabled();
  await page.getByRole("button", { name: "Create source dataset" }).click();
  await expect(page.getByRole("heading", { name: "Choose the next action" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Start annotating", exact: true })).toBeEnabled();
  expect(state.mutations.find((m) => m.path.endsWith("/import/pubmed"))?.body).toMatchObject({ pmids: ["12345678", "23456789"], include_abstract_only: true });
  expect(state.mutations.find((m) => m.path.endsWith("/versions/project-corpus"))?.body).toEqual({ document_ids: [1, 2] });
  await audit(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("source-next-action.png"), fullPage: true });
  await page.reload();
  await expect(page.getByRole("button", { name: "Start annotating" })).toBeVisible();
  await page.getByRole("button", { name: "Start annotating" }).click();
  await expect(page).toHaveURL(/\/my-work\/rounds\/78/);
  expect(state.mutations.find((m) => m.path === "/api/rounds")?.body).toMatchObject({ dataset_version_id: 22, task_version_id: 12, assistance_policy: "blind", annotator_user_ids: [2] });
});

test("training preparation previews external labels and hands the immutable version to training", async ({ page }, testInfo) => {
  const state = await installReusableMock(page, true);
  await page.goto("/training/data?projectId=1&flow=prepare&datasetVersionId=22&taskVersionId=12");
  await page.getByLabel("Training dataset name").fill("Tucatinib training");
  await page.getByRole("button", { name: "Preview training dataset" }).click();
  await expect(page.getByRole("region", { name: "Training dataset preview" })).toBeVisible();
  await audit(page);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("training-preview.png"), fullPage: true });
  await page.getByRole("region", { name: "Training dataset preview" }).getByRole("button", { name: "Create training dataset" }).click();
  await expect(page).toHaveURL(/tab=training/);
  expect(state.mutations.find((m) => m.path.endsWith("/prepare"))?.body).toMatchObject({ task_version_id: 12, sources: [{ dataset_version_id: 22, label_field: "label", input_mapping: { text: "text" } }], preview_manifest_hash: hash });
  await page.getByRole("button", { name: "Train model", exact: true }).click();
  await expect(page).toHaveURL(/trainingDatasetVersionId=51/);
});

test("unlabeled inference without Active Learning exports and reviews selected original papers", async ({ page }, testInfo) => {
  const state = await installReusableMock(page, true);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto("/projects/1/inference?datasetVersionId=22");
  await expect(page.getByRole("heading", { name: "Inference", exact: true })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "Source dataset", exact: true })).toHaveValue("22");
  // The setup has explicit review steps; each next action can be keyboard activated.
  await page.getByRole("button", { name: "Continue to model" }).click();
  await page.getByLabel("Trained model").selectOption("82");
  await page.getByLabel("Prediction name").fill("Tucatinib predictions");
  await page.getByRole("button", { name: "Review predictions" }).click();
  await page.getByRole("button", { name: "Run predictions", exact: true }).click();
  await expect(page.getByRole("region", { name: "Prediction results", exact: true })).toBeVisible();
  await expect(page.getByLabel("Review pmid:23456789")).toBeDisabled();
  await expect(page.getByLabel("Review pmid:34567890")).toBeDisabled();
  await audit(page);
  for (const button of await page.locator(".platform-prediction-pagination").getByRole("button").all()) {
    const bounds = await button.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.x).toBeGreaterThanOrEqual(0);
    expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(375);
  }
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: testInfo.outputPath("inference-results.png"), fullPage: true });
  await page.context().route(/\/api\/projects\/1\/prediction-runs\/91\/download/, (route) => route.fulfill({ contentType: "text/csv", headers: { "Content-Disposition": 'attachment; filename="predictions-91.csv"' }, body: "run_id,dataset_item_id,model_version_id,prediction\n91,221,82,include\n" }));
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download CSV" }).click();
  expect((await download).suggestedFilename()).toBe("predictions-91.csv");
  await page.getByLabel("Review pmid:12345678").check();
  await page.getByLabel("Review round name").fill("Tucatinib manual review");
  await page.getByRole("button", { name: "Create review round" }).click();
  await expect(page).toHaveURL(/\/my-work\/rounds\/71/);
  expect(state.mutations.find((m) => m.path.endsWith("/review-round"))?.body).toMatchObject({ dataset_item_ids: [221], include_submitted: false });
  expect(state.mutations.some((m) => m.path.includes("selection-run"))).toBe(false);
});
