import { expect, test, type Locator, type Page } from "@playwright/test";

import {
  installPlatformApiMock,
  type PlatformApiRequest,
} from "./fixtures/mockPlatformApi";

test.setTimeout(90_000);

const publicRevision = "2fdd8b9bcadd6e7055e742a7069a522d8f4cce56";

function requestFor(
  requests: PlatformApiRequest[],
  pathname: string,
): PlatformApiRequest {
  const request = requests.find(
    (item) => item.method === "POST" && item.pathname === pathname,
  );
  expect(request, `Expected POST ${pathname}`).toBeDefined();
  return request as PlatformApiRequest;
}

async function submitDialog(dialog: Locator): Promise<void> {
  await dialog.getByRole("button", { name: "Create", exact: true }).click();
  await expect(dialog).toBeHidden();
}

async function selectTrainingOnlyContext(page: Page): Promise<void> {
  await page
    .getByRole("radio", { name: /Training-only project/ })
    .check();
  await page
    .getByRole("combobox", { name: "Existing training-only project" })
    .selectOption("2");
  await page
    .getByRole("button", { name: "Use selected project" })
    .click();
}

test("individual owner completes standalone public-dataset training with immutable model lineage", async ({
  page,
}) => {
  const api = await installPlatformApiMock(page, {
    role: "personal",
    trainingReleaseGate: true,
  });

  await page.goto("/training/new");
  await expect(
    page.getByRole("heading", { level: 1, name: "New training" }),
  ).toBeVisible();

  await page
    .getByRole("radio", { name: /Training-only project/ })
    .check();
  await page.getByRole("textbox", { name: "Project name" }).fill(
    "Public benchmark training",
  );
  await page.getByRole("textbox", { name: "Description" }).fill(
    "Standalone IMDb training with pinned public provenance.",
  );
  await page.getByRole("button", { name: "Create project" }).click();

  await expect(
    page.getByText(/Project context: Public benchmark training/),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Prepare training data" })
    .click();

  await expect(page).toHaveURL("/training/data?projectId=2");
  await expect(
    page.getByRole("heading", { level: 1, name: "Training data" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Add dataset" }).first().click();

  let dialog = page.getByRole("dialog", { name: "Add dataset version" });
  await dialog.getByRole("textbox", { name: "Name" }).fill(
    "IMDb pinned benchmark",
  );
  await dialog.getByRole("combobox", { name: "Source" }).selectOption(
    "public_registry",
  );
  await dialog.getByRole("textbox", { name: "Hugging Face dataset" }).fill(
    "stanfordnlp/imdb",
  );
  await dialog.getByRole("combobox", { name: "Format" }).selectOption("jsonl");
  await dialog.getByRole("textbox", { name: "Exact revision" }).fill(
    publicRevision,
  );
  await dialog.getByRole("textbox", { name: "License identifier" }).fill(
    "apache-2.0",
  );
  await dialog
    .getByLabel("Pinned dataset snapshot")
    .setInputFiles({
      name: "imdb-pinned.jsonl",
      mimeType: "application/json",
      buffer: Buffer.from(
        '{"id":"train-1","text":"A precise review.","label":"positive"}\n',
      ),
    });
  await submitDialog(dialog);

  const datasetRegistry = page.getByRole("region", {
    name: "Dataset registry table",
  });
  await expect(datasetRegistry).toContainText("IMDb pinned benchmark");
  await expect(datasetRegistry).toContainText("Public Registry");
  await expect(datasetRegistry).toContainText("50000");

  await page.getByRole("button", { name: "Define task" }).click();
  dialog = page.getByRole("dialog", { name: "Define NLP task" });
  await dialog.getByRole("textbox", { name: "Name" }).fill("IMDb sentiment");
  await dialog
    .getByRole("textbox", { name: "Stable key" })
    .fill("imdb_sentiment");
  await dialog
    .getByRole("textbox", { name: "Allowed labels" })
    .fill("negative, positive");
  await submitDialog(dialog);

  await datasetRegistry
    .getByRole("button", { name: "Prepare training" })
    .click();
  dialog = page.getByRole("dialog", { name: "Prepare training data" });
  await dialog
    .getByRole("textbox", { name: "Training dataset name" })
    .fill("IMDb sentiment training set");
  await expect(
    dialog.getByRole("combobox", { name: "Task version" }),
  ).toHaveValue("212");
  await expect(
    dialog.getByRole("combobox", { name: "Dataset version" }),
  ).toHaveValue("222");
  await expect(
    dialog.getByRole("radio", { name: /Dataset field/ }),
  ).toBeChecked();
  await expect(dialog).toContainText(
    "protected from training, selection, prompt tuning, and guideline mining",
  );
  await submitDialog(dialog);

  await expect(
    page.getByRole("region", { name: "Composed training datasets" }),
  ).toContainText("IMDb sentiment training set");

  await page
    .getByRole("navigation", { name: "Training sections" })
    .getByRole("link", { name: "New training" })
    .click();
  await expect(page).toHaveURL("/training/new");
  await selectTrainingOnlyContext(page);

  await expect(
    page.getByRole("button", {
      name: "Step 1 of 5: Task & Data",
      exact: true,
    }),
  ).toHaveAttribute("aria-current", "step");
  await expect(
    page.getByRole("combobox", { name: "Task version" }),
  ).toHaveValue("212");
  await expect(
    page.getByRole("combobox", { name: "Training dataset" }),
  ).toHaveValue("251");
  await page.getByRole("button", { name: "Continue" }).click();

  await page
    .getByRole("textbox", { name: "Model name" })
    .fill("IMDb sentiment linear");
  await expect(
    page.getByRole("combobox", { name: "Training recipe" }),
  ).toHaveValue("tfidf_logistic_regression");
  await expect(page.locator(".platform-recipe-summary")).toContainText(
    "TF-IDF logistic regression",
  );
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(
    page.getByRole("radio", { name: /Verified classical CPU/ }),
  ).toBeChecked();
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(
    page.getByRole("combobox", { name: "Storage policy" }),
  ).toHaveValue("341");
  await expect(
    page.getByRole("combobox", { name: "Evaluation split" }),
  ).toHaveValue("test");
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(page.getByText("Review & Launch", { exact: true })).toBeVisible();
  await expect(page.getByText("IMDb sentiment linear", { exact: true }))
    .toBeVisible();
  await expect(
    page.getByText("IMDb sentiment training set · source v1", {
      exact: true,
    }),
  ).toBeVisible();
  const reviewGrid = page.locator(".platform-review-grid");
  await expect(reviewGrid).toContainText("Verified classical CPU");
  await expect(reviewGrid).toContainText("Immutable MinIO artifacts");
  await page.getByRole("button", { name: "Launch training" }).click();

  await expect(page).toHaveURL("/training/runs/252");
  await expect(
    page.getByRole("heading", { level: 1, name: "Training run 252" }),
  ).toBeVisible();
  const runManifest = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Run manifest" }),
  });
  await expect(runManifest).toContainText("Public benchmark training");
  await expect(runManifest).toContainText("IMDb sentiment linear");
  await expect(runManifest).toContainText("Linear");
  await expect(runManifest).toContainText("scikit-learn");
  await expect(runManifest).toContainText("IMDb sentiment training set");
  await expect(runManifest).toContainText("source v1");
  await expect(runManifest).toContainText("Verified classical CPU");
  await expect(runManifest).toContainText("Immutable MinIO artifacts");
  await expect(runManifest).toContainText("Succeeded · test");
  await expect(runManifest).toContainText("Alice Owner");

  await page
    .getByRole("navigation", { name: "Primary" })
    .getByRole("link", { name: "Models", exact: true })
    .click();
  const registry = page.getByRole("region", {
    name: "Workspace model registry",
  });
  await expect(registry).toContainText("IMDb sentiment linear");
  await expect(registry).toContainText("Public benchmark training");
  await registry
    .getByRole("link", { name: "IMDb sentiment linear" })
    .click();

  await expect(page).toHaveURL("/models/281");
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "IMDb sentiment linear",
    }),
  ).toBeVisible();
  const versions = page.getByRole("region", {
    name: "IMDb sentiment linear versions",
  });
  await expect(versions).toContainText("v1");
  await expect(versions).toContainText("IMDb sentiment training set");
  await expect(versions).toContainText("Source dataset v1");
  await expect(versions).toContainText("Verified classical CPU");
  await expect(versions).toContainText("Immutable MinIO artifacts");
  await versions.getByRole("link", { name: "v1" }).click();

  await expect(page).toHaveURL("/models/281/versions/282");
  const lineage = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Version lineage" }),
  });
  await expect(lineage).toContainText("Public benchmark training");
  await expect(lineage).toContainText("IMDb sentiment linear");
  await expect(lineage).toContainText("IMDb sentiment");
  await expect(lineage).toContainText("IMDb sentiment training set");
  await expect(lineage).toContainText("Source dataset v1");
  await expect(lineage).toContainText("Verified classical CPU");
  await expect(lineage).toContainText("Immutable MinIO artifacts");
  await expect(lineage).toContainText("252");
  await expect(lineage).toContainText("Artifact package 481");
  await expect(lineage).toContainText("Succeeded");

  expect(requestFor(api.requests, "/api/projects").body).toMatchObject({
    workspace_id: 1,
    settings: {
      modules: ["data", "train", "models", "activity"],
      project_purpose: "training_only",
    },
  });
  expect(
    api.requests.some(
      (request) =>
        request.method === "POST" &&
        request.pathname ===
          "/api/projects/2/datasets/221/versions/public-registry-snapshot",
    ),
  ).toBe(true);
  expect(
    requestFor(
      api.requests,
      "/api/datasets/training-versions/compose",
    ).body,
  ).toMatchObject({
    project_id: 2,
    dataset_version_id: 222,
    task_version_id: 212,
    input_field: "text",
    label_field: "label",
    train_percent: 80,
    validation_percent: 10,
  });
  expect(requestFor(api.requests, "/api/models").body).toMatchObject({
    project_id: 2,
    name: "IMDb sentiment linear",
  });
  expect(requestFor(api.requests, "/api/training-runs").body).toMatchObject({
    project_id: 2,
    registered_model_id: 281,
    task_version_id: 212,
    training_dataset_version_id: 251,
    recipe_version_id: 322,
    environment_id: 331,
    storage_policy_id: 341,
    evaluation_plan: { splits: ["test"] },
    seed: 42,
  });
});

test("annotation project overview deep-links into the independent Training workspace", async ({
  page,
}) => {
  await installPlatformApiMock(page, { role: "trainer" });
  await page.goto("/projects/1/overview");

  const modelDevelopment = page.locator("section").filter({
    has: page.getByRole("heading", { name: "Model development" }),
  });
  await modelDevelopment.getByRole("link", { name: /Training/ }).click();

  await expect(page).toHaveURL("/training?projectId=1");
  await expect(
    page.getByRole("heading", { level: 1, name: "Training" }),
  ).toBeVisible();
  await expect(
    page.getByRole("combobox", { name: "Project" }),
  ).toHaveValue("1");
  await expect(
    page.getByRole("region", { name: "Workspace training runs" }),
  ).toContainText("Evidence Review Demo");
  await expect(
    page.getByRole("region", { name: "Workspace training runs" }),
  ).toContainText("Relevance training set");
});
