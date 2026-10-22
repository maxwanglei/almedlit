import { expect, test, type Page } from "@playwright/test";
import { execFileSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { installEvidenceApiMock } from "./fixtures/mockEvidenceApi";
import { installPlatformApiMock } from "./fixtures/mockPlatformApi";
import accessCases from "./fixtures/backendAccessCases.json";
import contract from "./fixtures/backendContract.json";
import type { CapabilityPreset, ProjectModule } from "./fixtures/mockAccess";

function isPreset(value: string): value is CapabilityPreset {
  return value === "custom" || Object.hasOwn(contract.policy.presets, value);
}

function isModule(value: string): value is ProjectModule {
  return Object.hasOwn(contract.policy.module_requirements, value);
}

function numericField(body: unknown, key: string): number {
  if (body !== null && typeof body === "object" && key in body) {
    const value = Reflect.get(body, key);
    if (typeof value === "number" && Number.isInteger(value)) return value;
  }
  throw new Error(`Missing integer ${key} in fixture response`);
}

async function openProbe(page: Page): Promise<void> {
  await page.route("**/__fixture_probe", (route) => route.fulfill({
    contentType: "text/html",
    body: "<!doctype html><title>API fixture probe</title>",
  }));
  await page.goto("/__fixture_probe");
}

async function api(page: Page, path: string, body?: object) {
  return page.evaluate(async ({ path, body }) => {
    const response = await fetch(`/api${path}`, body === undefined ? undefined : {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const responseBody: unknown = await response.json();
    return { status: response.status, body: responseBody };
  }, { path, body });
}

test("annotators cannot read trainer-only project details", async ({ page }) => {
  await installEvidenceApiMock(page, "annotator");
  await openProbe(page);
  expect((await api(page, "/projects/1")).status).toBe(403);
});

test("trainers cannot create manager-owned rounds", async ({ page }) => {
  await installPlatformApiMock(page, { role: "trainer" });
  await openProbe(page);
  const response = await api(page, "/rounds", {
    project_id: 1, name: "Forbidden round", dataset_version_id: 22,
    task_version_id: 12, assistance_policy: "blind", reannotation_mode: "full_dataset",
    annotator_user_ids: [2], open_to_all_annotators: false,
  });
  expect(response.status).toBe(403);
});

test("data screen APIs have handlers and report actual source progress", async ({ page }) => {
  const fixture = await installPlatformApiMock(page, { role: "manager" });
  await openProbe(page);
  const progress = await api(page, "/projects/1/datasets/annotation-progress");
  expect(progress.status).toBe(200);
  expect(progress.body).toEqual(expect.arrayContaining([
    expect.objectContaining({ dataset_id: 21, dataset_version_id: 22, total: 240 }),
  ]));
  const predictions = await api(page, "/projects/1/prediction-runs");
  expect(predictions.status).toBe(200);
  expect(Array.isArray(predictions.body)).toBe(true);
  expect(fixture.evidence.unhandledRequests).toEqual([]);
});

test("round work context validates workspace and missing identities", async ({ page }) => {
  await installPlatformApiMock(page, { role: "manager", includeSecondWorkspace: true });
  await openProbe(page);
  expect((await api(page, "/rounds/71/work-context?workspace_id=1")).status).toBe(200);
  expect((await api(page, "/rounds/71/work-context?workspace_id=2")).status).toBe(404);
  expect((await api(page, "/rounds/9999/work-context")).status).toBe(404);
});

test("project summaries, presets and retention values match backend vocabulary", async ({ page }) => {
  await installPlatformApiMock(page, { role: "manager", includeSecondWorkspace: true });
  await openProbe(page);
  const projects = await api(page, "/projects?workspace_id=1");
  expect(projects.body).toEqual(expect.arrayContaining([
    expect.objectContaining({ workflow_task_count: 1 }),
  ]));
  for (const workspaceId of [1, 2]) {
    const capabilities = await api(page, `/workspaces/${workspaceId}/capabilities`);
    expect(capabilities.body).toMatchObject({
      preset: expect.stringMatching(/^(annotate|train|annotate_train|annotate_train_al|full|custom)$/),
    });
  }
  const storage = await api(page, "/storage/policies?project_id=1");
  expect(storage.body).toEqual(expect.arrayContaining([
    expect.objectContaining({ retention_class: "indefinite" }),
  ]));
});

test("creating a project returns backend HTTP 200", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal", trainingReleaseGate: true });
  await openProbe(page);
  const response = await api(page, "/projects", {
    name: "Training probe", workspace_id: 1, annotation_schema: { labels: {} },
    settings: { modules: ["data", "train", "models", "activity"] },
  });
  expect(response.status).toBe(200);
});

for (const scenario of accessCases) {
  test(`backend access agreement: ${scenario.name}`, async ({ page }) => {
    const role = scenario.role;
    if (role !== "annotator" && role !== "trainer" && role !== "manager" && role !== "personal") {
      throw new Error(`Unknown role in backend access case: ${role}`);
    }
    const preset = scenario.preset;
    const modules = scenario.modules;
    if (!isPreset(preset) || (modules !== null && !modules.every(isModule))) {
      throw new Error(`Invalid preset or modules in backend access case: ${scenario.name}`);
    }
    await installPlatformApiMock(page, {
      role, capabilityPreset: preset, projectModules: modules,
      actorUserId: "actorUserId" in scenario ? scenario.actorUserId : undefined,
      includeSecondWorkspace: scenario.path.includes("{other_workspace}"),
    });
    await openProbe(page);
    const path = scenario.path.replaceAll("{project}", "1").replaceAll("{workspace}", "1")
      .replaceAll("{round}", "71").replaceAll("{other_workspace}", "2");
    const response = await api(page, path);
    expect(response.status).toBe(scenario.status);
    if ("expected_body" in scenario) expect(response.body).toEqual(scenario.expected_body);
  });
}

test("created projects retain requested modules and project lists retain current round counts", async ({ page }) => {
  await installPlatformApiMock(page, { role: "personal", trainingReleaseGate: true });
  await openProbe(page);
  expect((await api(page, "/projects", {
    name: "Annotation only", workspace_id: 1, annotation_schema: { labels: {} },
    settings: { modules: ["annotate"] },
  })).status).toBe(200);
  const modules = await api(page, "/projects/2/modules");
  expect(modules.body).toMatchObject({ selected: ["data", "annotate"], effective: ["data", "annotate"] });
  expect((await api(page, "/training-runs?project_id=2")).status).toBe(403);
  expect((await api(page, "/rounds", {
    project_id: 1, name: "Another round", dataset_version_id: 22, task_version_id: 12,
    assistance_policy: "blind", reannotation_mode: "full_dataset", annotator_user_ids: [2],
  })).status).toBe(201);
  for (const path of ["/projects?workspace_id=1", "/projects/my-work?workspace_id=1"]) {
    expect((await api(page, path)).body).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: 1, workflow_round_count: 2 }),
    ]));
  }
});

test("assignment visibility and submission follow actor identity", async ({ page }) => {
  await installEvidenceApiMock(page, "annotator", { actorUserId: 4, capabilityPreset: "annotate" });
  await openProbe(page);
  expect((await api(page, "/projects/1/assignments")).body).toEqual([]);
  expect((await api(page, "/projects/my-work?workspace_id=1")).body).toEqual([]);
  expect((await api(page, "/auth/me")).body).not.toMatchObject({ user: { username: "alice" } });
  expect((await api(page, "/projects/1/assignments?annotator_id=alice")).status).toBe(403);
  expect((await api(page, "/projects/1/documents/41/submissions", { assignment_id: 51 })).status).toBe(403);
});

test("captured fixture responses validate against real backend models", async ({ page }, testInfo) => {
  await installPlatformApiMock(page, { role: "personal", trainingReleaseGate: true });
  await openProbe(page);
  const captures = [];
  for (const path of [
    "/projects?workspace_id=1", "/projects/my-work?workspace_id=1", "/projects/1",
    "/projects/1/datasets/annotation-progress", "/projects/1/prediction-runs",
    "/rounds/71/work-context?workspace_id=1", "/storage/policies?project_id=1",
    "/workspaces/1/capabilities", "/workspaces/1/models", "/workspaces/1/training-contexts",
    "/workspaces/1/training-runs",
  ]) captures.push({ method: "GET", path: `/api${path}`, ...await api(page, path) });
  expect((await api(page, "/projects/my-work?workspace_id=1")).body).toEqual(expect.arrayContaining([
    expect.objectContaining({ id: 1, settings: {} }),
  ]));
  captures.push({ method: "POST", path: "/api/projects", ...await api(page, "/projects", {
    name: "Contract project", workspace_id: 1, annotation_schema: { labels: {} },
    settings: { modules: ["data", "train", "models", "activity"] },
  }) });
  const predictionPath = "/projects/1/prediction-runs";
  const predictionBody = { name: "Schema-checked predictions", dataset_version_id: 22,
    model_version_id: 82, request_key: "schema-prediction-request" };
  const created = await api(page, predictionPath, predictionBody);
  captures.push({ method: "POST", path: `/api${predictionPath}`, ...created });
  expect(created.status).toBe(202);
  const runId = numericField(created.body, "id");
  expect((await api(page, predictionPath, predictionBody)).body).toMatchObject({ id: runId });
  for (const [method, path] of [["POST", `${predictionPath}/${runId}/retry`],
    ["GET", `${predictionPath}/${runId}`], ["GET", `${predictionPath}/${runId}/results?limit=1`]]) {
    const response = await api(page, path, method === "POST" ? {} : undefined);
    captures.push({ method, path: `/api${path}`, ...response });
  }
  const reviewPath = `${predictionPath}/${runId}/review-round`;
  const review = await api(page, reviewPath, { name: "Prediction review", request_key: "schema-review-request",
    dataset_item_ids: [221, 222], include_submitted: false });
  expect(review.body).toMatchObject({ item_count: 1, excluded_protected: 1 });
  captures.push({ method: "POST", path: `/api${reviewPath}`, ...review });
  const roundId = numericField(review.body, "round_id");
  const roundPath = `/rounds/${roundId}/work-context?workspace_id=1`;
  captures.push({ method: "GET", path: `/api${roundPath}`, ...await api(page, roundPath) });
  const capturePath = testInfo.outputPath("backend-responses.json");
  writeFileSync(capturePath, JSON.stringify(captures));
  const configFile = testInfo.config.configFile;
  if (!configFile) throw new Error("Backend contract checks require a Playwright configuration file");
  const root = resolve(dirname(configFile), "..");
  const virtualenv = resolve(root, ".venv/bin/python");
  const python = process.env.AL_MEDLIT_PYTHON ?? (existsSync(virtualenv) ? virtualenv : "python3");
  const output = execFileSync(python, [resolve(root, "scripts/generate_e2e_contract.py"),
    "--responses", capturePath], { encoding: "utf8", env: { ...process.env, PYTHONPATH: resolve(root, "backend") } });
  expect(output).toContain(`Validated ${captures.length} fixture responses`);
});

for (const [route, region] of [["/training", "Workspace training runs"], ["/models", "Workspace model registry"]]) {
  test(`${route} loads the next real cursor page`, async ({ page }) => {
    const fixture = await installPlatformApiMock(page, { role: "trainer", pagination: true });
    await page.goto(route);
    const rows = page.getByRole("region", { name: region, exact: true }).locator("tbody tr");
    await expect(rows).toHaveCount(50);
    await page.getByRole("button", { name: "Load more", exact: true }).click();
    await expect(rows).toHaveCount(51);
    await expect(page.getByRole("button", { name: "Load more", exact: true })).toHaveCount(0);
    expect(fixture.requests.filter((request) => request.search.includes("cursor="))).toHaveLength(1);
  });
}
