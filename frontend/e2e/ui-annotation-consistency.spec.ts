import { expect, test } from "./fixtures/test";
import { installEvidenceApiMock } from "./fixtures/mockEvidenceApi";
import type { Annotation, AnnotationCreate, AnnotationWorkbenchTask } from "../src/types/api";

test("keyboard text selection creates a span and deletion requires confirmation", async ({ page }) => {
  const fixture = await installEvidenceApiMock(page, "annotator");
  const task: AnnotationWorkbenchTask = {
    id: 11, project_id: 1, annotation_type: "entity", display_name: "Entity labels", description: null,
    enabled: true, sort_order: 0, labels: [{ name: "Disease", color: "var(--annotation-label-1)", description: null }], settings: {},
    annotation_type_spec: { name: "entity", requires_span: true, requires_head_tail: false,
      description: "Entity spans", selection_mode: "character_span", renderer_key: "legacy",
      relation_endpoint_allowed: true, handler_key: "generic" },
  };
  const text = "Trial enrolled adults at three centers.";
  let annotations: Annotation[] = [];
  const writes: AnnotationCreate[] = [];
  let deletes = 0;
  await page.route("**/api/annotation-workbench/documents/41", (route) => route.fulfill({
    json: {
      project: { id: 1, name: "Evidence Review Demo", description: null, tasks: [task],
        annotation_schema: { labels: { entity: task.labels } }, annotation_validation_mode: "strict", settings: {}, workspace_id: 1 },
      document: { id: 41, project_id: 1, external_id: "DOC-41", title: "Keyboard annotation", text,
        source: "test", metadata_: {}, sentences: [[0, text.length]], active_structure_version_id: 201 },
      tasks: [task], annotations, assignments: fixture.assignments.filter((item) => item.assignee_user_id === 2),
      active_guideline: null, guideline_versions_by_id: {}, annotation_type_specs: [task.annotation_type_spec], correction_locked_annotation_ids: [],
    },
  }));
  await page.route("**/api/annotations", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    const payload: AnnotationCreate = route.request().postDataJSON();
    writes.push(payload);
    const annotation: Annotation = {
      id: 801, project_id: 1, document_id: 41, annotation_type: "entity", label: payload.label,
      start_offset: payload.start_offset ?? null, end_offset: payload.end_offset ?? null, text_span: payload.text_span ?? null,
      source: "human", status: "draft", confidence: null, annotator_user_id: 2, annotator_id: "alice", model_checkpoint_id: null,
      guideline_version_id: 301, structure_version_id: 201, head_annotation_id: null, tail_annotation_id: null,
      evidence: {}, attributes: {}, created_at: "2026-10-22T12:00:00Z", updated_at: "2026-10-22T12:00:00Z",
    };
    annotations = [annotation];
    await route.fulfill({ status: 201, json: annotation });
  });
  await page.route("**/api/annotations/801", async (route) => {
    if (route.request().method() !== "DELETE") return route.fallback();
    deletes += 1;
    annotations = [];
    await route.fulfill({ status: 204 });
  });
  await page.goto("/my-work?view=annotate&project=1&document=41&assignment=51&tool=legacy&pane=document");
  const sentence = page.locator(".aw-sentence-body").first();
  await expect(sentence).toContainText(text);
  await sentence.focus();
  await page.keyboard.press("Home");
  for (let index = 0; index < 5; index += 1) await page.keyboard.press("Shift+ArrowRight");
  await expect.poll(() => page.evaluate(() => window.getSelection()?.toString())).toBe("Trial");
  await page.keyboard.press("Enter");
  const picker = page.getByRole("group", { name: "Choose a span label" });
  await expect(picker.getByRole("button", { name: /^Disease/ })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect.poll(() => writes.length).toBe(1);
  expect(writes[0]).toMatchObject({ start_offset: 0, end_offset: 5, text_span: "Trial" });
  await expect(sentence).toBeFocused();
  const span = sentence.getByRole("button", { name: "Disease: Trial" });
  await span.focus();
  await page.keyboard.press("Enter");
  const menu = page.getByRole("menu", { name: "Annotation actions" });
  await expect(menu.getByRole("menuitem").first()).toBeFocused();
  await page.keyboard.press("End");
  await expect(menu.getByRole("menuitem", { name: "Delete annotation" })).toBeFocused();
  await page.keyboard.press("Enter");
  const dialog = page.getByRole("alertdialog", { name: "Delete annotation?" });
  await expect(dialog).toBeVisible();
  expect(deletes).toBe(0);
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(span).toBeVisible();
  await span.focus();
  await page.keyboard.press("Delete");
  await expect(dialog).toBeVisible();
  expect(deletes).toBe(0);
  await dialog.getByRole("button", { name: "Delete annotation" }).click();
  await expect(span).toHaveCount(0);
  expect(deletes).toBe(1);
  await expect(page.locator("main .aw-tabs-row")).toBeVisible();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("tab", { name: "Queue", exact: true }).click();
  await expect(page.getByRole("tabpanel", { name: "Queue" }).locator(".aw-mini-status")).toBeVisible();
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});
