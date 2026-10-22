import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import postcss from "postcss";

const spacingScale = new Set([0, 4, 8, 12, 16, 24, 32, 40]);
const retiredClasses = new Set([
  "sidebar", "right-panel", "manager-panel", "workbench-grid", "manager-shell",
  "project-rail", "manager-main", "selectors", "custom-label", "workspace-nav",
  "label-list", "annotation-list", "compact-list", "task-stack", "label-editor-list",
  "label-button", "annotation-row", "project-button", "primary-action", "swatch",
  "selection-meter", "document-pane", "document-meta", "document-text", "text-segment",
  "annotation-range", "annotation-lock-note", "guideline-panel", "assignment-queue",
  "my-work-shell", "my-work-rail", "my-work-main", "training-shell", "workspace-tabs",
  "personal-task-grid", "personal-task-card", "compact-row", "project-list",
  "create-project-form", "starter-task-fieldset", "check-row", "project-summary-panel",
  "project-setting-control", "summary-metrics", "assignment-form", "label-editor-row",
  "assignment-row", "add-label-row", "assignment-table", "progress-row", "progress-grid",
  "progress-panel", "task-editor", "inline-form", "row-actions", "training-main",
  "guideline-doc", "xds-button", "empty-state",
]);

function inTokenLayer(node) {
  for (let parent = node.parent; parent; parent = parent.parent) {
    if (parent.type === "rule" && parent.selector === ":root") return true;
  }
  return false;
}

function inspect(source) {
  const diagnostics = [];
  const root = postcss.parse(source);
  const report = (kind, node) => diagnostics.push({ kind, line: node.source.start.line, selector: node.parent.selector ?? node.selector, declaration: node.toString() });
  root.walkRules((rule) => {
    const classes = [...rule.selector.matchAll(/\.([a-zA-Z_][\w-]*)/g)].map((match) => match[1]);
    if (classes.some((name) => retiredClasses.has(name) || /^(mc|tc)-/.test(name))) report("retired selector", rule);
  });
  root.walkDecls((declaration) => {
    if (inTokenLayer(declaration)) return;
    const { prop, value, parent } = declaration;
    if (/#(?:[\da-f]{3,8})\b|\b(?:rgb|hsl)a?\(/i.test(value)) report("raw color", declaration);
    if (prop === "font-size") {
      const size = value.match(/^([\d.]+)(px|rem)$/);
      if (size && Number(size[1]) * (size[2] === "rem" ? 16 : 1) < 13) report("small UI text", declaration);
    }
    if (/^(?:margin|padding)(?:-|$)|^(?:gap|row-gap|column-gap)$/.test(prop)) {
      const geometric = /\.aw-entity(?:\b|\.)|\.sr-only|\.visually-hidden|\.eb-sentence-number/.test(parent.selector);
      if (!geometric && [...value.matchAll(/(-?[\d.]+)(px|rem)\b/g)].some((match) => !spacingScale.has(Number(match[1]) * (match[2] === "rem" ? 16 : 1)))) report("off-scale spacing", declaration);
    }
    if (prop === "outline-offset" && value !== "var(--focus-ring-offset)") report("focus offset", declaration);
    if (prop === "z-index" && !/^(?:var\(--z-[\w-]+\)|auto)$/.test(value)) report("raw stacking level", declaration);
    if (prop === "border-radius" && [...value.matchAll(/([\d.]+)px\b/g)].some((match) => Number(match[1]) > 8 && Number(match[1]) < 99)) report("large container radius", declaration);
  });
  return diagnostics;
}

if (process.argv.includes("--self-test")) {
  const failures = inspect(".manager-panel { color: #fff; font-size: 0.75rem; padding: 10px; outline-offset: 1px; z-index: 2; border-radius: 12px; }");
  assert.deepEqual(new Set(failures.map(({ kind }) => kind)), new Set(["retired selector", "raw color", "small UI text", "off-scale spacing", "focus offset", "raw stacking level", "large container radius"]));
  assert.equal(inspect(":root { --test: #fff; } .live { padding: 8px 12px; font-size: 13px; outline-offset: var(--focus-ring-offset); z-index: var(--z-content); }").length, 0);
  process.stdout.write("UI style audit detects each seeded regression.\n");
}

const filename = process.argv.find((argument) => argument.endsWith(".css")) ?? path.resolve(import.meta.dirname, "../src/styles.css");
const diagnostics = inspect(fs.readFileSync(filename, "utf8"));
for (const diagnostic of diagnostics) process.stderr.write(`${filename}:${diagnostic.line} ${diagnostic.kind} ${diagnostic.declaration}\n`);
process.stdout.write(`UI style audit found ${diagnostics.length} violations.\n`);
if (diagnostics.length) process.exitCode = 1;
