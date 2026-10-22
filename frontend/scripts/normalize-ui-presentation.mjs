import fs from "node:fs";
import path from "node:path";
import ts from "typescript";

const sourceRoot = path.resolve(import.meta.dirname, "../src");
const check = process.argv.includes("--check");
const changed = [];

function visitFiles(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) visitFiles(filename);
    else if (filename.endsWith(".tsx") && !filename.endsWith(".test.tsx") && !["LoadingState.tsx", "NativeField.tsx", "ValidatedForm.tsx"].includes(entry.name)) normalize(filename);
  }
}

function normalize(filename) {
  const source = fs.readFileSync(filename, "utf8");
  const ast = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const edits = [];
  let loading = false;
  let form = false;
  const controls = new Set();
  const nativeTags = { input: "NativeInput", select: "NativeSelect", textarea: "NativeTextarea", form: "ValidatedForm" };
  const authPage = ["LoginPage.tsx", "AcceptInvitePage.tsx", "AccountActionPage.tsx"].includes(path.basename(filename));
  function visit(node) {
    if (ts.isJsxElement(node)) {
      const opening = node.openingElement;
      const attributes = opening.attributes.properties;
      const role = attributes.find((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(ast) === "role");
      const textOnly = node.children.every((child) => ts.isJsxText(child) || (ts.isJsxSelfClosingElement(child) && child.tagName.getText(ast) === "span" && child.attributes.properties.some((attribute) => ts.isJsxAttribute(attribute) && attribute.name.getText(ast) === "aria-hidden")));
      const text = textOnly ? node.children.filter(ts.isJsxText).map((child) => child.text).join("").trim().replace(/\s+/g, " ") : "";
      if (["p", "div"].includes(opening.tagName.getText(ast)) && role?.initializer && ts.isStringLiteral(role.initializer) && role.initializer.text === "status" && /^(Loading|Opening|Checking|Signing out)(?:\s|…)/.test(text)) {
        edits.push({ start: node.getStart(ast), end: node.end, replacement: `<LoadingState label=${JSON.stringify(text)} />` });
        loading = true;
        return;
      }
    }
    if (ts.isJsxSelfClosingElement(node) || ts.isJsxOpeningElement(node)) {
      const attributes = node.attributes.properties.filter(ts.isJsxAttribute);
      const tag = node.tagName.getText(ast);
      if (!authPage && nativeTags[tag]) {
        const replacement = nativeTags[tag];
        edits.push({ start: node.tagName.getStart(ast), end: node.tagName.end, replacement });
        if (tag === "form") form = true;
        else controls.add(replacement);
      }
      const icon = tag === "IconButton" || (tag === "Button" && attributes.some((attribute) => attribute.name.getText(ast) === "isIconOnly"));
      const label = attributes.find((attribute) => attribute.name.getText(ast) === "label");
      if (icon && label?.initializer && !attributes.some((attribute) => attribute.name.getText(ast) === "tooltip")) {
        edits.push({ start: label.end, end: label.end, replacement: ` tooltip=${label.initializer.getText(ast)}` });
      }
    }
    if (ts.isJsxClosingElement(node) && !authPage) {
      const replacement = nativeTags[node.tagName.getText(ast)];
      if (replacement) edits.push({ start: node.tagName.getStart(ast), end: node.tagName.end, replacement });
    }
    ts.forEachChild(node, visit);
  }
  visit(ast);
  if (!edits.length) return;
  if (loading && !source.includes('from "@/components/LoadingState"')) {
    edits.push({ start: 0, end: 0, replacement: 'import LoadingState from "@/components/LoadingState";\n' });
  }
  if (form) edits.push({ start: 0, end: 0, replacement: 'import ValidatedForm from "@/components/ValidatedForm";\n' });
  if (controls.size) edits.push({ start: 0, end: 0, replacement: `import { ${[...controls].sort().join(", ")} } from "@/components/NativeField";\n` });
  let updated = source;
  for (const edit of edits.sort((a, b) => b.start - a.start)) updated = updated.slice(0, edit.start) + edit.replacement + updated.slice(edit.end);
  changed.push(path.relative(sourceRoot, filename));
  if (!check) fs.writeFileSync(filename, updated);
}

visitFiles(sourceRoot);
process.stdout.write(`${check ? "Pending" : "Updated"} UI presentation files: ${changed.length}\n${changed.join("\n")}\n`);
if (check && changed.length) process.exitCode = 1;
