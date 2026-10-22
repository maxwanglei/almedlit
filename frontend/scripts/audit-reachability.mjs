import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const defaultRoot = fileURLToPath(new URL("../src", import.meta.url));
const ignored = /(?:^|\/)(?:test|__tests__)(?:\/|$)|\.(?:test|spec)\.[cm]?[jt]sx?$|\.d\.ts$/;
const sourceExtension = /\.[cm]?[jt]sx?$/;

function sourceFiles(root) {
  return fs.readdirSync(root, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(root, entry.name);
    const relative = path.relative(root, absolute).replaceAll(path.sep, "/");
    if (ignored.test(relative)) return [];
    if (entry.isDirectory()) return sourceFiles(absolute);
    return entry.isFile() && sourceExtension.test(entry.name) ? [absolute] : [];
  });
}

export function auditReachability(root, entry = "main.tsx") {
  root = path.resolve(root);
  const files = sourceFiles(root);
  const known = new Set(files);
  const entrypoint = path.resolve(root, entry);
  if (!known.has(entrypoint)) throw new Error(`Missing production entrypoint: ${entrypoint}`);
  const options = {
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    baseUrl: root,
    paths: { "@/*": ["*"] },
    allowJs: true,
  };
  const edges = new Map();
  const diagnostics = [];
  const relative = (file) => path.relative(root, file).replaceAll(path.sep, "/");
  for (const file of files) {
    const dependencies = new Set();
    const source = ts.createSourceFile(file, fs.readFileSync(file, "utf8"), ts.ScriptTarget.Latest, true);
    const add = (specifier) => {
      if (!specifier.startsWith(".") && !specifier.startsWith("@/")) return;
      if (/\.(?:css|svg|png|jpe?g|webp|gif|woff2?|json)(?:\?.*)?$/.test(specifier)) return;
      const resolved = ts.resolveModuleName(specifier, file, options, ts.sys).resolvedModule;
      if (!resolved) {
        diagnostics.push({ file: relative(file), import: specifier, reason: "Unresolved source import" });
      } else if (known.has(resolved.resolvedFileName)) {
        dependencies.add(resolved.resolvedFileName);
      } else if (resolved.resolvedFileName.startsWith(`${root}${path.sep}`) && !resolved.resolvedFileName.endsWith(".d.ts")) {
        diagnostics.push({ file: relative(file), import: specifier, reason: "Production import points to excluded test source" });
      }
    };
    const visit = (node) => {
      if ((ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) && node.moduleSpecifier) {
        if (ts.isStringLiteralLike(node.moduleSpecifier)) add(node.moduleSpecifier.text);
      } else if (ts.isImportEqualsDeclaration(node) && ts.isExternalModuleReference(node.moduleReference)) {
        const expression = node.moduleReference.expression;
        if (expression && ts.isStringLiteralLike(expression)) add(expression.text);
      } else if (ts.isImportTypeNode(node) && ts.isLiteralTypeNode(node.argument)) {
        if (ts.isStringLiteralLike(node.argument.literal)) add(node.argument.literal.text);
      } else if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword) {
        const specifier = node.arguments[0];
        if (specifier && ts.isStringLiteralLike(specifier)) add(specifier.text);
        else diagnostics.push({ file: relative(file), reason: "Nonliteral dynamic import needs an explicit root" });
      }
      ts.forEachChild(node, visit);
    };
    visit(source);
    edges.set(file, dependencies);
  }
  const reached = new Set();
  const queue = [entrypoint];
  while (queue.length) {
    const file = queue.pop();
    if (reached.has(file)) continue;
    reached.add(file);
    queue.push(...edges.get(file));
  }
  return {
    entry: relative(entrypoint),
    sourceFiles: files.length,
    reachable: [...reached].map(relative).sort(),
    unreachable: files.filter((file) => !reached.has(file)).map(relative).sort(),
    diagnostics,
  };
}

function selfTest() {
  const fixture = fs.mkdtempSync(path.join(os.tmpdir(), "al-medlit-reachability-"));
  try {
    fs.writeFileSync(path.join(fixture, "main.tsx"), 'import type { Value } from "@/types"; import("./lazy");');
    fs.writeFileSync(path.join(fixture, "lazy.ts"), 'export { helper } from "./helper";');
    fs.writeFileSync(path.join(fixture, "helper.ts"), "export const helper = 1;");
    fs.writeFileSync(path.join(fixture, "types.ts"), "export type Value = number;");
    fs.writeFileSync(path.join(fixture, "dead.ts"), "export const dead = 1;");
    fs.writeFileSync(path.join(fixture, "dead.test.ts"), 'import { dead } from "./dead";');
    const initial = auditReachability(fixture);
    assert.deepEqual(initial.unreachable, ["dead.ts"]);
    assert.equal(initial.reachable.length, 4);
    assert.equal(initial.diagnostics.length, 0);
    fs.unlinkSync(path.join(fixture, "helper.ts"));
    const brokenImport = auditReachability(fixture);
    assert.deepEqual(brokenImport.diagnostics, [{ file: "lazy.ts", import: "./helper", reason: "Unresolved source import" }]);
    fs.writeFileSync(path.join(fixture, "lazy.ts"), "import(variable);");
    assert.equal(auditReachability(fixture).diagnostics[0].reason, "Nonliteral dynamic import needs an explicit root");
    fs.writeFileSync(path.join(fixture, "lazy.ts"), 'import "./dead.test";');
    assert.equal(auditReachability(fixture).diagnostics[0].reason, "Production import points to excluded test source");
    fs.unlinkSync(path.join(fixture, "main.tsx"));
    assert.throws(() => auditReachability(fixture), /Missing production entrypoint/);
    process.stdout.write("Reachability audit detects test-only modules, broken imports, dynamic imports, excluded dependencies, and missing roots.\n");
  } finally {
    fs.rmSync(fixture, { recursive: true, force: true });
  }
}

const args = process.argv.slice(2);
const value = (flag, fallback) => {
  const index = args.indexOf(flag);
  if (index < 0) return fallback;
  if (!args[index + 1] || args[index + 1].startsWith("--")) throw new Error(`Missing value for ${flag}`);
  return args[index + 1];
};
if (args.includes("--self-test")) selfTest();
const result = auditReachability(value("--root", defaultRoot), value("--entry", "main.tsx"));
if (args.includes("--json")) process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
else {
  process.stdout.write(`${result.reachable.length}/${result.sourceFiles} production source modules reachable from ${result.entry}.\n`);
  for (const file of result.unreachable) process.stdout.write(`Unreachable: ${file}\n`);
  for (const diagnostic of result.diagnostics) process.stdout.write(`${diagnostic.reason}: ${diagnostic.file} ${diagnostic.import ?? ""}\n`);
}
if (args.includes("--check") && (result.unreachable.length || result.diagnostics.length)) process.exitCode = 1;
