import { describe, expect, it } from "vitest";
import { createPaperEntityTaskVersionPayload, derivePaperEntityTaskVersionPayload,
  paperAnnotationCompatibility, paperTaskLabels, supportsPaperAnnotation } from "./paperTaskContracts";
import type { TaskVersion } from "./types";

function task(overrides: Partial<TaskVersion> = {}): TaskVersion {
  return {
    id: 21, project_id: 7, task_definition_id: 12, version_number: 1,
    task_kind: "classification", input_schema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    output_schema: { type: "string", enum: ["relevant", "irrelevant"] }, label_rules: {},
    annotation_ui: {}, metrics: [], trainer_compatibility: [], content_hash: "unchanged",
    ...overrides,
  };
}

describe("paper annotation task contracts", () => {
  it("chooses the classification editor only when paper text satisfies the input contract", () => {
    expect(paperAnnotationCompatibility(task())).toEqual({ supported: true, editor: "classification", reason: null });
    expect(paperAnnotationCompatibility(task({ input_schema: { properties: { text: { type: "string" } }, required: ["text", "question"] } })).supported).toBe(false);
    expect(supportsPaperAnnotation(task({ input_schema: { properties: { prompt: { type: "string" } } } }))).toBe(false);
    expect(paperTaskLabels(task())).toEqual(["relevant", "irrelevant"]);
    expect(paperAnnotationCompatibility(task({ output_schema: { type: "string" } })).reason).toContain("no configured labels");
  });

  it("creates a document entity task that supports zero or multiple spans and implies no executable trainer", () => {
    const payload = createPaperEntityTaskVersionPayload(7, 12, [" Drug ", "Gene", "Drug", ""]);
    expect(payload.task_kind).toBe("span_extraction");
    expect(payload.input_schema.required).toEqual(["text"]);
    expect(payload.label_rules).toEqual({ values: ["Drug", "Gene"], closed_set: true });
    expect(payload.output_schema).toEqual(expect.objectContaining({
      required: ["entities"], properties: { entities: {
        type: "array", items: expect.objectContaining({ required: ["start", "end", "label"] }),
      } },
    }));
    expect(payload.annotation_ui).toEqual({ preset: "document_entities", offset_unit: "utf16_code_unit", end_offset: "exclusive" });
    expect(payload.trainer_compatibility).toEqual([]);
    expect(paperAnnotationCompatibility(task(payload))).toEqual({ supported: true, editor: "entities", reason: null });
    expect(paperTaskLabels(task(payload))).toEqual(["Drug", "Gene"]);
    expect(supportsPaperAnnotation(task({ ...payload, annotation_ui: {} }))).toBe(false);
    expect(supportsPaperAnnotation(task({ ...payload, annotation_ui: { ...payload.annotation_ui, offset_unit: "unicode_code_point" } }))).toBe(false);
  });

  it("does not silently treat token or single-span tasks as document entity tasks", () => {
    const token = task({ task_kind: "token_labeling", input_schema: { required: ["tokens"], properties: { tokens: { type: "array" } } }, output_schema: { type: "array", items: { type: "string" } } });
    expect(paperAnnotationCompatibility(token).reason).toContain("pretokenized inputs");
    expect(supportsPaperAnnotation(token)).toBe(false);
    expect(paperAnnotationCompatibility(task({ task_kind: "span_extraction", output_schema: { type: "object", properties: { start: { type: "integer" }, end: { type: "integer" } } } })).reason).toContain("different output format");
    const extraRequired = createPaperEntityTaskVersionPayload(7, 12, ["Drug"]);
    extraRequired.output_schema.required = ["entities", "relations"];
    expect(supportsPaperAnnotation(task(extraRequired))).toBe(false);
  });

  it("derives a new document version on the same definition without mutating the immutable token version", () => {
    const token = task({ task_kind: "token_labeling", label_rules: { values: ["O", "B-Drug", "I-Drug", "U-Gene"] }, input_schema: { type: "object", required: ["tokens"] } });
    const original = structuredClone(token);
    const derived = derivePaperEntityTaskVersionPayload(Object.freeze(token));
    expect(token).toEqual(original);
    expect(derived.project_id).toBe(7);
    expect(derived.task_definition_id).toBe(12);
    expect(derived.label_rules.values).toEqual(["Drug", "Gene"]);
    expect(derived.annotation_ui.source_task_version_id).toBe(21);
    expect(derived).not.toHaveProperty("id");
    expect(derived).not.toHaveProperty("version_number");
    expect(derived).not.toHaveProperty("content_hash");
    expect(derivePaperEntityTaskVersionPayload(token, ["Disease"]).label_rules.values).toEqual(["Disease"]);
  });

  it("requires meaningful entity labels and refuses unrelated task conversion", () => {
    expect(() => createPaperEntityTaskVersionPayload(7, 12, [" "])).toThrow("at least one entity label");
    expect(() => derivePaperEntityTaskVersionPayload(task({ task_kind: "token_labeling", label_rules: { values: ["O"] } }))).toThrow("at least one entity label");
    expect(() => derivePaperEntityTaskVersionPayload(task())).toThrow("Only token or span");
  });
});
