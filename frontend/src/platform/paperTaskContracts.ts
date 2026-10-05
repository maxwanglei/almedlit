import type { TaskVersion } from "./types";

export type PaperAnnotationEditor = "classification" | "entities";
export interface PaperAnnotationCompatibility {
  supported: boolean;
  editor: PaperAnnotationEditor | null;
  reason: string | null;
}

export type PaperTaskVersionPayload = Pick<TaskVersion,
  "project_id" | "task_definition_id" | "task_kind" | "input_schema" |
  "output_schema" | "label_rules" | "annotation_ui" | "metrics" | "trainer_compatibility"
>;

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function requiresOnly(schema: Record<string, unknown>, fields: string[]): boolean {
  return schema.required === undefined || (Array.isArray(schema.required) &&
    schema.required.every((field) => typeof field === "string" && fields.includes(field)));
}

function unsupported(reason: string): PaperAnnotationCompatibility {
  return { supported: false, editor: null, reason };
}

/** Choose an editor from the pinned data contract, never from a task's display name. */
export function paperAnnotationCompatibility(task: TaskVersion): PaperAnnotationCompatibility {
  if (task.task_kind === "token_labeling") {
    return unsupported("This task requires pretokenized inputs and one label per token. PubMed collections contain document text. Create a new document entity task version to annotate these papers; the existing version and its rounds will be preserved.");
  }
  const input = record(task.input_schema);
  const output = record(task.output_schema);
  if ((input.type !== undefined && input.type !== "object") ||
    record(record(input.properties).text).type !== "string" || !requiresOnly(input, ["text"])) {
    return unsupported("This task requires inputs other than paper text. Choose a task that accepts the collection's text field.");
  }
  if (task.task_kind === "classification" && output.type === "string") {
    if (paperTaskLabels(task).length === 0) return unsupported("This classification task has no configured labels. Add a task version with a defined label list to use the paper annotation editor.");
    return { supported: true, editor: "classification", reason: null };
  }
  if (task.task_kind === "span_extraction") {
    if (task.annotation_ui?.preset !== "document_entities" || task.annotation_ui.offset_unit !== "utf16_code_unit" || task.annotation_ui.end_offset !== "exclusive") {
      return unsupported("This span task uses a different output format or offset convention. Create a document entity task version with UTF-16 character offsets to use the paper annotation editor.");
    }
    const entities = record(record(output.properties).entities);
    const entity = record(entities.items);
    const fields = record(entity.properties);
    if (output.type === "object" && requiresOnly(output, ["entities"]) &&
      entities.type === "array" && entity.type === "object" && requiresOnly(entity, ["start", "end", "label"]) &&
      record(fields.start).type === "integer" && record(fields.end).type === "integer" && record(fields.label).type === "string") {
      if (paperTaskLabels(task).length === 0) return unsupported("This entity task has no configured labels. Add entity labels in a new task version to use the paper annotation editor.");
      return { supported: true, editor: "entities", reason: null };
    }
    return unsupported("This span task uses a different output format. Document entity annotation requires an entities list with start, end, and label for every selected span. Create a new compatible task version.");
  }
  return unsupported("This task does not have a supported paper annotation editor. Choose document classification or document entity annotation.");
}

export function supportsPaperAnnotation(task: TaskVersion): boolean {
  return paperAnnotationCompatibility(task).supported;
}

function uniqueLabels(values: unknown[]): string[] {
  return [...new Set(values.filter((value): value is string => typeof value === "string")
    .map((value) => value.trim()).filter(Boolean))];
}

export function paperTaskLabels(task: TaskVersion): string[] {
  const rules = record(task.label_rules);
  const output = record(task.output_schema);
  if (Array.isArray(rules.values) && uniqueLabels(rules.values).length) return uniqueLabels(rules.values);
  if (Array.isArray(output.enum)) return uniqueLabels(output.enum);
  const items = record(output.items);
  if (Array.isArray(items.enum)) return uniqueLabels(items.enum);
  const entities = record(record(output.properties).entities);
  const label = record(record(record(entities.items).properties).label);
  return Array.isArray(label.enum) ? uniqueLabels(label.enum) : [];
}

/** A human NER contract for the existing character-selection editor; no trainer is implied. */
export function createPaperEntityTaskVersionPayload(
  projectId: number,
  taskDefinitionId: number,
  labels: string[],
): PaperTaskVersionPayload {
  const values = uniqueLabels(labels);
  if (values.length === 0) throw new Error("Enter at least one entity label for document annotation.");
  return {
    project_id: projectId,
    task_definition_id: taskDefinitionId,
    task_kind: "span_extraction",
    input_schema: {
      type: "object", required: ["text"],
      properties: { text: { type: "string", minLength: 1 } },
    },
    output_schema: {
      type: "object", required: ["entities"], additionalProperties: false,
      properties: {
        entities: {
          type: "array",
          items: {
            type: "object", required: ["start", "end", "label"], additionalProperties: false,
            properties: {
              start: { type: "integer", minimum: 0 },
              end: { type: "integer", minimum: 1 },
              label: { type: "string", enum: values },
            },
          },
        },
      },
    },
    label_rules: { values, closed_set: true },
    annotation_ui: { preset: "document_entities", offset_unit: "utf16_code_unit", end_offset: "exclusive" },
    metrics: ["entity_f1"],
    trainer_compatibility: [],
  };
}

/** Construct a new version only. Calling this function never changes or saves the old task. */
export function derivePaperEntityTaskVersionPayload(
  existing: TaskVersion,
  labels?: string[],
): PaperTaskVersionPayload {
  if (existing.task_kind !== "token_labeling" && existing.task_kind !== "span_extraction") {
    throw new Error("Only token or span annotation tasks can be adapted to a document entity task version.");
  }
  const values = labels ?? (existing.task_kind === "token_labeling"
    ? paperTaskLabels(existing).filter((label) => label !== "O").map((label) => label.replace(/^(?:B|I|E|S|L|U)-/, ""))
    : paperTaskLabels(existing));
  const payload = createPaperEntityTaskVersionPayload(existing.project_id, existing.task_definition_id, values);
  payload.annotation_ui = { ...payload.annotation_ui, source_task_version_id: existing.id };
  return payload;
}
