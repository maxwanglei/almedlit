import type { TaskKind } from "@/platform/types";

const TASK_KIND_LABELS: Record<TaskKind, string> = {
  regression: "Regression",
  classification: "Classification",
  multilabel_classification: "Multilabel classification",
  token_labeling: "Token labeling (tokenized data)",
  span_extraction: "Span extraction",
  relation_extraction: "Relation extraction",
  ranking: "Ranking",
  generation: "Generation",
  instruction_tuning: "Instruction tuning",
};

export function formatTaskKind(kind: TaskKind, preset?: unknown): string {
  return kind === "span_extraction" && preset === "document_entities"
    ? "Named entities (NER)"
    : TASK_KIND_LABELS[kind];
}
