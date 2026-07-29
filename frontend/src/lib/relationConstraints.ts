import type {
  LabelDef,
  ProjectTask,
  RelationConstraint,
  RelationConstraintMap,
} from "@/types/api";

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

export function relationConstraintsOf(
  task: Pick<ProjectTask, "settings"> | null | undefined,
): RelationConstraintMap {
  const raw = task?.settings?.["relation_constraints"];
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return {};
  }

  const result: RelationConstraintMap = {};
  for (const [label, value] of Object.entries(raw as Record<string, unknown>)) {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      continue;
    }
    const entry = value as { head?: unknown; tail?: unknown };
    result[label] = {
      head: stringList(entry.head),
      tail: stringList(entry.tail),
    };
  }
  return result;
}

export function relationLabelAllows(
  constraint: RelationConstraint | undefined,
  headLabel: string,
  tailLabel: string,
): boolean {
  if (!constraint) {
    return true;
  }
  const headAllowed = constraint.head.length === 0 || constraint.head.includes(headLabel);
  const tailAllowed = constraint.tail.length === 0 || constraint.tail.includes(tailLabel);
  return headAllowed && tailAllowed;
}

export function filterRelationLabels(
  labels: LabelDef[],
  constraints: RelationConstraintMap,
  headLabel: string,
  tailLabel: string,
): LabelDef[] {
  return labels.filter((label) =>
    relationLabelAllows(constraints[label.name], headLabel, tailLabel),
  );
}
