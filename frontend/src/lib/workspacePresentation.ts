import type { WorkspaceRole } from "@/types/api";

export const WORKSPACE_PRESET_LABELS = {
  annotate: "Annotate",
  train: "Train models",
  annotate_train: "Annotate and train",
  annotate_train_al: "Annotate and train with active learning",
  full: "Full",
};

export type WorkspacePreset = keyof typeof WORKSPACE_PRESET_LABELS;

export const WORKSPACE_ROLE_OPTIONS = [
  { value: "annotator", label: "Annotator" },
  { value: "trainer", label: "Model trainer" },
  { value: "manager", label: "Manager" },
  { value: "admin", label: "Administrator" },
] satisfies Array<{ value: WorkspaceRole; label: string }>;

export function workspaceRoleLabel(role: string, workspaceKind = "team"): string {
  if (workspaceKind === "individual" && role === "admin") return "Owner";
  return WORKSPACE_ROLE_OPTIONS.find((option) => option.value === role)?.label ?? role;
}
