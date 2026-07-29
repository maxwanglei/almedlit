export type CapabilityKey =
  | "annotation"
  | "multi_annotator_iaa"
  | "training"
  | "inference"
  | "active_learning"
  | "co_learning"
  | "lineage"
  | "export"
  | "hpc_training"
  | "llm_serving";

export function hasCapability(
  effective: readonly string[],
  key: CapabilityKey,
): boolean {
  return effective.includes(key);
}

export function canTrain(effective: readonly string[]): boolean {
  return hasCapability(effective, "training");
}
