import type { EvidenceCommandSummary } from "@/types/api";

export function nextUndoCommand(
  commandsNewestFirst: EvidenceCommandSummary[],
): EvidenceCommandSummary | null {
  return commandsNewestFirst.find((command) => command.status === "applied") ?? null;
}

export function nextRedoCommand(
  commandsNewestFirst: EvidenceCommandSummary[],
): EvidenceCommandSummary | null {
  if (commandsNewestFirst[0]?.status !== "undone") {
    return null;
  }
  let candidate: EvidenceCommandSummary | null = null;
  for (const command of commandsNewestFirst) {
    if (command.status !== "undone") {
      break;
    }
    candidate = command;
  }
  return candidate;
}
