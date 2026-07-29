import { describe, expect, it } from "vitest";

import { nextRedoCommand, nextUndoCommand } from "./evidenceCommandHistory";
import type { EvidenceCommandSummary } from "@/types/api";

function command(
  key: string,
  status: EvidenceCommandSummary["status"],
): EvidenceCommandSummary {
  return {
    command_group_key: key,
    operation: "update",
    status,
    project_id: 1,
    document_id: 2,
    target_version_id: 3,
    structure_version_id: 5,
    guideline_version_id: 6,
    actor_user_id: 4,
    created_at: "2026-07-15T12:00:00Z",
  };
}

describe("persisted evidence command history", () => {
  it("undoes the newest command that remains applied", () => {
    const commands = [command("new", "undone"), command("middle", "applied"), command("old", "applied")];
    expect(nextUndoCommand(commands)?.command_group_key).toBe("middle");
  });

  it("redoes in reverse undo order", () => {
    const commands = [command("new", "undone"), command("middle", "undone"), command("old", "applied")];
    expect(nextRedoCommand(commands)?.command_group_key).toBe("middle");
    expect(nextRedoCommand([command("new", "undone"), command("middle", "applied")])?.command_group_key).toBe(
      "new",
    );
    expect(nextRedoCommand([command("branch", "applied"), command("old", "undone")])).toBeNull();
  });
});
