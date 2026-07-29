import { describe, expect, it } from "vitest";

import {
  documentQueueStatusFromAssignments,
  getPersonalProjectReadiness,
  managerTabsForWorkspace,
} from "@/pages/ManagerConsole";
import type {
  AnnotationType,
  EvidenceTarget,
  Project,
  ProjectTask,
  TaskAssignment,
} from "@/types/api";

function task(
  id: number,
  annotationType: AnnotationType,
  options: {
    enabled?: boolean;
    labels?: string[];
    displayName?: string;
  } = {},
): ProjectTask {
  return {
    id,
    project_id: 1,
    annotation_type: annotationType,
    display_name: options.displayName ?? `${annotationType} task`,
    description: null,
    enabled: options.enabled ?? true,
    sort_order: id,
    labels: (options.labels ?? []).map((name) => ({
      name,
      color: "#4d6e5b",
      description: null,
    })),
    settings: {},
  };
}

function project(
  tasks: ProjectTask[],
  mode: Project["annotation_validation_mode"] = "relaxed",
): Project {
  return {
    id: 1,
    name: "Personal test",
    description: null,
    annotation_schema: { labels: {} },
    annotation_validation_mode: mode,
    tasks,
    settings: {},
    workspace_id: 1,
  };
}

function activeEvidenceTarget(taskId: number): EvidenceTarget {
  return {
    id: 10,
    project_id: 1,
    task_id: taskId,
    key: "primary-outcome",
    name: "Primary outcome",
    description: null,
    is_active: true,
    active_version_id: 100,
    versions: [],
    created_by_user_id: 1,
    created_at: "2026-07-23T00:00:00Z",
    updated_at: "2026-07-23T00:00:00Z",
  };
}

function assignment(id: number, status: TaskAssignment["status"]): TaskAssignment {
  return {
    id,
    project_id: 1,
    task_id: id,
    document_id: 1,
    assignee_user_id: 1,
    annotator_id: "solo",
    status,
    assigned_by_user_id: 1,
    assigned_by: "solo",
    notes: null,
    metadata_: {},
    target_version_id: null,
    structure_version_id: null,
    guideline_version_id: null,
    assignment_scope_key: `task:${id}`,
  };
}

describe("personal project readiness", () => {
  it("blocks strict non-evidence tasks until labels are configured", () => {
    const readiness = getPersonalProjectReadiness(
      project([task(1, "entity", { displayName: "Entities" })], "strict"),
      1,
      [],
    );

    expect(readiness.canStart).toBe(false);
    expect(readiness.nextAction).toBe("tasks");
    expect(readiness.steps[0]).toMatchObject({
      id: "tasks",
      status: "blocked",
    });
    expect(readiness.steps[0].detail).toContain(
      "Entities needs at least one label in strict validation.",
    );
  });

  it("makes relaxed free-form annotation an explicit non-blocking path", () => {
    const readiness = getPersonalProjectReadiness(
      project([task(1, "entity", { displayName: "Entities" })]),
      2,
      [],
    );

    expect(readiness.canStart).toBe(true);
    expect(readiness.usesFreeFormLabels).toBe(true);
    expect(readiness.nextAction).toBe("my-work");
    expect(readiness.steps[0].status).toBe("warning");
    expect(readiness.steps[2]).toMatchObject({
      status: "ready",
      detail:
        "Ready to start with free-form labels. You can add preset labels at any time.",
    });
  });

  it("uses legacy schema labels when task labels are empty", () => {
    const legacyProject = project([task(1, "entity")], "strict");
    legacyProject.annotation_schema.labels.entity = [
      { name: "Disease", color: "#4d6e5b", description: null },
    ];

    const readiness = getPersonalProjectReadiness(legacyProject, 1, []);

    expect(readiness.canStart).toBe(true);
    expect(readiness.usesFreeFormLabels).toBe(false);
    expect(readiness.steps[0].status).toBe("complete");
  });

  it("blocks evidence work until the task has an active target version", () => {
    const evidenceTask = task(3, "evidence_block", {
      displayName: "Evidence blocks",
    });
    const missingTarget = getPersonalProjectReadiness(
      project([evidenceTask]),
      1,
      [],
    );
    const withTarget = getPersonalProjectReadiness(
      project([evidenceTask]),
      1,
      [activeEvidenceTarget(evidenceTask.id)],
    );

    expect(missingTarget.canStart).toBe(false);
    expect(missingTarget.steps[0].detail).toContain(
      "Evidence blocks needs an active evidence target.",
    );
    expect(withTarget.canStart).toBe(true);
  });

  it("requires an enabled entity task before relation annotation", () => {
    const relationTask = task(2, "relation", {
      labels: ["treats"],
      displayName: "Drug relation",
    });
    const disabledEntityTask = task(1, "entity", {
      enabled: false,
      labels: ["Drug"],
    });
    const missingEntity = getPersonalProjectReadiness(
      project([disabledEntityTask, relationTask], "strict"),
      1,
      [],
    );
    const withEntity = getPersonalProjectReadiness(
      project([
        { ...disabledEntityTask, enabled: true },
        relationTask,
      ], "strict"),
      1,
      [],
    );

    expect(missingEntity.canStart).toBe(false);
    expect(missingEntity.steps[0].detail).toContain(
      "Drug relation needs an enabled entity task first.",
    );
    expect(withEntity.canStart).toBe(true);
  });

  it("routes the next setup action to document import when tasks are ready", () => {
    const readiness = getPersonalProjectReadiness(
      project([task(1, "entity", { labels: ["Disease"] })], "strict"),
      0,
      [],
    );

    expect(readiness.canStart).toBe(false);
    expect(readiness.nextAction).toBe("documents");
    expect(readiness.steps[1].status).toBe("blocked");
  });
});

describe("manager project tabs", () => {
  it("uses personal gold-review language and removes the redundant Models tab", () => {
    const personalTabs = managerTabsForWorkspace(true);
    const teamTabs = managerTabsForWorkspace(false);

    expect(personalTabs.find((tab) => tab.id === "loop")?.label).toBe(
      "Gold review",
    );
    expect(teamTabs.find((tab) => tab.id === "loop")?.label).toBe(
      "Learning Loop",
    );
    expect(personalTabs.some((tab) => tab.id === "models")).toBe(false);
    expect(teamTabs.some((tab) => tab.id === "models")).toBe(false);
  });
});

describe("personal document progress", () => {
  it("treats a multi-task document as partial until every task is finished", () => {
    expect(
      documentQueueStatusFromAssignments([
        assignment(1, "submitted"),
        assignment(2, "assigned"),
      ]),
    ).toBe("partial");
    expect(
      documentQueueStatusFromAssignments([
        assignment(1, "submitted"),
        assignment(2, "completed"),
      ]),
    ).toBe("done");
  });
});
