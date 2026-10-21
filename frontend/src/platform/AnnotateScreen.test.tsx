// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ProjectTask } from "@/types/api";

import AnnotateScreen from "./AnnotateScreen";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
} from "./types";

const roundMocks = vi.hoisted(() => ({ transitionRound: vi.fn() }));
vi.mock("./api", async (original) => ({ ...await original<Record<string, unknown>>(), ...roundMocks }));

afterEach(cleanup);

function projectData(): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    taskDefinitions: [
      {
        id: 11,
        project_id: 1,
        key: "relevance",
        name: "Abstract relevance",
        description: null,
      },
    ],
    taskVersions: [
      {
        id: 12,
        project_id: 1,
        task_definition_id: 11,
        version_number: 2,
        task_kind: "classification",
        input_schema: {},
        output_schema: {},
        label_rules: {},
        annotation_ui: {},
        metrics: ["f1"],
        trainer_compatibility: [],
        content_hash: "task-hash",
      },
    ],
    datasetVersions: [
      {
        id: 22,
        project_id: 1,
        dataset_id: 21,
        version_number: 3,
        source_uri: null,
        source_revision: "upload-3",
        source_format: "jsonl",
        data_schema: {},
        provenance: {},
        license_info: {},
        content_hash: "dataset-hash",
        item_count: 10,
        artifact_package_id: null,
      },
    ],
    rounds: [
      {
        id: 71,
        project_id: 1,
        cycle_id: null,
        name: "Initial labels",
        sequence: 1,
        dataset_version_id: 22,
        task_version_id: 12,
        guideline_revision_id: null,
        selection_set_version_id: null,
        feedback_set_version_id: null,
        assistance_policy: "blind",
        reannotation_mode: "full_dataset",
        annotator_user_ids: [7],
        open_to_all_annotators: false,
        reason: null,
        status: "open",
        opened_at: "2026-07-28T12:00:00Z",
        closed_at: null,
      },
    ],
  };
}

describe("AnnotateScreen", () => {
  it("lists and opens legacy document tasks", () => {
    const onOpenDocumentTask = vi.fn();
    const documentTask = { id: 90, project_id: 1, annotation_type: "entity", display_name: "Document entities", description: null, enabled: true, sort_order: 0, labels: [], settings: {} } satisfies ProjectTask;
    render(<AnnotateScreen view="tasks" data={projectData()} projectTasks={[documentTask]} onOpenDocumentTask={onOpenDocumentTask} onCreateRound={vi.fn()} onCreateTask={vi.fn()} onOpenRound={vi.fn()} currentUserId={7} canManage />);
    fireEvent.click(screen.getByRole("button", { name: "Open Document entities" }));
    expect(onOpenDocumentTask).toHaveBeenCalledWith(documentTask);
  });

  it("recovers a failed draft opening and allows cancellation", async () => {
    const data = projectData();
    data.rounds = data.rounds.map((round) => ({ ...round, status: "draft" }));
    const refresh = vi.fn().mockResolvedValue(undefined);
    roundMocks.transitionRound.mockRejectedValueOnce(new Error("Opening failed")).mockResolvedValue({});
    render(<AnnotateScreen view="rounds" data={data} projectId={1} onCreateRound={vi.fn()} onCreateTask={vi.fn()} onOpenRound={vi.fn()} currentUserId={7} canManage onRefresh={refresh} />);
    fireEvent.click(screen.getByRole("button", { name: "Open draft Initial labels" }));
    expect(await screen.findByRole("alert")).toHaveProperty("textContent", "Opening failed");
    fireEvent.click(screen.getByRole("button", { name: "Cancel draft Initial labels" }));
    await waitFor(() => expect(roundMocks.transitionRound).toHaveBeenLastCalledWith(1, 71, "cancelled"));
    await waitFor(() => expect(refresh).toHaveBeenCalledOnce());
  });

  it("presents separate task and round administration surfaces", () => {
    const onCreateRound = vi.fn();
    const onCreateTask = vi.fn();
    const onOpenRound = vi.fn();

    const { rerender } = render(
      <AnnotateScreen
        view="tasks"
        data={projectData()}
        onCreateRound={onCreateRound}
        onCreateTask={onCreateTask}
        onOpenRound={onOpenRound}
        currentUserId={7}
        canManage
      />,
    );

    expect(
      screen.getByRole("heading", { level: 1, name: "Tasks" }),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "New task" }));
    expect(onCreateTask).toHaveBeenCalledTimes(1);

    rerender(
      <AnnotateScreen
        view="rounds"
        data={projectData()}
        onCreateRound={onCreateRound}
        onCreateTask={onCreateTask}
        onOpenRound={onOpenRound}
        currentUserId={7}
        canManage
      />,
    );

    expect(
      screen.getByRole("heading", { level: 1, name: "Team & Rounds" }),
    ).toBeTruthy();
    expect(screen.queryByText("Assistance policy")).toBeNull();
    expect(screen.queryByRole("columnheader", { name: /assistance/i })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "New round" }));
    fireEvent.click(screen.getByRole("button", { name: "Annotate" }));
    expect(onCreateRound).toHaveBeenCalledTimes(1);
    expect(onOpenRound).toHaveBeenCalledWith(71);
  });

  it("hides manager commands from annotators", () => {
    render(
      <AnnotateScreen
        view="rounds"
        data={projectData()}
        onCreateRound={vi.fn()}
        onCreateTask={vi.fn()}
        onOpenRound={vi.fn()}
        currentUserId={7}
        canManage={false}
      />,
    );

    expect(screen.queryByRole("button", { name: "New round" })).toBeNull();
    expect(screen.getByRole("button", { name: "Annotate" })).toBeTruthy();
    expect(screen.queryByRole("heading", { name: "Model scoring" })).toBeNull();
  });

  it("shows model scoring only when the caller has the scoring capability", () => {
    const onCreateScore = vi.fn();
    render(
      <AnnotateScreen
        view="rounds"
        data={projectData()}
        onCreateRound={vi.fn()}
        onCreateTask={vi.fn()}
        onOpenRound={vi.fn()}
        currentUserId={7}
        canManage
        canScore
        projectId={1}
        onCreateScore={onCreateScore}
        onRefresh={vi.fn().mockResolvedValue(undefined)}
      />,
    );

    expect(screen.getByRole("heading", { name: "Model scoring" })).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Score with model" }));
    expect(onCreateScore).toHaveBeenCalledTimes(1);
  });
});
