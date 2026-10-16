// @vitest-environment jsdom

import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DatasetDraft, TaskDraft } from "./api";
import type { Dataset, TaskDefinition } from "./types";
import { useResourceCreation } from "./useResourceCreation";

const mocks = vi.hoisted(() => ({
  createDatasetWithVersion: vi.fn(),
  createTaskWithVersion: vi.fn(),
}));
vi.mock("./api", () => mocks);

const task: TaskDraft = {
  key: "screening",
  name: "Screening",
  description: "Clinical review",
  taskKind: "classification",
  labelValues: ["Include", "Exclude"],
};
const dataset: DatasetDraft = {
  name: "Clinical papers",
  description: "Imported abstracts",
  sourceType: "upload",
  sourceUri: "",
  sourceRevision: "revision-1",
  sourceFormat: "csv",
  license: "",
  file: new File(["id,text\n1,abstract"], "papers.csv", { type: "text/csv" }),
  stableKeyField: "id",
  groupKeyField: "",
  registryDatasetId: "",
  registryConfigName: "",
  purposes: ["annotation", "training_source"],
};

type Kind = "task" | "dataset";
type Creation = ReturnType<typeof useResourceCreation>;

function create(kind: Kind, creation: Creation, identity?: string): Promise<unknown> {
  return kind === "task"
    ? creation.createTask({ ...task, key: identity ?? task.key })
    : creation.createDataset({ ...dataset, name: identity ?? dataset.name });
}

function api(kind: Kind): typeof mocks.createTaskWithVersion {
  return kind === "task" ? mocks.createTaskWithVersion : mocks.createDatasetWithVersion;
}

function failAfterParent(kind: Kind): void {
  if (kind === "task") {
    mocks.createTaskWithVersion.mockImplementationOnce(async (
      projectId: number, draft: TaskDraft, onCreated: (definition: TaskDefinition) => void,
    ) => {
      onCreated({ id: 31, project_id: projectId, key: draft.key.trim(), name: draft.name.trim(), description: draft.description.trim() || null });
      throw new Error("Version failed");
    });
  } else {
    mocks.createDatasetWithVersion.mockImplementationOnce(async (
      projectId: number, draft: DatasetDraft, onCreated: (definition: Dataset) => void,
    ) => {
      onCreated({ id: 31, project_id: projectId, name: draft.name.trim(), description: draft.description.trim() || null,
        source_type: draft.sourceType, purposes: draft.purposes });
      throw new Error("Version failed");
    });
  }
}

beforeEach(() => {
  vi.resetAllMocks();
  mocks.createTaskWithVersion.mockResolvedValue({ id: 51 });
  mocks.createDatasetWithVersion.mockResolvedValue({ id: 61 });
});

describe("useResourceCreation pending parents", () => {
  it.each<Kind>(["task", "dataset"])("starts a new %s identity without forgetting the failed one", async (kind) => {
    failAfterParent(kind);
    const { result } = renderHook(() => useResourceCreation(7, 5));
    await expect(create(kind, result.current)).rejects.toThrow("Version failed");

    await create(kind, result.current, "different_resource");
    expect(api(kind).mock.calls[1][3]).toBeUndefined();
    await create(kind, result.current);
    expect(api(kind).mock.calls[2][3]).toBe(31);
  });

  it.each([
    { kind: "task", scope: "project", projectId: 8, workspaceId: 5 },
    { kind: "task", scope: "workspace", projectId: 7, workspaceId: 6 },
    { kind: "dataset", scope: "project", projectId: 8, workspaceId: 5 },
    { kind: "dataset", scope: "workspace", projectId: 7, workspaceId: 6 },
  ] as const)("isolates a pending $kind when the $scope changes and retains it on return", async ({ kind, projectId, workspaceId }) => {
    failAfterParent(kind);
    const { result, rerender } = renderHook(
      ({ projectId, workspaceId }) => useResourceCreation(projectId, workspaceId),
      { initialProps: { projectId: 7, workspaceId: 5 } },
    );
    await expect(create(kind, result.current)).rejects.toThrow("Version failed");

    rerender({ projectId, workspaceId });
    await create(kind, result.current);
    expect(api(kind).mock.calls[1][0]).toBe(projectId);
    expect(api(kind).mock.calls[1][3]).toBeUndefined();
    rerender({ projectId: 7, workspaceId: 5 });
    await create(kind, result.current);
    expect(api(kind).mock.calls[2][3]).toBe(31);
  });

  it.each<Kind>(["task", "dataset"])("records a late %s parent only under the project that created it", async (kind) => {
    let recordParent!: () => void;
    let failVersion!: (error: Error) => void;
    const version = new Promise<never>((_resolve, reject) => { failVersion = reject; });
    if (kind === "task") {
      mocks.createTaskWithVersion.mockImplementationOnce((projectId: number, draft: TaskDraft, onCreated: (definition: TaskDefinition) => void) => {
        recordParent = () => onCreated({ id: 31, project_id: projectId, key: draft.key, name: draft.name, description: draft.description });
        return version;
      });
    } else {
      mocks.createDatasetWithVersion.mockImplementationOnce((projectId: number, draft: DatasetDraft, onCreated: (definition: Dataset) => void) => {
        recordParent = () => onCreated({ id: 31, project_id: projectId, name: draft.name, description: draft.description,
          source_type: draft.sourceType, purposes: draft.purposes });
        return version;
      });
    }
    const { result, rerender } = renderHook(({ projectId }) => useResourceCreation(projectId, 5), {
      initialProps: { projectId: 7 },
    });
    const firstCreation = create(kind, result.current);
    rerender({ projectId: 8 });
    recordParent();
    await create(kind, result.current);
    expect(api(kind).mock.calls[1][3]).toBeUndefined();
    failVersion(new Error("Late version failure"));
    await expect(firstCreation).rejects.toThrow("Late version failure");

    rerender({ projectId: 7 });
    await create(kind, result.current);
    expect(api(kind).mock.calls[2][3]).toBe(31);
  });

  it("rejects changed task parent metadata while allowing corrected labels", async () => {
    failAfterParent("task");
    const { result } = renderHook(() => useResourceCreation(7, 5));
    await expect(result.current.createTask(task)).rejects.toThrow("Version failed");

    await expect(result.current.createTask({ ...task, name: "Changed name" })).rejects.toThrow("Restore the original name and description");
    expect(mocks.createTaskWithVersion).toHaveBeenCalledTimes(1);
    await result.current.createTask({ ...task, name: ` ${task.name} `, description: ` ${task.description} `, labelValues: ["New label"] });
    expect(mocks.createTaskWithVersion.mock.calls[1][3]).toBe(31);
  });

  it("rejects changing an already-created dataset's type or metadata without issuing another parent request", async () => {
    failAfterParent("dataset");
    const { result } = renderHook(() => useResourceCreation(7, 5));
    await expect(result.current.createDataset(dataset)).rejects.toThrow("Version failed");

    await expect(result.current.createDataset({ ...dataset, sourceType: "project_corpus" })).rejects.toThrow('source type "upload"');
    await expect(result.current.createDataset({ ...dataset, description: "Different metadata" })).rejects.toThrow("Restore the original description and purposes");
    expect(mocks.createDatasetWithVersion).toHaveBeenCalledTimes(1);
    await result.current.createDataset({ ...dataset, description: ` ${dataset.description} `,
      purposes: ["training_source", "annotation", "annotation"], sourceRevision: "corrected-revision" });
    expect(mocks.createDatasetWithVersion.mock.calls[1][3]).toBe(31);
  });
});
