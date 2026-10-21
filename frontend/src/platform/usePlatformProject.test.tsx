// @vitest-environment jsdom

import { act, cleanup, render, renderHook, screen, waitFor } from "@testing-library/react";
import { Profiler } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { WorkspaceRole } from "@/types/api";

import type { DatasetDraft, RoundDraft, TaskDraft } from "./api";
import type { Dataset, TaskDefinition } from "./types";
import { EMPTY_PLATFORM_PROJECT_DATA } from "./types";
import { usePlatformProject } from "./usePlatformProject";

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
} {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((nextResolve, nextReject) => {
    resolve = nextResolve;
    reject = nextReject;
  });
  return { promise, resolve, reject };
}

const mocks = vi.hoisted(() => ({
  createCycle: vi.fn(),
  createDatasetWithVersion: vi.fn(),
  createGuidelineWithRevision: vi.fn(),
  createRound: vi.fn(),
  getAnnotationRound: vi.fn(),
  transitionRound: vi.fn(),
  createTaskWithVersion: vi.fn(),
  createTrainingDataset: vi.fn(),
  launchTrainingRun: vi.fn(),
  loadPlatformProject: vi.fn(),
  updateProjectModules: vi.fn(),
}));

vi.mock("./api", () => mocks);

beforeEach(() => {
  vi.resetAllMocks();
  mocks.loadPlatformProject.mockImplementation(
    async (projectId: number, scope: string) => ({
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
        project_id: projectId,
        selected: scope === "data" ? ["data"] : ["train"],
        effective: scope === "data" ? ["data"] : ["train"],
      },
    }),
  );
});

afterEach(cleanup);

describe("usePlatformProject load scope", () => {
  it("passes the active scope and reloads when the section changes", async () => {
    const { result, rerender } = renderHook(
      ({ scope }: { scope: "data" | "train" }) =>
        usePlatformProject(7, 5, true, scope),
      { initialProps: { scope: "data" as "data" | "train" } },
    );

    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(7, "data", 5, undefined),
    );
    await waitFor(() =>
      expect(result.current.data.projectModules.effective).toEqual(["data"]),
    );

    rerender({ scope: "train" });

    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "train", 5, undefined),
    );
    await waitFor(() =>
      expect(result.current.data.projectModules.effective).toEqual(["train"]),
    );

    await act(() => result.current.reload());
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "train", 5, undefined);
  });

  it("does not load section data while disabled", async () => {
    const { result } = renderHook(() =>
      usePlatformProject(7, 5, false, "overview"),
    );

    await act(() => result.current.reload());

    expect(mocks.loadPlatformProject).not.toHaveBeenCalled();
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);
  });

  it("exposes initial and retry failures and recovers when a later reload succeeds", async () => {
    const failedRetry = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    const successfulRetry = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    const projectData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
        project_id: 7,
      },
    };
    mocks.loadPlatformProject
      .mockRejectedValueOnce(new Error("Initial project load failed"))
      .mockReturnValueOnce(failedRetry.promise)
      .mockReturnValueOnce(successfulRetry.promise);
    const { result } = renderHook(() => usePlatformProject(7, 5, true, "data"));

    await waitFor(() => expect(result.current.error).toBe("Initial project load failed"));
    expect(result.current.loading).toBe(false);
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);

    let retryPromise!: Promise<void>;
    act(() => { retryPromise = result.current.reload(); });

    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);
    await act(async () => {
      failedRetry.reject(new Error("Project retry failed"));
      await retryPromise;
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe("Project retry failed");
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);

    act(() => { retryPromise = result.current.reload(); });

    expect(result.current.loading).toBe(true);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);
    expect(mocks.loadPlatformProject).toHaveBeenCalledTimes(3);
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "data", 5, undefined);
    await act(async () => {
      successfulRetry.resolve(projectData);
      await retryPromise;
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.data).toBe(projectData);
  });

  it.each([
    { description: "project", projectId: 8, scope: "data" as const },
    { description: "section", projectId: 7, scope: "rounds" as const },
  ])("does not commit the previous error when the $description changes", async ({ projectId, scope }) => {
    const nextLoad = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    mocks.loadPlatformProject
      .mockRejectedValueOnce(new Error("Previous project context failed"))
      .mockReturnValueOnce(nextLoad.promise);
    function ProjectStatus({ projectId, scope }: { projectId: number; scope: "data" | "rounds" }) {
      const project = usePlatformProject(projectId, 5, true, scope);
      return <p>{project.error ?? "No project error"}</p>;
    }
    const committedPreviousErrors: boolean[] = [];
    const onRender = () => {
      committedPreviousErrors.push(screen.queryByText("Previous project context failed") !== null);
    };
    const { rerender } = render(
      <Profiler id="project" onRender={onRender}>
        <ProjectStatus projectId={7} scope="data" />
      </Profiler>,
    );
    await screen.findByText("Previous project context failed");
    committedPreviousErrors.length = 0;

    rerender(
      <Profiler id="project" onRender={onRender}>
        <ProjectStatus projectId={projectId} scope={scope} />
      </Profiler>,
    );

    expect(committedPreviousErrors.length).toBeGreaterThan(0);
    expect(committedPreviousErrors).not.toContain(true);
    expect(screen.getByText("No project error")).toBeTruthy();
  });

  it("ignores an older project response and its finally state", async () => {
    const first = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    const second = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    mocks.loadPlatformProject.mockImplementation((projectId: number) =>
      projectId === 7 ? first.promise : second.promise,
    );
    const projectData = (projectId: number) => ({
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
        project_id: projectId,
      },
    });

    const { result, rerender } = renderHook(
      ({ projectId }: { projectId: number }) =>
        usePlatformProject(projectId, 5, true, "data"),
      { initialProps: { projectId: 7 } },
    );

    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(7, "data", 5, undefined),
    );
    rerender({ projectId: 8 });
    await waitFor(() =>
      expect(mocks.loadPlatformProject).toHaveBeenCalledWith(8, "data", 5, undefined),
    );

    await act(async () => {
      first.resolve(projectData(7));
      await first.promise;
    });
    expect(result.current.loading).toBe(true);
    expect(result.current.data.projectModules.project_id).not.toBe(7);

    await act(async () => {
      second.resolve(projectData(8));
      await second.promise;
    });
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data.projectModules.project_id).toBe(8);
    expect(result.current.error).toBeNull();
  });

  it("clears the previous section's data while navigation loads and after failure", async () => {
    const roundsLoad = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    const previousData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      projectModules: {
        ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
        project_id: 7,
      },
    };
    mocks.loadPlatformProject
      .mockResolvedValueOnce(previousData)
      .mockReturnValueOnce(roundsLoad.promise);
    const { result, rerender } = renderHook(
      ({ scope }: { scope: "data" | "rounds" }) =>
        usePlatformProject(7, 5, true, scope),
      { initialProps: { scope: "data" as "data" | "rounds" } },
    );
    await waitFor(() => expect(result.current.data).toBe(previousData));

    rerender({ scope: "rounds" });

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);

    await act(async () => {
      roundsLoad.reject(new Error("Project data could not be loaded"));
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBe("Project data could not be loaded");
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);
  });

  it("reloads with changed roles and ignores a response from the previous access context", async () => {
    const managerRoles: readonly WorkspaceRole[] = ["annotator", "trainer", "manager"];
    const trainerRoles: readonly WorkspaceRole[] = ["annotator", "trainer"];
    const managerLoad = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    const trainerLoad = deferred<typeof EMPTY_PLATFORM_PROJECT_DATA>();
    mocks.loadPlatformProject
      .mockReturnValueOnce(managerLoad.promise)
      .mockReturnValueOnce(trainerLoad.promise);
    const { result, rerender } = renderHook(
      ({ roles }: { roles: readonly WorkspaceRole[] }) =>
        usePlatformProject(7, 5, true, "rounds", roles),
      { initialProps: { roles: managerRoles } },
    );
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "rounds", 5, managerRoles);

    rerender({ roles: trainerRoles });
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "rounds", 5, trainerRoles);

    await act(async () => {
      managerLoad.resolve({
        ...EMPTY_PLATFORM_PROJECT_DATA,
        projectModules: {
          ...EMPTY_PLATFORM_PROJECT_DATA.projectModules,
          project_id: 7,
        },
      });
    });
    expect(result.current.data).toBe(EMPTY_PLATFORM_PROJECT_DATA);
    expect(result.current.loading).toBe(true);

    await act(async () => {
      trainerLoad.resolve(EMPTY_PLATFORM_PROJECT_DATA);
    });
    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();

    rerender({ roles: [...trainerRoles] });
    expect(mocks.loadPlatformProject).toHaveBeenCalledTimes(2);
  });

  it("reloads a saved draft after opening fails and preserves the original error", async () => {
    mocks.createRound.mockImplementation(async (_id: number, _draft: RoundDraft, onCreated?: (round: { id: number }) => void) => { onCreated?.({ id: 71 }); throw new Error("Opening failed"); });
    const { result } = renderHook(() => usePlatformProject(7, 5, true, "rounds"));
    await waitFor(() => expect(result.current.loading).toBe(false));
    const draft: RoundDraft = { name: "Recover me", datasetVersionId: 2, taskVersionId: 3, cycleId: null, splitMapId: null, guidelineRevisionId: null, feedbackSetVersionId: null, assistancePolicy: "blind", reannotationMode: "full_dataset", selectionStrategy: "all", selectionLimit: 1, annotatorUserIds: [7], openToAllAnnotators: false, reason: "" };
    await act(async () => { await expect(result.current.addRound(draft)).rejects.toThrow("Opening failed"); });
    expect(mocks.loadPlatformProject).toHaveBeenCalledTimes(2);
    expect(result.current.error).toBe("Opening failed");
    mocks.getAnnotationRound.mockResolvedValue({ id: 71, status: "draft" });
    mocks.transitionRound.mockResolvedValue({ id: 71, status: "open" });
    await act(() => result.current.addRound(draft));
    expect(mocks.createRound).toHaveBeenCalledOnce();
    expect(mocks.transitionRound).toHaveBeenCalledWith(7, 71, "open");
    expect(result.current.error).toBeNull();
  });

  it("retains the active roles when reloading after a mutation", async () => {
    const roles: readonly WorkspaceRole[] = ["annotator", "trainer", "manager"];
    mocks.createRound.mockResolvedValueOnce({ id: 21 });
    const { result } = renderHook(() =>
      usePlatformProject(7, 5, true, "rounds", roles),
    );
    await waitFor(() => expect(result.current.loading).toBe(false));
    const draft: RoundDraft = {
      name: "Round 2",
      datasetVersionId: 2,
      taskVersionId: 3,
      cycleId: null,
      splitMapId: null,
      guidelineRevisionId: null,
      feedbackSetVersionId: null,
      assistancePolicy: "reveal_after_first_pass",
      reannotationMode: "targeted_subset",
      selectionStrategy: "all",
      selectionLimit: 0,
      annotatorUserIds: [],
      openToAllAnnotators: true,
      reason: "Follow-up review",
    };

    await act(() => result.current.addRound(draft));

    expect(mocks.createRound).toHaveBeenCalledWith(7, draft, expect.any(Function));
    expect(mocks.loadPlatformProject).toHaveBeenCalledTimes(2);
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "rounds", 5, roles);
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBeNull();
  });
});

describe("usePlatformProject resource creation retries", () => {
  const task: TaskDraft = {
    key: "screening",
    name: "Screening",
    description: "Clinical review",
    taskKind: "classification",
    labelValues: ["Include", "Exclude"],
  };
  const taskDefinition: TaskDefinition = {
    id: 31,
    project_id: 7,
    key: task.key,
    name: task.name,
    description: task.description,
  };
  const dataset: DatasetDraft = {
    name: "Clinical papers",
    description: "Imported abstracts",
    sourceType: "upload",
    sourceUri: "",
    sourceRevision: "revision-1",
    sourceFormat: "csv",
    license: "",
    file: new File(["bad input"], "papers.csv", { type: "text/csv" }),
    stableKeyField: "id",
    groupKeyField: "",
    registryDatasetId: "",
    registryConfigName: "",
    purposes: ["training_source"],
  };
  const datasetDefinition: Dataset = {
    id: 41,
    project_id: 7,
    name: dataset.name,
    description: dataset.description,
    source_type: dataset.sourceType,
    purposes: dataset.purposes,
  };

  it("reuses a created task definition after correcting a failed version and clears it after success", async () => {
    mocks.createTaskWithVersion.mockImplementationOnce(async (
      _projectId: number, _draft: TaskDraft, onCreated?: (definition: TaskDefinition) => void,
    ) => {
      onCreated?.(taskDefinition);
      throw new Error("Task version rejected");
    }).mockResolvedValue({ id: 51 });
    const { result } = renderHook(() => usePlatformProject(7, 5, true, "data"));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await expect(result.current.addTask(task)).rejects.toThrow("Task version rejected");
    });
    await act(() => result.current.reload());
    const corrected = { ...task, key: " screening ", labelValues: ["Relevant", "Not relevant"] };
    await act(() => result.current.addTask(corrected));

    expect(mocks.createTaskWithVersion).toHaveBeenNthCalledWith(2, 7, corrected, expect.any(Function), 31);
    await act(() => result.current.addTask(task));
    expect(mocks.createTaskWithVersion).toHaveBeenNthCalledWith(3, 7, task, expect.any(Function), undefined);
  });

  it("reuses a created dataset after replacing a failed file and revision, then clears it after success", async () => {
    mocks.createDatasetWithVersion.mockImplementationOnce(async (
      _projectId: number, _draft: DatasetDraft, onCreated?: (definition: Dataset) => void,
    ) => {
      onCreated?.(datasetDefinition);
      throw new Error("Dataset version rejected");
    }).mockResolvedValue({ id: 61 });
    const { result } = renderHook(() => usePlatformProject(7, 5, true, "data"));
    await waitFor(() => expect(result.current.loading).toBe(false));

    await act(async () => {
      await expect(result.current.addDataset(dataset)).rejects.toThrow("Dataset version rejected");
    });
    await act(() => result.current.reload());
    const corrected = { ...dataset, name: " Clinical papers ", sourceRevision: "revision-2",
      file: new File(["id,text\n1,abstract"], "fixed.csv", { type: "text/csv" }) };
    await act(() => result.current.addDataset(corrected));

    expect(mocks.createDatasetWithVersion).toHaveBeenNthCalledWith(2, 7, corrected, expect.any(Function), 41);
    await act(() => result.current.addDataset(dataset));
    expect(mocks.createDatasetWithVersion).toHaveBeenNthCalledWith(3, 7, dataset, expect.any(Function), undefined);
  });
});

it("retries project launches with one attempt and handles the displayed failure", async () => {
  const draft = { modelName: "Baseline", taskVersionId: 1, trainingDatasetVersionId: 2,
    recipeKey: "linear", recipeVersionId: null, environmentId: 3, storagePolicyId: 4,
    seed: 42, config: {}, evaluationPlan: {} };
  mocks.launchTrainingRun.mockRejectedValueOnce(new Error("Response lost")).mockResolvedValue({ id: 151 });
  const { result, rerender } = renderHook(({ projectId }) => usePlatformProject(projectId, 5, true, "train"),
    { initialProps: { projectId: 7 } });
  await waitFor(() => expect(result.current.loading).toBe(false));
  await act(async () => { await expect(result.current.launch(draft)).resolves.toBeUndefined(); });
  expect(result.current.error).toBe("Response lost");
  await act(() => result.current.launch(draft));
  const firstAttempt = mocks.launchTrainingRun.mock.calls[0]?.[3];
  expect(firstAttempt).toMatchObject({ key: expect.any(String) });
  expect(mocks.launchTrainingRun.mock.calls[1]?.[3]).toBe(firstAttempt);
  await act(() => result.current.launch({ ...draft, seed: 73 }));
  expect(mocks.launchTrainingRun.mock.calls[2]?.[3].key).not.toBe(firstAttempt.key);
  rerender({ projectId: 8 });
  await act(() => result.current.launch(draft));
  expect(mocks.launchTrainingRun.mock.calls[3]?.[3].key).not.toBe(firstAttempt.key);
});
