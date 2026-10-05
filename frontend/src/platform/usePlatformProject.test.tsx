// @vitest-environment jsdom

import { act, cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { WorkspaceRole } from "@/types/api";

import type { RoundDraft } from "./api";
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

    expect(mocks.createRound).toHaveBeenCalledWith(7, draft);
    expect(mocks.loadPlatformProject).toHaveBeenCalledTimes(2);
    expect(mocks.loadPlatformProject).toHaveBeenLastCalledWith(7, "rounds", 5, roles);
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBeNull();
  });
});
