import { useCallback, useEffect, useRef, useState } from "react";

import type { WorkspaceRole } from "@/types/api";

import {
  createRound,
  getAnnotationRound,
  transitionRound,
  loadPlatformProject,
  scoreFeedbackRun,
  updateProjectModules,
  type DatasetDraft,
  type FeedbackScoringDraft,
  type PlatformLoadScope,
  type RoundDraft,
  type TaskDraft,
} from "./api";
import { useResourceCreation } from "./useResourceCreation";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
  type ProjectModule,
  type ProjectModules,
} from "./types";

export function usePlatformProject(
  projectId: number | null,
  workspaceId: number | null,
  enabled: boolean,
  scope: PlatformLoadScope,
  effectiveRoles?: readonly WorkspaceRole[],
): {
  data: PlatformProjectData;
  loading: boolean;
  busy: boolean;
  error: string | null;
  reload: () => Promise<void>;
  addDataset: (draft: DatasetDraft) => Promise<void>;
  addRound: (draft: RoundDraft) => Promise<void>;
  scoreFeedback: (draft: FeedbackScoringDraft) => Promise<void>;
  addTask: (draft: TaskDraft) => Promise<void>;
  setModules: (selected: ProjectModule[]) => Promise<ProjectModules>;
} {
  const [loadedData, setLoadedData] = useState<{
    contextKey: string;
    data: PlatformProjectData;
  } | null>(null);
  const [loadState, setLoadState] = useState<{ contextKey: string; loading: boolean } | null>(null);
  const [busy, setBusy] = useState(false);
  const [loadError, setError] = useState<{
    contextKey: string;
    message: string;
  } | null>(null);
  const loadRequestIdRef = useRef(0);
  const mutationRequestIdRef = useRef(0);
  const roundCreation = useRef<{ contextKey: string; requestKey: string; roundId: number } | null>(null);
  const rolesKey = effectiveRoles?.join(",") ?? "";
  const contextKey = `${enabled}:${workspaceId ?? "none"}:${projectId ?? "none"}:${scope}:${rolesKey}`;
  const loading = enabled && projectId !== null && workspaceId !== null &&
    (loadState?.contextKey !== contextKey || loadState.loading);
  const contextKeyRef = useRef(contextKey);
  contextKeyRef.current = contextKey;
  const effectiveRolesRef = useRef(effectiveRoles);
  effectiveRolesRef.current = effectiveRoles;
  const data = loadedData?.contextKey === contextKey
    ? loadedData.data
    : EMPTY_PLATFORM_PROJECT_DATA;
  const error = loadError?.contextKey === contextKey ? loadError.message : null;
  const resourceCreation = useResourceCreation(projectId, workspaceId);

  const reload = useCallback(async (): Promise<void> => {
    if (contextKeyRef.current !== contextKey) return;
    const requestId = ++loadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === loadRequestIdRef.current &&
      contextKey === contextKeyRef.current;
    if (!enabled || projectId === null || workspaceId === null) {
      if (isCurrentRequest()) {
        setLoadedData(null);
        setLoadState({ contextKey, loading: false });
        setError(null);
      }
      return;
    }
    setLoadState({ contextKey, loading: true });
    setError(null);
    try {
      const projectData = await loadPlatformProject(
        projectId,
        scope,
        workspaceId,
        effectiveRolesRef.current,
      );
      if (isCurrentRequest()) {
        setLoadedData({ contextKey, data: projectData });
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError({
          contextKey,
          message: caught instanceof Error ? caught.message : "Unable to load project resources.",
        });
      }
    } finally {
      if (isCurrentRequest()) {
        setLoadState({ contextKey, loading: false });
      }
    }
  }, [contextKey, enabled, projectId, scope, workspaceId]);

  useEffect(() => {
    setLoadedData(null);
    void reload();
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [reload]);

  const mutate = useCallback(
    async (operation: (activeProjectId: number) => Promise<unknown>): Promise<void> => {
      if (projectId === null) throw new Error("Select a project first.");
      if (contextKeyRef.current !== contextKey) throw new Error("The active project changed. Review the current project before trying again.");
      const requestId = ++mutationRequestIdRef.current;
      const isCurrentRequest = (): boolean =>
        requestId === mutationRequestIdRef.current &&
        contextKey === contextKeyRef.current;
      setBusy(true);
      setError(null);
      try {
        await operation(projectId);
        if (contextKey !== contextKeyRef.current) throw new Error("The active project changed while saving. The update may have been saved in the previous project. Review it before trying again.");
        if (isCurrentRequest()) {
          await reload();
        }
        if (contextKey !== contextKeyRef.current) throw new Error("The active project changed while refreshing. Review the previous project before trying again.");
      } catch (caught) {
        const message =
          caught instanceof Error ? caught.message : "The project update could not be completed.";
        if (isCurrentRequest()) {
          setError({ contextKey, message });
        }
        throw caught;
      } finally {
        if (isCurrentRequest()) {
          setBusy(false);
        }
      }
    },
    [contextKey, projectId, reload],
  );

  const setModules = useCallback(
    async (selected: ProjectModule[]): Promise<ProjectModules> => {
      if (projectId === null) throw new Error("Select a project first.");
      if (contextKeyRef.current !== contextKey) {
        throw new Error("The active project changed. Try this update again.");
      }
      const requestId = ++mutationRequestIdRef.current;
      const isCurrentRequest = (): boolean =>
        requestId === mutationRequestIdRef.current &&
        contextKey === contextKeyRef.current;
      setBusy(true);
      setError(null);
      try {
        const configuration = await updateProjectModules(projectId, selected);
        if (contextKey !== contextKeyRef.current) throw new Error("The active project changed while saving. Review the previous project's modules before trying again.");
        if (isCurrentRequest()) {
          await reload();
        }
        if (contextKey !== contextKeyRef.current) throw new Error("The active project changed while refreshing. Review the previous project's modules before trying again.");
        return configuration;
      } catch (caught) {
        const message =
          caught instanceof Error
            ? caught.message
            : "The project modules could not be updated.";
        if (isCurrentRequest()) {
          setError({ contextKey, message });
        }
        throw caught;
      } finally {
        if (isCurrentRequest()) {
          setBusy(false);
        }
      }
    },
    [contextKey, projectId, reload],
  );

  useEffect(() => {
    mutationRequestIdRef.current += 1;
    setBusy(false);
  }, [contextKey]);

  return {
    data,
    loading,
    busy,
    error,
    reload,
    addDataset: (draft) => mutate(() => resourceCreation.createDataset(draft)),
    addRound: (draft) => mutate(async (id) => {
      const requestKey = JSON.stringify(draft);
      const previous = roundCreation.current;
      try {
        if (previous?.contextKey === contextKey && previous.requestKey === requestKey) {
          const saved = await getAnnotationRound(previous.roundId);
          if (saved.status === "draft") await transitionRound(id, saved.id, "open");
          else if (saved.status !== "open") throw new Error("This saved round is no longer a draft. Review it in Team & Rounds.");
        } else {
          await createRound(id, draft, (round) => {
            roundCreation.current = { contextKey, requestKey, roundId: round.id };
          });
        }
        roundCreation.current = null;
      } catch (caught) {
        if (roundCreation.current?.contextKey === contextKey) await reload();
        throw caught;
      }
    }),
    scoreFeedback: (draft) => mutate((id) => scoreFeedbackRun(id, draft)),
    addTask: (draft) => mutate(() => resourceCreation.createTask(draft)),
    setModules,
  };
}
