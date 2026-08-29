import { useCallback, useEffect, useRef, useState } from "react";

import {
  createCycle,
  createDatasetWithVersion,
  createGuidelineWithRevision,
  createRound,
  createTaskWithVersion,
  createTrainingDataset,
  launchTrainingRun,
  loadPlatformProject,
  scoreFeedbackRun,
  updateProjectModules,
  type CycleDraft,
  type DatasetDraft,
  type FeedbackScoringDraft,
  type GuidelineDraft,
  type PlatformLoadScope,
  type RoundDraft,
  type TaskDraft,
  type TrainingDataDraft,
} from "./api";
import type { TrainingLaunchDraft } from "./TrainingScreen";
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
): {
  data: PlatformProjectData;
  loading: boolean;
  busy: boolean;
  error: string | null;
  reload: () => Promise<void>;
  addDataset: (draft: DatasetDraft) => Promise<void>;
  addCycle: (draft: CycleDraft) => Promise<void>;
  addRound: (draft: RoundDraft) => Promise<void>;
  scoreFeedback: (draft: FeedbackScoringDraft) => Promise<void>;
  addGuideline: (draft: GuidelineDraft) => Promise<void>;
  addTask: (draft: TaskDraft) => Promise<void>;
  prepareTrainingData: (draft: TrainingDataDraft) => Promise<void>;
  launch: (draft: TrainingLaunchDraft) => Promise<void>;
  setModules: (selected: ProjectModule[]) => Promise<ProjectModules>;
} {
  const [data, setData] = useState(EMPTY_PLATFORM_PROJECT_DATA);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadRequestIdRef = useRef(0);
  const mutationRequestIdRef = useRef(0);
  const contextKey = `${enabled}:${workspaceId ?? "none"}:${projectId ?? "none"}:${scope}`;
  const contextKeyRef = useRef(contextKey);
  contextKeyRef.current = contextKey;

  const reload = useCallback(async (): Promise<void> => {
    if (contextKeyRef.current !== contextKey) return;
    const requestId = ++loadRequestIdRef.current;
    const isCurrentRequest = (): boolean =>
      requestId === loadRequestIdRef.current &&
      contextKey === contextKeyRef.current;
    if (!enabled || projectId === null || workspaceId === null) {
      if (isCurrentRequest()) {
        setData(EMPTY_PLATFORM_PROJECT_DATA);
        setLoading(false);
        setError(null);
      }
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const projectData = await loadPlatformProject(
        projectId,
        scope,
        workspaceId,
      );
      if (isCurrentRequest()) {
        setData(projectData);
      }
    } catch (caught) {
      if (isCurrentRequest()) {
        setError(caught instanceof Error ? caught.message : "Unable to load project resources.");
      }
    } finally {
      if (isCurrentRequest()) {
        setLoading(false);
      }
    }
  }, [contextKey, enabled, projectId, scope, workspaceId]);

  useEffect(() => {
    void reload();
    return () => {
      loadRequestIdRef.current += 1;
    };
  }, [reload]);

  const mutate = useCallback(
    async (operation: (activeProjectId: number) => Promise<unknown>): Promise<void> => {
      if (projectId === null) throw new Error("Select a project first.");
      if (contextKeyRef.current !== contextKey) return;
      const requestId = ++mutationRequestIdRef.current;
      const isCurrentRequest = (): boolean =>
        requestId === mutationRequestIdRef.current &&
        contextKey === contextKeyRef.current;
      setBusy(true);
      setError(null);
      try {
        await operation(projectId);
        if (isCurrentRequest()) {
          await reload();
        }
      } catch (caught) {
        const message =
          caught instanceof Error ? caught.message : "The project update could not be completed.";
        if (isCurrentRequest()) {
          setError(message);
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
        if (isCurrentRequest()) {
          await reload();
        }
        return configuration;
      } catch (caught) {
        const message =
          caught instanceof Error
            ? caught.message
            : "The project modules could not be updated.";
        if (isCurrentRequest()) {
          setError(message);
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
    addDataset: (draft) => mutate((id) => createDatasetWithVersion(id, draft)),
    addCycle: (draft) => mutate((id) => createCycle(id, draft)),
    addRound: (draft) => mutate((id) => createRound(id, draft)),
    scoreFeedback: (draft) => mutate((id) => scoreFeedbackRun(id, draft)),
    addGuideline: (draft) => mutate((id) => createGuidelineWithRevision(id, draft)),
    addTask: (draft) => mutate((id) => createTaskWithVersion(id, draft)),
    prepareTrainingData: (draft) => mutate((id) => createTrainingDataset(id, draft)),
    launch: (draft) => mutate((id) => launchTrainingRun(id, draft, data)),
    setModules,
  };
}
