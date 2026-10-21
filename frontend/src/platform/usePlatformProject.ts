import { useCallback, useEffect, useRef, useState } from "react";

import type { WorkspaceRole } from "@/types/api";

import {
  createCycle,
  createGuidelineWithRevision,
  createRound,
  getAnnotationRound,
  transitionRound,
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
  type TrainingLaunchAttempt,
} from "./api";
import type { TrainingLaunchDraft } from "./TrainingScreen";
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
  addCycle: (draft: CycleDraft) => Promise<void>;
  addRound: (draft: RoundDraft) => Promise<void>;
  scoreFeedback: (draft: FeedbackScoringDraft) => Promise<void>;
  addGuideline: (draft: GuidelineDraft) => Promise<void>;
  addTask: (draft: TaskDraft) => Promise<void>;
  prepareTrainingData: (draft: TrainingDataDraft) => Promise<void>;
  launch: (draft: TrainingLaunchDraft) => Promise<void>;
  setModules: (selected: ProjectModule[]) => Promise<ProjectModules>;
} {
  const [loadedData, setLoadedData] = useState<{
    contextKey: string;
    data: PlatformProjectData;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [loadError, setError] = useState<{
    contextKey: string;
    message: string;
  } | null>(null);
  const loadRequestIdRef = useRef(0);
  const mutationRequestIdRef = useRef(0);
  const launchAttempt = useRef<{ signature: string; attempt: TrainingLaunchAttempt } | null>(null);
  const roundCreation = useRef<{ contextKey: string; requestKey: string; roundId: number } | null>(null);
  const rolesKey = effectiveRoles?.join(",") ?? "";
  const contextKey = `${enabled}:${workspaceId ?? "none"}:${projectId ?? "none"}:${scope}:${rolesKey}`;
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
        setLoading(false);
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
    addCycle: (draft) => mutate((id) => createCycle(id, draft)),
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
    addGuideline: (draft) => mutate((id) => createGuidelineWithRevision(id, draft)),
    addTask: (draft) => mutate(() => resourceCreation.createTask(draft)),
    prepareTrainingData: (draft) => mutate((id) => createTrainingDataset(id, draft)),
    launch: async (draft) => {
      const signature = JSON.stringify({ projectId, draft });
      if (launchAttempt.current?.signature !== signature) {
        launchAttempt.current = { signature, attempt: { key: crypto.randomUUID(), modelId: null, recipeVersionId: null } };
      }
      const attempt = launchAttempt.current.attempt;
      await mutate((id) => launchTrainingRun(id, draft, data, attempt)).catch(() => undefined);
    },
    setModules,
  };
}
