import { useCallback, useRef } from "react";

import { createDatasetWithVersion, createTaskWithVersion, type DatasetDraft, type TaskDraft } from "./api";
import type { Dataset, DatasetVersion, TaskVersion } from "./types";

function description(value: string | null): string | null {
  return value?.trim() || null;
}

function purposes(value: Dataset["purposes"]): string {
  return JSON.stringify([...new Set(value ?? [])].sort());
}

interface PendingTask {
  id: number;
  name: string;
  description: string | null;
}

interface PendingDataset {
  id: number;
  sourceType: Dataset["source_type"];
  description: string | null;
  purposes: string;
}

export function useResourceCreation(projectId: number | null, workspaceId: number | null): {
  createDataset: (draft: DatasetDraft) => Promise<DatasetVersion>;
  createTask: (draft: TaskDraft) => Promise<TaskVersion>;
} {
  // Keep successful parent creations until their version succeeds. Section
  // reloads and edits to version fields must not issue another parent POST.
  const tasks = useRef(new Map<string, PendingTask>());
  const datasets = useRef(new Map<string, PendingDataset>());
  const contextKey = JSON.stringify([workspaceId, projectId]);
  const currentContext = useRef(contextKey);
  currentContext.current = contextKey;

  const requireProject = useCallback((): number => {
    if (projectId === null) throw new Error("Select a project first.");
    if (workspaceId === null) throw new Error("Select a workspace first.");
    if (currentContext.current !== contextKey) {
      throw new Error("The active project changed. Try this update again.");
    }
    return projectId;
  }, [contextKey, projectId, workspaceId]);

  const createTask = useCallback(async (draft: TaskDraft): Promise<TaskVersion> => {
    const activeProjectId = requireProject();
    const key = JSON.stringify([contextKey, draft.key.trim()]);
    const pending = tasks.current.get(key);
    if (pending && (pending.name !== draft.name.trim() || pending.description !== description(draft.description))) {
      throw new Error("This task key was already created with a different name or description. Restore the original name and description to retry, or use a new task key.");
    }
    let definitionId = pending?.id;
    const version = await createTaskWithVersion(activeProjectId, draft, (definition) => {
      definitionId = definition.id;
      tasks.current.set(key, {
        id: definition.id,
        name: definition.name.trim(),
        description: description(definition.description),
      });
    }, pending?.id);
    if (tasks.current.get(key)?.id === definitionId) tasks.current.delete(key);
    return version;
  }, [contextKey, requireProject]);

  const createDataset = useCallback(async (draft: DatasetDraft): Promise<DatasetVersion> => {
    const activeProjectId = requireProject();
    const key = JSON.stringify([contextKey, draft.name.trim()]);
    const pending = datasets.current.get(key);
    if (pending && pending.sourceType !== draft.sourceType) {
      throw new Error(`This dataset was already created with source type "${pending.sourceType}". Restore that source type to retry, or use a new dataset name.`);
    }
    if (pending && (pending.description !== description(draft.description) || pending.purposes !== purposes(draft.purposes))) {
      throw new Error("This dataset name was already created with a different description or purposes. Restore the original description and purposes to retry, or use a new dataset name.");
    }
    let definitionId = pending?.id;
    const version = await createDatasetWithVersion(activeProjectId, draft, (definition) => {
      definitionId = definition.id;
      datasets.current.set(key, {
        id: definition.id,
        sourceType: definition.source_type,
        description: description(definition.description),
        purposes: purposes(definition.purposes),
      });
    }, pending?.id);
    if (datasets.current.get(key)?.id === definitionId) datasets.current.delete(key);
    return version;
  }, [contextKey, requireProject]);

  return { createDataset, createTask };
}
