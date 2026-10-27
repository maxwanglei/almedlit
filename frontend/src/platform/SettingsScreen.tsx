import { NativeInput, NativeSelect, NativeTextarea } from "@/components/NativeField";
import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import ConfirmDialog from "@/components/ConfirmDialog";

import type {
  AnnotationValidationMode,
  Project,
  ProjectUpdate,
} from "@/types/api";

import { PlatformPageHeader, PlatformSection, formatStatus } from "./components";
import type { PlatformProjectData, ProjectModule } from "./types";

type ReleasedProjectModule = Exclude<ProjectModule, "learning" | "guidelines">;

const MODULE_OPTIONS: Array<{
  id: ReleasedProjectModule;
  label: string;
  description: string;
}> = [
  { id: "data", label: "Data", description: "Versioned source data and label sets." },
  { id: "annotate", label: "Annotation", description: "Tasks, rounds, and human decisions." },
  { id: "train", label: "Training", description: "Independent training workspace access." },
  { id: "models", label: "Models", description: "Named model identities and versions." },
  { id: "activity", label: "Activity", description: "Project lineage and released activity." },
];

const MODULE_DEPENDENTS: Partial<
  Record<ReleasedProjectModule, ReleasedProjectModule[]>
> = {
  data: ["annotate", "train"],
  models: ["train"],
};

const MODULE_CAPABILITIES: Record<ReleasedProjectModule, string[]> = {
  data: ["annotation", "training", "inference", "lineage"],
  annotate: ["annotation"],
  train: ["training"],
  models: ["training", "inference", "lineage"],
  activity: ["lineage"],
};

function released(module: ProjectModule): module is ReleasedProjectModule {
  return module !== "learning" && module !== "guidelines";
}

export default function SettingsScreen({
  project,
  data,
  busy,
  onUpdateProject,
  onUpdateModules,
}: {
  project: Project;
  data: PlatformProjectData;
  busy: boolean;
  onUpdateProject: (payload: ProjectUpdate) => Promise<void>;
  onUpdateModules: (selected: ProjectModule[]) => Promise<void>;
}): React.ReactElement {
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description ?? "");
  const [validationMode, setValidationMode] =
    useState<AnnotationValidationMode>(project.annotation_validation_mode);
  const [environmentId, setEnvironmentId] = useState(
    String(project.settings.default_environment_id ?? ""),
  );
  const [storagePolicyId, setStoragePolicyId] = useState(
    String(project.settings.default_storage_policy_id ?? ""),
  );
  const [evaluationSplit, setEvaluationSplit] = useState(
    String(project.settings.default_evaluation_split ?? "validation"),
  );
  const [moduleToDisable, setModuleToDisable] = useState<ReleasedProjectModule | null>(null);
  const [nameTouched, setNameTouched] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [localError, setLocalError] = useState<string | null>(null);
  const workspaceCapabilities = new Set(data.projectModules.workspace_capabilities);

  useEffect(() => {
    setName(project.name);
    setDescription(project.description ?? "");
    setValidationMode(project.annotation_validation_mode);
    setEnvironmentId(
      String(project.settings.default_environment_id ?? ""),
    );
    setStoragePolicyId(
      String(project.settings.default_storage_policy_id ?? ""),
    );
    setEvaluationSplit(
      String(project.settings.default_evaluation_split ?? "validation"),
    );
    setMessage(null);
    setLocalError(null);
    setNameTouched(false);
    setModuleToDisable(null);
  }, [project]);

  async function updateModules(module: ReleasedProjectModule, enabled: boolean): Promise<void> {
    const selected = new Set(data.projectModules.selected);
    if (enabled) {
      selected.add(module);
    } else {
      selected.delete(module);
      for (const dependent of MODULE_DEPENDENTS[module] ?? []) selected.delete(dependent);
    }
    setMessage(null);
    setLocalError(null);
    try {
      await onUpdateModules([...selected]);
      setMessage("Project modules saved.");
      setModuleToDisable(null);
    } catch (caught) {
      setLocalError(caught instanceof Error ? caught.message : "Project modules could not be saved.");
    }
  }

  async function saveIdentity(): Promise<void> {
    setNameTouched(true);
    if (!name.trim()) return;
    setMessage(null);
    setLocalError(null);
    try {
      await onUpdateProject({
        name: name.trim(),
        description: description.trim() || null,
        annotation_validation_mode: validationMode,
      });
      setMessage("Project settings saved.");
    } catch (caught) {
      setLocalError(
        caught instanceof Error
          ? caught.message
          : "Project settings could not be saved.",
      );
    }
  }

  async function saveDefaults(): Promise<void> {
    setMessage(null);
    setLocalError(null);
    try {
      await onUpdateProject({
        settings: {
          ...project.settings,
          default_environment_id: environmentId
            ? Number(environmentId)
            : null,
          default_storage_policy_id: storagePolicyId
            ? Number(storagePolicyId)
            : null,
          default_evaluation_split: evaluationSplit,
        },
      });
      setMessage("Training defaults saved.");
    } catch (caught) {
      setLocalError(
        caught instanceof Error
          ? caught.message
          : "Project defaults could not be saved.",
      );
    }
  }

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Project settings"
        description="Configuration stays with the scientific project even when work happens in independent functional workspaces."
      />
      {localError ? (
        <p className="platform-form-warning" role="alert">
          {localError}
        </p>
      ) : null}
      {message ? (
        <p className="platform-form-success" role="status">
          {message}
        </p>
      ) : null}

      <PlatformSection title="Project identity">
        <div className="platform-settings-form">
          <label>
            <span>Project name</span>
            <NativeInput
              value={name}
              maxLength={255}
              required
              aria-invalid={nameTouched && !name.trim() ? true : undefined}
              aria-describedby={nameTouched && !name.trim() ? "project-name-error" : undefined}
              onBlur={() => setNameTouched(true)}
              onChange={(event) => setName(event.target.value)}
            />
            {nameTouched && !name.trim() ? <small id="project-name-error" aria-hidden="true">Enter a project name.</small> : null}
          </label>
          <label>
            <span>Description</span>
            <NativeTextarea
              value={description}
              rows={3}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>
          <label>
            <span>Annotation validation</span>
            <NativeSelect
              value={validationMode}
              onChange={(event) =>
                setValidationMode(event.target.value as AnnotationValidationMode)
              }
            >
              <option value="relaxed">Relaxed</option>
              <option value="strict">Strict</option>
            </NativeSelect>
          </label>
          <div className="platform-settings-actions">
            <Button
              label={busy ? "Saving…" : "Save project"}
              variant="primary"
              isDisabled={busy || !name.trim()}
              onClick={() => void saveIdentity()}
            />
          </div>
        </div>
      </PlatformSection>

      <PlatformSection
        title="Enabled modules"
        description="Training and Models open in their own workspaces. Module selection controls this project's data and authorization scope."
      >
        <fieldset className="platform-module-options" disabled={busy}>
          <legend className="sr-only">Enabled project modules</legend>
          {MODULE_OPTIONS.map((module) => {
            const checked = data.projectModules.selected.includes(module.id);
            const available = MODULE_CAPABILITIES[module.id].some((capability) =>
              workspaceCapabilities.has(capability),
            );
            return (
              <label key={module.id}>
                <NativeInput
                  type="checkbox"
                  checked={checked}
                  disabled={!available}
                  onChange={(event) => {
                    if (event.target.checked) void updateModules(module.id, true);
                    else {
                      setLocalError(null);
                      setModuleToDisable(module.id);
                    }
                  }}
                />
                <span>
                  <strong>{module.label}</strong>
                  <small>{module.description}</small>
                </span>
              </label>
            );
          })}
        </fieldset>
      </PlatformSection>

      <PlatformSection
        title="Configuration ownership"
        description="Mutable setup stays here; scientific records remain versioned in their dedicated project sections."
      >
        <dl className="platform-policy-grid">
          <div>
            <dt>Annotation schema</dt>
            <dd>{project.tasks.length + data.taskDefinitions.length} task definitions · {validationMode}</dd>
          </div>
          <div>
            <dt>Corpus</dt>
            <dd>{data.datasets.length} source datasets · configured in Data</dd>
          </div>
          <div>
            <dt>Assignments</dt>
            <dd>{data.rounds.length} versioned rounds · configured in Team &amp; Rounds</dd>
          </div>
          <div>
            <dt>Modules</dt>
            <dd>{data.projectModules.selected.filter(released).length} selected for this project</dd>
          </div>
        </dl>
      </PlatformSection>

      {data.projectModules.effective.includes("train") ? <PlatformSection
        title="Training defaults"
        description="These defaults prefill project-linked workflows; a Trainer may still choose another available runtime or policy."
      >
        <div className="platform-settings-form">
          <label>
            <span>Default runtime</span>
            <NativeSelect
              value={environmentId}
              onChange={(event) => setEnvironmentId(event.target.value)}
            >
              <option value="">Choose in Training</option>
              {data.environments.map((environment) => (
                <option key={environment.id} value={environment.id}>
                  {environment.name} · {formatStatus(environment.status)}
                </option>
              ))}
            </NativeSelect>
          </label>
          <label>
            <span>Default storage policy</span>
            <NativeSelect
              value={storagePolicyId}
              onChange={(event) => setStoragePolicyId(event.target.value)}
            >
              <option value="">Choose in Training</option>
              {data.storagePolicies.map((policy) => (
                <option key={policy.id} value={policy.id}>
                  {policy.name} · {formatStatus(policy.backend)}
                </option>
              ))}
            </NativeSelect>
          </label>
          <label>
            <span>Default evaluation split</span>
            <NativeSelect
              value={evaluationSplit}
              onChange={(event) => setEvaluationSplit(event.target.value)}
            >
              <option value="validation">Validation</option>
              <option value="test">Protected test</option>
            </NativeSelect>
          </label>

          <div className="platform-settings-actions">
            <Button
              label={busy ? "Saving…" : "Save defaults"}
              variant="primary"
              isDisabled={busy}
              onClick={() => void saveDefaults()}
            />
          </div>
        </div>
      </PlatformSection> : null}
      <ConfirmDialog
        open={moduleToDisable !== null}
        title="Disable project module?"
        description={moduleToDisable ? `Disable ${MODULE_OPTIONS.find((module) => module.id === moduleToDisable)?.label.toLowerCase()} for this project? Dependent modules will also be disabled. Existing scientific records remain saved.` : ""}
        confirmLabel="Disable module"
        busy={busy}
        error={localError}
        onCancel={() => { setModuleToDisable(null); setLocalError(null); }}
        onConfirm={async () => { if (moduleToDisable) await updateModules(moduleToDisable, false); }}
      />
    </div>
  );
}
