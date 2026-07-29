import { useEffect, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import type {
  AnnotationValidationMode,
  Project,
  ProjectUpdate,
} from "@/types/api";

import { PlatformPageHeader, PlatformSection } from "./components";
import type { PlatformProjectData, ProjectModule } from "./types";

type ReleasedProjectModule = Exclude<ProjectModule, "learning" | "guidelines">;

const MODULE_OPTIONS: Array<{
  id: ReleasedProjectModule;
  label: string;
  description: string;
}> = [
  { id: "data", label: "Data", description: "Versioned source data and label layers." },
  { id: "annotate", label: "Annotation", description: "Tasks, rounds, and human decisions." },
  { id: "train", label: "Training", description: "Independent Training workspace access." },
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
  data: ["annotation", "training", "lineage"],
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
  const [exportFormat, setExportFormat] = useState(
    String(project.settings.default_export_format ?? "jsonl"),
  );
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
    setExportFormat(
      String(project.settings.default_export_format ?? "jsonl"),
    );
  }, [project]);

  async function saveIdentity(): Promise<void> {
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
          default_export_format: exportFormat,
        },
      });
      setMessage("Training and export defaults saved.");
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
            <input
              value={name}
              maxLength={255}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <label>
            <span>Description</span>
            <textarea
              value={description}
              rows={3}
              onChange={(event) => setDescription(event.target.value)}
            />
          </label>
          <label>
            <span>Annotation validation</span>
            <select
              value={validationMode}
              onChange={(event) =>
                setValidationMode(event.target.value as AnnotationValidationMode)
              }
            >
              <option value="relaxed">Relaxed</option>
              <option value="strict">Strict</option>
            </select>
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
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={!available}
                  onChange={(event) => {
                    const selected = new Set(
                      data.projectModules.selected.filter(released),
                    );
                    if (event.target.checked) {
                      selected.add(module.id);
                    } else {
                      selected.delete(module.id);
                      for (const dependent of MODULE_DEPENDENTS[module.id] ?? []) {
                        selected.delete(dependent);
                      }
                    }
                    setMessage(null);
                    setLocalError(null);
                    void onUpdateModules([...selected])
                      .then(() => setMessage("Project modules saved."))
                      .catch((caught: unknown) =>
                        setLocalError(
                          caught instanceof Error
                            ? caught.message
                            : "Project modules could not be saved.",
                        ),
                      );
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
            <dd>{project.tasks.length} task definitions · {validationMode}</dd>
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
            <dd>{data.projectModules.selected.length} selected for this project</dd>
          </div>
        </dl>
      </PlatformSection>

      <PlatformSection
        title="Training and export defaults"
        description="These defaults prefill project-linked workflows; a Trainer may still choose another available runtime or policy."
      >
        <div className="platform-settings-form">
          <label>
            <span>Default runtime</span>
            <select
              value={environmentId}
              onChange={(event) => setEnvironmentId(event.target.value)}
            >
              <option value="">Choose in Training</option>
              {data.environments.map((environment) => (
                <option key={environment.id} value={environment.id}>
                  {environment.name} · {environment.status}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Default storage policy</span>
            <select
              value={storagePolicyId}
              onChange={(event) => setStoragePolicyId(event.target.value)}
            >
              <option value="">Choose in Training</option>
              {data.storagePolicies.map((policy) => (
                <option key={policy.id} value={policy.id}>
                  {policy.name} · {policy.backend}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>Default evaluation split</span>
            <select
              value={evaluationSplit}
              onChange={(event) => setEvaluationSplit(event.target.value)}
            >
              <option value="validation">Validation</option>
              <option value="test">Protected test</option>
            </select>
          </label>
          <label>
            <span>Default export format</span>
            <select
              value={exportFormat}
              onChange={(event) => setExportFormat(event.target.value)}
            >
              <option value="jsonl">JSONL</option>
              <option value="csv">CSV</option>
              <option value="parquet">Parquet</option>
            </select>
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
      </PlatformSection>
    </div>
  );
}
