import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@astryxdesign/core/Badge";
import { Button } from "@astryxdesign/core/Button";

import type { BaseModelAsset } from "@/types/api";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
} from "./components";
import type {
  PlatformProjectData,
  TaskVersion,
  TrainingRecipeDescriptor,
} from "./types";

export interface TrainingLaunchDraft {
  modelName: string;
  taskVersionId: number;
  trainingDatasetVersionId: number;
  recipeKey: string;
  recipeVersionId: number | null;
  environmentId: number;
  storagePolicyId: number;
  seed: number;
  config: Record<string, unknown>;
  evaluationPlan: Record<string, unknown>;
}

export function resolveInitialTrainingDataset(
  data: PlatformProjectData,
  datasetId: number | null | undefined,
): PlatformProjectData["trainingDatasets"][number] | undefined {
  if (!datasetId) return undefined;
  const sourceVersionIds = new Set(
    data.datasetVersions
      .filter((item) => item.dataset_id === datasetId)
      .map((item) => item.id),
  );
  const sourceDataset = data.trainingDatasets.find((item) =>
    sourceVersionIds.has(item.dataset_version_id),
  );
  if (sourceDataset || data.datasets.some((item) => item.id === datasetId)) {
    return sourceDataset;
  }
  return (
    data.trainingDatasets.find(
      (item) => item.dataset_version_id === datasetId,
    ) ?? data.trainingDatasets.find((item) => item.id === datasetId)
  );
}

const STEPS = [
  "Task & Data",
  "Model & Recipe",
  "Environment",
  "Evaluation & Storage",
  "Review & Launch",
];

function defaultConfig(recipe: TrainingRecipeDescriptor | undefined): Record<string, unknown> {
  if (!recipe) return {};
  if (recipe.key.startsWith("tfidf_")) {
    return {
      fields: { input_field: "text", target_field: "label" },
      max_features: 50_000,
      ngram_min: 1,
      ngram_max: 2,
      min_document_frequency: 2,
    };
  }
  if (recipe.key.startsWith("transformer_")) {
    return {
      input_field: recipe.key === "transformer_token_classification" ? "tokens" : "text",
      target_field: "label",
      max_sequence_length: 512,
      learning_rate: 0.00002,
      epochs: 3,
      batch_size: 8,
      gradient_accumulation_steps: 1,
      mixed_precision: "none",
    };
  }
  return {
    prompt_field: "prompt",
    completion_field: "completion",
    prompt_template_version: "v1",
    max_sequence_length: 2048,
    learning_rate: 0.00002,
    epochs: 1,
    batch_size: 1,
    gradient_accumulation_steps: 8,
    mixed_precision: "bf16",
  };
}

function parseTrainingConfiguration(configText: string): {
  config: Record<string, unknown> | null;
  error: string | null;
} {
  try {
    const value = JSON.parse(configText) as unknown;
    if (!value || Array.isArray(value) || typeof value !== "object") {
      return {
        config: null,
        error: "Configuration must be a JSON object.",
      };
    }
    return {
      config: value as Record<string, unknown>,
      error: null,
    };
  } catch {
    return {
      config: null,
      error: "Configuration contains invalid JSON.",
    };
  }
}

function modelNameMatches(left: string, right: string): boolean {
  return (
    left.localeCompare(right, undefined, {
      sensitivity: "accent",
    }) === 0
  );
}

function runRecipeLabel(
  data: PlatformProjectData,
  recipeVersionId: number,
): string {
  const version = data.recipeVersions.find(
    (item) => item.id === recipeVersionId,
  );
  if (!version) return `Recipe version ${recipeVersionId}`;
  const registered = data.projectRecipes.find(
    (item) => item.id === version.training_recipe_id,
  );
  const descriptor = data.recipes.find((item) => item.key === registered?.key);
  return `${descriptor?.label ?? registered?.name ?? "Training recipe"} · v${version.version_number}`;
}

function storageEncryptionLabel(
  encryption: Record<string, unknown>,
): string {
  if (encryption.enabled === false || encryption.at_rest === false) {
    return "Encryption: disabled";
  }
  const values = [
    encryption.mode,
    encryption.algorithm,
    encryption.key_management,
  ]
    .filter(
      (value): value is string =>
        typeof value === "string" && Boolean(value.trim()),
    )
    .map((value) => value.replace(/_/g, " "));
  if (values.length) return `Encryption: ${values.join(" · ")}`;
  return encryption.enabled === true || encryption.at_rest === true
    ? "Encryption: enabled"
    : "Encryption: policy managed";
}

function trainerFrameworkLabel(trainerKey: string): string {
  if (trainerKey === "sklearn_tfidf") return "scikit-learn";
  if (trainerKey.startsWith("huggingface_")) return "Transformers";
  return trainerKey.replace(/_/g, " ");
}

export function recipeRequiresBaseModel(
  recipe: TrainingRecipeDescriptor | undefined,
): boolean {
  if (!recipe) return false;
  const required = recipe.config_schema.required;
  return (
    recipe.model_family === "deep_learning" ||
    recipe.model_family === "llm_finetune" ||
    (Array.isArray(required) && required.includes("base_model_asset_id"))
  );
}

export function recipeCompatibleWithTask(
  recipe: TrainingRecipeDescriptor,
  taskVersion: TaskVersion | undefined,
): boolean {
  if (!taskVersion) return true;
  if (!recipe.supported_task_kinds.includes(taskVersion.task_kind)) {
    return false;
  }
  const compatibleTrainers = new Set(
    taskVersion.trainer_compatibility
      .map((identifier) => identifier.trim())
      .filter(Boolean),
  );
  return (
    compatibleTrainers.has(recipe.key) ||
    compatibleTrainers.has(recipe.trainer_key)
  );
}

function stringArray(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : null;
}

export function compatibleBaseModels(
  assets: BaseModelAsset[],
  recipe: TrainingRecipeDescriptor | undefined,
  taskKind: string | undefined,
): BaseModelAsset[] {
  if (!recipeRequiresBaseModel(recipe)) return [];
  return assets.filter((asset) => {
    if (
      asset.readiness !== "ready" ||
      asset.archived_at !== null ||
      asset.package.readiness !== "ready" ||
      asset.package.archived_at != null ||
      asset.package.purged_at != null ||
      asset.model_family !== recipe?.model_family
    ) {
      return false;
    }
    const supportedTasks = stringArray(asset.metadata.supported_task_kinds);
    if (taskKind && supportedTasks && !supportedTasks.includes(taskKind)) {
      return false;
    }
    const architecture = asset.metadata.architecture_family;
    if (
      typeof architecture === "string" &&
      architecture !== recipe.architecture_family
    ) {
      return false;
    }
    if (
      architecture === undefined &&
      ["causal_lm", "seq2seq_lm"].includes(recipe.architecture_family) &&
      asset.model_type !== recipe.architecture_family
    ) {
      return false;
    }
    return true;
  });
}

export default function TrainingScreen({
  data,
  busy,
  onLaunch,
  contextLabel,
  creatorLabel,
  title = "Train",
  showRecentRuns = true,
  onOpenRun,
  initialDatasetId,
  initialEnvironmentId,
  initialStoragePolicyId,
  initialEvaluationSplit,
  onOpenTrainingData,
}: {
  data: PlatformProjectData;
  busy: boolean;
  onLaunch: (draft: TrainingLaunchDraft) => Promise<void>;
  contextLabel?: string;
  creatorLabel?: string;
  title?: string;
  showRecentRuns?: boolean;
  onOpenRun?: (runId: number) => void;
  initialDatasetId?: number | null;
  initialEnvironmentId?: number | null;
  initialStoragePolicyId?: number | null;
  initialEvaluationSplit?: string | null;
  onOpenTrainingData?: () => void;
}): React.ReactElement {
  const [step, setStep] = useState(0);
  const [modelName, setModelName] = useState("");
  const [taskVersionId, setTaskVersionId] = useState(0);
  const [trainingDatasetVersionId, setTrainingDatasetVersionId] = useState(0);
  const [recipeKey, setRecipeKey] = useState("");
  const [baseModelAssetId, setBaseModelAssetId] = useState(0);
  const [environmentId, setEnvironmentId] = useState(
    initialEnvironmentId ?? 0,
  );
  const [storagePolicyId, setStoragePolicyId] = useState(
    initialStoragePolicyId ?? 0,
  );
  const [evaluationSplit, setEvaluationSplit] = useState(
    initialEvaluationSplit === "validation" ? "validation" : "test",
  );
  const [seed, setSeed] = useState(42);
  const [advanced, setAdvanced] = useState(false);
  const [configText, setConfigText] = useState("{}");
  const [configError, setConfigError] = useState<string | null>(null);
  const configInputRef = useRef<HTMLTextAreaElement | null>(null);
  const focusConfigAfterNavigation = useRef(false);

  const taskVersion = data.taskVersions.find((item) => item.id === taskVersionId);
  const compatibleRecipes = useMemo(
    () =>
      data.recipes.filter(
        (recipe) => recipeCompatibleWithTask(recipe, taskVersion),
      ),
    [data.recipes, taskVersion],
  );
  const recipe = data.recipes.find((item) => item.key === recipeKey);
  const baseModelRequired = recipeRequiresBaseModel(recipe);
  const availableBaseModels = useMemo(
    () => compatibleBaseModels(data.baseModels, recipe, taskVersion?.task_kind),
    [data.baseModels, recipe, taskVersion?.task_kind],
  );
  const baseModel = availableBaseModels.find(
    (item) => item.id === baseModelAssetId,
  );
  const existingModel = data.models.find((item) =>
    modelNameMatches(item.name, modelName.trim()),
  );
  const compatibleReadyEnvironments = useMemo(
    () =>
      recipe
        ? data.environments.filter(
            (item) =>
              item.status === "available" &&
              item.environment_class === recipe.environment.runtime_class,
          )
        : [],
    [data.environments, recipe],
  );
  const environment = data.environments.find((item) => item.id === environmentId);
  const environmentReady = Boolean(
    recipe &&
      environment?.status === "available" &&
      environment.environment_class === recipe.environment.runtime_class,
  );
  const storage = data.storagePolicies.find((item) => item.id === storagePolicyId);
  const trainingDataset = data.trainingDatasets.find(
    (item) => item.id === trainingDatasetVersionId,
  );
  const sourceDatasetVersion = trainingDataset
    ? data.datasetVersions.find(
        (item) => item.id === trainingDataset.dataset_version_id,
      )
    : undefined;
  const projectRecipe = data.projectRecipes.find((item) => item.key === recipeKey);
  const recipeVersion =
    data.recipeVersions
      .filter((item) => item.training_recipe_id === projectRecipe?.id)
      .sort((left, right) => right.version_number - left.version_number)[0] ?? null;
  const preferredInitialDataset = useMemo(
    () => resolveInitialTrainingDataset(data, initialDatasetId),
    [data, initialDatasetId],
  );
  const requestedDatasetNeedsPreparation =
    Boolean(initialDatasetId) && preferredInitialDataset === undefined;
  const requestedDatasetName = data.datasets.find(
    (item) => item.id === initialDatasetId,
  )?.name;

  useEffect(() => {
    if (!taskVersionId) {
      const initialTaskVersionId =
        preferredInitialDataset?.task_version_id ?? data.taskVersions[0]?.id;
      if (initialTaskVersionId) setTaskVersionId(initialTaskVersionId);
    }
    if (!trainingDatasetVersionId) {
      const initialTrainingDatasetId =
        preferredInitialDataset !== undefined
          ? preferredInitialDataset.id
          : initialDatasetId
            ? undefined
            : data.trainingDatasets[0]?.id;
      if (initialTrainingDatasetId) {
        setTrainingDatasetVersionId(initialTrainingDatasetId);
      }
    }
    if (
      !storagePolicyId ||
      !data.storagePolicies.some((item) => item.id === storagePolicyId)
    ) {
      const preferred =
        data.storagePolicies.find((item) => item.is_default) ??
        data.storagePolicies[0];
      setStoragePolicyId(preferred?.id ?? 0);
    }
  }, [
    data.storagePolicies,
    data.taskVersions,
    data.trainingDatasets,
    initialDatasetId,
    preferredInitialDataset,
    requestedDatasetNeedsPreparation,
    storagePolicyId,
    taskVersionId,
    trainingDatasetVersionId,
  ]);

  useEffect(() => {
    if (
      requestedDatasetNeedsPreparation &&
      trainingDatasetVersionId === 0
    ) {
      return;
    }
    const selected = data.trainingDatasets.find(
      (item) =>
        item.id === trainingDatasetVersionId &&
        item.task_version_id === taskVersionId,
    );
    if (!selected) {
      setTrainingDatasetVersionId(
        data.trainingDatasets.find((item) => item.task_version_id === taskVersionId)?.id ?? 0,
      );
    }
  }, [
    data.trainingDatasets,
    requestedDatasetNeedsPreparation,
    taskVersionId,
    trainingDatasetVersionId,
  ]);

  useEffect(() => {
    if (!compatibleRecipes.some((item) => item.key === recipeKey)) {
      const next = compatibleRecipes[0];
      setRecipeKey(next?.key ?? "");
      setBaseModelAssetId(0);
      setConfigText(JSON.stringify(defaultConfig(next), null, 2));
      setConfigError(null);
    }
  }, [compatibleRecipes, recipeKey]);

  useEffect(() => {
    const selectedIsReady =
      environment?.status === "available" &&
      (!recipe ||
        environment.environment_class === recipe.environment.runtime_class);
    if (selectedIsReady) return;
    const next = compatibleReadyEnvironments[0];
    setEnvironmentId(next?.id ?? 0);
  }, [compatibleReadyEnvironments, environment, recipe]);

  useEffect(() => {
    if (!baseModelRequired) {
      setBaseModelAssetId(0);
      return;
    }
    if (
      baseModelAssetId &&
      !availableBaseModels.some((item) => item.id === baseModelAssetId)
    ) {
      setBaseModelAssetId(0);
    }
  }, [availableBaseModels, baseModelAssetId, baseModelRequired]);

  useEffect(() => {
    if (step === 1 && focusConfigAfterNavigation.current) {
      focusConfigAfterNavigation.current = false;
      configInputRef.current?.focus();
    }
  }, [step, configError]);

  const configurationValidation = useMemo(
    () => parseTrainingConfiguration(configText),
    [configText],
  );

  function parsedConfig(focusOnError = false): Record<string, unknown> | null {
    const result = parseTrainingConfiguration(configText);
    setConfigError(result.error);
    if (result.error && focusOnError) {
      if (step === 1) {
        configInputRef.current?.focus();
      } else {
        focusConfigAfterNavigation.current = true;
        setStep(1);
      }
    }
    return result.config;
  }

  const modelRecipeReady = Boolean(
    modelName.trim() &&
      recipeKey &&
      (!baseModelRequired || Boolean(baseModel)),
  );
  const stepReady = [
    Boolean(taskVersionId && trainingDatasetVersionId),
    modelRecipeReady && configurationValidation.error === null,
    environmentReady,
    Boolean(storage),
    Boolean(recipe),
  ];
  const launchReady = stepReady.every(Boolean);
  const launchPrerequisitesReady = [
    stepReady[0],
    modelRecipeReady,
    stepReady[2],
    stepReady[3],
    stepReady[4],
  ].every(Boolean);

  function navigateToStep(nextStep: number): void {
    if (nextStep > 1 && !parsedConfig(true)) return;
    setStep(nextStep);
  }

  async function launch(): Promise<void> {
    const config = parsedConfig(true);
    if (!config || !launchPrerequisitesReady) return;
    const pinnedConfig = { ...config };
    if (baseModelRequired) {
      if (!baseModel) return;
      pinnedConfig.base_model_asset_id = baseModel.id;
    } else {
      delete pinnedConfig.base_model_asset_id;
    }
    await onLaunch({
      modelName: modelName.trim(),
      taskVersionId,
      trainingDatasetVersionId,
      recipeKey,
      recipeVersionId: recipeVersion?.id ?? null,
      environmentId,
      storagePolicyId,
      seed,
      config: pinnedConfig,
      evaluationPlan: {
        splits: [evaluationSplit],
      },
    });
  }

  return (
    <div className="platform-page">
      <PlatformPageHeader
        title={title}
        description="Create a named, reproducible model version from task-neutral recipes."
      />

      <nav className="platform-wizard-steps" aria-label="Training setup">
        {STEPS.map((label, index) => (
          <button
            key={label}
            type="button"
            aria-current={step === index ? "step" : undefined}
            onClick={() => navigateToStep(index)}
          >
            <span aria-hidden="true">{index + 1}</span>
            {step === index
              ? `Step ${index + 1} of ${STEPS.length}: ${label}`
              : label}
          </button>
        ))}
      </nav>

      <section className="platform-wizard-panel">
        {step === 0 ? (
          <>
            <div className="platform-section-header">
              <div><h2>Task & Data</h2><p>Versions are pinned when the run launches.</p></div>
            </div>
            {requestedDatasetNeedsPreparation ? (
              <div className="platform-form-warning" role="status">
                <strong>
                  {requestedDatasetName ?? "The requested source dataset"} is
                  not yet a composed training dataset.
                </strong>
                <span>
                  Define its label source, preprocessing, and protected split
                  before selecting it for this run.
                </span>
                {onOpenTrainingData ? (
                  <Button
                    label="Prepare requested dataset"
                    size="sm"
                    onClick={onOpenTrainingData}
                  />
                ) : null}
              </div>
            ) : null}
            {data.taskVersions.length && data.trainingDatasets.length ? (
              <div className="platform-form-grid">
                <label>
                  <span>Task version</span>
                  <select value={taskVersionId} onChange={(event) => setTaskVersionId(Number(event.target.value))}>
                    {data.taskVersions.map((task) => (
                      <option key={task.id} value={task.id}>
                        {task.task_kind.replace(/_/g, " ")} · v{task.version_number}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Training dataset</span>
                  <select
                    value={trainingDatasetVersionId}
                    onChange={(event) => setTrainingDatasetVersionId(Number(event.target.value))}
                  >
                    {trainingDatasetVersionId === 0 ? (
                      <option value={0}>Select a composed dataset</option>
                    ) : null}
                    {data.trainingDatasets
                      .filter((dataset) => dataset.task_version_id === taskVersionId)
                      .map((dataset) => (
                        <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                      ))}
                  </select>
                </label>
              </div>
            ) : (
              <PlatformEmpty
                title="Training data is not ready"
                detail="Create a task version, label composition, and protected split map first."
                actionLabel={
                  onOpenTrainingData ? "Prepare training data" : undefined
                }
                onAction={onOpenTrainingData}
              />
            )}
          </>
        ) : null}

        {step === 1 ? (
          <>
            <div className="platform-section-header">
              <div><h2>Model & Recipe</h2><p>The model name remains stable while each run creates a new version.</p></div>
            </div>
            <div className="platform-form-grid">
              <label>
                <span>Model name</span>
                <input
                  value={modelName}
                  maxLength={255}
                  placeholder="e.g. Sentiment baseline"
                  aria-label="Model name"
                  aria-describedby={
                    existingModel ? "training-model-reuse" : undefined
                  }
                  onChange={(event) => setModelName(event.target.value)}
                />
                {existingModel ? (
                  <small
                    id="training-model-reuse"
                    className="platform-model-reuse-hint"
                    aria-live="polite"
                  >
                    Existing model identity: {existingModel.name}. This run
                    creates its next version.
                  </small>
                ) : null}
              </label>
              <label>
                <span>Training recipe</span>
                <select
                  value={recipeKey}
                  onChange={(event) => {
                    const next = data.recipes.find((item) => item.key === event.target.value);
                    setRecipeKey(event.target.value);
                    setBaseModelAssetId(0);
                    setConfigText(JSON.stringify(defaultConfig(next), null, 2));
                  }}
                >
                  {compatibleRecipes.map((item) => (
                    <option key={item.key} value={item.key}>{item.label}</option>
                  ))}
                </select>
              </label>
              {baseModelRequired ? (
                <label>
                  <span>Immutable base model</span>
                  <select
                    required
                    value={baseModelAssetId}
                    disabled={!availableBaseModels.length}
                    onChange={(event) =>
                      setBaseModelAssetId(Number(event.target.value))
                    }
                  >
                    <option value={0}>Choose a ready base model</option>
                    {availableBaseModels.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.display_name} · Asset {item.id} · Source{" "}
                        {item.source_model_id} · Revision {item.exact_revision}
                      </option>
                    ))}
                  </select>
                  {!availableBaseModels.length ? (
                    <small>
                      No ready base model matches this recipe and task.
                    </small>
                  ) : null}
                </label>
              ) : null}
            </div>
            {recipe ? (
              <div className="platform-recipe-summary">
                <div>
                  <strong>{recipe.label}</strong>
                  <p>{recipe.description}</p>
                </div>
                <Badge label={recipe.model_family.replace(/_/g, " ")} variant="info" />
                <Badge label={recipe.parameterization} variant="neutral" />
              </div>
            ) : (
              <p className="platform-inline-empty">No compatible recipe is registered.</p>
            )}
            <label className="platform-switch-row">
              <input
                type="checkbox"
                checked={advanced}
                onChange={(event) => {
                  const checked = event.target.checked;
                  setAdvanced(checked);
                  if (checked) {
                    setConfigError(
                      parseTrainingConfiguration(configText).error,
                    );
                  } else {
                    setConfigText(JSON.stringify(defaultConfig(recipe), null, 2));
                    setConfigError(null);
                  }
                }}
              />
              <span>Advanced JSON configuration</span>
            </label>
            {advanced ? (
              <label className="platform-json-field">
                <span>Recipe configuration</span>
                <textarea
                  ref={configInputRef}
                  value={configText}
                  rows={14}
                  spellCheck={false}
                  onChange={(event) => {
                    const nextText = event.target.value;
                    setConfigText(nextText);
                    setConfigError(
                      parseTrainingConfiguration(nextText).error,
                    );
                  }}
                  onBlur={() => parsedConfig()}
                  aria-invalid={configError ? "true" : undefined}
                  aria-describedby={configError ? "training-config-error" : undefined}
                />
                {configError ? <small id="training-config-error" role="alert">{configError}</small> : null}
              </label>
            ) : null}
          </>
        ) : null}

        {step === 2 ? (
          <>
            <div className="platform-section-header">
              <div><h2>Execution environment</h2><p>Only verified, image-bound runtimes can launch.</p></div>
            </div>
            {data.environments.length ? (
              <div className="platform-option-list" role="radiogroup" aria-label="Execution environment">
                {data.environments.map((item) => {
                  const compatible = !recipe || item.environment_class === recipe.environment.runtime_class;
                  const selectable = compatible && item.status === "available";
                  return (
                    <label key={item.id} data-disabled={!selectable}>
                      <input
                        type="radio"
                        name="environment"
                        value={item.id}
                        checked={environmentId === item.id}
                        disabled={!selectable}
                        onChange={() => setEnvironmentId(item.id)}
                      />
                      <span>
                        <strong>{item.name}</strong>
                        <small>{item.environment_class}</small>
                      </span>
                      <PlatformStatus value={item.status} />
                    </label>
                  );
                })}
              </div>
            ) : (
              <PlatformEmpty
                title="No runtime enabled"
                detail={recipe?.environment.setup_hint ?? "Ask an administrator to enable a worker runtime."}
              />
            )}
            {data.environments.length &&
            recipe &&
            !compatibleReadyEnvironments.length ? (
              <p className="platform-form-warning" role="status">
                {recipe.environment.setup_hint}
              </p>
            ) : null}
            {recipe ? (
              <div className="platform-runtime-requirements">
                <strong>{recipe.environment.runtime_class}</strong>
                <span>{recipe.environment.devices.join(", ")}</span>
                <span>{recipe.environment.minimum_memory_gb} GB minimum</span>
                <span>{recipe.environment.packages.join(", ")}</span>
              </div>
            ) : null}
          </>
        ) : null}

        {step === 3 ? (
          <>
            <div className="platform-section-header">
              <div><h2>Evaluation & Storage</h2><p>Protected test data remains outside selection and tuning.</p></div>
            </div>
            <div className="platform-form-grid">
              <label>
                <span>Storage policy</span>
                <select value={storagePolicyId} onChange={(event) => setStoragePolicyId(Number(event.target.value))}>
                  <option value={0}>Choose storage</option>
                  {data.storagePolicies.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.name} · {item.backend}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span>Evaluation split</span>
                <select
                  value={evaluationSplit}
                  onChange={(event) => setEvaluationSplit(event.target.value)}
                >
                  <option value="validation">Validation</option>
                  <option value="test">Protected test</option>
                </select>
              </label>
              <label>
                <span>Random seed</span>
                <input
                  type="number"
                  inputMode="numeric"
                  value={seed}
                  onChange={(event) => setSeed(Number(event.target.value))}
                />
              </label>
            </div>
            {storage ? (
              <div className="platform-storage-summary">
                <PlatformStatus value={storage.backend} />
                <span>Artifact prefix: {storage.artifact_prefix}</span>
                <span>Retention: {storage.retention_class}</span>
                <span>{storageEncryptionLabel(storage.encryption)}</span>
                <span>
                  Evaluation split:{" "}
                  {evaluationSplit === "test"
                    ? "protected test"
                    : "validation"}
                </span>
              </div>
            ) : (
              <p className="platform-inline-empty">An administrator-managed storage policy is required.</p>
            )}
          </>
        ) : null}

        {step === 4 ? (
          <>
            <div className="platform-section-header">
              <div><h2>Review & Launch</h2><p>The run manifest will pin every selection below.</p></div>
            </div>
            <dl className="platform-review-grid">
              {contextLabel ? (
                <div><dt>Project context</dt><dd>{contextLabel}</dd></div>
              ) : null}
              {creatorLabel ? (
                <div><dt>Created by</dt><dd>{creatorLabel}</dd></div>
              ) : null}
              <div><dt>Model name</dt><dd>{modelName || "Not named"}</dd></div>
              <div><dt>Model family</dt><dd>{recipe?.model_family.replace(/_/g, " ") ?? "Not selected"}</dd></div>
              <div><dt>Architecture</dt><dd>{recipe?.architecture_family.replace(/_/g, " ") ?? "Not selected"}</dd></div>
              <div><dt>Framework</dt><dd>{recipe ? trainerFrameworkLabel(recipe.trainer_key) : "Not selected"}</dd></div>
              <div><dt>Training method</dt><dd>{recipe?.parameterization.replace(/_/g, " ") ?? "Not selected"}</dd></div>
              <div>
                <dt>Task</dt>
                <dd>
                  {taskVersion
                    ? `${taskVersion.task_kind.replace(/_/g, " ")} · v${taskVersion.version_number}`
                    : "Not selected"}
                </dd>
              </div>
              <div>
                <dt>Training dataset</dt>
                <dd>
                  {trainingDataset
                    ? `${trainingDataset.name} · source v${
                        sourceDatasetVersion?.version_number ??
                        trainingDataset.dataset_version_id
                      }`
                    : "Not selected"}
                </dd>
              </div>
              <div><dt>Recipe</dt><dd>{recipe?.label ?? "Not selected"}</dd></div>
              <div>
                <dt>Base model</dt>
                <dd>
                  {baseModelRequired
                    ? baseModel
                      ? (
                          <>
                            {baseModel.display_name}
                            <span className="platform-review-detail">
                              Asset ID: {baseModel.id} · Source:{" "}
                              {baseModel.source_model_id} · Revision:{" "}
                              {baseModel.exact_revision}
                            </span>
                          </>
                        )
                      : "Not selected"
                    : (
                        <>
                          From scratch
                          <span className="platform-review-detail">
                            Asset ID: Not required · Source: From scratch ·
                            Revision: Not applicable
                          </span>
                        </>
                      )}
                </dd>
              </div>
              <div>
                <dt>Environment</dt>
                <dd>
                  {environment ? (
                    <>
                      {environment.name}
                      <span className="platform-review-detail">
                        {environment.environment_class}
                        {environment.image_digest
                          ? ` · ${environment.image_digest.slice(0, 19)}…`
                          : ""}
                      </span>
                    </>
                  ) : (
                    "Not selected"
                  )}
                </dd>
              </div>
              <div>
                <dt>Storage</dt>
                <dd>
                  {storage ? (
                    <>
                      {storage.name}
                      <span className="platform-review-detail">
                        {storage.backend} · {storage.artifact_prefix} ·{" "}
                        {storage.retention_class} ·{" "}
                        {storageEncryptionLabel(storage.encryption)}
                      </span>
                    </>
                  ) : (
                    "Not selected"
                  )}
                </dd>
              </div>
              <div>
                <dt>Evaluation</dt>
                <dd>
                  {evaluationSplit === "test"
                    ? "Protected test split"
                    : "Validation split"}
                  <span className="platform-review-detail">
                    Metrics: {taskVersion?.metrics.join(", ") || "recipe defaults"}
                  </span>
                </dd>
              </div>
              <div><dt>Seed</dt><dd>{seed}</dd></div>
              <div><dt>Recipe version</dt><dd>{recipeVersion ? `v${recipeVersion.version_number}` : `${recipe?.version ?? "Unknown"} · registered at launch`}</dd></div>
            </dl>
            {!launchReady ? (
              <p className="platform-form-warning" role="status">
                Complete each step and choose an available, verified runtime before launch.
              </p>
            ) : null}
          </>
        ) : null}

        <div className="platform-wizard-actions">
          <Button
            label="Previous"
            isDisabled={step === 0 || busy}
            onClick={() => setStep((current) => Math.max(0, current - 1))}
          />
          {step < STEPS.length - 1 ? (
            <Button
              label="Continue"
              variant="primary"
              isDisabled={!stepReady[step] || busy}
              onClick={() =>
                navigateToStep(Math.min(STEPS.length - 1, step + 1))
              }
            />
          ) : (
            <Button
              label={busy ? "Launching…" : "Launch training"}
              variant="primary"
              isDisabled={!launchPrerequisitesReady || busy}
              onClick={() => void launch()}
            />
          )}
        </div>
      </section>

      {showRecentRuns ? (
        <PlatformSection title="Recent runs" description="Run state is separate from model identity.">
        {data.trainingRuns.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Recent training runs table"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead><tr><th scope="col">Run</th><th scope="col">Model</th><th scope="col">Training data</th><th scope="col">Recipe</th><th scope="col">Environment</th><th scope="col">Status</th></tr></thead>
              <tbody>
                {data.trainingRuns.map((run) => (
                  <tr key={run.id}>
                    <td data-label="Run" data-priority="identity">
                      {onOpenRun ? (
                        <button
                          type="button"
                          className="platform-text-action"
                          onClick={() => onOpenRun(run.id)}
                        >
                          Run {run.id}
                        </button>
                      ) : (
                        <strong>Run {run.id}</strong>
                      )}
                      <span>Seed {run.seed}</span>
                    </td>
                    <td data-label="Model">{data.models.find((item) => item.id === run.registered_model_id)?.name ?? `Model ${run.registered_model_id}`}</td>
                    <td data-label="Training data">{data.trainingDatasets.find((item) => item.id === run.training_dataset_version_id)?.name ?? `Dataset version ${run.training_dataset_version_id}`}</td>
                    <td data-label="Recipe">{runRecipeLabel(data, run.recipe_version_id)}</td>
                    <td data-label="Environment">{data.environments.find((item) => item.id === run.environment_id)?.name ?? `Environment ${run.environment_id}`}</td>
                    <td data-label="Status" data-priority="status">
                      <PlatformStatus value={run.status} />
                      {run.failure_reason ? (
                        <span className="platform-run-failure">
                          {run.failure_reason}
                        </span>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="platform-inline-empty">No training runs yet.</p>
        )}
        </PlatformSection>
      ) : null}
    </div>
  );
}
