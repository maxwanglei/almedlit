// @vitest-environment jsdom

import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { BaseModelAsset } from "@/types/api";

import TrainingScreen, {
  compatibleBaseModels,
  recipeCompatibleWithTask,
  recipeRequiresBaseModel,
  resolveInitialTrainingDataset,
} from "./TrainingScreen";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
  type TrainingRecipeDescriptor,
} from "./types";

const classicalRecipe: TrainingRecipeDescriptor = {
  schema_version: "training-recipe-descriptor-v1",
  key: "tfidf_logistic_regression",
  version: "1",
  label: "TF-IDF logistic regression",
  description: "Linear baseline",
  model_family: "conventional_ml",
  architecture_family: "logistic_regression",
  parameterization: "full",
  supported_task_kinds: ["classification"],
  trainer_key: "sklearn_tfidf",
  implementation_status: "implemented",
  environment: {
    runtime_class: "classical-cpu",
    packages: ["scikit-learn"],
    devices: ["cpu"],
    minimum_memory_gb: 2,
    requires_verified_environment: true,
    setup_hint: "Use verified CPU.",
  },
  config_schema: { type: "object" },
  artifact_formats: ["skops"],
};

const transformerRecipe: TrainingRecipeDescriptor = {
  ...classicalRecipe,
  key: "transformer_sequence_classification",
  label: "Transformer sequence classification",
  model_family: "deep_learning",
  architecture_family: "transformer_encoder",
  trainer_key: "huggingface_sequence",
  environment: {
    ...classicalRecipe.environment,
    runtime_class: "transformer-cpu",
    packages: ["torch", "transformers", "safetensors"],
    minimum_memory_gb: 8,
    setup_hint: "Ask an administrator to enable the transformer CPU runtime.",
  },
  config_schema: {
    type: "object",
    required: ["base_model_asset_id"],
  },
  artifact_formats: ["safetensors", "tokenizer"],
};

function baseModel(
  overrides: Partial<BaseModelAsset> = {},
): BaseModelAsset {
  return {
    id: 161,
    project_id: 1,
    package_id: 261,
    provider: "hugging_face",
    source_model_id: "medical/abstract-encoder",
    exact_revision: "encoder-commit-42",
    display_name: "Medical abstract encoder",
    model_family: "deep_learning",
    model_type: "bert",
    license_name: "Apache-2.0",
    license_url: null,
    license_terms_sha256: null,
    access_mode: "downloadable",
    readiness: "ready",
    archived_at: null,
    metadata: {
      architecture_family: "transformer_encoder",
      supported_task_kinds: ["classification"],
    },
    package: {
      id: 261,
      project_id: 1,
      kind: "base_model",
      format: "safetensors",
      schema_version: "1",
      model_family: "deep_learning",
      model_type: "bert",
      readiness: "ready",
      deployable: true,
      manifest_digest: "a".repeat(64),
      logical_size_bytes: 1024,
      file_count: 1,
      archived_at: null,
      purged_at: null,
      created_at: "2026-07-27T14:00:00Z",
      files: [],
      references: [],
    },
    created_at: "2026-07-27T14:00:00Z",
    ...overrides,
  };
}

function trainingData(
  recipes: TrainingRecipeDescriptor[],
  baseModels: BaseModelAsset[],
): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    taskDefinitions: [
      {
        id: 11,
        project_id: 1,
        key: "abstract_relevance",
        name: "Abstract relevance",
        description: null,
      },
    ],
    taskVersions: [
      {
        id: 12,
        project_id: 1,
        task_definition_id: 11,
        version_number: 3,
        task_kind: "classification",
        input_schema: {},
        output_schema: {},
        label_rules: {},
        annotation_ui: {},
        metrics: ["f1"],
        trainer_compatibility: recipes.map((item) => item.key),
        content_hash: "b".repeat(64),
      },
    ],
    trainingDatasets: [
      {
        id: 51,
        project_id: 1,
        name: "Relevance training set",
        dataset_version_id: 22,
        task_version_id: 12,
        label_set_version_ids: [31],
        split_map_id: 41,
        composition: [],
        preprocessing: {},
        content_hash: "c".repeat(64),
      },
    ],
    recipes,
    baseModels,
    environments: [
      {
        id: 131,
        project_id: 1,
        name: "Verified transformer CPU",
        environment_class: "transformer-cpu",
        image_digest:
          "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        package_manifest: {},
        hardware_constraints: {},
        status: "available",
        verification_report: { passed: true },
        verified_at: "2026-07-27T14:00:00Z",
      },
    ],
    storagePolicies: [
      {
        id: 141,
        project_id: 1,
        name: "Project artifacts",
        backend: "minio",
        artifact_prefix: "projects/1",
        retention_class: "standard",
        encryption: {
          enabled: true,
          algorithm: "AES-256-GCM",
          key_management: "platform_managed",
        },
        cache_policy: {},
        is_default: true,
      },
    ],
  };
}

afterEach(() => cleanup());

describe("TrainingScreen base-model selection", () => {
  it("filters the immutable catalog by readiness, family, task, and architecture", () => {
    const wrongFamily = baseModel({
      id: 162,
      model_family: "llm_finetune",
      model_type: "causal_lm",
    });
    const wrongTask = baseModel({
      id: 163,
      metadata: {
        architecture_family: "transformer_encoder",
        supported_task_kinds: ["token_labeling"],
      },
    });
    const quarantined = baseModel({
      id: 164,
      readiness: "quarantined",
    });

    expect(recipeRequiresBaseModel(classicalRecipe)).toBe(false);
    expect(recipeRequiresBaseModel(transformerRecipe)).toBe(true);
    expect(
      compatibleBaseModels(
        [baseModel(), wrongFamily, wrongTask, quarantined],
        transformerRecipe,
        "classification",
      ).map((item) => item.id),
    ).toEqual([161]);
  });

  it("blocks transformer progression until a ready base is selected and pins it at launch", async () => {
    const user = userEvent.setup();
    const onLaunch = vi.fn().mockResolvedValue(undefined);
    const data = trainingData([transformerRecipe], [baseModel()]);
    data.storagePolicies[0].encryption.secret_access_key = "hidden-secret";
    render(
      <TrainingScreen
        data={data}
        busy={false}
        onLaunch={onLaunch}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Abstract relevance encoder",
    );
    const continueButton = screen.getByRole("button", { name: "Continue" });
    expect((continueButton as HTMLButtonElement).disabled).toBe(true);

    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    expect((continueButton as HTMLButtonElement).disabled).toBe(false);
    await user.click(continueButton);
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(screen.getByText("Artifact prefix: projects/1")).toBeTruthy();
    expect(
      screen.getByText("Encryption: AES-256-GCM · platform managed"),
    ).toBeTruthy();
    expect(screen.queryByText("hidden-secret")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(
      screen.getByText(
        /minio · projects\/1 · standard · Encryption: AES-256-GCM · platform managed/,
      ),
    ).toBeTruthy();
    expect(screen.getByText("deep learning")).toBeTruthy();
    expect(screen.getByText("transformer encoder")).toBeTruthy();
    expect(screen.getByText("Transformers")).toBeTruthy();
    expect(screen.getByText("full")).toBeTruthy();
    expect(
      screen.getByText("Relevance training set · source v22"),
    ).toBeTruthy();
    expect(
      screen.getByText(
        /Asset ID: 161 · Source: medical\/abstract-encoder · Revision: encoder-commit-42/,
      ),
    ).toBeTruthy();
    expect(screen.getByText("Protected test split")).toBeTruthy();
    expect(screen.getByText("Metrics: f1")).toBeTruthy();
    expect(screen.queryByText("hidden-secret")).toBeNull();
    await user.click(screen.getByRole("button", { name: "Launch training" }));

    await waitFor(() =>
      expect(onLaunch).toHaveBeenCalledWith(
        expect.objectContaining({
          config: expect.objectContaining({ base_model_asset_id: 161 }),
        }),
      ),
    );
  });

  it("keeps classical recipes available without a base-model catalog", async () => {
    const user = userEvent.setup();
    render(
      <TrainingScreen
        data={trainingData([classicalRecipe], [])}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Linear relevance baseline",
    );

    expect(
      screen.queryByRole("combobox", { name: "Immutable base model" }),
    ).toBeNull();
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("blocks forward navigation until advanced configuration is a JSON object", async () => {
    const user = userEvent.setup();
    render(
      <TrainingScreen
        data={trainingData([transformerRecipe], [baseModel()])}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Validated encoder",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    await user.click(
      screen.getByRole("checkbox", {
        name: "Advanced JSON configuration",
      }),
    );

    const configuration = screen.getByRole("textbox", {
      name: "Recipe configuration",
    });
    fireEvent.change(configuration, { target: { value: "{" } });

    expect(
      screen.getByRole("alert").textContent,
    ).toBe("Configuration contains invalid JSON.");
    expect(configuration.getAttribute("aria-invalid")).toBe("true");
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await user.click(
      screen.getByRole("button", { name: "Review & Launch" }),
    );
    expect(
      screen.getByRole("heading", { name: "Model & Recipe" }),
    ).toBeTruthy();
    expect(document.activeElement).toBe(configuration);

    fireEvent.change(configuration, { target: { value: "[]" } });
    expect(
      screen.getByRole("alert").textContent,
    ).toBe("Configuration must be a JSON object.");

    fireEvent.change(configuration, {
      target: { value: '{"epochs": 2}' },
    });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(configuration.getAttribute("aria-invalid")).toBeNull();
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);

    await user.click(
      screen.getByRole("button", { name: "Review & Launch" }),
    );
    expect(
      screen.getByRole("heading", { name: "Review & Launch" }),
    ).toBeTruthy();
  });

  it("offers only recipes allowed by the task trainer contract", async () => {
    const data = trainingData(
      [classicalRecipe, transformerRecipe],
      [baseModel()],
    );
    data.taskVersions[0].trainer_compatibility = [
      transformerRecipe.trainer_key,
    ];

    expect(
      recipeCompatibleWithTask(classicalRecipe, data.taskVersions[0]),
    ).toBe(false);
    expect(
      recipeCompatibleWithTask(transformerRecipe, data.taskVersions[0]),
    ).toBe(true);

    render(
      <TrainingScreen
        data={data}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Continue" }),
    );

    const recipeSelect = screen.getByRole("combobox", {
      name: "Training recipe",
    }) as HTMLSelectElement;
    expect(Array.from(recipeSelect.options, (option) => option.value)).toEqual([
      transformerRecipe.key,
    ]);
  });

  it("shows model reuse and resolves recent run recipe, data, and failure details", async () => {
    const user = userEvent.setup();
    const data = trainingData([classicalRecipe], []);
    data.models = [
      {
        id: 81,
        project_id: 1,
        name: "Relevance baseline",
        description: null,
        lifecycle_status: "active",
      },
    ];
    data.projectRecipes = [
      {
        id: 121,
        project_id: 1,
        key: classicalRecipe.key,
        name: classicalRecipe.label,
        description: classicalRecipe.description,
      },
    ];
    data.recipeVersions = [
      {
        id: 122,
        project_id: 1,
        training_recipe_id: 121,
        version_number: 3,
        trainer_plugin_key: classicalRecipe.trainer_key,
        trainer_plugin_version: classicalRecipe.version,
        compatible_task_kinds: ["classification"],
        environment_class: "classical-cpu",
        config_schema: classicalRecipe.config_schema,
        default_config: {},
        evaluation_defaults: { splits: ["test"] },
        content_hash: "d".repeat(64),
      },
    ];
    data.trainingRuns = [
      {
        id: 151,
        project_id: 1,
        registered_model_id: 81,
        task_version_id: 12,
        training_dataset_version_id: 51,
        recipe_version_id: 122,
        environment_id: 131,
        storage_policy_id: 141,
        idempotency_key: "failed-run",
        status: "failed",
        seed: 42,
        config: {},
        evaluation_plan: { splits: ["test"] },
        output_model_version_id: null,
        failure_code: "runtime_verification_failed",
        failure_reason: "Worker image verification failed.",
      },
    ];

    render(
      <TrainingScreen
        data={data}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );

    expect(
      screen.getByRole("cell", { name: "Relevance training set" }),
    ).toBeTruthy();
    expect(
      screen.getByRole("cell", {
        name: "TF-IDF logistic regression · v3",
      }),
    ).toBeTruthy();
    expect(screen.getByText("Worker image verification failed.")).toBeTruthy();

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    const modelName = screen.getByRole("textbox", { name: "Model name" });
    await user.type(modelName, "relevance baseline");
    expect(
      screen.getByText(
        "Existing model identity: Relevance baseline. This run creates its next version.",
      ),
    ).toBeTruthy();
    expect(modelName.getAttribute("aria-describedby")).toBe(
      "training-model-reuse",
    );
  });

  it("replaces a stale environment when the selected recipe needs another runtime", async () => {
    const user = userEvent.setup();
    const data = trainingData(
      [classicalRecipe, transformerRecipe],
      [baseModel()],
    );
    data.environments = [
      {
        ...data.environments[0],
        id: 130,
        name: "Verified classical CPU",
        environment_class: "classical-cpu",
      },
      data.environments[0],
    ];

    render(
      <TrainingScreen
        data={data}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Runtime-pinned encoder",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Training recipe" }),
      transformerRecipe.key,
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));

    const classicalEnvironment = screen.getByRole("radio", {
      name: /Verified classical CPU/,
    }) as HTMLInputElement;
    const transformerEnvironment = screen.getByRole("radio", {
      name: /Verified transformer CPU/,
    }) as HTMLInputElement;
    expect(classicalEnvironment.disabled).toBe(true);
    expect(classicalEnvironment.checked).toBe(false);
    expect(transformerEnvironment.checked).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });

  it("shows the setup hint when no listed environment can run the recipe", async () => {
    const user = userEvent.setup();
    const data = trainingData([transformerRecipe], [baseModel()]);
    data.environments = [
      {
        ...data.environments[0],
        id: 130,
        name: "Verified classical CPU",
        environment_class: "classical-cpu",
      },
    ];

    render(
      <TrainingScreen
        data={data}
        busy={false}
        onLaunch={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Runtime-gated encoder",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(
      screen.getByText(
        "Ask an administrator to enable the transformer CPU runtime.",
      ),
    ).toBeTruthy();
    expect(
      (
        screen.getByRole("radio", {
          name: /Verified classical CPU/,
        }) as HTMLInputElement
      ).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(screen.queryByRole("button", { name: /install/i })).toBeNull();
  });

  it("applies project runtime, storage, and evaluation defaults to the run", async () => {
    const user = userEvent.setup();
    const onLaunch = vi.fn().mockResolvedValue(undefined);
    const data = trainingData([transformerRecipe], [baseModel()]);
    data.environments.push({
      ...data.environments[0],
      id: 132,
      name: "Preferred transformer CPU",
      image_digest:
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    });
    data.storagePolicies.push({
      ...data.storagePolicies[0],
      id: 142,
      name: "Preferred retained artifacts",
      artifact_prefix: "projects/1/retained",
      is_default: false,
    });

    render(
      <TrainingScreen
        data={data}
        busy={false}
        initialEnvironmentId={132}
        initialStoragePolicyId={142}
        initialEvaluationSplit="validation"
        onLaunch={onLaunch}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Project-default encoder",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(
      (
        screen.getByRole("radio", {
          name: /Preferred transformer CPU/,
        }) as HTMLInputElement
      ).checked,
    ).toBe(true);
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(
      (
        screen.getByRole("combobox", {
          name: "Storage policy",
        }) as HTMLSelectElement
      ).value,
    ).toBe("142");
    expect(
      (
        screen.getByRole("combobox", {
          name: "Evaluation split",
        }) as HTMLSelectElement
      ).value,
    ).toBe("validation");

    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Launch training" }));

    await waitFor(() =>
      expect(onLaunch).toHaveBeenCalledWith(
        expect.objectContaining({
          environmentId: 132,
          storagePolicyId: 142,
          evaluationPlan: { splits: ["validation"] },
        }),
      ),
    );
  });

  it("clears a stale storage default and keeps review unavailable", async () => {
    const user = userEvent.setup();
    const data = trainingData([transformerRecipe], [baseModel()]);
    data.storagePolicies = [];
    const onLaunch = vi.fn();

    render(
      <TrainingScreen
        data={data}
        busy={false}
        initialStoragePolicyId={999}
        onLaunch={onLaunch}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "Continue" }));
    await user.type(
      screen.getByRole("textbox", { name: "Model name" }),
      "Storage-gated encoder",
    );
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Immutable base model" }),
      "161",
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(
      (screen.getByRole("combobox", {
        name: "Storage policy",
      }) as HTMLSelectElement).value,
    ).toBe("0");
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(onLaunch).not.toHaveBeenCalled();
  });

  it("binds a source dataset deep link to its composed training dataset", async () => {
    const data = trainingData([classicalRecipe], []);
    data.taskVersions.push({
      ...data.taskVersions[0],
      id: 13,
      version_number: 4,
    });
    data.trainingDatasets = [
      {
        ...data.trainingDatasets[0],
        id: 21,
        name: "Other training set",
        dataset_version_id: 32,
        task_version_id: 13,
      },
      data.trainingDatasets[0],
    ];
    data.datasetVersions = [
      {
        id: 22,
        project_id: 1,
        dataset_id: 21,
        version_number: 3,
        source_uri: null,
        source_revision: "dataset-21-revision",
        source_format: "jsonl",
        data_schema: {},
        provenance: {},
        license_info: {},
        content_hash: "d".repeat(64),
        item_count: 100,
        artifact_package_id: null,
      },
      {
        id: 32,
        project_id: 1,
        dataset_id: 31,
        version_number: 1,
        source_uri: null,
        source_revision: "dataset-31-revision",
        source_format: "jsonl",
        data_schema: {},
        provenance: {},
        license_info: {},
        content_hash: "e".repeat(64),
        item_count: 100,
        artifact_package_id: null,
      },
    ];

    expect(resolveInitialTrainingDataset(data, 21)?.id).toBe(51);
    expect(resolveInitialTrainingDataset(data, 32)?.id).toBe(21);

    render(
      <TrainingScreen
        data={data}
        busy={false}
        initialDatasetId={21}
        onLaunch={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(
        (
          screen.getByRole("combobox", {
            name: "Training dataset",
          }) as HTMLSelectElement
        ).value,
      ).toBe("51");
      expect(
        (
          screen.getByRole("combobox", {
            name: "Task version",
          }) as HTMLSelectElement
        ).value,
      ).toBe("12");
    });
  });

  it("never substitutes another training dataset for an uncomposed source", async () => {
    const data = trainingData([classicalRecipe], []);
    data.datasets = [
      {
        id: 99,
        project_id: 1,
        name: "Requested public corpus",
        description: null,
        source_type: "public_registry",
      },
    ];
    data.datasetVersions = [
      {
        id: 199,
        project_id: 1,
        dataset_id: 99,
        version_number: 1,
        source_uri: "hf://example/corpus",
        source_revision: "f".repeat(40),
        source_format: "parquet",
        data_schema: {},
        provenance: {},
        license_info: {},
        content_hash: "e".repeat(64),
        item_count: 100,
        artifact_package_id: null,
      },
    ];
    const onOpenTrainingData = vi.fn();

    render(
      <TrainingScreen
        data={data}
        busy={false}
        initialDatasetId={99}
        onOpenTrainingData={onOpenTrainingData}
        onLaunch={vi.fn()}
      />,
    );

    await waitFor(() =>
      expect(
        (
          screen.getByRole("combobox", {
            name: "Training dataset",
          }) as HTMLSelectElement
        ).value,
      ).toBe("0"),
    );
    expect(
      screen.getByText(
        /Requested public corpus is not yet a composed training dataset/,
      ),
    ).toBeTruthy();
    expect(
      (screen.getByRole("button", { name: "Continue" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    await userEvent.click(
      screen.getByRole("button", { name: "Prepare requested dataset" }),
    );
    expect(onOpenTrainingData).toHaveBeenCalledTimes(1);
  });
});
