import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
  shortHash,
} from "./components";
import type { PlatformProjectData } from "./types";

function baseModelName(baseModel: Record<string, unknown>): string {
  for (const key of ["display_name", "source_model_id", "model_id", "name"]) {
    const value = baseModel[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return "From scratch";
}

function metricSummary(metrics: Record<string, number | null>): string {
  const values = Object.entries(metrics)
    .filter((entry): entry is [string, number] => typeof entry[1] === "number")
    .slice(0, 2)
    .map(([name, value]) => `${name.replace(/_/g, " ")} ${value.toFixed(3)}`);
  return values.join(" · ");
}

export default function ModelsScreen({
  data,
  onTrain,
}: {
  data: PlatformProjectData;
  onTrain: () => void;
}): React.ReactElement {
  return (
    <div className="platform-page">
      <PlatformPageHeader
        title="Models"
        description="Named model identities with immutable, traceable versions."
        actionLabel="Train model"
        onAction={onTrain}
      />
      <PlatformSection
        title="Model registry"
        description="A training run creates a version under a stable project model name."
      >
        {data.models.length ? (
          <div
            className="platform-table-scroll platform-table-scroll--summary"
            role="region"
            aria-label="Model registry table"
            tabIndex={0}
          >
            <table className="platform-table platform-table--summary">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Version</th>
                  <th scope="col">Family</th>
                  <th scope="col">Framework</th>
                  <th scope="col">Base model</th>
                  <th scope="col">Training method</th>
                  <th scope="col">Task</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Status</th>
                  <th scope="col">Evaluation</th>
                </tr>
              </thead>
              <tbody>
                {data.models.flatMap((model) => {
                  const versions = data.modelVersions
                    .filter((version) => version.registered_model_id === model.id)
                    .sort((left, right) => right.version_number - left.version_number);
                  if (!versions.length) {
                    return [
                      <tr key={`${model.id}:empty`}>
                        <td data-label="Model" data-priority="identity">
                          <strong>{model.name}</strong>
                          <span>{model.description}</span>
                        </td>
                        <td data-label="Version">Not trained</td>
                        <td data-label="Details" colSpan={6}>—</td>
                        <td data-label="Status" data-priority="status">
                          <PlatformStatus value={model.lifecycle_status} />
                        </td>
                        <td data-label="Evaluation">—</td>
                      </tr>,
                    ];
                  }
                  return versions.map((version, index) => {
                    const evaluation = [...data.modelEvaluations]
                      .filter((item) => item.model_version_id === version.id)
                      .sort((left, right) => right.id - left.id)[0];
                    const versionStatus = evaluation
                      ? evaluation.status === "succeeded"
                        ? "validated"
                        : "candidate"
                      : model.lifecycle_status;
                    const taskVersion = data.taskVersions.find(
                      (item) => item.id === version.task_version_id,
                    );
                    const task = taskVersion
                      ? data.taskDefinitions.find(
                          (item) => item.id === taskVersion.task_definition_id,
                        )
                      : null;
                    const trainingDataset = version.training_dataset_version_id
                      ? data.trainingDatasets.find(
                          (item) => item.id === version.training_dataset_version_id,
                        )
                      : null;
                    return (
                      <tr key={version.id}>
                        <td data-label="Model" data-priority="identity">
                          <strong>{model.name}</strong>
                          {index === 0 && model.description ? (
                            <span>{model.description}</span>
                          ) : null}
                        </td>
                        <td data-label="Version">
                          v{version.version_number}
                          <span><code>{shortHash(version.content_hash)}</code></span>
                        </td>
                        <td data-label="Family">{version.family.replace(/_/g, " ")}</td>
                        <td data-label="Framework">{version.framework}</td>
                        <td data-label="Base model">{baseModelName(version.base_model)}</td>
                        <td data-label="Training method">{version.training_method.replace(/_/g, " ")}</td>
                        <td data-label="Task">
                          {task?.name ?? taskVersion?.task_kind.replace(/_/g, " ") ?? "Unknown"}
                          {taskVersion ? <span>Task v{taskVersion.version_number}</span> : null}
                        </td>
                        <td data-label="Dataset">{trainingDataset?.name ?? "External"}</td>
                        <td data-label="Status" data-priority="status">
                          <PlatformStatus value={versionStatus} />
                        </td>
                        <td data-label="Evaluation">
                          {evaluation ? (
                            <>
                              <PlatformStatus value={evaluation.status} />
                              <span>
                                {metricSummary(evaluation.metrics) ||
                                  evaluation.status_reason ||
                                  "No aggregate metrics"}
                              </span>
                            </>
                          ) : (
                            "Not evaluated"
                          )}
                        </td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <PlatformEmpty
            title="No named models"
            detail="Choose a name during training so every resulting version has a clear identity."
            actionLabel="Train first model"
            onAction={onTrain}
          />
        )}
      </PlatformSection>
    </div>
  );
}
