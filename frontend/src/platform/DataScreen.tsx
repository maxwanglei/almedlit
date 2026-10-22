import { useEffect, useId, useState, type ReactNode } from "react";
import { Button } from "@astryxdesign/core/Button";

import { request } from "@/api/client";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  formatStatus,
  formatCount,
  shortHash,
} from "./components";
import type { Dataset, DatasetVersion, PlatformProjectData, TrainingDatasetVersion } from "./types";

function datasetName(data: PlatformProjectData, datasetId: number): string {
  return data.datasets.find((dataset) => dataset.id === datasetId)?.name ?? `Dataset ${datasetId}`;
}

interface AnnotationProgress {
  dataset_id: number;
  dataset_version_id: number;
  submitted: number;
  total: number;
}

export default function DataScreen({
  data,
  legacyDocumentCount,
  onCreate,
  onPrepareTraining,
  title = "Data",
  showHeader = true,
  description = "Immutable source records, independent label sets, and stable split policies.",
  secondary,
  activeTab = "source",
  onTabChange,
  onImport,
  onAnnotate,
  onPredict,
  onCreateTraining,
  onTrain,
  onNewTrainingVersion,
}: {
  data: PlatformProjectData;
  legacyDocumentCount: number;
  onCreate: () => void;
  onPrepareTraining?: (dataset: Dataset) => void;
  title?: string;
  showHeader?: boolean;
  description?: string;
  secondary?: ReactNode;
  activeTab?: "source" | "training";
  onTabChange?: (tab: "source" | "training") => void;
  onImport?: (dataset?: Dataset) => void;
  onAnnotate?: (dataset: Dataset, version: DatasetVersion) => void;
  onPredict?: (version: DatasetVersion) => void;
  onCreateTraining?: () => void;
  onTrain?: (version: TrainingDatasetVersion) => void;
  onNewTrainingVersion?: (version: TrainingDatasetVersion) => void;
}): React.ReactElement {
  const tabsId = useId();
  const projectId = data.projectModules.project_id;
  const [progress, setProgress] = useState<{ projectId: number; rows: AnnotationProgress[] } | null>(null);
  useEffect(() => {
    let cancelled = false;
    if (!projectId || activeTab !== "source") return;
    void request<AnnotationProgress[]>(`/projects/${projectId}/datasets/annotation-progress`)
      .then((rows) => { if (!cancelled) setProgress({ projectId, rows }); })
      .catch(() => { if (!cancelled) setProgress(null); });
    return () => { cancelled = true; };
  }, [projectId, activeTab, data.datasetVersions, data.rounds]);
  const progressByVersion = new Map(
    (progress?.projectId === projectId ? progress.rows : []).map((row) => [row.dataset_version_id, row]),
  );
  const internalDatasetIds = new Set(data.datasetVersions.filter((version) => version.provenance.ingestion === "training_preparation_v1").map((version) => version.dataset_id));
  const sourceDatasets = data.datasets.filter((dataset) => !internalDatasetIds.has(dataset.id));
  return (
    <div className="platform-page">
      {showHeader ? <PlatformPageHeader
        title={title}
        description={description}
        actionLabel={(activeTab === "training" ? data.trainingDatasets.length : sourceDatasets.length || legacyDocumentCount) ? activeTab === "training" ? "Create training dataset" : onImport ? "Import PMIDs" : "Import source data" : undefined}
        onAction={activeTab === "training" ? onCreateTraining : onImport ? () => onImport() : onCreate}
        secondary={secondary}
      /> : secondary}

      {onTabChange ? <div className="platform-page-actions" role="tablist" aria-label="Dataset types" onKeyDown={(event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        const tabs = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]'));
        const index = tabs.findIndex((tab) => tab === document.activeElement);
        const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + 1) % tabs.length;
        event.preventDefault();
        tabs[next]?.focus();
        tabs[next]?.click();
      }}>
        <button type="button" className="platform-text-action" id={`${tabsId}-source-tab`} role="tab" aria-selected={activeTab === "source"} aria-controls={`${tabsId}-source-panel`} tabIndex={activeTab === "source" ? 0 : -1} onClick={() => onTabChange("source")}>Source datasets</button>
        <button type="button" className="platform-text-action" id={`${tabsId}-training-tab`} role="tab" aria-selected={activeTab === "training"} aria-controls={`${tabsId}-training-panel`} tabIndex={activeTab === "training" ? 0 : -1} onClick={() => onTabChange("training")}>Training datasets</button>
      </div> : null}
      <div id={`${tabsId}-${activeTab}-panel`} role={onTabChange ? "tabpanel" : undefined} aria-labelledby={onTabChange ? `${tabsId}-${activeTab}-tab` : undefined}>
      {activeTab === "training" ? <PlatformSection title="Training datasets" description="Saved versions prepared from annotations, external labels, or both. Source data and earlier versions remain available.">
        {data.trainingDatasets.length ? <div className="platform-table-scroll" role="region" aria-label="Training datasets" tabIndex={0}><table className="platform-table"><thead><tr><th scope="col">Training dataset</th><th scope="col">Version</th><th scope="col">Task</th><th scope="col">Labeled records / splits</th><th scope="col">Sources</th><th scope="col">Actions</th></tr></thead><tbody>
          {[...data.trainingDatasets].sort((a, b) => (b.version_number ?? 1) - (a.version_number ?? 1)).map((version) => {
            const preview = version.preparation_manifest?.preview as { item_count?: number; split_counts?: Record<string, number> } | undefined;
            const references = Array.isArray(version.preparation_manifest?.sources) ? version.preparation_manifest.sources as Array<Record<string, unknown>> : [{ dataset_version_id: version.dataset_version_id }];
            const sourceVersion = data.datasetVersions.find((item) => item.id === version.dataset_version_id);
            const split = data.splitMaps.find((item) => item.id === version.split_map_id);
            const splitCounts = preview?.split_counts ?? Object.values(split?.assignments ?? {}).reduce<Record<string, number>>((counts, value) => ({ ...counts, [value]: (counts[value] ?? 0) + 1 }), {});
            return <tr key={version.id}>
            <td><strong>{version.name}</strong></td><td>v{version.version_number ?? 1}</td>
            <td>{data.taskDefinitions.find((task) => task.id === data.taskVersions.find((item) => item.id === version.task_version_id)?.task_definition_id)?.name ?? `Task ${version.task_version_id}`}</td>
            <td>{(preview?.item_count ?? sourceVersion?.item_count ?? 0).toLocaleString()}<span>{Object.entries(splitCounts).map(([name, count]) => `${name}: ${count.toLocaleString()}`).join(" · ") || "Splits recorded"}</span></td>
            <td><details><summary>View sources ({references.length})</summary><ul>{references.map((reference, index) => {
              const source = data.datasetVersions.find((item) => item.id === reference.dataset_version_id);
              return <li key={index}>{source ? datasetName(data, source.dataset_id) : `Source version ${reference.dataset_version_id}`} · {source ? `v${source.version_number}` : "pinned version"}{reference.label_set_version_id ? ` · label set ${reference.label_set_version_id}` : ""}{reference.annotation_round_id ? ` · submitted round ${reference.annotation_round_id}` : ""}</li>;
            })}</ul><small>Fingerprint: {shortHash(version.content_hash)}</small></details></td>
            <td><div className="platform-row-actions">{onTrain ? <Button label="Train model" size="sm" onClick={() => onTrain(version)} /> : null}{onNewTrainingVersion && version.training_dataset_id ? <Button label="Create new version" size="sm" onClick={() => onNewTrainingVersion(version)} /> : null}</div></td>
          </tr>; })}
        </tbody></table></div> : <PlatformEmpty title="No training datasets yet" detail="Choose submitted annotations, import external labeled data, or combine both sources." actionLabel={onCreateTraining ? "Create training dataset" : undefined} onAction={onCreateTraining} />}
      </PlatformSection> : <>

      <PlatformSection
        title="Source datasets"
        description="Reusable source records for annotation and predictions. Create training datasets from these sources when labels are available."
      >
        {sourceDatasets.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Dataset registry table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Dataset</th>
                  <th scope="col">Source</th>
                  <th scope="col">Latest version</th>
                  <th scope="col">Records</th>
                  <th scope="col">Labels / annotation status</th>
                  {onPrepareTraining || onAnnotate || onPredict || onImport ? (
                    <th scope="col"><span className="sr-only">Actions</span></th>
                  ) : null}
                </tr>
              </thead>
              <tbody>
                {sourceDatasets.map((dataset) => {
                  const versions = data.datasetVersions
                    .filter((version) => version.dataset_id === dataset.id)
                    .sort((left, right) => right.version_number - left.version_number);
                  const latest = versions[0];
                  const layers = data.labelSets.filter((layer) => layer.dataset_version_id === latest?.id);
                  const labeledKeys = new Set(layers.flatMap((layer) => Object.keys(layer.labels ?? {})));
                  const rounds = data.rounds.filter((round) => versions.some((version) => version.id === round.dataset_version_id));
                  const submissionProgress = latest ? progressByVersion.get(latest.id) : undefined;
                  return (
                    <tr key={dataset.id}>
                      <td>
                        <strong>{dataset.name}</strong>
                        {dataset.description ? <span>{dataset.description}</span> : null}
                        {dataset.purposes?.length ? <span>{dataset.purposes.map((purpose) => purpose === "training_source" ? "Training preparation" : purpose === "inference" ? "Predictions" : "Annotation").join(" · ")}</span> : null}
                        <details><summary>Source versions</summary><ul>{versions.map((version) => <li key={version.id}>v{version.version_number} · {formatCount(version.item_count, "record")} · revision {version.source_revision}<br /><small>Fingerprint: {shortHash(version.content_hash)}</small></li>)}</ul></details>
                      </td>
                      <td><span>{formatStatus(dataset.source_type)}</span></td>
                      <td>{latest ? `v${latest.version_number}` : "No version"}</td>
                      <td>{(latest?.item_count ?? 0).toLocaleString()}</td>
                      <td>
                        {submissionProgress ? <strong>{submissionProgress.submitted}/{submissionProgress.total} submitted</strong> : null}
                        {layers.length ? formatCount(labeledKeys.size || Math.max(...layers.map((layer) => layer.label_count)), "saved label") : "No saved label set"}
                        <span>{rounds.length ? `${rounds.filter((round) => round.status === "open").length} open · ${rounds.filter((round) => round.status === "closed").length} closed rounds` : "Annotation not started"}</span>
                        {versions.filter((version) => version.id !== latest?.id).map((version) => {
                          const prior = progressByVersion.get(version.id);
                          return prior?.submitted ? <span key={version.id}>v{version.version_number}: {prior.submitted}/{prior.total} submitted (preserved)</span> : null;
                        })}
                      </td>
                      {onPrepareTraining || onAnnotate || onPredict || onImport ? (
                        <td>
                          <div className="platform-row-actions">
                            {onPrepareTraining ? <Button
                              label="Prepare for training"
                              size="sm"
                              isDisabled={!latest?.item_count}
                              onClick={() => onPrepareTraining(dataset)}
                            /> : null}
                            {onAnnotate && latest ? <Button label="Annotate" size="sm" isDisabled={!latest.item_count} onClick={() => onAnnotate(dataset, latest)} /> : null}
                            {onPredict && latest ? <Button label="Run predictions" size="sm" isDisabled={!latest.item_count} onClick={() => onPredict(latest)} /> : null}
                            {onImport && dataset.source_type === "project_corpus" ? <Button label="Import PMIDs" size="sm" onClick={() => onImport(dataset)} /> : null}
                          </div>
                        </td>
                      ) : null}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : legacyDocumentCount ? (
          <PlatformEmpty
            title="Project corpus awaiting a dataset snapshot"
            detail={`${formatCount(legacyDocumentCount, "existing document")} ${legacyDocumentCount === 1 ? "remains" : "remain"} available and can be versioned without changing ${legacyDocumentCount === 1 ? "it" : "them"}.`}
            actionLabel="Create dataset snapshot"
            onAction={onCreate}
          />
        ) : (
          <PlatformEmpty
            title="No source datasets yet"
            detail="Import papers by PMID or upload source records. Give each dataset a name, then choose annotation, predictions, or training preparation."
            actionLabel={onImport ? "Import PMIDs" : "Import source data"}
            onAction={onImport ? () => onImport() : onCreate}
          />
        )}
      </PlatformSection>
      {onImport ? <p><button type="button" className="platform-text-action" onClick={onCreate}>Import CSV, JSONL, Parquet, or a public dataset</button></p> : null}

      <details><summary>Label and split version details</summary><PlatformSection
        title="Label sets"
        description="Imported labels remain separate from human, adjudicated, and derived labels."
      >
        {data.labelSets.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Label sets table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Label set</th>
                  <th scope="col">Dataset</th>
                  <th scope="col">Source</th>
                  <th scope="col">Composition</th>
                  <th scope="col">Labels</th>
                  <th scope="col">Fingerprint</th>
                </tr>
              </thead>
              <tbody>
                {data.labelSets.map((labels) => (
                  <tr key={labels.id}>
                    <td><strong>{labels.name}</strong><span>v{labels.version_number}</span></td>
                    <td>{datasetName(data, data.datasetVersions.find(
                      (version) => version.id === labels.dataset_version_id,
                    )?.dataset_id ?? -1)}</td>
                    <td><span>{formatStatus(labels.source_kind)}</span></td>
                    <td>{labels.composition_policy}</td>
                    <td>{labels.label_count.toLocaleString()}</td>
                    <td><code>{shortHash(labels.content_hash)}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="platform-inline-empty">
            No label sets yet. Imported labels will be preserved here rather than overwritten.
          </p>
        )}
      </PlatformSection>

      <PlatformSection
        title="Split governance"
        description="Stable group assignments prevent leakage across repeated learning cycles."
      >
        {data.splitMaps.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Split governance table"
            tabIndex={0}
          >
            <table className="platform-table">
              <thead>
                <tr>
                  <th scope="col">Split map</th>
                  <th scope="col">Strategy</th>
                  <th scope="col">Seed</th>
                  <th scope="col">Protected</th>
                  <th scope="col">Fingerprint</th>
                </tr>
              </thead>
              <tbody>
                {data.splitMaps.map((split) => (
                  <tr key={split.id}>
                    <td><strong>{split.name}</strong></td>
                    <td>{split.strategy}</td>
                    <td>{split.seed}</td>
                    <td>{split.protected_splits.join(", ") || "None"}</td>
                    <td><code>{shortHash(split.content_hash)}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="platform-inline-empty">
            No split policy recorded. The protected test cohort must be fixed before comparative training.
          </p>
        )}
      </PlatformSection>
      </details>
      </>}
      </div>
      {onTabChange ? <div hidden id={`${tabsId}-${activeTab === "source" ? "training" : "source"}-panel`} role="tabpanel" aria-labelledby={`${tabsId}-${activeTab === "source" ? "training" : "source"}-tab`} /> : null}
    </div>
  );
}
