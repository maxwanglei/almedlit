import type { ReactNode } from "react";
import { Button } from "@astryxdesign/core/Button";

import {
  PlatformEmpty,
  PlatformPageHeader,
  PlatformSection,
  PlatformStatus,
  shortHash,
} from "./components";
import type { Dataset, PlatformProjectData } from "./types";

function datasetName(data: PlatformProjectData, datasetId: number): string {
  return data.datasets.find((dataset) => dataset.id === datasetId)?.name ?? `Dataset ${datasetId}`;
}

export default function DataScreen({
  data,
  legacyDocumentCount,
  onCreate,
  onPrepareTraining,
  title = "Data",
  description = "Immutable source records, independent label layers, and stable split policies.",
  secondary,
}: {
  data: PlatformProjectData;
  legacyDocumentCount: number;
  onCreate: () => void;
  onPrepareTraining?: (dataset: Dataset) => void;
  title?: string;
  description?: string;
  secondary?: ReactNode;
}): React.ReactElement {
  return (
    <div className="platform-page">
      <PlatformPageHeader
        title={title}
        description={description}
        actionLabel="Add dataset"
        onAction={onCreate}
        secondary={secondary}
      />

      <PlatformSection
        title="Dataset registry"
        description="Origin affects provenance, not whether a dataset can be annotated or trained."
      >
        {data.datasets.length ? (
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
                  <th scope="col">Revision</th>
                  {onPrepareTraining ? (
                    <th scope="col"><span className="sr-only">Actions</span></th>
                  ) : null}
                </tr>
              </thead>
              <tbody>
                {data.datasets.map((dataset) => {
                  const versions = data.datasetVersions
                    .filter((version) => version.dataset_id === dataset.id)
                    .sort((left, right) => right.version_number - left.version_number);
                  const latest = versions[0];
                  return (
                    <tr key={dataset.id}>
                      <td>
                        <strong>{dataset.name}</strong>
                        {dataset.description ? <span>{dataset.description}</span> : null}
                      </td>
                      <td><PlatformStatus value={dataset.source_type} /></td>
                      <td>{latest ? `v${latest.version_number}` : "No version"}</td>
                      <td>{latest?.item_count ?? 0}</td>
                      <td><code>{latest?.source_revision ?? "Not pinned"}</code></td>
                      {onPrepareTraining ? (
                        <td>
                          <div className="platform-row-actions">
                            <Button
                              label="Prepare training"
                              size="sm"
                              isDisabled={!latest?.item_count}
                              onClick={() => onPrepareTraining(dataset)}
                            />
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
            detail={`${legacyDocumentCount} existing documents remain available and can be versioned without changing them.`}
            actionLabel="Create dataset snapshot"
            onAction={onCreate}
          />
        ) : (
          <PlatformEmpty
            title="No source data"
            detail="Add CSV, JSONL, Parquet, a project corpus snapshot, or a pinned public registry revision."
            actionLabel="Add dataset"
            onAction={onCreate}
          />
        )}
      </PlatformSection>

      <PlatformSection
        title="Label layers"
        description="Imported labels remain separate from human, adjudicated, and derived labels."
      >
        {data.labelSets.length ? (
          <div
            className="platform-table-scroll"
            role="region"
            aria-label="Label layers table"
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
                    <td><PlatformStatus value={labels.source_kind} /></td>
                    <td>{labels.composition_policy}</td>
                    <td>{labels.label_count}</td>
                    <td><code>{shortHash(labels.content_hash)}</code></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="platform-inline-empty">
            No label layers yet. Imported labels will be preserved here rather than overwritten.
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
    </div>
  );
}
