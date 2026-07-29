import { useId, useMemo, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import {
  BookOpen,
  BrainCircuit,
  Database,
  Repeat2,
  RefreshCw,
  Split,
  X,
} from "lucide-react";

import DialogFrame from "@/components/DialogFrame";

import type {
  CycleDraft,
  DatasetDraft,
  GuidelineDraft,
  RoundDraft,
  TaskDraft,
  TrainingDataDraft,
} from "./api";
import type { PlatformDialogKind } from "./ProjectPlatform";
import type {
  LearningFeedbackProducer,
  PlatformProjectData,
} from "./types";

interface PlatformDialogProps {
  kind: Exclude<PlatformDialogKind, null>;
  data: PlatformProjectData;
  busy: boolean;
  initialDatasetId?: number | null;
  onClose: () => void;
  onCreateDataset: (draft: DatasetDraft) => Promise<void>;
  onCreateCycle: (draft: CycleDraft) => Promise<void>;
  onCreateRound: (draft: RoundDraft) => Promise<void>;
  onCreateGuideline: (draft: GuidelineDraft) => Promise<void>;
  onCreateTask: (draft: TaskDraft) => Promise<void>;
  onPrepareTrainingData: (draft: TrainingDataDraft) => Promise<void>;
}

const ICONS = {
  dataset: Database,
  task: BrainCircuit,
  trainingData: Split,
  cycle: RefreshCw,
  round: Repeat2,
  guideline: BookOpen,
};

const TITLES = {
  dataset: "Add dataset version",
  task: "Define NLP task",
  trainingData: "Prepare training data",
  cycle: "Start learning cycle",
  round: "Create annotation round",
  guideline: "Create guideline",
};

export default function PlatformDialog({
  kind,
  data,
  busy,
  initialDatasetId,
  onClose,
  onCreateDataset,
  onCreateCycle,
  onCreateRound,
  onCreateGuideline,
  onCreateTask,
  onPrepareTrainingData,
}: PlatformDialogProps): React.ReactElement {
  const titleId = useId();
  const Icon = ICONS[kind];
  const defaultDatasetVersion = useMemo(() => {
    const matching = data.datasetVersions
      .filter((version) => !initialDatasetId || version.dataset_id === initialDatasetId)
      .sort((left, right) => right.version_number - left.version_number)[0];
    return matching?.id ?? data.datasetVersions[0]?.id ?? 0;
  }, [data.datasetVersions, initialDatasetId]);
  const defaultTaskVersion = data.taskVersions[0]?.id ?? 0;
  const [formError, setFormError] = useState<string | null>(null);

  async function submit(operation: () => Promise<void>): Promise<void> {
    setFormError(null);
    try {
      await operation();
      onClose();
    } catch (caught) {
      setFormError(caught instanceof Error ? caught.message : "Unable to create this resource.");
    }
  }

  return (
    <DialogFrame
      busy={busy}
      error={formError}
      labelledBy={titleId}
      backdropClassName="platform-dialog-backdrop"
      dialogClassName="platform-dialog"
      dialogElement="section"
      initialFocusSelector="form input:not([disabled]):not([type='hidden']), form select:not([disabled]), form textarea:not([disabled])"
      onDismiss={onClose}
    >
      <header>
        <div>
          <Icon size={18} aria-hidden="true" />
          <h2 id={titleId}>{TITLES[kind]}</h2>
        </div>
        <Button
          label="Close"
          icon={<X size={17} />}
          isIconOnly
          variant="ghost"
          isDisabled={busy}
          onClick={onClose}
        />
      </header>
      {kind === "dataset" ? (
        <DatasetForm busy={busy} onSubmit={(draft) => submit(() => onCreateDataset(draft))} />
      ) : null}
      {kind === "task" ? (
        <TaskForm busy={busy} onSubmit={(draft) => submit(() => onCreateTask(draft))} />
      ) : null}
      {kind === "trainingData" ? (
        <TrainingDataForm
          data={data}
          busy={busy}
          defaultDatasetVersion={defaultDatasetVersion}
          defaultTaskVersion={defaultTaskVersion}
          onSubmit={(draft) => submit(() => onPrepareTrainingData(draft))}
        />
      ) : null}
      {kind === "cycle" ? (
        <CycleForm
          data={data}
          busy={busy}
          defaultDatasetVersion={defaultDatasetVersion}
          defaultTaskVersion={defaultTaskVersion}
          onSubmit={(draft) => submit(() => onCreateCycle(draft))}
        />
      ) : null}
      {kind === "round" ? (
        <RoundForm
          data={data}
          busy={busy}
          defaultDatasetVersion={defaultDatasetVersion}
          defaultTaskVersion={defaultTaskVersion}
          onSubmit={(draft) => submit(() => onCreateRound(draft))}
        />
      ) : null}
      {kind === "guideline" ? (
        <GuidelineForm
          data={data}
          busy={busy}
          defaultTaskVersion={defaultTaskVersion}
          onSubmit={(draft) => submit(() => onCreateGuideline(draft))}
        />
      ) : null}
      {formError ? <p className="platform-dialog-error" role="alert">{formError}</p> : null}
    </DialogFrame>
  );
}

function FormActions({ busy }: { busy: boolean }): React.ReactElement {
  return (
    <div className="platform-dialog-actions">
      <Button
        label={busy ? "Creating…" : "Create"}
        type="submit"
        variant="primary"
        isDisabled={busy}
        isLoading={busy}
      />
    </div>
  );
}

function DatasetForm({
  busy,
  onSubmit,
}: {
  busy: boolean;
  onSubmit: (draft: DatasetDraft) => Promise<void>;
}): React.ReactElement {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [sourceType, setSourceType] = useState<DatasetDraft["sourceType"]>("upload");
  const [sourceUri, setSourceUri] = useState("");
  const [sourceRevision, setSourceRevision] = useState("");
  const [sourceFormat, setSourceFormat] = useState<DatasetDraft["sourceFormat"]>("jsonl");
  const [license, setLicense] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [stableKeyField, setStableKeyField] = useState("");
  const [groupKeyField, setGroupKeyField] = useState("");
  const [registryDatasetId, setRegistryDatasetId] = useState("");
  const [registryConfigName, setRegistryConfigName] = useState("");

  return (
    <form
      className="platform-dialog-form"
      onSubmit={(event) => {
        event.preventDefault();
        void onSubmit({
          name,
          description,
          sourceType,
          sourceUri,
          sourceRevision,
          sourceFormat,
          license,
          file,
          stableKeyField,
          groupKeyField,
          registryDatasetId,
          registryConfigName,
        });
      }}
    >
      <label><span>Name</span><input required maxLength={255} value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>
        <span>Source</span>
        <select value={sourceType} onChange={(event) => setSourceType(event.target.value as DatasetDraft["sourceType"])}>
          <option value="upload">File upload</option>
          <option value="public_registry">Public registry</option>
          <option value="project_corpus">Existing project corpus</option>
          <option value="generated">Generated data</option>
          <option value="other">Other source</option>
        </select>
      </label>
      <label><span>Description</span><textarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      {sourceType === "upload" ? (
        <>
          <label>
            <span>Dataset file</span>
            <input
              required
              type="file"
              accept=".csv,.jsonl,.json,.parquet,text/csv,application/json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div className="platform-form-grid">
            <label><span>Stable ID field</span><input value={stableKeyField} placeholder="Generated when empty" onChange={(event) => setStableKeyField(event.target.value)} /></label>
            <label><span>Group ID field</span><input value={groupKeyField} placeholder="Optional leakage group" onChange={(event) => setGroupKeyField(event.target.value)} /></label>
          </div>
        </>
      ) : null}
      {sourceType === "public_registry" ? (
        <>
          <div className="platform-form-grid">
            <label><span>Hugging Face dataset</span><input required value={registryDatasetId} placeholder="owner/dataset" onChange={(event) => setRegistryDatasetId(event.target.value)} /></label>
            <label><span>Configuration</span><input value={registryConfigName} placeholder="Optional" onChange={(event) => setRegistryConfigName(event.target.value)} /></label>
          </div>
          <label>
            <span>Pinned dataset snapshot</span>
            <input
              required
              type="file"
              accept=".csv,.jsonl,.json,.parquet,text/csv,application/json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div className="platform-form-grid">
            <label><span>Stable ID field</span><input value={stableKeyField} placeholder="Generated when empty" onChange={(event) => setStableKeyField(event.target.value)} /></label>
            <label><span>Group ID field</span><input value={groupKeyField} placeholder="Optional leakage group" onChange={(event) => setGroupKeyField(event.target.value)} /></label>
          </div>
        </>
      ) : null}
      {sourceType !== "project_corpus" ? (
        <div className="platform-form-grid">
          <label>
            <span>Format</span>
            <select value={sourceFormat} onChange={(event) => setSourceFormat(event.target.value as DatasetDraft["sourceFormat"])}>
              <option value="csv">CSV</option>
              <option value="jsonl">JSONL</option>
              <option value="parquet">Parquet</option>
              <option value="other">Other</option>
            </select>
          </label>
          <label>
            <span>Exact revision</span>
            <input
              required={sourceType !== "upload"}
              value={sourceRevision}
              placeholder={sourceType === "public_registry" ? "40-character commit" : "commit, tag, or checksum"}
              onChange={(event) => setSourceRevision(event.target.value)}
            />
          </label>
        </div>
      ) : null}
      {sourceType !== "upload" && sourceType !== "public_registry" && sourceType !== "project_corpus" ? (
        <label><span>Source URI</span><input value={sourceUri} placeholder="Immutable object or corpus URI" onChange={(event) => setSourceUri(event.target.value)} /></label>
      ) : null}
      {sourceType !== "project_corpus" ? (
        <label><span>License identifier</span><input value={license} placeholder="e.g. apache-2.0" onChange={(event) => setLicense(event.target.value)} /></label>
      ) : null}
      <FormActions busy={busy} />
    </form>
  );
}

function TaskForm({
  busy,
  onSubmit,
}: {
  busy: boolean;
  onSubmit: (draft: TaskDraft) => Promise<void>;
}): React.ReactElement {
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskKind, setTaskKind] = useState<TaskDraft["taskKind"]>("classification");
  const [labels, setLabels] = useState("");
  return (
    <form className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      void onSubmit({
        key,
        name,
        description,
        taskKind,
        labelValues: labels.split(",").map((value) => value.trim()).filter(Boolean),
      });
    }}>
      <div className="platform-form-grid">
        <label><span>Name</span><input required value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label><span>Stable key</span><input required pattern="[a-z0-9][a-z0-9_.-]*" value={key} placeholder="sentiment_classification" onChange={(event) => setKey(event.target.value)} /></label>
      </div>
      <label>
        <span>Task kind</span>
        <select value={taskKind} onChange={(event) => setTaskKind(event.target.value as TaskDraft["taskKind"])}>
          <option value="classification">Classification</option>
          <option value="multilabel_classification">Multilabel classification</option>
          <option value="regression">Regression</option>
          <option value="token_labeling">Token labeling / NER</option>
          <option value="span_extraction">Span extraction</option>
          <option value="relation_extraction">Relation extraction</option>
          <option value="ranking">Ranking</option>
          <option value="generation">Generation</option>
          <option value="instruction_tuning">Instruction tuning</option>
        </select>
      </label>
      <label><span>Description</span><textarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label>
        <span>Allowed labels</span>
        <input
          value={labels}
          placeholder="Comma-separated; optional for open outputs"
          onChange={(event) => setLabels(event.target.value)}
        />
      </label>
      <FormActions busy={busy} />
    </form>
  );
}

function TrainingDataForm({
  data,
  busy,
  defaultDatasetVersion,
  defaultTaskVersion,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultDatasetVersion: number;
  defaultTaskVersion: number;
  onSubmit: (draft: TrainingDataDraft) => Promise<void>;
}): React.ReactElement {
  const defaultLabelSet = data.labelSets.find(
    (labels) =>
      labels.dataset_version_id === defaultDatasetVersion &&
      labels.task_version_id === defaultTaskVersion &&
      labels.composition_policy !== "exclude",
  );
  const [name, setName] = useState("");
  const [taskVersionId, setTaskVersionId] = useState(defaultTaskVersion);
  const [datasetVersionId, setDatasetVersionId] = useState(defaultDatasetVersion);
  const [inputField, setInputField] = useState("text");
  const [labelField, setLabelField] = useState("label");
  const [labelSource, setLabelSource] = useState<
    TrainingDataDraft["labelSource"]
  >(defaultLabelSet ? "existing_label_set" : "dataset_field");
  const [labelSetVersionId, setLabelSetVersionId] = useState(
    defaultLabelSet?.id ?? 0,
  );
  const [trainPercent, setTrainPercent] = useState(80);
  const [validationPercent, setValidationPercent] = useState(10);
  const testPercent = 100 - trainPercent - validationPercent;
  const compatibleLabelSets = data.labelSets.filter(
    (labels) =>
      labels.dataset_version_id === datasetVersionId &&
      labels.task_version_id === taskVersionId &&
      labels.composition_policy !== "exclude",
  );
  const selectCompatibleLabelSet = (
    nextDatasetVersionId: number,
    nextTaskVersionId: number,
  ): void => {
    const next = data.labelSets.find(
      (labels) =>
        labels.dataset_version_id === nextDatasetVersionId &&
        labels.task_version_id === nextTaskVersionId &&
        labels.composition_policy !== "exclude",
    );
    setLabelSetVersionId(next?.id ?? 0);
  };
  const labelSourceIncomplete =
    labelSource === "existing_label_set"
      ? !compatibleLabelSets.some((labels) => labels.id === labelSetVersionId)
      : !labelField.trim();
  return (
    <form className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      void onSubmit({
        name,
        taskVersionId,
        datasetVersionId,
        inputField,
        labelSource,
        labelSetVersionId:
          labelSource === "existing_label_set" ? labelSetVersionId : null,
        labelField,
        trainPercent,
        validationPercent,
      });
    }}>
      <label><span>Training dataset name</span><input required value={name} onChange={(event) => setName(event.target.value)} /></label>
      <ResourceSelectors
        data={data}
        taskVersionId={taskVersionId}
        datasetVersionId={datasetVersionId}
        onTaskChange={(nextTaskVersionId) => {
          setTaskVersionId(nextTaskVersionId);
          selectCompatibleLabelSet(datasetVersionId, nextTaskVersionId);
        }}
        onDatasetChange={(nextDatasetVersionId) => {
          setDatasetVersionId(nextDatasetVersionId);
          selectCompatibleLabelSet(nextDatasetVersionId, taskVersionId);
        }}
      />
      <label><span>Input field</span><input required value={inputField} onChange={(event) => setInputField(event.target.value)} /></label>
      <fieldset className="platform-fieldset">
        <legend>Label source</legend>
        <div className="platform-option-list">
          <label data-disabled={!compatibleLabelSets.length}>
            <input
              type="radio"
              name="training-label-source"
              checked={labelSource === "existing_label_set"}
              disabled={!compatibleLabelSets.length}
              onChange={() => setLabelSource("existing_label_set")}
            />
            <span>
              <strong>Existing label layer</strong>
              <small>Use imported, human, or adjudicated labels without changing them.</small>
            </span>
          </label>
          <label>
            <input
              type="radio"
              name="training-label-source"
              checked={labelSource === "dataset_field"}
              onChange={() => setLabelSource("dataset_field")}
            />
            <span>
              <strong>Dataset field</strong>
              <small>Extract preserved source labels into a separate immutable layer.</small>
            </span>
          </label>
        </div>
      </fieldset>
      {labelSource === "existing_label_set" ? (
        <label>
          <span>Label layer</span>
          <select
            required
            value={labelSetVersionId}
            onChange={(event) => setLabelSetVersionId(Number(event.target.value))}
          >
            <option value={0}>Select a compatible label layer</option>
            {compatibleLabelSets.map((labels) => (
              <option key={labels.id} value={labels.id}>
                {labels.name} · v{labels.version_number} · {labels.source_kind} · {labels.label_count} labels
              </option>
            ))}
          </select>
        </label>
      ) : (
        <label>
          <span>Imported label field</span>
          <input
            required
            value={labelField}
            onChange={(event) => setLabelField(event.target.value)}
          />
        </label>
      )}
      <div className="platform-form-grid">
        <label>
          <span>Train percent</span>
          <input type="number" min={1} max={98} value={trainPercent} onChange={(event) => setTrainPercent(Number(event.target.value))} />
        </label>
        <label>
          <span>Validation percent</span>
          <input type="number" min={1} max={98} value={validationPercent} onChange={(event) => setValidationPercent(Number(event.target.value))} />
        </label>
      </div>
      <p className="platform-form-note">
        Test: {testPercent}% · protected from training, selection, prompt tuning, and guideline mining.
      </p>
      <FormActions
        busy={
          busy ||
          !taskVersionId ||
          !datasetVersionId ||
          labelSourceIncomplete ||
          testPercent < 1
        }
      />
    </form>
  );
}

function CycleForm({
  data,
  busy,
  defaultDatasetVersion,
  defaultTaskVersion,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultDatasetVersion: number;
  defaultTaskVersion: number;
  onSubmit: (draft: CycleDraft) => Promise<void>;
}): React.ReactElement {
  type SourceChoice = "registered_model" | LearningFeedbackProducer;
  const [name, setName] = useState("");
  const [goal, setGoal] = useState<CycleDraft["goal"]>("reannotate");
  const [taskVersionId, setTaskVersionId] = useState(defaultTaskVersion);
  const [datasetVersionId, setDatasetVersionId] = useState(defaultDatasetVersion);
  const [baselineModelVersionId, setBaselineModelVersionId] = useState(0);
  const compatibleModelVersions = data.modelVersions.filter(
    (version) => version.task_version_id === taskVersionId,
  );
  const [sourceType, setSourceType] = useState<SourceChoice>(
    compatibleModelVersions.length ? "registered_model" : "rule",
  );
  const [sourceName, setSourceName] = useState("");
  const [provider, setProvider] = useState("");
  const [externalModelId, setExternalModelId] = useState("");
  const [exactRevision, setExactRevision] = useState("");
  const [egressPolicy, setEgressPolicy] = useState("");
  const sourceIncomplete =
    (sourceType === "registered_model" && !baselineModelVersionId) ||
    (sourceType === "external_llm" &&
      (!sourceName.trim() ||
        !provider.trim() ||
        !externalModelId.trim() ||
        !exactRevision.trim() ||
        !egressPolicy.trim())) ||
    (sourceType !== "registered_model" &&
      sourceType !== "external_llm" &&
      !sourceName.trim());
  return (
    <form className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      const feedbackSources =
        sourceType === "registered_model"
          ? []
          : [{
              producer_type: sourceType,
              name: sourceName.trim(),
              provider: sourceType === "external_llm" ? provider.trim() : null,
              external_model_id:
                sourceType === "external_llm" ? externalModelId.trim() : null,
              exact_revision:
                sourceType === "external_llm" ? exactRevision.trim() : null,
              configuration: {},
              data_egress_policy:
                sourceType === "external_llm"
                  ? { policy_reference: egressPolicy.trim() }
                  : {},
            }];
      void onSubmit({
        name,
        goal,
        taskVersionId,
        datasetVersionId,
        baselineModelVersionId:
          sourceType === "registered_model" ? baselineModelVersionId || null : null,
        feedbackSources,
      });
    }}>
      <label><span>Cycle name</span><input required value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>
        <span>Goal</span>
        <select value={goal} onChange={(event) => setGoal(event.target.value as CycleDraft["goal"])}>
          <option value="reannotate">Re-annotate with feedback</option>
          <option value="expand_pool">Expand the data pool</option>
          <option value="guideline_pilot">Pilot a guideline</option>
          <option value="error_remediation">Remediate errors</option>
        </select>
      </label>
      <ResourceSelectors
        data={data}
        taskVersionId={taskVersionId}
        datasetVersionId={datasetVersionId}
        onTaskChange={(nextTaskVersionId) => {
          setTaskVersionId(nextTaskVersionId);
          setBaselineModelVersionId(0);
        }}
        onDatasetChange={setDatasetVersionId}
      />
      <label>
        <span>Feedback source</span>
        <select
          value={sourceType}
          onChange={(event) => {
            setSourceType(event.target.value as SourceChoice);
            setBaselineModelVersionId(0);
          }}
        >
          <option value="registered_model" disabled={!compatibleModelVersions.length}>
            Registered model
          </option>
          <option value="external_llm">External LLM</option>
          <option value="rule">Rule system</option>
          <option value="dictionary">Dictionary</option>
          <option value="ensemble">Ensemble</option>
          <option value="human_disagreement">Human disagreement</option>
        </select>
      </label>
      {sourceType === "registered_model" ? (
        <label>
          <span>Model version</span>
          <select
            required
            value={baselineModelVersionId}
            onChange={(event) => setBaselineModelVersionId(Number(event.target.value))}
          >
            <option value={0}>Select a compatible model</option>
            {compatibleModelVersions.map((version) => {
              const model = data.models.find(
                (item) => item.id === version.registered_model_id,
              );
              return (
                <option key={version.id} value={version.id}>
                  {model?.name ?? "Model"} · v{version.version_number}
                </option>
              );
            })}
          </select>
        </label>
      ) : (
        <label>
          <span>Source name</span>
          <input
            required
            value={sourceName}
            onChange={(event) => setSourceName(event.target.value)}
          />
        </label>
      )}
      {sourceType === "external_llm" ? (
        <>
          <label>
            <span>Provider</span>
            <input
              required
              value={provider}
              onChange={(event) => setProvider(event.target.value)}
            />
          </label>
          <label>
            <span>Provider model ID</span>
            <input
              required
              value={externalModelId}
              onChange={(event) => setExternalModelId(event.target.value)}
            />
          </label>
          <label>
            <span>Exact model revision</span>
            <input
              required
              value={exactRevision}
              onChange={(event) => setExactRevision(event.target.value)}
            />
          </label>
          <label>
            <span>Data-egress policy</span>
            <input
              required
              value={egressPolicy}
              onChange={(event) => setEgressPolicy(event.target.value)}
            />
          </label>
        </>
      ) : null}
      <FormActions
        busy={busy || !taskVersionId || !datasetVersionId || sourceIncomplete}
      />
    </form>
  );
}

function RoundForm({
  data,
  busy,
  defaultDatasetVersion,
  defaultTaskVersion,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultDatasetVersion: number;
  defaultTaskVersion: number;
  onSubmit: (draft: RoundDraft) => Promise<void>;
}): React.ReactElement {
  const [name, setName] = useState("");
  const [taskVersionId, setTaskVersionId] = useState(defaultTaskVersion);
  const [datasetVersionId, setDatasetVersionId] = useState(defaultDatasetVersion);
  const [guidelineRevisionId, setGuidelineRevisionId] = useState(0);
  const [annotatorUserIds, setAnnotatorUserIds] = useState<number[]>([]);
  const [openToAllAnnotators, setOpenToAllAnnotators] = useState(false);
  const [reason, setReason] = useState("");

  return (
    <form className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      void onSubmit({
        name,
        taskVersionId,
        datasetVersionId,
        cycleId: null,
        guidelineRevisionId: guidelineRevisionId || null,
        feedbackSetVersionId: null,
        assistancePolicy: "blind",
        reannotationMode: "full_dataset",
        selectionStrategy: "all",
        selectionLimit: 1,
        annotatorUserIds,
        openToAllAnnotators,
        reason,
      });
    }}>
      <label><span>Round name</span><input required value={name} onChange={(event) => setName(event.target.value)} /></label>
      <ResourceSelectors
        data={data}
        taskVersionId={taskVersionId}
        datasetVersionId={datasetVersionId}
        onTaskChange={setTaskVersionId}
        onDatasetChange={setDatasetVersionId}
      />
      <label>
        <span>Pinned guideline</span>
        <select
          value={guidelineRevisionId}
          onChange={(event) => setGuidelineRevisionId(Number(event.target.value))}
        >
          <option value={0}>No guideline</option>
          {data.guidelineRevisions.map((revision) => {
            const guideline = data.guidelines.find(
              (item) => item.id === revision.guideline_id,
            );
            return (
              <option key={revision.id} value={revision.id}>
                {guideline?.name ?? "Guideline"} · v{revision.version_number}
              </option>
            );
          })}
        </select>
      </label>
      <fieldset className="platform-form-fieldset">
        <legend>Annotators</legend>
        <div className="platform-checkbox-list">
          {data.workspaceMembers
            .filter((member) => member.is_active)
            .map((member) => (
              <label key={member.user_id} className="platform-checkbox-row">
                <input
                  type="checkbox"
                  checked={annotatorUserIds.includes(member.user_id)}
                  disabled={openToAllAnnotators}
                  onChange={(event) => {
                    setAnnotatorUserIds((current) =>
                      event.target.checked
                        ? [...current, member.user_id]
                        : current.filter((userId) => userId !== member.user_id),
                    );
                  }}
                />
                <span>
                  {member.display_name || member.username} · {member.role}
                </span>
              </label>
            ))}
        </div>
      </fieldset>
      <label className="platform-checkbox-row">
        <input
          type="checkbox"
          checked={openToAllAnnotators}
          onChange={(event) => {
            setOpenToAllAnnotators(event.target.checked);
            if (event.target.checked) setAnnotatorUserIds([]);
          }}
        />
        <span>Open to all workspace annotators</span>
      </label>
      <label><span>Reason</span><textarea rows={3} value={reason} onChange={(event) => setReason(event.target.value)} /></label>
      <FormActions
        busy={
          busy ||
          !taskVersionId ||
          !datasetVersionId ||
          (!openToAllAnnotators && !annotatorUserIds.length)
        }
      />
    </form>
  );
}

function GuidelineForm({
  data,
  busy,
  defaultTaskVersion,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultTaskVersion: number;
  onSubmit: (draft: GuidelineDraft) => Promise<void>;
}): React.ReactElement {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskVersionId, setTaskVersionId] = useState(defaultTaskVersion);
  const [markdown, setMarkdown] = useState("");
  const [rationale, setRationale] = useState("");
  const taskVersion = data.taskVersions.find((item) => item.id === taskVersionId);
  return (
    <form className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      if (!taskVersion) return;
      void onSubmit({
        name,
        description,
        taskDefinitionId: taskVersion.task_definition_id,
        taskVersionId,
        markdown,
        rationale,
      });
    }}>
      <label><span>Name</span><input required value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>
        <span>Task version</span>
        <select value={taskVersionId} onChange={(event) => setTaskVersionId(Number(event.target.value))}>
          {data.taskVersions.map((task) => <option key={task.id} value={task.id}>{task.task_kind.replace(/_/g, " ")} · v{task.version_number}</option>)}
        </select>
      </label>
      <label><span>Description</span><input value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label><span>Guideline content</span><textarea required rows={10} value={markdown} onChange={(event) => setMarkdown(event.target.value)} /></label>
      <label><span>Rationale</span><textarea rows={3} value={rationale} onChange={(event) => setRationale(event.target.value)} /></label>
      <FormActions busy={busy || !taskVersion} />
    </form>
  );
}

function ResourceSelectors({
  data,
  taskVersionId,
  datasetVersionId,
  onTaskChange,
  onDatasetChange,
}: {
  data: PlatformProjectData;
  taskVersionId: number;
  datasetVersionId: number;
  onTaskChange: (value: number) => void;
  onDatasetChange: (value: number) => void;
}): React.ReactElement {
  return (
    <div className="platform-form-grid">
      <label>
        <span>Task version</span>
        <select required value={taskVersionId} onChange={(event) => onTaskChange(Number(event.target.value))}>
          {data.taskVersions.map((task) => <option key={task.id} value={task.id}>{task.task_kind.replace(/_/g, " ")} · v{task.version_number}</option>)}
        </select>
      </label>
      <label>
        <span>Dataset version</span>
        <select required value={datasetVersionId} onChange={(event) => onDatasetChange(Number(event.target.value))}>
          {data.datasetVersions.map((version) => {
            const dataset = data.datasets.find((item) => item.id === version.dataset_id);
            return <option key={version.id} value={version.id}>{dataset?.name ?? "Dataset"} · v{version.version_number}</option>;
          })}
        </select>
      </label>
    </div>
  );
}
