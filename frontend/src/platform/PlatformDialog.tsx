import { formatTaskKind } from "@/lib/taskPresentation";
import { NativeInput, NativeSelect, NativeTextarea } from "@/components/NativeField";
import ValidatedForm from "@/components/ValidatedForm";
import { useId, useState } from "react";
import { Button } from "@astryxdesign/core/Button";
import {
  BrainCircuit,
  Database,
  Repeat2,
  X,
} from "lucide-react";

import DialogFrame from "@/components/DialogFrame";

import { validateDatasetDraft, validateTaskDraft } from "./api";
import { formatCount } from "./components";
import type {
  DatasetDraft,
  FeedbackScoringDraft,
  RoundDraft,
  TaskDraft,
} from "./api";
import type { PlatformDialogKind } from "./ProjectPlatform";
import type { PlatformProjectData } from "./types";

interface PlatformDialogProps {
  kind: Exclude<PlatformDialogKind, null>;
  data: PlatformProjectData;
  busy: boolean;
  currentUserId?: number | null;
  isPersonalWorkspace?: boolean;
  taskPurpose?: "annotation" | "training";
  onClose: () => void;
  onCreateDataset: (draft: DatasetDraft) => Promise<void>;
  onCreateRound: (draft: RoundDraft) => Promise<void>;
  onScoreFeedback: (draft: FeedbackScoringDraft) => Promise<void>;
  onCreateTask: (draft: TaskDraft) => Promise<void>;
}

const ICONS = {
  dataset: Database,
  task: BrainCircuit,
  round: Repeat2,
  feedbackScore: BrainCircuit,
};

const TITLES = {
  dataset: "Import source data",
  task: "Define NLP task",
  round: "Create annotation round",
  feedbackScore: "Score dataset with model",
};

export default function PlatformDialog({
  kind,
  data,
  busy,
  currentUserId,
  isPersonalWorkspace = false,
  taskPurpose = "annotation",
  onClose,
  onCreateDataset,
  onCreateRound,
  onScoreFeedback,
  onCreateTask,
}: PlatformDialogProps): React.ReactElement {
  const titleId = useId();
  const Icon = ICONS[kind];
  const defaultDatasetVersion = [...data.datasetVersions]
    .sort((left, right) => right.version_number - left.version_number)[0]?.id ?? 0;
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
          tooltip="Close dialog"
          icon={<X size={17} />}
          isIconOnly
          variant="ghost"
          isDisabled={busy}
          onClick={onClose}
        />
      </header>
      {kind === "dataset" ? (
        <DatasetForm busy={busy} onSubmit={(draft) => submit(() => {
          validateDatasetDraft(draft);
          return onCreateDataset(draft);
        })} />
      ) : null}
      {kind === "task" ? (
        <TaskForm busy={busy} purpose={taskPurpose} onSubmit={(draft) => submit(() => {
          validateTaskDraft(draft);
          return onCreateTask(draft);
        })} />
      ) : null}
      {kind === "round" ? (
        <RoundForm
          data={data}
          busy={busy}
          defaultDatasetVersion={defaultDatasetVersion}
          defaultTaskVersion={defaultTaskVersion}
          defaultAnnotatorId={isPersonalWorkspace ? currentUserId : null}
          onSubmit={(draft) => submit(() => onCreateRound(draft))}
        />
      ) : null}
      {kind === "feedbackScore" ? (
        <FeedbackScoringForm
          data={data}
          busy={busy}
          defaultDatasetVersion={defaultDatasetVersion}
          onSubmit={(draft) => submit(() => onScoreFeedback(draft))}
        />
      ) : null}
      {formError ? <p className="platform-dialog-error" role="alert">{formError}</p> : null}
    </DialogFrame>
  );
}

function FormActions({
  busy,
  disabled = false,
  label = "Create",
}: {
  busy: boolean;
  disabled?: boolean;
  label?: string;
}): React.ReactElement {
  return (
    <div className="platform-dialog-actions">
      <Button
        label={busy ? `${label}…` : label}
        type="submit"
        variant="primary"
        isDisabled={busy || disabled}
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
    <ValidatedForm
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
      <label><span>Name</span><NativeInput required maxLength={255} value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>
        <span>Source</span>
        <NativeSelect value={sourceType} onChange={(event) => {
          const nextSourceType = event.target.value as DatasetDraft["sourceType"];
          setSourceType(nextSourceType);
          if (nextSourceType === "public_registry" && !["csv", "jsonl", "parquet"].includes(sourceFormat)) {
            setSourceFormat("jsonl");
          }
        }}>
          <option value="upload">File upload</option>
          <option value="public_registry">Public registry</option>
          <option value="project_corpus">Existing project corpus</option>
          <option value="generated">Generated data</option>
          <option value="other">Other source</option>
        </NativeSelect>
      </label>
      <label><span>Description</span><NativeTextarea rows={2} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      {sourceType === "upload" ? (
        <>
          <label>
            <span>Dataset file</span>
            <NativeInput
              required
              type="file"
              accept=".csv,.jsonl,.json,.parquet,text/csv,application/json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div className="platform-form-grid">
            <label><span>Stable ID field</span><NativeInput value={stableKeyField} placeholder="Generated when empty" onChange={(event) => setStableKeyField(event.target.value)} /></label>
            <label><span>Group ID field</span><NativeInput value={groupKeyField} placeholder="Optional leakage group" onChange={(event) => setGroupKeyField(event.target.value)} /></label>
          </div>
        </>
      ) : null}
      {sourceType === "public_registry" ? (
        <>
          <div className="platform-form-grid">
            <label><span>Hugging Face dataset</span><NativeInput required value={registryDatasetId} placeholder="owner/dataset" onChange={(event) => setRegistryDatasetId(event.target.value)} /></label>
            <label><span>Configuration</span><NativeInput value={registryConfigName} placeholder="Optional" onChange={(event) => setRegistryConfigName(event.target.value)} /></label>
          </div>
          <label>
            <span>Pinned dataset snapshot</span>
            <NativeInput
              required
              type="file"
              accept=".csv,.jsonl,.json,.parquet,text/csv,application/json"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div className="platform-form-grid">
            <label><span>Stable ID field</span><NativeInput value={stableKeyField} placeholder="Generated when empty" onChange={(event) => setStableKeyField(event.target.value)} /></label>
            <label><span>Group ID field</span><NativeInput value={groupKeyField} placeholder="Optional leakage group" onChange={(event) => setGroupKeyField(event.target.value)} /></label>
          </div>
        </>
      ) : null}
      {sourceType !== "project_corpus" ? (
        <div className="platform-form-grid">
          <label>
            <span>Format</span>
            <NativeSelect value={sourceFormat} onChange={(event) => setSourceFormat(event.target.value as DatasetDraft["sourceFormat"])}>
              <option value="csv">CSV</option>
              <option value="jsonl">JSONL</option>
              <option value="parquet">Parquet</option>
              {sourceType !== "public_registry" ? <option value="other">Other</option> : null}
            </NativeSelect>
          </label>
          <label>
            <span>Exact revision</span>
            <NativeInput
              required={sourceType !== "upload"}
              pattern={sourceType === "public_registry" ? "[a-fA-F0-9]{40}" : undefined}
              title={sourceType === "public_registry" ? "Enter the full 40-character hexadecimal commit hash." : undefined}
              value={sourceRevision}
              placeholder={sourceType === "public_registry" ? "40-character hexadecimal commit hash" : "commit, tag, or checksum"}
              onChange={(event) => setSourceRevision(event.target.value)}
            />
          </label>
        </div>
      ) : null}
      {sourceType !== "upload" && sourceType !== "public_registry" && sourceType !== "project_corpus" ? (
        <label><span>Source URI</span><NativeInput value={sourceUri} placeholder="Immutable object or corpus URI" onChange={(event) => setSourceUri(event.target.value)} /></label>
      ) : null}
      {sourceType !== "project_corpus" ? (
        <label><span>License identifier</span><NativeInput value={license} placeholder="e.g. apache-2.0" onChange={(event) => setLicense(event.target.value)} /></label>
      ) : null}
      <FormActions busy={busy} />
    </ValidatedForm>
  );
}

function TaskForm({
  busy,
  purpose,
  onSubmit,
}: {
  busy: boolean;
  purpose: "annotation" | "training";
  onSubmit: (draft: TaskDraft) => Promise<void>;
}): React.ReactElement {
  const labelsHelpId = useId();
  const [key, setKey] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [taskKind, setTaskKind] = useState<TaskDraft["taskKind"] | "paper_entities">("classification");
  const [labels, setLabels] = useState("");
  return (
    <ValidatedForm className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      void onSubmit({
        key,
        name,
        description,
        taskKind: taskKind === "paper_entities" ? "span_extraction" : taskKind,
        annotationMode: taskKind === "paper_entities" ? "paper_entities" : undefined,
        labelValues: labels.split(",").map((value) => value.trim()).filter(Boolean),
      });
    }}>
      <div className="platform-form-grid">
        <label><span>Name</span><NativeInput required value={name} onChange={(event) => setName(event.target.value)} /></label>
        <label><span>Stable key</span><NativeInput required pattern="[a-z0-9][a-z0-9_.-]*" value={key} placeholder="sentiment_classification" onChange={(event) => setKey(event.target.value)} /></label>
      </div>
      <label>
        <span>Task kind</span>
        <NativeSelect value={taskKind} onChange={(event) => setTaskKind(event.target.value as TaskDraft["taskKind"])}>
          <option value="classification">Classification</option>
          <option value="multilabel_classification">Multilabel classification</option>
          <option value="regression">Regression</option>
          {purpose === "annotation" ? <option value="paper_entities">Named entities (NER) in papers</option> : null}
          <option value="token_labeling">Token labeling (tokenized data)</option>
          <option value="span_extraction">Span extraction</option>
          <option value="relation_extraction">Relation extraction</option>
          <option value="ranking">Ranking</option>
          <option value="generation">Generation</option>
          <option value="instruction_tuning">Instruction tuning</option>
        </NativeSelect>
      </label>
      <label><span>Description</span><NativeTextarea rows={3} value={description} onChange={(event) => setDescription(event.target.value)} /></label>
      <label>
        <span>Allowed labels</span>
        <NativeInput
          required={taskKind === "paper_entities"}
          pattern={taskKind === "paper_entities" ? ".*[^\\s,].*" : undefined}
          aria-describedby={taskKind === "paper_entities" ? labelsHelpId : undefined}
          value={labels}
          placeholder={taskKind === "paper_entities" ? "Disease, Drug, Gene" : "Comma-separated; optional for open outputs"}
          onChange={(event) => setLabels(event.target.value)}
        />
      </label>
      {taskKind === "paper_entities" ? (
        <small id={labelsHelpId}>NER requires at least one entity label. Separate multiple labels with commas.</small>
      ) : null}
      <FormActions busy={busy} />
    </ValidatedForm>
  );
}

function FeedbackScoringForm({
  data,
  busy,
  defaultDatasetVersion,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultDatasetVersion: number;
  onSubmit: (draft: FeedbackScoringDraft) => Promise<void>;
}): React.ReactElement {
  const classificationTasks = data.taskVersions.filter(
    (task) => task.task_kind === "classification",
  );
  const initialTaskVersionId = classificationTasks[0]?.id ?? 0;
  const initialDatasetVersionId =
    data.datasetVersions.find(
      (version) => version.id === defaultDatasetVersion && version.item_count > 0,
    )?.id ?? data.datasetVersions.find((version) => version.item_count > 0)?.id ?? 0;
  const scorableCycles = data.cycles.filter((cycle) => {
    const task = data.taskVersions.find((item) => item.id === cycle.task_version_id);
    const dataset = data.datasetVersions.find(
      (item) => item.id === cycle.source_dataset_version_id,
    );
    const model = data.modelVersions.find(
      (item) => item.id === cycle.baseline_model_version_id,
    );
    return Boolean(
      task?.task_kind === "classification" &&
      dataset &&
      dataset.item_count > 0 &&
      model?.recipe_key === "tfidf_logistic_regression" &&
      model.framework === "scikit-learn" &&
      model.family === "conventional_ml" &&
      model.checkpoint_package_id,
    );
  });
  const [cycleId, setCycleId] = useState(0);
  const [taskVersionId, setTaskVersionId] = useState(initialTaskVersionId);
  const [datasetVersionId, setDatasetVersionId] = useState(initialDatasetVersionId);
  const [modelVersionId, setModelVersionId] = useState(0);
  const selectedCycle = data.cycles.find((cycle) => cycle.id === cycleId);
  const compatibleModels = data.modelVersions.filter(
    (version) =>
      version.task_version_id === taskVersionId &&
      version.recipe_key === "tfidf_logistic_regression" &&
      version.framework === "scikit-learn" &&
      version.family === "conventional_ml" &&
      Boolean(version.checkpoint_package_id),
  );

  function chooseCompatibleModel(nextTaskVersionId: number): void {
    const version = data.modelVersions.find(
      (item) =>
        item.task_version_id === nextTaskVersionId &&
        item.recipe_key === "tfidf_logistic_regression" &&
        item.framework === "scikit-learn" &&
        item.family === "conventional_ml" &&
        Boolean(item.checkpoint_package_id),
    );
    setModelVersionId(version?.id ?? 0);
  }

  return (
    <ValidatedForm className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      void onSubmit({
        cycleId: cycleId || null,
        taskVersionId,
        datasetVersionId,
        modelVersionId,
      });
    }}>
      <p className="platform-form-note">
        The worker scores the complete immutable dataset. Test and validation protection is
        applied later, when a targeted round is selected.
      </p>
      <label>
        <span>Learning cycle</span>
        <NativeSelect
          value={cycleId}
          onChange={(event) => {
            const nextCycleId = Number(event.target.value);
            const cycle = data.cycles.find((item) => item.id === nextCycleId);
            setCycleId(nextCycleId);
            if (cycle) {
              setDatasetVersionId(cycle.source_dataset_version_id);
              setTaskVersionId(cycle.task_version_id);
              setModelVersionId(cycle.baseline_model_version_id ?? 0);
            }
          }}
        >
          <option value={0}>No cycle</option>
          {scorableCycles.map((cycle) => (
            <option key={cycle.id} value={cycle.id}>
              Cycle {cycle.sequence} · {cycle.name}
            </option>
          ))}
        </NativeSelect>
        {selectedCycle ? <small>Dataset, task, and baseline model are pinned by this cycle.</small> : null}
      </label>
      <section className="platform-form-grid" aria-label="Scoring resources">
        <label>
          <span>Classification task</span>
          <NativeSelect
            required
            disabled={Boolean(selectedCycle)}
            value={taskVersionId}
            onChange={(event) => {
              const nextTaskVersionId = Number(event.target.value);
              setTaskVersionId(nextTaskVersionId);
              chooseCompatibleModel(nextTaskVersionId);
            }}
          >
            <option value={0}>Select a task</option>
            {classificationTasks.map((task) => (
              <option key={task.id} value={task.id}>
                Classification · v{task.version_number}
              </option>
            ))}
          </NativeSelect>
        </label>
        <label>
          <span>Dataset version</span>
          <NativeSelect
            required
            disabled={Boolean(selectedCycle)}
            value={datasetVersionId}
            onChange={(event) => setDatasetVersionId(Number(event.target.value))}
          >
            <option value={0}>Select a non-empty dataset</option>
            {data.datasetVersions.filter((version) => version.item_count > 0).map((version) => {
              const dataset = data.datasets.find((item) => item.id === version.dataset_id);
              return (
                <option key={version.id} value={version.id}>
                  {dataset?.name ?? "Dataset"} · v{version.version_number} · {formatCount(version.item_count, "record")}
                </option>
              );
            })}
          </NativeSelect>
        </label>
      </section>
      <label>
        <span>TF-IDF model</span>
        <NativeSelect
          required
          disabled={Boolean(selectedCycle)}
          value={modelVersionId}
          onChange={(event) => setModelVersionId(Number(event.target.value))}
        >
          <option value={0}>Select a compatible model</option>
          {compatibleModels.map((version) => {
            const model = data.models.find(
              (item) => item.id === version.registered_model_id,
            );
            return (
              <option key={version.id} value={version.id}>
                {model?.name ?? "Model"} · v{version.version_number}
              </option>
            );
          })}
        </NativeSelect>
      </label>
      {!classificationTasks.length ? (
        <p className="platform-dialog-error" role="alert">
          Create a classification task before scoring.
        </p>
      ) : !compatibleModels.length ? (
        <p className="platform-dialog-error" role="alert">
          No ready TF-IDF logistic-regression model matches this task.
        </p>
      ) : null}
      <FormActions
        busy={busy}
        disabled={!taskVersionId || !datasetVersionId || !modelVersionId}
        label="Start scoring"
      />
    </ValidatedForm>
  );
}

function RoundForm({
  data,
  busy,
  defaultDatasetVersion,
  defaultTaskVersion,
  defaultAnnotatorId,
  onSubmit,
}: {
  data: PlatformProjectData;
  busy: boolean;
  defaultDatasetVersion: number;
  defaultTaskVersion: number;
  defaultAnnotatorId?: number | null;
  onSubmit: (draft: RoundDraft) => Promise<void>;
}): React.ReactElement {
  const [name, setName] = useState("");
  const [cycleId, setCycleId] = useState(0);
  const [taskVersionId, setTaskVersionId] = useState(defaultTaskVersion);
  const [datasetVersionId, setDatasetVersionId] = useState(data.datasetVersions.find((version) => version.id === defaultDatasetVersion && version.item_count > 0)?.id ?? data.datasetVersions.find((version) => version.item_count > 0)?.id ?? 0);
  const [guidelineRevisionId, setGuidelineRevisionId] = useState(0);
  const [reannotationMode, setReannotationMode] =
    useState<RoundDraft["reannotationMode"]>("full_dataset");
  const [selectionStrategy, setSelectionStrategy] =
    useState<RoundDraft["selectionStrategy"]>("random");
  const [splitMapId, setSplitMapId] = useState(0);
  const [selectionLimit, setSelectionLimit] = useState(25);
  const [feedbackSetVersionId, setFeedbackSetVersionId] = useState(0);
  const [annotatorUserIds, setAnnotatorUserIds] = useState<number[]>(defaultAnnotatorId ? [defaultAnnotatorId] : []);
  const [openToAllAnnotators, setOpenToAllAnnotators] = useState(false);
  const [reason, setReason] = useState("");
  const learningEnabled = data.projectModules.effective.includes("learning");
  const availableCycles = learningEnabled ? data.cycles : [];
  const selectedCycle = data.cycles.find((cycle) => cycle.id === cycleId);
  const compatibleSplitMaps = data.splitMaps.filter(
    (splitMap) => splitMap.dataset_version_id === datasetVersionId,
  );
  const compatibleFeedbackSets = data.feedbackSets.filter((feedbackSet) => {
    if (!learningEnabled) return false;
    const run = data.feedbackRuns.find(
      (item) =>
        item.id === feedbackSet.feedback_run_id &&
        item.status === "completed" &&
        item.output_feedback_set_version_id === feedbackSet.id,
    );
    return Boolean(
      run &&
      feedbackSet.candidate_count > 0 &&
      feedbackSet.dataset_version_id === datasetVersionId &&
      feedbackSet.task_version_id === taskVersionId &&
      run.cycle_id === (cycleId || null),
    );
  });
  const selectedDataset = data.datasetVersions.find(
    (version) => version.id === datasetVersionId,
  );
  const compatibleGuidelineRevisions = data.guidelineRevisions.filter((revision) =>
    revision.task_version_id === taskVersionId && revision.status === "active",
  );

  function resetSelectionResources(
    nextDatasetVersionId: number,
    nextTaskVersionId: number,
    nextCycleId: number,
  ): void {
    setSplitMapId(
      data.splitMaps.find(
        (splitMap) => splitMap.dataset_version_id === nextDatasetVersionId,
      )?.id ?? 0,
    );
    const nextFeedbackSet = data.feedbackSets.find((feedbackSet) => {
      const run = data.feedbackRuns.find(
        (item) =>
          item.id === feedbackSet.feedback_run_id &&
          item.status === "completed" &&
          item.output_feedback_set_version_id === feedbackSet.id,
      );
      return Boolean(
        run &&
        feedbackSet.candidate_count > 0 &&
        feedbackSet.dataset_version_id === nextDatasetVersionId &&
        feedbackSet.task_version_id === nextTaskVersionId &&
        run.cycle_id === (nextCycleId || null),
      );
    });
    setFeedbackSetVersionId(nextFeedbackSet?.id ?? 0);
  }

  const targeted = reannotationMode === "targeted_subset" && compatibleSplitMaps.length > 0;
  const effectiveSelectionStrategy = compatibleFeedbackSets.length ? selectionStrategy : "random";
  const uncertaintyIncomplete =
    targeted &&
    effectiveSelectionStrategy === "uncertainty" &&
    !compatibleFeedbackSets.some((feedbackSet) => feedbackSet.id === feedbackSetVersionId);

  return (
    <ValidatedForm className="platform-dialog-form" onSubmit={(event) => {
      event.preventDefault();
      if (!selectedDataset?.item_count) return;
      void onSubmit({
        name,
        taskVersionId,
        datasetVersionId,
        cycleId: cycleId || null,
        splitMapId: targeted ? splitMapId || null : null,
        guidelineRevisionId: compatibleGuidelineRevisions.some((revision) => revision.id === guidelineRevisionId) ? guidelineRevisionId : null,
        feedbackSetVersionId: compatibleFeedbackSets.some((feedbackSet) => feedbackSet.id === feedbackSetVersionId) ? feedbackSetVersionId : null,
        assistancePolicy: "blind",
        reannotationMode: targeted ? "targeted_subset" : "full_dataset",
        selectionStrategy: targeted ? effectiveSelectionStrategy : "all",
        selectionLimit,
        annotatorUserIds,
        openToAllAnnotators,
        reason,
      });
    }}>
      <label><span>Round name</span><NativeInput required value={name} onChange={(event) => setName(event.target.value)} /></label>
      {availableCycles.length ? <label>
        <span>Learning cycle</span>
        <NativeSelect
          value={cycleId}
          onChange={(event) => {
            const nextCycleId = Number(event.target.value);
            const cycle = data.cycles.find((item) => item.id === nextCycleId);
            setCycleId(nextCycleId);
            if (cycle) {
              setTaskVersionId(cycle.task_version_id);
              setDatasetVersionId(cycle.source_dataset_version_id);
              resetSelectionResources(
                cycle.source_dataset_version_id,
                cycle.task_version_id,
                cycle.id,
              );
            } else {
              resetSelectionResources(datasetVersionId, taskVersionId, 0);
            }
          }}
        >
          <option value={0}>No cycle</option>
          {availableCycles.map((cycle) => (
            <option key={cycle.id} value={cycle.id}>Cycle {cycle.sequence} · {cycle.name}</option>
          ))}
        </NativeSelect>
        {selectedCycle ? <small>Dataset and task are pinned by this cycle.</small> : null}
      </label> : null}
      <ResourceSelectors
        data={{ ...data, datasetVersions: data.datasetVersions.filter((version) => version.item_count > 0) }}
        taskVersionId={taskVersionId}
        datasetVersionId={datasetVersionId}
        disabled={Boolean(selectedCycle)}
        onTaskChange={(nextTaskVersionId) => {
          setTaskVersionId(nextTaskVersionId);
          resetSelectionResources(datasetVersionId, nextTaskVersionId, cycleId);
        }}
        onDatasetChange={(nextDatasetVersionId) => {
          setDatasetVersionId(nextDatasetVersionId);
          resetSelectionResources(nextDatasetVersionId, taskVersionId, cycleId);
        }}
      />
      <fieldset className="platform-fieldset">
        <legend>Round coverage</legend>
        <section className="platform-option-list">
          <label>
            <NativeInput
              type="radio"
              name="round-coverage"
              checked={!targeted}
              onChange={() => setReannotationMode("full_dataset")}
            />
            <span><strong>Full dataset</strong><small>Annotate every dataset item.</small></span>
          </label>
          {compatibleSplitMaps.length ? <label>
            <NativeInput
              type="radio"
              name="round-coverage"
              checked={targeted}
              onChange={() => {
                setReannotationMode("targeted_subset");
                if (!splitMapId) setSplitMapId(compatibleSplitMaps[0]?.id ?? 0);
              }}
            />
            <span><strong>Targeted subset</strong><small>Select eligible items while preserving protected splits.</small></span>
          </label> : null}
        </section>
      </fieldset>
      {targeted ? (
        <>
          <section className="platform-form-grid" aria-label="Selection configuration">
            <label>
              <span>Split map</span>
              <NativeSelect required value={splitMapId} onChange={(event) => setSplitMapId(Number(event.target.value))}>
                <option value={0}>Select a governed split</option>
                {compatibleSplitMaps.map((splitMap) => (
                  <option key={splitMap.id} value={splitMap.id}>
                    {splitMap.name} · protects {splitMap.protected_splits.join(", ") || "none"}
                  </option>
                ))}
              </NativeSelect>
            </label>
            <label>
              <span>Selection strategy</span>
              <NativeSelect
                value={effectiveSelectionStrategy}
                onChange={(event) => setSelectionStrategy(event.target.value as "random" | "uncertainty")}
              >
                <option value="random">Random</option>
                {compatibleFeedbackSets.length ? <option value="uncertainty">Uncertainty</option> : null}
              </NativeSelect>
            </label>
          </section>
          <label>
            <span>Selection limit</span>
            <NativeInput
              required
              type="number"
              min={1}
              max={selectedDataset?.item_count || undefined}
              value={selectionLimit}
              onChange={(event) => setSelectionLimit(Number(event.target.value))}
            />
          </label>
        </>
      ) : null}
      {compatibleFeedbackSets.length ? <label>
        <span>Model feedback set</span>
        <NativeSelect
          required={targeted && effectiveSelectionStrategy === "uncertainty"}
          value={feedbackSetVersionId}
          onChange={(event) => setFeedbackSetVersionId(Number(event.target.value))}
        >
          <option value={0}>No feedback set</option>
          {compatibleFeedbackSets.map((feedbackSet) => (
            <option key={feedbackSet.id} value={feedbackSet.id}>
              Feedback set v{feedbackSet.version_number} · {formatCount(feedbackSet.candidate_count, "candidate")}
            </option>
          ))}
        </NativeSelect>
        <small>Assistance remains blind: scores affect order but predictions are not shown to annotators.</small>
      </label> : null}
      {uncertaintyIncomplete ? (
        <p className="platform-dialog-error" role="alert">
          Uncertainty selection requires a completed, non-empty feedback set for this dataset, task, and cycle.
        </p>
      ) : null}
      {compatibleGuidelineRevisions.length ? <label>
        <span>Pinned guideline</span>
        <NativeSelect value={guidelineRevisionId} onChange={(event) => setGuidelineRevisionId(Number(event.target.value))}>
          <option value={0}>No guideline</option>
          {compatibleGuidelineRevisions.map((revision) => {
            const guideline = data.guidelines.find((item) => item.id === revision.guideline_id);
            return <option key={revision.id} value={revision.id}>{guideline?.name ?? "Guideline"} · v{revision.version_number}</option>;
          })}
        </NativeSelect>
      </label> : null}
      <fieldset className="platform-form-fieldset">
        <legend>Annotators</legend>
        <section className="platform-checkbox-list">
          {data.workspaceMembers.filter((member) => member.is_active).map((member) => (
            <label key={member.user_id} className="platform-checkbox-row">
              <NativeInput
                type="checkbox"
                checked={annotatorUserIds.includes(member.user_id)}
                disabled={openToAllAnnotators}
                onChange={(event) => setAnnotatorUserIds((current) =>
                  event.target.checked
                    ? [...current, member.user_id]
                    : current.filter((userId) => userId !== member.user_id))}
              />
              <span>{member.display_name || member.username} · {member.role}</span>
            </label>
          ))}
        </section>
      </fieldset>
      <label className="platform-checkbox-row">
        <NativeInput
          type="checkbox"
          checked={openToAllAnnotators}
          onChange={(event) => {
            setOpenToAllAnnotators(event.target.checked);
            if (event.target.checked) setAnnotatorUserIds([]);
          }}
        />
        <span>Open to all workspace annotators</span>
      </label>
      <label><span>Reason</span><NativeTextarea rows={3} value={reason} onChange={(event) => setReason(event.target.value)} /></label>
      <FormActions busy={busy} disabled={
        !taskVersionId ||
        !selectedDataset?.item_count ||
        (targeted && !splitMapId) ||
        uncertaintyIncomplete ||
        (!openToAllAnnotators && !annotatorUserIds.length)
      } />
    </ValidatedForm>
  );
}

function ResourceSelectors({
  data,
  taskVersionId,
  datasetVersionId,
  onTaskChange,
  onDatasetChange,
  disabled = false,
}: {
  data: PlatformProjectData;
  taskVersionId: number;
  datasetVersionId: number;
  onTaskChange: (value: number) => void;
  onDatasetChange: (value: number) => void;
  disabled?: boolean;
}): React.ReactElement {
  return (
    <div className="platform-form-grid">
      <label>
        <span>Task version</span>
        <NativeSelect required disabled={disabled} value={taskVersionId} onChange={(event) => onTaskChange(Number(event.target.value))}>
          {data.taskVersions.map((task) => <option key={task.id} value={task.id}>{formatTaskKind(task.task_kind, task.annotation_ui?.preset)} · v{task.version_number}</option>)}
        </NativeSelect>
      </label>
      <label>
        <span>Dataset version</span>
        <NativeSelect required disabled={disabled} value={datasetVersionId} onChange={(event) => onDatasetChange(Number(event.target.value))}>
          {data.datasetVersions.map((version) => {
            const dataset = data.datasets.find((item) => item.id === version.dataset_id);
            return <option key={version.id} value={version.id}>{dataset?.name ?? "Dataset"} · v{version.version_number}</option>;
          })}
        </NativeSelect>
      </label>
    </div>
  );
}
