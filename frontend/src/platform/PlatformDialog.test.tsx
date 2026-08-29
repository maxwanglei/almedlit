// @vitest-environment jsdom

import { useState } from "react";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PlatformDialog from "./PlatformDialog";
import {
  EMPTY_PLATFORM_PROJECT_DATA,
  type PlatformProjectData,
} from "./types";

function Harness(): React.ReactElement {
  const [open, setOpen] = useState(false);
  const complete = async (): Promise<void> => undefined;
  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>
        Open dialog
      </button>
      {open ? (
        <PlatformDialog
          kind="task"
          data={EMPTY_PLATFORM_PROJECT_DATA}
          busy={false}
          onClose={() => setOpen(false)}
          onCreateDataset={complete}
          onCreateCycle={complete}
          onCreateRound={complete}
          onScoreFeedback={complete}
          onCreateGuideline={complete}
          onCreateTask={complete}
          onPrepareTrainingData={complete}
        />
      ) : null}
    </>
  );
}

function learningProjectData(): PlatformProjectData {
  return {
    ...EMPTY_PLATFORM_PROJECT_DATA,
    workspaceMembers: [],
    taskVersions: [
      {
        id: 12,
        project_id: 1,
        task_definition_id: 11,
        version_number: 1,
        task_kind: "classification",
        input_schema: {},
        output_schema: { type: "string", enum: ["include", "exclude"] },
        label_rules: {},
        annotation_ui: {},
        metrics: ["f1"],
        trainer_compatibility: ["tfidf_logistic_regression"],
        content_hash: "task-hash",
      },
    ],
    datasets: [
      {
        id: 21,
        project_id: 1,
        name: "Screening pool",
        description: null,
        source_type: "upload",
      },
    ],
    datasetVersions: [
      {
        id: 22,
        project_id: 1,
        dataset_id: 21,
        version_number: 1,
        source_uri: null,
        source_revision: "upload-1",
        source_format: "jsonl",
        data_schema: {},
        provenance: {},
        license_info: {},
        content_hash: "dataset-hash",
        item_count: 100,
        artifact_package_id: null,
      },
    ],
    models: [
      {
        id: 80,
        project_id: 1,
        name: "Screening baseline",
        description: null,
        lifecycle_status: "candidate",
      },
    ],
    modelVersions: [
      {
        id: 81,
        project_id: 1,
        registered_model_id: 80,
        version_number: 2,
        parent_version_id: null,
        task_version_id: 12,
        training_dataset_version_id: null,
        family: "conventional_ml",
        framework: "scikit-learn",
        base_model: {},
        training_method: "full",
        recipe_key: "tfidf_logistic_regression",
        recipe_version: "1",
        parameters: { fields: { input_field: "text", target_field: "label" } },
        metrics: {},
        runtime_digest: "runtime-hash",
        content_hash: "model-hash",
        checkpoint_package_id: 501,
      },
    ],
    cycles: [
      {
        id: 5,
        project_id: 1,
        name: "Uncertain abstracts",
        sequence: 2,
        status: "active",
        current_stage: "feedback",
        task_version_id: 12,
        source_dataset_version_id: 22,
        baseline_model_version_id: 81,
        feedback_sources: [],
        output_training_dataset_version_id: null,
        output_model_version_id: null,
        metadata: {},
      },
    ],
    splitMaps: [
      {
        id: 41,
        project_id: 1,
        dataset_version_id: 22,
        name: "Governed split",
        strategy: "group_stratified",
        seed: 42,
        group_key_field: null,
        assignments: {},
        protected_splits: ["test", "validation"],
        content_hash: "split-hash",
      },
    ],
    feedbackRuns: [
      {
        id: 72,
        project_id: 1,
        dataset_version_id: 22,
        task_version_id: 12,
        producer_type: "registered_model",
        cycle_id: null,
        model_version_id: 81,
        provider: null,
        external_model_id: null,
        exact_revision: null,
        prompt_template_hash: null,
        configuration: {},
        data_egress_policy: {},
        status: "completed",
        output_feedback_set_version_id: 73,
        failure_code: null,
        failure_reason: null,
        started_at: null,
        heartbeat_at: null,
        completed_at: "2026-08-18T12:00:00Z",
      },
    ],
    feedbackSets: [
      {
        id: 73,
        project_id: 1,
        feedback_run_id: 72,
        dataset_version_id: 22,
        task_version_id: 12,
        version_number: 1,
        output_schema: { type: "string" },
        candidate_count: 100,
        content_hash: "feedback-hash",
        artifact_package_id: null,
      },
    ],
  };
}

beforeEach(() => {
  Object.defineProperty(window, "requestAnimationFrame", {
    configurable: true,
    value: (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    },
  });
  Object.defineProperty(window, "cancelAnimationFrame", {
    configurable: true,
    value: vi.fn(),
  });
});

afterEach(() => {
  cleanup();
});

describe("PlatformDialog accessibility", () => {
  it("moves focus inside, traps Tab, closes with Escape, and restores focus", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const trigger = screen.getByRole("button", { name: "Open dialog" });

    await user.click(trigger);

    const dialog = screen.getByRole("dialog", { name: "Define NLP task" });
    expect(dialog.getAttribute("aria-modal")).toBe("true");
    const name = screen.getByRole("textbox", { name: "Name" });
    await waitFor(() => expect(document.activeElement).toBe(name));

    const close = screen.getByRole("button", { name: "Close" });
    const create = screen.getByRole("button", { name: "Create" });
    close.focus();
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(create);
    await user.tab();
    expect(document.activeElement).toBe(close);

    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog")).toBeNull();
    expect(document.activeElement).toBe(trigger);
    expect(document.body.style.overflow).toBe("");
  });

  it("creates a blind full-dataset round while exposing optional learning controls", async () => {
    const user = userEvent.setup();
    const onCreateRound = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    const complete = async (): Promise<void> => undefined;
    const data: PlatformProjectData = {
      ...EMPTY_PLATFORM_PROJECT_DATA,
      workspaceMembers: [
        {
          id: 1,
          workspace_id: 1,
          user_id: 7,
          username: "annotator",
          display_name: "Project Annotator",
          email: null,
          role: "annotator",
          is_active: true,
        },
      ],
      taskVersions: [
        {
          id: 12,
          project_id: 1,
          task_definition_id: 11,
          version_number: 1,
          task_kind: "classification",
          input_schema: {},
          output_schema: {},
          label_rules: {},
          annotation_ui: {},
          metrics: ["f1"],
          trainer_compatibility: [],
          content_hash: "task-hash",
        },
      ],
      datasetVersions: [
        {
          id: 22,
          project_id: 1,
          dataset_id: 21,
          version_number: 1,
          source_uri: null,
          source_revision: "upload-1",
          source_format: "jsonl",
          data_schema: {},
          provenance: {},
          license_info: {},
          content_hash: "dataset-hash",
          item_count: 10,
          artifact_package_id: null,
        },
      ],
    };

    render(
      <PlatformDialog
        kind="round"
        data={data}
        busy={false}
        onClose={onClose}
        onCreateDataset={complete}
        onCreateCycle={complete}
        onCreateRound={onCreateRound}
        onScoreFeedback={complete}
        onCreateGuideline={complete}
        onCreateTask={complete}
        onPrepareTrainingData={complete}
      />,
    );

    expect(screen.getByLabelText("Learning cycle")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: /Model feedback set/ })).toBeTruthy();
    expect(screen.queryByLabelText("Feedback visibility")).toBeNull();
    expect(screen.queryByLabelText("Selection strategy")).toBeNull();

    await user.type(screen.getByLabelText("Round name"), "Initial annotation");
    await user.click(
      screen.getByRole("checkbox", { name: /Project Annotator/ }),
    );
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(onCreateRound).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Initial annotation",
          cycleId: null,
          splitMapId: null,
          feedbackSetVersionId: null,
          assistancePolicy: "blind",
          reannotationMode: "full_dataset",
          selectionStrategy: "all",
          annotatorUserIds: [7],
        }),
      ),
    );
  });

  it("offers the existing project corpus as a dataset source", async () => {
    const user = userEvent.setup();
    const onCreateDataset = vi.fn().mockResolvedValue(undefined);
    const complete = async (): Promise<void> => undefined;

    render(
      <PlatformDialog
        kind="dataset"
        data={EMPTY_PLATFORM_PROJECT_DATA}
        busy={false}
        onClose={vi.fn()}
        onCreateDataset={onCreateDataset}
        onCreateCycle={complete}
        onCreateRound={complete}
        onScoreFeedback={complete}
        onCreateGuideline={complete}
        onCreateTask={complete}
        onPrepareTrainingData={complete}
      />,
    );

    await user.type(screen.getByRole("textbox", { name: "Name" }), "Project corpus");
    await user.selectOptions(
      screen.getByRole("combobox", { name: "Source" }),
      "project_corpus",
    );
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() =>
      expect(onCreateDataset).toHaveBeenCalledWith(
        expect.objectContaining({
          name: "Project corpus",
          sourceType: "project_corpus",
          file: null,
        }),
      ),
    );
  });

  it("pins scoring resources from a compatible cycle", async () => {
    const user = userEvent.setup();
    const onScoreFeedback = vi.fn().mockResolvedValue(undefined);
    const complete = async (): Promise<void> => undefined;
    render(
      <PlatformDialog
        kind="feedbackScore"
        data={learningProjectData()}
        busy={false}
        onClose={vi.fn()}
        onCreateDataset={complete}
        onCreateCycle={complete}
        onCreateRound={complete}
        onScoreFeedback={onScoreFeedback}
        onCreateGuideline={complete}
        onCreateTask={complete}
        onPrepareTrainingData={complete}
      />,
    );

    await user.selectOptions(screen.getByLabelText("Learning cycle"), "5");
    expect(screen.getByLabelText("Classification task").hasAttribute("disabled")).toBe(true);
    expect(screen.getByLabelText("Dataset version").hasAttribute("disabled")).toBe(true);
    expect(screen.getByLabelText("TF-IDF model").hasAttribute("disabled")).toBe(true);
    await user.click(screen.getByRole("button", { name: "Start scoring" }));

    await waitFor(() => expect(onScoreFeedback).toHaveBeenCalledWith({
      cycleId: 5,
      taskVersionId: 12,
      datasetVersionId: 22,
      modelVersionId: 81,
    }));
  });

  it("creates uncertainty rounds with an explicit split and compatible feedback", async () => {
    const user = userEvent.setup();
    const onCreateRound = vi.fn().mockResolvedValue(undefined);
    const complete = async (): Promise<void> => undefined;
    render(
      <PlatformDialog
        kind="round"
        data={learningProjectData()}
        busy={false}
        onClose={vi.fn()}
        onCreateDataset={complete}
        onCreateCycle={complete}
        onCreateRound={onCreateRound}
        onScoreFeedback={complete}
        onCreateGuideline={complete}
        onCreateTask={complete}
        onPrepareTrainingData={complete}
      />,
    );

    await user.type(screen.getByLabelText("Round name"), "Highest uncertainty");
    await user.click(screen.getByRole("radio", { name: /Targeted subset/ }));
    await user.selectOptions(screen.getByLabelText("Selection strategy"), "uncertainty");
    await user.selectOptions(
      screen.getByRole("combobox", { name: /Model feedback set/ }),
      "73",
    );
    await user.click(screen.getByLabelText("Open to all workspace annotators"));
    await user.click(screen.getByRole("button", { name: "Create" }));

    await waitFor(() => expect(onCreateRound).toHaveBeenCalledWith(
      expect.objectContaining({
        splitMapId: 41,
        feedbackSetVersionId: 73,
        reannotationMode: "targeted_subset",
        selectionStrategy: "uncertainty",
        assistancePolicy: "blind",
      }),
    ));
  });
});
