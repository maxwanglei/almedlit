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
          onCreateGuideline={complete}
          onCreateTask={complete}
          onPrepareTrainingData={complete}
        />
      ) : null}
    </>
  );
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

  it("creates ordinary blind full-dataset rounds without future learning controls", async () => {
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
        onCreateGuideline={complete}
        onCreateTask={complete}
        onPrepareTrainingData={complete}
      />,
    );

    expect(screen.queryByLabelText("Learning cycle")).toBeNull();
    expect(screen.queryByLabelText("Pinned feedback set")).toBeNull();
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
});
