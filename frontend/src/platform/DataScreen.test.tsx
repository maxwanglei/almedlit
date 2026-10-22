// @vitest-environment jsdom

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import DataScreen from "./DataScreen";
import { EMPTY_PLATFORM_PROJECT_DATA, type PlatformProjectData } from "./types";

const mocks = vi.hoisted(() => ({ request: vi.fn() }));
vi.mock("@/api/client", () => ({ request: mocks.request }));
const data: PlatformProjectData = {
  ...EMPTY_PLATFORM_PROJECT_DATA,
  projectModules: { ...EMPTY_PLATFORM_PROJECT_DATA.projectModules, project_id: 7 },
  datasets: [{ id: 2, project_id: 7, name: "Tucatinib papers", source_type: "project_corpus", description: null }],
  datasetVersions: [{ id: 22, project_id: 7, dataset_id: 2, version_number: 1, item_count: 3, source_uri: null, source_revision: "revision", source_format: "jsonl", data_schema: {}, provenance: {}, license_info: {}, content_hash: "hash", artifact_package_id: null }],
};

beforeEach(() => { vi.clearAllMocks(); });
afterEach(cleanup);

describe("Source annotation progress", () => {
  it("retains submitted work and closed rounds when papers create a new source version", async () => {
    mocks.request.mockResolvedValue([{ dataset_id: 2, dataset_version_id: 22, total: 3, submitted: 3 }, { dataset_id: 2, dataset_version_id: 23, total: 4, submitted: 0 }]);
    const prior = data.datasetVersions[0];
    if (!prior) throw new Error("Fixture requires a source version");
    render(<DataScreen data={{ ...data, datasetVersions: [prior, { ...prior, id: 23, version_number: 2, item_count: 4 }], rounds: [{ id: 71, project_id: 7, cycle_id: null, name: "Prior work", sequence: 1, dataset_version_id: 22, task_version_id: 12, guideline_revision_id: null, selection_set_version_id: null, feedback_set_version_id: null, assistance_policy: "blind", reannotation_mode: "full_dataset", annotator_user_ids: [7], open_to_all_annotators: false, reason: null, status: "closed", opened_at: null, closed_at: null }] }} legacyDocumentCount={0} onCreate={vi.fn()} />);
    expect(await screen.findByText("v1: 3/3 submitted (preserved)")).toBeTruthy();
    expect(screen.getByText("0 open · 1 closed rounds")).toBeTruthy();
    expect(screen.queryByText("Annotation not started")).toBeNull();
  });

  it("shows submitted examples separately from saved labels", async () => {
    mocks.request.mockResolvedValue([{ dataset_id: 2, dataset_version_id: 22, total: 3, submitted: 2 }]);
    render(<DataScreen data={data} legacyDocumentCount={0} onCreate={vi.fn()} />);
    await screen.findByText("2/3 submitted");
    expect(mocks.request).toHaveBeenCalledWith("/projects/7/datasets/annotation-progress");
    expect(screen.getByText("No saved label set")).toBeTruthy();
  });

  it("keeps source actions usable when optional progress fails", async () => {
    mocks.request.mockRejectedValue(new Error("Progress unavailable"));
    render(<DataScreen data={data} legacyDocumentCount={0} onCreate={vi.fn()} />);
    await waitFor(() => expect(mocks.request).toHaveBeenCalled());
    expect(screen.getByText("Tucatinib papers")).toBeTruthy();
    expect(screen.getByText("Annotation not started")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("does not display another project's optional counts after a project change", async () => {
    mocks.request.mockResolvedValueOnce([{ dataset_id: 2, dataset_version_id: 22, total: 3, submitted: 2 }]);
    const rendered = render(<DataScreen data={data} legacyDocumentCount={0} onCreate={vi.fn()} />);
    await screen.findByText("2/3 submitted");
    mocks.request.mockReturnValue(new Promise(() => undefined));
    rendered.rerender(<DataScreen data={{ ...data, projectModules: { ...data.projectModules, project_id: 8 } }} legacyDocumentCount={0} onCreate={vi.fn()} />);
    expect(screen.queryByText("2/3 submitted")).toBeNull();
  });
});
