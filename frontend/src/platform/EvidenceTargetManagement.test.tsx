// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { EvidenceTarget, EvidenceTargetVersion, ProjectTask } from "@/types/api";

import EvidenceTargetManagement from "./EvidenceTargetManagement";

const mocks = vi.hoisted(() => ({
  createEvidenceTarget: vi.fn(),
  createEvidenceTargetVersion: vi.fn(),
  activateEvidenceTarget: vi.fn(),
  deactivateEvidenceTarget: vi.fn(),
}));
vi.mock("@/api/client", () => mocks);

const task: ProjectTask = {
  id: 1, project_id: 7, annotation_type: "evidence_block", display_name: "Treatment evidence",
  description: null, enabled: true, sort_order: 0, labels: [], settings: {},
};
function version(id = 101, number = 1): EvidenceTargetVersion {
  return {
    id, target_id: 41, version_number: number, text: `Find treatment benefits ${number}`,
    guidance: "Include reported outcomes", inclusion_guidance: null, exclusion_guidance: null,
    metadata_: {}, created_by_user_id: 9, created_at: "2026-10-15T12:00:00Z", updated_at: "2026-10-15T12:00:00Z",
  };
}
const target: EvidenceTarget = {
  id: 41, project_id: 7, task_id: 1, key: "benefits", name: "Treatment benefits", description: null,
  is_active: false, active_version_id: null, versions: [version()], created_by_user_id: 9,
  created_at: "2026-10-15T12:00:00Z", updated_at: "2026-10-15T12:00:00Z",
};

function fillTarget(): void {
  fireEvent.change(screen.getByRole("textbox", { name: "Target key" }), { target: { value: "benefits" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Target name" }), { target: { value: "Treatment benefits" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Target text" }), { target: { value: "Find treatment benefits 1" } });
  fireEvent.change(screen.getByRole("textbox", { name: "Target guidance" }), { target: { value: "Include reported outcomes" } });
}

beforeEach(() => {
  vi.resetAllMocks();
  mocks.createEvidenceTarget.mockResolvedValue(target);
  mocks.createEvidenceTargetVersion.mockResolvedValue(version(103, 3));
  mocks.activateEvidenceTarget.mockImplementation(async (_projectId: number, _targetId: number, versionId: number) => ({
    ...target, is_active: true, active_version_id: versionId,
  }));
  mocks.deactivateEvidenceTarget.mockResolvedValue(target);
});
afterEach(cleanup);

describe("EvidenceTargetManagement", () => {
  it("creates a target before separately activating its selected version", async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    render(<EvidenceTargetManagement projectId={7} tasks={[task]} targets={[]} onChanged={onChanged} />);
    fillTarget();
    fireEvent.click(screen.getByRole("button", { name: "Create target" }));

    const card = await screen.findByRole("article", { name: "Evidence target Treatment benefits" });
    expect(mocks.createEvidenceTarget).toHaveBeenCalledWith(7, {
      task_id: 1, key: "benefits", name: "Treatment benefits",
      initial_version: { text: "Find treatment benefits 1", guidance: "Include reported outcomes" },
    });
    expect(mocks.activateEvidenceTarget).not.toHaveBeenCalled();
    fireEvent.click(within(card).getByRole("button", { name: "Activate version" }));

    await waitFor(() => expect(mocks.activateEvidenceTarget).toHaveBeenCalledWith(7, 41, 101));
    expect(await within(card).findByText("Active · v1")).toBeTruthy();
    expect(onChanged).toHaveBeenCalledTimes(2);
  });

  it("keeps the created target available after activation fails and retries only activation", async () => {
    mocks.activateEvidenceTarget.mockRejectedValueOnce(new Error("Activation failed"));
    render(<EvidenceTargetManagement projectId={7} tasks={[task]} targets={[target]} onChanged={async () => undefined} />);
    const card = screen.getByRole("article", { name: "Evidence target Treatment benefits" });
    fireEvent.click(within(card).getByRole("button", { name: "Activate version" }));

    expect((await screen.findByRole("alert")).textContent).toContain("Activation failed");
    expect(within(card).getByText("Inactive")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Activate version" }));

    expect(await within(card).findByText("Active · v1")).toBeTruthy();
    expect(mocks.activateEvidenceTarget).toHaveBeenCalledTimes(2);
    expect(mocks.createEvidenceTarget).not.toHaveBeenCalled();
  });

  it("filters targets by project and evidence task and preserves the active version when adding an immutable version", async () => {
    const original = { ...target, is_active: true, active_version_id: 101, versions: [version(), version(102, 2)] };
    mocks.activateEvidenceTarget.mockResolvedValue({ ...original, active_version_id: 103, versions: [...original.versions, version(103, 3)] });
    render(<EvidenceTargetManagement projectId={7} tasks={[task, { ...task, id: 2, annotation_type: "entity", display_name: "Entities" }]}
      targets={[original, { ...target, id: 42, task_id: 2, name: "Wrong task" }, { ...target, id: 43, project_id: 8, name: "Wrong project" }]}
      onChanged={async () => undefined} />);
    expect(screen.queryByText("Wrong task")).toBeNull();
    expect(screen.queryByText("Wrong project")).toBeNull();
    expect(screen.queryByRole("option", { name: "Entities" })).toBeNull();
    const card = screen.getByRole("article", { name: "Evidence target Treatment benefits" });
    expect((within(card).getByRole("combobox", { name: "Target version" }) as HTMLSelectElement).value).toBe("101");

    fireEvent.click(within(card).getByText("Create a new version"));
    fireEvent.change(within(card).getByRole("textbox", { name: "New target text" }), { target: { value: "Find treatment benefits 3" } });
    fireEvent.change(within(card).getByRole("textbox", { name: "New target guidance" }), { target: { value: "Include reported outcomes" } });
    fireEvent.click(within(card).getByRole("button", { name: "Create version" }));

    await waitFor(() => expect(mocks.createEvidenceTargetVersion).toHaveBeenCalledWith(7, 41, {
      text: "Find treatment benefits 3", guidance: "Include reported outcomes",
    }));
    await waitFor(() => expect((within(card).getByRole("combobox", { name: "Target version" }) as HTMLSelectElement).value).toBe("103"));
    expect(within(card).getByText("Active · v1")).toBeTruthy();
    expect(original.versions).toHaveLength(2);
    expect(mocks.activateEvidenceTarget).not.toHaveBeenCalled();
    fireEvent.click(within(card).getByRole("button", { name: "Activate version" }));
    expect(await within(card).findByText("Active · v3")).toBeTruthy();
    fireEvent.click(within(card).getByRole("button", { name: "Deactivate target" }));
    expect(await within(card).findByText("Inactive")).toBeTruthy();
    expect(mocks.deactivateEvidenceTarget).toHaveBeenCalledWith(7, 41);
  });

  it("retains a saved target when refreshing setup fails and retries refresh without recreating it", async () => {
    const onChanged = vi.fn().mockRejectedValueOnce(new Error("Refresh failed")).mockResolvedValue(undefined);
    render(<EvidenceTargetManagement projectId={7} tasks={[task]} targets={[]} onChanged={onChanged} />);
    fillTarget();
    fireEvent.click(screen.getByRole("button", { name: "Create target" }));

    expect(await screen.findByRole("article", { name: "Evidence target Treatment benefits" })).toBeTruthy();
    expect((await screen.findByRole("alert")).textContent).toContain("saved");
    expect((screen.getByRole("textbox", { name: "Target key" }) as HTMLInputElement).value).toBe("");
    fireEvent.click(screen.getByRole("button", { name: "Retry refresh" }));

    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(mocks.createEvidenceTarget).toHaveBeenCalledTimes(1);
    expect(onChanged).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("article", { name: "Evidence target Treatment benefits" })).toBeTruthy();
  });

  it.each([
    { reason: "Create an active evidence guideline before activating targets.", enabled: true },
    { reason: undefined, enabled: false },
  ])("blocks activation when setup is incomplete (enabled=$enabled)", ({ reason, enabled }) => {
    render(<EvidenceTargetManagement projectId={7} tasks={[{ ...task, enabled }]} targets={[target]}
      activationBlockedReason={reason} onChanged={async () => undefined} />);
    const card = screen.getByRole("article", { name: "Evidence target Treatment benefits" });
    const activate = within(card).getByRole("button", { name: "Activate version" }) as HTMLButtonElement;

    expect(activate.disabled).toBe(true);
    fireEvent.click(activate);
    expect(mocks.activateEvidenceTarget).not.toHaveBeenCalled();
    expect(screen.getByText(reason ?? "Enable this evidence task before activating a target.")).toBeTruthy();
    fireEvent.click(within(card).getByText("Create a new version"));
    expect((within(card).getByRole("button", { name: "Create version" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
