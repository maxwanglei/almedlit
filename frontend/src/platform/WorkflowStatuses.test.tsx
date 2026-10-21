// @vitest-environment jsdom
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import OverviewScreen from "./OverviewScreen";
import QualityScreen from "./QualityScreen";
import { PlatformStatus } from "./components";
import { EMPTY_PLATFORM_PROJECT_DATA, type AnnotationRound, type PlatformProjectData } from "./types";

vi.mock("./EvidenceAdjudicationPanel", () => ({ default: () => null }));
afterEach(cleanup);
const rounds: AnnotationRound[] = ["draft", "open", "closed", "cancelled"].map((status, index) => ({
  id: index + 1, project_id: 7, name: `${status} round`, sequence: index + 1, status,
  dataset_version_id: 22, task_version_id: 12, annotator_user_ids: [7], open_to_all_annotators: false,
  cycle_id: null, guideline_revision_id: null, selection_set_version_id: null, feedback_set_version_id: null,
  assistance_policy: "blind", reannotation_mode: "full_dataset", reason: null, opened_at: null, closed_at: null,
}));
const data: PlatformProjectData = { ...EMPTY_PLATFORM_PROJECT_DATA, rounds, projectModules: { project_id: 7, selected: ["annotate"], effective: ["annotate"], workspace_capabilities: ["annotation"] } };
function statValue(label: string): string | null {
  const container = screen.getByText(label).parentElement;
  if (!container) throw new Error(`Missing stat ${label}`);
  return within(container).getAllByRole("definition")[0].textContent;
}
describe("round lifecycle vocabulary", () => {
  it("counts only open rounds as open work", () => {
    render(<OverviewScreen data={data} documents={[]} assignments={[]} progress={null} onOpenData={vi.fn()} />);
    expect(statValue("Open work")).toBe("1");
  });
  it("counts closed rounds as available for review", () => {
    render(<QualityScreen data={data} projectId={7} documents={[]} assignments={[]} progress={null} allowSoloGold={false} />);
    expect(statValue("Reviewable rounds")).toBe("1");
    expect(statValue("Final rounds")).toBe("1");
  });
  it("distinguishes open work from cancelled rounds", () => {
    render(<><PlatformStatus value="open" /><PlatformStatus value="cancelled" /></>);
    expect(screen.getByText("Open").getAttribute("data-variant")).not.toBe(screen.getByText("Cancelled").getAttribute("data-variant"));
  });
});
