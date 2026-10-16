// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { GuidelineVersion } from "@/types/api";

import EvidenceGuidelineManagement from "./EvidenceGuidelineManagement";

const mocks = vi.hoisted(() => ({ createGuidelineVersion: vi.fn() }));
vi.mock("@/api/client", () => mocks);

const SAVED_GUIDELINE: GuidelineVersion = {
  id: 12,
  project_id: 7,
  version_label: "v2",
  markdown: "# Evidence rules\nReview every sentence.",
  author_id: "manager",
  status: "active",
};

function fillForm(label = "v2", markdown = SAVED_GUIDELINE.markdown): void {
  fireEvent.change(screen.getByLabelText("Guideline version"), { target: { value: label } });
  fireEvent.change(screen.getByLabelText("Guideline text"), { target: { value: markdown } });
}

function submitForm(): void {
  fireEvent.submit(screen.getByRole("form", { name: "Create evidence guideline" }));
}

beforeEach(() => {
  vi.resetAllMocks();
  mocks.createGuidelineVersion.mockResolvedValue(SAVED_GUIDELINE);
});
afterEach(cleanup);

describe("EvidenceGuidelineManagement", () => {
  it("creates an active legacy guideline and keeps its saved version visible before props refresh", async () => {
    const previous = { ...SAVED_GUIDELINE, id: 11, version_label: "v1", markdown: "Previous rules" };
    const onChanged = vi.fn().mockResolvedValue(undefined);
    render(<EvidenceGuidelineManagement projectId={7} guidelines={[previous]} onChanged={onChanged} />);
    expect(screen.getByRole("heading", { name: "Active guideline: v1" })).toBeTruthy();
    expect(screen.getByText(/Existing assignments keep their pinned guideline/)).toBeTruthy();
    expect(screen.getByLabelText("Guideline version").getAttribute("maxlength")).toBe("50");

    fillForm(" v2 ");
    submitForm();

    await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
    expect(mocks.createGuidelineVersion).toHaveBeenCalledExactlyOnceWith({
      project_id: 7,
      version_label: "v2",
      markdown: SAVED_GUIDELINE.markdown,
      status: "active",
    });
    expect(screen.getByRole("heading", { name: "Active guideline: v2" })).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("Guideline v2 is active.");
    expect((screen.getByLabelText("Guideline version") as HTMLInputElement).value).toBe("");
    expect((screen.getByLabelText("Guideline text") as HTMLTextAreaElement).value).toBe("");
    const versions = screen.getByRole("table", { name: "Evidence guideline versions" });
    expect(versions.textContent).toContain("v1");
    expect(versions.textContent).toContain("Superseded");
    expect(versions.textContent).toContain("v2");
  });

  it.each([
    { label: " ", markdown: "Review all evidence.", error: "Enter a guideline version label." },
    { label: "v".repeat(51), markdown: "Review all evidence.", error: "Guideline version labels must be 50 characters or fewer." },
    { label: "v1", markdown: " \n ", error: "Enter guideline text before activating a version." },
  ])("validates the guideline draft before saving ($error)", ({ label, markdown, error }) => {
    const onChanged = vi.fn();
    render(<EvidenceGuidelineManagement projectId={7} guidelines={[]} onChanged={onChanged} />);
    fillForm(label, markdown);

    submitForm();

    expect(screen.getByRole("alert").textContent).toBe(error);
    expect(mocks.createGuidelineVersion).not.toHaveBeenCalled();
    expect(onChanged).not.toHaveBeenCalled();
    expect((screen.getByLabelText("Guideline version") as HTMLInputElement).value).toBe(label);
    expect((screen.getByLabelText("Guideline text") as HTMLTextAreaElement).value).toBe(markdown);
  });

  it("retains a failed save's draft and retries the mutation", async () => {
    mocks.createGuidelineVersion.mockRejectedValueOnce(new Error("Guideline save failed"));
    const onChanged = vi.fn().mockResolvedValue(undefined);
    render(<EvidenceGuidelineManagement projectId={7} guidelines={[]} onChanged={onChanged} />);
    fillForm();
    submitForm();

    expect((await screen.findByRole("alert")).textContent).toBe("Guideline save failed");
    expect((screen.getByLabelText("Guideline version") as HTMLInputElement).value).toBe("v2");
    expect((screen.getByLabelText("Guideline text") as HTMLTextAreaElement).value).toBe(SAVED_GUIDELINE.markdown);
    expect(onChanged).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Create active guideline" }));

    await waitFor(() => expect(onChanged).toHaveBeenCalledOnce());
    expect(mocks.createGuidelineVersion).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("heading", { name: "Active guideline: v2" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("preserves a successful save when refresh fails and retries only refresh", async () => {
    const onChanged = vi.fn()
      .mockRejectedValueOnce(new Error("Refresh failed"))
      .mockRejectedValueOnce(new Error("Still offline"))
      .mockResolvedValueOnce(undefined);
    render(<EvidenceGuidelineManagement projectId={7} guidelines={[]} onChanged={onChanged} />);
    fillForm();
    submitForm();

    expect((await screen.findByRole("alert")).textContent).toContain("Guideline was saved");
    expect(screen.getByRole("heading", { name: "Active guideline: v2" })).toBeTruthy();
    expect(screen.getByRole("status").textContent).toContain("Guideline v2 is active.");
    expect((screen.getByRole("button", { name: "Create active guideline" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText("Guideline version") as HTMLInputElement).value).toBe("");

    fireEvent.click(screen.getByRole("button", { name: "Retry refresh" }));
    await waitFor(() => expect(onChanged).toHaveBeenCalledTimes(2));
    expect(await screen.findByRole("button", { name: "Retry refresh" })).toBeTruthy();
    expect(mocks.createGuidelineVersion).toHaveBeenCalledOnce();

    fireEvent.click(screen.getByRole("button", { name: "Retry refresh" }));

    await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
    expect(onChanged).toHaveBeenCalledTimes(3);
    expect(mocks.createGuidelineVersion).toHaveBeenCalledOnce();
    expect(screen.getByRole("heading", { name: "Active guideline: v2" })).toBeTruthy();
    expect((screen.getByRole("button", { name: "Create active guideline" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
