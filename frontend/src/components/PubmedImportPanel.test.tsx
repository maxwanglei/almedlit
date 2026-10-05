// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import PubmedImportPanel from "@/components/PubmedImportPanel";

const mocks = vi.hoisted(() => ({
  importPubmed: vi.fn(),
  previewPubmedImport: vi.fn(),
}));

vi.mock("@/api/client", () => ({
  ApiError: class ApiError extends Error {},
  importPubmed: mocks.importPubmed,
  previewPubmedImport: mocks.previewPubmedImport,
}));

beforeEach(() => {
  vi.clearAllMocks();
  mocks.previewPubmedImport.mockResolvedValue({
    items: [
      {
        pmid: "29860986",
        title: "Full-text clinical study",
        journal: "Journal",
        year: "2024",
        pmcid: "PMC100",
        status: "full_text",
        has_full_text: true,
        has_abstract: true,
      },
      {
        pmid: "29717446",
        title: "Abstract-only oncology study",
        journal: "Journal",
        year: "2023",
        pmcid: null,
        status: "abstract_only",
        has_full_text: false,
        has_abstract: true,
      },
    ],
  });
  mocks.importPubmed.mockResolvedValue({
    created: [
      {
        pmid: "29860986",
        status: "full_text",
        title: "Full-text clinical study",
        document_id: 1,
        reason: null,
      },
      {
        pmid: "29717446",
        status: "abstract_only",
        title: "Abstract-only oncology study",
        document_id: 2,
        reason: null,
      },
    ],
    skipped: [],
  });
});

afterEach(cleanup);

describe("PubmedImportPanel", () => {
  it("treats existing papers as a successful reusable import and closes the preview", async () => {
    mocks.importPubmed.mockResolvedValue({ created: [], skipped: [
      { pmid: "29860986", document_id: 42, status: "full_text", title: "Existing paper", reason: "Duplicate: already in this project; available to reuse" },
      { pmid: "29717446", document_id: null, status: "error", title: "Unavailable paper", reason: "No full text or abstract available" },
    ] });
    const onImported = vi.fn();
    render(<PubmedImportPanel projectId={7} onImported={onImported} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "29860986,29717446" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    fireEvent.click(await screen.findByRole("button", { name: "Import 2 selected documents" }));
    await waitFor(() => expect(onImported).toHaveBeenCalledOnce());
    const status = screen.getByRole("status");
    expect(status.textContent).toContain("Reused 1 document(s) already in this project.");
    expect(status.textContent).toContain("1 document(s) could not be imported.");
    expect(status.className).toContain("success");
    expect(screen.queryByText("Choose documents to import")).toBeNull();
    expect(screen.queryByText("No documents imported.")).toBeNull();
    expect(screen.queryByText(/adjust the selection, and try again/)).toBeNull();
  });
  it("makes preview and final document import distinct actions", async () => {
    const onImported = vi.fn();
    render(
      <PubmedImportPanel
        projectId={7}
        variant="personal"
        onImported={onImported}
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Import PubMed documents" }),
    ).toBeTruthy();
    fireEvent.change(screen.getByLabelText("PMIDs"), {
      target: { value: "29860986, 29717446" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));

    await waitFor(() =>
      expect(mocks.previewPubmedImport).toHaveBeenCalledWith(7, [
        "29860986",
        "29717446",
      ]),
    );
    expect(
      await screen.findByText("Choose documents to import"),
    ).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Import 2 selected documents" }),
    ).toBeTruthy();

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Include abstract-only documents",
      }),
    );
    expect(screen.getByRole("button", { name: "Import 1 selected document" })).toBeTruthy();
    fireEvent.click(screen.getByRole("checkbox", { name: "Include abstract-only documents" }));
    const importButton = screen.getByRole("button", {
      name: "Import 2 selected documents",
    });
    expect(importButton.className).toContain("import-submit-button");
    fireEvent.click(importButton);

    await waitFor(() =>
      expect(mocks.importPubmed).toHaveBeenCalledWith(
        7,
        ["29860986", "29717446"],
        true,
      ),
    );
    await waitFor(() => expect(onImported).toHaveBeenCalledTimes(1));
    expect(await screen.findByText(/Imported/)).toBeTruthy();
  });
  it("rejects arbitrary identifiers and clears a preview when the PMID input changes", async () => {
    render(<PubmedImportPanel projectId={7} onImported={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "10.1001/journal.2024.55" } });
    expect((screen.getByRole("button", { name: "Preview documents" }) as HTMLButtonElement).disabled).toBe(true);
    expect(screen.getByRole("alert").textContent).toContain("numeric PMIDs");
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "PMID: 29860986" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    await screen.findByText("Choose documents to import");
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "29717446" } });
    expect(screen.queryByText("Choose documents to import")).toBeNull();
  });
});
