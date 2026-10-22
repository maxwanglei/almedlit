// @vitest-environment jsdom

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  vi.resetAllMocks();
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
  const previewItems = (pmids: string[]) => pmids.map((pmid) => ({
    pmid, title: `Paper ${pmid}`, journal: "Journal", year: "2024", pmcid: null,
    status: "abstract_only", has_full_text: false, has_abstract: true,
  }));

  it("previews and imports bounded batches with progress and one combined result", async () => {
    const pmids = Array.from({ length: 201 }, (_, index) => String(10000 + index));
    mocks.previewPubmedImport.mockImplementation(async (_projectId: number, values: string[]) => ({
      items: previewItems(values),
    }));
    let finishFirst: (value: { created: unknown[]; skipped: unknown[] }) => void = () => undefined;
    const firstImport = new Promise<{ created: unknown[]; skipped: unknown[] }>((resolve) => {
      finishFirst = resolve;
    });
    const outcomes = (values: string[]) => values.map((pmid) => ({
      pmid, status: "abstract_only", title: `Paper ${pmid}`, document_id: Number(pmid), reason: null,
    }));
    mocks.importPubmed.mockImplementation(async (_projectId: number, values: string[]) => ({
      created: outcomes(values), skipped: [],
    })).mockReturnValueOnce(firstImport);
    const onImported = vi.fn();
    render(<PubmedImportPanel projectId={7} onImported={onImported} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: pmids.join(",") } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    fireEvent.click(await screen.findByRole("button", { name: "Import 201 selected documents" }));
    expect(screen.getByText(/Importing.*0 of 201/)).toBeTruthy();
    await act(async () => finishFirst({ created: outcomes(pmids.slice(0, 100)), skipped: [] }));
    await waitFor(() => expect(onImported).toHaveBeenCalledOnce());
    expect(mocks.previewPubmedImport.mock.calls.map((call) => call[1].length)).toEqual([100, 100, 1]);
    expect(mocks.importPubmed.mock.calls.map((call) => call[1].length)).toEqual([100, 100, 1]);
    expect(onImported.mock.calls[0]?.[0].created).toHaveLength(201);
  });

  it("disables importing while re-previewing the selected items", async () => {
    let finishPreview: (value: { items: ReturnType<typeof previewItems> }) => void = () => undefined;
    render(<PubmedImportPanel projectId={7} onImported={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "29860986,29717446" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    await screen.findByRole("button", { name: "Import 2 selected documents" });
    mocks.previewPubmedImport.mockReturnValueOnce(new Promise((resolve) => { finishPreview = resolve; }));
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    const button = screen.getByRole("button", { name: "Import 2 selected documents" });
    expect(button).toHaveProperty("disabled", true);
    fireEvent.click(button);
    expect(mocks.importPubmed).not.toHaveBeenCalled();
    await act(async () => finishPreview({ items: previewItems(["29860986", "29717446"]) }));
  });

  it("accepts an arbitrary CSV column heading and ignores other columns", async () => {
    render(<PubmedImportPanel projectId={7} onImported={vi.fn()} />);
    const file = new File(['Article identifier,Title\r\n"29860986","A trial, with commas"\r\n29717446,Review'], "papers.csv", { type: "text/csv" });
    fireEvent.change(screen.getByLabelText("Upload PMID file"), { target: { files: [file] } });
    await waitFor(() => expect(screen.getByLabelText("PMIDs")).toHaveProperty("value", "29860986\n29717446"));
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    await waitFor(() => expect(mocks.previewPubmedImport).toHaveBeenCalledWith(7, ["29860986", "29717446"]));
  });

  it("retains unprocessed selections and reports successful earlier batches after failure", async () => {
    const pmids = Array.from({ length: 201 }, (_, index) => String(10000 + index));
    mocks.previewPubmedImport.mockImplementation(async (_projectId: number, values: string[]) => ({ items: previewItems(values) }));
    mocks.importPubmed.mockResolvedValueOnce({
      created: pmids.slice(0, 100).map((pmid) => ({ pmid, document_id: Number(pmid), status: "abstract_only", title: "Saved", reason: null })),
      skipped: [],
    }).mockRejectedValueOnce(new Error("Connection interrupted"));
    const onImported = vi.fn();
    render(<PubmedImportPanel projectId={7} onImported={onImported} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: pmids.join(",") } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    fireEvent.click(await screen.findByRole("button", { name: "Import 201 selected documents" }));
    await waitFor(() => expect(onImported).toHaveBeenCalledOnce());
    expect(onImported.mock.calls[0]?.[0].created).toHaveLength(100);
    expect(screen.getByRole("button", { name: "Import 101 selected documents" })).toBeTruthy();
    expect(screen.getByRole("alert").textContent).toContain("100 of 201");
  });

  it("ignores previews from a previous project", async () => {
    let finishPreview: (value: { items: ReturnType<typeof previewItems> }) => void = () => undefined;
    mocks.previewPubmedImport.mockReturnValueOnce(new Promise((resolve) => { finishPreview = resolve; }));
    const view = render(<PubmedImportPanel projectId={7} onImported={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("PMIDs"), { target: { value: "29860986" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview documents" }));
    view.rerender(<PubmedImportPanel projectId={8} onImported={vi.fn()} />);
    await act(async () => finishPreview({ items: previewItems(["29860986"]) }));
    expect(screen.queryByText("Choose documents to import")).toBeNull();
    expect(screen.getByLabelText("PMIDs")).toHaveProperty("value", "");
  });

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
    const status = screen.getByRole("status", { name: "Import result" });
    expect(status.textContent).toContain("Reused 1 document already in this project.");
    expect(status.textContent).toContain("1 document could not be imported.");
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
      screen.getByRole("heading", { name: "Add papers from PubMed" }),
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
