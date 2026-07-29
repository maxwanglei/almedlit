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
      screen.getByRole("button", { name: "Import 1 selected document" }),
    ).toBeTruthy();

    fireEvent.click(
      screen.getByRole("checkbox", {
        name: "Include abstract-only documents",
      }),
    );
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
});
