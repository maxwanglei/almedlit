import { forwardRef, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ForwardedRef } from "react";

import { importPubmed, previewPubmedImport } from "@/api/client";
import { ApiError } from "@/api/client";
import type { ImportPreviewItem, ImportResponse, ImportStatus } from "@/types/api";

interface PubmedImportPanelProps {
  projectId: number;
  onImported: (result: ImportResponse) => void | Promise<void>;
  onOpenWork?: () => void;
  variant?: "team" | "personal";
}

export interface PubmedImportPanelHandle {
  focusInput: () => void;
}

const STATUS_LABEL: Record<ImportStatus, string> = {
  full_text: "Full text",
  abstract_only: "Abstract only",
  not_found: "Not found",
  error: "Error",
};

function parsePmids(raw: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const token of raw.replace(/\bPMID:?\s*(\d+)/gi, "$1").split(/[\s,;]+/)) {
    const digits = /^\d+$/.test(token) ? token : "";
    if (digits && !seen.has(digits)) {
      seen.add(digits);
      out.push(digits);
    }
  }
  return out;
}

function isImportable(status: ImportStatus): boolean {
  return status === "full_text" || status === "abstract_only";
}

function PubmedImportPanel(
  { projectId, onImported, onOpenWork, variant = "team" }: PubmedImportPanelProps,
  ref: ForwardedRef<PubmedImportPanelHandle>,
) {
  const [rawInput, setRawInput] = useState("");
  const [items, setItems] = useState<ImportPreviewItem[] | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [previewing, setPreviewing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImportResponse | null>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const parsedCount = useMemo(() => parsePmids(rawInput).length, [rawInput]);
  const invalidTokens = rawInput.replace(/\bPMID:?\s*(\d+)/gi, "$1").split(/[\s,;]+/).filter((token) => token && !/^\d+$/.test(token));
  const isPersonal = variant === "personal";
  const previewCounts = useMemo(() => {
    const rows = items ?? [];
    return {
      fullText: rows.filter((item) => item.status === "full_text").length,
      abstractOnly: rows.filter((item) => item.status === "abstract_only").length,
      unavailable: rows.filter((item) => !isImportable(item.status)).length,
    };
  }, [items]);

  useImperativeHandle(ref, () => ({
    focusInput: () => {
      sectionRef.current?.scrollIntoView({ behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
      textareaRef.current?.focus({ preventScroll: true });
    },
  }));

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const text = typeof reader.result === "string" ? reader.result : "";
      setRawInput((previous) => (previous ? `${previous}\n${text}` : text));
      setItems(null); setSelected(new Set()); setResult(null); setError(null);
    };
    reader.readAsText(file);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  async function handlePreview() {
    const pmids = parsePmids(rawInput);
    if (pmids.length === 0) {
      setError("Enter at least one PMID.");
      return;
    }
    setError(null);
    setResult(null);
    setPreviewing(true);
    try {
      const response = await previewPubmedImport(projectId, pmids);
      setItems(response.items);
      const importablePmids = response.items
        .filter((item) => isImportable(item.status))
        .map((item) => item.pmid);
      setSelected(new Set(importablePmids));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Preview failed.");
      setItems(null);
    } finally {
      setPreviewing(false);
    }
  }

  function selectAllImportable() {
    if (!items) {
      return;
    }
    setSelected(new Set(items.filter((item) => isImportable(item.status)).map((item) => item.pmid)));
  }

  function clearSelection() {
    setSelected(new Set());
  }

  function toggleRow(pmid: string) {
    setSelected((previous) => {
      const next = new Set(previous);
      if (next.has(pmid)) {
        next.delete(pmid);
      } else {
        next.add(pmid);
      }
      return next;
    });
  }

  function toggleAbstractOnly(checked: boolean) {
    if (!items) {
      return;
    }
    const abstractPmids = items.filter((item) => item.status === "abstract_only").map((item) => item.pmid);
    setSelected((previous) => {
      const next = new Set(previous);
      for (const pmid of abstractPmids) {
        if (checked) {
          next.add(pmid);
        } else {
          next.delete(pmid);
        }
      }
      return next;
    });
  }

  async function handleImport() {
    if (!items) {
      return;
    }
    const pmids = items.filter((item) => selected.has(item.pmid)).map((item) => item.pmid);
    if (pmids.length === 0) {
      setError("Select at least one document to import.");
      return;
    }
    const includeAbstractOnly = items.some(
      (item) => selected.has(item.pmid) && item.status === "abstract_only",
    );
    setError(null);
    setImporting(true);
    try {
      const response = await importPubmed(projectId, pmids, includeAbstractOnly);
      setResult(response);
      if (response.created.length > 0 || response.skipped.some((outcome) => typeof outcome.document_id === "number")) {
        setItems(null);
        setSelected(new Set());
        setRawInput("");
      }
      try {
        await onImported(response);
      } catch (refreshError) {
        setError(
          refreshError instanceof Error
            ? `Import completed, but refresh failed: ${refreshError.message}`
            : "Import completed, but refresh failed.",
        );
      }
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Import failed.");
    } finally {
      setImporting(false);
    }
  }

  const selectableCount = items?.filter((item) => isImportable(item.status)).length ?? 0;
  const createdCount = result?.created.length ?? 0;
  const reusedCount = result?.skipped.filter((outcome) => typeof outcome.document_id === "number").length ?? 0;
  const unavailable = result?.skipped.filter((outcome) => typeof outcome.document_id !== "number") ?? [];
  const readyCount = createdCount + reusedCount;
  const selectedDocumentLabel =
    selected.size === 1
      ? "Import 1 selected document"
      : `Import ${selected.size} selected documents`;

  return (
    <section ref={sectionRef} className="manager-panel import-panel">
      <div className="section-heading">
        <h2>Import PubMed documents</h2>
        <span>PubMed / PMC</span>
      </div>

      <p className="import-hint">
        {isPersonal
          ? "Paste PMIDs or upload a .txt/.csv file. Full text is preferred when PMC has it; abstracts are available as a fallback."
          : "Paste PMIDs (separated by spaces, commas, or new lines) or upload a .txt/.csv file. We fetch full text from PMC when available and fall back to the PubMed abstract."}
      </p>

      <label htmlFor={`pubmed-pmids-${projectId}`}>PMIDs</label>
      <textarea
        id={`pubmed-pmids-${projectId}`}
        ref={textareaRef}
        className="import-textarea"
        aria-label="PMIDs"
        placeholder="e.g., 31452104, 29622564, 33301246…"
        value={rawInput}
        disabled={previewing || importing}
        onChange={(event) => { setRawInput(event.target.value); setItems(null); setSelected(new Set()); setResult(null); setError(null); }}
        rows={3}
      />
      {invalidTokens.length ? <p role="alert" className="import-error">Enter numeric PMIDs only, separated by spaces, commas, or new lines. Remove invalid entries before previewing.</p> : null}

      <div className="import-actions">
        <label className="import-file-button">
          Upload PMID file
          <input
            ref={fileInputRef}
            className="visually-hidden import-file-input"
            type="file"
            disabled={previewing || importing}
            accept=".txt,.csv,text/plain,text/csv"
            onChange={handleFile}
          />
        </label>
        <span className="import-count" aria-live="polite">
          {parsedCount} PMID(s) detected
        </span>
        <button
          className="import-preview-button"
          type="button"
          onClick={() => void handlePreview()}
          disabled={previewing || parsedCount === 0 || invalidTokens.length > 0}
        >
          {previewing ? "Checking documents…" : "Preview documents"}
        </button>
      </div>

      {error ? (
        <p className="import-error" role="alert">
          {error}
        </p>
      ) : null}

      {items ? (
        <div className="import-preview">
          <div className="import-preview-heading">
            <strong>Choose documents to import</strong>
            <span aria-live="polite">
              {selected.size} of {selectableCount} selected
            </span>
          </div>
          <div className="import-preview-summary" aria-label="Import preview summary">
            <span>
              <strong>{previewCounts.fullText}</strong> full text
            </span>
            <span>
              <strong>{previewCounts.abstractOnly}</strong> abstract only
            </span>
            <span>
              <strong>{previewCounts.unavailable}</strong> unavailable
            </span>
          </div>
          <table className="import-table">
            <thead>
              <tr>
                <th aria-label="Select" />
                <th>PMID</th>
                <th>Title</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.pmid} className={item.status}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.has(item.pmid)}
                      disabled={!isImportable(item.status)}
                      onChange={() => toggleRow(item.pmid)}
                      aria-label={`Select PMID ${item.pmid}`}
                    />
                  </td>
                  <td>{item.pmid}</td>
                  <td title={item.title || undefined}>{item.title || <em>—</em>}</td>
                  <td>
                    <span className={`status-badge status-${item.status}`}>
                      {STATUS_LABEL[item.status]}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="import-footer">
            <label className="import-toggle">
              <input
                type="checkbox"
                checked={items
                  .filter((item) => item.status === "abstract_only")
                  .every((item) => selected.has(item.pmid)) &&
                  items.some((item) => item.status === "abstract_only")}
                disabled={!items.some((item) => item.status === "abstract_only")}
                onChange={(event) => toggleAbstractOnly(event.target.checked)}
              />
              Include abstract-only documents
            </label>
            <div className="import-selection-actions">
              <button type="button" onClick={selectAllImportable} disabled={selectableCount === 0}>
                Select all
              </button>
              <button type="button" onClick={clearSelection} disabled={selected.size === 0}>
                Clear
              </button>
            </div>
            <button
              className="import-submit-button"
              type="button"
              onClick={() => void handleImport()}
              disabled={importing || selected.size === 0}
            >
              {importing ? "Importing selected documents…" : selectedDocumentLabel}
            </button>
          </div>
        </div>
      ) : null}

      {result ? (
        <div role="status" className={readyCount > 0 ? "import-result success" : "import-result warning"}>
          <p>
            {createdCount > 0 ? (
              <>
                Imported <strong>{createdCount}</strong> document(s).
              </>
            ) : reusedCount === 0 ? (
              "No documents imported."
            ) : null}
            {reusedCount > 0 ? ` Reused ${reusedCount} document(s) already in this project.` : ""}
            {unavailable.length > 0 ? ` ${unavailable.length} document(s) could not be imported.` : ""}
          </p>
          {readyCount === 0 && unavailable.length > 0 ? (
            <p className="import-result-detail">
              Nothing was created. Review the skipped PMIDs below, adjust the selection, and try again.
            </p>
          ) : null}
          {readyCount > 0 && isPersonal && onOpenWork ? (
            <button className="import-next-button" type="button" onClick={onOpenWork}>
              Open My Work
            </button>
          ) : null}
          {unavailable.length > 0 ? (
            <details>
            <summary>View {unavailable.length} unavailable PMID(s)</summary>
            <ul className="import-skipped">
              {unavailable.map((outcome) => (
                <li key={outcome.pmid}>
                  PMID {outcome.pmid}: {outcome.reason ?? "skipped"}
                </li>
              ))}
            </ul>
            </details>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

export default forwardRef(PubmedImportPanel);
