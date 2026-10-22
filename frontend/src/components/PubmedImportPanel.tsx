import { NativeInput, NativeTextarea } from "@/components/NativeField";
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ForwardedRef } from "react";
import { Button } from "@astryxdesign/core/Button";

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

const PMID_BATCH_SIZE = 100;
type ImportOperation = { kind: "file" | "preview" | "import"; completed: number; total: number };

function pmidsFromCsv(text: string): string {
  const rows: string[][] = [];
  let row: string[] = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') { value += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) {
      row.push(value.trim()); value = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      row.push(value.trim());
      if (row.some(Boolean)) rows.push(row);
      row = []; value = "";
      if (character === "\r" && text[index + 1] === "\n") index += 1;
    } else value += character;
  }
  if (quoted) throw new Error("The CSV contains an unclosed quoted field.");
  row.push(value.trim());
  if (row.some(Boolean)) rows.push(row);
  const first = rows[0];
  if (!first) return "";
  const namedColumn = first.findIndex((heading) => /^(?:pmids?|pubmed[\s_-]*ids?)$/i.test(heading));
  const column = namedColumn < 0 ? 0 : namedColumn;
  const hasHeader = namedColumn >= 0 || !/^(?:PMID:?\s*)?\d+$/i.test(first[column] ?? "");
  return rows.slice(hasHeader ? 1 : 0).map((cells) => cells[column] ?? "").join("\n");
}

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
  const [operation, setOperation] = useState<ImportOperation | null>(null);
  const operationRef = useRef<ImportOperation | null>(null);
  const previewing = operation?.kind === "preview";
  const importing = operation?.kind === "import";
  const working = operation !== null;
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ImportResponse | null>(null);
  const sectionRef = useRef<HTMLElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setRawInput(""); setItems(null); setSelected(new Set()); setResult(null); setError(null);
    setOperation(null);
    return () => { operationRef.current = null; };
  }, [projectId]);

  function startOperation(kind: ImportOperation["kind"], total: number): ImportOperation | null {
    if (operationRef.current !== null) return null;
    const next = { kind, completed: 0, total };
    operationRef.current = next;
    setOperation(next);
    return next;
  }

  function finishOperation(current: ImportOperation) {
    if (operationRef.current !== current) return;
    operationRef.current = null;
    setOperation(null);
  }

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
    const current = startOperation("file", 0);
    if (!current) return;
    const reader = new FileReader();
    reader.onload = () => {
      if (operationRef.current !== current) return;
      const text = typeof reader.result === "string" ? reader.result : "";
      try {
        const identifiers = /\.csv$/i.test(file.name) || file.type === "text/csv"
          ? pmidsFromCsv(text.replace(/^\uFEFF/, "")) : text;
        setRawInput((previous) => previous ? `${previous}\n${identifiers}` : identifiers);
        setItems(null); setSelected(new Set()); setResult(null); setError(null);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "The PMID file could not be read.");
      } finally { finishOperation(current); }
    };
    reader.onerror = () => {
      if (operationRef.current !== current) return;
      setError("The PMID file could not be read."); finishOperation(current);
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
    const current = startOperation("preview", pmids.length);
    if (!current) return;
    setError(null);
    setResult(null);
    try {
      const previewItems: ImportPreviewItem[] = [];
      for (let start = 0; start < pmids.length; start += PMID_BATCH_SIZE) {
        if (operationRef.current !== current) return;
        const batch = pmids.slice(start, start + PMID_BATCH_SIZE);
        const response = await previewPubmedImport(projectId, batch);
        if (operationRef.current !== current) return;
        previewItems.push(...response.items);
        setOperation({ ...current, completed: start + batch.length });
      }
      setItems(previewItems);
      const importablePmids = previewItems
        .filter((item) => isImportable(item.status))
        .map((item) => item.pmid);
      setSelected(new Set(importablePmids));
    } catch (caught) {
      if (operationRef.current !== current) return;
      setError(caught instanceof ApiError ? caught.message : "Preview failed.");
      setItems(null);
    } finally {
      finishOperation(current);
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
    const current = startOperation("import", pmids.length);
    if (!current) return;
    const response: ImportResponse = { created: [], skipped: [] };
    let processed = 0;
    setError(null);
    try {
      try {
        for (let start = 0; start < pmids.length; start += PMID_BATCH_SIZE) {
          if (operationRef.current !== current) return;
          const batch = pmids.slice(start, start + PMID_BATCH_SIZE);
          const imported = await importPubmed(projectId, batch, includeAbstractOnly);
          if (operationRef.current !== current) return;
          response.created.push(...imported.created);
          response.skipped.push(...imported.skipped);
          processed += batch.length;
          setResult({ created: [...response.created], skipped: [...response.skipped] });
          setOperation({ ...current, completed: processed });
        }
      } catch (caught) {
        if (operationRef.current !== current) return;
        setError(`Import stopped after ${processed} of ${pmids.length} PMIDs. ${caught instanceof Error ? caught.message : "Import failed."}`);
      }
      if (operationRef.current !== current) return;
      const saved = new Set([...response.created, ...response.skipped]
        .filter((outcome) => typeof outcome.document_id === "number").map((outcome) => outcome.pmid));
      if (saved.size > 0) {
        if (processed === pmids.length) {
          setItems(null); setSelected(new Set()); setRawInput("");
        } else {
          setItems(items.filter((item) => !saved.has(item.pmid)));
          setSelected(new Set(pmids.filter((pmid) => !saved.has(pmid))));
          setRawInput(parsePmids(rawInput).filter((pmid) => !saved.has(pmid)).join("\n"));
        }
      }
      if (processed === 0) return;
      try {
        await onImported(response);
      } catch (refreshError) {
        if (operationRef.current !== current) return;
        setError(
          refreshError instanceof Error
            ? `Import completed, but refresh failed: ${refreshError.message}`
            : "Import completed, but refresh failed.",
        );
      }
    } finally {
      finishOperation(current);
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
    <section ref={sectionRef} className="import-panel" aria-label="PubMed import">
      <div className="section-heading">
        <h3>Add papers from PubMed</h3>
        <span>PubMed / PMC</span>
      </div>

      <p className="import-hint">
        {isPersonal
          ? "Paste PMIDs or upload a .txt/.csv file. Full text is preferred when PMC has it; abstracts are available as a fallback."
          : "Paste PMIDs (separated by spaces, commas, or new lines) or upload a .txt/.csv file. We fetch full text from PMC when available and fall back to the PubMed abstract."}
      </p>

      <label htmlFor={`pubmed-pmids-${projectId}`}>PMIDs</label>
      <NativeTextarea
        id={`pubmed-pmids-${projectId}`}
        ref={textareaRef}
        className="import-textarea"
        aria-label="PMIDs"
        placeholder="e.g., 31452104, 29622564, 33301246…"
        value={rawInput}
        disabled={working}
        aria-invalid={invalidTokens.length > 0 || undefined}
        aria-describedby={invalidTokens.length ? `pubmed-error-${projectId}` : undefined}
        onChange={(event) => { setRawInput(event.target.value); setItems(null); setSelected(new Set()); setResult(null); setError(null); }}
        rows={3}
      />
      {invalidTokens.length ? <p id={`pubmed-error-${projectId}`} role="alert" className="import-error">Enter numeric PMIDs only, separated by spaces, commas, or new lines. Remove invalid entries before previewing.</p> : null}

      <div className="import-actions">
        <label className="import-file-button">
          Upload PMID file
          <NativeInput
            ref={fileInputRef}
            className="sr-only import-file-input"
            type="file"
            disabled={working}
            accept=".txt,.csv,text/plain,text/csv"
            onChange={handleFile}
          />
        </label>
        <span className="import-count" aria-live="polite">
          {parsedCount.toLocaleString()} {parsedCount === 1 ? "PMID" : "PMIDs"} detected
        </span>
        <Button
          label={previewing ? "Checking documents…" : "Preview documents"}
          className={items ? undefined : "import-preview-button"}
          type="button"
          variant={items ? "secondary" : "primary"}
          onClick={() => void handlePreview()}
          isDisabled={working || parsedCount === 0 || invalidTokens.length > 0}
          isLoading={previewing}
        />
      </div>

      {operation && operation.kind !== "file" ? (
        <p role="status" aria-live="polite">
          {importing ? "Importing" : "Checking"} documents, {operation.completed} of {operation.total} PMIDs.
        </p>
      ) : null}

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
                    <NativeInput
                      type="checkbox"
                      checked={selected.has(item.pmid)}
                      disabled={working || !isImportable(item.status)}
                      onChange={() => toggleRow(item.pmid)}
                      aria-label={`Select PMID ${item.pmid}`}
                    />
                  </td>
                  <td>{item.pmid}</td>
                  <td>{item.title || <em>No title available</em>}</td>
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
              <NativeInput
                type="checkbox"
                checked={items
                  .filter((item) => item.status === "abstract_only")
                  .every((item) => selected.has(item.pmid)) &&
                  items.some((item) => item.status === "abstract_only")}
                disabled={working || !items.some((item) => item.status === "abstract_only")}
                onChange={(event) => toggleAbstractOnly(event.target.checked)}
              />
              Include abstract-only documents
            </label>
            <div className="import-selection-actions">
              <Button label="Select all" variant="secondary" onClick={selectAllImportable} isDisabled={working || selectableCount === 0} />
              <Button label="Clear" variant="secondary" onClick={clearSelection} isDisabled={working || selected.size === 0} />
            </div>
            <Button
              label={importing ? "Importing selected documents…" : selectedDocumentLabel}
              className="import-submit-button"
              type="button"
              variant="primary"
              onClick={() => void handleImport()}
              isDisabled={working || selected.size === 0}
              isLoading={importing}
            />
          </div>
        </div>
      ) : null}

      {result ? (
        <div role="status" aria-label="Import result" className={readyCount > 0 ? "import-result success" : "import-result warning"}>
          <p>
            {createdCount > 0 ? (
              <>
                Imported <strong>{createdCount}</strong> {createdCount === 1 ? "document" : "documents"}.
              </>
            ) : reusedCount === 0 ? (
              "No documents imported."
            ) : null}
            {reusedCount > 0 ? ` Reused ${reusedCount} ${reusedCount === 1 ? "document" : "documents"} already in this project.` : ""}
            {unavailable.length > 0 ? ` ${unavailable.length} ${unavailable.length === 1 ? "document" : "documents"} could not be imported.` : ""}
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
            <summary>View {unavailable.length} unavailable {unavailable.length === 1 ? "PMID" : "PMIDs"}</summary>
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
