import { NativeInput, NativeSelect, NativeTextarea } from "@/components/NativeField";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import {
  adjudicateEvidence,
  getEvidenceAdjudication,
  getDocumentStructure,
} from "@/api/client";
import type {
  DocumentSummary,
  EvidenceAdjudicationRead,
  EvidenceAdjudicationCreate,
  TaskAssignment,
} from "@/types/api";

import { PlatformEmpty, PlatformSection, PlatformStatus } from "./components";

type Strategy = EvidenceAdjudicationCreate["strategy"];

interface AdjudicationScope {
  documentId: number;
  targetVersionId: number;
  structureVersionId: number;
  guidelineVersionId: number;
}

function scopeKey(scope: AdjudicationScope): string {
  return [
    scope.documentId,
    scope.targetVersionId,
    scope.structureVersionId,
    scope.guidelineVersionId,
  ].join(":");
}

function documentName(documents: DocumentSummary[], documentId: number): string {
  const document = documents.find((item) => item.id === documentId);
  return (
    document?.title ??
    document?.external_id ??
    `Document ${documentId}`
  );
}

export default function EvidenceAdjudicationPanel({
  projectId,
  documents,
  assignments,
  allowSoloGold,
  onSetup,
}: {
  projectId: number;
  documents: DocumentSummary[];
  assignments: TaskAssignment[];
  allowSoloGold: boolean;
  onSetup?: () => void;
}): React.ReactElement {
  const scopes = useMemo(() => {
    const unique = new Map<string, AdjudicationScope>();
    for (const assignment of assignments) {
      if (
        assignment.target_version_id === null ||
        assignment.structure_version_id === null ||
        assignment.guideline_version_id === null
      ) {
        continue;
      }
      const scope = {
        documentId: assignment.document_id,
        targetVersionId: assignment.target_version_id,
        structureVersionId: assignment.structure_version_id,
        guidelineVersionId: assignment.guideline_version_id,
      };
      unique.set(scopeKey(scope), scope);
    }
    return [...unique.values()].sort(
      (left, right) =>
        left.documentId - right.documentId ||
        left.targetVersionId - right.targetVersionId,
    );
  }, [assignments]);
  const [selectedScopeKey, setSelectedScopeKey] = useState("");
  const [comparison, setComparison] =
    useState<EvidenceAdjudicationRead | null>(null);
  const [selectedAnnotationIds, setSelectedAnnotationIds] = useState<number[]>(
    [],
  );
  const [strategy, setStrategy] = useState<Strategy>(
    allowSoloGold ? "a" : "union",
  );
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [note, setNote] = useState("");
  const [goldLabels, setGoldLabels] = useState("");
  const [busy, setBusy] = useState(false);
  const [rangeError, setRangeError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const comparisonRequest = useRef(0);
  const activeScope = useRef("");
  const selectedScope =
    scopes.find((scope) => scopeKey(scope) === selectedScopeKey) ?? null;
  activeScope.current = `${projectId}:${selectedScope ? scopeKey(selectedScope) : ""}`;

  useEffect(() => {
    setSelectedScopeKey((current) =>
      scopes.some((scope) => scopeKey(scope) === current)
        ? current
        : scopes[0]
          ? scopeKey(scopes[0])
          : "",
    );
  }, [scopes]);

  useEffect(() => {
    comparisonRequest.current += 1;
    setBusy(false);
    setComparison(null);
    setSelectedAnnotationIds([]);
    setGoldLabels("");
    setError(null);
    setRangeError(null);
    setStatus(null);
    return () => { comparisonRequest.current += 1; };
  }, [projectId, selectedScopeKey]);

  async function loadComparison(): Promise<void> {
    if (!selectedScope) return;
    const request = ++comparisonRequest.current;
    const scope = activeScope.current;
    const isCurrent = (): boolean => request === comparisonRequest.current && scope === activeScope.current;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const result = await getEvidenceAdjudication(
        projectId,
        selectedScope.documentId,
        selectedScope.targetVersionId,
        selectedScope.structureVersionId,
        selectedScope.guidelineVersionId,
      );
      if (!isCurrent()) return;
      setComparison(result);
      setSelectedAnnotationIds((current) =>
        current.filter((id) =>
          result.blocks.some(
            (block) => block.annotation_id === id && block.status !== "gold",
          ),
        ),
      );
    } catch (caught) {
      if (!isCurrent()) return;
      setComparison(null);
      setError(
        caught instanceof Error
          ? caught.message
          : "The evidence comparison could not be loaded.",
      );
    } finally {
      if (isCurrent()) setBusy(false);
    }
  }

  function toggleSource(annotationId: number): void {
    if (!selectedAnnotationIds.includes(annotationId)) {
      const block = comparison?.blocks.find((item) => item.annotation_id === annotationId);
      if (block) setGoldLabels([...new Set([...goldLabels.split(",").map((label) => label.trim()).filter(Boolean), ...block.labels])].join(", "));
    }
    setSelectedAnnotationIds((current) =>
      current.includes(annotationId)
        ? current.filter((id) => id !== annotationId)
        : [...current, annotationId],
    );
  }

  async function createGold(): Promise<void> {
    if (!selectedScope || !comparison) {
      setError("Load a comparison before creating gold evidence.");
      return;
    }
    const operationScope = activeScope.current;
    const isCurrentScope = (): boolean => operationScope === activeScope.current;
    const effectiveStrategy = strategy;
    const selectedBlocks = comparison.blocks.filter((block) =>
      selectedAnnotationIds.includes(block.annotation_id),
    );
    const sourceAnnotators = new Set(
      selectedBlocks
        .map((block) => block.annotator_user_id)
        .filter((id): id is number => id !== null),
    );
    if (!allowSoloGold && sourceAnnotators.size < 2) {
      setError(
        "Select reviewed evidence from at least two distinct annotators.",
      );
      return;
    }
    if (
      effectiveStrategy === "a" && selectedAnnotationIds.length === 0
    ) {
      setError("Select a source block to retain its boundaries and labels.");
      return;
    }
    if (
      ["b", "union", "intersection"].includes(effectiveStrategy) &&
      selectedAnnotationIds.length < 2
    ) {
      setError("This boundary strategy requires two source blocks.");
      return;
    }
    const startOrdinal = Number(customStart);
    const endOrdinal = Number(customEnd);
    if (
      effectiveStrategy === "custom" &&
      (
        customStart.trim() === "" ||
        customEnd.trim() === "" ||
        !Number.isSafeInteger(startOrdinal) ||
        !Number.isSafeInteger(endOrdinal) ||
        startOrdinal < 0 ||
        endOrdinal < startOrdinal
      )
    ) {
      setRangeError("Enter valid start and end sentence ordinals in ascending order.");
      const invalidStart = customStart.trim() === "" || !Number.isSafeInteger(startOrdinal) || startOrdinal < 0;
      document.getElementById(invalidStart ? "gold-start-ordinal" : "gold-end-ordinal")?.focus();
      return;
    }

    setBusy(true);
    setError(null);
    setRangeError(null);
    setStatus(null);
    try {
      let startSentenceId: number | null = null;
      let endSentenceId: number | null = null;
      if (effectiveStrategy === "custom") {
        const [startPage, endPage] = await Promise.all([
          getDocumentStructure(selectedScope.documentId, {
            versionId: selectedScope.structureVersionId,
            sentenceStart: startOrdinal,
            sentenceLimit: 1,
          }),
          getDocumentStructure(selectedScope.documentId, {
            versionId: selectedScope.structureVersionId,
            sentenceStart: endOrdinal,
            sentenceLimit: 1,
          }),
        ]);
        const start = startPage.sentences.find((sentence) => sentence.ordinal === startOrdinal);
        const end = endPage.sentences.find((sentence) => sentence.ordinal === endOrdinal);
        if (!start || !end) {
          throw new Error("Choose sentence ordinals within this document's pinned structure.");
        }
        startSentenceId = start.id;
        endSentenceId = end.id;
      }
      if (!isCurrentScope()) return;
      const created = await adjudicateEvidence(
        projectId,
        selectedScope.documentId,
        {
          target_version_id: selectedScope.targetVersionId,
          structure_version_id: selectedScope.structureVersionId,
          guideline_version_id: selectedScope.guidelineVersionId,
          strategy: effectiveStrategy,
          source_annotation_ids: selectedAnnotationIds,
          ...(effectiveStrategy === "custom" ? { labels: goldLabels.split(",").map((label) => label.trim()).filter(Boolean) } : {}),
          start_sentence_id:
            effectiveStrategy === "custom" ? startSentenceId : null,
          end_sentence_id:
            effectiveStrategy === "custom" ? endSentenceId : null,
          note: note.trim() || null,
          solo_gold: allowSoloGold,
        },
      );
      if (!isCurrentScope()) return;
      setStatus(`Gold evidence block #${created.id} created`);
      setSelectedAnnotationIds([]);
      setGoldLabels("");
      setNote("");
      await loadComparison();
      if (isCurrentScope()) setStatus(`Gold evidence block #${created.id} created`);
    } catch (caught) {
      if (isCurrentScope()) setError(
        caught instanceof Error
          ? caught.message
          : "Gold evidence could not be created.",
      );
    } finally {
      if (isCurrentScope()) setBusy(false);
    }
  }

  return (
    <PlatformSection
      title="Evidence adjudication"
      description="Compare reviewed evidence within one pinned document, target, structure, and guideline scope."
    >
      {scopes.length === 0 ? (
        <PlatformEmpty
          title="No pinned evidence scope"
          detail="Evidence adjudication becomes available after assignments pin target, structure, and guideline versions."
          actionLabel={onSetup ? "Set up evidence annotation" : undefined}
          onAction={onSetup}
        />
      ) : (
        <div className="platform-adjudication">
          <div className="platform-form-grid">
            <label>
              <span>Review scope</span>
              <NativeSelect
                value={selectedScopeKey}
                disabled={busy}
                onChange={(event) => setSelectedScopeKey(event.target.value)}
              >
                {scopes.map((scope) => (
                  <option key={scopeKey(scope)} value={scopeKey(scope)}>
                    {`${documentName(documents, scope.documentId)} · target ID ${scope.targetVersionId} · guideline ID ${scope.guidelineVersionId}`}
                  </option>
                ))}
              </NativeSelect>
            </label>
            <div className="platform-field-action">
              <Button
                label={busy ? "Loading…" : "Load comparison"}
                isDisabled={busy}
                onClick={() => void loadComparison()}
              />
            </div>
          </div>

          {error ? (
            <p className="platform-dialog-error" role="alert">{error}</p>
          ) : null}
          {status ? (
            <p className="platform-form-success" role="status">{status}</p>
          ) : null}

          {comparison ? (
            <>
              <div
                className="platform-table-scroll platform-table-scroll--summary"
                role="region"
                aria-label="Evidence comparison"
                tabIndex={0}
              >
                <table className="platform-table platform-table--summary">
                  <thead>
                    <tr>
                      <th scope="col">Use</th>
                      <th scope="col">Annotator</th>
                      <th scope="col">Boundary</th>
                      <th scope="col">Status</th>
                      <th scope="col">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparison.blocks.map((block) => {
                      const isGold = block.status === "gold";
                      return (
                        <tr key={block.annotation_id}>
                          <td data-label="Use">
                            <NativeInput
                              type="checkbox"
                              aria-label={`Use evidence block ${block.annotation_id}`}
                              checked={selectedAnnotationIds.includes(
                                block.annotation_id,
                              )}
                              disabled={busy || isGold}
                              onChange={() =>
                                toggleSource(block.annotation_id)
                              }
                            />
                          </td>
                          <td data-label="Annotator" data-priority="identity">
                            <strong>
                              {block.annotator_id ??
                                `User ${block.annotator_user_id ?? "unknown"}`}
                            </strong>
                            <span>Block {block.annotation_id}</span>
                          </td>
                          <td data-label="Boundary">
                            Sentences {block.start_sentence_ordinal}-
                            {block.end_sentence_ordinal}
                          </td>
                          <td data-label="Status" data-priority="status">
                            <PlatformStatus value={block.status} />
                          </td>
                          <td data-label="Note">{block.note ?? "No note"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              <div className="platform-form-grid">
                <label>
                  <span>Gold boundary</span>
                  <NativeSelect
                    value={strategy}
                    disabled={busy}
                    onChange={(event) =>
                      setStrategy(event.target.value as Strategy)
                    }
                  >
                    <option value="a">Source A</option>
                    <option value="b">Source B</option>
                    <option value="union">Union</option>
                    <option value="intersection">Intersection</option>
                    <option value="custom">Custom</option>
                  </NativeSelect>
                </label>
                {strategy === "custom" ? (
                  <>
                    <label><span>Gold labels</span><NativeInput value={goldLabels} disabled={busy} placeholder="Comma-separated labels" onChange={(event) => setGoldLabels(event.target.value)} /></label>
                    <label>
                      <span>Start sentence ordinal</span>
                      <NativeInput
                        inputMode="numeric"
                        id="gold-start-ordinal"
                        aria-invalid={Boolean(rangeError)}
                        aria-describedby={rangeError ? "gold-range-error" : undefined}
                        value={customStart}
                        onChange={(event) => { setCustomStart(event.target.value); setRangeError(null); }}
                      />
                    </label>
                    <label>
                      <span>End sentence ordinal</span>
                      <NativeInput
                        inputMode="numeric"
                        id="gold-end-ordinal"
                        aria-invalid={Boolean(rangeError)}
                        aria-describedby={rangeError ? "gold-range-error" : undefined}
                        value={customEnd}
                        onChange={(event) => { setCustomEnd(event.target.value); setRangeError(null); }}
                      />
                    </label>
                    {rangeError ? <p id="gold-range-error" className="platform-dialog-error platform-form-wide" role="alert">{rangeError}</p> : null}
                  </>
                ) : null}
                <label className="platform-form-wide">
                  <span>Gold note</span>
                  <NativeTextarea
                    rows={3}
                    value={note}
                    onChange={(event) => setNote(event.target.value)}
                  />
                </label>
              </div>
              <div className="platform-settings-actions">
                <Button
                  label={busy ? "Saving…" : "Create gold block"}
                  variant="primary"
                  isDisabled={busy}
                  onClick={() => void createGold()}
                />
              </div>
            </>
          ) : (
            <p className="platform-inline-empty">
              Load a comparison to review source boundaries and create an
              explicit gold decision.
            </p>
          )}
        </div>
      )}
    </PlatformSection>
  );
}
