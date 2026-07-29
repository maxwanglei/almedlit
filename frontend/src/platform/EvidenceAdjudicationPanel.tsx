import { useEffect, useMemo, useState } from "react";
import { Button } from "@astryxdesign/core/Button";

import {
  adjudicateEvidence,
  getEvidenceAdjudication,
} from "@/api/client";
import type {
  Document,
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

function documentName(documents: Document[], documentId: number): string {
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
}: {
  projectId: number;
  documents: Document[];
  assignments: TaskAssignment[];
  allowSoloGold: boolean;
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
    allowSoloGold ? "custom" : "union",
  );
  const [customStart, setCustomStart] = useState("");
  const [customEnd, setCustomEnd] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const selectedScope =
    scopes.find((scope) => scopeKey(scope) === selectedScopeKey) ?? null;

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
    setComparison(null);
    setSelectedAnnotationIds([]);
    setError(null);
    setStatus(null);
  }, [selectedScopeKey]);

  async function loadComparison(): Promise<void> {
    if (!selectedScope) return;
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
      setComparison(result);
      setSelectedAnnotationIds((current) =>
        current.filter((id) =>
          result.blocks.some(
            (block) => block.annotation_id === id && block.status !== "gold",
          ),
        ),
      );
    } catch (caught) {
      setComparison(null);
      setError(
        caught instanceof Error
          ? caught.message
          : "The evidence comparison could not be loaded.",
      );
    } finally {
      setBusy(false);
    }
  }

  function toggleSource(annotationId: number): void {
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
    const effectiveStrategy: Strategy = allowSoloGold ? "custom" : strategy;
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
      ["b", "union", "intersection"].includes(effectiveStrategy) &&
      selectedAnnotationIds.length < 2
    ) {
      setError("This boundary strategy requires two source blocks.");
      return;
    }
    const startSentenceId = Number(customStart);
    const endSentenceId = Number(customEnd);
    if (
      effectiveStrategy === "custom" &&
      (
        !Number.isInteger(startSentenceId) ||
        !Number.isInteger(endSentenceId)
      )
    ) {
      setError("Enter valid start and end sentence IDs.");
      return;
    }

    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const created = await adjudicateEvidence(
        projectId,
        selectedScope.documentId,
        {
          target_version_id: selectedScope.targetVersionId,
          structure_version_id: selectedScope.structureVersionId,
          guideline_version_id: selectedScope.guidelineVersionId,
          strategy: effectiveStrategy,
          source_annotation_ids: selectedAnnotationIds,
          start_sentence_id:
            effectiveStrategy === "custom" ? startSentenceId : null,
          end_sentence_id:
            effectiveStrategy === "custom" ? endSentenceId : null,
          note: note.trim() || null,
          solo_gold: allowSoloGold,
        },
      );
      setStatus(`Gold evidence block #${created.id} created`);
      setSelectedAnnotationIds([]);
      setNote("");
      await loadComparison();
      setStatus(`Gold evidence block #${created.id} created`);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Gold evidence could not be created.",
      );
    } finally {
      setBusy(false);
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
        />
      ) : (
        <div className="platform-adjudication">
          <div className="platform-form-grid">
            <label>
              <span>Review scope</span>
              <select
                value={selectedScopeKey}
                disabled={busy}
                onChange={(event) => setSelectedScopeKey(event.target.value)}
              >
                {scopes.map((scope) => (
                  <option key={scopeKey(scope)} value={scopeKey(scope)}>
                    {documentName(documents, scope.documentId)} · target v
                    {scope.targetVersionId} · guideline v
                    {scope.guidelineVersionId}
                  </option>
                ))}
              </select>
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
            <p className="platform-form-warning" role="alert">{error}</p>
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
                            <input
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
                  <select
                    value={allowSoloGold ? "custom" : strategy}
                    disabled={busy || allowSoloGold}
                    onChange={(event) =>
                      setStrategy(event.target.value as Strategy)
                    }
                  >
                    <option value="a">Source A</option>
                    <option value="b">Source B</option>
                    <option value="union">Union</option>
                    <option value="intersection">Intersection</option>
                    <option value="custom">Custom</option>
                  </select>
                </label>
                {allowSoloGold || strategy === "custom" ? (
                  <>
                    <label>
                      <span>Start sentence ID</span>
                      <input
                        inputMode="numeric"
                        value={customStart}
                        onChange={(event) => setCustomStart(event.target.value)}
                      />
                    </label>
                    <label>
                      <span>End sentence ID</span>
                      <input
                        inputMode="numeric"
                        value={customEnd}
                        onChange={(event) => setCustomEnd(event.target.value)}
                      />
                    </label>
                  </>
                ) : null}
                <label className="platform-form-wide">
                  <span>Gold note</span>
                  <textarea
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
