import hashlib
import json
from collections.abc import Iterable
from typing import Any

from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage, StoredObject
from al_medlit.corpus.models import Document, DocumentSentence, DocumentStructureVersion
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceReviewCoverage,
    EvidenceTargetVersion,
)
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.lineage.models import (
    AnnotationSet,
    AnnotationSetItem,
    AnnotationSetReviewRegion,
    CorpusSnapshot,
    CorpusSnapshotDocument,
    ExportArtifact,
    LineageArtifact,
    LineageEdge,
)
from al_medlit.lineage.schemas import (
    AnnotationSetCreate,
    AnnotationSetItemCreate,
    AnnotationSetReviewRegionCreate,
    CorpusSnapshotCreate,
)
from al_medlit.project.models import Project, TaskAssignment
from al_medlit.training.windowing import assign_grouped_splits


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _artifact_for_stored_object(
    db: Session,
    *,
    project_id: int,
    artifact_type: str,
    stored: StoredObject,
    manifest: dict,
    created_by_user_id: int | None,
    schema_version: str = "1.0",
) -> LineageArtifact:
    existing = db.query(LineageArtifact).filter(
        LineageArtifact.storage_key == stored.key
    ).first()
    if existing is not None:
        if (
            existing.project_id != project_id
            or existing.artifact_type != artifact_type
            or existing.content_hash != stored.checksum_sha256
            or existing.size_bytes != stored.size_bytes
        ):
            raise ConflictError("Storage key is already registered with different metadata")
        return existing
    artifact = LineageArtifact(
        project_id=project_id,
        artifact_type=artifact_type,
        schema_version=schema_version,
        content_hash=stored.checksum_sha256,
        storage_key=stored.key,
        content_type=stored.content_type,
        size_bytes=stored.size_bytes,
        manifest=manifest,
        created_by_user_id=created_by_user_id,
    )
    db.add(artifact)
    db.flush()
    return artifact


def register_stored_artifact(
    db: Session,
    *,
    project_id: int,
    artifact_type: str,
    stored: StoredObject,
    manifest: dict,
    created_by_user_id: int | None,
    schema_version: str = "1.0",
) -> LineageArtifact:
    """Register bytes already written by a streaming producer."""
    _require_project(db, project_id)
    return _artifact_for_stored_object(
        db,
        project_id=project_id,
        artifact_type=artifact_type,
        stored=stored,
        manifest=manifest,
        created_by_user_id=created_by_user_id,
        schema_version=schema_version,
    )


def _store_manifest(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    artifact_type: str,
    manifest: dict,
    created_by_user_id: int | None,
) -> LineageArtifact:
    payload = canonical_json_bytes(manifest)
    digest = hashlib.sha256(payload).hexdigest()
    existing = (
        db.query(LineageArtifact)
        .filter(
            LineageArtifact.project_id == project_id,
            LineageArtifact.artifact_type == artifact_type,
            LineageArtifact.content_hash == digest,
        )
        .first()
    )
    if existing is not None:
        return existing
    key = f"projects/{project_id}/lineage/{artifact_type}/{digest}.json"
    stored = storage.put_stream(
        key,
        _BytesReader(payload),
        length=len(payload),
        content_type="application/json",
    )
    return _artifact_for_stored_object(
        db,
        project_id=project_id,
        artifact_type=artifact_type,
        stored=stored,
        manifest=manifest,
        created_by_user_id=created_by_user_id,
    )


def freeze_corpus_snapshot(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: CorpusSnapshotCreate,
    actor_user_id: int | None,
) -> CorpusSnapshot:
    _require_project(db, project_id)
    if data.split_strategy != "document_grouped_80_10_10":
        raise ValidationError(
            "Only the deterministic document_grouped_80_10_10 split strategy is supported"
        )
    document_groups = {item.document_id: item.group_key for item in data.documents}
    assigned_splits = assign_grouped_splits(document_groups, seed=data.split_seed)
    manifest_documents: list[dict] = []
    for item in sorted(data.documents, key=lambda value: value.document_id):
        document = db.get(Document, item.document_id)
        if document is None or document.project_id != project_id:
            raise ValidationError(f"Document {item.document_id} is not in project")
        structure = db.get(DocumentStructureVersion, item.structure_version_id)
        if structure is None or structure.document_id != document.id:
            raise ValidationError(
                f"Structure version {item.structure_version_id} does not belong to "
                f"document {document.id}"
            )
        if structure.status != "ready":
            raise ValidationError(f"Structure version {structure.id} is not ready")
        if structure.source_hash != item.source_hash:
            raise ConflictError(f"Source hash does not match structure version {structure.id}")
        manifest_documents.append(
            {
                **item.model_dump(),
                "split": assigned_splits[item.document_id],
            }
        )

    manifest = {
        "schema_version": "corpus-snapshot-v1",
        "project_id": project_id,
        "name": data.name,
        "split_strategy": data.split_strategy,
        "split_seed": data.split_seed,
        "documents": manifest_documents,
        "metadata": data.metadata_,
    }
    artifact = _store_manifest(
        db,
        storage,
        project_id=project_id,
        artifact_type="corpus_snapshot",
        manifest=manifest,
        created_by_user_id=actor_user_id,
    )
    existing = db.query(CorpusSnapshot).filter(CorpusSnapshot.artifact_id == artifact.id).first()
    if existing is not None:
        return existing

    snapshot = CorpusSnapshot(
        project_id=project_id,
        artifact_id=artifact.id,
        name=data.name,
        split_strategy=data.split_strategy,
        split_seed=data.split_seed,
        document_count=len(data.documents),
        metadata_=data.metadata_,
    )
    db.add(snapshot)
    db.flush()
    for item in manifest_documents:
        db.add(CorpusSnapshotDocument(snapshot_id=snapshot.id, **item))
    db.commit()
    db.refresh(snapshot)
    return snapshot


_FROZEN_ASSIGNMENT_STATUSES = {
    "submitted",
    "adjudication_ready",
    "adjudicated",
    "completed",
}


def _derive_frozen_annotation_contents(
    db: Session,
    *,
    project_id: int,
    snapshot_documents: dict[int, CorpusSnapshotDocument],
    requested_targets: set[int],
) -> tuple[list[AnnotationSetItemCreate], list[AnnotationSetReviewRegionCreate]]:
    reviewed_by_scope: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    regions: list[AnnotationSetReviewRegionCreate] = []

    for snapshot_document in snapshot_documents.values():
        for target_version_id in sorted(requested_targets):
            assignments = (
                db.query(TaskAssignment)
                .filter(
                    TaskAssignment.project_id == project_id,
                    TaskAssignment.document_id == snapshot_document.document_id,
                    TaskAssignment.structure_version_id
                    == snapshot_document.structure_version_id,
                    TaskAssignment.target_version_id == target_version_id,
                    TaskAssignment.status.in_(_FROZEN_ASSIGNMENT_STATUSES),
                )
                .all()
            )
            reviewer_ids = sorted({assignment.assignee_user_id for assignment in assignments})
            if not reviewer_ids:
                continue
            reviewer_intervals: list[list[tuple[int, int]]] = []
            for reviewer_id in reviewer_ids:
                rows = (
                    db.query(EvidenceReviewCoverage)
                    .filter(
                        EvidenceReviewCoverage.project_id == project_id,
                        EvidenceReviewCoverage.document_id
                        == snapshot_document.document_id,
                        EvidenceReviewCoverage.structure_version_id
                        == snapshot_document.structure_version_id,
                        EvidenceReviewCoverage.target_version_id == target_version_id,
                        EvidenceReviewCoverage.reviewer_user_id == reviewer_id,
                    )
                    .order_by(EvidenceReviewCoverage.start_sentence_ordinal)
                    .all()
                )
                reviewer_intervals.append(
                    [
                        (row.start_sentence_ordinal, row.end_sentence_ordinal)
                        for row in rows
                    ]
                )
            effective = _intersect_interval_sets(reviewer_intervals)
            scope = (
                snapshot_document.document_id,
                snapshot_document.structure_version_id,
                target_version_id,
            )
            reviewed_by_scope[scope] = effective
            regions.extend(
                AnnotationSetReviewRegionCreate(
                    document_id=snapshot_document.document_id,
                    structure_version_id=snapshot_document.structure_version_id,
                    target_version_id=target_version_id,
                    start_sentence_ordinal=start,
                    end_sentence_ordinal=end,
                    metadata_={
                        "coverage_policy": "intersection_of_submitted_reviewers",
                        "reviewer_user_ids": reviewer_ids,
                    },
                )
                for start, end in effective
            )

    items: list[AnnotationSetItemCreate] = []
    gold_annotations = (
        db.query(Annotation)
        .join(EvidenceBlockAnnotation)
        .filter(
            Annotation.project_id == project_id,
            Annotation.annotation_type == "evidence_block",
            Annotation.status == "gold",
            Annotation.document_id.in_(snapshot_documents),
            EvidenceBlockAnnotation.target_version_id.in_(requested_targets),
        )
        .order_by(
            Annotation.document_id,
            EvidenceBlockAnnotation.target_version_id,
            EvidenceBlockAnnotation.start_sentence_ordinal,
            Annotation.id,
        )
        .all()
    )
    for annotation in gold_annotations:
        block = annotation.evidence_block
        snapshot_document = snapshot_documents[annotation.document_id]
        if block.structure_version_id != snapshot_document.structure_version_id:
            continue
        adjudication = (annotation.attributes or {}).get("adjudication")
        if not isinstance(adjudication, dict):
            # Preserve legacy gold rows as history, but never train from them.
            continue
        scope = (
            annotation.document_id,
            block.structure_version_id,
            block.target_version_id,
        )
        effective_intervals = reviewed_by_scope.get(scope, [])
        if not any(
            start <= block.start_sentence_ordinal
            and end >= block.end_sentence_ordinal
            for start, end in effective_intervals
        ):
            raise ValidationError(
                f"Gold annotation {annotation.id} is outside frozen reviewed coverage"
            )
        document = db.get(Document, annotation.document_id)
        sentences = (
            db.query(DocumentSentence)
            .filter(
                DocumentSentence.structure_version_id == block.structure_version_id,
                DocumentSentence.ordinal >= block.start_sentence_ordinal,
                DocumentSentence.ordinal <= block.end_sentence_ordinal,
            )
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        source_ids = adjudication.get("source_annotation_ids") or []
        items.append(
            AnnotationSetItemCreate(
                source_annotation_id=annotation.id,
                document_id=annotation.document_id,
                structure_version_id=block.structure_version_id,
                target_version_id=block.target_version_id,
                guideline_version_id=annotation.guideline_version_id,
                start_sentence_id=block.start_sentence_id,
                end_sentence_id=block.end_sentence_id,
                start_sentence_ordinal=block.start_sentence_ordinal,
                end_sentence_ordinal=block.end_sentence_ordinal,
                start_char=annotation.start_offset,
                end_char=annotation.end_offset,
                block_text=document.text[annotation.start_offset : annotation.end_offset],
                section_paths=[list(sentence.section.path or []) for sentence in sentences],
                labels=list(block.labels or []),
                source="adjudicated" if source_ids else "solo_gold",
                metadata_={
                    "adjudication": adjudication,
                    "note": block.note,
                },
            )
        )
    return items, regions


def _intersect_interval_sets(
    interval_sets: list[list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    if not interval_sets:
        return []
    intersection = list(interval_sets[0])
    for candidates in interval_sets[1:]:
        next_intersection: list[tuple[int, int]] = []
        left_index = 0
        right_index = 0
        while left_index < len(intersection) and right_index < len(candidates):
            left = intersection[left_index]
            right = candidates[right_index]
            start = max(left[0], right[0])
            end = min(left[1], right[1])
            if start <= end:
                next_intersection.append((start, end))
            if left[1] <= right[1]:
                left_index += 1
            else:
                right_index += 1
        intersection = next_intersection
        if not intersection:
            break
    return intersection


def freeze_annotation_set(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: AnnotationSetCreate,
    actor_user_id: int | None,
) -> AnnotationSet:
    _require_project(db, project_id)
    snapshot = db.get(CorpusSnapshot, data.corpus_snapshot_id)
    if snapshot is None or snapshot.project_id != project_id:
        raise NotFoundError("Corpus snapshot not found in project")
    snapshot_documents = {item.document_id: item for item in snapshot.documents}
    requested_targets = set(data.target_version_ids)
    if data.items or data.reviewed_regions:
        raise ValidationError(
            "Frozen annotation items and reviewed regions are server-derived; omit them"
        )
    for target_version_id in requested_targets:
        target_version = db.get(EvidenceTargetVersion, target_version_id)
        if target_version is None or target_version.target.project_id != project_id:
            raise ValidationError(
                f"Evidence target version {target_version_id} is not in project"
            )
    for guideline_version_id in set(data.guideline_version_ids):
        guideline = db.get(GuidelineVersion, guideline_version_id)
        if guideline is None or guideline.project_id != project_id:
            raise ValidationError(
                f"Guideline version {guideline_version_id} is not in project"
            )

    derived_items, derived_regions = _derive_frozen_annotation_contents(
        db,
        project_id=project_id,
        snapshot_documents=snapshot_documents,
        requested_targets=requested_targets,
    )
    if not derived_regions:
        raise ValidationError(
            "No reviewed evidence coverage is available for the selected frozen scope"
        )
    derived_guidelines = {
        item.guideline_version_id
        for item in derived_items
        if item.guideline_version_id is not None
    }
    guideline_version_ids = sorted(
        set(data.guideline_version_ids) | derived_guidelines
    )

    manifest = {
        "schema_version": "annotation-set-v1",
        "project_id": project_id,
        "name": data.name,
        "corpus_snapshot_artifact_id": snapshot.artifact_id,
        "target_version_ids": sorted(requested_targets),
        "guideline_version_ids": guideline_version_ids,
        "items": [item.model_dump() for item in derived_items],
        "reviewed_regions": [item.model_dump() for item in derived_regions],
        "metadata": data.metadata_,
    }
    artifact = _store_manifest(
        db,
        storage,
        project_id=project_id,
        artifact_type="annotation_set",
        manifest=manifest,
        created_by_user_id=actor_user_id,
    )
    existing = db.query(AnnotationSet).filter(AnnotationSet.artifact_id == artifact.id).first()
    if existing is not None:
        return existing

    annotation_set = AnnotationSet(
        project_id=project_id,
        artifact_id=artifact.id,
        corpus_snapshot_id=snapshot.id,
        name=data.name,
        target_version_ids=sorted(requested_targets),
        guideline_version_ids=guideline_version_ids,
        block_count=len(derived_items),
        reviewed_region_count=len(derived_regions),
        metadata_=data.metadata_,
    )
    db.add(annotation_set)
    db.flush()
    for item in derived_items:
        db.add(AnnotationSetItem(annotation_set_id=annotation_set.id, **item.model_dump()))
    for region in derived_regions:
        db.add(
            AnnotationSetReviewRegion(
                annotation_set_id=annotation_set.id,
                **region.model_dump(),
            )
        )
    add_lineage_edge(
        db,
        upstream_artifact_id=snapshot.artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="frozen_from",
    )
    db.commit()
    db.refresh(annotation_set)
    return annotation_set


def register_export_artifact(
    db: Session,
    *,
    project_id: int,
    stored: StoredObject,
    format_key: str,
    file_name: str,
    row_count: int,
    actor_user_id: int | None,
    corpus_snapshot_id: int | None = None,
    annotation_set_id: int | None = None,
    metadata: dict | None = None,
) -> ExportArtifact:
    _require_project(db, project_id)
    if corpus_snapshot_id is None and annotation_set_id is None:
        raise ValidationError("An export must reference a corpus snapshot or annotation set")
    annotation_set = db.get(AnnotationSet, annotation_set_id) if annotation_set_id else None
    snapshot = db.get(CorpusSnapshot, corpus_snapshot_id) if corpus_snapshot_id else None
    if annotation_set is not None:
        if annotation_set.project_id != project_id:
            raise ValidationError("Annotation set is not in project")
        snapshot = annotation_set.corpus_snapshot
        corpus_snapshot_id = snapshot.id
    if snapshot is None or snapshot.project_id != project_id:
        raise ValidationError("Corpus snapshot is not in project")

    manifest = {
        "schema_version": "export-artifact-v1",
        "format_key": format_key,
        "file_name": file_name,
        "row_count": row_count,
        "corpus_snapshot_id": corpus_snapshot_id,
        "annotation_set_id": annotation_set_id,
        "checksum_sha256": stored.checksum_sha256,
        "metadata": metadata or {},
    }
    artifact = _artifact_for_stored_object(
        db,
        project_id=project_id,
        artifact_type="export",
        stored=stored,
        manifest=manifest,
        created_by_user_id=actor_user_id,
    )
    existing = db.query(ExportArtifact).filter(ExportArtifact.artifact_id == artifact.id).first()
    if existing is not None:
        return existing
    export = ExportArtifact(
        project_id=project_id,
        artifact_id=artifact.id,
        corpus_snapshot_id=corpus_snapshot_id,
        annotation_set_id=annotation_set_id,
        format_key=format_key,
        file_name=file_name,
        row_count=row_count,
        metadata_=metadata or {},
    )
    db.add(export)
    upstream_artifact_id = (
        annotation_set.artifact_id if annotation_set is not None else snapshot.artifact_id
    )
    add_lineage_edge(
        db,
        upstream_artifact_id=upstream_artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="exported_as",
    )
    db.commit()
    db.refresh(export)
    return export


def add_lineage_edge(
    db: Session,
    *,
    upstream_artifact_id: int,
    downstream_artifact_id: int,
    relationship_type: str = "derived_from",
    metadata: dict | None = None,
) -> LineageEdge:
    if upstream_artifact_id == downstream_artifact_id:
        raise ValidationError("A lineage artifact cannot derive from itself")
    upstream = db.get(LineageArtifact, upstream_artifact_id)
    downstream = db.get(LineageArtifact, downstream_artifact_id)
    if upstream is None or downstream is None:
        raise NotFoundError("Lineage artifact not found")
    if upstream.project_id != downstream.project_id:
        raise ValidationError("Lineage edges may not cross projects")
    if _is_reachable(db, start_id=downstream_artifact_id, target_id=upstream_artifact_id):
        raise ConflictError("Lineage edge would create a cycle")
    existing = (
        db.query(LineageEdge)
        .filter(
            LineageEdge.upstream_artifact_id == upstream_artifact_id,
            LineageEdge.downstream_artifact_id == downstream_artifact_id,
            LineageEdge.relationship_type == relationship_type,
        )
        .first()
    )
    if existing is not None:
        return existing
    edge = LineageEdge(
        upstream_artifact_id=upstream_artifact_id,
        downstream_artifact_id=downstream_artifact_id,
        relationship_type=relationship_type,
        metadata_=metadata or {},
    )
    db.add(edge)
    db.flush()
    return edge


def _is_reachable(db: Session, *, start_id: int, target_id: int) -> bool:
    frontier = [start_id]
    seen: set[int] = set()
    while frontier:
        current = frontier.pop()
        if current == target_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(
            edge.downstream_artifact_id
            for edge in db.query(LineageEdge)
            .filter(LineageEdge.upstream_artifact_id == current)
            .all()
        )
    return False


def lineage_graph(db: Session, artifact_id: int) -> tuple[list[LineageArtifact], list[LineageEdge]]:
    if db.get(LineageArtifact, artifact_id) is None:
        raise NotFoundError("Lineage artifact not found")
    artifact_ids = {artifact_id}
    frontier = [artifact_id]
    edges: dict[int, LineageEdge] = {}
    while frontier:
        current = frontier.pop()
        adjacent = (
            db.query(LineageEdge)
            .filter(
                (LineageEdge.upstream_artifact_id == current)
                | (LineageEdge.downstream_artifact_id == current)
            )
            .all()
        )
        for edge in adjacent:
            edges[edge.id] = edge
            for candidate in (edge.upstream_artifact_id, edge.downstream_artifact_id):
                if candidate not in artifact_ids:
                    artifact_ids.add(candidate)
                    frontier.append(candidate)
    artifacts = (
        db.query(LineageArtifact)
        .filter(LineageArtifact.id.in_(artifact_ids))
        .order_by(LineageArtifact.id)
        .all()
    )
    return artifacts, sorted(edges.values(), key=lambda edge: edge.id)


class _BytesReader:
    def __init__(self, data: bytes) -> None:
        self._stream = memoryview(data)
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        if self._position >= len(self._stream):
            return b""
        if size < 0:
            size = len(self._stream) - self._position
        end = min(len(self._stream), self._position + size)
        chunk = bytes(self._stream[self._position : end])
        self._position = end
        return chunk


class IterableBytesReader:
    """Adapt generated export chunks to the file-like storage interface."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = iter(chunks)
        self._buffer = bytearray()
        self._finished = False

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            for chunk in self._chunks:
                self._append(chunk)
            self._finished = True
            result = bytes(self._buffer)
            self._buffer.clear()
            return result
        while len(self._buffer) < size and not self._finished:
            try:
                self._append(next(self._chunks))
            except StopIteration:
                self._finished = True
        result = bytes(self._buffer[:size])
        del self._buffer[:size]
        return result

    def _append(self, chunk: bytes) -> None:
        if not isinstance(chunk, bytes):
            raise TypeError("Export chunks must be bytes")
        self._buffer.extend(chunk)
