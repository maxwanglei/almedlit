from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from sqlalchemy import func
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assignment_access_clause, resource_access_clause
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.corpus.models import (
    Document,
    DocumentParagraph,
    DocumentSection,
    DocumentSentence,
    DocumentStructureVersion,
)
from al_medlit.corpus.schemas import (
    DocumentCreate,
    DocumentParagraphRead,
    DocumentSectionRead,
    DocumentSentenceRead,
    DocumentStructureRead,
    StructureRangeRead,
    StructureVersionRead,
)
from al_medlit.corpus.segmentation import (
    SEGMENTER_NAME,
    SEGMENTER_VERSION,
    segment_document,
    text_sha256,
)
from al_medlit.project.models import Project, ProjectTask, TaskAssignment

OPEN_ASSIGNMENT_STATUSES = ("assigned", "in_progress", "blocked")


def create_document(
    db: Session,
    data: DocumentCreate,
    *,
    structure_source_metadata: dict | None = None,
) -> Document:
    payload = data.model_dump()
    project = db.get(Project, payload["project_id"])
    if project is None:
        raise ValidationError(f"Project {payload['project_id']} not found")

    doc = Document(**payload)
    db.add(doc)
    db.flush()
    structure_version = _build_structure_version(
        db,
        doc,
        source_metadata=structure_source_metadata,
    )
    doc.active_structure_version = structure_version
    db.commit()
    db.refresh(doc)
    return doc


def list_documents(
    db: Session,
    project_id: int | None = None,
    user: User | None = None,
    assignee_user_id: int | None = None,
) -> list[Document]:
    query = db.query(Document)
    if project_id is not None:
        query = query.filter(Document.project_id == project_id)
    if user is not None:
        query = query.join(Project).filter(
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=Document.project_id,
                document_id=Document.id,
            )
        )
        if assignee_user_id is not None:
            # Routers only use this for the authenticated user's ``scope=mine``.
            # Keep the parameter check explicit so internal callers cannot use it
            # to query another annotator's assignments accidentally.
            if assignee_user_id != user.id:
                return []
            query = query.filter(
                assignment_access_clause(
                    user,
                    project_id=Document.project_id,
                    document_id=Document.id,
                )
            )
    return query.order_by(Document.created_at.desc()).all()


def get_document(db: Session, document_id: int) -> Document | None:
    return db.get(Document, document_id)


def _structure_source_metadata(document: Document) -> dict | None:
    metadata = document.metadata_ or {}
    source_metadata = metadata.get("structure_source")
    return deepcopy(source_metadata) if isinstance(source_metadata, dict) else None


def _build_structure_version(
    db: Session,
    document: Document,
    *,
    source_metadata: dict | None = None,
) -> DocumentStructureVersion:
    """Persist a new immutable structure version without committing it."""
    latest_version = (
        db.query(func.max(DocumentStructureVersion.version))
        .filter(DocumentStructureVersion.document_id == document.id)
        .scalar()
    )
    version_number = int(latest_version or 0) + 1
    source_metadata = deepcopy(source_metadata) if source_metadata else {}
    segmented = segment_document(document.text or "", source_metadata=source_metadata)

    structure_version = DocumentStructureVersion(
        document_id=document.id,
        version=version_number,
        segmenter_name=SEGMENTER_NAME,
        segmenter_version=SEGMENTER_VERSION,
        source_hash=text_sha256(document.text or ""),
        text_length=len(document.text or ""),
        status="ready",
        source_metadata=source_metadata,
    )
    db.add(structure_version)
    db.flush()

    for section_plan in segmented.sections:
        section = DocumentSection(
            structure_version_id=structure_version.id,
            ordinal=section_plan.ordinal,
            title=section_plan.title,
            path=list(section_plan.path),
            kind=section_plan.kind,
            start_offset=section_plan.start_offset,
            end_offset=section_plan.end_offset,
            locator=deepcopy(section_plan.locator),
        )
        db.add(section)
        db.flush()
        for paragraph_plan in section_plan.paragraphs:
            paragraph = DocumentParagraph(
                structure_version_id=structure_version.id,
                section_id=section.id,
                ordinal=paragraph_plan.ordinal,
                section_ordinal=paragraph_plan.section_ordinal,
                start_offset=paragraph_plan.start_offset,
                end_offset=paragraph_plan.end_offset,
                locator=deepcopy(paragraph_plan.locator),
            )
            db.add(paragraph)
            db.flush()
            db.add_all(
                [
                    DocumentSentence(
                        structure_version_id=structure_version.id,
                        section_id=section.id,
                        paragraph_id=paragraph.id,
                        ordinal=sentence_plan.ordinal,
                        paragraph_ordinal=sentence_plan.paragraph_ordinal,
                        start_offset=sentence_plan.start_offset,
                        end_offset=sentence_plan.end_offset,
                        text_hash=sentence_plan.text_hash,
                    )
                    for sentence_plan in paragraph_plan.sentences
                ]
            )
    db.flush()
    return structure_version


def _get_structure_version(
    db: Session,
    document: Document,
    structure_version_id: int | None,
) -> DocumentStructureVersion:
    resolved_id = (
        structure_version_id
        if structure_version_id is not None
        else document.active_structure_version_id
    )
    if resolved_id is None:
        raise NotFoundError("Document has no active structure version")
    structure_version = db.get(DocumentStructureVersion, resolved_id)
    if structure_version is None or structure_version.document_id != document.id:
        raise NotFoundError("Document structure version not found")
    return structure_version


def get_document_structure(
    db: Session,
    document_id: int,
    *,
    structure_version_id: int | None = None,
    sentence_start: int = 0,
    sentence_limit: int = 500,
) -> DocumentStructureRead:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    structure_version = _get_structure_version(db, document, structure_version_id)

    total_sentences = (
        db.query(func.count(DocumentSentence.id))
        .filter(DocumentSentence.structure_version_id == structure_version.id)
        .scalar()
        or 0
    )
    if sentence_start > total_sentences:
        raise ValidationError(
            f"sentence_start must be between 0 and {total_sentences}"
        )
    sentences = (
        db.query(DocumentSentence)
        .filter(
            DocumentSentence.structure_version_id == structure_version.id,
            DocumentSentence.ordinal >= sentence_start,
        )
        .order_by(DocumentSentence.ordinal)
        .limit(sentence_limit)
        .all()
    )
    paragraph_ids = {sentence.paragraph_id for sentence in sentences}
    section_ids = {sentence.section_id for sentence in sentences}
    paragraphs = (
        db.query(DocumentParagraph)
        .filter(DocumentParagraph.id.in_(paragraph_ids))
        .order_by(DocumentParagraph.ordinal)
        .all()
        if paragraph_ids
        else []
    )
    sections = (
        db.query(DocumentSection)
        .filter(DocumentSection.id.in_(section_ids))
        .order_by(DocumentSection.ordinal)
        .all()
        if section_ids
        else []
    )

    # ``end_ordinal`` is exclusive, matching Python slicing and all stored text
    # ranges. On an empty page it remains equal to ``sentence_start``.
    end_ordinal = sentences[-1].ordinal + 1 if sentences else sentence_start
    return DocumentStructureRead(
        document_id=document.id,
        active_structure_version_id=document.active_structure_version_id,
        structure_version=StructureVersionRead.model_validate(structure_version),
        range=StructureRangeRead(
            start_ordinal=sentence_start,
            end_ordinal=end_ordinal,
            total_sentences=total_sentences,
            has_more=end_ordinal < total_sentences,
        ),
        sections=[DocumentSectionRead.model_validate(section) for section in sections],
        paragraphs=[
            DocumentParagraphRead.model_validate(paragraph) for paragraph in paragraphs
        ],
        sentences=[
            DocumentSentenceRead(
                id=sentence.id,
                section_id=sentence.section_id,
                paragraph_id=sentence.paragraph_id,
                ordinal=sentence.ordinal,
                paragraph_ordinal=sentence.paragraph_ordinal,
                start_offset=sentence.start_offset,
                end_offset=sentence.end_offset,
                text=document.text[sentence.start_offset : sentence.end_offset],
            )
            for sentence in sentences
        ],
    )


def _assert_structure_activation_allowed(
    db: Session,
    document: Document,
    structure_version: DocumentStructureVersion,
) -> None:
    if structure_version.source_hash != text_sha256(document.text or ""):
        raise ConflictError(
            "Structure was built from a different document text and cannot be activated"
        )
    current_id = document.active_structure_version_id
    if current_id is None or current_id == structure_version.id:
        return

    if _has_open_evidence_assignments(db, document.id, current_id):
        raise ConflictError(
            "Cannot replace the active structure while evidence assignments are open"
        )


def _has_open_evidence_assignments(
    db: Session,
    document_id: int,
    structure_version_id: int,
) -> bool:
    return (
        db.query(TaskAssignment.id)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.document_id == document_id,
            TaskAssignment.structure_version_id == structure_version_id,
            TaskAssignment.status.in_(OPEN_ASSIGNMENT_STATUSES),
            ProjectTask.annotation_type == "evidence_block",
        )
        .first()
        is not None
    )


def activate_document_structure(
    db: Session,
    document_id: int,
    structure_version_id: int,
) -> DocumentStructureVersion:
    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if document is None:
        raise NotFoundError("Document not found")
    structure_version = _get_structure_version(db, document, structure_version_id)
    _assert_structure_activation_allowed(db, document, structure_version)
    document.active_structure_version = structure_version
    db.commit()
    db.refresh(structure_version)
    return structure_version


def rebuild_document_structure(
    db: Session,
    document_id: int,
    *,
    activate: bool = True,
) -> DocumentStructureVersion:
    document = (
        db.query(Document)
        .filter(Document.id == document_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if document is None:
        raise NotFoundError("Document not found")
    if (
        activate
        and document.active_structure_version_id is not None
        and _has_open_evidence_assignments(
            db,
            document.id,
            document.active_structure_version_id,
        )
    ):
        # Check before doing the comparatively expensive segmentation. Managers
        # can set ``activate=false`` to build an inspectable candidate version.
        raise ConflictError(
            "Cannot replace the active structure while evidence assignments are open"
        )

    structure_version = _build_structure_version(
        db,
        document,
        source_metadata=_structure_source_metadata(document),
    )
    if activate:
        document.active_structure_version = structure_version
    db.commit()
    db.refresh(structure_version)
    return structure_version


@dataclass(slots=True)
class StructureBackfillResult:
    created: int = 0
    activated_existing: int = 0
    skipped: int = 0
    failures: dict[int, str] = field(default_factory=dict)


def backfill_document_structures(
    db: Session,
    *,
    document_ids: list[int] | None = None,
) -> StructureBackfillResult:
    """Idempotently provide every selected document with an active structure."""
    query = db.query(Document.id).order_by(Document.id)
    if document_ids is not None:
        query = query.filter(Document.id.in_(document_ids))
    ids = [document_id for (document_id,) in query.all()]
    result = StructureBackfillResult()

    for document_id in ids:
        try:
            document = db.get(Document, document_id)
            if document is None:
                result.skipped += 1
                continue
            if document.active_structure_version_id is not None:
                result.skipped += 1
                continue
            current_hash = text_sha256(document.text or "")
            existing = (
                db.query(DocumentStructureVersion)
                .filter(
                    DocumentStructureVersion.document_id == document.id,
                    DocumentStructureVersion.source_hash == current_hash,
                    DocumentStructureVersion.status == "ready",
                )
                .order_by(DocumentStructureVersion.version.desc())
                .first()
            )
            if existing is not None:
                document.active_structure_version = existing
                db.commit()
                result.activated_existing += 1
                continue
            structure_version = _build_structure_version(
                db,
                document,
                source_metadata=_structure_source_metadata(document),
            )
            document.active_structure_version = structure_version
            db.commit()
            result.created += 1
        except Exception as exc:  # keep a large backfill retryable per document
            db.rollback()
            result.failures[document_id] = str(exc)
    return result
