from __future__ import annotations

from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from al_medlit.annotation.models import Annotation
from al_medlit.annotation.types import AnnotationValidationError
from al_medlit.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from al_medlit.corpus.models import (
    Document,
    DocumentSentence,
    DocumentStructureVersion,
)
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceBlockRevision,
    EvidenceReviewCoverage,
    EvidenceReviewEvent,
    EvidenceTarget,
    EvidenceTargetVersion,
)
from al_medlit.evidence.schemas import (
    EvidenceAdjudicationCreate,
    EvidenceBlockPayloadV1,
    EvidenceMergeRequest,
    EvidenceReviewIntervalRequest,
    EvidenceSplitRequest,
    EvidenceTargetCreate,
    EvidenceTargetVersionCreate,
)
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.project.schemas import EvidenceBlockTaskSettingsV1

MUTABLE_ASSIGNMENT_STATUSES = ("assigned", "in_progress", "blocked")


def _required_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _required_evidence_task(db: Session, project_id: int, task_id: int) -> ProjectTask:
    task = db.get(ProjectTask, task_id)
    if (
        task is None
        or task.project_id != project_id
        or task.annotation_type != "evidence_block"
    ):
        raise ValidationError("task_id must identify the project's evidence-block task")
    return task


def _task_settings(task: ProjectTask) -> EvidenceBlockTaskSettingsV1:
    return EvidenceBlockTaskSettingsV1.model_validate(task.settings or {})


def create_target(
    db: Session,
    project_id: int,
    data: EvidenceTargetCreate,
    *,
    actor_user_id: int | None,
) -> EvidenceTarget:
    _required_project(db, project_id)
    task = _required_evidence_task(db, project_id, data.task_id)
    target = EvidenceTarget(
        project_id=project_id,
        task_id=task.id,
        key=data.key,
        name=data.name.strip(),
        description=data.description,
        is_active=False,
        created_by_user_id=actor_user_id,
    )
    initial = data.initial_version
    target.versions.append(
        EvidenceTargetVersion(
            version_number=1,
            text=initial.text,
            guidance=initial.guidance,
            inclusion_guidance=initial.inclusion_guidance,
            exclusion_guidance=initial.exclusion_guidance,
            metadata_=initial.metadata_,
            created_by_user_id=actor_user_id,
        )
    )
    db.add(target)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(f"Evidence target key {data.key!r} already exists") from exc
    db.refresh(target)
    return target


def list_targets(db: Session, project_id: int) -> list[EvidenceTarget]:
    _required_project(db, project_id)
    return (
        db.query(EvidenceTarget)
        .filter(EvidenceTarget.project_id == project_id)
        .order_by(EvidenceTarget.created_at.asc(), EvidenceTarget.id.asc())
        .all()
    )


def _required_target(db: Session, project_id: int, target_id: int) -> EvidenceTarget:
    target = db.get(EvidenceTarget, target_id)
    if target is None or target.project_id != project_id:
        raise NotFoundError("Evidence target not found")
    return target


def create_target_version(
    db: Session,
    project_id: int,
    target_id: int,
    data: EvidenceTargetVersionCreate,
    *,
    actor_user_id: int | None,
) -> EvidenceTargetVersion:
    target = (
        db.query(EvidenceTarget)
        .filter(EvidenceTarget.id == target_id, EvidenceTarget.project_id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if target is None:
        raise NotFoundError("Evidence target not found")
    next_number = (
        db.query(func.max(EvidenceTargetVersion.version_number))
        .filter(EvidenceTargetVersion.target_id == target.id)
        .scalar()
        or 0
    ) + 1
    version = EvidenceTargetVersion(
        target_id=target.id,
        version_number=next_number,
        text=data.text,
        guidance=data.guidance,
        inclusion_guidance=data.inclusion_guidance,
        exclusion_guidance=data.exclusion_guidance,
        metadata_=data.metadata_,
        created_by_user_id=actor_user_id,
    )
    db.add(version)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError("Target version was created concurrently; retry") from exc
    db.refresh(version)
    return version


def _write_active_target_settings(
    task: ProjectTask,
    *,
    target_id: int,
    active: bool,
) -> None:
    settings = _task_settings(task)
    active_ids = set(settings.active_target_ids)
    if active:
        active_ids.add(target_id)
    else:
        active_ids.discard(target_id)
    settings.active_target_ids = sorted(active_ids)
    task.settings = settings.model_dump()
    flag_modified(task, "settings")


def _locked_target_and_task(
    db: Session,
    project_id: int,
    target_id: int,
) -> tuple[EvidenceTarget, ProjectTask]:
    candidate = _required_target(db, project_id, target_id)
    task = (
        db.query(ProjectTask)
        .filter(
            ProjectTask.id == candidate.task_id,
            ProjectTask.project_id == project_id,
            ProjectTask.annotation_type == "evidence_block",
        )
        .populate_existing()
        .with_for_update(of=ProjectTask)
        .first()
    )
    if task is None:
        raise ValidationError("Evidence target task was not found")
    target = (
        db.query(EvidenceTarget)
        .filter(
            EvidenceTarget.id == target_id,
            EvidenceTarget.project_id == project_id,
            EvidenceTarget.task_id == task.id,
        )
        .populate_existing()
        .with_for_update(of=EvidenceTarget)
        .one()
    )
    return target, task


def activate_target(
    db: Session,
    project_id: int,
    target_id: int,
    version_id: int,
) -> EvidenceTarget:
    target, task = _locked_target_and_task(db, project_id, target_id)
    version = db.get(EvidenceTargetVersion, version_id)
    if version is None or version.target_id != target.id:
        raise ValidationError("version_id must belong to the evidence target")
    target.active_version_id = version.id
    target.is_active = True
    _write_active_target_settings(task, target_id=target.id, active=True)
    db.commit()
    db.refresh(target)
    return target


def deactivate_target(db: Session, project_id: int, target_id: int) -> EvidenceTarget:
    target, task = _locked_target_and_task(db, project_id, target_id)
    target.is_active = False
    _write_active_target_settings(task, target_id=target.id, active=False)
    db.commit()
    db.refresh(target)
    return target


def require_target_version_scope(
    db: Session,
    *,
    project_id: int,
    target_version_id: int,
    task_id: int | None = None,
) -> tuple[EvidenceTargetVersion, EvidenceTarget]:
    version = db.get(EvidenceTargetVersion, target_version_id)
    target = version.target if version is not None else None
    if (
        version is None
        or target is None
        or target.project_id != project_id
        or (task_id is not None and target.task_id != task_id)
    ):
        raise ValidationError("target_version_id does not belong to this evidence scope")
    return version, target


def _lock_evidence_scope(db: Session, target_version_id: int) -> None:
    """Serialize every mutable operation for one immutable target version.

    The target-version row always exists even when coverage and annotations do
    not, making it a stable PostgreSQL lock for normalization, boundary edits,
    command replay, and submission.
    """

    locked = (
        db.query(EvidenceTargetVersion.id)
        .filter(EvidenceTargetVersion.id == target_version_id)
        .order_by(EvidenceTargetVersion.id)
        .with_for_update()
        .first()
    )
    if locked is None:
        raise NotFoundError("Evidence target version not found")


def _lock_actor_assignment_if_present(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    target_version_id: int,
    structure_version_id: int,
    guideline_version_id: int | None,
    actor_user_id: int | None,
    required_assignment_id: int | None = None,
) -> TaskAssignment | None:
    """Lock and enforce an actor's exact assignment after the target lock.

    Managers may work without an assignment, but an exact assignment that has
    reached a final state is immutable for every role, including personal-workspace
    admins. This service-layer check also covers inference and command replay paths.
    """

    if actor_user_id is None:
        return None
    assignment = (
        db.query(TaskAssignment)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.document_id == document_id,
            TaskAssignment.assignee_user_id == actor_user_id,
            TaskAssignment.target_version_id == target_version_id,
            TaskAssignment.structure_version_id == structure_version_id,
            TaskAssignment.guideline_version_id == guideline_version_id,
            ProjectTask.project_id == project_id,
            ProjectTask.annotation_type == "evidence_block",
            ProjectTask.enabled.is_(True),
        )
        .filter(
            TaskAssignment.id == required_assignment_id
            if required_assignment_id is not None
            else True
        )
        .populate_existing()
        .with_for_update(of=TaskAssignment)
        .first()
    )
    if assignment is None and required_assignment_id is not None:
        raise ForbiddenError(
            "Evidence assignment is no longer assigned to the current user"
        )
    if assignment is not None and assignment.status not in MUTABLE_ASSIGNMENT_STATUSES:
        raise ConflictError("Evidence assignment is finalized and read-only")
    return assignment


def _required_document_structure(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
) -> tuple[Document, DocumentStructureVersion]:
    document = db.get(Document, document_id)
    if document is None or document.project_id != project_id:
        raise AnnotationValidationError(
            f"Document {document_id} does not belong to project {project_id}"
        )
    structure = db.get(DocumentStructureVersion, structure_version_id)
    if structure is None or structure.document_id != document.id:
        raise AnnotationValidationError(
            "structure_version_id must belong to the annotation document"
        )
    return document, structure


def _sentence_range(
    db: Session,
    *,
    structure_version_id: int,
    start_sentence_id: int,
    end_sentence_id: int,
) -> tuple[DocumentSentence, DocumentSentence]:
    start = db.get(DocumentSentence, start_sentence_id)
    end = db.get(DocumentSentence, end_sentence_id)
    if (
        start is None
        or end is None
        or start.structure_version_id != structure_version_id
        or end.structure_version_id != structure_version_id
    ):
        raise AnnotationValidationError(
            "start_sentence_id and end_sentence_id must belong to structure_version_id"
        )
    if start.ordinal > end.ordinal:
        raise AnnotationValidationError(
            "start_sentence_id must not follow end_sentence_id"
        )
    count = (
        db.query(func.count(DocumentSentence.id))
        .filter(
            DocumentSentence.structure_version_id == structure_version_id,
            DocumentSentence.ordinal >= start.ordinal,
            DocumentSentence.ordinal <= end.ordinal,
        )
        .scalar()
    )
    if count != end.ordinal - start.ordinal + 1:
        raise AnnotationValidationError("Evidence block sentence ordinals are not contiguous")
    return start, end


def _block_state(annotation: Annotation, block: EvidenceBlockAnnotation) -> dict:
    return {
        "annotation_id": annotation.id,
        "project_id": annotation.project_id,
        "document_id": annotation.document_id,
        "annotation_type": annotation.annotation_type,
        "label": annotation.label,
        "source": annotation.source,
        "status": annotation.status,
        "confidence": annotation.confidence,
        "annotator_user_id": annotation.annotator_user_id,
        "annotator_id": annotation.annotator_id,
        "model_checkpoint_id": annotation.model_checkpoint_id,
        "guideline_version_id": annotation.guideline_version_id,
        "evidence": dict(annotation.evidence or {}),
        "attributes": dict(annotation.attributes or {}),
        "structure_version_id": block.structure_version_id,
        "target_version_id": block.target_version_id,
        "start_sentence_id": block.start_sentence_id,
        "end_sentence_id": block.end_sentence_id,
        "start_sentence_ordinal": block.start_sentence_ordinal,
        "end_sentence_ordinal": block.end_sentence_ordinal,
        "start_offset": annotation.start_offset,
        "end_offset": annotation.end_offset,
        "text_span": annotation.text_span,
        "labels": list(block.labels or []),
        "note": block.note,
        "boundary_policy": block.boundary_policy,
        "revision": block.revision,
        "locked": block.locked,
        "last_command_group_key": block.last_command_group_key,
    }


_INVALIDATED_COMMAND_PREFIX = "invalidated:"


def _state_guideline_version_id(state: dict | None) -> int | None:
    return state.get("guideline_version_id") if state is not None else None


def command_guideline_version_id(rows: list[EvidenceBlockRevision]) -> int | None:
    """Return the immutable guideline scope shared by every row in a command."""

    guideline_ids = {
        _state_guideline_version_id(state)
        for row in rows
        for state in (row.before, row.after)
        if state is not None
    }
    if len(guideline_ids) > 1:
        raise ConflictError("Evidence command spans multiple guideline versions")
    return next(iter(guideline_ids), None)


def _invalidate_redo_branch(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int | None,
    actor_user_id: int | None,
    command_group_key: str,
) -> None:
    """Discard an undone branch when a new command is recorded in its scope."""

    candidates = (
        db.query(EvidenceBlockRevision)
        .filter(
            EvidenceBlockRevision.project_id == project_id,
            EvidenceBlockRevision.document_id == document_id,
            EvidenceBlockRevision.structure_version_id == structure_version_id,
            EvidenceBlockRevision.target_version_id == target_version_id,
            EvidenceBlockRevision.actor_user_id == actor_user_id,
            EvidenceBlockRevision.is_undone.is_(True),
            EvidenceBlockRevision.command_group_key != command_group_key,
            ~EvidenceBlockRevision.operation.like("adjudicate%"),
            ~EvidenceBlockRevision.operation.like(
                f"{_INVALIDATED_COMMAND_PREFIX}%"
            ),
        )
        .order_by(EvidenceBlockRevision.id.asc())
        .populate_existing()
        .with_for_update()
        .all()
    )
    grouped: dict[str, list[EvidenceBlockRevision]] = {}
    for row in candidates:
        grouped.setdefault(row.command_group_key, []).append(row)
    for command_rows_ in grouped.values():
        if command_guideline_version_id(command_rows_) != guideline_version_id:
            continue
        for row in command_rows_:
            row.operation = f"{_INVALIDATED_COMMAND_PREFIX}{row.operation}"


def _record_revision(
    db: Session,
    annotation: Annotation,
    block: EvidenceBlockAnnotation,
    *,
    actor_user_id: int | None,
    operation: str,
    before: dict | None,
    after: dict | None,
    command_group_key: str,
) -> None:
    state = after or before
    if state is None:
        raise ConflictError("Evidence revision must contain a before or after state")
    if not operation.startswith("adjudicate"):
        _invalidate_redo_branch(
            db,
            project_id=annotation.project_id,
            document_id=annotation.document_id,
            structure_version_id=block.structure_version_id,
            target_version_id=block.target_version_id,
            guideline_version_id=_state_guideline_version_id(state),
            actor_user_id=actor_user_id,
            command_group_key=command_group_key,
        )
    db.add(
        EvidenceBlockRevision(
            annotation_id=annotation.id,
            project_id=annotation.project_id,
            document_id=annotation.document_id,
            structure_version_id=block.structure_version_id,
            target_version_id=block.target_version_id,
            actor_user_id=actor_user_id,
            revision=block.revision,
            operation=operation,
            before=before,
            after=after,
            command_group_key=command_group_key,
        )
    )


def _validate_block_policy(
    db: Session,
    *,
    project_id: int,
    payload: EvidenceBlockPayloadV1,
    start: DocumentSentence,
    end: DocumentSentence,
    annotator_user_id: int | None,
    status: str,
    exclude_annotation_ids: tuple[int, ...] = (),
) -> None:
    # Serialize the validate-then-insert overlap check on PostgreSQL. Target
    # versions are immutable and therefore provide a stable lock row even when
    # this target has no existing blocks yet.
    _lock_evidence_scope(db, payload.target_version_id)
    _version, target = require_target_version_scope(
        db,
        project_id=project_id,
        target_version_id=payload.target_version_id,
    )
    task = _required_evidence_task(db, project_id, target.task_id)
    settings = _task_settings(task)
    if not settings.multi_paragraph_allowed and start.paragraph_id != end.paragraph_id:
        raise AnnotationValidationError("Evidence blocks may not cross paragraphs")
    if not settings.cross_section_allowed and start.section_id != end.section_id:
        raise AnnotationValidationError("Evidence blocks may not cross sections")
    if not settings.same_target_overlap_allowed:
        query = (
            db.query(EvidenceBlockAnnotation.annotation_id)
            .join(Annotation, Annotation.id == EvidenceBlockAnnotation.annotation_id)
            .filter(
                Annotation.project_id == project_id,
                Annotation.document_id == start.structure_version.document_id,
                Annotation.status == status,
                EvidenceBlockAnnotation.structure_version_id
                == payload.structure_version_id,
                EvidenceBlockAnnotation.target_version_id
                == payload.target_version_id,
                EvidenceBlockAnnotation.start_sentence_ordinal <= end.ordinal,
                EvidenceBlockAnnotation.end_sentence_ordinal >= start.ordinal,
            )
        )
        # Human drafts are private to an annotator. Gold is the canonical shared
        # layer, so it must remain non-overlapping regardless of which manager
        # performed adjudication.
        if status != "gold":
            query = query.filter(Annotation.annotator_user_id == annotator_user_id)
        if exclude_annotation_ids:
            query = query.filter(
                EvidenceBlockAnnotation.annotation_id.not_in(exclude_annotation_ids)
            )
        if query.first() is not None:
            raise AnnotationValidationError(
                "Evidence blocks for the same annotator and target may not overlap"
            )
    if not settings.adjacency_allowed:
        adjacent = (
            db.query(EvidenceBlockAnnotation.annotation_id)
            .join(Annotation, Annotation.id == EvidenceBlockAnnotation.annotation_id)
            .filter(
                Annotation.project_id == project_id,
                Annotation.document_id == start.structure_version.document_id,
                Annotation.status == status,
                EvidenceBlockAnnotation.structure_version_id
                == payload.structure_version_id,
                EvidenceBlockAnnotation.target_version_id == payload.target_version_id,
                (
                    (EvidenceBlockAnnotation.end_sentence_ordinal == start.ordinal - 1)
                    | (EvidenceBlockAnnotation.start_sentence_ordinal == end.ordinal + 1)
                ),
            )
        )
        if status != "gold":
            adjacent = adjacent.filter(
                Annotation.annotator_user_id == annotator_user_id
            )
        if exclude_annotation_ids:
            adjacent = adjacent.filter(
                EvidenceBlockAnnotation.annotation_id.not_in(exclude_annotation_ids)
            )
        if adjacent.first() is not None:
            raise AnnotationValidationError(
                "Adjacent evidence blocks are disabled for this task"
            )


def _create_block_internal(
    db: Session,
    values: dict,
    payload: EvidenceBlockPayloadV1,
    *,
    actor_user_id: int | None,
    source: str,
    status: str,
    command_group_key: str,
    operation: str = "create",
    exclude_annotation_ids: tuple[int, ...] = (),
) -> Annotation:
    document, _structure = _required_document_structure(
        db,
        project_id=values["project_id"],
        document_id=values["document_id"],
        structure_version_id=payload.structure_version_id,
    )
    start, end = _sentence_range(
        db,
        structure_version_id=payload.structure_version_id,
        start_sentence_id=payload.start_sentence_id,
        end_sentence_id=payload.end_sentence_id,
    )
    _validate_block_policy(
        db,
        project_id=values["project_id"],
        payload=payload,
        start=start,
        end=end,
        annotator_user_id=values.get("annotator_user_id"),
        status=status,
        exclude_annotation_ids=exclude_annotation_ids,
    )
    parent_values = dict(values)
    parent_values.pop("evidence_block", None)
    parent_values.update(
        {
            "annotation_type": "evidence_block",
            "label": "evidence_block",
            "start_offset": start.start_offset,
            "end_offset": end.end_offset,
            "text_span": (document.text or "")[start.start_offset : end.end_offset],
            "source": source,
            "status": status,
            "confidence": None,
            "model_checkpoint_id": None,
            "structure_version_id": payload.structure_version_id,
            "head_annotation_id": None,
            "tail_annotation_id": None,
        }
    )
    annotation = Annotation(**parent_values)
    block = EvidenceBlockAnnotation(
        structure_version_id=payload.structure_version_id,
        target_version_id=payload.target_version_id,
        start_sentence_id=start.id,
        end_sentence_id=end.id,
        start_sentence_ordinal=start.ordinal,
        end_sentence_ordinal=end.ordinal,
        labels=payload.labels,
        note=payload.note,
        boundary_policy=payload.boundary_policy,
        revision=1,
        locked=False,
        last_command_group_key=command_group_key,
    )
    annotation.evidence_block = block
    db.add(annotation)
    db.flush()
    _record_revision(
        db,
        annotation,
        block,
        actor_user_id=actor_user_id,
        operation=operation,
        before=None,
        after=_block_state(annotation, block),
        command_group_key=command_group_key,
    )
    return annotation


def create_evidence_block(
    db: Session,
    values: dict,
    payload: EvidenceBlockPayloadV1,
    *,
    actor_user_id: int | None,
    required_assignment_id: int | None = None,
    commit: bool = True,
) -> Annotation:
    _lock_evidence_scope(db, payload.target_version_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=values["project_id"],
        document_id=values["document_id"],
        target_version_id=payload.target_version_id,
        structure_version_id=payload.structure_version_id,
        guideline_version_id=values.get("guideline_version_id"),
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    annotation = _create_block_internal(
        db,
        values,
        payload,
        actor_user_id=actor_user_id,
        source="human",
        status="draft",
        command_group_key=uuid4().hex,
    )
    block = annotation.evidence_block
    _reopen_coverage_ordinals(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        structure_version_id=block.structure_version_id,
        target_version_id=block.target_version_id,
        guideline_version_id=annotation.guideline_version_id,
        reviewer_user_id=annotation.annotator_user_id,
        actor_user_id=actor_user_id,
        start_ordinal=block.start_sentence_ordinal,
        end_ordinal=block.end_sentence_ordinal,
        reason="Evidence block created",
        action="boundary_mutation",
    )
    if commit:
        db.commit()
        db.refresh(annotation)
    return annotation


def _required_block_for_update(
    db: Session, annotation_id: int
) -> tuple[Annotation, EvidenceBlockAnnotation]:
    candidate = db.get(Annotation, annotation_id)
    candidate_block = candidate.evidence_block if candidate is not None else None
    if candidate is None or candidate_block is None:
        raise NotFoundError("Evidence block annotation not found")
    # Always acquire the stable target lock before locking mutable annotation rows.
    _lock_evidence_scope(db, candidate_block.target_version_id)
    annotation = (
        db.query(Annotation)
        .filter(Annotation.id == annotation_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    block = (
        db.query(EvidenceBlockAnnotation)
        .filter(EvidenceBlockAnnotation.annotation_id == annotation_id)
        .populate_existing()
        .with_for_update()
        .one_or_none()
        if annotation is not None
        else None
    )
    if annotation is None or block is None:
        raise NotFoundError("Evidence block annotation not found")
    return annotation, block


def _check_revision(block: EvidenceBlockAnnotation, expected_revision: int | None) -> None:
    if expected_revision is None:
        raise ValidationError("expected_revision is required for evidence mutations")
    if block.revision != expected_revision:
        raise ConflictError(
            f"Evidence block revision is {block.revision}; expected {expected_revision}"
        )
    if block.locked:
        raise ConflictError("Evidence block is locked")


def update_evidence_block(
    db: Session,
    annotation_id: int,
    payload: EvidenceBlockPayloadV1 | None,
    *,
    expected_revision: int | None,
    actor_user_id: int | None,
    required_assignment_id: int | None = None,
) -> Annotation:
    annotation, block = _required_block_for_update(db, annotation_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        target_version_id=block.target_version_id,
        structure_version_id=block.structure_version_id,
        guideline_version_id=annotation.guideline_version_id,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    _check_revision(block, expected_revision)
    if actor_user_id is not None:
        if annotation.annotator_user_id != actor_user_id:
            raise ForbiddenError("Evidence block belongs to another annotator")
        if annotation.source != "human" or annotation.status == "gold":
            raise ConflictError("Model and gold evidence blocks are read-only")
    if payload is None:
        raise ValidationError("evidence_block payload is required")
    if annotation.status == "gold":
        raise ConflictError("Gold evidence blocks are immutable")
    if (
        payload.structure_version_id != block.structure_version_id
        or payload.target_version_id != block.target_version_id
    ):
        raise ValidationError(
            "Evidence block target and structure pins cannot be changed"
        )
    document, _structure = _required_document_structure(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        structure_version_id=payload.structure_version_id,
    )
    start, end = _sentence_range(
        db,
        structure_version_id=payload.structure_version_id,
        start_sentence_id=payload.start_sentence_id,
        end_sentence_id=payload.end_sentence_id,
    )
    _validate_block_policy(
        db,
        project_id=annotation.project_id,
        payload=payload,
        start=start,
        end=end,
        annotator_user_id=annotation.annotator_user_id,
        status=annotation.status,
        exclude_annotation_ids=(annotation.id,),
    )
    before = _block_state(annotation, block)
    command_group_key = uuid4().hex
    old_start = block.start_sentence_ordinal
    old_end = block.end_sentence_ordinal
    block.structure_version_id = payload.structure_version_id
    block.target_version_id = payload.target_version_id
    block.start_sentence_id = start.id
    block.end_sentence_id = end.id
    block.start_sentence_ordinal = start.ordinal
    block.end_sentence_ordinal = end.ordinal
    block.labels = payload.labels
    block.note = payload.note
    block.boundary_policy = payload.boundary_policy
    block.revision += 1
    block.last_command_group_key = command_group_key
    annotation.start_offset = start.start_offset
    annotation.end_offset = end.end_offset
    annotation.text_span = (document.text or "")[start.start_offset : end.end_offset]
    after = _block_state(annotation, block)
    _record_revision(
        db,
        annotation,
        block,
        actor_user_id=actor_user_id,
        operation="update",
        before=before,
        after=after,
        command_group_key=command_group_key,
    )
    if old_start != start.ordinal or old_end != end.ordinal:
        _reopen_coverage_ordinals(
            db,
            project_id=annotation.project_id,
            document_id=annotation.document_id,
            structure_version_id=before["structure_version_id"],
            target_version_id=before["target_version_id"],
            guideline_version_id=annotation.guideline_version_id,
            reviewer_user_id=annotation.annotator_user_id,
            actor_user_id=actor_user_id,
            start_ordinal=min(old_start, start.ordinal),
            end_ordinal=max(old_end, end.ordinal),
            reason="Evidence block boundary changed",
            action="boundary_mutation",
        )
    db.commit()
    db.refresh(annotation)
    return annotation


def delete_evidence_block(
    db: Session,
    annotation_id: int,
    *,
    expected_revision: int | None,
    actor_user_id: int | None,
    required_assignment_id: int | None = None,
) -> bool:
    annotation, block = _required_block_for_update(db, annotation_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        target_version_id=block.target_version_id,
        structure_version_id=block.structure_version_id,
        guideline_version_id=annotation.guideline_version_id,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    if actor_user_id is not None:
        if annotation.annotator_user_id != actor_user_id:
            raise ForbiddenError("Evidence block belongs to another annotator")
        if annotation.source != "human" or annotation.status == "gold":
            raise ConflictError("Model and gold evidence blocks are read-only")
    from al_medlit.annotation.service import assert_annotations_deletable

    assert_annotations_deletable(db, (annotation_id,))
    _check_revision(block, expected_revision)
    if annotation.status == "gold":
        raise ConflictError("Gold evidence blocks are immutable")
    before = _block_state(annotation, block)
    _record_revision(
        db,
        annotation,
        block,
        actor_user_id=actor_user_id,
        operation="delete",
        before=before,
        after=None,
        command_group_key=uuid4().hex,
    )
    _reopen_coverage_ordinals(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        structure_version_id=block.structure_version_id,
        target_version_id=block.target_version_id,
        guideline_version_id=annotation.guideline_version_id,
        reviewer_user_id=annotation.annotator_user_id,
        actor_user_id=actor_user_id,
        start_ordinal=block.start_sentence_ordinal,
        end_ordinal=block.end_sentence_ordinal,
        reason="Evidence block deleted",
        action="boundary_mutation",
    )
    db.delete(annotation)
    db.commit()
    return True


def merge_blocks(
    db: Session,
    data: EvidenceMergeRequest,
    *,
    actor_user_id: int,
    required_assignment_id: int | None = None,
) -> Annotation:
    ids = list(dict.fromkeys(data.annotation_ids))
    if len(ids) != len(data.annotation_ids):
        raise ValidationError("annotation_ids must not contain duplicates")
    candidates = db.query(Annotation).filter(Annotation.id.in_(ids)).all()
    if len(candidates) != len(ids) or any(
        row.evidence_block is None for row in candidates
    ):
        raise NotFoundError("One or more evidence blocks were not found")
    for target_version_id in sorted(
        {row.evidence_block.target_version_id for row in candidates}
    ):
        _lock_evidence_scope(db, target_version_id)
    rows = (
        db.query(Annotation)
        .filter(Annotation.id.in_(ids))
        .populate_existing()
        .with_for_update()
        .all()
    )
    blocks_by_annotation_id = {
        block.annotation_id: block
        for block in (
            db.query(EvidenceBlockAnnotation)
            .filter(EvidenceBlockAnnotation.annotation_id.in_(ids))
            .populate_existing()
            .with_for_update()
            .all()
        )
    }
    if len(rows) != len(ids) or len(blocks_by_annotation_id) != len(ids):
        raise NotFoundError("One or more evidence blocks were not found")
    for row in rows:
        block = blocks_by_annotation_id[row.id]
        _check_revision(block, data.expected_revisions.get(row.id))
        if row.annotator_user_id != actor_user_id:
            raise ForbiddenError("Evidence block belongs to another annotator")
        if row.source != "human" or row.status == "gold":
            raise ConflictError("Model and gold evidence blocks are read-only")
    rows.sort(key=lambda row: row.evidence_block.start_sentence_ordinal)
    first = rows[0]
    first_block = first.evidence_block
    scope = (
        first.project_id,
        first.document_id,
        first.annotator_user_id,
        first.guideline_version_id,
        first_block.structure_version_id,
        first_block.target_version_id,
    )
    if any(
        (
            row.project_id,
            row.document_id,
            row.annotator_user_id,
            row.guideline_version_id,
            row.evidence_block.structure_version_id,
            row.evidence_block.target_version_id,
        )
        != scope
        for row in rows[1:]
    ):
        raise ValidationError(
            "Merged blocks must have the same owner, document, target, structure, and guideline"
        )
    _lock_actor_assignment_if_present(
        db,
        project_id=first.project_id,
        document_id=first.document_id,
        target_version_id=first_block.target_version_id,
        structure_version_id=first_block.structure_version_id,
        guideline_version_id=first.guideline_version_id,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    from al_medlit.annotation.service import assert_annotations_deletable

    assert_annotations_deletable(db, ids)
    for previous, current in zip(rows, rows[1:], strict=False):
        if (
            current.evidence_block.start_sentence_ordinal
            != previous.evidence_block.end_sentence_ordinal + 1
        ):
            raise ValidationError("Only adjacent evidence blocks can be merged")
    labels_values = {tuple(row.evidence_block.labels or []) for row in rows}
    note_values = {row.evidence_block.note for row in rows}
    if len(labels_values) > 1 and data.labels is None:
        raise ValidationError("labels must explicitly resolve differing block metadata")
    if len(note_values) > 1 and "note" not in data.model_fields_set:
        raise ValidationError("note must explicitly resolve differing block metadata")
    labels = data.labels if data.labels is not None else list(next(iter(labels_values)))
    note = data.note if "note" in data.model_fields_set else next(iter(note_values))
    start = first_block.start_sentence
    end = rows[-1].evidence_block.end_sentence
    block_payload = EvidenceBlockPayloadV1(
        structure_version_id=first_block.structure_version_id,
        target_version_id=first_block.target_version_id,
        start_sentence_id=start.id,
        end_sentence_id=end.id,
        labels=labels,
        note=note,
        boundary_policy=data.boundary_policy,
    )
    group = uuid4().hex
    parent_values = {
        "project_id": first.project_id,
        "document_id": first.document_id,
        "annotator_user_id": first.annotator_user_id,
        "annotator_id": first.annotator_id,
        "guideline_version_id": first.guideline_version_id,
        "evidence": dict(first.evidence or {}),
        "attributes": dict(first.attributes or {}),
    }
    merged = _create_block_internal(
        db,
        parent_values,
        block_payload,
        actor_user_id=actor_user_id,
        source="human",
        status="draft",
        command_group_key=group,
        operation="merge_create",
        exclude_annotation_ids=tuple(ids),
    )
    for row in rows:
        old_block = row.evidence_block
        _record_revision(
            db,
            row,
            old_block,
            actor_user_id=actor_user_id,
            operation="merge_delete",
            before=_block_state(row, old_block),
            after=None,
            command_group_key=group,
        )
        db.delete(row)
    _reopen_coverage_ordinals(
        db,
        project_id=first.project_id,
        document_id=first.document_id,
        structure_version_id=first_block.structure_version_id,
        target_version_id=first_block.target_version_id,
        guideline_version_id=first.guideline_version_id,
        reviewer_user_id=first.annotator_user_id,
        actor_user_id=actor_user_id,
        start_ordinal=start.ordinal,
        end_ordinal=end.ordinal,
        reason="Evidence blocks merged",
        action="boundary_mutation",
    )
    db.commit()
    db.refresh(merged)
    return merged


def split_block(
    db: Session,
    annotation_id: int,
    data: EvidenceSplitRequest,
    *,
    actor_user_id: int,
    required_assignment_id: int | None = None,
) -> list[Annotation]:
    annotation, block = _required_block_for_update(db, annotation_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        target_version_id=block.target_version_id,
        structure_version_id=block.structure_version_id,
        guideline_version_id=annotation.guideline_version_id,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    if annotation.annotator_user_id != actor_user_id:
        raise ForbiddenError("Evidence block belongs to another annotator")
    if annotation.source != "human" or annotation.status == "gold":
        raise ConflictError("Model and gold evidence blocks are read-only")
    from al_medlit.annotation.service import assert_annotations_deletable

    assert_annotations_deletable(db, (annotation_id,))
    _check_revision(block, data.expected_revision)
    if annotation.status == "gold":
        raise ConflictError("Gold evidence blocks are immutable")
    split_sentence = db.get(DocumentSentence, data.split_before_sentence_id)
    if (
        split_sentence is None
        or split_sentence.structure_version_id != block.structure_version_id
        or split_sentence.ordinal <= block.start_sentence_ordinal
        or split_sentence.ordinal > block.end_sentence_ordinal
    ):
        raise ValidationError("split_before_sentence_id must be inside the evidence block")
    previous_sentence = (
        db.query(DocumentSentence)
        .filter(
            DocumentSentence.structure_version_id == block.structure_version_id,
            DocumentSentence.ordinal == split_sentence.ordinal - 1,
        )
        .one()
    )
    payloads = [
        EvidenceBlockPayloadV1(
            structure_version_id=block.structure_version_id,
            target_version_id=block.target_version_id,
            start_sentence_id=block.start_sentence_id,
            end_sentence_id=previous_sentence.id,
            labels=list(block.labels or []),
            note=block.note,
            boundary_policy=block.boundary_policy,
        ),
        EvidenceBlockPayloadV1(
            structure_version_id=block.structure_version_id,
            target_version_id=block.target_version_id,
            start_sentence_id=split_sentence.id,
            end_sentence_id=block.end_sentence_id,
            labels=list(block.labels or []),
            note=block.note,
            boundary_policy=block.boundary_policy,
        ),
    ]
    group = uuid4().hex
    parent_values = {
        "project_id": annotation.project_id,
        "document_id": annotation.document_id,
        "annotator_user_id": annotation.annotator_user_id,
        "annotator_id": annotation.annotator_id,
        "guideline_version_id": annotation.guideline_version_id,
        "evidence": dict(annotation.evidence or {}),
        "attributes": dict(annotation.attributes or {}),
    }
    created = [
        _create_block_internal(
            db,
            parent_values,
            payload,
            actor_user_id=actor_user_id,
            source="human",
            status="draft",
            command_group_key=group,
            operation="split_create",
            exclude_annotation_ids=(annotation.id,),
        )
        for payload in payloads
    ]
    _record_revision(
        db,
        annotation,
        block,
        actor_user_id=actor_user_id,
        operation="split_delete",
        before=_block_state(annotation, block),
        after=None,
        command_group_key=group,
    )
    old_start = block.start_sentence_ordinal
    old_end = block.end_sentence_ordinal
    db.delete(annotation)
    _reopen_coverage_ordinals(
        db,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        structure_version_id=block.structure_version_id,
        target_version_id=block.target_version_id,
        guideline_version_id=annotation.guideline_version_id,
        reviewer_user_id=annotation.annotator_user_id,
        actor_user_id=actor_user_id,
        start_ordinal=old_start,
        end_ordinal=old_end,
        reason="Evidence block split",
        action="boundary_mutation",
    )
    db.commit()
    for item in created:
        db.refresh(item)
    return created


def _sentence_for_ordinal(
    db: Session, structure_version_id: int, ordinal: int
) -> DocumentSentence:
    sentence = (
        db.query(DocumentSentence)
        .filter(
            DocumentSentence.structure_version_id == structure_version_id,
            DocumentSentence.ordinal == ordinal,
        )
        .first()
    )
    if sentence is None:
        raise ValidationError(f"Sentence ordinal {ordinal} not found")
    return sentence


def _coverage_query(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int | None,
    reviewer_user_id: int,
):
    return db.query(EvidenceReviewCoverage).filter(
        EvidenceReviewCoverage.project_id == project_id,
        EvidenceReviewCoverage.document_id == document_id,
        EvidenceReviewCoverage.structure_version_id == structure_version_id,
        EvidenceReviewCoverage.target_version_id == target_version_id,
        EvidenceReviewCoverage.guideline_version_id == guideline_version_id,
        EvidenceReviewCoverage.reviewer_user_id == reviewer_user_id,
    )


def _validate_review_scope(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    data: EvidenceReviewIntervalRequest,
) -> tuple[DocumentSentence, DocumentSentence]:
    _required_document_structure(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=data.structure_version_id,
    )
    require_target_version_scope(
        db, project_id=project_id, target_version_id=data.target_version_id
    )
    if data.guideline_version_id is not None:
        guideline = db.get(GuidelineVersion, data.guideline_version_id)
        if guideline is None or guideline.project_id != project_id:
            raise ValidationError("guideline_version_id must belong to the project")
    return _sentence_range(
        db,
        structure_version_id=data.structure_version_id,
        start_sentence_id=data.start_sentence_id,
        end_sentence_id=data.end_sentence_id,
    )


def mark_reviewed(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    reviewer_user_id: int,
    data: EvidenceReviewIntervalRequest,
    required_assignment_id: int | None = None,
) -> dict:
    _lock_evidence_scope(db, data.target_version_id)
    assignment = _lock_actor_assignment_if_present(
        db,
        project_id=project_id,
        document_id=document_id,
        target_version_id=data.target_version_id,
        structure_version_id=data.structure_version_id,
        guideline_version_id=data.guideline_version_id,
        actor_user_id=reviewer_user_id,
        required_assignment_id=required_assignment_id,
    )
    start, end = _validate_review_scope(
        db, project_id=project_id, document_id=document_id, data=data
    )
    from al_medlit.inference.models import (
        EvidenceCandidatePrediction,
        EvidencePredictionReview,
    )

    intersecting_predictions = (
        db.query(EvidenceCandidatePrediction.id)
        .filter(
            EvidenceCandidatePrediction.project_id == project_id,
            EvidenceCandidatePrediction.document_id == document_id,
            EvidenceCandidatePrediction.structure_version_id
            == data.structure_version_id,
            EvidenceCandidatePrediction.target_version_id == data.target_version_id,
            EvidenceCandidatePrediction.start_sentence_ordinal <= end.ordinal,
            EvidenceCandidatePrediction.end_sentence_ordinal >= start.ordinal,
        )
        .all()
    )
    unresolved = []
    for (prediction_id,) in intersecting_predictions:
        review_query = db.query(EvidencePredictionReview.id).filter(
            EvidencePredictionReview.prediction_id == prediction_id,
            EvidencePredictionReview.reviewer_user_id == reviewer_user_id,
            EvidencePredictionReview.action.in_(["accept", "modify", "reject"]),
            EvidencePredictionReview.guideline_version_id
            == data.guideline_version_id,
        )
        review_query = (
            review_query.filter(
                EvidencePredictionReview.assignment_id == assignment.id
            )
            if assignment is not None
            else review_query.filter(EvidencePredictionReview.assignment_id.is_(None))
        )
        if review_query.first() is None:
            unresolved.append(prediction_id)
    if unresolved:
        raise ValidationError(
            "All intersecting predictions must be accepted, modified, or rejected "
            f"before review; unresolved prediction ids: {unresolved}"
        )
    touching = (
        _coverage_query(
            db,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=data.structure_version_id,
            target_version_id=data.target_version_id,
            guideline_version_id=data.guideline_version_id,
            reviewer_user_id=reviewer_user_id,
        )
        .filter(
            EvidenceReviewCoverage.start_sentence_ordinal <= end.ordinal + 1,
            EvidenceReviewCoverage.end_sentence_ordinal >= start.ordinal - 1,
        )
        .all()
    )
    merged_start = min([start.ordinal, *(row.start_sentence_ordinal for row in touching)])
    merged_end = max([end.ordinal, *(row.end_sentence_ordinal for row in touching)])
    for row in touching:
        db.delete(row)
    merged_start_sentence = _sentence_for_ordinal(
        db, data.structure_version_id, merged_start
    )
    merged_end_sentence = _sentence_for_ordinal(db, data.structure_version_id, merged_end)
    db.add(
        EvidenceReviewCoverage(
            project_id=project_id,
            document_id=document_id,
            structure_version_id=data.structure_version_id,
            target_version_id=data.target_version_id,
            guideline_version_id=data.guideline_version_id,
            reviewer_user_id=reviewer_user_id,
            start_sentence_id=merged_start_sentence.id,
            end_sentence_id=merged_end_sentence.id,
            start_sentence_ordinal=merged_start,
            end_sentence_ordinal=merged_end,
        )
    )
    db.add(
        EvidenceReviewEvent(
            project_id=project_id,
            document_id=document_id,
            structure_version_id=data.structure_version_id,
            target_version_id=data.target_version_id,
            guideline_version_id=data.guideline_version_id,
            actor_user_id=reviewer_user_id,
            action="mark_reviewed",
            start_sentence_id=start.id,
            end_sentence_id=end.id,
            start_sentence_ordinal=start.ordinal,
            end_sentence_ordinal=end.ordinal,
            reason=data.reason,
            metadata_={},
        )
    )
    db.commit()
    return get_review_coverage(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=data.structure_version_id,
        target_version_id=data.target_version_id,
        guideline_version_id=data.guideline_version_id,
        reviewer_user_id=reviewer_user_id,
    )


def _reopen_coverage_ordinals(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int | None,
    reviewer_user_id: int | None,
    actor_user_id: int | None,
    start_ordinal: int,
    end_ordinal: int,
    reason: str | None,
    action: str,
) -> None:
    _lock_evidence_scope(db, target_version_id)
    if reviewer_user_id is None:
        return
    start = _sentence_for_ordinal(db, structure_version_id, start_ordinal)
    end = _sentence_for_ordinal(db, structure_version_id, end_ordinal)
    overlapping = (
        _coverage_query(
            db,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            guideline_version_id=guideline_version_id,
            reviewer_user_id=reviewer_user_id,
        )
        .filter(
            EvidenceReviewCoverage.start_sentence_ordinal <= end_ordinal,
            EvidenceReviewCoverage.end_sentence_ordinal >= start_ordinal,
        )
        .all()
    )
    for row in overlapping:
        db.delete(row)
        if row.start_sentence_ordinal < start_ordinal:
            left_end = _sentence_for_ordinal(db, structure_version_id, start_ordinal - 1)
            db.add(
                EvidenceReviewCoverage(
                    project_id=project_id,
                    document_id=document_id,
                    structure_version_id=structure_version_id,
                    target_version_id=target_version_id,
                    guideline_version_id=guideline_version_id,
                    reviewer_user_id=reviewer_user_id,
                    start_sentence_id=row.start_sentence_id,
                    end_sentence_id=left_end.id,
                    start_sentence_ordinal=row.start_sentence_ordinal,
                    end_sentence_ordinal=left_end.ordinal,
                )
            )
        if row.end_sentence_ordinal > end_ordinal:
            right_start = _sentence_for_ordinal(db, structure_version_id, end_ordinal + 1)
            db.add(
                EvidenceReviewCoverage(
                    project_id=project_id,
                    document_id=document_id,
                    structure_version_id=structure_version_id,
                    target_version_id=target_version_id,
                    guideline_version_id=guideline_version_id,
                    reviewer_user_id=reviewer_user_id,
                    start_sentence_id=right_start.id,
                    end_sentence_id=row.end_sentence_id,
                    start_sentence_ordinal=right_start.ordinal,
                    end_sentence_ordinal=row.end_sentence_ordinal,
                )
            )
    db.add(
        EvidenceReviewEvent(
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            guideline_version_id=guideline_version_id,
            actor_user_id=actor_user_id or reviewer_user_id,
            action=action,
            start_sentence_id=start.id,
            end_sentence_id=end.id,
            start_sentence_ordinal=start_ordinal,
            end_sentence_ordinal=end_ordinal,
            reason=reason,
            metadata_={"reviewer_user_id": reviewer_user_id},
        )
    )


def reopen_reviewed(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    reviewer_user_id: int,
    data: EvidenceReviewIntervalRequest,
    required_assignment_id: int | None = None,
) -> dict:
    _lock_evidence_scope(db, data.target_version_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=project_id,
        document_id=document_id,
        target_version_id=data.target_version_id,
        structure_version_id=data.structure_version_id,
        guideline_version_id=data.guideline_version_id,
        actor_user_id=reviewer_user_id,
        required_assignment_id=required_assignment_id,
    )
    start, end = _validate_review_scope(
        db, project_id=project_id, document_id=document_id, data=data
    )
    _reopen_coverage_ordinals(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=data.structure_version_id,
        target_version_id=data.target_version_id,
        guideline_version_id=data.guideline_version_id,
        reviewer_user_id=reviewer_user_id,
        actor_user_id=reviewer_user_id,
        start_ordinal=start.ordinal,
        end_ordinal=end.ordinal,
        reason=data.reason,
        action="reopen",
    )
    db.commit()
    return get_review_coverage(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=data.structure_version_id,
        target_version_id=data.target_version_id,
        guideline_version_id=data.guideline_version_id,
        reviewer_user_id=reviewer_user_id,
    )


def get_review_coverage(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int | None,
    reviewer_user_id: int,
) -> dict:
    _required_document_structure(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=structure_version_id,
    )
    require_target_version_scope(
        db, project_id=project_id, target_version_id=target_version_id
    )
    intervals = (
        _coverage_query(
            db,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            guideline_version_id=guideline_version_id,
            reviewer_user_id=reviewer_user_id,
        )
        .order_by(EvidenceReviewCoverage.start_sentence_ordinal.asc())
        .all()
    )
    events = (
        db.query(EvidenceReviewEvent)
        .filter(
            EvidenceReviewEvent.project_id == project_id,
            EvidenceReviewEvent.document_id == document_id,
            EvidenceReviewEvent.structure_version_id == structure_version_id,
            EvidenceReviewEvent.target_version_id == target_version_id,
            EvidenceReviewEvent.guideline_version_id == guideline_version_id,
            EvidenceReviewEvent.actor_user_id == reviewer_user_id,
        )
        .order_by(EvidenceReviewEvent.created_at.asc(), EvidenceReviewEvent.id.asc())
        .all()
    )
    bounds = (
        db.query(
            func.min(DocumentSentence.ordinal),
            func.max(DocumentSentence.ordinal),
        )
        .filter(DocumentSentence.structure_version_id == structure_version_id)
        .one()
    )
    fully_reviewed = bool(
        bounds[0] is not None
        and len(intervals) == 1
        and intervals[0].start_sentence_ordinal == bounds[0]
        and intervals[0].end_sentence_ordinal == bounds[1]
    )
    return {
        "project_id": project_id,
        "document_id": document_id,
        "structure_version_id": structure_version_id,
        "target_version_id": target_version_id,
        "guideline_version_id": guideline_version_id,
        "reviewer_user_id": reviewer_user_id,
        "intervals": intervals,
        "events": events,
        "fully_reviewed": fully_reviewed,
    }


def assignment_has_full_review_coverage(
    db: Session, assignment: TaskAssignment
) -> bool:
    if assignment.target_version_id is None or assignment.structure_version_id is None:
        return True
    _lock_evidence_scope(db, assignment.target_version_id)
    report = get_review_coverage(
        db,
        project_id=assignment.project_id,
        document_id=assignment.document_id,
        structure_version_id=assignment.structure_version_id,
        target_version_id=assignment.target_version_id,
        guideline_version_id=assignment.guideline_version_id,
        reviewer_user_id=assignment.assignee_user_id,
    )
    return bool(report["fully_reviewed"])


_ADJUDICATION_ASSIGNMENT_STATUSES = {
    "submitted",
    "adjudication_ready",
    "adjudicated",
    "completed",
}
_ADJUDICATION_ANNOTATION_STATUSES = {"draft", "accepted"}


def _required_guideline_scope(
    db: Session,
    *,
    project_id: int,
    guideline_version_id: int,
) -> GuidelineVersion:
    guideline = db.get(GuidelineVersion, guideline_version_id)
    if guideline is None or guideline.project_id != project_id:
        raise ValidationError("guideline_version_id must belong to the project")
    return guideline


def _coverage_contains(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int,
    reviewer_user_id: int,
    start_ordinal: int,
    end_ordinal: int,
) -> bool:
    return (
        _coverage_query(
            db,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            guideline_version_id=guideline_version_id,
            reviewer_user_id=reviewer_user_id,
        )
        .filter(
            EvidenceReviewCoverage.start_sentence_ordinal <= start_ordinal,
            EvidenceReviewCoverage.end_sentence_ordinal >= end_ordinal,
        )
        .first()
        is not None
    )


def _has_adjudication_ready_assignment(
    db: Session,
    *,
    annotation: Annotation,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int,
    lock: bool = False,
) -> bool:
    query = (
        db.query(TaskAssignment.id)
        .filter(
            TaskAssignment.project_id == annotation.project_id,
            TaskAssignment.document_id == annotation.document_id,
            TaskAssignment.assignee_user_id == annotation.annotator_user_id,
            TaskAssignment.structure_version_id == structure_version_id,
            TaskAssignment.target_version_id == target_version_id,
            TaskAssignment.guideline_version_id == guideline_version_id,
            TaskAssignment.status.in_(_ADJUDICATION_ASSIGNMENT_STATUSES),
        )
    )
    if lock:
        query = query.populate_existing().with_for_update()
    return query.first() is not None


def _is_eligible_adjudication_source(
    db: Session,
    *,
    annotation: Annotation,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int,
    require_ready_assignment: bool,
    lock_assignment: bool = False,
) -> bool:
    block = annotation.evidence_block
    if (
        annotation.project_id != project_id
        or annotation.document_id != document_id
        or annotation.annotation_type != "evidence_block"
        or annotation.source != "human"
        or annotation.status not in _ADJUDICATION_ANNOTATION_STATUSES
        or annotation.annotator_user_id is None
        or annotation.guideline_version_id != guideline_version_id
        or block is None
        or block.locked
        or block.target_version_id != target_version_id
        or block.structure_version_id != structure_version_id
    ):
        return False
    if not _coverage_contains(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=structure_version_id,
        target_version_id=target_version_id,
        guideline_version_id=guideline_version_id,
        reviewer_user_id=annotation.annotator_user_id,
        start_ordinal=block.start_sentence_ordinal,
        end_ordinal=block.end_sentence_ordinal,
    ):
        return False
    return not require_ready_assignment or _has_adjudication_ready_assignment(
        db,
        annotation=annotation,
        structure_version_id=structure_version_id,
        target_version_id=target_version_id,
        guideline_version_id=guideline_version_id,
        lock=lock_assignment,
    )


def comparison(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    target_version_id: int,
    structure_version_id: int,
    guideline_version_id: int,
    annotator_user_ids: list[int] | None = None,
) -> dict:
    project = _required_project(db, project_id)
    require_target_version_scope(
        db, project_id=project_id, target_version_id=target_version_id
    )
    _required_guideline_scope(
        db,
        project_id=project_id,
        guideline_version_id=guideline_version_id,
    )
    _required_document_structure(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=structure_version_id,
    )
    query = (
        db.query(Annotation)
        .join(EvidenceBlockAnnotation)
        .filter(
            Annotation.project_id == project_id,
            Annotation.document_id == document_id,
            Annotation.annotation_type == "evidence_block",
            Annotation.source == "human",
            Annotation.status.in_(_ADJUDICATION_ANNOTATION_STATUSES),
            Annotation.guideline_version_id == guideline_version_id,
            EvidenceBlockAnnotation.target_version_id == target_version_id,
            EvidenceBlockAnnotation.structure_version_id == structure_version_id,
            EvidenceBlockAnnotation.locked.is_(False),
        )
    )
    if annotator_user_ids:
        query = query.filter(Annotation.annotator_user_id.in_(annotator_user_ids))
    candidates = query.order_by(
        EvidenceBlockAnnotation.start_sentence_ordinal.asc(), Annotation.id.asc()
    ).all()
    require_ready_assignment = project.workspace.kind == "team"
    annotations = [
        annotation
        for annotation in candidates
        if _is_eligible_adjudication_source(
            db,
            annotation=annotation,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            guideline_version_id=guideline_version_id,
            require_ready_assignment=require_ready_assignment,
        )
    ]
    return {
        "project_id": project_id,
        "document_id": document_id,
        "target_version_id": target_version_id,
        "structure_version_id": structure_version_id,
        "guideline_version_id": guideline_version_id,
        "blocks": [
            {
                "annotation_id": annotation.id,
                "annotator_user_id": annotation.annotator_user_id,
                "annotator_id": annotation.annotator_id,
                "status": annotation.status,
                "start_sentence_id": annotation.evidence_block.start_sentence_id,
                "end_sentence_id": annotation.evidence_block.end_sentence_id,
                "start_sentence_ordinal": annotation.evidence_block.start_sentence_ordinal,
                "end_sentence_ordinal": annotation.evidence_block.end_sentence_ordinal,
                "labels": annotation.evidence_block.labels or [],
                "note": annotation.evidence_block.note,
            }
            for annotation in annotations
        ],
    }


def adjudicate(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    actor_user_id: int,
    actor_username: str,
    data: EvidenceAdjudicationCreate,
) -> Annotation:
    project = _required_project(db, project_id)
    _required_document_structure(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=data.structure_version_id,
    )
    require_target_version_scope(
        db, project_id=project_id, target_version_id=data.target_version_id
    )
    _lock_evidence_scope(db, data.target_version_id)
    _required_guideline_scope(
        db,
        project_id=project_id,
        guideline_version_id=data.guideline_version_id,
    )
    workspace_kind = project.workspace.kind
    if workspace_kind == "team" and data.solo_gold:
        raise ValidationError("Solo-gold promotion is available only in individual workspaces")
    sources = []
    if data.source_annotation_ids:
        if len(data.source_annotation_ids) != len(set(data.source_annotation_ids)):
            raise ValidationError("source_annotation_ids must be unique")
        sources = (
            db.query(Annotation)
            .filter(Annotation.id.in_(data.source_annotation_ids))
            .order_by(Annotation.id)
            .populate_existing()
            .with_for_update()
            .all()
        )
        source_blocks = {
            block.annotation_id: block
            for block in (
                db.query(EvidenceBlockAnnotation)
                .filter(
                    EvidenceBlockAnnotation.annotation_id.in_(
                        data.source_annotation_ids
                    )
                )
                .populate_existing()
                .with_for_update()
                .all()
            )
        }
        if (
            len(sources) != len(set(data.source_annotation_ids))
            or len(source_blocks) != len(set(data.source_annotation_ids))
        ):
            raise ValidationError("One or more source evidence blocks were not found")
        ineligible_source_ids = [
            source.id
            for source in sources
            if not _is_eligible_adjudication_source(
                db,
                annotation=source,
                project_id=project_id,
                document_id=document_id,
                structure_version_id=data.structure_version_id,
                target_version_id=data.target_version_id,
                guideline_version_id=data.guideline_version_id,
                require_ready_assignment=workspace_kind == "team",
                lock_assignment=workspace_kind == "team",
            )
        ]
        if ineligible_source_ids:
            raise ValidationError(
                "Adjudication source blocks must be matching, human, unlocked, "
                "reviewed, and assignment-eligible; "
                f"ineligible annotation ids: {ineligible_source_ids}"
            )
        by_id = {source.id: source for source in sources}
        sources = [by_id[source_id] for source_id in data.source_annotation_ids]

    source_annotator_ids = {
        source.annotator_user_id
        for source in sources
        if source.annotator_user_id is not None
    }
    if workspace_kind == "team" and len(source_annotator_ids) < 2:
        raise ValidationError(
            "Team adjudication requires reviewed sources from at least two annotators"
        )
    if workspace_kind != "team" and len(source_annotator_ids) < 2:
        if not data.solo_gold:
            raise ValidationError(
                "Individual-workspace promotion requires solo_gold=true"
            )
        if source_annotator_ids and source_annotator_ids != {actor_user_id}:
            raise ValidationError("Solo-gold sources must belong to the acting owner")
    if data.solo_gold and len(source_annotator_ids) > 1:
        raise ValidationError("solo_gold cannot combine multiple annotators")

    if data.strategy == "custom":
        if data.start_sentence_id is None or data.end_sentence_id is None:
            raise ValidationError("Custom adjudication requires sentence boundaries")
        start_sentence_id = data.start_sentence_id
        end_sentence_id = data.end_sentence_id
    else:
        if not sources:
            raise ValidationError("Adjudication strategy requires source_annotation_ids")
        if data.strategy == "a":
            chosen = sources[0]
            start_sentence_id = chosen.evidence_block.start_sentence_id
            end_sentence_id = chosen.evidence_block.end_sentence_id
        elif data.strategy == "b":
            if len(sources) < 2:
                raise ValidationError("Strategy 'b' requires at least two source blocks")
            chosen = sources[1]
            start_sentence_id = chosen.evidence_block.start_sentence_id
            end_sentence_id = chosen.evidence_block.end_sentence_id
        elif data.strategy == "union":
            chosen_start = min(
                sources, key=lambda source: source.evidence_block.start_sentence_ordinal
            )
            chosen_end = max(
                sources, key=lambda source: source.evidence_block.end_sentence_ordinal
            )
            start_sentence_id = chosen_start.evidence_block.start_sentence_id
            end_sentence_id = chosen_end.evidence_block.end_sentence_id
        else:
            chosen_start = max(
                sources, key=lambda source: source.evidence_block.start_sentence_ordinal
            )
            chosen_end = min(
                sources, key=lambda source: source.evidence_block.end_sentence_ordinal
            )
            if (
                chosen_start.evidence_block.start_sentence_ordinal
                > chosen_end.evidence_block.end_sentence_ordinal
            ):
                raise ValidationError("Source blocks have an empty intersection")
            start_sentence_id = chosen_start.evidence_block.start_sentence_id
            end_sentence_id = chosen_end.evidence_block.end_sentence_id

    adjudicated_start, adjudicated_end = _sentence_range(
        db,
        structure_version_id=data.structure_version_id,
        start_sentence_id=start_sentence_id,
        end_sentence_id=end_sentence_id,
    )
    if data.strategy == "custom":
        reviewers = source_annotator_ids or {actor_user_id}
        uncovered_reviewers = [
            reviewer_user_id
            for reviewer_user_id in sorted(reviewers)
            if not _coverage_contains(
                db,
                project_id=project_id,
                document_id=document_id,
                structure_version_id=data.structure_version_id,
                target_version_id=data.target_version_id,
                guideline_version_id=data.guideline_version_id,
                reviewer_user_id=reviewer_user_id,
                start_ordinal=adjudicated_start.ordinal,
                end_ordinal=adjudicated_end.ordinal,
            )
        ]
        if uncovered_reviewers:
            raise ValidationError(
                "Custom adjudication must be inside reviewed coverage for every "
                f"source annotator; uncovered user ids: {uncovered_reviewers}"
            )

    payload = EvidenceBlockPayloadV1(
        structure_version_id=data.structure_version_id,
        target_version_id=data.target_version_id,
        start_sentence_id=start_sentence_id,
        end_sentence_id=end_sentence_id,
        labels=data.labels,
        note=data.note,
    )
    annotation = _create_block_internal(
        db,
        {
            "project_id": project_id,
            "document_id": document_id,
            "annotator_user_id": actor_user_id,
            "annotator_id": actor_username,
            "guideline_version_id": data.guideline_version_id,
            "evidence": {},
            "attributes": {
                "adjudication": {
                    "strategy": data.strategy,
                    "source_annotation_ids": data.source_annotation_ids,
                    **(
                        {
                            "source_revisions": {
                                str(source.id): source.evidence_block.revision
                                for source in sources
                            }
                        }
                        if sources
                        else {}
                    ),
                    "solo_gold": data.solo_gold,
                }
            },
        },
        payload,
        actor_user_id=actor_user_id,
        source="human",
        status="gold",
        command_group_key=uuid4().hex,
        operation="adjudicate",
    )
    annotation.evidence_block.locked = True
    db.commit()
    db.refresh(annotation)
    return annotation


def command_rows(
    db: Session,
    command_group_key: str,
    *,
    lock: bool = False,
) -> list[EvidenceBlockRevision]:
    query = db.query(EvidenceBlockRevision).filter(
        EvidenceBlockRevision.command_group_key == command_group_key
    )
    if lock:
        query = query.populate_existing().with_for_update()
    rows = query.order_by(EvidenceBlockRevision.id.asc()).all()
    if not rows:
        raise NotFoundError("Evidence command not found")
    return rows


def list_commands(
    db: Session,
    *,
    project_id: int,
    document_id: int | None = None,
    target_version_id: int | None = None,
    structure_version_id: int | None = None,
    guideline_version_id: int | None = None,
    actor_user_id: int | None = None,
) -> list[dict]:
    query = db.query(EvidenceBlockRevision).filter(
        EvidenceBlockRevision.project_id == project_id,
        ~EvidenceBlockRevision.operation.like("adjudicate%"),
        ~EvidenceBlockRevision.operation.like(f"{_INVALIDATED_COMMAND_PREFIX}%"),
    )
    if document_id is not None:
        query = query.filter(EvidenceBlockRevision.document_id == document_id)
    if target_version_id is not None:
        query = query.filter(
            EvidenceBlockRevision.target_version_id == target_version_id
        )
    if structure_version_id is not None:
        query = query.filter(
            EvidenceBlockRevision.structure_version_id == structure_version_id
        )
    if actor_user_id is not None:
        query = query.filter(EvidenceBlockRevision.actor_user_id == actor_user_id)
    rows = query.order_by(
        EvidenceBlockRevision.created_at.desc(), EvidenceBlockRevision.id.desc()
    ).all()
    command_groups: dict[str, list[EvidenceBlockRevision]] = {}
    for row in rows:
        command_groups.setdefault(row.command_group_key, []).append(row)
    summaries = []
    for group_rows in command_groups.values():
        guideline_id = command_guideline_version_id(group_rows)
        if (
            guideline_version_id is not None
            and guideline_id != guideline_version_id
        ):
            continue
        row = group_rows[0]
        summaries.append(
            {
                "command_group_key": row.command_group_key,
                "operation": row.operation.split("_", 1)[0],
                "status": "undone" if row.is_undone else "applied",
                "project_id": row.project_id,
                "document_id": row.document_id,
                "target_version_id": row.target_version_id,
                "structure_version_id": row.structure_version_id,
                "guideline_version_id": guideline_id,
                "actor_user_id": row.actor_user_id,
                "created_at": row.created_at,
            }
        )
    return summaries


def _apply_block_state(db: Session, state: dict) -> Annotation:
    annotation_id = state["annotation_id"]
    annotation = db.get(Annotation, annotation_id)
    if annotation is None:
        annotation = Annotation(id=annotation_id)
        db.add(annotation)
    elif (
        annotation.annotation_type != "evidence_block"
        or annotation.evidence_block is None
    ):
        raise ConflictError(
            "Evidence command cannot overwrite an unrelated annotation"
        )
    annotation.project_id = state["project_id"]
    annotation.document_id = state["document_id"]
    annotation.annotation_type = "evidence_block"
    annotation.label = "evidence_block"
    annotation.start_offset = state["start_offset"]
    annotation.end_offset = state["end_offset"]
    annotation.text_span = state["text_span"]
    annotation.source = state["source"]
    annotation.status = state["status"]
    annotation.confidence = state["confidence"]
    annotation.annotator_user_id = state["annotator_user_id"]
    annotation.annotator_id = state["annotator_id"]
    annotation.model_checkpoint_id = state["model_checkpoint_id"]
    annotation.guideline_version_id = state["guideline_version_id"]
    annotation.structure_version_id = state["structure_version_id"]
    annotation.head_annotation_id = None
    annotation.tail_annotation_id = None
    annotation.evidence = state["evidence"]
    annotation.attributes = state["attributes"]
    block = annotation.evidence_block
    if block is None:
        block = EvidenceBlockAnnotation(annotation_id=annotation_id)
        annotation.evidence_block = block
    block.structure_version_id = state["structure_version_id"]
    block.target_version_id = state["target_version_id"]
    block.start_sentence_id = state["start_sentence_id"]
    block.end_sentence_id = state["end_sentence_id"]
    block.start_sentence_ordinal = state["start_sentence_ordinal"]
    block.end_sentence_ordinal = state["end_sentence_ordinal"]
    block.labels = state["labels"]
    block.note = state["note"]
    block.boundary_policy = state["boundary_policy"]
    block.revision = state["revision"]
    block.locked = state["locked"]
    block.last_command_group_key = state.get("last_command_group_key")
    return annotation


def _command_scope(
    rows: list[EvidenceBlockRevision],
) -> tuple[int, int, int, int, int | None, int | None]:
    scopes = {
        (
            row.project_id,
            row.document_id,
            row.structure_version_id,
            row.target_version_id,
            row.actor_user_id,
        )
        for row in rows
    }
    if len(scopes) != 1:
        raise ConflictError("Evidence command spans multiple mutation scopes")
    project_id, document_id, structure_id, target_id, actor_id = next(iter(scopes))
    return (
        project_id,
        document_id,
        structure_id,
        target_id,
        command_guideline_version_id(rows),
        actor_id,
    )


def _scope_command_groups(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    guideline_version_id: int | None,
    actor_user_id: int | None,
) -> list[list[EvidenceBlockRevision]]:
    rows = (
        db.query(EvidenceBlockRevision)
        .filter(
            EvidenceBlockRevision.project_id == project_id,
            EvidenceBlockRevision.document_id == document_id,
            EvidenceBlockRevision.structure_version_id == structure_version_id,
            EvidenceBlockRevision.target_version_id == target_version_id,
            EvidenceBlockRevision.actor_user_id == actor_user_id,
            ~EvidenceBlockRevision.operation.like("adjudicate%"),
            ~EvidenceBlockRevision.operation.like(
                f"{_INVALIDATED_COMMAND_PREFIX}%"
            ),
        )
        .order_by(EvidenceBlockRevision.id.asc())
        .populate_existing()
        .with_for_update()
        .all()
    )
    grouped: dict[str, list[EvidenceBlockRevision]] = {}
    for row in rows:
        grouped.setdefault(row.command_group_key, []).append(row)
    return [
        group_rows
        for group_rows in grouped.values()
        if command_guideline_version_id(group_rows) == guideline_version_id
    ]


def _assert_command_stack_position(
    groups: list[list[EvidenceBlockRevision]],
    *,
    command_group_key: str,
    undo: bool,
) -> None:
    group_states: list[tuple[str, bool]] = []
    undone_seen = False
    for group_rows in groups:
        states = {row.is_undone for row in group_rows}
        if len(states) != 1:
            raise ConflictError("Evidence command has inconsistent revision state")
        is_undone = next(iter(states))
        if is_undone:
            undone_seen = True
        elif undone_seen:
            raise ConflictError("Evidence command history is not a valid undo stack")
        group_states.append((group_rows[0].command_group_key, is_undone))

    requested = next(
        (item for item in group_states if item[0] == command_group_key), None
    )
    if requested is None:
        raise ConflictError("Evidence command is no longer available for replay")
    applied = [key for key, is_undone in group_states if not is_undone]
    undone = [key for key, is_undone in group_states if is_undone]
    if undo and (not applied or applied[-1] != command_group_key):
        raise ConflictError("Only the newest applied evidence command can be undone")
    if not undo and (not undone or undone[0] != command_group_key):
        raise ConflictError("Only the next undone evidence command can be redone")


def _locked_annotation(db: Session, annotation_id: int) -> Annotation | None:
    annotation = (
        db.query(Annotation)
        .filter(Annotation.id == annotation_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if annotation is not None and annotation.annotation_type == "evidence_block":
        # Refresh the one-to-one payload as well as the parent. Querying the
        # parent with populate_existing does not refresh an already-loaded
        # relationship in SQLAlchemy's identity map.
        (
            db.query(EvidenceBlockAnnotation)
            .filter(EvidenceBlockAnnotation.annotation_id == annotation_id)
            .populate_existing()
            .with_for_update()
            .one_or_none()
        )
    return annotation


def _validate_command_source_states(
    db: Session,
    source_states: list[dict | None],
    desired_states: list[dict | None],
) -> None:
    checked: set[int] = set()
    for source, desired in zip(source_states, desired_states, strict=True):
        state = source or desired
        if state is None:
            continue
        annotation_id = state["annotation_id"]
        if annotation_id in checked:
            continue
        checked.add(annotation_id)
        current = _locked_annotation(db, annotation_id)
        if source is None:
            if current is not None:
                raise ConflictError(
                    "Evidence command source state changed; annotation id is in use"
                )
            continue
        if current is None or current.evidence_block is None:
            raise ConflictError("Evidence command source state no longer exists")
        if _block_state(current, current.evidence_block) != source:
            raise ConflictError("Evidence command source state has changed")


def _reopen_command_coverage(
    db: Session,
    rows: list[EvidenceBlockRevision],
    *,
    actor_user_id: int,
    action: str,
) -> None:
    scopes: dict[
        tuple[int, int, int, int, int | None, int | None], tuple[int, int]
    ] = {}
    for row in rows:
        for state in (row.before, row.after):
            if not state:
                continue
            key = (
                state["project_id"],
                state["document_id"],
                state["structure_version_id"],
                state["target_version_id"],
                state["guideline_version_id"],
                state["annotator_user_id"],
            )
            previous = scopes.get(key)
            start = state["start_sentence_ordinal"]
            end = state["end_sentence_ordinal"]
            scopes[key] = (
                min(previous[0], start) if previous else start,
                max(previous[1], end) if previous else end,
            )
    for scope, (start, end) in scopes.items():
        (
            project_id,
            document_id,
            structure_id,
            target_id,
            guideline_id,
            reviewer_id,
        ) = scope
        _reopen_coverage_ordinals(
            db,
            project_id=project_id,
            document_id=document_id,
            structure_version_id=structure_id,
            target_version_id=target_id,
            guideline_version_id=guideline_id,
            reviewer_user_id=reviewer_id,
            actor_user_id=actor_user_id,
            start_ordinal=start,
            end_ordinal=end,
            reason=f"Evidence command {action}",
            action="boundary_mutation",
        )


def _set_command_state(
    db: Session,
    command_group_key: str,
    *,
    undo: bool,
    actor_user_id: int,
    required_assignment_id: int | None = None,
) -> dict:
    initial_rows = command_rows(db, command_group_key)
    if any(
        row.operation.startswith(_INVALIDATED_COMMAND_PREFIX)
        for row in initial_rows
    ):
        raise ConflictError("Evidence command was invalidated by a newer edit")
    (
        project_id,
        document_id,
        structure_version_id,
        target_version_id,
        guideline_version_id,
        command_actor_user_id,
    ) = _command_scope(initial_rows)
    _lock_evidence_scope(db, target_version_id)
    _lock_actor_assignment_if_present(
        db,
        project_id=project_id,
        document_id=document_id,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
    rows = command_rows(db, command_group_key, lock=True)
    if _command_scope(rows) != (
        project_id,
        document_id,
        structure_version_id,
        target_version_id,
        guideline_version_id,
        command_actor_user_id,
    ):
        raise ConflictError("Evidence command scope changed while replaying it")
    groups = _scope_command_groups(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=structure_version_id,
        target_version_id=target_version_id,
        guideline_version_id=guideline_version_id,
        actor_user_id=command_actor_user_id,
    )
    _assert_command_stack_position(
        groups,
        command_group_key=command_group_key,
        undo=undo,
    )
    if any(
        row.operation.startswith("adjudicate")
        or any(
            state and (state.get("status") == "gold" or state.get("locked") is True)
            for state in (row.before, row.after)
        )
        for row in rows
    ):
        raise ConflictError("Adjudicated gold evidence is immutable and cannot be undone")
    currently_undone = {row.is_undone for row in rows}
    if len(currently_undone) != 1:
        raise ConflictError("Evidence command has inconsistent revision state")
    if undo and True in currently_undone:
        raise ConflictError("Evidence command is already undone")
    if not undo and False in currently_undone:
        raise ConflictError("Evidence command is already applied")

    source_states = [row.after if undo else row.before for row in rows]
    desired_states = [row.before if undo else row.after for row in rows]
    _validate_command_source_states(db, source_states, desired_states)
    source_ids = {state["annotation_id"] for state in source_states if state}
    desired_ids = {state["annotation_id"] for state in desired_states if state}
    from al_medlit.annotation.service import assert_annotations_deletable

    assert_annotations_deletable(db, tuple(source_ids - desired_ids))
    for annotation_id in source_ids - desired_ids:
        annotation = _locked_annotation(db, annotation_id)
        if annotation is None:
            raise ConflictError("Evidence command source state no longer exists")
        db.delete(annotation)
    db.flush()

    affected = []
    seen = set()
    for state in desired_states:
        if not state or state["annotation_id"] in seen:
            continue
        affected.append(_apply_block_state(db, state))
        seen.add(state["annotation_id"])
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(
            "Evidence command cannot restore an annotation because its id is in use"
        ) from exc
    for row in rows:
        row.is_undone = undo
    _reopen_command_coverage(
        db,
        rows,
        actor_user_id=actor_user_id,
        action="undo" if undo else "redo",
    )
    db.commit()
    for annotation in affected:
        db.refresh(annotation)
    return {
        "command_group_key": command_group_key,
        "status": "undone" if undo else "applied",
        "annotations": affected,
    }


def undo_command(
    db: Session,
    command_group_key: str,
    *,
    actor_user_id: int,
    required_assignment_id: int | None = None,
) -> dict:
    return _set_command_state(
        db,
        command_group_key,
        undo=True,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )


def redo_command(
    db: Session,
    command_group_key: str,
    *,
    actor_user_id: int,
    required_assignment_id: int | None = None,
) -> dict:
    return _set_command_state(
        db,
        command_group_key,
        undo=False,
        actor_user_id=actor_user_id,
        required_assignment_id=required_assignment_id,
    )
