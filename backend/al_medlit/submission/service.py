import hashlib
import json
import logging
from uuid import uuid4

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import resource_access_clause
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.corpus.models import Document
from al_medlit.project import service as project_service
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.storage_reclaim import service as storage_reclaim_service
from al_medlit.submission.models import AnnotationSubmission
from al_medlit.submission.schemas import SubmissionCreate
from al_medlit.submission.snapshot import build_snapshot

OPEN_ASSIGNMENT_STATUSES = ("assigned", "in_progress", "blocked")
logger = logging.getLogger(__name__)


def snapshot_annotation_types(data: bytes) -> set[str] | None:
    """Return stored annotation types, or ``None`` for an unverifiable snapshot."""
    try:
        snapshot = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(snapshot, dict):
        return None
    annotations = snapshot.get("annotations")
    if not isinstance(annotations, list):
        return None
    annotation_types: set[str] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            return None
        annotation_type = annotation.get("annotation_type")
        if not isinstance(annotation_type, str):
            return None
        annotation_types.add(annotation_type)
    return annotation_types


def snapshot_checksum_matches(data: bytes, expected_sha256: str) -> bool:
    return hashlib.sha256(data).hexdigest() == expected_sha256


def _clean_annotator_id(annotator_id: str | None) -> str | None:
    if annotator_id is None:
        return None
    cleaned = annotator_id.strip()
    return cleaned or None


def _finalize_assignments(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    annotator_user_id: int,
    assignment_id: int | None = None,
) -> None:
    query = db.query(TaskAssignment).filter(
        TaskAssignment.project_id == project_id,
        TaskAssignment.document_id == document_id,
        TaskAssignment.status.in_(OPEN_ASSIGNMENT_STATUSES),
        TaskAssignment.assignee_user_id == annotator_user_id,
    )
    if assignment_id is not None:
        query = query.filter(TaskAssignment.id == assignment_id)
    for assignment in query.populate_existing().all():
        assignment.status = "submitted"


def create_submission(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    document_id: int,
    data: SubmissionCreate,
) -> AnnotationSubmission:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    document = db.get(Document, document_id)
    if document is None or document.project_id != project_id:
        raise NotFoundError("Document not found in project")

    annotator_user_id = data.annotator_user_id
    annotator_id = _clean_annotator_id(data.annotator_id)
    annotator_user: User | None = None
    if annotator_user_id is not None:
        annotator_user = db.get(User, annotator_user_id)
        if annotator_user is None:
            raise NotFoundError("Annotator user not found")
        annotator_id = annotator_user.username
    elif annotator_id is not None:
        annotator_user = db.query(User).filter(User.username == annotator_id).first()
        if annotator_user is not None:
            annotator_user_id = annotator_user.id

    if data.kind == "submission" and annotator_user_id is None:
        raise ValidationError("annotator_user_id is required when kind is 'submission'")
    if data.kind == "submission" and annotator_user is not None:
        project_service.ensure_personal_project_self_assignments(
            db,
            project_id,
            annotator_user,
            document_ids=[document_id],
            assigned_by_user=annotator_user,
            commit=False,
        )

    assignment: TaskAssignment | None = None
    if data.assignment_id is not None:
        assignment = db.get(TaskAssignment, data.assignment_id)
        if (
            assignment is None
            or assignment.project_id != project_id
            or assignment.document_id != document_id
        ):
            raise ValidationError("assignment_id does not belong to this project document")
        if assignment.assignee_user_id != annotator_user_id:
            raise ValidationError("assignment_id belongs to another annotator")
    elif data.kind == "submission" and annotator_user_id is not None:
        evidence_assignment = (
            db.query(TaskAssignment.id)
            .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
            .filter(
                TaskAssignment.project_id == project_id,
                TaskAssignment.document_id == document_id,
                TaskAssignment.assignee_user_id == annotator_user_id,
                ProjectTask.annotation_type == "evidence_block",
            )
            .first()
        )
        if evidence_assignment is not None:
            raise ValidationError(
                "assignment_id is required when submitting evidence-block work"
            )
        if project.workspace.kind == "team":
            project_has_assignments = (
                db.query(TaskAssignment.id)
                .filter(TaskAssignment.project_id == project_id)
                .first()
                is not None
            )
            actor_has_assignment = (
                db.query(TaskAssignment.id)
                .filter(
                    TaskAssignment.project_id == project_id,
                    TaskAssignment.document_id == document_id,
                    TaskAssignment.assignee_user_id == annotator_user_id,
                )
                .first()
                is not None
            )
            if project_has_assignments and not actor_has_assignment:
                raise ValidationError(
                    "assignment_id is required for assignment-managed team work"
                )
    if assignment is not None and assignment.task.annotation_type == "evidence_block":
        from al_medlit.evidence.service import (
            _lock_evidence_scope,
            assignment_has_full_review_coverage,
        )

        if assignment.target_version_id is not None:
            _lock_evidence_scope(db, assignment.target_version_id)
        assignment = (
            db.query(TaskAssignment)
            .filter(TaskAssignment.id == assignment.id)
            .populate_existing()
            .with_for_update()
            .one()
        )
        if not assignment_has_full_review_coverage(db, assignment):
            raise ValidationError(
                "Evidence assignment requires full reviewed coverage before submission"
            )
    elif assignment is not None:
        assignment = (
            db.query(TaskAssignment)
            .filter(TaskAssignment.id == assignment.id)
            .populate_existing()
            .with_for_update()
            .one()
        )

    if data.kind == "submission" and assignment is not None:
        if assignment.status not in OPEN_ASSIGNMENT_STATUSES:
            raise ConflictError("Task assignment has already reached a final status")
    elif data.kind == "submission" and annotator_user_id is not None:
        actor_assignments = (
            db.query(TaskAssignment)
            .filter(
                TaskAssignment.project_id == project_id,
                TaskAssignment.document_id == document_id,
                TaskAssignment.assignee_user_id == annotator_user_id,
            )
            .order_by(TaskAssignment.id)
            .populate_existing()
            .with_for_update()
            .all()
        )
        if actor_assignments and not any(
            candidate.status in OPEN_ASSIGNMENT_STATUSES
            for candidate in actor_assignments
        ):
            raise ConflictError("All task assignments for this document are finalized")

    snapshot = build_snapshot(
        db,
        project=project,
        document=document,
        annotator_user_id=annotator_user_id,
        annotator_id=annotator_id,
        user=annotator_user,
        assignment=assignment,
    )
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False).encode("utf-8")

    token = uuid4().hex
    submission = AnnotationSubmission(
        project_id=project_id,
        document_id=document_id,
        assignment_id=assignment.id if assignment is not None else None,
        annotator_user_id=annotator_user_id,
        annotator_id=annotator_id,
        kind=data.kind,
        storage_key=(
            f"projects/{project_id}/documents/{document_id}/submission-{token}.json"
        ),
        file_name=f"project-{project_id}-document-{document_id}-{token[:8]}.json",
        content_type="application/json",
        size_bytes=len(payload),
        checksum_sha256=hashlib.sha256(payload).hexdigest(),
        annotation_count=len(snapshot["annotations"]),
        metadata_=data.metadata_,
    )
    db.add(submission)

    if data.kind == "submission":
        _finalize_assignments(
            db,
            project_id=project_id,
            document_id=document_id,
            annotator_user_id=annotator_user_id,
            assignment_id=assignment.id if assignment is not None else None,
        )
        if annotator_user is not None:
            # A personal workspace may have activated a new guideline or
            # structure while the just-submitted round was still open. The
            # pre-submit provisioning pass intentionally preserves that open
            # round; after finalization, provision the now-unblocked active
            # version in the same transaction.
            db.flush()
            project_service.ensure_personal_project_self_assignments(
                db,
                project_id,
                annotator_user,
                document_ids=[document_id],
                assigned_by_user=annotator_user,
                commit=False,
            )

    storage_key = submission.storage_key
    try:
        # Treat the key as potentially published before calling storage: an
        # implementation may persist the object and then raise (for example,
        # after a failed acknowledgement). In every pre-commit failure case,
        # roll back the database unit of work and best-effort remove the key.
        storage.put_bytes(storage_key, payload, content_type="application/json")
        db.commit()
    except Exception:
        db.rollback()
        try:
            storage.delete(storage_key)
        except Exception as storage_error:
            logger.exception(
                "Failed to remove submission object after submission creation failure",
                extra={"storage_key": storage_key},
            )
            storage_reclaim_service.record_orphaned_object(
                db,
                storage_key,
                origin="submission.create_rollback",
                error=storage_error,
            )
        raise
    db.refresh(submission)
    return submission


def list_submissions(
    db: Session,
    project_id: int,
    *,
    document_id: int | None = None,
    annotator_user_id: int | None = None,
    annotator_id: str | None = None,
    kind: str | None = None,
    user: User | None = None,
) -> list[AnnotationSubmission]:
    if db.get(Project, project_id) is None:
        raise NotFoundError("Project not found")
    query = db.query(AnnotationSubmission).filter(
        AnnotationSubmission.project_id == project_id
    )
    if document_id is not None:
        query = query.filter(AnnotationSubmission.document_id == document_id)
    if annotator_user_id is not None:
        query = query.filter(AnnotationSubmission.annotator_user_id == annotator_user_id)
    elif annotator_id is not None:
        query = query.filter(AnnotationSubmission.annotator_id == annotator_id)
    if kind is not None:
        query = query.filter(AnnotationSubmission.kind == kind)
    if user is not None:
        query = query.join(
            Project,
            Project.id == AnnotationSubmission.project_id,
        ).join(
            Document,
            Document.id == AnnotationSubmission.document_id,
        ).filter(
            Document.project_id == AnnotationSubmission.project_id,
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=AnnotationSubmission.project_id,
                document_id=AnnotationSubmission.document_id,
                owner_user_id=AnnotationSubmission.annotator_user_id,
            )
        )
    return query.order_by(AnnotationSubmission.created_at.desc()).all()


def get_submission(db: Session, submission_id: int) -> AnnotationSubmission:
    submission = db.get(AnnotationSubmission, submission_id)
    if submission is None:
        raise NotFoundError("Submission not found")
    return submission


def delete_submission(db: Session, storage: ObjectStorage, submission_id: int) -> None:
    submission = get_submission(db, submission_id)
    storage_key = submission.storage_key
    db.delete(submission)
    db.commit()
    try:
        storage.delete(storage_key)
    except Exception as storage_error:
        # The authoritative database deletion succeeded. Keep the API operation
        # successful and hand the now-unreferenced object to the reclaim sweep
        # so a failed delete cannot leak it.
        logger.exception(
            "Failed to remove unreferenced submission object",
            extra={"storage_key": storage_key},
        )
        storage_reclaim_service.record_orphaned_object(
            db,
            storage_key,
            origin="submission.delete",
            error=storage_error,
        )
