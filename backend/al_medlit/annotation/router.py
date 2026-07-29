from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from al_medlit.annotation import service
from al_medlit.annotation.schemas import (
    AnnotationCorrectionCreate,
    AnnotationCorrectionRead,
    AnnotationCreate,
    AnnotationRead,
    AnnotationUpdate,
    EvidenceCommandResult,
    EvidenceCommandSummary,
)
from al_medlit.annotation.types import AnnotationValidationError
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    MUTABLE_ASSIGNMENT_STATUSES,
    assert_annotation_member_with_assignment,
    assert_annotation_references_access,
    assert_assigned_document_member,
    assert_assigned_document_member_if_exists,
    assert_global_manager_scope,
    assert_guideline_version_member_if_exists,
    assert_project_member,
    assert_project_member_if_exists,
    assert_task_assigned,
    has_assignment_bypass,
    lock_document_resource_for_mutation,
    require_annotation_access,
    require_project_access,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, ValidationError
from al_medlit.corpus.models import Document, DocumentStructureVersion
from al_medlit.evidence import service as evidence_service
from al_medlit.evidence.schemas import EvidenceMergeRequest, EvidenceSplitRequest
from al_medlit.guideline import service as guideline_service
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.project.models import ProjectTask, TaskAssignment

router = APIRouter(prefix="/annotations", tags=["annotations"])


def _assert_annotation_owner(annotation, current_user: User) -> None:
    if annotation is not None and annotation.annotator_user_id != current_user.id:
        raise ForbiddenError(
            "Managers must use comparison, corrections, or adjudication for another "
            "annotator's work"
        )


@router.post("", response_model=AnnotationRead)
def create_annotation(
    payload: AnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_assignment_id: int | None = None
    requested_guideline_version_id = payload.guideline_version_id
    requested_structure_version_id = payload.structure_version_id
    if payload.evidence_block is not None:
        payload.structure_version_id = payload.evidence_block.structure_version_id
    else:
        # Ordinary annotation version provenance is selected from the open
        # assignment round, never from caller-supplied ids.
        payload.structure_version_id = None
        payload.guideline_version_id = None
    if payload.evidence_block is not None and payload.guideline_version_id is None:
        assignment = (
            db.query(TaskAssignment)
            .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
            .filter(
                TaskAssignment.project_id == payload.project_id,
                TaskAssignment.document_id == payload.document_id,
                TaskAssignment.assignee_user_id == current_user.id,
                TaskAssignment.target_version_id == payload.evidence_block.target_version_id,
                TaskAssignment.structure_version_id == payload.evidence_block.structure_version_id,
                TaskAssignment.status.in_(MUTABLE_ASSIGNMENT_STATUSES),
                ProjectTask.annotation_type == "evidence_block",
            )
            .order_by(TaskAssignment.id.desc())
            .first()
        )
        if assignment is not None:
            payload.guideline_version_id = assignment.guideline_version_id
        else:
            active_guideline = guideline_service.get_active_guideline(db, payload.project_id)
            if active_guideline is not None:
                payload.guideline_version_id = active_guideline.id
    member = assert_project_member_if_exists(db, current_user, payload.project_id)
    assert_assigned_document_member_if_exists(db, current_user, payload.document_id)
    document = db.get(Document, payload.document_id)
    if member is not None and document is not None and document.project_id == payload.project_id:
        member = lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=payload.project_id,
            document_id=payload.document_id,
            lock_assignment=False,
        )
        if payload.evidence_block is None:
            if requested_guideline_version_id is not None:
                requested_guideline = db.get(
                    GuidelineVersion,
                    requested_guideline_version_id,
                )
                if (
                    requested_guideline is None
                    or requested_guideline.project_id != payload.project_id
                ):
                    raise ValidationError(
                        f"GuidelineVersion {requested_guideline_version_id} does not "
                        f"belong to project {payload.project_id}"
                    )
                assert_guideline_version_member_if_exists(
                    db,
                    current_user,
                    requested_guideline_version_id,
                )
            if requested_structure_version_id is not None:
                requested_structure = db.get(
                    DocumentStructureVersion,
                    requested_structure_version_id,
                )
                if (
                    requested_structure is None
                    or requested_structure.document_id != payload.document_id
                ):
                    raise ValidationError(
                        "structure_version_id must belong to the annotation document"
                    )
        assignment = assert_task_assigned(
            db,
            current_user,
            member,
            project_id=payload.project_id,
            document_id=payload.document_id,
            annotation_type=payload.annotation_type,
            target_version_id=(
                payload.evidence_block.target_version_id
                if payload.evidence_block is not None
                else None
            ),
            structure_version_id=payload.structure_version_id,
            guideline_version_id=payload.guideline_version_id,
            require_mutable_assignment=True,
        )
        if (
            payload.evidence_block is not None
            and assignment is not None
            and not has_assignment_bypass(current_user, member)
        ):
            required_assignment_id = assignment.id
        if assignment is not None:
            payload.guideline_version_id = assignment.guideline_version_id
            payload.structure_version_id = assignment.structure_version_id
        elif payload.guideline_version_id is None:
            active_guideline = guideline_service.get_active_guideline(
                db,
                payload.project_id,
            )
            if active_guideline is not None:
                payload.guideline_version_id = active_guideline.id
        if assignment is None and payload.structure_version_id is None:
            payload.structure_version_id = document.active_structure_version_id
        if payload.annotation_type == "relation":
            assert_annotation_references_access(
                db,
                current_user,
                project_id=payload.project_id,
                document_id=payload.document_id,
                annotation_ids=(payload.head_annotation_id, payload.tail_annotation_id),
            )
    if payload.guideline_version_id is not None:
        assert_guideline_version_member_if_exists(
            db,
            current_user,
            payload.guideline_version_id,
        )
    payload.annotator_user_id = current_user.id
    payload.annotator_id = current_user.username
    payload.source = "human"
    payload.status = "draft"
    payload.confidence = None
    payload.model_checkpoint_id = None
    try:
        return service.create_annotation(
            db,
            payload,
            required_assignment_id=required_assignment_id,
        )
    except AnnotationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get("", response_model=list[AnnotationRead])
def list_annotations(
    project_id: int | None = Query(None),
    document_id: int | None = Query(None),
    annotator_id: str | None = Query(None),
    target_version_id: int | None = Query(None),
    structure_version_id: int | None = Query(None),
    guideline_version_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = None
    if project_id is not None:
        member = assert_project_member(db, current_user, project_id)
    if document_id is not None:
        assert_assigned_document_member(db, current_user, document_id)
    if annotator_id is not None and annotator_id != current_user.username:
        if member is not None:
            if not has_assignment_bypass(current_user, member):
                raise ForbiddenError("Cannot view another annotator's annotations")
        else:
            assert_global_manager_scope(db, current_user)
    return service.list_annotations(
        db,
        project_id=project_id,
        document_id=document_id,
        annotator_id=annotator_id,
        user=current_user,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
        filter_guideline_version=guideline_version_id is not None,
    )


@router.post("/corrections", response_model=AnnotationCorrectionRead)
def create_correction(
    payload: AnnotationCorrectionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    member = assert_project_member_if_exists(db, current_user, payload.project_id)
    assert_assigned_document_member_if_exists(db, current_user, payload.document_id)
    document = db.get(Document, payload.document_id)
    if member is not None and document is not None and document.project_id == payload.project_id:
        lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=payload.project_id,
            document_id=payload.document_id,
        )
        assert_annotation_references_access(
            db,
            current_user,
            project_id=payload.project_id,
            document_id=payload.document_id,
            annotation_ids=(
                payload.original_annotation_id,
                payload.corrected_annotation_id,
            ),
        )
    # This public endpoint records user feedback. Adjudication/model provenance
    # is reserved for trusted workflow services and must not be caller-forgeable.
    payload.correction_source = "human"
    try:
        return service.create_correction(
            db,
            payload,
            created_by_user_id=current_user.id,
        )
    except AnnotationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc


@router.get("/corrections", response_model=list[AnnotationCorrectionRead])
def list_corrections(
    project_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if project_id is not None:
        assert_project_member(db, current_user, project_id)
    return service.list_corrections(db, project_id, user=current_user)


@router.get(
    "/evidence-blocks/commands",
    response_model=list[EvidenceCommandSummary],
)
def list_evidence_commands(
    project_id: int = Query(...),
    document_id: int | None = Query(None),
    target_version_id: int | None = Query(None),
    structure_version_id: int | None = Query(None),
    guideline_version_id: int | None = Query(None),
    member=Depends(require_project_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return evidence_service.list_commands(
        db,
        project_id=project_id,
        document_id=document_id,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
        actor_user_id=(None if has_assignment_bypass(current_user, member) else current_user.id),
    )


def _authorize_evidence_command(
    db: Session,
    current_user: User,
    command_group_key: str,
) -> int | None:
    rows = evidence_service.command_rows(db, command_group_key)
    member = lock_document_resource_for_mutation(
        db,
        current_user,
        project_id=rows[0].project_id,
        document_id=rows[0].document_id,
        lock_assignment=False,
    )
    if any(row.actor_user_id != current_user.id for row in rows) or any(
        state.get("annotator_user_id") != current_user.id
        for row in rows
        for state in (row.before, row.after)
        if state is not None
    ):
        raise ForbiddenError("Evidence command belongs to another annotator")
    assignment = assert_task_assigned(
        db,
        current_user,
        member,
        project_id=rows[0].project_id,
        document_id=rows[0].document_id,
        annotation_type="evidence_block",
        target_version_id=rows[0].target_version_id,
        structure_version_id=rows[0].structure_version_id,
        guideline_version_id=evidence_service.command_guideline_version_id(rows),
        require_mutable_assignment=True,
    )
    if assignment is not None and not has_assignment_bypass(current_user, member):
        return assignment.id
    return None


@router.post(
    "/evidence-blocks/commands/{command_group_key}/undo",
    response_model=EvidenceCommandResult,
)
def undo_evidence_command(
    command_group_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_assignment_id = _authorize_evidence_command(db, current_user, command_group_key)
    return evidence_service.undo_command(
        db,
        command_group_key,
        actor_user_id=current_user.id,
        required_assignment_id=required_assignment_id,
    )


@router.post(
    "/evidence-blocks/commands/{command_group_key}/redo",
    response_model=EvidenceCommandResult,
)
def redo_evidence_command(
    command_group_key: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_assignment_id = _authorize_evidence_command(db, current_user, command_group_key)
    return evidence_service.redo_command(
        db,
        command_group_key,
        actor_user_id=current_user.id,
        required_assignment_id=required_assignment_id,
    )


@router.post("/evidence-blocks/merge", response_model=AnnotationRead)
def merge_evidence_blocks(
    payload: EvidenceMergeRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    first = service.get_annotation(db, payload.annotation_ids[0])
    if first is not None:
        lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=first.project_id,
            document_id=first.document_id,
            lock_assignment=False,
        )
    required_assignment_ids: set[int] = set()
    for annotation_id in payload.annotation_ids:
        member, assignment = assert_annotation_member_with_assignment(
            db,
            current_user,
            annotation_id,
            require_mutable_assignment=True,
        )
        if assignment is not None and not has_assignment_bypass(current_user, member):
            required_assignment_ids.add(assignment.id)
        _assert_annotation_owner(service.get_annotation(db, annotation_id), current_user)
    if len(required_assignment_ids) > 1:
        raise ValidationError("Evidence blocks span multiple assignments")
    return evidence_service.merge_blocks(
        db,
        payload,
        actor_user_id=current_user.id,
        required_assignment_id=next(iter(required_assignment_ids), None),
    )


@router.post("/{annotation_id}/split", response_model=list[AnnotationRead])
def split_evidence_block(
    annotation_id: int,
    payload: EvidenceSplitRequest,
    _member=Depends(require_annotation_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = service.get_annotation(db, annotation_id)
    if existing is not None:
        lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=existing.project_id,
            document_id=existing.document_id,
            lock_assignment=False,
        )
    member, assignment = assert_annotation_member_with_assignment(
        db,
        current_user,
        annotation_id,
        require_mutable_assignment=True,
    )
    _assert_annotation_owner(service.get_annotation(db, annotation_id), current_user)
    return evidence_service.split_block(
        db,
        annotation_id,
        payload,
        actor_user_id=current_user.id,
        required_assignment_id=(
            assignment.id
            if assignment is not None and not has_assignment_bypass(current_user, member)
            else None
        ),
    )


@router.get("/{annotation_id}", response_model=AnnotationRead)
def get_annotation(
    annotation_id: int,
    _member=Depends(require_annotation_access()),
    db: Session = Depends(get_db),
):
    annotation = service.get_annotation(db, annotation_id)
    if annotation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found")
    return annotation


@router.patch("/{annotation_id}", response_model=AnnotationRead)
def update_annotation(
    annotation_id: int,
    payload: AnnotationUpdate,
    _member=Depends(require_annotation_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_assignment_id: int | None = None
    existing = service.get_annotation(db, annotation_id)
    if existing is not None:
        lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=existing.project_id,
            document_id=existing.document_id,
            lock_assignment=False,
        )
        member, assignment = assert_annotation_member_with_assignment(
            db,
            current_user,
            annotation_id,
            require_mutable_assignment=True,
        )
        if (
            existing.annotation_type == "evidence_block"
            and assignment is not None
            and not has_assignment_bypass(current_user, member)
        ):
            required_assignment_id = assignment.id
        _assert_annotation_owner(existing, current_user)
        if existing.source != "human" or existing.status == "gold":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Model and gold annotations are read-only",
            )
        if payload.status == "gold":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Gold annotations can only be created by adjudication",
            )
        if existing.annotation_type == "relation":
            head_id = (
                payload.head_annotation_id
                if "head_annotation_id" in payload.model_fields_set
                else existing.head_annotation_id
            )
            tail_id = (
                payload.tail_annotation_id
                if "tail_annotation_id" in payload.model_fields_set
                else existing.tail_annotation_id
            )
            assert_annotation_references_access(
                db,
                current_user,
                project_id=existing.project_id,
                document_id=existing.document_id,
                annotation_ids=(head_id, tail_id),
            )
    try:
        annotation = service.update_annotation(
            db,
            annotation_id,
            payload,
            actor_user_id=current_user.id,
            required_assignment_id=required_assignment_id,
        )
    except AnnotationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    if annotation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found")
    return annotation


@router.delete("/{annotation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_annotation(
    annotation_id: int,
    expected_revision: int | None = Query(None, ge=1),
    _member=Depends(require_annotation_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    required_assignment_id: int | None = None
    existing = service.get_annotation(db, annotation_id)
    if existing is not None:
        lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=existing.project_id,
            document_id=existing.document_id,
            lock_assignment=False,
        )
        member, assignment = assert_annotation_member_with_assignment(
            db,
            current_user,
            annotation_id,
            require_mutable_assignment=True,
        )
        if (
            existing.annotation_type == "evidence_block"
            and assignment is not None
            and not has_assignment_bypass(current_user, member)
        ):
            required_assignment_id = assignment.id
        _assert_annotation_owner(existing, current_user)
    if existing is not None and (existing.source != "human" or existing.status == "gold"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Model and gold annotations are read-only",
        )
    try:
        deleted = service.delete_annotation(
            db,
            annotation_id,
            expected_revision=expected_revision,
            actor_user_id=current_user.id,
            required_assignment_id=required_assignment_id,
        )
    except service.AnnotationInUseError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Annotation not found")
