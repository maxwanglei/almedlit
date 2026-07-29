from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.annotation.schemas import AnnotationRead
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_document_resource_access,
    assert_task_assigned,
    has_assignment_bypass,
    lock_document_resource_for_mutation,
    lock_project_member_for_mutation,
    require_project_access,
)
from al_medlit.core.database import get_db
from al_medlit.evidence import service
from al_medlit.evidence.schemas import (
    EvidenceAdjudicationCreate,
    EvidenceAdjudicationRead,
    EvidenceReviewCoverageRead,
    EvidenceReviewIntervalRequest,
    EvidenceTargetActivate,
    EvidenceTargetCreate,
    EvidenceTargetRead,
    EvidenceTargetVersionCreate,
    EvidenceTargetVersionRead,
)
from al_medlit.guideline import service as guideline_service
from al_medlit.project import service as project_service
from al_medlit.project.models import ProjectTask, TaskAssignment

router = APIRouter(prefix="/projects", tags=["evidence-blocks"])


@router.get(
    "/{project_id}/evidence-targets",
    response_model=list[EvidenceTargetRead],
)
def list_evidence_targets(
    project_id: int,
    _member=Depends(require_project_access()),
    db: Session = Depends(get_db),
):
    return service.list_targets(db, project_id)


@router.post(
    "/{project_id}/evidence-targets",
    response_model=EvidenceTargetRead,
)
def create_evidence_target(
    project_id: int,
    payload: EvidenceTargetCreate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    return service.create_target(
        db,
        project_id,
        payload,
        actor_user_id=current_user.id,
    )


@router.post(
    "/{project_id}/evidence-targets/{target_id}/versions",
    response_model=EvidenceTargetVersionRead,
)
def create_evidence_target_version(
    project_id: int,
    target_id: int,
    payload: EvidenceTargetVersionCreate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    return service.create_target_version(
        db,
        project_id,
        target_id,
        payload,
        actor_user_id=current_user.id,
    )


@router.post(
    "/{project_id}/evidence-targets/{target_id}/activate",
    response_model=EvidenceTargetRead,
)
def activate_evidence_target(
    project_id: int,
    target_id: int,
    payload: EvidenceTargetActivate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    target = service.activate_target(db, project_id, target_id, payload.version_id)
    project_service.ensure_personal_project_self_assignments(
        db,
        project_id,
        current_user,
        task_ids=[target.task_id],
        assigned_by_user=current_user,
    )
    return target


@router.post(
    "/{project_id}/evidence-targets/{target_id}/deactivate",
    response_model=EvidenceTargetRead,
)
def deactivate_evidence_target(
    project_id: int,
    target_id: int,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="manager",
    )
    return service.deactivate_target(db, project_id, target_id)


def _authorize_review_scope(
    db: Session,
    current_user: User,
    *,
    project_id: int,
    document_id: int,
    target_version_id: int,
    structure_version_id: int,
    guideline_version_id: int | None,
    require_mutable_assignment: bool = False,
) -> tuple[int | None, int | None]:
    if require_mutable_assignment:
        member = lock_document_resource_for_mutation(
            db,
            current_user,
            project_id=project_id,
            document_id=document_id,
            lock_assignment=False,
        )
    else:
        member = assert_document_resource_access(
            db,
            current_user,
            project_id=project_id,
            document_id=document_id,
        )
    assignment = assert_task_assigned(
        db,
        current_user,
        member,
        project_id=project_id,
        document_id=document_id,
        annotation_type="evidence_block",
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
        require_mutable_assignment=require_mutable_assignment,
    )
    if assignment is not None:
        return (
            assignment.guideline_version_id,
            assignment.id
            if require_mutable_assignment and not has_assignment_bypass(current_user, member)
            else None,
        )

    # Managers retain assignment-bypass behavior, but when they also have an
    # exact assignment its immutable pin is still the least surprising scope.
    manager_assignment = (
        db.query(TaskAssignment)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.document_id == document_id,
            TaskAssignment.assignee_user_id == current_user.id,
            TaskAssignment.target_version_id == target_version_id,
            TaskAssignment.structure_version_id == structure_version_id,
            ProjectTask.annotation_type == "evidence_block",
        )
        .first()
    )
    if manager_assignment is not None:
        if (
            guideline_version_id is not None
            and manager_assignment.guideline_version_id != guideline_version_id
        ):
            return guideline_version_id, None
        return manager_assignment.guideline_version_id, None
    if guideline_version_id is not None:
        return guideline_version_id, None
    active_guideline = guideline_service.get_active_guideline(db, project_id)
    return (active_guideline.id if active_guideline is not None else None), None


@router.get(
    "/{project_id}/documents/{document_id}/evidence-review-coverage",
    response_model=EvidenceReviewCoverageRead,
)
def get_evidence_review_coverage(
    project_id: int,
    document_id: int,
    target_version_id: int = Query(...),
    structure_version_id: int = Query(...),
    guideline_version_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resolved_guideline_version_id, _required_assignment_id = _authorize_review_scope(
        db,
        current_user,
        project_id=project_id,
        document_id=document_id,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
    )
    return service.get_review_coverage(
        db,
        project_id=project_id,
        document_id=document_id,
        structure_version_id=structure_version_id,
        target_version_id=target_version_id,
        guideline_version_id=resolved_guideline_version_id,
        reviewer_user_id=current_user.id,
    )


@router.post(
    "/{project_id}/documents/{document_id}/evidence-review-coverage/mark-reviewed",
    response_model=EvidenceReviewCoverageRead,
)
def mark_evidence_reviewed(
    project_id: int,
    document_id: int,
    payload: EvidenceReviewIntervalRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payload.guideline_version_id, required_assignment_id = _authorize_review_scope(
        db,
        current_user,
        project_id=project_id,
        document_id=document_id,
        target_version_id=payload.target_version_id,
        structure_version_id=payload.structure_version_id,
        guideline_version_id=payload.guideline_version_id,
        require_mutable_assignment=True,
    )
    return service.mark_reviewed(
        db,
        project_id=project_id,
        document_id=document_id,
        reviewer_user_id=current_user.id,
        data=payload,
        required_assignment_id=required_assignment_id,
    )


@router.post(
    "/{project_id}/documents/{document_id}/evidence-review-coverage/reopen",
    response_model=EvidenceReviewCoverageRead,
)
def reopen_evidence_reviewed(
    project_id: int,
    document_id: int,
    payload: EvidenceReviewIntervalRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payload.guideline_version_id, required_assignment_id = _authorize_review_scope(
        db,
        current_user,
        project_id=project_id,
        document_id=document_id,
        target_version_id=payload.target_version_id,
        structure_version_id=payload.structure_version_id,
        guideline_version_id=payload.guideline_version_id,
        require_mutable_assignment=True,
    )
    return service.reopen_reviewed(
        db,
        project_id=project_id,
        document_id=document_id,
        reviewer_user_id=current_user.id,
        data=payload,
        required_assignment_id=required_assignment_id,
    )


@router.get(
    "/{project_id}/documents/{document_id}/evidence-adjudication",
    response_model=EvidenceAdjudicationRead,
)
def get_evidence_adjudication(
    project_id: int,
    document_id: int,
    target_version_id: int = Query(...),
    structure_version_id: int = Query(...),
    guideline_version_id: int = Query(...),
    annotator_user_ids: list[int] | None = Query(None),
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
):
    return service.comparison(
        db,
        project_id=project_id,
        document_id=document_id,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
        annotator_user_ids=annotator_user_ids,
    )


@router.post(
    "/{project_id}/documents/{document_id}/evidence-adjudication",
    response_model=AnnotationRead,
    status_code=status.HTTP_201_CREATED,
)
def create_evidence_adjudication(
    project_id: int,
    document_id: int,
    payload: EvidenceAdjudicationCreate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_document_resource_for_mutation(
        db,
        current_user,
        project_id=project_id,
        document_id=document_id,
        min_role="manager",
        lock_assignment=False,
    )
    return service.adjudicate(
        db,
        project_id=project_id,
        document_id=document_id,
        actor_user_id=current_user.id,
        actor_username=current_user.username,
        data=payload,
    )
