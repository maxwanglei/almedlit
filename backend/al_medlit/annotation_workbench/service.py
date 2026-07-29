from sqlalchemy.orm import Session

from al_medlit.annotation import service as annotation_service
from al_medlit.annotation.schemas import AnnotationRead
from al_medlit.annotation.types import REGISTRY as ANNOTATION_TYPE_REGISTRY
from al_medlit.annotation_workbench.schemas import (
    AnnotationTypeSpecRead,
    AnnotationWorkbenchRead,
    AnnotationWorkbenchTaskRead,
)
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_document_assigned, has_assignment_bypass
from al_medlit.core.exceptions import NotFoundError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.schemas import DocumentRead
from al_medlit.guideline import service as guideline_service
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.guideline.schemas import GuidelineVersionRead
from al_medlit.project import service as project_service
from al_medlit.project.models import TaskAssignment
from al_medlit.project.schemas import ProjectRead, ProjectTaskRead, TaskAssignmentRead
from al_medlit.workspace.models import WorkspaceMember


def _annotation_type_spec(annotation_type: str) -> AnnotationTypeSpecRead:
    spec = ANNOTATION_TYPE_REGISTRY[annotation_type]
    return AnnotationTypeSpecRead(
        name=spec.name,
        requires_span=spec.requires_span,
        requires_head_tail=spec.requires_head_tail,
        description=spec.description,
        selection_mode=spec.selection_mode,
        renderer_key=spec.renderer_key,
        relation_endpoint_allowed=spec.relation_endpoint_allowed,
        handler_key=spec.handler_key,
    )


def get_annotation_workbench(
    db: Session,
    document_id: int,
    *,
    current_user: User,
    member: WorkspaceMember,
) -> AnnotationWorkbenchRead:
    document = corpus_service.get_document(db, document_id)
    if document is None:
        raise NotFoundError("Document not found")

    project = project_service.get_project(db, document.project_id)
    if project is None:
        raise NotFoundError("Project not found")
    assert_document_assigned(
        db,
        current_user,
        member,
        project_id=project.id,
        document_id=document.id,
    )

    assignments = project_service.list_task_assignments(
        db,
        project.id,
        document_id=document.id,
        # This is the current user's editing workbench. Managers compare or
        # adjudicate other annotators in the dedicated manager workflows; mixing
        # every assignee here overlays foreign annotations as if they were local.
        assignee_user_id=current_user.id,
    )
    active_guideline = guideline_service.get_active_guideline(db, project.id)
    project_tasks = project_service.list_project_tasks(
        db,
        project.id,
        enabled_only=True,
    )
    if not has_assignment_bypass(current_user, member):
        project_has_assignments = (
            db.query(TaskAssignment.id)
            .filter(TaskAssignment.project_id == project.id)
            .first()
            is not None
        )
        if project_has_assignments:
            assigned_task_ids = {assignment.task_id for assignment in assignments}
            project_tasks = [
                task for task in project_tasks if task.id in assigned_task_ids
            ]
    visible_task_ids = {task.id for task in project_tasks}
    assignments = [
        assignment
        for assignment in assignments
        if assignment.task_id in visible_task_ids
    ]
    pinned_guideline_ids = {
        assignment.guideline_version_id
        for assignment in assignments
        if assignment.guideline_version_id is not None
    }
    guideline_versions_by_id = {
        guideline.id: GuidelineVersionRead.model_validate(guideline)
        for guideline in (
            db.query(GuidelineVersion)
            .filter(
                GuidelineVersion.project_id == project.id,
                GuidelineVersion.id.in_(pinned_guideline_ids),
            )
            .all()
            if pinned_guideline_ids
            else []
        )
    }
    annotations = annotation_service.list_annotations(
        db,
        project_id=project.id,
        document_id=document.id,
        annotator_user_id=current_user.id,
        user=current_user,
    )
    correction_locked_annotation_ids = (
        annotation_service.get_correction_locked_annotation_ids(
            db,
            document.id,
            user=current_user,
        )
    )
    visible_annotation_ids = {annotation.id for annotation in annotations}
    correction_locked_annotation_ids = [
        annotation_id
        for annotation_id in correction_locked_annotation_ids
        if annotation_id in visible_annotation_ids
    ]
    annotation_type_specs = [
        _annotation_type_spec(task.annotation_type) for task in project_tasks
    ]

    return AnnotationWorkbenchRead(
        project=ProjectRead.model_validate(project).model_copy(
            update={
                "tasks": [
                    ProjectTaskRead.model_validate(task) for task in project_tasks
                ]
            }
        ),
        document=DocumentRead.model_validate(document),
        active_guideline=(
            GuidelineVersionRead.model_validate(active_guideline)
            if active_guideline is not None
            else None
        ),
        guideline_versions_by_id=guideline_versions_by_id,
        tasks=[
            AnnotationWorkbenchTaskRead(
                **ProjectTaskRead.model_validate(task).model_dump(),
                annotation_type_spec=_annotation_type_spec(task.annotation_type),
            )
            for task in project_tasks
        ],
        annotation_type_specs=annotation_type_specs,
        annotations=[
            AnnotationRead.model_validate(annotation) for annotation in annotations
        ],
        assignments=[
            TaskAssignmentRead.model_validate(assignment) for assignment in assignments
        ],
        correction_locked_annotation_ids=correction_locked_annotation_ids,
    )
