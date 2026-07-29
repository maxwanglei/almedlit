from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from al_medlit.annotation.models import Annotation
from al_medlit.annotation.types import REGISTRY as ANNOTATION_TYPE_REGISTRY
from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    ValidationError,
)
from al_medlit.core.models import utc_now
from al_medlit.corpus.models import Document
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceBlockRevision,
    EvidenceReviewCoverage,
    EvidenceReviewEvent,
    EvidenceTarget,
    EvidenceTargetVersion,
)
from al_medlit.guideline import service as guideline_service
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.project.schemas import (
    EvidenceBlockTaskSettingsV1,
    ProjectCreate,
    ProjectProgressRead,
    ProjectTaskCreate,
    ProjectTaskUpdate,
    ProjectUpdate,
    TaskAssignmentCreate,
    TaskAssignmentUpdate,
)
from al_medlit.submission.models import AnnotationSubmission
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.models import WorkspaceMember

TASK_DISPLAY_NAMES = {
    "entity": "Entity Annotation",
    "relation": "Relation Annotation",
    "doc_label": "Document Annotation",
    "sentence_label": "Sentence Annotation",
    "passage_label": "Passage Annotation",
    "evidence_block": "Evidence Block Identification",
}


def _default_task_display_name(annotation_type: str) -> str:
    return TASK_DISPLAY_NAMES.get(annotation_type, annotation_type.replace("_", " ").title())


def _normalize_task_payload(data: dict, *, sort_order: int | None = None) -> dict:
    annotation_type = data["annotation_type"]
    return {
        "annotation_type": annotation_type,
        "display_name": data.get("display_name") or _default_task_display_name(annotation_type),
        "description": data.get("description"),
        "enabled": data.get("enabled", True),
        "sort_order": data.get("sort_order", sort_order if sort_order is not None else 0),
        "labels": data.get("labels") or [],
        "settings": data.get("settings") or {},
    }


def _derive_tasks_from_annotation_schema(annotation_schema: dict) -> list[dict]:
    labels_by_type = annotation_schema.get("labels", {})
    if not isinstance(labels_by_type, dict):
        return []

    task_payloads = []
    for index, annotation_type in enumerate(labels_by_type):
        if annotation_type not in ANNOTATION_TYPE_REGISTRY:
            continue
        task_payloads.append(
            _normalize_task_payload(
                {
                    "annotation_type": annotation_type,
                    "labels": labels_by_type.get(annotation_type) or [],
                },
                sort_order=index,
            )
        )
    return task_payloads


def _validate_unique_task_types(task_payloads: list[dict]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for task_payload in task_payloads:
        annotation_type = task_payload["annotation_type"]
        if annotation_type in seen:
            duplicates.add(annotation_type)
        seen.add(annotation_type)
    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValidationError(f"Duplicate project task annotation_type: {duplicate_list}")


def _flush_or_conflict(db: Session, message: str) -> None:
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(message) from exc


def _commit_or_conflict(db: Session, message: str) -> None:
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(message) from exc


def _annotation_schema_from_tasks(tasks: list[ProjectTask | dict]) -> dict:
    labels: dict[str, list] = {}
    for task in tasks:
        if isinstance(task, ProjectTask):
            enabled = task.enabled
            annotation_type = task.annotation_type
            task_labels = task.labels
        else:
            enabled = task.get("enabled", True)
            annotation_type = task["annotation_type"]
            task_labels = task.get("labels") or []

        if enabled:
            labels[annotation_type] = task_labels
    return {"labels": labels}


def _sync_project_annotation_schema(db: Session, project_id: int) -> None:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    tasks = list_project_tasks(db, project_id)
    project.annotation_schema = _annotation_schema_from_tasks(tasks)
    flag_modified(project, "annotation_schema")


def _get_required_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _lock_assignment_project(db: Session, project_id: int) -> Project:
    """Lock an assignment's workspace before locking its project scope.

    Workspace membership removal uses the same workspace row lock. Holding it
    until assignment commit makes the later refreshed membership check atomic
    with respect to removal: either removal wins and assignment sees no member,
    or assignment wins and removal waits until the assignment is durable.
    """

    workspace_id = db.query(Project.workspace_id).filter(Project.id == project_id).scalar()
    if workspace_id is None:
        raise NotFoundError("Project not found")
    workspace_service.lock_workspace_for_update(db, workspace_id)
    project = (
        db.query(Project)
        .filter(Project.id == project_id, Project.workspace_id == workspace_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _get_project_task(db: Session, project_id: int, task_id: int) -> ProjectTask:
    task = db.get(ProjectTask, task_id)
    if task is None or task.project_id != project_id:
        raise NotFoundError("Project task not found")
    return task


def _get_task_assignment(
    db: Session,
    project_id: int,
    assignment_id: int,
) -> TaskAssignment:
    assignment = db.get(TaskAssignment, assignment_id)
    if assignment is None or assignment.project_id != project_id:
        raise NotFoundError("Task assignment not found")
    return assignment


def _resolve_user_for_assignment(
    db: Session,
    project: Project,
    *,
    assignee_user_id: int | None,
    annotator_id: str | None,
) -> User:
    if assignee_user_id is not None:
        user = db.query(User).filter(User.id == assignee_user_id).populate_existing().first()
        if user is None:
            raise NotFoundError("Assignee user not found")
    else:
        cleaned_annotator_id = annotator_id.strip() if annotator_id else ""
        if not cleaned_annotator_id:
            raise ValidationError("assignee_user_id is required")
        user = (
            db.query(User).filter(User.username == cleaned_annotator_id).populate_existing().first()
        )
        if user is None:
            raise NotFoundError("Assignee user not found")

    if not user.is_active:
        raise ValidationError("Assignee user must be active")

    if project.workspace_id is None:
        raise ValidationError("Project is not assigned to a workspace")
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == project.workspace_id,
            WorkspaceMember.user_id == user.id,
        )
        .populate_existing()
        .first()
    )
    if member is None:
        raise ValidationError("Assignee must be a member of the project workspace")
    return user


def _validate_assignment_scope(
    db: Session,
    *,
    project_id: int,
    task_id: int,
    document_id: int,
    target_version_id: int | None,
    structure_version_id: int | None,
    guideline_version_id: int | None,
) -> tuple[ProjectTask, int | None, int | None, int | None, str]:
    task = _get_project_task(db, project_id, task_id)
    if not task.enabled:
        raise ValidationError("Cannot assign a disabled project task")

    document = db.get(Document, document_id)
    if document is None:
        raise ValidationError(f"Document {document_id} not found")
    if document.project_id != project_id:
        raise ValidationError(f"Document {document_id} does not belong to project {project_id}")
    active_structure_version_id = getattr(document, "active_structure_version_id", None)
    if structure_version_id is None:
        structure_version_id = active_structure_version_id
    if structure_version_id is not None:
        from al_medlit.corpus.models import DocumentStructureVersion

        structure = db.get(DocumentStructureVersion, structure_version_id)
        if structure is None or structure.document_id != document.id:
            raise ValidationError("structure_version_id must belong to the assigned document")
        if (
            active_structure_version_id is not None
            and structure_version_id != active_structure_version_id
        ):
            raise ValidationError("New assignments must use the active document structure")

    if guideline_version_id is None:
        active_guideline = guideline_service.get_active_guideline(db, project_id)
        guideline_version_id = active_guideline.id if active_guideline is not None else None
    if guideline_version_id is not None:
        guideline = db.get(GuidelineVersion, guideline_version_id)
        if guideline is None or guideline.project_id != project_id:
            raise ValidationError("guideline_version_id must belong to the project")

    if task.annotation_type == "evidence_block":
        if target_version_id is None:
            raise ValidationError("target_version_id is required for evidence assignments")
        target_version = db.get(EvidenceTargetVersion, target_version_id)
        target = target_version.target if target_version is not None else None
        if target is None or target.project_id != project_id or target.task_id != task.id:
            raise ValidationError("target_version_id must belong to the evidence task")
        if not target.is_active or target.active_version_id != target_version_id:
            raise ValidationError("New assignments must use an active target version")
    else:
        if target_version_id is not None:
            raise ValidationError("target_version_id is only valid for evidence-block assignments")

    assignment_scope_key = _assignment_scope_key(
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
        guideline_version_id=guideline_version_id,
    )

    return (
        task,
        target_version_id,
        structure_version_id,
        guideline_version_id,
        assignment_scope_key,
    )


def _assignment_scope_key(
    *,
    target_version_id: int | None,
    structure_version_id: int | None,
    guideline_version_id: int | None,
) -> str:
    scope_prefix = f"target:{target_version_id}" if target_version_id is not None else "document"
    return (
        f"{scope_prefix}:structure:{structure_version_id or 'none'}:"
        f"guideline:{guideline_version_id or 'none'}"
    )


def _logical_target_id_for_version(
    db: Session,
    target_version_id: int,
) -> int:
    target_id = (
        db.query(EvidenceTargetVersion.target_id)
        .filter(EvidenceTargetVersion.id == target_version_id)
        .scalar()
    )
    if target_id is None:
        raise ValidationError("target_version_id does not exist")
    return target_id


def _has_open_assignment_round(
    db: Session,
    *,
    task_id: int,
    document_id: int,
    assignee_user_id: int,
    target_version_id: int | None,
    exclude_assignment_id: int | None = None,
) -> bool:
    """Lock and find a writable round for the same logical target.

    Evidence target versions are immutable revisions of one logical target.
    Different targets under the same evidence task remain independently
    writable, while two versions of one target may not be open together.
    """

    query = db.query(TaskAssignment.id).filter(
        TaskAssignment.task_id == task_id,
        TaskAssignment.document_id == document_id,
        TaskAssignment.assignee_user_id == assignee_user_id,
        TaskAssignment.status.in_(("assigned", "in_progress", "blocked")),
    )
    if target_version_id is None:
        query = query.filter(TaskAssignment.target_version_id.is_(None))
    else:
        logical_target_id = _logical_target_id_for_version(db, target_version_id)
        query = query.join(
            EvidenceTargetVersion,
            EvidenceTargetVersion.id == TaskAssignment.target_version_id,
        ).filter(EvidenceTargetVersion.target_id == logical_target_id)
    if exclude_assignment_id is not None:
        query = query.filter(TaskAssignment.id != exclude_assignment_id)
    return query.order_by(TaskAssignment.id).with_for_update(of=TaskAssignment).first() is not None


def create_project(db: Session, data: ProjectCreate) -> Project:
    payload = data.model_dump()
    task_payloads = payload.pop("tasks", [])
    if task_payloads:
        normalized_tasks = [_normalize_task_payload(task_payload) for task_payload in task_payloads]
        payload["annotation_schema"] = _annotation_schema_from_tasks(normalized_tasks)
    else:
        normalized_tasks = _derive_tasks_from_annotation_schema(
            payload.get("annotation_schema") or {}
        )

    _validate_unique_task_types(normalized_tasks)

    if payload.get("workspace_id") is None:
        payload["workspace_id"] = workspace_service.ensure_default_workspace(db).id

    conflict_message = f"Project with name {payload['name']!r} already exists"
    existing = (
        db.query(Project.id)
        .filter(
            Project.workspace_id == payload["workspace_id"],
            Project.name == payload["name"],
        )
        .first()
    )
    if existing is not None:
        raise ConflictError(conflict_message)

    project = Project(**payload)
    project.tasks = [ProjectTask(**task_payload) for task_payload in normalized_tasks]
    db.add(project)
    _commit_or_conflict(db, conflict_message)
    db.refresh(project)
    return project


def list_projects(
    db: Session,
    user: User | None = None,
    *,
    workspace_id: int | None = None,
) -> list[Project]:
    query = db.query(Project)
    if workspace_id is not None:
        query = query.filter(Project.workspace_id == workspace_id)
    if user is not None and not user.is_superuser:
        workspace_ids = [
            member_workspace_id
            for (member_workspace_id,) in (
                db.query(WorkspaceMember.workspace_id)
                .filter(
                    WorkspaceMember.user_id == user.id,
                    WorkspaceMember.role.in_(("trainer", "manager", "admin")),
                )
                .all()
            )
        ]
        if not workspace_ids:
            return []
        query = query.filter(Project.workspace_id.in_(workspace_ids))
    return query.order_by(Project.created_at.desc()).all()


def list_my_work_projects(
    db: Session,
    user: User,
    *,
    workspace_id: int,
) -> list[Project]:
    legacy_project_ids = {
        project_id
        for (project_id,) in (
            db.query(TaskAssignment.project_id)
            .filter(
                TaskAssignment.assignee_user_id == user.id,
                TaskAssignment.status != "withdrawn",
            )
            .all()
        )
    }

    from al_medlit.workflow.models import AnnotationRound

    for annotation_round in (
        db.query(AnnotationRound)
        .join(Project, Project.id == AnnotationRound.project_id)
        .filter(
            Project.workspace_id == workspace_id,
            AnnotationRound.status == "open",
        )
        .all()
    ):
        if (
            annotation_round.open_to_all_annotators
            or user.id in annotation_round.annotator_user_ids
        ):
            legacy_project_ids.add(annotation_round.project_id)

    if not legacy_project_ids:
        return []
    return (
        db.query(Project)
        .filter(
            Project.workspace_id == workspace_id,
            Project.id.in_(legacy_project_ids),
        )
        .order_by(Project.created_at.desc())
        .all()
    )


def get_project(db: Session, project_id: int) -> Project | None:
    return db.get(Project, project_id)


def update_project(db: Session, project_id: int, data: ProjectUpdate) -> Project:
    project = _get_required_project(db, project_id)
    updates = data.model_dump(exclude_unset=True)

    if updates.get("name") is None and "name" in updates:
        raise ValidationError("Project name is required")
    if (
        updates.get("annotation_validation_mode") is None
        and "annotation_validation_mode" in updates
    ):
        raise ValidationError("Project validation mode is required")
    if updates.get("settings") is None and "settings" in updates:
        raise ValidationError("Project settings are required")

    next_name = updates.get("name")
    if isinstance(next_name, str):
        next_name = next_name.strip()
        if not next_name:
            raise ValidationError("Project name is required")
        updates["name"] = next_name
    if next_name is not None and next_name != project.name:
        conflict_message = f"Project with name {next_name!r} already exists"
        existing = (
            db.query(Project.id)
            .filter(
                Project.workspace_id == project.workspace_id,
                Project.name == next_name,
                Project.id != project.id,
            )
            .first()
        )
        if existing is not None:
            raise ConflictError(conflict_message)
    else:
        conflict_message = "Project update violates a uniqueness constraint"

    for field, value in updates.items():
        setattr(project, field, value)

    _commit_or_conflict(db, conflict_message)
    db.refresh(project)
    return project


def list_project_tasks(
    db: Session,
    project_id: int,
    *,
    enabled_only: bool = False,
) -> list[ProjectTask]:
    _get_required_project(db, project_id)
    query = db.query(ProjectTask).filter(ProjectTask.project_id == project_id)
    if enabled_only:
        query = query.filter(ProjectTask.enabled.is_(True))
    return query.order_by(ProjectTask.sort_order.asc(), ProjectTask.id.asc()).all()


def create_project_task(
    db: Session,
    project_id: int,
    data: ProjectTaskCreate,
) -> ProjectTask:
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None:
        raise NotFoundError("Project not found")
    payload = _normalize_task_payload(data.model_dump())
    conflict_message = (
        f"Project task already exists for annotation_type {payload['annotation_type']!r}"
    )
    existing = (
        db.query(ProjectTask.id)
        .filter(
            ProjectTask.project_id == project_id,
            ProjectTask.annotation_type == payload["annotation_type"],
        )
        .first()
    )
    if existing is not None:
        raise ConflictError(conflict_message)

    task = ProjectTask(project_id=project_id, **payload)
    db.add(task)
    _flush_or_conflict(db, conflict_message)
    _sync_project_annotation_schema(db, project_id)
    _commit_or_conflict(db, conflict_message)
    db.refresh(task)
    return task


def update_project_task(
    db: Session,
    project_id: int,
    task_id: int,
    data: ProjectTaskUpdate,
) -> ProjectTask:
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None:
        raise NotFoundError("Project not found")
    task = (
        db.query(ProjectTask)
        .filter(
            ProjectTask.id == task_id,
            ProjectTask.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if task is None:
        raise NotFoundError("Project task not found")
    updates = data.model_dump(exclude_unset=True)
    if task.annotation_type == "evidence_block" and "settings" in updates:
        try:
            updates["settings"] = EvidenceBlockTaskSettingsV1.model_validate(
                updates["settings"] or {}
            ).model_dump()
        except PydanticValidationError as exc:
            first_error = exc.errors()[0]
            raise ValidationError(str(first_error["msg"])) from exc
    for field, value in updates.items():
        setattr(task, field, value)

    db.flush()
    _sync_project_annotation_schema(db, project_id)
    db.commit()
    db.refresh(task)
    return task


def delete_project_task(db: Session, project_id: int, task_id: int) -> bool:
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None:
        raise NotFoundError("Project not found")
    task = (
        db.query(ProjectTask)
        .filter(ProjectTask.id == task_id, ProjectTask.project_id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if task is None:
        raise NotFoundError("Project task not found")
    if task.annotation_type == "evidence_block" and task.evidence_targets:
        raise ConflictError(
            "Evidence-block tasks with versioned targets cannot be deleted; disable the task"
        )
    db.delete(task)
    db.flush()
    _sync_project_annotation_schema(db, project_id)
    db.commit()
    return True


def list_task_assignments(
    db: Session,
    project_id: int,
    *,
    document_id: int | None = None,
    task_id: int | None = None,
    assignee_user_id: int | None = None,
    annotator_id: str | None = None,
    status: str | None = None,
    target_version_id: int | None = None,
) -> list[TaskAssignment]:
    _get_required_project(db, project_id)
    query = db.query(TaskAssignment).filter(TaskAssignment.project_id == project_id)
    if document_id is not None:
        query = query.filter(TaskAssignment.document_id == document_id)
    if task_id is not None:
        query = query.filter(TaskAssignment.task_id == task_id)
    if assignee_user_id is not None:
        query = query.filter(TaskAssignment.assignee_user_id == assignee_user_id)
    if annotator_id is not None:
        query = query.filter(TaskAssignment.annotator_id == annotator_id)
    if status is not None:
        query = query.filter(TaskAssignment.status == status)
    if target_version_id is not None:
        query = query.filter(TaskAssignment.target_version_id == target_version_id)
    return query.order_by(TaskAssignment.created_at.desc()).all()


def create_task_assignment(
    db: Session,
    project_id: int,
    data: TaskAssignmentCreate,
    *,
    assigned_by_user: User | None = None,
    assigned_by: str | None = None,
) -> TaskAssignment:
    payload = data.model_dump()
    if payload["status"] == "withdrawn":
        raise ValidationError("The withdrawn status is reserved for workspace offboarding")
    # Lock the workspace first so assignee membership cannot be removed after
    # validation but before this assignment commits. Then lock the remaining
    # stable scope owners before resolving active version defaults. Guideline
    # activation uses the project lock, task/target changes use the task lock,
    # and structure rebuilds use the document lock.
    project = _lock_assignment_project(db, project_id)
    task = (
        db.query(ProjectTask)
        .filter(
            ProjectTask.id == payload["task_id"],
            ProjectTask.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if task is None:
        raise NotFoundError("Project task not found")
    document = (
        db.query(Document)
        .filter(
            Document.id == payload["document_id"],
            Document.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if document is None:
        raise ValidationError(
            f"Document {payload['document_id']} does not belong to project {project_id}"
        )
    (
        task,
        payload["target_version_id"],
        payload["structure_version_id"],
        payload["guideline_version_id"],
        payload["assignment_scope_key"],
    ) = _validate_assignment_scope(
        db,
        project_id=project_id,
        task_id=payload["task_id"],
        document_id=payload["document_id"],
        target_version_id=payload.get("target_version_id"),
        structure_version_id=payload.get("structure_version_id"),
        guideline_version_id=payload.get("guideline_version_id"),
    )
    assignee = _resolve_user_for_assignment(
        db,
        project,
        assignee_user_id=payload.pop("assignee_user_id", None),
        annotator_id=payload.pop("annotator_id", None),
    )
    payload["assignee_user_id"] = assignee.id
    payload["annotator_id"] = assignee.username
    if assigned_by_user is not None:
        payload["assigned_by_user_id"] = assigned_by_user.id
        payload["assigned_by"] = assigned_by_user.username
    elif assigned_by is not None:
        payload["assigned_by"] = assigned_by
    conflict_message = (
        "Task assignment already exists for this task, document, and annotator "
        "(within this assignment scope)"
    )
    existing = (
        db.query(TaskAssignment.id)
        .filter(
            TaskAssignment.task_id == payload["task_id"],
            TaskAssignment.document_id == payload["document_id"],
            TaskAssignment.assignee_user_id == payload["assignee_user_id"],
            TaskAssignment.assignment_scope_key == payload["assignment_scope_key"],
        )
        .first()
    )
    if existing is not None:
        raise ConflictError(conflict_message)
    if _has_open_assignment_round(
        db,
        task_id=task.id,
        document_id=payload["document_id"],
        assignee_user_id=payload["assignee_user_id"],
        target_version_id=payload["target_version_id"],
    ):
        raise ConflictError("Finish the current assignment round before assigning a new version")

    assignment = TaskAssignment(project_id=project_id, **payload)
    db.add(assignment)
    _commit_or_conflict(db, conflict_message)
    db.refresh(assignment)
    return assignment


def ensure_personal_project_self_assignments(
    db: Session,
    project_id: int,
    user: User,
    *,
    document_ids: Iterable[int] | None = None,
    task_ids: Iterable[int] | None = None,
    assigned_by_user: User | None = None,
    commit: bool = True,
) -> list[TaskAssignment]:
    project = _get_required_project(db, project_id)
    if project.workspace_id is None:
        return []

    workspace = workspace_service.require_workspace(db, project.workspace_id)
    if workspace.kind != "individual":
        return []
    if not user.is_active:
        raise ValidationError("Assignee user must be active")
    if workspace_service.get_member(db, workspace.id, user.id) is None:
        raise ValidationError("Assignee must be a member of the project workspace")
    project = (
        db.query(Project)
        .filter(Project.id == project_id)
        .populate_existing()
        .with_for_update()
        .one()
    )

    requested_document_ids = set(document_ids) if document_ids is not None else None
    requested_task_ids = set(task_ids) if task_ids is not None else None
    if requested_document_ids == set() or requested_task_ids == set():
        return []

    task_query = db.query(ProjectTask).filter(
        ProjectTask.project_id == project_id,
        ProjectTask.enabled.is_(True),
    )
    if requested_task_ids is not None:
        task_query = task_query.filter(ProjectTask.id.in_(requested_task_ids))
    tasks = (
        task_query.order_by(ProjectTask.sort_order.asc(), ProjectTask.id.asc())
        .populate_existing()
        .with_for_update()
        .all()
    )

    document_query = db.query(Document).filter(Document.project_id == project_id)
    if requested_document_ids is not None:
        document_query = document_query.filter(Document.id.in_(requested_document_ids))
    documents = (
        document_query.order_by(Document.id.asc()).populate_existing().with_for_update().all()
    )
    if not tasks or not documents:
        return []

    task_id_values = [task.id for task in tasks]
    document_id_values = [document.id for document in documents]
    existing = {
        (task_id, document_id, scope_key)
        for task_id, document_id, scope_key in db.query(
            TaskAssignment.task_id,
            TaskAssignment.document_id,
            TaskAssignment.assignment_scope_key,
        )
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.assignee_user_id == user.id,
            TaskAssignment.task_id.in_(task_id_values),
            TaskAssignment.document_id.in_(document_id_values),
        )
        .all()
    }
    open_rounds = {
        (task_id, document_id, logical_target_id)
        for task_id, document_id, logical_target_id in db.query(
            TaskAssignment.task_id,
            TaskAssignment.document_id,
            EvidenceTargetVersion.target_id,
        )
        .outerjoin(
            EvidenceTargetVersion,
            EvidenceTargetVersion.id == TaskAssignment.target_version_id,
        )
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.assignee_user_id == user.id,
            TaskAssignment.task_id.in_(task_id_values),
            TaskAssignment.document_id.in_(document_id_values),
            TaskAssignment.status.in_(("assigned", "in_progress", "blocked")),
        )
        .all()
    }

    assigner = assigned_by_user or user
    created: list[TaskAssignment] = []
    active_targets_by_task: dict[int, list[EvidenceTarget]] = defaultdict(list)
    logical_target_by_version: dict[int, int] = {}
    evidence_task_ids = [task.id for task in tasks if task.annotation_type == "evidence_block"]
    if evidence_task_ids:
        for target in (
            db.query(EvidenceTarget)
            .filter(
                EvidenceTarget.task_id.in_(evidence_task_ids),
                EvidenceTarget.is_active.is_(True),
                EvidenceTarget.active_version_id.is_not(None),
            )
            .populate_existing()
            .all()
        ):
            active_targets_by_task[target.task_id].append(target)
            if target.active_version_id is not None:
                logical_target_by_version[target.active_version_id] = target.id

    active_guideline = guideline_service.get_active_guideline(db, project_id)
    for task in tasks:
        target_versions: list[int | None]
        if task.annotation_type == "evidence_block":
            target_versions = sorted(
                target.active_version_id
                for target in active_targets_by_task[task.id]
                if target.active_version_id is not None
            )
        else:
            target_versions = [None]
        for document in documents:
            for target_version_id in target_versions:
                logical_target_id = (
                    logical_target_by_version[target_version_id]
                    if target_version_id is not None
                    else None
                )
                structure_version_id = getattr(
                    document,
                    "active_structure_version_id",
                    None,
                )
                guideline_version_id = active_guideline.id if active_guideline is not None else None
                scope_key = _assignment_scope_key(
                    target_version_id=target_version_id,
                    structure_version_id=structure_version_id,
                    guideline_version_id=guideline_version_id,
                )
                if (task.id, document.id, scope_key) in existing:
                    continue
                if (task.id, document.id, logical_target_id) in open_rounds:
                    continue
                assignment = TaskAssignment(
                    project_id=project_id,
                    task_id=task.id,
                    document_id=document.id,
                    assignee_user_id=user.id,
                    annotator_id=user.username,
                    target_version_id=target_version_id,
                    structure_version_id=structure_version_id,
                    guideline_version_id=guideline_version_id,
                    assignment_scope_key=scope_key,
                    status="assigned",
                    assigned_by_user_id=assigner.id,
                    assigned_by=assigner.username,
                    metadata_={
                        "auto_created": True,
                        "source": "personal_workspace",
                    },
                )
                db.add(assignment)
                created.append(assignment)
                existing.add((task.id, document.id, scope_key))
                open_rounds.add((task.id, document.id, logical_target_id))

    if not created:
        return []

    if commit:
        _commit_or_conflict(
            db,
            "Task assignment already exists for this task, document, and annotator "
            "(within this assignment scope)",
        )
        for assignment in created:
            db.refresh(assignment)
    else:
        _flush_or_conflict(
            db,
            "Task assignment already exists for this task, document, and annotator "
            "(within this assignment scope)",
        )
    return created


def _assignment_has_persisted_work(
    db: Session,
    assignment: TaskAssignment,
) -> bool:
    """Return whether changing assignment ownership would orphan scoped work."""
    if (
        db.query(AnnotationSubmission.id)
        .filter(AnnotationSubmission.assignment_id == assignment.id)
        .first()
        is not None
    ):
        return True

    annotation_query = db.query(Annotation.id).filter(
        Annotation.project_id == assignment.project_id,
        Annotation.document_id == assignment.document_id,
        Annotation.annotation_type == assignment.task.annotation_type,
        Annotation.annotator_user_id == assignment.assignee_user_id,
        Annotation.structure_version_id == assignment.structure_version_id,
        Annotation.guideline_version_id == assignment.guideline_version_id,
    )
    if assignment.task.annotation_type == "evidence_block":
        annotation_query = annotation_query.join(EvidenceBlockAnnotation).filter(
            EvidenceBlockAnnotation.target_version_id == assignment.target_version_id,
            EvidenceBlockAnnotation.structure_version_id == assignment.structure_version_id,
        )
    if annotation_query.first() is not None:
        return True

    if assignment.task.annotation_type != "evidence_block":
        return False

    # Reject-only prediction decisions do not create an annotation or evidence
    # revision, so the review's explicit assignment pin is the only durable
    # provenance tying that work to this exact round.
    from al_medlit.inference.models import EvidencePredictionReview

    if (
        db.query(EvidencePredictionReview.id)
        .filter(EvidencePredictionReview.assignment_id == assignment.id)
        .first()
        is not None
    ):
        return True
    coverage_scope = (
        EvidenceReviewCoverage.project_id == assignment.project_id,
        EvidenceReviewCoverage.document_id == assignment.document_id,
        EvidenceReviewCoverage.target_version_id == assignment.target_version_id,
        EvidenceReviewCoverage.structure_version_id == assignment.structure_version_id,
        EvidenceReviewCoverage.guideline_version_id == assignment.guideline_version_id,
        EvidenceReviewCoverage.reviewer_user_id == assignment.assignee_user_id,
    )
    if db.query(EvidenceReviewCoverage.id).filter(*coverage_scope).first() is not None:
        return True
    revisions = (
        db.query(EvidenceBlockRevision.before, EvidenceBlockRevision.after)
        .filter(
            EvidenceBlockRevision.project_id == assignment.project_id,
            EvidenceBlockRevision.document_id == assignment.document_id,
            EvidenceBlockRevision.target_version_id == assignment.target_version_id,
            EvidenceBlockRevision.structure_version_id == assignment.structure_version_id,
        )
        .all()
    )
    if any(
        state
        and state.get("annotator_user_id") == assignment.assignee_user_id
        and state.get("guideline_version_id") == assignment.guideline_version_id
        for before, after in revisions
        for state in (before, after)
    ):
        return True
    return (
        db.query(EvidenceReviewEvent.id)
        .filter(
            EvidenceReviewEvent.project_id == assignment.project_id,
            EvidenceReviewEvent.document_id == assignment.document_id,
            EvidenceReviewEvent.target_version_id == assignment.target_version_id,
            EvidenceReviewEvent.structure_version_id == assignment.structure_version_id,
            EvidenceReviewEvent.guideline_version_id == assignment.guideline_version_id,
            EvidenceReviewEvent.actor_user_id == assignment.assignee_user_id,
        )
        .first()
        is not None
    )


def update_task_assignment(
    db: Session,
    project_id: int,
    assignment_id: int,
    data: TaskAssignmentUpdate,
) -> TaskAssignment:
    assignment_scope = (
        db.query(TaskAssignment.task_id, TaskAssignment.document_id)
        .filter(
            TaskAssignment.id == assignment_id,
            TaskAssignment.project_id == project_id,
        )
        .first()
    )
    if assignment_scope is None:
        raise NotFoundError("Task assignment not found")
    task_id, document_id = assignment_scope
    # Reassignment must use the same stable scope lock order as assignment
    # creation. Otherwise two managers can concurrently give one annotator two
    # writable version rounds for the same task/document/target.
    project = _lock_assignment_project(db, project_id)
    task = (
        db.query(ProjectTask)
        .filter(ProjectTask.id == task_id, ProjectTask.project_id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    document = (
        db.query(Document)
        .filter(Document.id == document_id, Document.project_id == project_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None or task is None or document is None:
        raise NotFoundError("Task assignment scope no longer exists")
    assignment = (
        db.query(TaskAssignment)
        .filter(
            TaskAssignment.id == assignment_id,
            TaskAssignment.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if assignment is None:
        raise NotFoundError("Task assignment not found")
    updates = data.model_dump(exclude_unset=True)
    next_status = updates.get("status")
    if assignment.status == "withdrawn" and updates not in ({}, {"status": "withdrawn"}):
        raise ConflictError("Withdrawn task assignments are immutable audit records")
    final_status_order = {
        "submitted": 0,
        "adjudication_ready": 1,
        "adjudicated": 2,
        "completed": 3,
        # Offboarding is a terminal audit state. Giving it the highest rank
        # prevents API updates from reopening or moving it into another state.
        "withdrawn": 4,
    }
    if assignment.status in final_status_order and next_status is not None:
        if (assignment.status == "withdrawn" and next_status != "withdrawn") or (
            assignment.status != "withdrawn" and next_status == "withdrawn"
        ):
            raise ConflictError("Finalized task assignments cannot be reopened or moved backward")
        if (
            next_status not in final_status_order
            or final_status_order[next_status] < final_status_order[assignment.status]
        ):
            raise ConflictError("Finalized task assignments cannot be reopened or moved backward")
    if "assignee_user_id" in updates or "annotator_id" in updates:
        project = _get_required_project(db, project_id)
        assignee = _resolve_user_for_assignment(
            db,
            project,
            assignee_user_id=updates.pop("assignee_user_id", None),
            annotator_id=updates.pop("annotator_id", None),
        )
        updates["assignee_user_id"] = assignee.id
        updates["annotator_id"] = assignee.username
        if assignee.id != assignment.assignee_user_id:
            if assignment.status in final_status_order:
                raise ConflictError("Finalized task assignments cannot be reassigned")
            if _assignment_has_persisted_work(db, assignment):
                raise ConflictError(
                    "Task assignment cannot be reassigned after submissions or work exist"
                )
            if _has_open_assignment_round(
                db,
                task_id=assignment.task_id,
                document_id=assignment.document_id,
                assignee_user_id=assignee.id,
                target_version_id=assignment.target_version_id,
                exclude_assignment_id=assignment.id,
            ):
                raise ConflictError(
                    "Finish the destination annotator's current assignment round "
                    "before reassigning a new version"
                )
    for field, value in updates.items():
        setattr(assignment, field, value)

    _commit_or_conflict(
        db,
        "Task assignment update conflicts with an existing assignment or persisted work",
    )
    db.refresh(assignment)
    return assignment


def reopen_personal_task_assignment(
    db: Session,
    project_id: int,
    assignment_id: int,
    *,
    current_user: User,
) -> TaskAssignment:
    """Reopen one submitted personal paper/task while preserving its snapshots.

    Generic assignment updates intentionally keep finalized states monotonic.
    This narrowly scoped operation is the exception for a user's own individual
    workspace: every prior submission remains immutable, while the same
    assignment becomes writable for a corrected follow-up submission.
    """

    assignment_scope = (
        db.query(TaskAssignment.task_id, TaskAssignment.document_id)
        .filter(
            TaskAssignment.id == assignment_id,
            TaskAssignment.project_id == project_id,
        )
        .first()
    )
    if assignment_scope is None:
        raise NotFoundError("Task assignment not found")
    task_id, document_id = assignment_scope

    project = _lock_assignment_project(db, project_id)
    workspace = workspace_service.require_workspace(db, project.workspace_id)
    if workspace.kind != "individual":
        raise ForbiddenError("Only personal workspace assignments can be reopened")

    task = (
        db.query(ProjectTask)
        .filter(
            ProjectTask.id == task_id,
            ProjectTask.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    document = (
        db.query(Document)
        .filter(
            Document.id == document_id,
            Document.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if task is None or document is None:
        raise NotFoundError("Task assignment scope no longer exists")

    assignment = (
        db.query(TaskAssignment)
        .filter(
            TaskAssignment.id == assignment_id,
            TaskAssignment.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if assignment is None:
        raise NotFoundError("Task assignment not found")
    if assignment.assignee_user_id != current_user.id:
        raise ForbiddenError("Only the assignment owner can reopen personal work")
    if assignment.status != "submitted":
        raise ConflictError("Only a submitted personal assignment can be reopened")
    if _has_open_assignment_round(
        db,
        task_id=assignment.task_id,
        document_id=assignment.document_id,
        assignee_user_id=assignment.assignee_user_id,
        target_version_id=assignment.target_version_id,
        exclude_assignment_id=assignment.id,
    ):
        raise ConflictError("This paper task already has an editable assignment round")

    metadata = dict(assignment.metadata_ or {})
    reopen_history = list(metadata.get("reopen_history") or [])
    reopen_history.append(
        {
            "reopened_at": utc_now().isoformat(),
            "actor_user_id": current_user.id,
            "from_status": "submitted",
        }
    )
    metadata["reopen_history"] = reopen_history
    assignment.metadata_ = metadata
    assignment.status = "in_progress"

    _commit_or_conflict(db, "Unable to reopen the personal assignment")
    db.refresh(assignment)
    return assignment


def delete_task_assignment(db: Session, project_id: int, assignment_id: int) -> bool:
    _lock_assignment_project(db, project_id)
    assignment = (
        db.query(TaskAssignment)
        .filter(
            TaskAssignment.id == assignment_id,
            TaskAssignment.project_id == project_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if assignment is None:
        raise NotFoundError("Task assignment not found")
    if _assignment_has_persisted_work(db, assignment):
        raise ConflictError("Task assignment cannot be deleted after submissions or work exist")
    if assignment.status not in {"assigned", "in_progress", "blocked"}:
        raise ConflictError("Finalized task assignments cannot be deleted")
    db.delete(assignment)
    _commit_or_conflict(
        db,
        "Task assignment cannot be deleted because dependent records exist",
    )
    return True


def get_project_progress(
    db: Session,
    project_id: int,
    *,
    assignee_user_id: int | None = None,
) -> ProjectProgressRead:
    _get_required_project(db, project_id)
    assignments = list_task_assignments(
        db,
        project_id,
        assignee_user_id=assignee_user_id,
    )
    tasks_by_id = {
        task.id: task
        for task in db.query(ProjectTask).filter(ProjectTask.project_id == project_id).all()
    }

    by_status = Counter(assignment.status for assignment in assignments)
    by_task: dict[int, Counter] = defaultdict(Counter)
    by_document: dict[int, Counter] = defaultdict(Counter)
    by_annotator: dict[tuple[int, str], Counter] = defaultdict(Counter)
    by_target: dict[int, Counter] = defaultdict(Counter)

    for assignment in assignments:
        by_task[assignment.task_id][assignment.status] += 1
        by_document[assignment.document_id][assignment.status] += 1
        by_annotator[(assignment.assignee_user_id, assignment.annotator_id)][assignment.status] += 1
        if assignment.target_version_id is not None:
            by_target[assignment.target_version_id][assignment.status] += 1

    target_versions = {
        version.id: version
        for version in db.query(EvidenceTargetVersion)
        .filter(EvidenceTargetVersion.id.in_(by_target))
        .all()
    }

    return ProjectProgressRead(
        project_id=project_id,
        total=len(assignments),
        by_status=dict(by_status),
        by_task=[
            {
                "task_id": task_id,
                "annotation_type": tasks_by_id[task_id].annotation_type,
                "display_name": tasks_by_id[task_id].display_name,
                "total": sum(status_counts.values()),
                "by_status": dict(status_counts),
            }
            for task_id, status_counts in sorted(by_task.items())
            if task_id in tasks_by_id
        ],
        by_document=[
            {
                "document_id": document_id,
                "total": sum(status_counts.values()),
                "by_status": dict(status_counts),
            }
            for document_id, status_counts in sorted(by_document.items())
        ],
        by_annotator=[
            {
                "assignee_user_id": assignee_user_id,
                "annotator_id": annotator_id,
                "total": sum(status_counts.values()),
                "by_status": dict(status_counts),
            }
            for (assignee_user_id, annotator_id), status_counts in sorted(
                by_annotator.items(),
                key=lambda item: item[0][1],
            )
        ],
        by_target=[
            {
                "target_version_id": target_version_id,
                "target_id": target_versions[target_version_id].target_id,
                "target_key": target_versions[target_version_id].target.key,
                "target_name": target_versions[target_version_id].target.name,
                "total": sum(status_counts.values()),
                "by_status": dict(status_counts),
            }
            for target_version_id, status_counts in sorted(by_target.items())
            if target_version_id in target_versions
        ],
    )
