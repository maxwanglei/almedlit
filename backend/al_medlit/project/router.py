from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_workspace_member,
    has_assignment_bypass,
    require_project_access,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError
from al_medlit.project import service
from al_medlit.project.schemas import (
    ProjectCreate,
    ProjectProgressRead,
    ProjectRead,
    ProjectTaskCreate,
    ProjectTaskRead,
    ProjectTaskUpdate,
    ProjectUpdate,
    TaskAssignmentCreate,
    TaskAssignmentRead,
    TaskAssignmentStatus,
    TaskAssignmentUpdate,
)
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.dependencies import ROLE_RANK
from al_medlit.workspace.models import WorkspaceMember

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectRead)
def create_project(
    payload: ProjectCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    workspace_id = payload.workspace_id
    if workspace_id is None:
        workspace_id = workspace_service.ensure_default_workspace(db).id
        payload = payload.model_copy(update={"workspace_id": workspace_id})
    assert_workspace_member(db, current_user, workspace_id, min_role="manager")
    return service.create_project(db, payload)


@router.get("", response_model=list[ProjectRead])
def list_projects(
    workspace_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if workspace_id is not None:
        assert_workspace_member(
            db,
            current_user,
            workspace_id,
            min_role="trainer",
        )
    elif (
        not current_user.is_superuser
        and not db.query(WorkspaceMember.id)
        .filter(
            WorkspaceMember.user_id == current_user.id,
            WorkspaceMember.role.in_(("trainer", "manager", "admin")),
        )
        .first()
    ):
        raise ForbiddenError("Insufficient workspace role")
    return service.list_projects(db, current_user, workspace_id=workspace_id)


@router.get("/my-work", response_model=list[ProjectRead])
def list_my_work_projects(
    workspace_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_workspace_member(db, current_user, workspace_id)
    projects = service.list_my_work_projects(
        db,
        current_user,
        workspace_id=workspace_id,
    )
    return [
        {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "annotation_schema": project.annotation_schema,
            "annotation_validation_mode": project.annotation_validation_mode,
            "tasks": [task for task in project.tasks if task.enabled],
            "settings": {},
            "workspace_id": project.workspace_id,
        }
        for project in projects
    ]


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(
    project_id: int,
    _member=Depends(require_project_access(min_role="trainer")),
    db: Session = Depends(get_db),
):
    project = service.get_project(db, project_id)
    if not project:
        raise NotFoundError("Project not found")
    return project


@router.patch("/{project_id}", response_model=ProjectRead)
def update_project(
    project_id: int,
    payload: ProjectUpdate,
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
):
    return service.update_project(db, project_id, payload)


@router.get("/{project_id}/tasks", response_model=list[ProjectTaskRead])
def list_project_tasks(
    project_id: int,
    enabled_only: bool = Query(False),
    _member=Depends(require_project_access(min_role="trainer")),
    db: Session = Depends(get_db),
):
    return service.list_project_tasks(db, project_id, enabled_only=enabled_only)


@router.post("/{project_id}/tasks", response_model=ProjectTaskRead)
def create_project_task(
    project_id: int,
    payload: ProjectTaskCreate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = service.create_project_task(db, project_id, payload)
    service.ensure_personal_project_self_assignments(
        db,
        project_id,
        current_user,
        task_ids=[task.id],
        assigned_by_user=current_user,
    )
    return task


@router.patch("/{project_id}/tasks/{task_id}", response_model=ProjectTaskRead)
def update_project_task(
    project_id: int,
    task_id: int,
    payload: ProjectTaskUpdate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    task = service.update_project_task(db, project_id, task_id, payload)
    service.ensure_personal_project_self_assignments(
        db,
        project_id,
        current_user,
        task_ids=[task.id],
        assigned_by_user=current_user,
    )
    return task


@router.delete("/{project_id}/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_task(
    project_id: int,
    task_id: int,
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
):
    service.delete_project_task(db, project_id, task_id)


@router.get("/{project_id}/assignments", response_model=list[TaskAssignmentRead])
def list_task_assignments(
    project_id: int,
    document_id: int | None = Query(None),
    task_id: int | None = Query(None),
    assignee_user_id: int | None = Query(None),
    annotator_id: str | None = Query(None),
    assignment_status: TaskAssignmentStatus | None = Query(None, alias="status"),
    target_version_id: int | None = Query(None),
    scope: str | None = Query(None, pattern="^(mine|all)$"),
    member: WorkspaceMember = Depends(require_project_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    requested_assignee_user_id = assignee_user_id
    restricted = not has_assignment_bypass(current_user, member)
    if restricted and assignee_user_id not in (None, current_user.id):
        raise ForbiddenError("Cannot view another annotator's assignments")
    if restricted and annotator_id not in (None, current_user.username):
        raise ForbiddenError("Cannot view another annotator's assignments")
    if scope == "mine" or (scope is None and restricted):
        requested_assignee_user_id = current_user.id
        annotator_id = None
    elif scope == "all" and restricted:
        raise ForbiddenError("Insufficient workspace role to view all assignments")
    return service.list_task_assignments(
        db,
        project_id,
        document_id=document_id,
        task_id=task_id,
        assignee_user_id=requested_assignee_user_id,
        annotator_id=annotator_id,
        status=assignment_status,
        target_version_id=target_version_id,
    )


@router.post("/{project_id}/assignments", response_model=TaskAssignmentRead)
def create_task_assignment(
    project_id: int,
    payload: TaskAssignmentCreate,
    _member=Depends(require_project_access(min_role="manager")),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.create_task_assignment(
        db,
        project_id,
        payload,
        assigned_by_user=current_user,
    )


@router.patch(
    "/{project_id}/assignments/{assignment_id}",
    response_model=TaskAssignmentRead,
)
def update_task_assignment(
    project_id: int,
    assignment_id: int,
    payload: TaskAssignmentUpdate,
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
):
    return service.update_task_assignment(db, project_id, assignment_id, payload)


@router.post(
    "/{project_id}/assignments/{assignment_id}/reopen",
    response_model=TaskAssignmentRead,
)
def reopen_personal_task_assignment(
    project_id: int,
    assignment_id: int,
    _member=Depends(require_project_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.reopen_personal_task_assignment(
        db,
        project_id,
        assignment_id,
        current_user=current_user,
    )


@router.delete(
    "/{project_id}/assignments/{assignment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_task_assignment(
    project_id: int,
    assignment_id: int,
    _member=Depends(require_project_access(min_role="manager")),
    db: Session = Depends(get_db),
):
    service.delete_task_assignment(db, project_id, assignment_id)


@router.get("/{project_id}/progress", response_model=ProjectProgressRead)
def get_project_progress(
    project_id: int,
    scope: str | None = Query(None, pattern="^(mine|all)$"),
    member: WorkspaceMember = Depends(require_project_access()),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assignee_user_id = None
    if scope == "mine" or (
        scope is None and ROLE_RANK.get(member.role, -1) < ROLE_RANK["manager"]
    ):
        assignee_user_id = current_user.id
    elif scope == "all" and ROLE_RANK.get(member.role, -1) < ROLE_RANK["manager"]:
        raise ForbiddenError("Insufficient workspace role to view all progress")
    return service.get_project_progress(
        db,
        project_id,
        assignee_user_id=assignee_user_id,
    )
