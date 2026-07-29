from collections.abc import Callable

from fastapi import Depends, Request
from sqlalchemy import and_, exists, or_, select, true
from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from al_medlit.corpus.models import Document
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.dependencies import ROLE_RANK
from al_medlit.workspace.models import Workspace, WorkspaceMember

ASSIGNMENT_BYPASS_ROLES = tuple(
    role for role, rank in ROLE_RANK.items() if rank >= ROLE_RANK["manager"]
)
MUTABLE_ASSIGNMENT_STATUSES = ("assigned", "in_progress", "blocked")


def _role_satisfies(actual: str, minimum: str | None) -> bool:
    if minimum is None:
        return True
    return ROLE_RANK.get(actual, -1) >= ROLE_RANK.get(minimum, 99)


def has_assignment_bypass(user: User, member: WorkspaceMember) -> bool:
    """Return whether a user may bypass assignment and record-owner checks."""
    return user.is_superuser or member.role in ASSIGNMENT_BYPASS_ROLES


def assert_global_manager_scope(db: Session, user: User) -> None:
    """Reject an unscoped ``scope=all`` request from any restricted membership."""
    if user.is_superuser:
        return
    restricted_membership = (
        db.query(WorkspaceMember.id)
        .filter(
            WorkspaceMember.user_id == user.id,
            ~WorkspaceMember.role.in_(ASSIGNMENT_BYPASS_ROLES),
        )
        .first()
    )
    if restricted_membership is not None:
        raise ForbiddenError("Insufficient workspace role to view all records")


def assignment_access_clause(
    user: User,
    *,
    project_id,
    document_id,
    annotation_type=None,
    target_version_id=None,
    structure_version_id=None,
):
    """Build the hybrid assignment predicate for a correlated resource query.

    A project with no assignment rows is open. Once the first assignment exists,
    restricted users need an assignment on an enabled task. Supplying an
    annotation type makes that assignment task-specific.
    """
    if user.is_superuser:
        return true()

    project_is_open = ~exists(
        select(TaskAssignment.id).where(TaskAssignment.project_id == project_id)
    )
    assignment_conditions = [
        TaskAssignment.project_id == project_id,
        TaskAssignment.document_id == document_id,
        TaskAssignment.assignee_user_id == user.id,
        ProjectTask.id == TaskAssignment.task_id,
        ProjectTask.project_id == project_id,
        ProjectTask.enabled.is_(True),
    ]
    if annotation_type is not None:
        assignment_conditions.append(ProjectTask.annotation_type == annotation_type)
    if target_version_id is not None:
        assignment_conditions.append(
            or_(
                ProjectTask.annotation_type != "evidence_block",
                TaskAssignment.target_version_id == target_version_id,
            )
        )
    if structure_version_id is not None:
        assignment_conditions.append(
            or_(
                ProjectTask.annotation_type != "evidence_block",
                TaskAssignment.structure_version_id == structure_version_id,
            )
        )
    has_assignment = exists(
        select(TaskAssignment.id)
        .select_from(TaskAssignment)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .where(*assignment_conditions)
    )
    return or_(project_is_open, has_assignment)


def resource_access_clause(
    user: User,
    *,
    workspace_id,
    project_id,
    document_id,
    annotation_type=None,
    owner_user_id=None,
    target_version_id=None,
    structure_version_id=None,
):
    """Build membership, assignment, task, and optional ownership filtering."""
    if user.is_superuser:
        return true()

    is_member = exists(
        select(WorkspaceMember.id).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
        )
    )
    is_manager = exists(
        select(WorkspaceMember.id).where(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user.id,
            WorkspaceMember.role.in_(ASSIGNMENT_BYPASS_ROLES),
        )
    )
    restricted_access = assignment_access_clause(
        user,
        project_id=project_id,
        document_id=document_id,
        annotation_type=annotation_type,
        target_version_id=target_version_id,
        structure_version_id=structure_version_id,
    )
    if owner_user_id is not None:
        restricted_access = and_(restricted_access, owner_user_id == user.id)
    return and_(is_member, or_(is_manager, restricted_access))


def annotation_reference_access_clause(
    user: User,
    annotation_id,
    *,
    project_id=None,
    document_id=None,
):
    """Require a nullable correction reference to point at a visible annotation."""
    reference_conditions = [Annotation.id == annotation_id]
    if not user.is_superuser:
        reference_conditions.append(
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=Annotation.project_id,
                document_id=Annotation.document_id,
                annotation_type=Annotation.annotation_type,
                owner_user_id=Annotation.annotator_user_id,
            )
        )
    if project_id is not None:
        reference_conditions.append(Annotation.project_id == project_id)
    if document_id is not None:
        reference_conditions.append(Annotation.document_id == document_id)
    visible_reference = exists(
        select(Annotation.id)
        .select_from(Annotation)
        .join(Project, Project.id == Annotation.project_id)
        .where(*reference_conditions)
    )
    return or_(annotation_id.is_(None), visible_reference)


def assert_workspace_member(
    db: Session,
    user: User,
    workspace_id: int | None,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    if workspace_id is None or db.get(Workspace, workspace_id) is None:
        raise NotFoundError("Workspace not found")

    if user.is_superuser:
        return WorkspaceMember(
            workspace_id=workspace_id,
            user_id=user.id,
            role="admin",
        )

    member = workspace_service.get_member(db, workspace_id, user.id)
    if member is None:
        raise ForbiddenError("You are not a member of this workspace")
    if not _role_satisfies(member.role, min_role):
        raise ForbiddenError("Insufficient workspace role")
    return member


def workspace_ids_for_user(db: Session, user: User) -> list[int]:
    return [
        workspace_id
        for (workspace_id,) in (
            db.query(WorkspaceMember.workspace_id).filter(WorkspaceMember.user_id == user.id).all()
        )
    ]


def _require_project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def _require_document(db: Session, document_id: int) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise NotFoundError("Document not found")
    return document


def _require_guideline_version(
    db: Session,
    guideline_version_id: int,
) -> GuidelineVersion:
    guideline = db.get(GuidelineVersion, guideline_version_id)
    if guideline is None:
        raise NotFoundError("Guideline version not found")
    return guideline


def _require_annotation(db: Session, annotation_id: int) -> Annotation:
    annotation = db.get(Annotation, annotation_id)
    if annotation is None:
        raise NotFoundError("Annotation not found")
    return annotation


def assert_project_member(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    project = _require_project(db, project_id)
    return assert_workspace_member(db, user, project.workspace_id, min_role=min_role)


def assert_project_member_if_exists(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember | None:
    project = db.get(Project, project_id)
    if project is None:
        return None
    return assert_workspace_member(db, user, project.workspace_id, min_role=min_role)


def lock_project_member_for_mutation(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    """Lock and refresh the actor's project membership before a mutation.

    Route dependencies are intentionally only an early rejection.  A concurrent
    membership removal or role demotion can make their cached membership stale,
    so every project-scoped write locks the active user and that user's membership
    before it locks domain rows or assignments and commits.  Offboarding and role
    changes lock the same membership row after their workspace lock.  This route
    never acquires the workspace lock afterward, so the ordering is deadlock-safe
    without serializing unrelated members' writes on one workspace row.
    """

    project = db.query(Project).filter(Project.id == project_id).populate_existing().first()
    if project is None:
        raise NotFoundError("Project not found")
    workspace_id = project.workspace_id

    actor = db.query(User).filter(User.id == user.id).populate_existing().with_for_update().first()
    if actor is None or not actor.is_active:
        raise ForbiddenError("Workspace mutation actor is not active")
    if actor.is_superuser:
        member = WorkspaceMember(
            workspace_id=workspace_id,
            user_id=actor.id,
            role="admin",
        )
    else:
        member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace_id,
                WorkspaceMember.user_id == actor.id,
            )
            .populate_existing()
            .with_for_update()
            .first()
        )
        if member is None:
            raise ForbiddenError("You are not a member of this workspace")
        if not _role_satisfies(member.role, min_role):
            raise ForbiddenError("Insufficient workspace role")

    if not has_assignment_bypass(actor, member):
        # A shared project lock permits concurrent annotators while preventing
        # assignment mode or task scope from changing underneath a restricted
        # user's authorization decision. Assignment/task mutations take the
        # corresponding exclusive project lock. Bypass roles do not depend on
        # assignment state and avoid holding this lock through longer writes
        # such as object-store submission.
        locked_project = (
            db.query(Project.id)
            .filter(Project.id == project_id, Project.workspace_id == workspace_id)
            .populate_existing()
            .with_for_update(read=True)
            .first()
        )
        if locked_project is None:
            raise NotFoundError("Project not found")
    return member


def lock_document_resource_for_mutation(
    db: Session,
    user: User,
    *,
    project_id: int,
    document_id: int,
    owner_user_id: int | None = None,
    enforce_owner: bool = False,
    min_role: str | None = None,
    lock_assignment: bool = True,
) -> WorkspaceMember:
    """Lock and authorize a document-scoped mutation in canonical order."""

    member = lock_project_member_for_mutation(
        db,
        user,
        project_id,
        min_role=min_role,
    )
    document = db.query(Document).filter(Document.id == document_id).populate_existing().first()
    if document is None:
        raise NotFoundError("Document not found")
    if document.project_id != project_id:
        raise ValidationError(f"Document {document_id} does not belong to project {project_id}")

    if not has_assignment_bypass(user, member):
        has_project_assignments = (
            db.query(TaskAssignment.id).filter(TaskAssignment.project_id == project_id).first()
            is not None
        )
        if has_project_assignments:
            assignment_query = (
                db.query(TaskAssignment)
                .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
                .filter(
                    TaskAssignment.project_id == project_id,
                    TaskAssignment.document_id == document_id,
                    TaskAssignment.assignee_user_id == user.id,
                    ProjectTask.project_id == project_id,
                    ProjectTask.enabled.is_(True),
                )
                .order_by(TaskAssignment.id)
                .populate_existing()
            )
            if lock_assignment:
                assignment_query = assignment_query.with_for_update(of=TaskAssignment)
            assignment = assignment_query.first()
            if assignment is None:
                raise ForbiddenError("Document is not assigned to the current user")

    if enforce_owner and not has_assignment_bypass(user, member) and owner_user_id != user.id:
        raise ForbiddenError("Resource belongs to another annotator")
    return member


def assert_document_member(
    db: Session,
    user: User,
    document_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    document = _require_document(db, document_id)
    project = _require_project(db, document.project_id)
    return assert_workspace_member(db, user, project.workspace_id, min_role=min_role)


def assert_document_member_if_exists(
    db: Session,
    user: User,
    document_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember | None:
    document = db.get(Document, document_id)
    if document is None:
        return None
    project = _require_project(db, document.project_id)
    return assert_workspace_member(db, user, project.workspace_id, min_role=min_role)


def assert_guideline_version_member(
    db: Session,
    user: User,
    guideline_version_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    guideline = _require_guideline_version(db, guideline_version_id)
    return assert_project_member(db, user, guideline.project_id, min_role=min_role)


def assert_guideline_version_member_if_exists(
    db: Session,
    user: User,
    guideline_version_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember | None:
    guideline = db.get(GuidelineVersion, guideline_version_id)
    if guideline is None:
        return None
    return assert_project_member(db, user, guideline.project_id, min_role=min_role)


def assert_document_assigned(
    db: Session,
    user: User,
    member: WorkspaceMember,
    *,
    project_id: int,
    document_id: int,
) -> None:
    if has_assignment_bypass(user, member):
        return
    has_project_assignments = (
        db.query(TaskAssignment.id).filter(TaskAssignment.project_id == project_id).first()
        is not None
    )
    if not has_project_assignments:
        return
    assigned = (
        db.query(TaskAssignment.id)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.document_id == document_id,
            TaskAssignment.assignee_user_id == user.id,
            ProjectTask.project_id == project_id,
            ProjectTask.enabled.is_(True),
        )
        .first()
        is not None
    )
    if not assigned:
        raise ForbiddenError("Document is not assigned to the current user")


def assert_task_assigned(
    db: Session,
    user: User,
    member: WorkspaceMember,
    *,
    project_id: int,
    document_id: int,
    annotation_type: str,
    target_version_id: int | None = None,
    structure_version_id: int | None = None,
    guideline_version_id: int | None = None,
    match_structure_version: bool = False,
    match_guideline_version: bool = False,
    require_mutable_assignment: bool = False,
) -> TaskAssignment | None:
    """Require an exact enabled task assignment in assignment-managed projects."""
    bypass = has_assignment_bypass(user, member)
    has_project_assignments = (
        db.query(TaskAssignment.id).filter(TaskAssignment.project_id == project_id).first()
        is not None
    )
    if not has_project_assignments:
        return None
    query = (
        db.query(TaskAssignment)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.document_id == document_id,
            TaskAssignment.assignee_user_id == user.id,
            ProjectTask.project_id == project_id,
            ProjectTask.annotation_type == annotation_type,
            ProjectTask.enabled.is_(True),
        )
    )
    if target_version_id is not None:
        query = query.filter(TaskAssignment.target_version_id == target_version_id)
    if structure_version_id is not None or match_structure_version:
        query = query.filter(TaskAssignment.structure_version_id == structure_version_id)
    if guideline_version_id is not None or match_guideline_version:
        query = query.filter(TaskAssignment.guideline_version_id == guideline_version_id)
    if require_mutable_assignment:
        if annotation_type != "evidence_block":
            # Evidence mutations acquire their stable target-version lock first and
            # re-check the assignment in the evidence service to keep one lock order.
            query = query.with_for_update(of=TaskAssignment)
        assignment = query.populate_existing().order_by(TaskAssignment.id.desc()).first()
        if assignment is not None and assignment.status not in MUTABLE_ASSIGNMENT_STATUSES:
            raise ForbiddenError(f"The {annotation_type!r} assignment is finalized and read-only")
    else:
        assignment = query.order_by(TaskAssignment.id.desc()).first()
    if assignment is None:
        if bypass:
            return None
        suffix = " with a mutable assignment" if require_mutable_assignment else ""
        raise ForbiddenError(
            f"The {annotation_type!r} task is not assigned to the current user{suffix}"
        )
    return assignment


def assert_document_resource_access(
    db: Session,
    user: User,
    *,
    project_id: int,
    document_id: int,
    owner_user_id: int | None = None,
    enforce_owner: bool = False,
    min_role: str | None = None,
) -> WorkspaceMember:
    """Authorize a document-scoped resource and optionally enforce its owner."""
    project = _require_project(db, project_id)
    document = _require_document(db, document_id)
    if document.project_id != project.id:
        raise ValidationError(f"Document {document_id} does not belong to project {project_id}")
    member = assert_workspace_member(
        db,
        user,
        project.workspace_id,
        min_role=min_role,
    )
    assert_document_assigned(
        db,
        user,
        member,
        project_id=project_id,
        document_id=document_id,
    )
    if enforce_owner and not has_assignment_bypass(user, member) and owner_user_id != user.id:
        raise ForbiddenError("Resource belongs to another annotator")
    return member


def assert_assigned_document_member(
    db: Session,
    user: User,
    document_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember:
    document = _require_document(db, document_id)
    member = assert_document_member(db, user, document_id, min_role=min_role)
    assert_document_assigned(
        db,
        user,
        member,
        project_id=document.project_id,
        document_id=document_id,
    )
    return member


def assert_assigned_document_member_if_exists(
    db: Session,
    user: User,
    document_id: int,
    *,
    min_role: str | None = None,
) -> WorkspaceMember | None:
    if db.get(Document, document_id) is None:
        return None
    return assert_assigned_document_member(db, user, document_id, min_role=min_role)


def assert_annotation_member_with_assignment(
    db: Session,
    user: User,
    annotation_id: int,
    *,
    min_role: str | None = None,
    require_mutable_assignment: bool = False,
) -> tuple[WorkspaceMember, TaskAssignment | None]:
    annotation = _require_annotation(db, annotation_id)
    member = assert_project_member(db, user, annotation.project_id, min_role=min_role)
    document = _require_document(db, annotation.document_id)
    if document.project_id != annotation.project_id:
        raise ValidationError("Annotation project and document do not match")
    evidence_block = (
        annotation.evidence_block if annotation.annotation_type == "evidence_block" else None
    )
    assignment = assert_task_assigned(
        db,
        user,
        member,
        project_id=annotation.project_id,
        document_id=annotation.document_id,
        annotation_type=annotation.annotation_type,
        target_version_id=(
            evidence_block.target_version_id if evidence_block is not None else None
        ),
        structure_version_id=(
            annotation.structure_version_id
            if annotation.structure_version_id is not None
            else (evidence_block.structure_version_id if evidence_block is not None else None)
        ),
        guideline_version_id=annotation.guideline_version_id,
        match_structure_version=True,
        match_guideline_version=True,
        require_mutable_assignment=require_mutable_assignment,
    )
    if not has_assignment_bypass(user, member) and annotation.annotator_user_id != user.id:
        raise ForbiddenError("Annotation belongs to another annotator")
    return member, assignment


def assert_annotation_member(
    db: Session,
    user: User,
    annotation_id: int,
    *,
    min_role: str | None = None,
    require_mutable_assignment: bool = False,
) -> WorkspaceMember:
    member, _assignment = assert_annotation_member_with_assignment(
        db,
        user,
        annotation_id,
        min_role=min_role,
        require_mutable_assignment=require_mutable_assignment,
    )
    return member


def assert_annotation_references_access(
    db: Session,
    user: User,
    *,
    project_id: int,
    document_id: int,
    annotation_ids: list[int | None] | tuple[int | None, ...],
) -> None:
    """Enforce private visibility for existing, in-scope annotation references.

    Missing or cross-scope references are left to domain validation so existing
    422 response semantics remain intact.
    """
    unavailable = False
    for annotation_id in annotation_ids:
        if annotation_id is None:
            continue
        annotation = db.get(Annotation, annotation_id)
        if annotation is None:
            unavailable = True
            continue
        try:
            assert_annotation_member(db, user, annotation.id)
        except (ForbiddenError, NotFoundError, ValidationError):
            unavailable = True
            continue
        if annotation.project_id != project_id or annotation.document_id != document_id:
            unavailable = True
    if unavailable:
        # Missing, cross-scope, and private foreign IDs intentionally share one
        # response so annotation identifiers cannot be probed as an existence
        # oracle by relation/correction payloads.
        raise ValidationError("One or more referenced annotations are unavailable")


def _route_int(request: Request, name: str) -> int:
    value = request.path_params.get(name)
    if value is None:
        value = request.query_params.get(name)
    if value is None:
        raise ValidationError(f"{name} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an integer") from exc


def require_project_access(
    min_role: str | None = None,
) -> Callable[..., WorkspaceMember]:
    def _dependency(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> WorkspaceMember:
        return assert_project_member(
            db,
            current_user,
            _route_int(request, "project_id"),
            min_role=min_role,
        )

    return _dependency


def require_document_access(
    min_role: str | None = None,
) -> Callable[..., WorkspaceMember]:
    def _dependency(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> WorkspaceMember:
        return assert_document_member(
            db,
            current_user,
            _route_int(request, "document_id"),
            min_role=min_role,
        )

    return _dependency


def require_assigned_document_access(
    min_role: str | None = None,
) -> Callable[..., WorkspaceMember]:
    def _dependency(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> WorkspaceMember:
        return assert_assigned_document_member(
            db,
            current_user,
            _route_int(request, "document_id"),
            min_role=min_role,
        )

    return _dependency


def require_guideline_access(
    min_role: str | None = None,
) -> Callable[..., WorkspaceMember]:
    return require_project_access(min_role=min_role)


def require_annotation_access(
    min_role: str | None = None,
) -> Callable[..., WorkspaceMember]:
    def _dependency(
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> WorkspaceMember:
        return assert_annotation_member(
            db,
            current_user,
            _route_int(request, "annotation_id"),
            min_role=min_role,
        )

    return _dependency
