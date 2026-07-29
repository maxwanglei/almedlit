import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from al_medlit.auth.models import User
from al_medlit.auth.schemas import MembershipRead, MeResponse, UserRead
from al_medlit.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from al_medlit.workspace.models import (
    Workspace,
    WorkspaceInvite,
    WorkspaceJoinRequest,
    WorkspaceMember,
)

VALID_ROLES = ("annotator", "trainer", "manager", "admin")
ROLE_RANK = {role: rank for rank, role in enumerate(VALID_ROLES)}


def _validate_role(role: str) -> None:
    if role not in VALID_ROLES:
        raise ConflictError(f"Unknown role {role!r}")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _new_join_code() -> str:
    return secrets.token_urlsafe(12)[:40]


def lock_workspace_for_update(db: Session, workspace_id: int) -> Workspace:
    """Lock and refresh the stable row used to serialize workspace changes."""

    workspace = (
        db.query(Workspace)
        .filter(Workspace.id == workspace_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if workspace is None:
        raise NotFoundError("Workspace not found")
    return workspace


def require_actor_role_after_workspace_lock(
    db: Session,
    workspace_id: int,
    *,
    actor_user_id: int,
    minimum_role: str,
) -> WorkspaceMember | None:
    """Revalidate a workspace mutation actor after its workspace is locked.

    Route dependencies provide an early rejection, but their membership object
    can be stale by the time a request waits for a concurrent role change. All
    membership-changing callers acquire ``lock_workspace_for_update`` first,
    so this refreshed read remains stable through the current transaction.
    Superusers are authorized without requiring a workspace membership.
    """

    _validate_role(minimum_role)
    actor = (
        db.query(User)
        .filter(User.id == actor_user_id)
        .populate_existing()
        .first()
    )
    if actor is None or not actor.is_active:
        raise ForbiddenError("Workspace mutation actor is not active")
    if actor.is_superuser:
        return None
    member = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == actor_user_id,
        )
        .populate_existing()
        .first()
    )
    if member is None or ROLE_RANK.get(member.role, -1) < ROLE_RANK[minimum_role]:
        raise ForbiddenError("Insufficient workspace role")
    return member


def _lock_member(
    db: Session,
    workspace_id: int,
    user_id: int,
) -> WorkspaceMember | None:
    return (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )


def _require_team_workspace(workspace: Workspace) -> None:
    if workspace.kind != "team":
        raise ConflictError("Individual workspaces cannot have additional members")


def create_personal_workspace(db: Session, user: User) -> Workspace:
    # The creator of a personal workspace always owns it outright, so the role is
    # fixed to "admin" rather than accepted from the caller. Roles are a
    # multi-person permission ladder and are meaningless in a solo workspace.
    label = user.display_name or user.username
    # Workspace names are scoped to their creator, so two users may use the
    # same friendly display name without competing for a global namespace.
    # Keep the SAVEPOINT so a duplicate personal-workspace attempt does not
    # roll back a surrounding registration transaction.
    ws = Workspace(name=f"{label}'s Workspace", kind="individual", created_by=user.id)
    try:
        with db.begin_nested():
            db.add(ws)
            db.flush()
    except IntegrityError as exc:
        raise ConflictError("User already has a personal workspace with this name") from exc
    add_member(db, ws.id, user.id, role="admin")
    return ws


def create_team_workspace(db: Session, user: User, name: str) -> Workspace:
    workspace_name = name.strip()
    if not workspace_name:
        label = user.display_name or user.username
        workspace_name = f"{label}'s Team"
    ws = Workspace(
        name=workspace_name,
        kind="team",
        created_by=user.id,
        join_code=_new_join_code(),
    )
    try:
        with db.begin_nested():
            db.add(ws)
            db.flush()
    except IntegrityError as exc:
        raise ConflictError(
            f"You already have a team workspace named {workspace_name!r}"
        ) from exc
    add_member(db, ws.id, user.id, role="admin")
    return ws


def ensure_default_workspace(db: Session) -> Workspace:
    ws = (
        db.query(Workspace)
        .filter(Workspace.name == "Default", Workspace.kind == "team")
        .order_by(Workspace.id)
        .first()
    )
    if ws is not None:
        if ws.capability_preset != "full":
            ws.capability_preset = "full"
            ws.capability_overrides = []
            db.flush()
        return ws

    ws = Workspace(
        name="Default",
        kind="team",
        created_by=None,
        capability_preset="full",
        capability_overrides=[],
    )
    db.add(ws)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        ws = (
            db.query(Workspace)
            .filter(Workspace.name == "Default", Workspace.kind == "team")
            .first()
        )
        if ws is None:
            raise
    return ws


def get_member(db: Session, workspace_id: int, user_id: int) -> WorkspaceMember | None:
    return (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.user_id == user_id,
        )
        .first()
    )


def add_member(db: Session, workspace_id: int, user_id: int, *, role: str) -> WorkspaceMember:
    _validate_role(role)
    workspace = require_workspace(db, workspace_id)
    if db.get(User, user_id) is None:
        raise NotFoundError("User not found")
    if workspace.kind == "individual":
        if workspace.created_by != user_id:
            raise ConflictError("Individual workspaces cannot have additional members")
        if role != "admin":
            raise ConflictError("An individual workspace owner must remain an admin")
    if get_member(db, workspace_id, user_id) is not None:
        raise ConflictError("User is already a member of this workspace")
    member = WorkspaceMember(workspace_id=workspace_id, user_id=user_id, role=role)
    try:
        with db.begin_nested():
            db.add(member)
            db.flush()
    except IntegrityError as exc:
        raise ConflictError("User is already a member of this workspace") from exc
    return member


def _admin_count(db: Session, workspace_id: int) -> int:
    return (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.role == "admin",
        )
        .count()
    )


def change_role(
    db: Session,
    workspace_id: int,
    user_id: int,
    *,
    role: str,
    actor_user_id: int,
) -> WorkspaceMember:
    _validate_role(role)
    # Serialize membership transitions per workspace on PostgreSQL so two
    # concurrent admin demotions cannot both observe the same stale count. The
    # actor check must happen after this lock; the dependency's earlier role
    # read may have waited behind another request and become stale.
    lock_workspace_for_update(db, workspace_id)
    require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=actor_user_id,
        minimum_role="admin",
    )
    member = _lock_member(db, workspace_id, user_id)
    if member is None:
        raise NotFoundError("Membership not found")
    if member.role == "admin" and role != "admin" and _admin_count(db, workspace_id) <= 1:
        raise ConflictError("A team workspace must keep at least one admin")
    member.role = role
    db.flush()
    return member


def remove_member(
    db: Session,
    workspace_id: int,
    user_id: int,
    *,
    actor_user_id: int,
) -> None:
    # Use the same deterministic lock as change_role to protect the last-admin
    # invariant across concurrent removals and demotions. Assignment creation
    # also takes this lock before validating assignee membership, preventing a
    # removal and a new assignment from both succeeding on stale observations.
    lock_workspace_for_update(db, workspace_id)
    require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=actor_user_id,
        minimum_role="admin",
    )
    member = _lock_member(db, workspace_id, user_id)
    if member is None:
        raise NotFoundError("Membership not found")
    if member.role == "admin" and _admin_count(db, workspace_id) <= 1:
        raise ConflictError("A team workspace must keep at least one admin")

    # Preserve the removed member's annotation history while ensuring no work
    # remains misleadingly writable after their workspace access is revoked.
    # Assignment creation/update takes the workspace lock first as well, so the
    # ordered assignment locks below form one atomic boundary with offboarding.
    from al_medlit.project.models import Project, TaskAssignment

    mutable_assignments = (
        db.query(TaskAssignment)
        .join(Project, Project.id == TaskAssignment.project_id)
        .filter(
            Project.workspace_id == workspace_id,
            TaskAssignment.assignee_user_id == user_id,
            TaskAssignment.status.in_(("assigned", "in_progress", "blocked")),
        )
        .order_by(TaskAssignment.id)
        .populate_existing()
        .with_for_update(of=TaskAssignment)
        .all()
    )
    withdrawn_at = datetime.now(UTC).isoformat()
    for assignment in mutable_assignments:
        prior_status = assignment.status
        metadata = dict(assignment.metadata_ or {})
        metadata["workspace_offboarding"] = {
            "reason": "workspace_member_removed",
            "workspace_id": workspace_id,
            "removed_user_id": user_id,
            "removed_by_user_id": actor_user_id,
            "withdrawn_at": withdrawn_at,
            "prior_status": prior_status,
        }
        assignment.metadata_ = metadata
        assignment.status = "withdrawn"

    db.delete(member)
    db.flush()


def list_members(db: Session, workspace_id: int) -> list[WorkspaceMember]:
    return (
        db.query(WorkspaceMember)
        .options(joinedload(WorkspaceMember.user))
        .filter(WorkspaceMember.workspace_id == workspace_id)
        .order_by(WorkspaceMember.id)
        .all()
    )


def list_workspaces_for_user(db: Session, user: User) -> list[Workspace]:
    memberships = (
        db.query(WorkspaceMember)
        .filter(WorkspaceMember.user_id == user.id)
        .order_by(WorkspaceMember.id)
        .all()
    )
    workspace_ids = [membership.workspace_id for membership in memberships]
    if not workspace_ids:
        return []
    return db.query(Workspace).filter(Workspace.id.in_(workspace_ids)).order_by(Workspace.id).all()


def me_response(db: Session, user: User) -> MeResponse:
    memberships = (
        db.query(WorkspaceMember)
        .join(Workspace)
        .filter(WorkspaceMember.user_id == user.id)
        .order_by(WorkspaceMember.id)
        .all()
    )
    return MeResponse(
        user=UserRead.model_validate(user),
        memberships=[
            MembershipRead(
                workspace_id=membership.workspace_id,
                workspace_name=membership.workspace.name,
                workspace_kind=membership.workspace.kind,
                role=membership.role,
            )
            for membership in memberships
        ],
    )


def require_workspace(db: Session, workspace_id: int) -> Workspace:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise NotFoundError("Workspace not found")
    return ws


def create_invite(
    db: Session,
    workspace_id: int,
    *,
    created_by: int,
    role: str,
    expires_minutes: int | None,
) -> WorkspaceInvite:
    _validate_role(role)
    # Serialize the inviter-role check with membership changes. After waiting
    # for the workspace lock, populate_existing ensures this request cannot use
    # a role cached by FastAPI's earlier authorization dependency.
    workspace = lock_workspace_for_update(db, workspace_id)
    _require_team_workspace(workspace)
    creator_member = require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=created_by,
        minimum_role="admin",
    )
    if creator_member is not None:
        if ROLE_RANK[role] > ROLE_RANK.get(creator_member.role, -1):
            raise ForbiddenError("Cannot invite a member with a higher role")
    expires_at = (
        datetime.now(UTC) + timedelta(minutes=expires_minutes)
        if expires_minutes is not None
        else None
    )
    invite = WorkspaceInvite(
        workspace_id=workspace_id,
        token=secrets.token_urlsafe(24),
        role=role,
        created_by=created_by,
        expires_at=expires_at,
    )
    db.add(invite)
    db.flush()
    return invite


def get_open_invite(db: Session, token: str) -> WorkspaceInvite:
    invite = db.query(WorkspaceInvite).filter(WorkspaceInvite.token == token).first()
    if invite is None or invite.accepted_at is not None:
        raise NotFoundError("Invite is invalid or already used")
    if invite.expires_at is not None and _aware(invite.expires_at) < datetime.now(UTC):
        raise NotFoundError("Invite has expired")
    return invite


def accept_invite(
    db: Session,
    invite: WorkspaceInvite,
    user: User,
) -> WorkspaceMember:
    # Membership and role changes serialize on the workspace row. Take that
    # canonical lock before the invite lock so acceptance cannot race an
    # inviter demotion/removal and deadlock with other workspace mutations.
    workspace = lock_workspace_for_update(db, invite.workspace_id)
    locked_invite = (
        db.query(WorkspaceInvite)
        .filter(
            WorkspaceInvite.id == invite.id,
            WorkspaceInvite.workspace_id == workspace.id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if locked_invite is None or locked_invite.accepted_at is not None:
        raise NotFoundError("Invite is invalid or already used")
    if locked_invite.expires_at is not None and _aware(locked_invite.expires_at) < datetime.now(
        UTC
    ):
        raise NotFoundError("Invite has expired")
    _require_team_workspace(workspace)

    # An invite is delegated authority, not a durable privilege grant. Recheck
    # the creator at acceptance time so removing, demoting, or deactivating the
    # inviter immediately invalidates outstanding invites. An active superuser
    # remains authorized without a workspace membership.
    inviter = (
        db.query(User)
        .filter(User.id == locked_invite.created_by)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if inviter is None or not inviter.is_active:
        raise ForbiddenError("Invite is no longer authorized by its creator")
    if not inviter.is_superuser:
        inviter_member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == inviter.id,
            )
            .populate_existing()
            .first()
        )
        if (
            inviter_member is None
            or ROLE_RANK.get(inviter_member.role, -1) < ROLE_RANK["admin"]
            or ROLE_RANK.get(locked_invite.role, len(ROLE_RANK))
            > ROLE_RANK.get(inviter_member.role, -1)
        ):
            raise ForbiddenError("Invite is no longer authorized by its creator")

    member = get_member(db, locked_invite.workspace_id, user.id)
    if member is None:
        member = add_member(
            db,
            locked_invite.workspace_id,
            user.id,
            role=locked_invite.role,
        )
    locked_invite.accepted_at = datetime.now(UTC)
    locked_invite.accepted_by = user.id
    db.flush()
    return member


def get_workspace_by_join_code(db: Session, join_code: str) -> Workspace:
    ws = db.query(Workspace).filter(Workspace.join_code == join_code).first()
    if ws is None:
        raise NotFoundError("No workspace found for that join code")
    return ws


def create_join_request(
    db: Session,
    workspace_id: int,
    user_id: int,
    *,
    message: str | None,
) -> WorkspaceJoinRequest:
    # Serialize with approval and membership changes so an approved member
    # cannot race a second pending request into existence.
    workspace = lock_workspace_for_update(db, workspace_id)
    _require_team_workspace(workspace)
    if get_member(db, workspace_id, user_id) is not None:
        raise ConflictError("You are already a member of this workspace")
    existing = (
        db.query(WorkspaceJoinRequest)
        .filter(
            WorkspaceJoinRequest.workspace_id == workspace_id,
            WorkspaceJoinRequest.user_id == user_id,
            WorkspaceJoinRequest.status == "pending",
        )
        .first()
    )
    if existing is not None:
        return existing
    req = WorkspaceJoinRequest(
        workspace_id=workspace_id,
        user_id=user_id,
        message=message,
    )
    try:
        with db.begin_nested():
            db.add(req)
            db.flush()
    except IntegrityError:
        # A concurrent request won the partial unique index. Treat applying to
        # join as idempotent and return that request after its transaction wins.
        existing = (
            db.query(WorkspaceJoinRequest)
            .filter(
                WorkspaceJoinRequest.workspace_id == workspace_id,
                WorkspaceJoinRequest.user_id == user_id,
                WorkspaceJoinRequest.status == "pending",
            )
            .first()
        )
        if existing is not None:
            return existing
        raise ConflictError("A pending join request already exists") from None
    return req


def list_pending_join_requests(
    db: Session,
    workspace_id: int,
) -> list[WorkspaceJoinRequest]:
    require_workspace(db, workspace_id)
    return (
        db.query(WorkspaceJoinRequest)
        .filter(
            WorkspaceJoinRequest.workspace_id == workspace_id,
            WorkspaceJoinRequest.status == "pending",
        )
        .order_by(WorkspaceJoinRequest.id)
        .all()
    )


def decide_join_request(
    db: Session,
    request_id: int,
    *,
    approve: bool,
    decided_by: int,
) -> WorkspaceJoinRequest:
    candidate = db.get(WorkspaceJoinRequest, request_id)
    if candidate is None:
        raise NotFoundError("Join request not found")
    workspace = lock_workspace_for_update(db, candidate.workspace_id)
    _require_team_workspace(workspace)
    decider = db.get(User, decided_by)
    if decider is None:
        raise NotFoundError("Join request decider not found")
    if not decider.is_superuser:
        decider_member = (
            db.query(WorkspaceMember)
            .filter(
                WorkspaceMember.workspace_id == workspace.id,
                WorkspaceMember.user_id == decided_by,
            )
            .populate_existing()
            .first()
        )
        if (
            decider_member is None
            or ROLE_RANK.get(decider_member.role, -1) < ROLE_RANK["admin"]
        ):
            raise ForbiddenError("Insufficient role to decide join requests")
    req = (
        db.query(WorkspaceJoinRequest)
        .filter(
            WorkspaceJoinRequest.id == request_id,
            WorkspaceJoinRequest.workspace_id == workspace.id,
        )
        .populate_existing()
        .with_for_update()
        .first()
    )
    if req is None:
        raise NotFoundError("Join request not found")
    if req.status != "pending":
        raise ConflictError("Join request already decided")
    req.status = "approved" if approve else "rejected"
    req.decided_by = decided_by
    req.decided_at = datetime.now(UTC)
    if approve and get_member(db, req.workspace_id, req.user_id) is None:
        add_member(db, req.workspace_id, req.user_id, role="annotator")
    db.flush()
    return req
