import hashlib
import secrets
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.administration.events import record_admin_event
from al_medlit.administration.models import AccountActionToken, InstancePolicy
from al_medlit.administration.policy import (
    DEFAULT_ACCOUNT_ACTION_EXPIRY_MINUTES,
    DEFAULT_INVITE_EXPIRY_MINUTES,
    get_effective_policy,
)
from al_medlit.administration.schemas import (
    AccountActionCompleteResponse,
    AccountActionLink,
    AccountActionPreview,
    AdminMembershipRead,
    AdminUserCreate,
    AdminUserCreateResponse,
    AdminUserDetail,
    AdminUserList,
    AdminUserSummary,
    InstanceSettingsRead,
    InstanceSettingsUpdate,
)
from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token, hash_password
from al_medlit.core.config import settings
from al_medlit.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from al_medlit.project.models import Project, TaskAssignment
from al_medlit.workspace.models import Workspace, WorkspaceMember

ACCOUNT_ACTION_ACTIVATION = "activation"
ACCOUNT_ACTION_PASSWORD_RESET = "password_reset"
ACCOUNT_ACTION_PURPOSES = {
    ACCOUNT_ACTION_ACTIVATION,
    ACCOUNT_ACTION_PASSWORD_RESET,
}
MUTABLE_ASSIGNMENT_STATUSES = ("assigned", "in_progress", "blocked")
MUTABLE_ANNOTATION_ROUND_STATUSES = ("draft", "open")
UNUSABLE_PASSWORD_PREFIX = "!"


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _unusable_password_hash() -> str:
    return f"{UNUSABLE_PASSWORD_PREFIX}{secrets.token_urlsafe(32)}"


def _has_usable_password(user: User) -> bool:
    return not user.password_hash.startswith(UNUSABLE_PASSWORD_PREFIX)


def _lock_and_require_active_superuser(
    db: Session,
    actor_user_id: int,
    *,
    target_user_ids: Iterable[int] = (),
) -> tuple[User, list[User], dict[int, User]]:
    """Serialize and revalidate system-administration authority.

    Locking active superusers and every target in one identifier-ordered query
    provides a stable boundary for authority and last-superuser checks. It also
    gives all account and cross-service mutations one canonical User-row order.
    """

    target_ids = set(target_user_ids)
    authority_clause = and_(
        User.is_active.is_(True),
        User.is_superuser.is_(True),
    )
    lock_clause = (
        or_(authority_clause, User.id.in_(target_ids))
        if target_ids
        else authority_clause
    )
    locked_users = (
        db.query(User)
        .filter(lock_clause)
        .order_by(User.id)
        .populate_existing()
        .with_for_update()
        .all()
    )
    locked_by_id = {user.id: user for user in locked_users}
    actor = locked_by_id.get(actor_user_id)
    if actor is None or not actor.is_active or not actor.is_superuser:
        raise ForbiddenError("Active system administrator access is required")
    active_superusers = [
        user for user in locked_users if user.is_active and user.is_superuser
    ]
    return actor, active_superusers, locked_by_id


def _user_summary(db: Session, user: User, membership_count: int | None = None) -> AdminUserSummary:
    if membership_count is None:
        membership_count = (
            db.query(func.count(WorkspaceMember.id))
            .filter(WorkspaceMember.user_id == user.id)
            .scalar()
            or 0
        )
    return AdminUserSummary(
        id=user.id,
        username=user.username,
        display_name=user.display_name,
        email=user.email,
        is_active=user.is_active,
        is_initialized=_has_usable_password(user),
        is_superuser=user.is_superuser,
        last_login_at=user.last_login_at,
        membership_count=int(membership_count),
        created_at=user.created_at,
    )


def list_users(
    db: Session,
    *,
    search: str | None,
    is_active: bool | None,
    is_superuser: bool | None,
    workspace_id: int | None,
    page: int,
    page_size: int,
) -> AdminUserList:
    query = db.query(User)
    cleaned_search = search.strip() if search else ""
    if cleaned_search:
        pattern = f"%{cleaned_search}%"
        query = query.filter(
            or_(
                User.username.ilike(pattern),
                User.display_name.ilike(pattern),
                User.email.ilike(pattern),
            )
        )
    if is_active is not None:
        query = query.filter(User.is_active.is_(is_active))
    if is_superuser is not None:
        query = query.filter(User.is_superuser.is_(is_superuser))
    if workspace_id is not None:
        query = query.filter(
            User.memberships.any(WorkspaceMember.workspace_id == workspace_id)
        )

    total = query.order_by(None).count()
    rows = (
        query.outerjoin(WorkspaceMember, WorkspaceMember.user_id == User.id)
        .add_columns(func.count(WorkspaceMember.id).label("membership_count"))
        .group_by(User.id)
        .order_by(User.username, User.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return AdminUserList(
        items=[_user_summary(db, user, count) for user, count in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_user_detail(db: Session, user_id: int) -> AdminUserDetail:
    user = db.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found")
    membership_rows = (
        db.query(WorkspaceMember, Workspace)
        .join(Workspace, Workspace.id == WorkspaceMember.workspace_id)
        .filter(WorkspaceMember.user_id == user.id)
        .order_by(Workspace.name, Workspace.id)
        .all()
    )
    summary = _user_summary(db, user, len(membership_rows))
    return AdminUserDetail(
        **summary.model_dump(),
        memberships=[
            AdminMembershipRead(
                workspace_id=workspace.id,
                workspace_name=workspace.name,
                workspace_kind=workspace.kind,
                role=membership.role,
            )
            for membership, workspace in membership_rows
        ],
    )


def get_instance_settings(db: Session) -> InstanceSettingsRead:
    policy = get_effective_policy(db)
    return InstanceSettingsRead(
        allow_self_registration=policy.allow_self_registration,
        default_invite_expiry_minutes=policy.default_invite_expiry_minutes,
        account_action_expiry_minutes=policy.account_action_expiry_minutes,
        deployment_profile=settings.deployment_profile,
        storage_backend=settings.storage_backend,
        storage_encryption=settings.storage_encryption_mode,
        task_execution="eager" if settings.celery_task_always_eager else "worker",
        jwt_lifetime_minutes=settings.jwt_expire_minutes,
    )


def update_instance_settings(
    db: Session,
    *,
    actor_user_id: int,
    updates: InstanceSettingsUpdate,
) -> InstanceSettingsRead:
    _lock_and_require_active_superuser(db, actor_user_id)
    policy = (
        db.query(InstancePolicy)
        .filter(InstancePolicy.id == 1)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if policy is None:
        policy = InstancePolicy(
            id=1,
            allow_self_registration=None,
            default_invite_expiry_minutes=DEFAULT_INVITE_EXPIRY_MINUTES,
            account_action_expiry_minutes=DEFAULT_ACCOUNT_ACTION_EXPIRY_MINUTES,
        )
        db.add(policy)
        db.flush()

    before = get_effective_policy(db)
    fields = updates.model_fields_set
    if "allow_self_registration" in fields and updates.allow_self_registration is not None:
        policy.allow_self_registration = updates.allow_self_registration
    if (
        "default_invite_expiry_minutes" in fields
        and updates.default_invite_expiry_minutes is not None
    ):
        policy.default_invite_expiry_minutes = updates.default_invite_expiry_minutes
    if (
        "account_action_expiry_minutes" in fields
        and updates.account_action_expiry_minutes is not None
    ):
        policy.account_action_expiry_minutes = updates.account_action_expiry_minutes
    policy.updated_by = actor_user_id
    db.flush()
    after = get_effective_policy(db)
    record_admin_event(
        db,
        event_type="instance.policy_updated",
        actor_user_id=actor_user_id,
        details={
            "before": {
                "allow_self_registration": before.allow_self_registration,
                "default_invite_expiry_minutes": before.default_invite_expiry_minutes,
                "account_action_expiry_minutes": before.account_action_expiry_minutes,
            },
            "after": {
                "allow_self_registration": after.allow_self_registration,
                "default_invite_expiry_minutes": after.default_invite_expiry_minutes,
                "account_action_expiry_minutes": after.account_action_expiry_minutes,
            },
        },
    )
    return get_instance_settings(db)


def _revoke_unused_actions(
    db: Session,
    user_id: int,
    *,
    now: datetime,
    purpose: str | None = None,
    except_id: int | None = None,
) -> int:
    query = db.query(AccountActionToken).filter(
        AccountActionToken.user_id == user_id,
        AccountActionToken.consumed_at.is_(None),
        AccountActionToken.revoked_at.is_(None),
    )
    if purpose is not None:
        query = query.filter(AccountActionToken.purpose == purpose)
    if except_id is not None:
        query = query.filter(AccountActionToken.id != except_id)
    actions = query.order_by(AccountActionToken.id).with_for_update().all()
    for action in actions:
        action.revoked_at = now
    return len(actions)


def _issue_account_action_locked(
    db: Session,
    *,
    actor_user_id: int,
    user: User,
    purpose: str,
) -> AccountActionLink:
    if purpose not in ACCOUNT_ACTION_PURPOSES:
        raise ValueError(f"Unknown account action purpose: {purpose}")
    now = datetime.now(UTC)
    _revoke_unused_actions(db, user.id, now=now, purpose=purpose)
    raw_token = secrets.token_urlsafe(32)
    expiry_minutes = get_effective_policy(db).account_action_expiry_minutes
    expires_at = now + timedelta(minutes=expiry_minutes)
    action = AccountActionToken(
        user_id=user.id,
        purpose=purpose,
        token_hash=_token_hash(raw_token),
        created_by=actor_user_id,
        expires_at=expires_at,
    )
    db.add(action)
    db.flush()
    record_admin_event(
        db,
        event_type=f"account.{purpose}_link_created",
        actor_user_id=actor_user_id,
        target_user_id=user.id,
        details={"expires_at": expires_at.isoformat()},
    )
    return AccountActionLink(
        url=f"/account-actions/{raw_token}",
        expires_at=expires_at,
        purpose=purpose,
    )


def create_inactive_user(
    db: Session,
    *,
    actor_user_id: int,
    data: AdminUserCreate,
) -> AdminUserCreateResponse:
    _lock_and_require_active_superuser(db, actor_user_id)
    username = data.username.strip()
    if not username:
        raise ConflictError("Username is required")
    if db.query(User).filter(User.username == username).first() is not None:
        raise ConflictError(f"Username {username!r} is already taken")
    user = User(
        username=username,
        email=data.email.strip() if data.email else None,
        password_hash=_unusable_password_hash(),
        display_name=data.display_name.strip(),
        is_active=False,
        is_superuser=False,
    )
    db.add(user)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(f"Username {username!r} is already taken") from exc
    record_admin_event(
        db,
        event_type="account.created",
        actor_user_id=actor_user_id,
        target_user_id=user.id,
        details={"initial_status": "inactive"},
    )
    action = _issue_account_action_locked(
        db,
        actor_user_id=actor_user_id,
        user=user,
        purpose=ACCOUNT_ACTION_ACTIVATION,
    )
    return AdminUserCreateResponse(user=_user_summary(db, user, 0), action=action)


def issue_activation_link(
    db: Session,
    *,
    actor_user_id: int,
    user_id: int,
) -> AccountActionLink:
    _actor, _active_superusers, locked_users = _lock_and_require_active_superuser(
        db,
        actor_user_id,
        target_user_ids=(user_id,),
    )
    user = locked_users.get(user_id)
    if user is None:
        raise NotFoundError("User not found")
    if user.is_active or _has_usable_password(user):
        raise ConflictError("Activation links are only available for uninitialized accounts")
    return _issue_account_action_locked(
        db,
        actor_user_id=actor_user_id,
        user=user,
        purpose=ACCOUNT_ACTION_ACTIVATION,
    )


def issue_password_reset_link(
    db: Session,
    *,
    actor_user_id: int,
    user_id: int,
) -> AccountActionLink:
    _actor, _active_superusers, locked_users = _lock_and_require_active_superuser(
        db,
        actor_user_id,
        target_user_ids=(user_id,),
    )
    user = locked_users.get(user_id)
    if user is None:
        raise NotFoundError("User not found")
    if not user.is_active or not _has_usable_password(user):
        raise ConflictError("Password reset links require an active initialized account")
    return _issue_account_action_locked(
        db,
        actor_user_id=actor_user_id,
        user=user,
        purpose=ACCOUNT_ACTION_PASSWORD_RESET,
    )


def _withdraw_user_assignments(
    db: Session,
    *,
    user_id: int,
    actor_user_id: int,
    now: datetime,
) -> tuple[int, list[int]]:
    from al_medlit.workflow.models import AnnotationRound, ReviewCase

    membership_workspace_ids = {
        workspace_id
        for (workspace_id,) in db.query(WorkspaceMember.workspace_id)
        .filter(WorkspaceMember.user_id == user_id)
        .all()
    }
    assignment_workspace_ids = {
        workspace_id
        for (workspace_id,) in db.query(Project.workspace_id)
        .join(TaskAssignment, TaskAssignment.project_id == Project.id)
        .filter(
            TaskAssignment.assignee_user_id == user_id,
            TaskAssignment.status.in_(MUTABLE_ASSIGNMENT_STATUSES),
        )
        .all()
    }
    review_case_workspace_ids = {
        workspace_id
        for (workspace_id,) in db.query(Project.workspace_id)
        .join(ReviewCase, ReviewCase.project_id == Project.id)
        .filter(
            ReviewCase.assigned_to_user_id == user_id,
            ReviewCase.status == "open",
        )
        .all()
    }
    mutable_round_rows = (
        db.query(
            AnnotationRound.id,
            Project.workspace_id,
            AnnotationRound.annotator_user_ids,
        )
        .join(Project, Project.id == AnnotationRound.project_id)
        .filter(AnnotationRound.status.in_(MUTABLE_ANNOTATION_ROUND_STATUSES))
        .all()
    )
    annotation_round_ids = sorted(
        round_id
        for round_id, _workspace_id, annotator_user_ids in mutable_round_rows
        if user_id in (annotator_user_ids or [])
    )
    annotation_round_workspace_ids = {
        workspace_id
        for round_id, workspace_id, annotator_user_ids in mutable_round_rows
        if round_id in annotation_round_ids and user_id in (annotator_user_ids or [])
    }
    workspace_ids = sorted(
        membership_workspace_ids
        | assignment_workspace_ids
        | review_case_workspace_ids
        | annotation_round_workspace_ids
    )
    if workspace_ids:
        (
            db.query(Workspace)
            .filter(Workspace.id.in_(workspace_ids))
            .order_by(Workspace.id)
            .populate_existing()
            .with_for_update()
            .all()
        )

    assignments = (
        db.query(TaskAssignment)
        .join(Project, Project.id == TaskAssignment.project_id)
        .filter(
            TaskAssignment.assignee_user_id == user_id,
            TaskAssignment.status.in_(MUTABLE_ASSIGNMENT_STATUSES),
        )
        .order_by(TaskAssignment.id)
        .populate_existing()
        .with_for_update(of=TaskAssignment)
        .all()
    )
    for assignment in assignments:
        prior_status = assignment.status
        metadata = dict(assignment.metadata_ or {})
        metadata["account_deactivation"] = {
            "reason": "user_deactivated",
            "user_id": user_id,
            "deactivated_by_user_id": actor_user_id,
            "withdrawn_at": now.isoformat(),
            "prior_status": prior_status,
        }
        assignment.metadata_ = metadata
        assignment.status = "withdrawn"
    return len(assignments), annotation_round_ids


def _unassign_user_annotation_rounds(
    db: Session,
    *,
    annotation_round_ids: list[int],
    user_id: int,
    actor_user_id: int,
    now: datetime,
) -> list[int]:
    from al_medlit.workflow.models import AnnotationRound

    if not annotation_round_ids:
        return []
    round_rows = (
        db.query(AnnotationRound, Project.workspace_id)
        .join(Project, Project.id == AnnotationRound.project_id)
        .filter(
            AnnotationRound.id.in_(annotation_round_ids),
            AnnotationRound.status.in_(MUTABLE_ANNOTATION_ROUND_STATUSES),
        )
        .order_by(AnnotationRound.id)
        .populate_existing()
        .with_for_update(of=AnnotationRound)
        .all()
    )
    withdrawn_round_ids: list[int] = []
    for annotation_round, workspace_id in round_rows:
        assigned_user_ids = list(annotation_round.annotator_user_ids or [])
        if user_id not in assigned_user_ids:
            continue
        annotation_round.annotator_user_ids = [
            assigned_user_id
            for assigned_user_id in assigned_user_ids
            if assigned_user_id != user_id
        ]
        withdrawn_round_ids.append(annotation_round.id)
        record_admin_event(
            db,
            event_type="workflow.annotation_round_annotator_withdrawn",
            actor_user_id=actor_user_id,
            target_user_id=user_id,
            workspace_id=workspace_id,
            details={
                "reason": "user_deactivated",
                "annotation_round_id": annotation_round.id,
                "project_id": annotation_round.project_id,
                "round_status": annotation_round.status,
                "withdrawn_at": now.isoformat(),
            },
        )
    return withdrawn_round_ids


def _unassign_user_review_cases(
    db: Session,
    *,
    user_id: int,
    actor_user_id: int,
    now: datetime,
) -> int:
    from al_medlit.workflow.models import ReviewCase

    review_cases = (
        db.query(ReviewCase)
        .filter(
            ReviewCase.assigned_to_user_id == user_id,
            ReviewCase.status == "open",
        )
        .order_by(ReviewCase.id)
        .populate_existing()
        .with_for_update()
        .all()
    )
    for review_case in review_cases:
        resolution = dict(review_case.resolution or {})
        resolution["account_deactivation"] = {
            "reason": "user_deactivated",
            "user_id": user_id,
            "deactivated_by_user_id": actor_user_id,
            "unassigned_at": now.isoformat(),
        }
        review_case.resolution = resolution
        review_case.assigned_to_user_id = None
    return len(review_cases)


def set_user_status(
    db: Session,
    *,
    actor_user_id: int,
    user_id: int,
    is_active: bool,
) -> AdminUserDetail:
    _actor, active_superusers, locked_users = _lock_and_require_active_superuser(
        db,
        actor_user_id,
        target_user_ids=(user_id,),
    )
    user = locked_users.get(user_id)
    if user is None:
        raise NotFoundError("User not found")
    if user.is_active == is_active:
        return get_user_detail(db, user.id)

    if not is_active:
        if user.id == actor_user_id:
            raise ConflictError("You cannot deactivate your own account")
        if user.is_superuser and len(active_superusers) <= 1:
            raise ConflictError("The deployment must keep at least one active superuser")
        now = datetime.now(UTC)
        withdrawn_count, annotation_round_ids = _withdraw_user_assignments(
            db,
            user_id=user.id,
            actor_user_id=actor_user_id,
            now=now,
        )
        withdrawn_annotation_round_ids = _unassign_user_annotation_rounds(
            db,
            annotation_round_ids=annotation_round_ids,
            user_id=user.id,
            actor_user_id=actor_user_id,
            now=now,
        )
        unassigned_review_case_count = _unassign_user_review_cases(
            db,
            user_id=user.id,
            actor_user_id=actor_user_id,
            now=now,
        )
        revoked_action_count = _revoke_unused_actions(db, user.id, now=now)
        user.is_active = False
        user.session_version += 1
        record_admin_event(
            db,
            event_type="account.deactivated",
            actor_user_id=actor_user_id,
            target_user_id=user.id,
            details={
                "withdrawn_assignment_count": withdrawn_count,
                "withdrawn_annotation_round_count": len(
                    withdrawn_annotation_round_ids
                ),
                "withdrawn_annotation_round_ids": withdrawn_annotation_round_ids,
                "unassigned_review_case_count": unassigned_review_case_count,
                "revoked_account_action_count": revoked_action_count,
            },
        )
    else:
        if not _has_usable_password(user):
            raise ConflictError("Complete account activation before enabling this account")
        user.is_active = True
        record_admin_event(
            db,
            event_type="account.activated",
            actor_user_id=actor_user_id,
            target_user_id=user.id,
        )
    db.flush()
    return get_user_detail(db, user.id)


def _valid_account_action(db: Session, token: str, *, for_update: bool) -> AccountActionToken:
    query = db.query(AccountActionToken).filter(AccountActionToken.token_hash == _token_hash(token))
    if for_update:
        query = query.populate_existing().with_for_update()
    action = query.first()
    now = datetime.now(UTC)
    if (
        action is None
        or action.purpose not in ACCOUNT_ACTION_PURPOSES
        or action.consumed_at is not None
        or action.revoked_at is not None
        or _aware(action.expires_at) <= now
    ):
        raise NotFoundError("Account action link is invalid or expired")
    return action


def preview_account_action(db: Session, token: str) -> AccountActionPreview:
    action = _valid_account_action(db, token, for_update=False)
    user = db.get(User, action.user_id)
    if user is None:
        raise NotFoundError("Account action link is invalid or expired")
    if action.purpose == ACCOUNT_ACTION_ACTIVATION:
        valid_state = not user.is_active and not _has_usable_password(user)
    else:
        valid_state = user.is_active and _has_usable_password(user)
    if not valid_state:
        raise NotFoundError("Account action link is invalid or expired")
    return AccountActionPreview(
        purpose=action.purpose,
        username=user.username,
        display_name=user.display_name,
        expires_at=action.expires_at,
    )


def complete_account_action(
    db: Session,
    *,
    token: str,
    password: str,
) -> AccountActionCompleteResponse:
    # Every account lifecycle path locks User before AccountActionToken. The
    # first lookup only discovers the subject; after acquiring the User lock we
    # lock and fully revalidate the token so replacement, reset, and
    # deactivation cannot deadlock or consume stale authority.
    candidate = _valid_account_action(db, token, for_update=False)
    user = (
        db.query(User)
        .filter(User.id == candidate.user_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if user is None:
        raise NotFoundError("Account action link is invalid or expired")
    action = _valid_account_action(db, token, for_update=True)
    if action.user_id != user.id:
        raise NotFoundError("Account action link is invalid or expired")
    if action.purpose == ACCOUNT_ACTION_ACTIVATION:
        if user.is_active or _has_usable_password(user):
            raise NotFoundError("Account action link is invalid or expired")
        user.is_active = True
    elif not user.is_active or not _has_usable_password(user):
        raise NotFoundError("Account action link is invalid or expired")

    now = datetime.now(UTC)
    user.password_hash = hash_password(password)
    user.session_version += 1
    action.consumed_at = now
    _revoke_unused_actions(db, user.id, now=now, except_id=action.id)
    record_admin_event(
        db,
        event_type=f"account.{action.purpose}_completed",
        target_user_id=user.id,
    )
    db.flush()
    return AccountActionCompleteResponse(
        purpose=action.purpose,
        access_token=create_access_token(
            user.id,
            session_version=user.session_version,
        ),
    )
