from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.orm import Session

from al_medlit.auth import service as auth_service
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.schemas import UserCreate
from al_medlit.auth.security import create_access_token
from al_medlit.core import capabilities as cap
from al_medlit.core.config import settings
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError, UnauthorizedError
from al_medlit.workspace import capability_service, service
from al_medlit.workspace.dependencies import ROLE_ORDER, require_role
from al_medlit.workspace.models import WorkspaceJoinRequest, WorkspaceMember
from al_medlit.workspace.schemas import (
    CapabilitiesRead,
    CapabilityUpdate,
    InviteAccept,
    InviteCreate,
    InvitePreview,
    InviteRead,
    InviteSummaryRead,
    JoinRequestCreate,
    JoinRequestRead,
    RoleUpdate,
    WorkspaceCreate,
    WorkspaceGovernanceRead,
    WorkspaceMemberRead,
    WorkspaceRead,
)

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceRead)
def create_workspace(
    payload: WorkspaceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ws = service.create_team_workspace(db, current_user, payload.name)
    db.commit()
    db.refresh(ws)
    return ws


@router.get("", response_model=list[WorkspaceRead])
def list_my_workspaces(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return service.list_workspaces_for_user(db, current_user)


@router.get("/{workspace_id}/members", response_model=list[WorkspaceMemberRead])
def list_members(
    workspace_id: int,
    _member: WorkspaceMember = Depends(require_role("annotator")),
    db: Session = Depends(get_db),
):
    return service.list_members(db, workspace_id)


@router.patch("/{workspace_id}/members/{user_id}", response_model=WorkspaceMemberRead)
def update_member_role(
    workspace_id: int,
    user_id: int,
    payload: RoleUpdate,
    admin: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    member = service.change_role(
        db,
        workspace_id,
        user_id,
        role=payload.role,
        actor_user_id=admin.user_id,
    )
    db.commit()
    db.refresh(member)
    return member


@router.delete("/{workspace_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_member(
    workspace_id: int,
    user_id: int,
    admin: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    service.remove_member(
        db,
        workspace_id,
        user_id,
        actor_user_id=admin.user_id,
    )
    db.commit()


@router.post("/{workspace_id}/invites", response_model=InviteRead)
def create_invite(
    workspace_id: int,
    payload: InviteCreate,
    member: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    invite = service.create_invite(
        db,
        workspace_id,
        created_by=member.user_id,
        role=payload.role,
        expires_minutes=payload.expires_minutes,
    )
    db.commit()
    db.refresh(invite)
    return invite


@router.get("/{workspace_id}/invites", response_model=list[InviteSummaryRead])
def list_invites(
    workspace_id: int,
    _member: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    return service.list_open_invites(db, workspace_id)


@router.delete(
    "/{workspace_id}/invites/{invite_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def revoke_invite(
    workspace_id: int,
    invite_id: int,
    admin: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    service.revoke_invite(
        db,
        workspace_id,
        invite_id,
        actor_user_id=admin.user_id,
    )
    db.commit()


@router.get("/{workspace_id}/governance", response_model=WorkspaceGovernanceRead)
def get_workspace_governance(
    workspace_id: int,
    _member: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    return service.workspace_governance(db, workspace_id)


@router.post(
    "/{workspace_id}/join-code/rotate",
    response_model=WorkspaceGovernanceRead,
)
def rotate_join_code(
    workspace_id: int,
    admin: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    service.rotate_join_code(
        db,
        workspace_id,
        actor_user_id=admin.user_id,
    )
    db.commit()
    return service.workspace_governance(db, workspace_id)


@router.post("/by-code/{join_code}/join-requests", response_model=JoinRequestRead)
def apply_to_join(
    join_code: str,
    payload: JoinRequestCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ws = service.get_workspace_by_join_code(db, join_code)
    req = service.create_join_request(db, ws.id, current_user.id, message=payload.message)
    db.commit()
    db.refresh(req)
    return req


@router.get("/{workspace_id}/join-requests", response_model=list[JoinRequestRead])
def list_join_requests(
    workspace_id: int,
    _member: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    return service.list_pending_join_requests(db, workspace_id)


def _capabilities_read(db: Session, workspace_id: int) -> CapabilitiesRead:
    ws = service.require_workspace(db, workspace_id)
    effective = capability_service.effective_capabilities_for_workspace(db, workspace_id)
    selected = cap.close_dependencies(
        cap.preset_keys(ws.capability_preset, ws.capability_overrides or [])
    )
    blocked: dict[str, str] = {}
    for key in selected - effective:
        capability = cap.CAPABILITIES[key]
        if capability.requires_profile == "lab" and settings.deployment_profile != "lab":
            blocked[key] = "requires the lab deployment profile"
        elif capability.requires_team and ws.kind != "team":
            blocked[key] = "available only in team workspaces"

    return CapabilitiesRead(
        preset=ws.capability_preset,
        overrides=list(ws.capability_overrides or []),
        effective=sorted(effective),
        blocked=blocked,
    )


@router.get("/{workspace_id}/capabilities", response_model=CapabilitiesRead)
def get_capabilities(
    workspace_id: int,
    _member: WorkspaceMember = Depends(require_role("annotator")),
    db: Session = Depends(get_db),
):
    return _capabilities_read(db, workspace_id)


@router.patch("/{workspace_id}/capability", response_model=CapabilitiesRead)
def update_capability(
    workspace_id: int,
    payload: CapabilityUpdate,
    admin: WorkspaceMember = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    capability_service.set_capability(
        db,
        workspace_id,
        preset=payload.preset,
        overrides=payload.overrides,
        actor_user_id=admin.user_id,
    )
    db.commit()
    return _capabilities_read(db, workspace_id)


invite_router = APIRouter(prefix="/invites", tags=["invites"])


@invite_router.get("/{token}", response_model=InvitePreview)
def preview_invite(token: str, db: Session = Depends(get_db)):
    """Describe an open invite so the acceptance page can name the workspace.

    Unauthenticated by necessity — the invitee may not have an account yet.
    ``get_open_invite`` already rejects used and expired tokens, and the public
    edge throttles this route alongside the other unauthenticated invite and
    auth endpoints.
    """
    invite = service.get_open_invite(db, token)
    workspace = service.require_workspace(db, invite.workspace_id)
    return InvitePreview(
        workspace_name=workspace.name,
        workspace_kind=workspace.kind,
        role=invite.role,
        expires_at=invite.expires_at,
    )


@invite_router.post("/{token}/accept")
def accept_invite(
    token: str,
    payload: InviteAccept,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    invite = service.get_open_invite(db, token)
    if authorization and authorization.lower().startswith("bearer "):
        user = get_current_user(authorization=authorization, db=db)
    elif payload.username and payload.password:
        user = auth_service.register_user(
            db,
            UserCreate(
                username=payload.username,
                password=payload.password,
                display_name=payload.display_name,
            ),
        )
    else:
        raise UnauthorizedError("Provide a bearer token or username+password to accept")

    service.accept_invite(db, invite, user)
    db.commit()
    return {
        "access_token": create_access_token(
            str(user.id),
            session_version=user.session_version,
        ),
        "token_type": "bearer",
    }


join_request_router = APIRouter(prefix="/join-requests", tags=["join-requests"])


def _require_admin_for_request(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkspaceJoinRequest:
    req = db.get(WorkspaceJoinRequest, request_id)
    if req is None:
        raise NotFoundError("Join request not found")
    if current_user.is_superuser:
        return req
    member = service.get_member(db, req.workspace_id, current_user.id)
    if member is None or ROLE_ORDER[member.role] < ROLE_ORDER["admin"]:
        raise ForbiddenError("Insufficient role to decide join requests")
    return req


@join_request_router.post("/{request_id}/approve", response_model=JoinRequestRead)
def approve_join_request(
    request_id: int,
    _req: WorkspaceJoinRequest = Depends(_require_admin_for_request),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    decided = service.decide_join_request(
        db,
        request_id,
        approve=True,
        decided_by=current_user.id,
    )
    db.commit()
    db.refresh(decided)
    return decided


@join_request_router.post("/{request_id}/reject", response_model=JoinRequestRead)
def reject_join_request(
    request_id: int,
    _req: WorkspaceJoinRequest = Depends(_require_admin_for_request),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    decided = service.decide_join_request(
        db,
        request_id,
        approve=False,
        decided_by=current_user.id,
    )
    db.commit()
    db.refresh(decided)
    return decided
