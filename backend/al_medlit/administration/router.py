from typing import Literal

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.administration import service
from al_medlit.administration.dependencies import require_active_superuser
from al_medlit.administration.schemas import (
    AccountActionComplete,
    AccountActionCompleteResponse,
    AccountActionLink,
    AccountActionPreview,
    AdminUserCreate,
    AdminUserCreateResponse,
    AdminUserDetail,
    AdminUserList,
    AdminUserStatusUpdate,
    InstanceSettingsRead,
    InstanceSettingsUpdate,
)
from al_medlit.auth.models import User
from al_medlit.core.database import get_db

admin_router = APIRouter(prefix="/admin", tags=["system-administration"])
account_action_router = APIRouter(prefix="/account-actions", tags=["account-actions"])


@admin_router.get("/settings", response_model=InstanceSettingsRead)
def get_settings(
    _admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    return service.get_instance_settings(db)


@admin_router.patch("/settings", response_model=InstanceSettingsRead)
def patch_settings(
    payload: InstanceSettingsUpdate,
    admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    result = service.update_instance_settings(
        db,
        actor_user_id=admin.id,
        updates=payload,
    )
    db.commit()
    return result


@admin_router.get("/users", response_model=AdminUserList)
def list_users(
    search: str | None = None,
    is_active: bool | None = None,
    status_filter: Literal["active", "inactive"] | None = Query(
        default=None,
        alias="status",
    ),
    is_superuser: bool | None = None,
    workspace_id: int | None = None,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    _admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    if is_active is None and status_filter is not None:
        is_active = status_filter == "active"
    return service.list_users(
        db,
        search=search,
        is_active=is_active,
        is_superuser=is_superuser,
        workspace_id=workspace_id,
        page=page,
        page_size=page_size,
    )


@admin_router.post(
    "/users",
    response_model=AdminUserCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_user(
    payload: AdminUserCreate,
    admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    result = service.create_inactive_user(
        db,
        actor_user_id=admin.id,
        data=payload,
    )
    db.commit()
    return result


@admin_router.get("/users/{user_id}", response_model=AdminUserDetail)
def get_user(
    user_id: int,
    _admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    return service.get_user_detail(db, user_id)


@admin_router.patch("/users/{user_id}/status", response_model=AdminUserDetail)
def patch_user_status(
    user_id: int,
    payload: AdminUserStatusUpdate,
    admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    result = service.set_user_status(
        db,
        actor_user_id=admin.id,
        user_id=user_id,
        is_active=payload.is_active,
    )
    db.commit()
    return result


@admin_router.post("/users/{user_id}/activation-link", response_model=AccountActionLink)
def create_activation_link(
    user_id: int,
    admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    action = service.issue_activation_link(
        db,
        actor_user_id=admin.id,
        user_id=user_id,
    )
    db.commit()
    return action


@admin_router.post("/users/{user_id}/password-reset-link", response_model=AccountActionLink)
def create_password_reset_link(
    user_id: int,
    admin: User = Depends(require_active_superuser),
    db: Session = Depends(get_db),
):
    action = service.issue_password_reset_link(
        db,
        actor_user_id=admin.id,
        user_id=user_id,
    )
    db.commit()
    return action


@account_action_router.get("/{token}", response_model=AccountActionPreview)
def preview_account_action(token: str, db: Session = Depends(get_db)):
    return service.preview_account_action(db, token)


@account_action_router.post("/{token}", response_model=AccountActionCompleteResponse)
def complete_account_action(
    token: str,
    payload: AccountActionComplete,
    db: Session = Depends(get_db),
):
    result = service.complete_account_action(
        db,
        token=token,
        password=payload.password,
    )
    db.commit()
    return result
