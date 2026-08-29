from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from al_medlit.auth.schemas import PublicPassword

AccountActionPurpose = Literal["activation", "password_reset"]


class InstanceSettingsRead(BaseModel):
    allow_self_registration: bool
    default_invite_expiry_minutes: int
    account_action_expiry_minutes: int
    deployment_profile: str
    storage_backend: str
    storage_encryption: str
    task_execution: str
    jwt_lifetime_minutes: int


class InstanceSettingsUpdate(BaseModel):
    allow_self_registration: bool | None = None
    default_invite_expiry_minutes: int | None = Field(default=None, ge=60, le=43_200)
    account_action_expiry_minutes: int | None = Field(default=None, ge=15, le=1_440)


class AdminMembershipRead(BaseModel):
    workspace_id: int
    workspace_name: str
    workspace_kind: str
    role: str


class AdminUserSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    email: str | None
    is_active: bool
    is_initialized: bool
    is_superuser: bool
    last_login_at: datetime | None
    membership_count: int
    created_at: datetime


class AdminUserDetail(AdminUserSummary):
    memberships: list[AdminMembershipRead]


class AdminUserList(BaseModel):
    items: list[AdminUserSummary]
    total: int
    page: int
    page_size: int


class AdminUserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    display_name: str = Field(default="", max_length=120)
    email: str | None = Field(default=None, max_length=255)


class AdminUserStatusUpdate(BaseModel):
    is_active: bool


class AccountActionLink(BaseModel):
    url: str
    expires_at: datetime
    purpose: AccountActionPurpose


class AdminUserCreateResponse(BaseModel):
    user: AdminUserSummary
    action: AccountActionLink


class AccountActionPreview(BaseModel):
    purpose: AccountActionPurpose
    username: str
    display_name: str
    expires_at: datetime


class AccountActionComplete(BaseModel):
    password: PublicPassword


class AccountActionCompleteResponse(BaseModel):
    completed: bool = True
    purpose: AccountActionPurpose
    user_id: int = Field(exclude=True)
