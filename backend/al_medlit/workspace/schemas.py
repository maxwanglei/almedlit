from datetime import datetime

from pydantic import BaseModel, Field

from al_medlit.auth.schemas import PublicPassword


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class WorkspaceRead(BaseModel):
    id: int
    name: str
    kind: str
    join_code: str | None = None
    created_by: int | None = None
    capability_preset: str
    capability_overrides: list[str]

    model_config = {"from_attributes": True}


class WorkspaceMemberRead(BaseModel):
    id: int
    workspace_id: int
    user_id: int
    username: str
    display_name: str
    email: str | None
    is_active: bool
    role: str

    model_config = {"from_attributes": True}


class WorkspaceMemberUpdate(BaseModel):
    role: str


class MemberRead(BaseModel):
    user_id: int
    role: str

    model_config = {"from_attributes": True}


class RoleUpdate(BaseModel):
    role: str


class InviteCreate(BaseModel):
    role: str = "annotator"
    expires_minutes: int | None = Field(default=None, ge=60, le=43_200)


class InviteRead(BaseModel):
    id: int
    token: str
    role: str
    workspace_id: int
    expires_at: datetime | None

    model_config = {"from_attributes": True}


class InviteSummaryRead(BaseModel):
    id: int
    workspace_id: int
    role: str
    created_by: int
    created_by_username: str
    expires_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class InvitePreview(BaseModel):
    """Unauthenticated view of an open invite, for the acceptance page.

    Deliberately narrow: a token holder learns which workspace they were
    invited to and at what role, and nothing else. No workspace id, member
    list, or join code.
    """

    workspace_name: str
    workspace_kind: str
    role: str
    expires_at: datetime | None = None


class InviteAccept(BaseModel):
    username: str | None = Field(default=None, min_length=1, max_length=120)
    password: PublicPassword | None = None
    display_name: str = Field(default="", max_length=120)
    create_account: bool = True


class JoinRequestCreate(BaseModel):
    message: str | None = None


class JoinRequestRead(BaseModel):
    id: int
    workspace_id: int
    user_id: int
    username: str
    display_name: str
    email: str | None
    status: str
    message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class WorkspaceGovernanceRead(BaseModel):
    workspace_id: int
    workspace_kind: str
    join_code: str | None
    default_invite_expiry_minutes: int


class CapabilitiesRead(BaseModel):
    preset: str
    overrides: list[str]
    effective: list[str]
    blocked: dict[str, str]


class CapabilityUpdate(BaseModel):
    preset: str
    overrides: list[str] = Field(default_factory=list)
