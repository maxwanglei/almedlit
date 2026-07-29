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
    expires_minutes: int | None = None


class InviteRead(BaseModel):
    token: str
    role: str
    workspace_id: int

    model_config = {"from_attributes": True}


class InviteAccept(BaseModel):
    username: str | None = Field(default=None, min_length=1, max_length=120)
    password: PublicPassword | None = None
    display_name: str = Field(default="", max_length=120)


class JoinRequestCreate(BaseModel):
    message: str | None = None


class JoinRequestRead(BaseModel):
    id: int
    workspace_id: int
    user_id: int
    status: str
    message: str | None

    model_config = {"from_attributes": True}


class CapabilitiesRead(BaseModel):
    preset: str
    overrides: list[str]
    effective: list[str]
    blocked: dict[str, str]


class CapabilityUpdate(BaseModel):
    preset: str
    overrides: list[str] = Field(default_factory=list)
