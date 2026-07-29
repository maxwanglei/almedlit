from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

WorkspaceKind = Literal["individual", "team"]
WorkspaceRole = Literal["annotator", "trainer", "manager", "admin"]
PUBLIC_PASSWORD_MIN_BYTES = 12
PUBLIC_PASSWORD_MAX_BYTES = 72


def validate_public_password(password: str) -> str:
    byte_length = len(password.encode("utf-8"))
    if byte_length < PUBLIC_PASSWORD_MIN_BYTES:
        raise ValueError(f"Password must be at least {PUBLIC_PASSWORD_MIN_BYTES} UTF-8 bytes")
    if byte_length > PUBLIC_PASSWORD_MAX_BYTES:
        raise ValueError(f"Password must be at most {PUBLIC_PASSWORD_MAX_BYTES} UTF-8 bytes")
    return password


PublicPassword = Annotated[str, AfterValidator(validate_public_password)]


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1)
    display_name: str = Field(default="", max_length=120)
    email: str | None = Field(default=None, max_length=255)


class RegistrationRequest(UserCreate):
    password: PublicPassword
    workspace_kind: WorkspaceKind = "individual"
    workspace_name: str | None = Field(default=None, max_length=255)


class LoginRequest(BaseModel):
    username: str
    password: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    email: str | None
    is_active: bool
    is_superuser: bool


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class WorkspaceMembershipRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    workspace_id: int
    role: str


class MembershipRead(WorkspaceMembershipRead):
    workspace_name: str
    workspace_kind: str


class MeResponse(BaseModel):
    user: UserRead
    memberships: list[MembershipRead]


# Backward-compatible names used by the existing router/frontend code.
UserLogin = LoginRequest
TokenRead = TokenResponse
