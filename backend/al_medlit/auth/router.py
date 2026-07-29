from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from al_medlit.auth import service
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.schemas import LoginRequest, MeResponse, RegistrationRequest, TokenResponse
from al_medlit.auth.security import create_access_token
from al_medlit.core.config import settings
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, UnauthorizedError
from al_medlit.workspace import service as workspace_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
def register(payload: RegistrationRequest, db: Session = Depends(get_db)):
    if not settings.allow_self_registration:
        raise ForbiddenError("Self-registration is disabled on this deployment")
    try:
        user = service.register_user(db, payload)
        if payload.workspace_kind == "team":
            workspace_service.create_team_workspace(
                db,
                user,
                payload.workspace_name or "",
            )
        else:
            workspace_service.create_personal_workspace(db, user)
        db.commit()
    except Exception:
        # User + initial workspace are one registration unit. This explicit
        # rollback is important for public web deployments and for SQLite local
        # mode, where request-session close should not be the atomicity boundary.
        db.rollback()
        raise
    db.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = service.authenticate_user(db, payload.username, payload.password)
    if user is None:
        raise UnauthorizedError("Invalid username or password")
    # authenticate_user may transparently upgrade a legacy password hash. The
    # request-scoped session otherwise rolls that flush back when it closes.
    db.commit()
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=MeResponse)
def me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return workspace_service.me_response(db, current_user)
