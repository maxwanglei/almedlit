from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from al_medlit.administration.policy import get_effective_policy
from al_medlit.auth import service
from al_medlit.auth.cookies import (
    SESSION_COOKIE_NAME,
    clear_session_cookies,
    set_session_cookies,
    validate_cookie_csrf,
)
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.schemas import LoginRequest, MeResponse, RegistrationRequest, SessionResponse
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, UnauthorizedError
from al_medlit.workspace import service as workspace_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=SessionResponse)
def register(
    payload: RegistrationRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    if not get_effective_policy(db).allow_self_registration:
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
    set_session_cookies(
        response,
        user_id=user.id,
        session_version=user.session_version,
    )
    return SessionResponse()


@router.post("/login", response_model=SessionResponse)
def login(
    payload: LoginRequest,
    response: Response,
    db: Session = Depends(get_db),
):
    service.assert_login_not_throttled(db, payload.username)
    user = service.authenticate_user(db, payload.username, payload.password)
    if user is None:
        service.record_login_failure(db, payload.username)
        # The audit row is the whole point of this branch, so it has to be
        # committed before the request unwinds -- get_db only closes.
        db.commit()
        raise UnauthorizedError("Invalid username or password")
    user.last_login_at = datetime.now(UTC)
    service.record_login_success(db, user)
    # authenticate_user may transparently upgrade a legacy password hash. The
    # request-scoped session otherwise rolls that flush back when it closes.
    db.commit()
    set_session_cookies(
        response,
        user_id=user.id,
        session_version=user.session_version,
    )
    return SessionResponse()


@router.post("/logout", response_model=SessionResponse)
def logout(request: Request, response: Response):
    if request.cookies.get(SESSION_COOKIE_NAME):
        validate_cookie_csrf(request)
    clear_session_cookies(response)
    return SessionResponse(authenticated=False)


@router.get("/me", response_model=MeResponse)
def me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return workspace_service.me_response(db, current_user)
