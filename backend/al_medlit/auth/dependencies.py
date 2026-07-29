import jwt
from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from al_medlit.auth import service
from al_medlit.auth.models import User
from al_medlit.auth.security import decode_access_token
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import UnauthorizedError

PUBLIC_PATHS = {
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/register"),
    ("GET", "/health"),
}


def _is_public_path(method: str, path: str) -> bool:
    method = method.upper()
    if (method, path) in PUBLIC_PATHS:
        return True
    return (
        method == "POST"
        and path.startswith("/api/invites/")
        and path.endswith("/accept")
    )


def _resolve_user_from_authorization(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise UnauthorizedError("Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        subject = decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise UnauthorizedError("Invalid or expired token") from exc
    try:
        user_id = int(subject)
    except ValueError as exc:
        raise UnauthorizedError("Invalid bearer token") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("Invalid bearer token")
    if service.user_has_forbidden_bootstrap_password(user):
        raise UnauthorizedError("Invalid bearer token")
    return user


def enforce_authentication(
    request: Request,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> None:
    if _is_public_path(request.method, request.url.path):
        return
    request.state.user = _resolve_user_from_authorization(authorization, db)


def get_current_user(
    request: Request = None,  # type: ignore[assignment]
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if request is not None:
        user = getattr(request.state, "user", None)
        if isinstance(user, User):
            return user
    return _resolve_user_from_authorization(authorization, db)
