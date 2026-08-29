import jwt
from fastapi import Depends, Header, Request
from sqlalchemy.orm import Session

from al_medlit.auth import service
from al_medlit.auth.cookies import SESSION_COOKIE_NAME, validate_cookie_csrf
from al_medlit.auth.models import User
from al_medlit.auth.security import decode_access_token_claims
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import UnauthorizedError

PUBLIC_PATHS = {
    ("POST", "/api/auth/login"),
    ("POST", "/api/auth/logout"),
    ("POST", "/api/auth/register"),
    ("GET", "/health"),
}


def _is_public_path(method: str, path: str) -> bool:
    method = method.upper()
    if (method, path) in PUBLIC_PATHS:
        return True
    if path.startswith("/api/invites/"):
        # An invitee may not have an account yet, so both reading an invite and
        # accepting it must work unauthenticated. Nothing else under this prefix
        # is public.
        token = path.removeprefix("/api/invites/")
        if method == "POST":
            return token.endswith("/accept") and "/" not in token.removesuffix("/accept")
        return method == "GET" and bool(token) and "/" not in token
    if path.startswith("/api/account-actions/"):
        token = path.removeprefix("/api/account-actions/")
        return method in {"GET", "POST"} and bool(token) and "/" not in token
    return False


def _resolve_user_from_token(token: str, db: Session) -> User:
    try:
        claims = decode_access_token_claims(token)
        subject = str(claims["sub"])
        session_version = int(claims.get("sv", 0))
    except (jwt.PyJWTError, TypeError, ValueError) as exc:
        raise UnauthorizedError("Invalid or expired token") from exc
    try:
        user_id = int(subject)
    except ValueError as exc:
        raise UnauthorizedError("Invalid bearer token") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("Invalid bearer token")
    if session_version != user.session_version:
        raise UnauthorizedError("Invalid bearer token")
    if service.user_has_forbidden_bootstrap_password(user):
        raise UnauthorizedError("Invalid bearer token")
    return user


def _authentication_token(
    request: Request | None,
    authorization: str | None,
) -> tuple[str | None, bool]:
    if authorization:
        if not authorization.lower().startswith("bearer "):
            raise UnauthorizedError("Invalid authorization header")
        token = authorization.split(" ", 1)[1].strip()
        if not token:
            raise UnauthorizedError("Missing bearer token")
        return token, False
    if request is not None:
        token = request.cookies.get(SESSION_COOKIE_NAME)
        if token:
            return token, True
    return None, False


def _resolve_user_from_authorization(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    token, _cookie_authenticated = _authentication_token(None, authorization)
    if token is None:
        raise UnauthorizedError("Missing bearer token")
    return _resolve_user_from_token(token, db)


def enforce_authentication(
    request: Request,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> None:
    if _is_public_path(request.method, request.url.path):
        return
    token, cookie_authenticated = _authentication_token(request, authorization)
    if token is None:
        raise UnauthorizedError("Missing bearer token")
    if cookie_authenticated:
        validate_cookie_csrf(request)
    request.state.user = _resolve_user_from_token(token, db)


def get_current_user(
    request: Request = None,  # type: ignore[assignment]
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User:
    if request is not None:
        user = getattr(request.state, "user", None)
        if isinstance(user, User):
            return user
    token, cookie_authenticated = _authentication_token(request, authorization)
    if token is None:
        raise UnauthorizedError("Missing bearer token")
    if cookie_authenticated and request is not None:
        validate_cookie_csrf(request)
    return _resolve_user_from_token(token, db)
