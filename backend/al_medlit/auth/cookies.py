import secrets

from fastapi import Request, Response

from al_medlit.auth.security import create_access_token
from al_medlit.core.config import settings
from al_medlit.core.exceptions import ForbiddenError

SESSION_COOKIE_NAME = "al_medlit_session"
CSRF_COOKIE_NAME = "al_medlit_csrf"
CSRF_HEADER_NAME = "X-CSRF-Token"
SESSION_COOKIE_PATH = "/api"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def set_session_cookies(
    response: Response,
    *,
    user_id: int,
    session_version: int,
) -> None:
    max_age = settings.jwt_expire_minutes * 60
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_access_token(user_id, session_version=session_version),
        max_age=max_age,
        path=SESSION_COOKIE_PATH,
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=secrets.token_urlsafe(32),
        max_age=max_age,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=False,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store"


def clear_session_cookies(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite="lax",
    )
    response.delete_cookie(
        key=CSRF_COOKIE_NAME,
        path="/",
        secure=settings.auth_cookie_secure,
        httponly=False,
        samesite="lax",
    )
    response.headers["Cache-Control"] = "no-store"


def validate_cookie_csrf(request: Request) -> None:
    if request.method.upper() in SAFE_METHODS:
        return
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME, "")
    header_token = request.headers.get(CSRF_HEADER_NAME, "")
    if not cookie_token or not header_token or not secrets.compare_digest(
        cookie_token,
        header_token,
    ):
        raise ForbiddenError("CSRF validation failed")
