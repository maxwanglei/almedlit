from datetime import UTC, datetime, timedelta

import jwt
from passlib.context import CryptContext
from passlib.exc import PasswordSizeError, UnknownHashError

from al_medlit.core.config import settings

_pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _pwd_context.verify(password, password_hash)
    except (PasswordSizeError, UnknownHashError):
        return False


def verify_password_and_update(password: str, password_hash: str) -> tuple[bool, str | None]:
    try:
        return _pwd_context.verify_and_update(password, password_hash)
    except (PasswordSizeError, UnknownHashError):
        return False, None


def create_access_token(
    subject: int | str,
    expires_minutes: int | None = None,
    *,
    session_version: int = 0,
) -> str:
    minutes = expires_minutes if expires_minutes is not None else settings.jwt_expire_minutes
    expire = datetime.now(UTC) + timedelta(minutes=minutes)
    payload = {"sub": str(subject), "exp": expire, "sv": session_version}
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token_claims(token: str) -> dict:
    """Decode validated claims, preserving legacy tokens without ``sv``."""

    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    if "sub" not in payload:
        raise jwt.MissingRequiredClaimError("sub")
    return payload


def decode_access_token(token: str) -> str:
    """Return the token subject, or raise jwt.PyJWTError on any failure."""

    payload = decode_access_token_claims(token)
    return str(payload["sub"])


def verify_access_token(token: str) -> str:
    """Backward-compatible alias for older internal call sites."""

    return decode_access_token(token)
