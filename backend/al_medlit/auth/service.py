from functools import lru_cache

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.schemas import UserCreate
from al_medlit.auth.security import hash_password, verify_password, verify_password_and_update
from al_medlit.core.config import FORBIDDEN_BOOTSTRAP_ADMIN_PASSWORDS, settings
from al_medlit.core.exceptions import ConflictError


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def register_user(db: Session, data: UserCreate, *, is_active: bool = True) -> User:
    username = data.username.strip()
    if not username:
        raise ConflictError("Username is required")
    if get_user_by_username(db, username) is not None:
        raise ConflictError(f"Username {username!r} is already taken")
    user = User(
        username=username,
        email=data.email,
        password_hash=hash_password(data.password),
        display_name=data.display_name or "",
        is_active=is_active,
    )
    try:
        # The read above gives a friendly fast path, while the unique index is
        # the authority when two registrations race. Do not release a SAVEPOINT
        # here: on SQLite, an outer transaction may not yet have emitted BEGIN,
        # so releasing the first savepoint can accidentally persist the user
        # before workspace provisioning succeeds.
        db.add(user)
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ConflictError(f"Username {username!r} is already taken") from exc
    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    # Password reset and bootstrap password rotation use the same User-row lock.
    # Refresh after waiting so a login can never verify an old hash and then
    # overwrite a newer password with a legacy-hash upgrade.
    user = (
        db.query(User)
        .filter(User.username == username)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if user is None or not user.is_active:
        return None
    valid, updated_hash = verify_password_and_update(password, user.password_hash)
    if not valid:
        return None
    if user_has_forbidden_bootstrap_password(user):
        return None
    if updated_hash is not None and updated_hash != user.password_hash:
        user.password_hash = updated_hash
        db.flush()
    return user


@lru_cache(maxsize=8)
def _hash_matches_forbidden_bootstrap_password(password_hash: str) -> bool:
    """Memoize the bcrypt comparison against the fixed forbidden password list.

    ``verify_password`` over a fixed password is a pure function of the stored
    hash, so this is safe to cache: rotating the bootstrap admin's password
    produces a new hash and therefore a new cache key. Callers check the
    username first, so only bootstrap-admin hashes ever reach this cache.
    """

    return any(
        verify_password(password, password_hash)
        for password in FORBIDDEN_BOOTSTRAP_ADMIN_PASSWORDS
    )


def user_has_forbidden_bootstrap_password(user: User) -> bool:
    # Every authenticated request reaches this check, so the cheap identity
    # guards run before the deliberately expensive password comparison.
    if user.username != settings.bootstrap_admin_username.strip():
        return False
    if not user.is_active or not user.is_superuser:
        return False
    return _hash_matches_forbidden_bootstrap_password(user.password_hash)


def assert_no_vulnerable_bootstrap_admin(db: Session) -> None:
    user = get_user_by_username(db, settings.bootstrap_admin_username.strip())
    if user is None or not user_has_forbidden_bootstrap_password(user):
        return
    raise RuntimeError(
        "The bootstrap admin account still uses a known default password. "
        "Rotate or disable that account before starting the API."
    )
