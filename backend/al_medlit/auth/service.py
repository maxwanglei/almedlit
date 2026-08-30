import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from functools import lru_cache

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.schemas import UserCreate
from al_medlit.auth.security import hash_password, verify_password, verify_password_and_update
from al_medlit.core.config import FORBIDDEN_BOOTSTRAP_ADMIN_PASSWORDS, settings
from al_medlit.core.exceptions import ConflictError, RateLimitedError

LOGIN_FAILED_EVENT = "auth.login_failed"
LOGIN_SUCCEEDED_EVENT = "auth.login_succeeded"
# The username field is attacker-controlled free text, so cap what reaches the
# audit row.
MAX_AUDITED_USERNAME_LENGTH = 120


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


@lru_cache(maxsize=1)
def _timing_equalizer_hash() -> str:
    """Return a throwaway hash used to keep the unknown-account path expensive.

    Hashing through the shared password context rather than hardcoding a digest
    keeps the cost identical to a real verification even if the bcrypt work
    factor is retuned later.
    """

    return hash_password(secrets.token_urlsafe(32))


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
        # Burn an equivalent bcrypt round before giving up. Returning here
        # directly would make the unknown-account path measurably faster than a
        # wrong password, turning login into a username oracle.
        verify_password(password, _timing_equalizer_hash())
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


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _username_fingerprint(username: str) -> str:
    """Correlate attempts on one name without retaining the name itself.

    A password mistyped into the username box would otherwise land in the audit
    log in the clear, so unresolvable attempts are recorded only as a digest.
    """

    return hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]


def _recent_login_failure_count(db: Session, user_id: int) -> int:
    # Imported lazily: the administration package imports this module, so a
    # module-level import would close a cycle.
    from al_medlit.administration.models import AdminAuditEvent

    window_start = datetime.now(UTC) - timedelta(
        minutes=settings.login_failure_window_minutes
    )
    # A successful login is the owner demonstrating control of the account, so
    # it retires the failures that preceded it. The audit table is append-only,
    # which makes "count since the last success" the only way to clear a streak.
    last_success = (
        db.query(func.max(AdminAuditEvent.created_at))
        .filter(
            AdminAuditEvent.event_type == LOGIN_SUCCEEDED_EVENT,
            AdminAuditEvent.target_user_id == user_id,
        )
        .scalar()
    )
    if last_success is not None:
        window_start = max(window_start, _aware(last_success))
    return (
        db.query(func.count(AdminAuditEvent.id))
        .filter(
            AdminAuditEvent.event_type == LOGIN_FAILED_EVENT,
            AdminAuditEvent.target_user_id == user_id,
            AdminAuditEvent.created_at > window_start,
        )
        .scalar()
        or 0
    )


def assert_login_not_throttled(db: Session, username: str) -> None:
    """Reject a login against an account with too many recent failures.

    The edge limiter counts per source address, so it never sees a distributed
    run against a single account, and it is absent from any topology that does
    not front the API with the bundled nginx. This backoff is keyed on the
    account instead, and it lapses on its own once the window passes or the
    owner logs in successfully -- deliberately not a sticky lockout, which would
    hand anyone a way to keep an administrator out of their own instance.

    Only accounts that exist are counted. Extending the backoff to unknown names
    would need a separate name-keyed counter, and the enumeration signal that
    omitting it leaves is bounded by the edge limiter and far weaker than the
    timing oracle ``authenticate_user`` already closes.
    """

    threshold = settings.login_failure_threshold
    if threshold <= 0:
        return
    user = get_user_by_username(db, username.strip())
    if user is None:
        return
    if _recent_login_failure_count(db, user.id) < threshold:
        return
    raise RateLimitedError(
        "Too many failed login attempts for this account. Try again later."
    )


def record_login_failure(db: Session, username: str) -> None:
    """Stage an audit event for a rejected login attempt.

    Without this row, a credential-stuffing run against one account leaves no
    trace an operator can query after the fact, and the backoff above has
    nothing to count.
    """

    from al_medlit.administration.events import record_admin_event

    candidate = get_user_by_username(db, username.strip())
    if candidate is None:
        reason = "unknown_user"
    elif not candidate.is_active:
        reason = "inactive_account"
    else:
        reason = "bad_password"
    details = {
        "reason": reason,
        "username_fingerprint": _username_fingerprint(username),
    }
    if candidate is not None:
        details["username"] = candidate.username[:MAX_AUDITED_USERNAME_LENGTH]
    record_admin_event(
        db,
        event_type=LOGIN_FAILED_EVENT,
        # The attempt is unauthenticated, so there is no actor to name. The
        # account under attack is the target.
        actor_user_id=None,
        target_user_id=candidate.id if candidate is not None else None,
        details=details,
    )


def record_login_success(db: Session, user: User) -> None:
    from al_medlit.administration.events import record_admin_event

    record_admin_event(
        db,
        event_type=LOGIN_SUCCEEDED_EVENT,
        actor_user_id=user.id,
        target_user_id=user.id,
        details={"is_superuser": user.is_superuser},
    )


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
