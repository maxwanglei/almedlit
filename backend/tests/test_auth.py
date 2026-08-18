import jwt
import pytest
from passlib.context import CryptContext

from al_medlit.auth import security


def test_password_hash_roundtrip():
    hashed = security.hash_password("s3cret")
    assert hashed != "s3cret"
    assert security.verify_password("s3cret", hashed) is True
    assert security.verify_password("wrong", hashed) is False


def test_access_token_roundtrip():
    token = security.create_access_token("42")
    assert security.decode_access_token(token) == "42"


def test_decode_rejects_tampered_token():
    token = security.create_access_token("42")
    with pytest.raises(jwt.PyJWTError):
        security.decode_access_token(token + "x")


def test_user_model_columns(db):
    from al_medlit.auth.models import User

    user = User(
        username="alice",
        password_hash="x",
        display_name="Alice",
    )
    db.add(user)
    db.flush()
    assert user.id is not None
    assert user.is_active is True
    assert user.is_superuser is False


def test_register_user_creates_active_user(db):
    from al_medlit.auth import service
    from al_medlit.auth.schemas import UserCreate

    user = service.register_user(
        db,
        UserCreate(username="carol", password="pw", display_name="Carol"),
    )
    assert user.id is not None
    assert user.is_active is True
    assert user.password_hash != "pw"


def test_register_user_rejects_duplicate_username(db):
    from al_medlit.auth import service
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.core.exceptions import ConflictError

    service.register_user(db, UserCreate(username="dave", password="pw"))
    db.flush()

    with pytest.raises(ConflictError):
        service.register_user(db, UserCreate(username="dave", password="pw2"))


def test_authenticate_user_checks_password(db):
    from al_medlit.auth import service
    from al_medlit.auth.schemas import UserCreate

    service.register_user(db, UserCreate(username="erin", password="pw"))
    db.flush()
    assert service.authenticate_user(db, "erin", "pw") is not None
    assert service.authenticate_user(db, "erin", "nope") is None
    assert service.authenticate_user(db, "ghost", "pw") is None


def test_authenticate_user_upgrades_legacy_pbkdf2_hash(db):
    from al_medlit.auth import service
    from al_medlit.auth.models import User

    legacy_context = CryptContext(schemes=["pbkdf2_sha256"])
    user = User(
        username="legacy",
        password_hash=legacy_context.hash("pw"),
        display_name="Legacy",
    )
    db.add(user)
    db.flush()

    authenticated = service.authenticate_user(db, "legacy", "pw")
    assert authenticated is not None
    assert authenticated.password_hash.startswith("$2")


def test_authenticate_user_rejects_unknown_hash_without_500(db):
    from al_medlit.auth import service
    from al_medlit.auth.models import User

    user = User(username="broken", password_hash="not-a-real-hash", display_name="Broken")
    db.add(user)
    db.flush()

    assert service.authenticate_user(db, "broken", "pw") is None


def test_authenticate_user_rejects_default_bootstrap_admin_password(db, monkeypatch):
    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password
    from al_medlit.core.config import DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, settings

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    user = User(
        username="admin",
        password_hash=hash_password(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD),
        display_name="Admin",
        is_active=True,
        is_superuser=True,
    )
    db.add(user)
    db.flush()

    assert service.authenticate_user(db, "admin", DEFAULT_BOOTSTRAP_ADMIN_PASSWORD) is None


def test_existing_default_bootstrap_admin_token_is_rejected(db, monkeypatch):
    from al_medlit.auth.dependencies import get_current_user
    from al_medlit.auth.models import User
    from al_medlit.auth.security import create_access_token, hash_password
    from al_medlit.core.config import DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, settings
    from al_medlit.core.exceptions import UnauthorizedError

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    user = User(
        username="admin",
        password_hash=hash_password(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD),
        display_name="Admin",
        is_active=True,
        is_superuser=True,
    )
    db.add(user)
    db.flush()

    token = create_access_token(user.id)
    with pytest.raises(UnauthorizedError):
        get_current_user(authorization=f"Bearer {token}", db=db)


def test_authenticate_user_allows_rotated_bootstrap_admin_password(db, monkeypatch):
    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    user = User(
        username="admin",
        password_hash=hash_password("rotated-admin-password"),
        display_name="Admin",
        is_active=True,
        is_superuser=True,
    )
    db.add(user)
    db.flush()

    assert service.authenticate_user(db, "admin", "rotated-admin-password") is not None


def test_bootstrap_password_check_hashes_once_per_stored_hash(monkeypatch):
    """The check runs on every authenticated request; bcrypt must not."""
    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    service._hash_matches_forbidden_bootstrap_password.cache_clear()

    calls = []
    real_verify_password = service.verify_password

    def counting_verify_password(password, password_hash):
        calls.append(password_hash)
        return real_verify_password(password, password_hash)

    monkeypatch.setattr(service, "verify_password", counting_verify_password)

    user = User(
        username="admin",
        password_hash=hash_password("rotated-admin-password"),
        display_name="Admin",
        is_active=True,
        is_superuser=True,
    )

    assert service.user_has_forbidden_bootstrap_password(user) is False
    first_call_count = len(calls)
    assert first_call_count > 0

    for _ in range(3):
        assert service.user_has_forbidden_bootstrap_password(user) is False
    assert len(calls) == first_call_count


def test_bootstrap_password_check_reevaluates_a_rotated_hash(monkeypatch):
    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password
    from al_medlit.core.config import DEFAULT_BOOTSTRAP_ADMIN_PASSWORD, settings

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    service._hash_matches_forbidden_bootstrap_password.cache_clear()

    user = User(
        username="admin",
        password_hash=hash_password(DEFAULT_BOOTSTRAP_ADMIN_PASSWORD),
        display_name="Admin",
        is_active=True,
        is_superuser=True,
    )
    assert service.user_has_forbidden_bootstrap_password(user) is True

    # Rotating the password produces a new hash, so the cached verdict for the
    # old hash must not be reused.
    user.password_hash = hash_password("rotated-admin-password")
    assert service.user_has_forbidden_bootstrap_password(user) is False


def test_bootstrap_password_check_skips_bcrypt_for_other_users(monkeypatch):
    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "bootstrap_admin_username", "admin")
    service._hash_matches_forbidden_bootstrap_password.cache_clear()

    def fail_on_verify(password, password_hash):
        raise AssertionError("verify_password must not run for non-bootstrap users")

    monkeypatch.setattr(service, "verify_password", fail_on_verify)

    assert (
        service.user_has_forbidden_bootstrap_password(
            User(username="alice", password_hash="x", is_active=True, is_superuser=True)
        )
        is False
    )
    assert (
        service.user_has_forbidden_bootstrap_password(
            User(username="admin", password_hash="x", is_active=True, is_superuser=False)
        )
        is False
    )


def test_explicit_bootstrap_admin_command_creates_superuser(db, monkeypatch):
    from al_medlit.auth import service
    from al_medlit.core.config import settings
    from al_medlit.workspace import service as workspace_service
    from scripts.bootstrap_admin import bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_username", "root")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "strong-admin-password")

    user = bootstrap_admin(db)
    default_workspace = workspace_service.ensure_default_workspace(db)
    member = workspace_service.get_member(db, default_workspace.id, user.id)

    assert user.username == "root"
    assert user.is_active is True
    assert user.is_superuser is True
    assert member is not None
    assert member.role == "admin"
    assert service.authenticate_user(db, "root", "strong-admin-password") is not None


def test_explicit_bootstrap_admin_refuses_existing_non_superuser(db, monkeypatch):
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password, verify_password
    from al_medlit.core.config import settings
    from scripts.bootstrap_admin import bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_username", "claimed-root")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "strong-admin-password")
    user = User(
        username="claimed-root",
        password_hash=hash_password("ordinary-user-password"),
        display_name="Ordinary User",
        is_active=True,
        is_superuser=False,
    )
    db.add(user)
    db.flush()
    original_hash = user.password_hash

    with pytest.raises(RuntimeError, match="existing non-superuser account"):
        bootstrap_admin(db)

    assert user.is_superuser is False
    assert user.password_hash == original_hash
    assert verify_password("ordinary-user-password", user.password_hash) is True
    assert verify_password("strong-admin-password", user.password_hash) is False


def test_explicit_bootstrap_admin_is_idempotent_for_existing_superuser(db, monkeypatch):
    from al_medlit.auth.models import User
    from al_medlit.auth.security import verify_password
    from al_medlit.core.config import settings
    from al_medlit.workspace import service as workspace_service
    from scripts.bootstrap_admin import bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_username", "existing-root")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "strong-admin-password")

    first = bootstrap_admin(db)
    second = bootstrap_admin(db)
    default_workspace = workspace_service.ensure_default_workspace(db)
    member = workspace_service.get_member(db, default_workspace.id, second.id)

    assert first.id == second.id
    assert db.query(User).filter(User.username == "existing-root").count() == 1
    assert verify_password("strong-admin-password", second.password_hash) is True
    assert member is not None
    assert member.role == "admin"


def test_bootstrap_admin_password_change_invalidates_existing_sessions(
    client,
    db,
    monkeypatch,
):
    from al_medlit.auth.security import create_access_token
    from al_medlit.core.config import settings
    from scripts.bootstrap_admin import bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_username", "rotated-root")
    monkeypatch.setattr(settings, "bootstrap_admin_password", "first-admin-password")
    user = bootstrap_admin(db)
    original_version = user.session_version
    original_token = create_access_token(
        user.id,
        session_version=user.session_version,
    )

    # Reapplying the same password is idempotent and does not revoke sessions.
    same_password = bootstrap_admin(db)
    assert same_password.session_version == original_version

    monkeypatch.setattr(settings, "bootstrap_admin_password", "second-admin-password")
    rotated = bootstrap_admin(db)
    assert rotated.session_version == original_version + 1
    assert client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {original_token}"},
    ).status_code == 401


def test_authenticate_user_requests_a_user_row_lock(db, monkeypatch):
    from sqlalchemy.orm import Query

    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password

    user = User(
        username="locked-login",
        password_hash=hash_password("locked-login-password"),
        display_name="Locked Login",
        is_active=True,
    )
    db.add(user)
    db.commit()
    lock_calls = 0
    real_with_for_update = Query.with_for_update

    def track_with_for_update(query, *args, **kwargs):
        nonlocal lock_calls
        lock_calls += 1
        return real_with_for_update(query, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", track_with_for_update)

    assert service.authenticate_user(
        db,
        "locked-login",
        "locked-login-password",
    ) is not None
    assert lock_calls == 1


def test_explicit_bootstrap_admin_command_requires_password(db, monkeypatch):
    from al_medlit.core.config import settings
    from scripts.bootstrap_admin import bootstrap_admin

    monkeypatch.setattr(settings, "bootstrap_admin_password", "")

    with pytest.raises(RuntimeError, match="AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD"):
        bootstrap_admin(db)


def test_get_current_user_resolves_token(db):
    from al_medlit.auth import service
    from al_medlit.auth.dependencies import get_current_user
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.auth.security import create_access_token

    user = service.register_user(db, UserCreate(username="frank", password="pw"))
    db.flush()
    token = create_access_token(str(user.id))
    resolved = get_current_user(authorization=f"Bearer {token}", db=db)
    assert resolved.id == user.id


def test_get_current_user_rejects_missing_header(db):
    from al_medlit.auth.dependencies import get_current_user
    from al_medlit.core.exceptions import UnauthorizedError

    with pytest.raises(UnauthorizedError):
        get_current_user(authorization=None, db=db)


def test_get_current_user_rejects_bad_token(db):
    from al_medlit.auth.dependencies import get_current_user
    from al_medlit.core.exceptions import UnauthorizedError

    with pytest.raises(UnauthorizedError):
        get_current_user(authorization="Bearer not-a-token", db=db)


def test_register_login_me_flow(client):
    password = "strong-password"
    reg = client.post(
        "/api/auth/register",
        json={"username": "newbie", "password": password},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    body = me.json()
    assert body["user"]["username"] == "newbie"
    assert len(body["memberships"]) == 1

    login = client.post(
        "/api/auth/login",
        json={"username": "newbie", "password": password},
    )
    assert login.status_code == 200

    bad = client.post("/api/auth/login", json={"username": "newbie", "password": "x"})
    assert bad.status_code == 401


@pytest.mark.parametrize(
    "password",
    [
        "elevenbytes",
        "a" * 73,
        chr(0x1F40D) * 19,
    ],
)
def test_public_registration_rejects_passwords_outside_utf8_byte_limits(client, password):
    response = client.post(
        "/api/auth/register",
        json={"username": f"invalid-{len(password)}", "password": password},
    )

    assert response.status_code == 422


def test_public_registration_accepts_twelve_utf8_bytes(client):
    password = chr(0x1F40D) * 3
    response = client.post(
        "/api/auth/register",
        json={"username": "unicode-password", "password": password},
    )

    assert response.status_code == 200
    login = client.post(
        "/api/auth/login",
        json={"username": "unicode-password", "password": password},
    )
    assert login.status_code == 200


def test_login_rejects_password_over_passlib_limit_without_server_error(client):
    password = "strong-password"
    registered = client.post(
        "/api/auth/register",
        json={"username": "oversized-login", "password": password},
    )
    assert registered.status_code == 200

    response = client.post(
        "/api/auth/login",
        json={"username": "oversized-login", "password": "x" * 4097},
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("display_name", "x" * 121),
        ("email", "x" * 256),
    ],
)
def test_public_registration_rejects_user_fields_over_database_limits(client, field, value):
    payload = {
        "username": f"oversized-{field}",
        "password": "strong-password",
        field: value,
    }

    response = client.post("/api/auth/register", json=payload)

    assert response.status_code == 422


def test_login_persists_legacy_password_hash_upgrade(client, db):
    from al_medlit.auth.models import User

    legacy_context = CryptContext(schemes=["pbkdf2_sha256"])
    password = "legacy-password"
    user = User(
        username="legacy-api",
        password_hash=legacy_context.hash(password),
        display_name="Legacy API",
    )
    db.add(user)
    db.commit()

    response = client.post(
        "/api/auth/login",
        json={"username": user.username, "password": password},
    )

    assert response.status_code == 200
    db.expire_all()
    assert db.get(User, user.id).password_hash.startswith("$2")


def test_register_user_translates_racing_unique_violation(db, monkeypatch):
    from sqlalchemy.exc import IntegrityError

    from al_medlit.auth import service
    from al_medlit.auth.models import User
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.core.exceptions import ConflictError

    original_flush = db.flush

    def racing_flush(*args, **kwargs):
        if any(isinstance(row, User) for row in db.new):
            raise IntegrityError("INSERT", {}, Exception("duplicate key"))
        return original_flush(*args, **kwargs)

    monkeypatch.setattr(db, "flush", racing_flush)

    with pytest.raises(ConflictError, match="already taken"):
        service.register_user(
            db,
            UserCreate(username="racing-user", password="internal-password"),
        )


def test_me_requires_auth(client):
    assert client.get("/api/auth/me", headers={"Authorization": ""}).status_code == 401
