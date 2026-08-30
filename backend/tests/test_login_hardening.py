"""Login auditing, per-account backoff, and administrative read auditing.

These cover the controls that make a credential-stuffing run visible and
bounded, plus the audit trail over the two superuser reads that expose the
cross-workspace user directory.
"""

import pytest


def _make_user(
    db,
    username: str,
    *,
    password: str = "a-secure-password",
    is_active: bool = True,
    is_superuser: bool = False,
):
    from al_medlit.auth.models import User
    from al_medlit.auth.security import hash_password

    user = User(
        username=username,
        password_hash=hash_password(password),
        display_name=username.title(),
        email=f"{username}@example.test",
        is_active=is_active,
        is_superuser=is_superuser,
    )
    db.add(user)
    db.flush()
    return user


def _headers(user) -> dict[str, str]:
    from al_medlit.auth.security import create_access_token

    return {
        "Authorization": (
            "Bearer " + create_access_token(user.id, session_version=user.session_version)
        )
    }


def _events(db, event_type: str):
    from al_medlit.administration.models import AdminAuditEvent

    return (
        db.query(AdminAuditEvent)
        .filter(AdminAuditEvent.event_type == event_type)
        .order_by(AdminAuditEvent.id)
        .all()
    )


def _login(client, username: str, password: str):
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
        headers={"Authorization": ""},
    )


def test_unknown_account_login_still_verifies_a_password_hash(db, monkeypatch):
    """The unknown-account path must not short-circuit ahead of bcrypt.

    Wall-clock timing is too flaky to assert on, so this pins the behaviour that
    produces the equal timing: a hash comparison happens either way.
    """
    from al_medlit.auth import service

    calls = []
    real_verify = service.verify_password

    def counting_verify(password, password_hash):
        calls.append(password_hash)
        return real_verify(password, password_hash)

    monkeypatch.setattr(service, "verify_password", counting_verify)

    assert service.authenticate_user(db, "no-such-account", "whatever") is None
    assert len(calls) == 1
    assert calls[0].startswith("$")


def test_inactive_account_login_also_verifies_a_password_hash(db, monkeypatch):
    from al_medlit.auth import service

    _make_user(db, "dormant", is_active=False)
    db.commit()

    calls = []
    monkeypatch.setattr(
        service,
        "verify_password",
        lambda password, password_hash: calls.append(password_hash) or False,
    )

    assert service.authenticate_user(db, "dormant", "a-secure-password") is None
    assert len(calls) == 1


def test_failed_login_on_real_account_is_audited(client, db):
    user = _make_user(db, "target-account")
    db.commit()

    assert _login(client, "target-account", "wrong-password").status_code == 401

    events = _events(db, "auth.login_failed")
    assert len(events) == 1
    event = events[0]
    assert event.target_user_id == user.id
    # The attempt is unauthenticated, so nobody is named as the actor.
    assert event.actor_user_id is None
    assert event.details["reason"] == "bad_password"
    assert event.details["username"] == "target-account"


def test_failed_login_on_unknown_account_records_only_a_fingerprint(client, db):
    # A password mistyped into the username box must not be retained verbatim.
    assert _login(client, "hunter2-not-a-user", "whatever").status_code == 401

    events = _events(db, "auth.login_failed")
    assert len(events) == 1
    assert events[0].target_user_id is None
    assert events[0].details["reason"] == "unknown_user"
    assert "username" not in events[0].details
    assert len(events[0].details["username_fingerprint"]) == 16


def test_failed_login_on_inactive_account_records_that_reason(client, db):
    _make_user(db, "suspended-account", is_active=False)
    db.commit()

    assert _login(client, "suspended-account", "a-secure-password").status_code == 401

    events = _events(db, "auth.login_failed")
    assert len(events) == 1
    assert events[0].details["reason"] == "inactive_account"


def test_successful_login_is_audited(client, db):
    user = _make_user(db, "good-login", is_superuser=True)
    db.commit()

    assert _login(client, "good-login", "a-secure-password").status_code == 200

    events = _events(db, "auth.login_succeeded")
    assert len(events) == 1
    assert events[0].actor_user_id == user.id
    assert events[0].details["is_superuser"] is True


def test_repeated_failures_throttle_the_account_then_lapse_after_the_window(
    client, db, monkeypatch
):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "login_failure_threshold", 3)
    _make_user(db, "stuffed-account")
    db.commit()

    for _ in range(3):
        assert _login(client, "stuffed-account", "wrong-password").status_code == 401

    throttled = _login(client, "stuffed-account", "wrong-password")
    assert throttled.status_code == 429
    # The correct password is refused too, or the throttle would be a no-op.
    assert _login(client, "stuffed-account", "a-secure-password").status_code == 429

    # The backoff is a window, never a sticky lockout: an operator must not be
    # lockable out of their own instance by a stranger.
    monkeypatch.setattr(settings, "login_failure_window_minutes", 0)
    assert _login(client, "stuffed-account", "a-secure-password").status_code == 200


def test_successful_login_clears_the_failure_streak(client, db, monkeypatch):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "login_failure_threshold", 3)
    _make_user(db, "recovering-account")
    db.commit()

    for _ in range(2):
        assert _login(client, "recovering-account", "wrong-password").status_code == 401
    assert _login(client, "recovering-account", "a-secure-password").status_code == 200

    # Two more failures would have tripped the old streak; the success retired it.
    for _ in range(2):
        assert _login(client, "recovering-account", "wrong-password").status_code == 401
    assert _login(client, "recovering-account", "a-secure-password").status_code == 200


def test_throttle_can_be_disabled(client, db, monkeypatch):
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "login_failure_threshold", 0)
    _make_user(db, "unthrottled-account")
    db.commit()

    for _ in range(4):
        assert _login(client, "unthrottled-account", "wrong-password").status_code == 401
    assert _login(client, "unthrottled-account", "wrong-password").status_code == 401


def test_unknown_usernames_are_not_throttled(client, db, monkeypatch):
    """Documents a deliberate choice, not an oversight.

    Throttling unknown names would need a separate name-keyed counter; without
    one, a 429 on an existing account and a 401 on a missing one differ. That
    residual signal is bounded by the edge rate limiter and costs an attacker a
    full threshold of attempts per guess.
    """
    from al_medlit.core.config import settings

    monkeypatch.setattr(settings, "login_failure_threshold", 2)

    for _ in range(4):
        assert _login(client, "ghost-account", "whatever").status_code == 401


def test_workspace_scoped_user_listing_is_audited(client, db):
    from al_medlit.workspace import service as workspace_service

    admin = _make_user(db, "audit-admin", is_superuser=True)
    member = _make_user(db, "audit-member")
    workspace = workspace_service.create_team_workspace(db, member, "Audited Team")
    db.commit()

    response = client.get(
        f"/api/admin/users?workspace_id={workspace.id}",
        headers=_headers(admin),
    )
    assert response.status_code == 200

    events = _events(db, "directory.users_listed")
    assert len(events) == 1
    assert events[0].actor_user_id == admin.id
    assert events[0].workspace_id == workspace.id
    assert events[0].details["returned"] == response.json()["total"]


def test_user_detail_read_is_audited(client, db):
    admin = _make_user(db, "detail-admin", is_superuser=True)
    subject = _make_user(db, "detail-subject")
    db.commit()

    response = client.get(f"/api/admin/users/{subject.id}", headers=_headers(admin))
    assert response.status_code == 200

    events = _events(db, "directory.user_viewed")
    assert len(events) == 1
    assert events[0].actor_user_id == admin.id
    assert events[0].target_user_id == subject.id


def test_status_mutation_does_not_emit_a_directory_read_event(client, db):
    """A status change renders the detail payload but is not a directory read."""
    admin = _make_user(db, "mutating-admin", is_superuser=True)
    subject = _make_user(db, "mutated-subject")
    db.commit()

    response = client.patch(
        f"/api/admin/users/{subject.id}/status",
        json={"is_active": False},
        headers=_headers(admin),
    )
    assert response.status_code == 200

    assert _events(db, "directory.user_viewed") == []
    assert len(_events(db, "account.deactivated")) == 1


@pytest.mark.parametrize("event_type", ["auth.login_failed", "auth.login_succeeded"])
def test_login_audit_events_are_append_only(client, db, event_type):
    _make_user(db, "immutable-audit")
    db.commit()
    _login(client, "immutable-audit", "wrong-password")
    _login(client, "immutable-audit", "a-secure-password")

    event = _events(db, event_type)[0]
    event.details = {"tampered": True}
    with pytest.raises(ValueError, match="append-only"):
        db.flush()
    db.rollback()
