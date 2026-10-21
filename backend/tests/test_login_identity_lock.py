"""The login identity lock precedes user lookup and remains transaction scoped."""

import hashlib
from types import SimpleNamespace

import pytest

from al_medlit.auth import service
from al_medlit.core.exceptions import UnauthorizedError


@pytest.mark.parametrize("account", ["known_valid", "known_invalid", "unknown"])
def test_postgres_login_locks_normalized_identity_before_verifying(monkeypatch, account):
    events = []

    class Session:
        def get_bind(self):
            return SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        def execute(self, statement, parameters):
            events.append(("sql", str(statement), parameters))

        def commit(self):
            events.append(("commit",))

        def rollback(self):
            events.append(("rollback",))

        def flush(self):
            events.append(("flush",))

    def verify(db, username, password):
        events.append(("verify", username))
        return SimpleNamespace(id=1) if account == "known_valid" else None

    monkeypatch.setattr(service, "authenticate_user", verify)
    monkeypatch.setattr(service, "_assert_failed_login_not_throttled", lambda *a: None)
    monkeypatch.setattr(service, "record_login_failure", lambda *a: events.append(("audit",)))
    monkeypatch.setattr(service, "record_login_success", lambda *a: events.append(("audit",)))
    if account == "known_valid":
        service.authenticate_login(Session(), "  Alice  ", "password")
    else:
        with pytest.raises(UnauthorizedError):
            service.authenticate_login(Session(), "  Alice  ", "password")
    assert events[0][0] == "sql", events
    assert "pg_advisory_xact_lock" in events[0][1]
    key = int.from_bytes(hashlib.sha256(b"al-medlit:login:Alice").digest()[:8], "big", signed=True)
    assert events[0][2] == {"identity_key": key}
    assert events[1] == ("verify", "Alice")
    assert [event[0] for event in events] == [
        "sql",
        "verify",
        "audit",
        "flush" if account == "known_valid" else "commit",
    ]
