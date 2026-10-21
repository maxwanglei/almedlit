"""Real PostgreSQL serialization of known and missing login identities."""

import os
import threading
import time
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_login_hardening import _make_user
from test_system_administration_postgres_contention import (
    _set_transaction_timeouts,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.auth import service
from al_medlit.core.config import settings
from al_medlit.core.exceptions import RateLimitedError, UnauthorizedError

postgres_session_factory = _shared_postgres_session_factory
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


@pytest.mark.parametrize("known", [True, False])
def test_simultaneous_failures_share_threshold_for_known_and_unknown_names(
    postgres_session_factory,
    monkeypatch,
    known,
):
    username = f"login-race-{uuid4().hex}"
    monkeypatch.setattr(settings, "login_failure_threshold", 1)
    if known:
        with postgres_session_factory() as db:
            _make_user(db, username)
            db.commit()
    entered_audit = threading.Event()
    release_audit = threading.Event()
    second_started = threading.Event()
    second_finished = threading.Event()
    results = {}
    backend_pids = {}
    record = service.record_login_failure

    def pause_first_audit(db, name):
        if threading.current_thread().name == "first-login":
            entered_audit.set()
            assert release_audit.wait(10), "First login audit was never released"
        record(db, name)

    monkeypatch.setattr(service, "record_login_failure", pause_first_audit)

    def attempt(name):
        try:
            with postgres_session_factory() as db:
                _set_transaction_timeouts(db)
                backend_pids[name] = db.execute(text("SELECT pg_backend_pid()")).scalar_one()
                if name == "second-login":
                    second_started.set()
                try:
                    service.authenticate_login(db, username, "incorrect")
                except UnauthorizedError:
                    results[name] = 401
                except RateLimitedError:
                    results[name] = 429
        except Exception as exc:
            results[name] = exc
        finally:
            if name == "second-login":
                second_finished.set()

    first = threading.Thread(target=attempt, args=("first-login",), name="first-login", daemon=True)
    second = threading.Thread(
        target=attempt, args=("second-login",), name="second-login", daemon=True
    )
    first.start()
    observed_blocking = False
    try:
        assert entered_audit.wait(10), results
        second.start()
        assert second_started.wait(5), results
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not second_finished.is_set():
            with postgres_session_factory() as observer:
                blockers = observer.execute(
                    text("SELECT pg_blocking_pids(:pid)"), {"pid": backend_pids["second-login"]}
                ).scalar_one()
            if blockers:
                observed_blocking = True
                break
            time.sleep(0.02)
    finally:
        release_audit.set()
        first.join(timeout=10)
        if second.ident is not None:
            second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive(), results
    assert observed_blocking, "Second login did not wait on the first transaction"
    assert results == {"first-login": 401, "second-login": 429}
