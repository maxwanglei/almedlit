import subprocess
import sys

import pytest

from scripts import start_api


def test_start_api_runs_migrations_before_exec_when_enabled(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        calls.append(("run", command))
        assert check is False
        return subprocess.CompletedProcess(command, 0)

    def fake_execvp(file: str, args: list[str]) -> None:
        assert file == sys.executable
        calls.append(("exec", args))
        raise SystemExit(0)

    def fake_security_checks() -> None:
        calls.append(("security", []))

    monkeypatch.setenv(start_api.MIGRATION_ENV, "true")
    monkeypatch.setattr(start_api.subprocess, "run", fake_run)
    monkeypatch.setattr(start_api, "_run_security_checks", fake_security_checks)
    monkeypatch.setattr(start_api.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit) as exc_info:
        start_api.main()

    assert exc_info.value.code == 0
    assert calls == [
        ("run", [sys.executable, "-m", "alembic", "upgrade", "head"]),
        ("security", []),
        (
            "exec",
            [
                sys.executable,
                "-m",
                "uvicorn",
                "al_medlit.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
            ],
        ),
    ]


def test_start_api_skips_migrations_when_disabled(monkeypatch):
    calls: list[tuple[str, list[str]]] = []

    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"unexpected migration command: {command}, check={check}")

    def fake_execvp(file: str, args: list[str]) -> None:
        assert file == sys.executable
        calls.append(("exec", args))
        raise SystemExit(0)

    def fake_security_checks() -> None:
        calls.append(("security", []))

    monkeypatch.delenv(start_api.MIGRATION_ENV, raising=False)
    monkeypatch.setattr(start_api.subprocess, "run", fake_run)
    monkeypatch.setattr(start_api, "_run_security_checks", fake_security_checks)
    monkeypatch.setattr(start_api.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit) as exc_info:
        start_api.main()

    assert exc_info.value.code == 0
    assert calls[0][0] == "security"
    assert calls[1][0] == "exec"


def test_start_api_exits_when_migration_fails(monkeypatch):
    def fake_run(command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert command == [sys.executable, "-m", "alembic", "upgrade", "head"]
        assert check is False
        return subprocess.CompletedProcess(command, 3)

    def fake_execvp(file: str, args: list[str]) -> None:
        raise AssertionError(f"unexpected API exec: {file} {args}")

    monkeypatch.setenv(start_api.MIGRATION_ENV, "true")
    monkeypatch.setattr(start_api.subprocess, "run", fake_run)
    monkeypatch.setattr(start_api.os, "execvp", fake_execvp)

    with pytest.raises(SystemExit) as exc_info:
        start_api.main()

    assert exc_info.value.code == 3


def test_security_checks_require_a_ready_schema_before_db_query(monkeypatch):
    from al_medlit.auth import service as auth_service
    from al_medlit.core import config, database

    calls: list[str] = []

    class FakeSettings:
        def validate_runtime_secrets(self) -> None:
            calls.append("validate")

        def validate_deployment_secrets(self) -> None:
            calls.append("validate-deployment")

    class FakeSession:
        def __enter__(self):
            calls.append("session-enter")
            return "db"

        def __exit__(self, exc_type, exc, traceback) -> None:
            calls.append("session-exit")

    def fake_ensure_schema_ready() -> None:
        calls.append("schema-ready")

    def fake_assert_no_vulnerable_bootstrap_admin(db) -> None:
        assert db == "db"
        calls.append("bootstrap-check")

    monkeypatch.setattr(config, "settings", FakeSettings())
    monkeypatch.setattr(database, "ensure_schema_ready", fake_ensure_schema_ready)
    monkeypatch.setattr(database, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(
        auth_service,
        "assert_no_vulnerable_bootstrap_admin",
        fake_assert_no_vulnerable_bootstrap_admin,
    )

    start_api._run_security_checks()

    assert calls == [
        "validate",
        "validate-deployment",
        "schema-ready",
        "session-enter",
        "bootstrap-check",
        "session-exit",
    ]
