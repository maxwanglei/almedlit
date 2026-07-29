from __future__ import annotations

import os
import subprocess
import sys
from typing import NoReturn

TRUE_VALUES = {"1", "true", "yes", "on"}
MIGRATION_ENV = "AL_MEDLIT_AUTO_MIGRATE_ON_STARTUP"


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in TRUE_VALUES


def _run_migrations() -> None:
    command = [sys.executable, "-m", "alembic", "upgrade", "head"]
    print("[start_api] Running Alembic migrations: python -m alembic upgrade head", flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"[start_api] Alembic migration failed with exit code {result.returncode}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(result.returncode)


def _run_security_checks() -> None:
    from al_medlit.auth import service as auth_service
    from al_medlit.core.config import settings
    from al_medlit.core.database import SessionLocal, register_models

    settings.validate_runtime_secrets()
    register_models()
    with SessionLocal() as db:
        auth_service.assert_no_vulnerable_bootstrap_admin(db)


def main() -> NoReturn:
    if _enabled(os.getenv(MIGRATION_ENV)):
        _run_migrations()
    else:
        print(f"[start_api] {MIGRATION_ENV} is not enabled; skipping migrations", flush=True)

    _run_security_checks()

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "al_medlit.main:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    print("[start_api] Starting API: python -m uvicorn al_medlit.main:app", flush=True)
    os.execvp(command[0], command)
    raise SystemExit(127)


if __name__ == "__main__":
    main()
