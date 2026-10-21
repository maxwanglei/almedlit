"""Own training processes independently of Celery's eager or worker pool mode."""

from __future__ import annotations

import logging
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from al_medlit.core.config import settings

logger = logging.getLogger(__name__)
_DATABASE_TIMEOUT_SECONDS = 10.0


def _acquire_process_accelerator_slot() -> None:
    from al_medlit.training.runtime_profiles import RUNTIME_PROFILES

    descriptor = RUNTIME_PROFILES.get(os.environ.get("AL_MEDLIT_RUNTIME_PROFILE", ""))
    if descriptor is None or descriptor.required_device == "cpu":
        return

    import fcntl

    root = Path(settings.local_attempt_root)
    root.mkdir(parents=True, exist_ok=True)
    slot = os.open(root / ".accelerator.lock", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(slot, fcntl.LOCK_EX)
    except BaseException:
        os.close(slot)
        raise
    # The descriptor stays open until process exit, including CUDA teardown.


@dataclass(frozen=True)
class _ProcessOutcome:
    reason: str
    database_operation_pending: bool


def _database_operation[T](operation: Callable[[], T]) -> queue.Queue:
    result: queue.Queue = queue.Queue(maxsize=1)

    def run() -> None:
        try:
            result.put((True, operation()))
        except Exception as exc:
            result.put((False, exc))

    threading.Thread(target=run, daemon=True, name="training-lease").start()
    return result


def _database_result[T](operation: Callable[[], T]) -> T:
    try:
        succeeded, value = _database_operation(operation).get(timeout=_DATABASE_TIMEOUT_SECONDS)
    except queue.Empty as exc:
        raise TimeoutError("Training supervision database operation timed out") from exc
    if not succeeded:
        raise value
    return value


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def _watch_process(
    process: subprocess.Popen,
    *,
    parent_fd: int,
    heartbeat: Callable[[], bool],
    max_runtime_seconds: float,
    poll_seconds: float,
    grace_seconds: float,
    database_timeout_seconds: float = _DATABASE_TIMEOUT_SECONDS,
    lease_failure_timeout_seconds: float = 240,
) -> _ProcessOutcome:
    deadline = time.monotonic() + max_runtime_seconds
    next_heartbeat = time.monotonic()
    last_heartbeat = next_heartbeat
    pending: queue.Queue | None = None
    heartbeat_deadline = deadline
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                return _ProcessOutcome("execution_deadline_exceeded", pending is not None)
            if process.poll() is not None:
                return _ProcessOutcome(
                    "completed" if process.returncode == 0 else "worker_process_failed",
                    pending is not None,
                )
            if select.select([parent_fd], [], [], 0)[0] and not os.read(parent_fd, 1):
                return _ProcessOutcome("task_supervisor_lost", pending is not None)
            if pending is not None:
                try:
                    succeeded, value = pending.get_nowait()
                except queue.Empty:
                    if now >= heartbeat_deadline:
                        return _ProcessOutcome("worker_lease_check_timed_out", pending is not None)
                else:
                    pending = None
                    if not succeeded:
                        logger.warning("Training lease renewal will retry: %s", value)
                    elif not value:
                        return _ProcessOutcome("worker_lease_lost", False)
                    else:
                        last_heartbeat = now
                    next_heartbeat = now + poll_seconds
            if now - last_heartbeat >= lease_failure_timeout_seconds:
                return _ProcessOutcome("worker_lease_renewal_expired", pending is not None)
            if pending is None and now >= next_heartbeat:
                pending = _database_operation(heartbeat)
                heartbeat_deadline = now + database_timeout_seconds
            select.select([parent_fd], [], [], min(poll_seconds, 0.1, max(0, deadline - now)))
    finally:
        _terminate_process_group(process, grace_seconds)


def _session_operation(operation):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import Session

    from al_medlit.core.database import register_models

    register_models()
    url = make_url(settings.database_url)
    connect_args = {}
    if url.get_backend_name() == "postgresql":
        connect_args = {
            "connect_timeout": 5,
            "options": "-c statement_timeout=5000 -c lock_timeout=5000",
        }
    elif url.get_backend_name() == "sqlite":
        connect_args = {"timeout": 5}
    engine = create_engine(url, connect_args=connect_args)
    try:
        with Session(engine, autoflush=False) as db:
            return operation(db)
    finally:
        engine.dispose()


def supervise_training_run(training_run_id: int, *, parent_fd: int) -> int:
    from al_medlit.training.run_execution import (
        _claim_training_run,
        _mark_failed,
        heartbeat_training_run,
    )

    token = uuid.uuid4().hex

    def claim(db):
        try:
            _, claimed = _claim_training_run(db, training_run_id, execution_token=token)
            return claimed
        except Exception as exc:
            _mark_failed(db, training_run_id, exc, expected_status="queued")
            raise

    if not _database_result(lambda: _session_operation(claim)):
        return training_run_id
    database_busy = False
    process = None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "al_medlit.training.supervision",
                "compute",
                str(training_run_id),
                token,
            ],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        outcome = _watch_process(
            process,
            parent_fd=parent_fd,
            heartbeat=lambda: _session_operation(
                lambda db: heartbeat_training_run(db, training_run_id, execution_token=token)
            ),
            max_runtime_seconds=settings.training_max_runtime_seconds,
            poll_seconds=settings.training_supervisor_poll_seconds,
            grace_seconds=settings.training_termination_grace_seconds,
        )
        database_busy = outcome.database_operation_pending
        if outcome.reason != "completed":
            raise RuntimeError(outcome.reason)
    except Exception as exc:
        failure = exc
        if not database_busy:
            try:
                _database_result(
                    lambda: _session_operation(
                        lambda db: _mark_failed(db, training_run_id, failure, execution_token=token)
                    )
                )
            except Exception:
                logger.exception("Could not persist training run %s failure", training_run_id)
        raise
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_group(process, settings.training_termination_grace_seconds)
    return training_run_id


def run_supervised_training(training_run_id: int) -> int:
    if os.name != "posix":
        raise RuntimeError("Training process supervision requires a POSIX runtime")
    process = subprocess.Popen(
        [sys.executable, "-m", "al_medlit.training.supervision", "supervise", str(training_run_id)],
        stdin=subprocess.PIPE,
    )
    try:
        if process.wait() != 0:
            raise RuntimeError(f"Training supervisor failed for run {training_run_id}")
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.wait(timeout=settings.training_termination_grace_seconds + 30)
    return training_run_id


def main() -> None:
    mode, run_id, *arguments = sys.argv[1:]
    if mode == "supervise":

        def stop(signum, _frame):
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGTERM, stop)
        supervise_training_run(int(run_id), parent_fd=sys.stdin.fileno())
    elif mode == "compute":
        _acquire_process_accelerator_slot()
        from al_medlit.core.database import SessionLocal, register_models
        from al_medlit.core.storage import get_object_storage
        from al_medlit.training.run_execution import execute_training_run

        register_models()
        with SessionLocal() as db:
            execute_training_run(
                db,
                get_object_storage(),
                training_run_id=int(run_id),
                execution_token=arguments[0],
            )
    else:
        raise ValueError("Unknown training supervision mode")


if __name__ == "__main__":
    main()
