"""Detached wrapper for durable laptop training processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from al_medlit.training.compute.local import LocalComputeBackend


def _write_status(path: Path, payload: dict) -> None:
    payload = {**payload, "updated_at": datetime.now(UTC).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def run(specification_path: str | Path) -> int:
    specification = json.loads(Path(specification_path).read_text(encoding="utf-8"))
    if specification.get("schema_version") != "local-process-v1":
        raise ValueError("Unsupported local process specification")
    command = specification.get("command")
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("Local process command is invalid")
    status_path = Path(specification["status_path"])
    stdout_path = Path(specification["stdout_path"])
    stderr_path = Path(specification["stderr_path"])
    cancel_path = Path(
        specification.get("cancel_path", status_path.parent / "cancel-requested.json")
    )
    launch_token = specification.get("launch_token")
    identity_path = Path(
        specification.get("identity_path", status_path.parent / "process-identity.json")
    )
    identity = {
        "schema_version": "local-process-identity-v1",
        "pid": os.getpid(),
        "start_identity": LocalComputeBackend._process_start_identity(os.getpid()),
        "launch_token": launch_token,
        "created_at": specification.get("created_at") or datetime.now(UTC).isoformat(),
    }
    temporary_identity = identity_path.with_suffix(identity_path.suffix + ".tmp")
    temporary_identity.write_text(json.dumps(identity, sort_keys=True), encoding="utf-8")
    os.replace(temporary_identity, identity_path)
    _write_status(
        status_path,
        {
            "status": "running",
            "exit_code": None,
            "reason": None,
            "raw_state": "RUNNING",
            "worker_pid": os.getpid(),
            "worker_start_identity": identity["start_identity"],
            "launch_token": launch_token,
        },
    )
    environment = {**os.environ, **(specification.get("environment") or {})}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            tuple(command),
            cwd=specification["cwd"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        heartbeat_interval = max(
            1,
            min(int(specification.get("heartbeat_interval_seconds", 15)), 60),
        )
        cancelled = False
        while True:
            try:
                returncode = process.wait(timeout=heartbeat_interval)
                break
            except subprocess.TimeoutExpired:
                if cancel_path.is_file():
                    cancelled = True
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait()
                    break
                _write_status(
                    status_path,
                    {
                        "status": "running",
                        "exit_code": None,
                        "reason": None,
                        "raw_state": "RUNNING",
                        "worker_pid": os.getpid(),
                        "worker_start_identity": identity["start_identity"],
                        "launch_token": launch_token,
                    },
                )
    cancelled = cancelled or cancel_path.is_file()
    failed = returncode != 0 and not cancelled
    reason = None
    if failed:
        try:
            reason = stderr_path.read_text(encoding="utf-8")[-4000:] or "Local job failed"
        except OSError:
            reason = "Local job failed"
    _write_status(
        status_path,
        {
            "status": "cancelled" if cancelled else ("failed" if failed else "succeeded"),
            "exit_code": returncode,
            "reason": "Cancelled by user" if cancelled else reason,
            "raw_state": "CANCELLED" if cancelled else ("FAILED" if failed else "COMPLETED"),
            "worker_pid": os.getpid(),
            "worker_start_identity": identity["start_identity"],
            "launch_token": launch_token,
        },
    )
    return returncode


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print(
            "usage: python -m al_medlit.training.compute.local_worker PROCESS.json",
            file=sys.stderr,
        )
        return 2
    try:
        return run(arguments[0])
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
