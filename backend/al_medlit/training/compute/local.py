import json
import os
import re
import secrets
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from al_medlit.training.compute.base import (
    CommandRunner,
    ComputeBackendError,
    ComputeSubmission,
    JobBundle,
    JobState,
    SubprocessCommandRunner,
)


class LocalComputeBackend:
    """Restart-recoverable local subprocess backend.

    Supplying a CommandRunner retains the synchronous dependency-injected path
    used by unit tests.
    """

    key = "local"

    def __init__(
        self,
        runner: CommandRunner | None = None,
        *,
        runtime_root: str | Path = "/tmp/al-medlit-attempts",
        synchronous: bool = False,
        stale_heartbeat_seconds: int = 180,
    ) -> None:
        self.runner = runner
        self.runtime_root = Path(runtime_root)
        self.synchronous = synchronous or runner is not None
        self.stale_heartbeat_seconds = stale_heartbeat_seconds
        self._states: dict[str, JobState] = {}

    @staticmethod
    def _validate_job_key(job_key: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", job_key):
            raise ComputeBackendError("Invalid local job key")
        return job_key

    def _job_root(self, job_key: str) -> Path:
        return self.runtime_root / self._validate_job_key(job_key)

    @staticmethod
    def _job_key(external_job_id: str) -> str:
        prefix, separator, job_key = external_job_id.partition(":")
        if prefix != "local" or not separator:
            raise ComputeBackendError("Invalid local external job ID")
        return LocalComputeBackend._validate_job_key(job_key)

    @staticmethod
    def _state_from_payload(payload: dict) -> JobState:
        return JobState(
            status=payload.get("status", "unknown"),
            exit_code=payload.get("exit_code"),
            reason=payload.get("reason"),
            raw_state=payload.get("raw_state"),
        )

    @staticmethod
    def _atomic_json(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _process_start_identity(pid: int) -> str | None:
        """Return an OS start identity so PID reuse is never trusted blindly."""

        proc_stat = Path(f"/proc/{pid}/stat")
        try:
            # Field 22 is the start time in clock ticks since boot. The command
            # name can contain spaces and parentheses, so split after the final
            # closing parenthesis rather than tokenizing the whole line.
            suffix = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            start_ticks = suffix[19]
            boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
                encoding="ascii"
            ).strip()
            return f"linux:{boot_id}:{start_ticks}"
        except (FileNotFoundError, IndexError, OSError, UnicodeDecodeError):
            pass
        try:
            result = subprocess.run(
                ("ps", "-o", "lstart=", "-p", str(pid)),
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        value = result.stdout.strip()
        return f"ps:{value}" if result.returncode == 0 and value else None

    @classmethod
    def _identity_matches_live_process(cls, identity: dict) -> bool:
        try:
            pid = int(identity["pid"])
        except (KeyError, TypeError, ValueError):
            return False
        expected_start = identity.get("start_identity")
        actual_start = cls._process_start_identity(pid)
        if expected_start is not None and actual_start is not None:
            return expected_start == actual_start
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _read_identity(self, job_root: Path) -> dict | None:
        try:
            payload = json.loads(
                (job_root / "process-identity.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _is_stale(self, payload: dict, status_path: Path) -> bool:
        raw_updated_at = payload.get("updated_at")
        try:
            updated_at = datetime.fromisoformat(str(raw_updated_at))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            updated_at = datetime.fromtimestamp(status_path.stat().st_mtime, tz=UTC)
        return (datetime.now(UTC) - updated_at).total_seconds() > self.stale_heartbeat_seconds

    def submit(self, bundle: JobBundle) -> ComputeSubmission:
        if not bundle.command:
            raise ComputeBackendError("Local job command cannot be empty")
        self._validate_job_key(bundle.job_key)
        external_job_id = f"local:{bundle.job_key}"
        if not self.synchronous:
            job_root = self._job_root(bundle.job_key)
            status_path = job_root / "status.json"
            if status_path.is_file():
                state = self._state_from_payload(
                    json.loads(status_path.read_text(encoding="utf-8"))
                )
                return ComputeSubmission(
                    external_job_id=external_job_id,
                    status=state.status,
                    metadata={"idempotent": True, "durable": True},
                )
            job_root.mkdir(parents=True, exist_ok=True)
            specification = {
                "schema_version": "local-process-v1",
                "command": list(bundle.command),
                "cwd": str(bundle.local_bundle_path.resolve()),
                "environment": bundle.environment,
                "status_path": str(status_path.resolve()),
                "stdout_path": str((job_root / "stdout.log").resolve()),
                "stderr_path": str((job_root / "stderr.log").resolve()),
                "identity_path": str((job_root / "process-identity.json").resolve()),
                "cancel_path": str((job_root / "cancel-requested.json").resolve()),
                "launch_token": secrets.token_hex(32),
                "heartbeat_interval_seconds": min(15, self.stale_heartbeat_seconds // 3),
                "created_at": datetime.now(UTC).isoformat(),
                "training_job_id": (
                    int(match.group(1))
                    if (
                        match := re.fullmatch(
                            r"training-(\d+)-attempt-\d+", bundle.job_key
                        )
                    )
                    else None
                ),
            }
            spec_path = job_root / "process.json"
            temporary = job_root / "process.json.tmp"
            temporary.write_text(json.dumps(specification, sort_keys=True), encoding="utf-8")
            os.replace(temporary, spec_path)
            process = subprocess.Popen(
                (
                    sys.executable,
                    "-m",
                    "al_medlit.training.compute.local_worker",
                    str(spec_path),
                ),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            identity = {
                "schema_version": "local-process-identity-v1",
                "pid": process.pid,
                "start_identity": self._process_start_identity(process.pid),
                "launch_token": specification["launch_token"],
                "created_at": specification["created_at"],
            }
            self._atomic_json(job_root / "process-identity.json", identity)
            (job_root / "pid").write_text(str(process.pid), encoding="ascii")
            return ComputeSubmission(
                external_job_id=external_job_id,
                status="submitted",
                metadata={"durable": True, "pid": process.pid},
            )
        existing = self._states.get(external_job_id)
        if existing is not None:
            return ComputeSubmission(
                external_job_id=external_job_id,
                status=existing.status,
                metadata={"idempotent": True},
            )
        runner = self.runner or SubprocessCommandRunner()
        result = runner.run(
            bundle.command,
            cwd=bundle.local_bundle_path,
            environment=bundle.environment,
        )
        state = JobState(
            status="succeeded" if result.returncode == 0 else "failed",
            exit_code=result.returncode,
            reason=result.stderr.strip() or None,
            raw_state="COMPLETED" if result.returncode == 0 else "FAILED",
        )
        self._states[external_job_id] = state
        return ComputeSubmission(
            external_job_id=external_job_id,
            status=state.status,
            metadata={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            },
        )

    def poll(self, external_job_id: str) -> JobState:
        if not self.synchronous:
            job_root = self._job_root(self._job_key(external_job_id))
            status_path = job_root / "status.json"
            if (job_root / "cancel-requested.json").is_file():
                return JobState(
                    status="cancelled",
                    reason="Cancelled by user",
                    raw_state="CANCELLED",
                )
            if not status_path.is_file():
                identity = self._read_identity(job_root)
                if identity is not None:
                    try:
                        created_at = datetime.fromisoformat(str(identity["created_at"]))
                        if created_at.tzinfo is None:
                            created_at = created_at.replace(tzinfo=UTC)
                    except (KeyError, TypeError, ValueError):
                        created_at = datetime.fromtimestamp(job_root.stat().st_mtime, tz=UTC)
                    stale_startup = (
                        datetime.now(UTC) - created_at
                    ).total_seconds() > self.stale_heartbeat_seconds
                    if stale_startup and not self._identity_matches_live_process(identity):
                        return JobState(
                            status="failed",
                            reason="Local worker exited before publishing status",
                            raw_state="WORKER_LOST",
                        )
                return JobState(status="submitted", raw_state="STARTING")
            try:
                payload = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                return JobState(status="unknown", reason="Invalid local status document")
            state = self._state_from_payload(payload)
            identity = self._read_identity(job_root)
            if (
                state.status in {"submitted", "running"}
                and identity is not None
                and payload.get("launch_token") is not None
                and payload.get("launch_token") != identity.get("launch_token")
            ):
                return JobState(
                    status="failed",
                    reason="Local worker identity no longer matches its attempt",
                    raw_state="WORKER_IDENTITY_MISMATCH",
                )
            if state.status in {"submitted", "running"} and self._is_stale(
                payload, status_path
            ):
                return JobState(
                    status="failed",
                    reason="Local worker heartbeat became stale",
                    raw_state="STALE_HEARTBEAT",
                )
            return state
        return self._states.get(external_job_id, JobState(status="unknown"))

    def cancel(self, external_job_id: str) -> JobState:
        if not self.synchronous:
            job_root = self._job_root(self._job_key(external_job_id))
            current = self.poll(external_job_id)
            if current.status in {"succeeded", "failed", "cancelled"}:
                return current
            cancellation = {
                "schema_version": "local-cancellation-v1",
                "requested_at": datetime.now(UTC).isoformat(),
            }
            self._atomic_json(job_root / "cancel-requested.json", cancellation)
            try:
                identity = self._read_identity(job_root)
                if identity is not None and self._identity_matches_live_process(identity):
                    os.killpg(int(identity["pid"]), signal.SIGTERM)
            except (FileNotFoundError, OSError, ProcessLookupError, ValueError):
                pass
            payload = {
                "status": "cancelled",
                "exit_code": None,
                "reason": "Cancelled by user",
                "raw_state": "CANCELLED",
                "updated_at": datetime.now(UTC).isoformat(),
            }
            self._atomic_json(job_root / "status.json", payload)
            return self._state_from_payload(payload)
        state = self._states.get(external_job_id)
        if state is None:
            return JobState(status="unknown")
        if state.status in {"succeeded", "failed", "cancelled"}:
            return state
        cancelled = JobState(status="cancelled", raw_state="CANCELLED")
        self._states[external_job_id] = cancelled
        return cancelled
