import math
import os
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

JobStatus = Literal[
    "queued",
    "submitted",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "unknown",
]


class ComputeBackendError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        environment: dict[str, str] | None = None,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Execute an argv vector directly; shell evaluation is never enabled."""

    def __init__(self, *, inherit_environment: bool = True) -> None:
        self.inherit_environment = inherit_environment

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        environment: dict[str, str] | None = None,
        input_text: str | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        process_environment = None if self.inherit_environment else {}
        if environment is not None:
            process_environment = {
                **(os.environ if self.inherit_environment else {}),
                **environment,
            }
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ComputeBackendError("Command timeout must be positive finite seconds")
        timed_process_group = timeout_seconds is not None and os.name == "posix"
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=process_environment,
            text=True,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=timed_process_group,
        )
        try:
            stdout, stderr = process.communicate(input=input_text, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            cleanup_error = None
            try:
                if timed_process_group:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            except OSError as error:
                cleanup_error = f"termination failed: {error}"
            try:
                process.communicate(timeout=1)
            except (subprocess.TimeoutExpired, OSError):
                # Detached descendants can retain pipes after the process group dies.
                for pipe in (process.stdin, process.stdout, process.stderr):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except OSError as error:
                            cleanup_error = f"pipe cleanup failed: {error}"
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    cleanup_error = "process did not exit during cleanup"
            message = f"{Path(argv[0]).name} timed out after {timeout_seconds:g} seconds"
            if cleanup_error is not None:
                message += f" ({cleanup_error})"
            raise ComputeBackendError(message) from exc
        return CommandResult(
            argv=argv,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )


@dataclass(frozen=True, slots=True)
class JobBundle:
    job_key: str
    command: tuple[str, ...]
    local_bundle_path: Path
    entrypoint_path: str = "run.sh"
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComputeSubmission:
    external_job_id: str
    status: JobStatus
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JobState:
    status: JobStatus
    exit_code: int | None = None
    reason: str | None = None
    raw_state: str | None = None


class ComputeBackend(Protocol):
    key: str

    def submit(self, bundle: JobBundle) -> ComputeSubmission: ...

    def poll(self, external_job_id: str) -> JobState: ...

    def cancel(self, external_job_id: str) -> JobState: ...
