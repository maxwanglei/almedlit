import os
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
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Execute an argv vector directly; shell evaluation is never enabled."""

    def run(
        self,
        argv: tuple[str, ...],
        *,
        cwd: str | Path | None = None,
        environment: dict[str, str] | None = None,
        input_text: str | None = None,
    ) -> CommandResult:
        process_environment = None
        if environment is not None:
            process_environment = {**os.environ, **environment}
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=process_environment,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        return CommandResult(
            argv=argv,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
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
