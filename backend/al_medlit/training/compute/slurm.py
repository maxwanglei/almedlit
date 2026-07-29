import hashlib
import json
import os
import re
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from al_medlit.training.compute.base import (
    CommandRunner,
    ComputeBackendError,
    ComputeSubmission,
    JobBundle,
    JobState,
    SubprocessCommandRunner,
)

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.@+-]+$")
_SAFE_ARTIFACT_SEGMENT = re.compile(r"^[A-Za-z0-9_.@+-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TERMINAL_STATES = {
    "COMPLETED": "succeeded",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "PREEMPTED": "failed",
    "TIMEOUT": "failed",
    "CANCELLED": "cancelled",
}
_ACTIVE_STATES = {
    "CONFIGURING": "submitted",
    "PENDING": "submitted",
    "REQUEUED": "submitted",
    "RESIZING": "running",
    "RUNNING": "running",
    "SUSPENDED": "running",
    "COMPLETING": "running",
}


@dataclass(frozen=True, slots=True)
class OutputTransferLimits:
    """Hard bounds applied before any remote output payload is downloaded."""

    max_manifest_bytes: int = 1024 * 1024
    max_files: int = 512
    max_file_bytes: int = 16 * 1024 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024 * 1024
    max_path_length: int = 512

    def __post_init__(self) -> None:
        values = (
            self.max_manifest_bytes,
            self.max_files,
            self.max_file_bytes,
            self.max_total_bytes,
            self.max_path_length,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in values
        ):
            raise ComputeBackendError("Output transfer limits must be positive integers")


@dataclass(frozen=True, slots=True)
class RemoteOutputFile:
    relative_path: str
    size_bytes: int
    checksum_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedOutputManifest:
    raw: dict[str, Any]
    files: tuple[RemoteOutputFile, ...]
    total_bytes: int


@dataclass(frozen=True, slots=True)
class InputTransferFile:
    relative_path: str
    local_path: Path
    checksum_sha256: str
    size_bytes: int
    cacheable_base_model: bool = False


@dataclass(frozen=True, slots=True)
class SlurmResources:
    account: str
    partition: str | None = None
    qos: str | None = None
    time_limit: str = "01:00:00"
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 4
    memory: str = "32G"
    gres: str | None = "gpu:1"

    def __post_init__(self) -> None:
        for name in ("account", "partition", "qos"):
            value = getattr(self, name)
            if value is not None and not _SAFE_IDENTIFIER.fullmatch(value):
                raise ComputeBackendError(f"Invalid Slurm {name}")
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", self.time_limit):
            raise ComputeBackendError("time_limit must use HH:MM:SS")
        if min(self.nodes, self.tasks_per_node, self.cpus_per_task) <= 0:
            raise ComputeBackendError("Slurm CPU and node resources must be positive")
        for value, label in ((self.memory, "memory"), (self.gres, "gres")):
            if value is not None and ("\n" in value or "\r" in value):
                raise ComputeBackendError(f"Invalid Slurm {label}")


@dataclass(frozen=True, slots=True)
class SSHSlurmConfig:
    host: str
    username: str
    known_hosts_file: Path
    remote_root: PurePosixPath
    resources: SlurmResources
    apptainer_image: PurePosixPath
    apptainer_module: str
    ssh_port: int = 22
    ssh_identity_file: Path | None = None
    apptainer_args: tuple[str, ...] = ("--nv",)
    module_prelude: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for value, label in ((self.host, "host"), (self.username, "username")):
            if not _SAFE_IDENTIFIER.fullmatch(value):
                raise ComputeBackendError(f"Invalid SSH {label}")
        if not 1 <= self.ssh_port <= 65535:
            raise ComputeBackendError("SSH port is outside the valid range")
        if not self.remote_root.is_absolute() or not self.apptainer_image.is_absolute():
            raise ComputeBackendError("Remote root and Apptainer image must be absolute")
        if any(
            part not in {"/", ""} and not _SAFE_ARTIFACT_SEGMENT.fullmatch(part)
            for part in self.remote_root.parts
        ):
            raise ComputeBackendError("Remote root contains an unsafe path segment")
        if "\n" in self.apptainer_module or "/" not in self.apptainer_module:
            raise ComputeBackendError("Apptainer module must be an explicitly versioned module")


class SSHSlurmComputeBackend:
    key = "ssh_slurm"

    def __init__(
        self,
        config: SSHSlurmConfig,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()

    def ssh_argv(self, remote_argv: tuple[str, ...]) -> tuple[str, ...]:
        return (
            "ssh",
            *self._ssh_options(),
            f"{self.config.username}@{self.config.host}",
            shlex.join(remote_argv),
        )

    def upload_commands(self, bundle: JobBundle) -> tuple[tuple[str, ...], ...]:
        remote_directory = self.remote_job_directory(bundle.job_key)
        scp_options = (
            "-P",
            str(self.config.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.config.known_hosts_file}",
        )
        if self.config.ssh_identity_file is not None:
            scp_options += ("-i", str(self.config.ssh_identity_file))
        return (
            self.ssh_argv(("mkdir", "-p", "--", str(remote_directory))),
            (
                "scp",
                *scp_options,
                "-r",
                str(bundle.local_bundle_path) + "/.",
                f"{self.config.username}@{self.config.host}:{remote_directory}/",
            ),
        )

    def submit(self, bundle: JobBundle) -> ComputeSubmission:
        self._upload_verified_bundle(bundle)
        remote_directory = self.remote_job_directory(bundle.job_key)
        sentinel = remote_directory / ".submitted-job-id"
        temporary_sentinel = remote_directory / ".submitted-job-id.tmp"
        submit_script = (
            "set -e; "
            f"if test -s {shlex.quote(str(sentinel))}; then "
            f"cat {shlex.quote(str(sentinel))}; "
            "else "
            f"submission=$(sbatch --parsable {shlex.quote(str(remote_directory / 'job.sbatch'))}); "
            f"printf '%s\\n' \"$submission\" > {shlex.quote(str(temporary_sentinel))}; "
            f"mv {shlex.quote(str(temporary_sentinel))} {shlex.quote(str(sentinel))}; "
            "printf '%s\\n' \"$submission\"; "
            "fi"
        )
        submit = self.runner.run(self.ssh_argv(("sh", "-c", submit_script)))
        if submit.returncode != 0:
            raise ComputeBackendError(submit.stderr or "sbatch submission failed")
        external_job_id = parse_sbatch_job_id(submit.stdout)
        return ComputeSubmission(
            external_job_id=external_job_id,
            status="submitted",
            metadata={
                "remote_directory": str(remote_directory),
                "submission_sentinel": str(sentinel),
            },
        )

    def _upload_verified_bundle(self, bundle: JobBundle) -> None:
        """Send an input manifest first, verify every file, and cache bases only."""

        job_manifest, files = self._input_transfer_plan(bundle)
        remote_directory = self.remote_job_directory(bundle.job_key)
        self._run_required(
            self.ssh_argv(("mkdir", "-p", "--", str(remote_directory))),
            "Unable to create the remote job directory",
        )
        transfer_manifest = {
            "schema_version": "al-medlit-input-transfer-v1",
            "job_key": bundle.job_key,
            "workspace_id": job_manifest.get("workspace_id"),
            "files": {
                item.relative_path: {
                    "checksum_sha256": item.checksum_sha256,
                    "size_bytes": item.size_bytes,
                    "cacheable_base_model": item.cacheable_base_model,
                }
                for item in files
            },
        }
        manifest_path = bundle.local_bundle_path / ".al-medlit-transfer-manifest.json"
        manifest_path.write_text(
            json.dumps(transfer_manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self._upload_regular_file(
            local_path=manifest_path,
            remote_path=remote_directory / "transfer-manifest.json",
            checksum_sha256=sha256_path(manifest_path),
            size_bytes=manifest_path.stat().st_size,
        )
        workspace_id = job_manifest.get("workspace_id")
        for item in files:
            remote_path = remote_directory / PurePosixPath(item.relative_path)
            if item.cacheable_base_model and isinstance(workspace_id, int):
                cache_path = (
                    self.config.remote_root
                    / ".base-model-cache"
                    / "workspaces"
                    / str(workspace_id)
                    / "sha256"
                    / item.checksum_sha256[:2]
                    / item.checksum_sha256
                )
                self._ensure_cached_base_model(item, cache_path, bundle.job_key)
                self._materialize_cached_file(
                    cache_path=cache_path,
                    destination=remote_path,
                    checksum_sha256=item.checksum_sha256,
                    size_bytes=item.size_bytes,
                )
            else:
                self._upload_regular_file(
                    local_path=item.local_path,
                    remote_path=remote_path,
                    checksum_sha256=item.checksum_sha256,
                    size_bytes=item.size_bytes,
                )

    def _input_transfer_plan(
        self,
        bundle: JobBundle,
    ) -> tuple[dict[str, Any], tuple[InputTransferFile, ...]]:
        root = bundle.local_bundle_path.resolve()
        job_path = root / "job.json"
        if job_path.is_symlink() or not job_path.is_file():
            raise ComputeBackendError("Slurm bundles require a regular job.json manifest")
        try:
            manifest = json.loads(job_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ComputeBackendError("Slurm bundle job.json is invalid") from exc
        if manifest.get("schema_version") != "al-medlit-job-v1":
            raise ComputeBackendError("Slurm bundle job manifest schema is unsupported")
        workspace_id = manifest.get("workspace_id")
        if not isinstance(workspace_id, int) or isinstance(workspace_id, bool) or workspace_id <= 0:
            raise ComputeBackendError("Slurm job manifest requires a trusted workspace ID")

        declarations: dict[str, tuple[str | None, int | None, bool]] = {
            "job.json": (None, None, False),
            "job.sbatch": (None, None, False),
        }
        for key in ("dataset", "checkpoint", "corpus", "staged_base_model", "base_model"):
            value = manifest.get(key)
            if not isinstance(value, dict):
                continue
            cacheable = key in {"staged_base_model", "base_model"} and isinstance(
                value.get("base_model_asset_id"), int
            )
            if value.get("path"):
                declarations[str(value["path"])] = (
                    value.get("checksum_sha256"),
                    value.get("size_bytes"),
                    cacheable,
                )
            package = value.get("package")
            if not isinstance(package, dict):
                continue
            package_files = package.get("files")
            if not isinstance(package_files, list) or not package_files:
                raise ComputeBackendError(f"Slurm {key} package declaration is incomplete")
            for package_file in package_files:
                if not isinstance(package_file, dict) or not package_file.get("path"):
                    raise ComputeBackendError(f"Slurm {key} package file is invalid")
                declarations[str(package_file["path"])] = (
                    package_file.get("checksum_sha256"),
                    package_file.get("size_bytes"),
                    cacheable,
                )

        files: list[InputTransferFile] = []
        total_bytes = 0
        for relative_path, (declared_checksum, declared_size, cacheable) in sorted(
            declarations.items()
        ):
            relative = validate_remote_relative_path(relative_path)
            local_path = (root / relative.as_posix()).resolve()
            if (
                not local_path.is_relative_to(root)
                or local_path.is_symlink()
                or not local_path.is_file()
            ):
                raise ComputeBackendError(
                    f"Slurm input is missing or unsafe: {relative.as_posix()}"
                )
            size_bytes = local_path.stat().st_size
            checksum = sha256_path(local_path)
            if declared_size is not None and declared_size != size_bytes:
                raise ComputeBackendError(
                    f"Slurm input size differs from job manifest: {relative.as_posix()}"
                )
            if declared_checksum is not None and declared_checksum != checksum:
                raise ComputeBackendError(
                    f"Slurm input checksum differs from job manifest: {relative.as_posix()}"
                )
            total_bytes += size_bytes
            if total_bytes > 128 * 1024 * 1024 * 1024:
                raise ComputeBackendError("Slurm input bundle exceeds the transfer byte limit")
            files.append(
                InputTransferFile(
                    relative_path=relative.as_posix(),
                    local_path=local_path,
                    checksum_sha256=checksum,
                    size_bytes=size_bytes,
                    cacheable_base_model=cacheable,
                )
            )
        return manifest, tuple(files)

    def _ensure_cached_base_model(
        self,
        item: InputTransferFile,
        cache_path: PurePosixPath,
        job_key: str,
    ) -> None:
        self._run_required(
            self.ssh_argv(("mkdir", "-p", "--", str(cache_path.parent))),
            "Unable to create the base-model cache directory",
        )
        exists = self.runner.run(self.ssh_argv(("test", "-f", str(cache_path))))
        if exists.returncode == 0:
            self._verify_remote_file(
                cache_path,
                checksum_sha256=item.checksum_sha256,
                size_bytes=item.size_bytes,
            )
            return
        if exists.returncode != 1:
            raise ComputeBackendError(exists.stderr or "Unable to inspect base-model cache")
        temporary = cache_path.parent / f".{item.checksum_sha256}.{job_key}.upload"
        self._upload_regular_file(
            local_path=item.local_path,
            remote_path=temporary,
            checksum_sha256=item.checksum_sha256,
            size_bytes=item.size_bytes,
        )
        linked = self.runner.run(self.ssh_argv(("ln", str(temporary), str(cache_path))))
        if linked.returncode not in {0, 1}:
            raise ComputeBackendError(linked.stderr or "Unable to publish cached base model")
        self._run_required(
            self.ssh_argv(("rm", "-f", "--", str(temporary))),
            "Unable to remove temporary cached input",
        )
        self._verify_remote_file(
            cache_path,
            checksum_sha256=item.checksum_sha256,
            size_bytes=item.size_bytes,
        )

    def _materialize_cached_file(
        self,
        *,
        cache_path: PurePosixPath,
        destination: PurePosixPath,
        checksum_sha256: str,
        size_bytes: int,
    ) -> None:
        self._run_required(
            self.ssh_argv(("mkdir", "-p", "--", str(destination.parent))),
            "Unable to create the remote input directory",
        )
        exists = self.runner.run(self.ssh_argv(("test", "-f", str(destination))))
        if exists.returncode == 1:
            self._run_required(
                self.ssh_argv(("ln", str(cache_path), str(destination))),
                "Unable to materialize cached base model",
            )
        elif exists.returncode != 0:
            raise ComputeBackendError(exists.stderr or "Unable to inspect remote input")
        self._verify_remote_file(
            destination,
            checksum_sha256=checksum_sha256,
            size_bytes=size_bytes,
        )

    def _upload_regular_file(
        self,
        *,
        local_path: Path,
        remote_path: PurePosixPath,
        checksum_sha256: str,
        size_bytes: int,
    ) -> None:
        self._run_required(
            self.ssh_argv(("mkdir", "-p", "--", str(remote_path.parent))),
            "Unable to create the remote input directory",
        )
        temporary = remote_path.parent / f".{remote_path.name}.upload"
        command = (
            "scp",
            *self._scp_options(),
            str(local_path),
            f"{self.config.username}@{self.config.host}:{temporary}",
        )
        self._run_required(command, "Unable to transfer a Slurm input")
        self._verify_remote_file(
            temporary,
            checksum_sha256=checksum_sha256,
            size_bytes=size_bytes,
        )
        self._run_required(
            self.ssh_argv(("mv", "-f", "--", str(temporary), str(remote_path))),
            "Unable to publish a verified Slurm input",
        )

    def _verify_remote_file(
        self,
        remote_path: PurePosixPath,
        *,
        checksum_sha256: str,
        size_bytes: int,
    ) -> None:
        stat = self.runner.run(self.ssh_argv(("stat", "--format=%s", "--", str(remote_path))))
        if stat.returncode != 0 or stat.stdout.strip() != str(size_bytes):
            raise ComputeBackendError("Remote input size verification failed")
        digest = self.runner.run(self.ssh_argv(("sha256sum", "--", str(remote_path))))
        if digest.returncode != 0:
            raise ComputeBackendError(digest.stderr or "Remote input checksum failed")
        remote_checksum = digest.stdout.strip().split(maxsplit=1)[0]
        if remote_checksum != checksum_sha256:
            raise ComputeBackendError("Remote input checksum verification failed")

    def _run_required(self, command: tuple[str, ...], message: str) -> None:
        result = self.runner.run(command)
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or message)

    def poll(self, external_job_id: str) -> JobState:
        job_id = validate_job_id(external_job_id)
        queue = self.runner.run(
            self.ssh_argv(("squeue", "--noheader", "--jobs", job_id, "--format", "%T"))
        )
        if queue.returncode != 0:
            raise ComputeBackendError(queue.stderr or "squeue failed")
        active_state = queue.stdout.strip().splitlines()
        if active_state:
            raw = normalize_slurm_state(active_state[0])
            return JobState(status=_ACTIVE_STATES.get(raw, "unknown"), raw_state=raw)
        accounting = self.runner.run(
            self.ssh_argv(
                (
                    "sacct",
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    job_id,
                    "--format",
                    "State,ExitCode",
                )
            )
        )
        if accounting.returncode != 0:
            raise ComputeBackendError(accounting.stderr or "sacct failed")
        return parse_sacct_state(accounting.stdout)

    def cancel(self, external_job_id: str) -> JobState:
        job_id = validate_job_id(external_job_id)
        result = self.runner.run(self.ssh_argv(("scancel", job_id)))
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or "scancel failed")
        return JobState(status="cancelled", raw_state="CANCELLED")

    def cleanup_job_directory(self, job_key: str) -> None:
        """Remove one validated, attempt-scoped remote copy after retention expiry."""

        remote_directory = self.remote_job_directory(job_key)
        result = self.runner.run(self.ssh_argv(("rm", "-rf", "--", str(remote_directory))))
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or "Remote job cleanup failed")

    def artifact_download_command(
        self,
        *,
        job_key: str,
        remote_relative_path: str,
        local_destination: Path,
    ) -> tuple[str, ...]:
        relative = validate_remote_relative_path(remote_relative_path)
        remote = self.remote_job_directory(job_key) / relative
        return (
            "scp",
            *self._scp_options(),
            f"{self.config.username}@{self.config.host}:{remote}",
            str(local_destination),
        )

    def remote_file_size(self, *, job_key: str, remote_relative_path: str) -> int:
        """Read bounded-file metadata without transferring the remote file."""

        relative = validate_remote_relative_path(remote_relative_path)
        remote = self.remote_job_directory(job_key) / relative
        result = self.runner.run(self.ssh_argv(("stat", "--format=%s", "--", str(remote))))
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or "Unable to stat remote artifact")
        output = result.stdout.strip()
        if not re.fullmatch(r"[0-9]+", output):
            raise ComputeBackendError("Remote artifact size is invalid")
        return int(output)

    def read_metric_lines(
        self,
        *,
        job_key: str,
        after_line: int = 0,
        max_lines: int = 5000,
        max_bytes: int = 4 * 1024 * 1024,
    ) -> tuple[str, ...]:
        """Read a bounded append-only metric snapshot from an active job.

        The remote file can be written while it is being read.  Only records
        ending in a newline are returned, so a partially written final JSONL
        record is retried from the same line on the next reconciliation pass.
        """

        for value, label, upper_bound in (
            (after_line, "metric cursor", 2**63 - 1),
            (max_lines, "metric line limit", 5000),
            (max_bytes, "metric byte limit", 4 * 1024 * 1024),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (0 if label == "metric cursor" else 1)
                or value > upper_bound
            ):
                raise ComputeBackendError(f"Invalid {label}")

        remote = self.remote_job_directory(job_key) / "outputs" / "metric-points.jsonl"
        first_line = after_line + 1
        quoted_remote = shlex.quote(str(remote))
        script = (
            f"if test -f {quoted_remote}; then "
            f"tail -n +{first_line} -- {quoted_remote} | "
            f"head -n {max_lines} | head -c {max_bytes}; "
            "fi"
        )
        result = self.runner.run(self.ssh_argv(("sh", "-c", script)))
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or "Unable to read remote training metrics")
        if not result.stdout:
            return ()
        if len(result.stdout.encode("utf-8")) > max_bytes:
            raise ComputeBackendError("Remote training metrics exceeded the byte limit")

        # JSONL writers publish a record by appending its trailing newline.  A
        # non-terminated suffix is either still being written or was clipped by
        # the byte bound and must not advance the durable cursor.
        complete_payload, separator, _partial = result.stdout.rpartition("\n")
        if not separator:
            return ()
        return tuple(complete_payload.splitlines())

    def collect_outputs(
        self,
        *,
        job_key: str,
        output_root: Path,
        limits: OutputTransferLimits | None = None,
        manifest_relative_path: str = "outputs/artifact-manifest.json",
    ) -> dict[str, Any]:
        """Collect declared outputs only after a manifest-first bounded preflight.

        Payload files are first downloaded to an isolated staging directory. They
        are published under ``output_root`` only after every declared size and
        SHA-256 checksum has been verified.
        """

        transfer_limits = limits or OutputTransferLimits()
        remote_manifest = validate_remote_relative_path(
            manifest_relative_path,
            max_length=transfer_limits.max_path_length,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        resolved_root = output_root.resolve()

        manifest_size = self.remote_file_size(
            job_key=job_key,
            remote_relative_path=remote_manifest.as_posix(),
        )
        if manifest_size > transfer_limits.max_manifest_bytes:
            raise ComputeBackendError("Remote artifact manifest exceeds the byte limit")

        with tempfile.TemporaryDirectory(
            prefix=".slurm-output-",
            dir=resolved_root,
        ) as temporary_directory:
            staging_root = Path(temporary_directory)
            staged_manifest = staging_root / "artifact-manifest.json"
            self._download_remote_file(
                job_key=job_key,
                remote_relative_path=remote_manifest.as_posix(),
                destination=staged_manifest,
            )
            if staged_manifest.stat().st_size != manifest_size:
                raise ComputeBackendError("Remote artifact manifest changed during transfer")
            try:
                manifest_payload = json.loads(staged_manifest.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ComputeBackendError("Remote artifact manifest is invalid JSON") from exc
            validated = validate_output_manifest(manifest_payload, transfer_limits)

            for artifact in validated.files:
                staged_path = safe_local_output_path(staging_root, artifact.relative_path)
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                remote_relative_path = (remote_manifest.parent / artifact.relative_path).as_posix()
                if (
                    self.remote_file_size(
                        job_key=job_key,
                        remote_relative_path=remote_relative_path,
                    )
                    != artifact.size_bytes
                ):
                    raise ComputeBackendError(
                        f"Remote artifact size differs from manifest: {artifact.relative_path}"
                    )
                self._download_remote_file(
                    job_key=job_key,
                    remote_relative_path=remote_relative_path,
                    destination=staged_path,
                )
                if staged_path.stat().st_size != artifact.size_bytes:
                    raise ComputeBackendError(
                        f"Remote artifact size mismatch: {artifact.relative_path}"
                    )
                if sha256_path(staged_path) != artifact.checksum_sha256:
                    raise ComputeBackendError(
                        f"Remote artifact checksum mismatch: {artifact.relative_path}"
                    )

            for artifact in validated.files:
                staged_path = safe_local_output_path(staging_root, artifact.relative_path)
                final_path = safe_local_output_path(resolved_root, artifact.relative_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_path, final_path)
            # The manifest is the publication marker and therefore moves last.
            final_manifest = safe_local_output_path(
                resolved_root,
                "artifact-manifest.json",
            )
            os.replace(staged_manifest, final_manifest)

        return validated.raw

    def _download_remote_file(
        self,
        *,
        job_key: str,
        remote_relative_path: str,
        destination: Path,
    ) -> None:
        result = self.runner.run(
            self.artifact_download_command(
                job_key=job_key,
                remote_relative_path=remote_relative_path,
                local_destination=destination,
            )
        )
        if result.returncode != 0:
            raise ComputeBackendError(result.stderr or "Unable to download remote artifact")
        if destination.is_symlink() or not destination.is_file():
            raise ComputeBackendError("Remote artifact download did not produce a regular file")

    def remote_job_directory(self, job_key: str) -> PurePosixPath:
        if not _SAFE_IDENTIFIER.fullmatch(job_key):
            raise ComputeBackendError("Invalid job key")
        return self.config.remote_root / job_key

    def render_sbatch_script(self, bundle: JobBundle) -> str:
        resources = self.config.resources
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={bundle.job_key[:80]}",
            f"#SBATCH --account={resources.account}",
            f"#SBATCH --time={resources.time_limit}",
            f"#SBATCH --nodes={resources.nodes}",
            f"#SBATCH --ntasks-per-node={resources.tasks_per_node}",
            f"#SBATCH --cpus-per-task={resources.cpus_per_task}",
            f"#SBATCH --mem={resources.memory}",
            "#SBATCH --output=slurm-%j.out",
            "#SBATCH --error=slurm-%j.err",
        ]
        if resources.partition:
            directives.append(f"#SBATCH --partition={resources.partition}")
        if resources.qos:
            directives.append(f"#SBATCH --qos={resources.qos}")
        if resources.gres:
            directives.append(f"#SBATCH --gres={resources.gres}")
        remote_directory = self.remote_job_directory(bundle.job_key)
        command = (
            "apptainer",
            "exec",
            *self.config.apptainer_args,
            str(self.config.apptainer_image),
            *bundle.command,
        )
        body = [
            "set -euo pipefail",
            *self.config.module_prelude,
            f"module load {shlex.quote(self.config.apptainer_module)}",
            f"cd {shlex.quote(str(remote_directory))}",
            shlex.join(command),
        ]
        return "\n".join((*directives, "", *body, ""))

    def _ssh_options(self) -> tuple[str, ...]:
        options = (
            "-p",
            str(self.config.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.config.known_hosts_file}",
        )
        if self.config.ssh_identity_file is not None:
            options += ("-i", str(self.config.ssh_identity_file))
        return options

    def _scp_options(self) -> tuple[str, ...]:
        options = (
            "-P",
            str(self.config.ssh_port),
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.config.known_hosts_file}",
        )
        if self.config.ssh_identity_file is not None:
            options += ("-i", str(self.config.ssh_identity_file))
        return options


def validate_remote_relative_path(
    value: str,
    *,
    max_length: int = 1024,
) -> PurePosixPath:
    """Return a canonical shell-safe relative POSIX artifact path."""

    if not isinstance(value, str) or not value or len(value) > max_length:
        raise ComputeBackendError("Artifact path is empty or exceeds the path limit")
    if "\\" in value or "\x00" in value:
        raise ComputeBackendError("Artifact path must be a canonical POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative.as_posix() != value:
        raise ComputeBackendError("Artifact path must be a canonical relative path")
    if not relative.parts or any(
        part in {"", ".", ".."} or not _SAFE_ARTIFACT_SEGMENT.fullmatch(part)
        for part in relative.parts
    ):
        raise ComputeBackendError("Artifact path contains an unsafe segment")
    return relative


def validate_output_manifest(
    payload: Any,
    limits: OutputTransferLimits | None = None,
) -> ValidatedOutputManifest:
    """Validate all output declarations and limits before payload transfer."""

    transfer_limits = limits or OutputTransferLimits()
    if not isinstance(payload, dict):
        raise ComputeBackendError("Remote artifact manifest must be a JSON object")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict):
        raise ComputeBackendError("Remote artifact manifest files must be an object")
    if len(raw_files) > transfer_limits.max_files:
        raise ComputeBackendError("Remote artifact manifest exceeds the file-count limit")

    files: list[RemoteOutputFile] = []
    total_bytes = 0
    for raw_path, raw_metadata in raw_files.items():
        relative = validate_remote_relative_path(
            raw_path,
            max_length=transfer_limits.max_path_length,
        )
        if relative.as_posix() == "artifact-manifest.json":
            raise ComputeBackendError("Output manifest cannot declare itself as a payload")
        if not isinstance(raw_metadata, dict):
            raise ComputeBackendError(f"Artifact metadata must be an object: {raw_path}")
        size_bytes = raw_metadata.get("size_bytes")
        checksum = raw_metadata.get("checksum_sha256")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ComputeBackendError(f"Artifact size is invalid: {raw_path}")
        if size_bytes > transfer_limits.max_file_bytes:
            raise ComputeBackendError(f"Artifact exceeds the per-file byte limit: {raw_path}")
        if not isinstance(checksum, str) or not _SHA256.fullmatch(checksum):
            raise ComputeBackendError(f"Artifact SHA-256 is invalid: {raw_path}")
        total_bytes += size_bytes
        if total_bytes > transfer_limits.max_total_bytes:
            raise ComputeBackendError("Remote artifacts exceed the total byte limit")
        files.append(
            RemoteOutputFile(
                relative_path=relative.as_posix(),
                size_bytes=size_bytes,
                checksum_sha256=checksum,
            )
        )

    return ValidatedOutputManifest(
        raw=payload,
        files=tuple(files),
        total_bytes=total_bytes,
    )


def safe_local_output_path(root: Path, relative_path: str) -> Path:
    relative = validate_remote_relative_path(relative_path)
    resolved_root = root.resolve()
    unresolved = resolved_root / Path(*relative.parts)
    cursor = resolved_root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ComputeBackendError("Artifact destination contains a symbolic link")
    candidate = unresolved.resolve()
    if not candidate.is_relative_to(resolved_root):
        raise ComputeBackendError("Artifact path escapes the local output directory")
    return candidate


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sbatch_job_id(output: str) -> str:
    candidate = output.strip().split(";", 1)[0]
    return validate_job_id(candidate)


def validate_job_id(value: str) -> str:
    if not re.fullmatch(r"\d+(?:_[0-9]+)?", value):
        raise ComputeBackendError("Scheduler returned an invalid Slurm job ID")
    return value


def normalize_slurm_state(value: str) -> str:
    return value.strip().upper().split("+", 1)[0].split(" ", 1)[0]


def parse_sacct_state(output: str) -> JobState:
    rows = [line for line in output.splitlines() if line.strip()]
    if not rows:
        return JobState(status="unknown")
    state_text, _, exit_text = rows[0].partition("|")
    raw_state = normalize_slurm_state(state_text)
    exit_code: int | None = None
    if exit_text:
        try:
            exit_code = int(exit_text.split(":", 1)[0])
        except ValueError:
            pass
    status = _TERMINAL_STATES.get(raw_state, _ACTIVE_STATES.get(raw_state, "unknown"))
    return JobState(status=status, exit_code=exit_code, raw_state=raw_state)


def osc_ascend_profile(
    *,
    username: str,
    account: str,
    known_hosts_file: Path,
    remote_root: PurePosixPath,
    apptainer_image: PurePosixPath,
    apptainer_module: str,
    partition: str,
    time_limit: str = "01:00:00",
    memory: str = "64G",
    gres: str = "gpu:1",
) -> SSHSlurmConfig:
    """Build an explicit, non-interactive OSC Ascend profile.

    Account, partition, image, and versioned module are deliberately required;
    AL-MedLit must not silently guess site allocations or mutable environments.
    """
    return SSHSlurmConfig(
        host="ascend.osc.edu",
        username=username,
        known_hosts_file=known_hosts_file,
        remote_root=remote_root,
        resources=SlurmResources(
            account=account,
            partition=partition,
            time_limit=time_limit,
            memory=memory,
            gres=gres,
        ),
        apptainer_image=apptainer_image,
        apptainer_module=apptainer_module,
    )
