import hashlib
import json
import shlex
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath

import pytest

from al_medlit.training.compute.base import CommandResult, JobBundle
from al_medlit.training.compute.config import validate_compute_config
from al_medlit.training.compute.local import LocalComputeBackend
from al_medlit.training.compute.local_worker import run as run_local_worker
from al_medlit.training.compute.slurm import (
    SlurmResources,
    SSHSlurmComputeBackend,
    SSHSlurmConfig,
    parse_sacct_state,
    parse_sbatch_job_id,
)


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.commands = []

    def run(self, argv, **_kwargs):
        self.commands.append(argv)
        if self.results:
            return self.results.pop(0)
        return CommandResult(argv=argv, returncode=0, stdout="", stderr="")


class VirtualRemoteRunner:
    def __init__(self):
        self.commands = []
        self.remote_files = {}

    def run(self, argv, **_kwargs):
        self.commands.append(argv)
        if argv[0] == "scp":
            source = Path(argv[-2])
            remote_path = argv[-1].split(":", 1)[1]
            self.remote_files[remote_path] = source.read_bytes()
            return _result()
        command = shlex.split(argv[-1])
        operation = command[0]
        if operation == "mkdir":
            return _result()
        if operation == "test":
            return _result(returncode=0 if command[-1] in self.remote_files else 1)
        if operation == "stat":
            payload = self.remote_files.get(command[-1])
            if payload is None:
                return _result(stderr="missing", returncode=1)
            return _result(stdout=f"{len(payload)}\n")
        if operation == "sha256sum":
            payload = self.remote_files.get(command[-1])
            if payload is None:
                return _result(stderr="missing", returncode=1)
            digest = hashlib.sha256(payload).hexdigest()
            return _result(stdout=f"{digest}  {command[-1]}\n")
        if operation == "mv":
            source, destination = command[-2:]
            self.remote_files[destination] = self.remote_files.pop(source)
            return _result()
        if operation == "ln":
            source, destination = command[-2:]
            if destination in self.remote_files:
                return _result(returncode=1)
            self.remote_files[destination] = self.remote_files[source]
            return _result()
        if operation == "rm":
            self.remote_files.pop(command[-1], None)
            return _result()
        raise AssertionError(f"Unsupported virtual remote command: {command}")


def _result(stdout="", stderr="", returncode=0):
    return CommandResult(argv=(), returncode=returncode, stdout=stdout, stderr=stderr)


def _slurm_config(tmp_path):
    return SSHSlurmConfig(
        host="cluster.example.org",
        username="researcher",
        known_hosts_file=tmp_path / "known_hosts",
        remote_root=PurePosixPath("/scratch/al-medlit"),
        resources=SlurmResources(account="project-1", partition="gpu"),
        apptainer_image=PurePosixPath("/images/al-medlit.sif"),
        apptainer_module="apptainer/1.3.6",
    )


def test_local_backend_is_idempotent_and_never_uses_a_shell(tmp_path):
    runner = FakeRunner([_result(stdout="trained")])
    backend = LocalComputeBackend(runner)
    bundle = JobBundle(
        job_key="job-1",
        command=("python", "train.py"),
        local_bundle_path=tmp_path,
    )
    first = backend.submit(bundle)
    second = backend.submit(bundle)

    assert first.status == "succeeded"
    assert second.metadata == {"idempotent": True}
    assert runner.commands == [("python", "train.py")]


def test_local_backend_classifies_a_stale_worker_heartbeat(tmp_path):
    backend = LocalComputeBackend(
        runtime_root=tmp_path,
        stale_heartbeat_seconds=30,
    )
    job_root = tmp_path / "job-1"
    job_root.mkdir()
    (job_root / "status.json").write_text(
        json.dumps(
            {
                "status": "running",
                "raw_state": "RUNNING",
                "updated_at": (datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    state = backend.poll("local:job-1")

    assert state.status == "failed"
    assert state.raw_state == "STALE_HEARTBEAT"
    assert state.reason == "Local worker heartbeat became stale"


def test_local_cancel_refuses_to_signal_a_reused_pid(tmp_path, monkeypatch):
    backend = LocalComputeBackend(runtime_root=tmp_path)
    job_root = tmp_path / "job-1"
    job_root.mkdir()
    now = datetime.now(UTC).isoformat()
    (job_root / "status.json").write_text(
        json.dumps({"status": "running", "raw_state": "RUNNING", "updated_at": now}),
        encoding="utf-8",
    )
    (job_root / "process-identity.json").write_text(
        json.dumps(
            {
                "pid": 1234,
                "start_identity": "old-process-start",
                "launch_token": "launch-token",
                "created_at": now,
            }
        ),
        encoding="utf-8",
    )
    signalled = []
    monkeypatch.setattr(
        LocalComputeBackend,
        "_process_start_identity",
        staticmethod(lambda _pid: "reused-process-start"),
    )
    monkeypatch.setattr("al_medlit.training.compute.local.os.killpg", signalled.append)

    state = backend.cancel("local:job-1")

    assert state.status == "cancelled"
    assert signalled == []
    assert (job_root / "cancel-requested.json").is_file()


def test_local_worker_honors_shared_cancellation_marker(tmp_path):
    cancel_path = tmp_path / "cancel-requested.json"
    cancel_path.write_text("{}", encoding="utf-8")
    specification = {
        "schema_version": "local-process-v1",
        "command": [sys.executable, "-c", "import time; time.sleep(60)"],
        "cwd": str(tmp_path),
        "environment": {},
        "status_path": str(tmp_path / "status.json"),
        "stdout_path": str(tmp_path / "stdout.log"),
        "stderr_path": str(tmp_path / "stderr.log"),
        "identity_path": str(tmp_path / "process-identity.json"),
        "cancel_path": str(cancel_path),
        "launch_token": "test-launch-token",
        "heartbeat_interval_seconds": 1,
    }
    spec_path = tmp_path / "process.json"
    spec_path.write_text(json.dumps(specification), encoding="utf-8")

    run_local_worker(spec_path)

    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "cancelled"
    assert status["raw_state"] == "CANCELLED"
    assert status["launch_token"] == "test-launch-token"
    assert status["updated_at"]


def test_slurm_commands_enforce_batch_mode_and_strict_host_keys(tmp_path):
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), FakeRunner([]))
    bundle = JobBundle(
        job_key="job-1",
        command=("python", "-m", "trainer"),
        local_bundle_path=Path(tmp_path),
    )
    commands = backend.upload_commands(bundle)
    script = backend.render_sbatch_script(bundle)

    assert all("BatchMode=yes" in command for command in commands)
    assert all("StrictHostKeyChecking=yes" in command for command in commands)
    assert "#SBATCH --account=project-1" in script
    assert "module load apptainer/1.3.6" in script
    assert "apptainer exec --nv /images/al-medlit.sif python -m trainer" in script


def test_slurm_submit_poll_terminal_and_cancel_commands(tmp_path, monkeypatch):
    runner = FakeRunner(
        [
            _result(stdout="12345;cluster\n"),
            _result(stdout=""),
            _result(stdout="COMPLETED|0:0\n"),
        ]
    )
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), runner)
    monkeypatch.setattr(backend, "_upload_verified_bundle", lambda _bundle: None)
    bundle = JobBundle(
        job_key="job-1",
        command=("python", "train.py"),
        local_bundle_path=tmp_path,
    )
    submission = backend.submit(bundle)
    state = backend.poll(submission.external_job_id)

    assert submission.external_job_id == "12345"
    assert state.status == "succeeded"
    assert state.exit_code == 0
    assert any("sbatch" in command[-1] for command in runner.commands)
    assert any("sacct" in command[-1] for command in runner.commands)


def test_slurm_live_metric_reader_returns_only_complete_bounded_records(tmp_path):
    runner = FakeRunner(
        [
            _result(
                stdout=(
                    '{"phase":"train","values":{"loss":0.8}}\n'
                    '{"phase":"train","values":{"loss":0.6}}'
                )
            )
        ]
    )
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), runner)

    lines = backend.read_metric_lines(
        job_key="training-7-attempt-2",
        after_line=4,
        max_lines=20,
        max_bytes=4096,
    )

    assert lines == ('{"phase":"train","values":{"loss":0.8}}',)
    remote_command = shlex.split(runner.commands[0][-1])
    assert remote_command[:2] == ["sh", "-c"]
    assert "tail -n +5 --" in remote_command[2]
    assert "head -n 20" in remote_command[2]
    assert "head -c 4096" in remote_command[2]
    with pytest.raises(RuntimeError, match="Invalid metric line limit"):
        backend.read_metric_lines(job_key="training-7-attempt-2", max_lines=5001)


def test_slurm_parsers_reject_untrusted_job_ids_and_map_failures():
    with pytest.raises(RuntimeError, match="invalid Slurm job ID"):
        parse_sbatch_job_id("123$(touch-pwned)\n")
    failed = parse_sacct_state("OUT_OF_MEMORY|137:0\n")
    assert failed.status == "failed"
    assert failed.exit_code == 137


def test_quantization_capability_attestation_is_image_bound():
    digest = "a" * 64
    validated = validate_compute_config(
        "local",
        {
            "verified_capabilities": ["cuda", "qlora_4bit"],
            "worker_image_digest": digest,
            "capability_report_sha256": "b" * 64,
        },
    )
    assert validated["verified_capabilities"] == ["cuda", "qlora_4bit"]

    with pytest.raises(ValueError, match="also requires verified CUDA"):
        validate_compute_config(
            "local",
            {
                "verified_capabilities": ["qlora_4bit"],
                "worker_image_digest": digest,
                "capability_report_sha256": "b" * 64,
            },
        )
    with pytest.raises(ValueError, match="worker_image_digest"):
        validate_compute_config(
            "local",
            {
                "verified_capabilities": ["cuda"],
                "capability_report_sha256": "b" * 64,
            },
        )


def test_slurm_input_transfer_is_manifest_first_verified_and_base_cached(tmp_path):
    dataset = tmp_path / "training.jsonl"
    dataset.write_bytes(b"private project dataset")
    base_model = tmp_path / "base-model.zip"
    base_model.write_bytes(b"approved immutable base model")
    (tmp_path / "job.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
    job = {
        "schema_version": "al-medlit-job-v1",
        "kind": "training",
        "workspace_id": 9,
        "dataset": {
            "path": dataset.name,
            "checksum_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "size_bytes": dataset.stat().st_size,
        },
        "staged_base_model": {
            "path": base_model.name,
            "checksum_sha256": hashlib.sha256(base_model.read_bytes()).hexdigest(),
            "size_bytes": base_model.stat().st_size,
            "base_model_asset_id": 4,
        },
    }
    (tmp_path / "job.json").write_text(json.dumps(job), encoding="utf-8")
    runner = VirtualRemoteRunner()
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), runner)
    first = JobBundle(
        job_key="training-1-attempt-1",
        command=("python", "-m", "trainer"),
        local_bundle_path=tmp_path,
    )

    backend._upload_verified_bundle(first)

    first_scp = [command for command in runner.commands if command[0] == "scp"]
    assert Path(first_scp[0][-2]).name == ".al-medlit-transfer-manifest.json"
    cache_paths = [path for path in runner.remote_files if "/.base-model-cache/" in path]
    assert len(cache_paths) == 1
    assert "/workspaces/9/sha256/" in cache_paths[0]
    assert runner.remote_files[cache_paths[0]] == base_model.read_bytes()
    assert all(
        payload != dataset.read_bytes()
        for path, payload in runner.remote_files.items()
        if "/.base-model-cache/" in path
    )

    command_offset = len(runner.commands)
    second = JobBundle(
        job_key="training-2-attempt-1",
        command=("python", "-m", "trainer"),
        local_bundle_path=tmp_path,
    )
    backend._upload_verified_bundle(second)
    second_scp = [command for command in runner.commands[command_offset:] if command[0] == "scp"]
    assert str(base_model) not in {command[-2] for command in second_scp}
    assert str(dataset) in {command[-2] for command in second_scp}


def test_slurm_transfers_sharded_checkpoint_packages_without_an_archive(tmp_path):
    checkpoint_file = tmp_path / "checkpoint-package" / "model-00001-of-00002.safetensors"
    checkpoint_file.parent.mkdir()
    checkpoint_file.write_bytes(b"project checkpoint shard")
    base_file = tmp_path / "base-model-package" / "model.safetensors"
    base_file.parent.mkdir()
    base_file.write_bytes(b"approved base model shard")
    (tmp_path / "job.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
    def package_item(path):
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "relative_path": path.name,
            "checksum_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    (tmp_path / "job.json").write_text(
        json.dumps(
            {
                "schema_version": "al-medlit-job-v1",
                "kind": "inference",
                "workspace_id": 9,
                "checkpoint": {
                    "checksum_sha256": "a" * 64,
                    "package": {
                        "path": "checkpoint-package",
                        "files": [package_item(checkpoint_file)],
                    },
                },
                "base_model": {
                    "base_model_asset_id": 4,
                    "package": {
                        "path": "base-model-package",
                        "files": [package_item(base_file)],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    runner = VirtualRemoteRunner()
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), runner)

    backend._upload_verified_bundle(
        JobBundle(
            job_key="inference-8",
            command=("python", "-m", "trainer"),
            local_bundle_path=tmp_path,
        )
    )

    remote_checkpoint = "/scratch/al-medlit/inference-8/checkpoint-package/"
    assert any(path.startswith(remote_checkpoint) for path in runner.remote_files)
    assert not any(path.endswith("checkpoint.zip") for path in runner.remote_files)
    cache_payloads = [
        payload
        for path, payload in runner.remote_files.items()
        if "/.base-model-cache/" in path
    ]
    assert cache_payloads == [base_file.read_bytes()]


def test_slurm_remote_cleanup_is_scoped_to_validated_attempt_directory(tmp_path):
    runner = FakeRunner([_result()])
    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path), runner)

    backend.cleanup_job_directory("training-7-attempt-2")

    remote_command = shlex.split(runner.commands[0][-1])
    assert remote_command == [
        "rm",
        "-rf",
        "--",
        "/scratch/al-medlit/training-7-attempt-2",
    ]
    with pytest.raises(RuntimeError, match="Invalid job key"):
        backend.cleanup_job_directory("../other-job")
