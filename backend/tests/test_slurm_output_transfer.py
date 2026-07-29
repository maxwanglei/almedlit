import hashlib
import json
from pathlib import Path, PurePosixPath

import pytest

from al_medlit.training.compute.base import CommandResult, ComputeBackendError
from al_medlit.training.compute.slurm import (
    OutputTransferLimits,
    SlurmResources,
    SSHSlurmComputeBackend,
    SSHSlurmConfig,
    validate_output_manifest,
    validate_remote_relative_path,
)


def _result(argv, *, stdout="", stderr="", returncode=0):
    return CommandResult(
        argv=tuple(argv),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


class TransferRunner:
    def __init__(self, responses):
        self.responses = list(responses)
        self.commands = []

    def run(self, argv, **_kwargs):
        self.commands.append(argv)
        kind, value = self.responses.pop(0)
        if kind == "stat":
            assert argv[0] == "ssh"
            return _result(argv, stdout=f"{value}\n")
        if kind == "download":
            assert argv[0] == "scp"
            Path(argv[-1]).write_bytes(value)
            return _result(argv)
        if kind == "error":
            return _result(argv, returncode=1, stderr=str(value))
        raise AssertionError(f"Unknown fake response {kind!r}")


def _backend(tmp_path, runner):
    return SSHSlurmComputeBackend(
        SSHSlurmConfig(
            host="cluster.example.org",
            username="researcher",
            known_hosts_file=tmp_path / "known_hosts",
            remote_root=PurePosixPath("/scratch/al-medlit"),
            resources=SlurmResources(account="project-1", partition="gpu"),
            apptainer_image=PurePosixPath("/images/al-medlit.sif"),
            apptainer_module="apptainer/1.3.6",
        ),
        runner,
    )


def _sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def _manifest(files):
    return json.dumps(
        {
            "schema_version": "al-medlit-job-result-v1",
            "kind": "training",
            "status": "succeeded",
            "files": files,
        },
        separators=(",", ":"),
    ).encode()


def test_collect_outputs_downloads_manifest_first_and_verifies_all_payloads(tmp_path):
    metrics = b'{"loss":0.2}'
    weights = b"safe model weights"
    manifest = _manifest(
        {
            "metrics.json": {
                "size_bytes": len(metrics),
                "checksum_sha256": _sha256(metrics),
            },
            "checkpoint/model.safetensors": {
                "size_bytes": len(weights),
                "checksum_sha256": _sha256(weights),
            },
        }
    )
    runner = TransferRunner(
        [
            ("stat", len(manifest)),
            ("download", manifest),
            ("stat", len(metrics)),
            ("download", metrics),
            ("stat", len(weights)),
            ("download", weights),
        ]
    )
    backend = _backend(tmp_path, runner)
    output_root = tmp_path / "outputs"

    result = backend.collect_outputs(job_key="training-7-attempt-1", output_root=output_root)

    assert result["status"] == "succeeded"
    assert (output_root / "metrics.json").read_bytes() == metrics
    assert (output_root / "checkpoint/model.safetensors").read_bytes() == weights
    assert (output_root / "artifact-manifest.json").read_bytes() == manifest
    assert runner.commands[0][0] == "ssh"
    assert "outputs/artifact-manifest.json" in runner.commands[0][-1]
    assert "outputs/artifact-manifest.json" in runner.commands[1][-2]
    assert all(
        "StrictHostKeyChecking=yes" in command
        for command in runner.commands
        if command[0] in {"ssh", "scp"}
    )


def test_collect_outputs_rejects_declared_limits_before_payload_download(tmp_path):
    payload = b"oversized"
    manifest = _manifest(
        {
            "checkpoint/model.safetensors": {
                "size_bytes": len(payload),
                "checksum_sha256": _sha256(payload),
            }
        }
    )
    runner = TransferRunner([("stat", len(manifest)), ("download", manifest)])
    backend = _backend(tmp_path, runner)

    with pytest.raises(ComputeBackendError, match="per-file byte limit"):
        backend.collect_outputs(
            job_key="training-7-attempt-1",
            output_root=tmp_path / "outputs",
            limits=OutputTransferLimits(max_file_bytes=len(payload) - 1),
        )

    assert len(runner.commands) == 2
    assert not (tmp_path / "outputs/artifact-manifest.json").exists()


def test_collect_outputs_bounds_manifest_before_downloading_it(tmp_path):
    runner = TransferRunner([("stat", 2048)])
    backend = _backend(tmp_path, runner)

    with pytest.raises(ComputeBackendError, match="manifest exceeds"):
        backend.collect_outputs(
            job_key="training-7-attempt-1",
            output_root=tmp_path / "outputs",
            limits=OutputTransferLimits(max_manifest_bytes=1024),
        )

    assert len(runner.commands) == 1
    assert runner.commands[0][0] == "ssh"


def test_collect_outputs_does_not_publish_checksum_mismatches(tmp_path):
    expected = b"expected"
    manifest = _manifest(
        {
            "metrics.json": {
                "size_bytes": len(expected),
                "checksum_sha256": _sha256(expected),
            }
        }
    )
    runner = TransferRunner(
        [
            ("stat", len(manifest)),
            ("download", manifest),
            ("stat", len(expected)),
            ("download", b"tampered"),
        ]
    )
    backend = _backend(tmp_path, runner)
    output_root = tmp_path / "outputs"

    with pytest.raises(ComputeBackendError, match="checksum mismatch"):
        backend.collect_outputs(job_key="training-7-attempt-1", output_root=output_root)

    assert not (output_root / "metrics.json").exists()
    assert not (output_root / "artifact-manifest.json").exists()


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "../secret",
        "/absolute/file",
        "checkpoint//model.safetensors",
        "checkpoint/./model.safetensors",
        "checkpoint/model name.safetensors",
        "checkpoint/model$(id).safetensors",
        "checkpoint\\model.safetensors",
    ),
)
def test_remote_artifact_paths_are_canonical_and_shell_safe(unsafe_path):
    with pytest.raises(ComputeBackendError, match="Artifact path"):
        validate_remote_relative_path(unsafe_path)


def test_manifest_requires_checksums_and_bounds_file_count_and_total_bytes():
    checksum = "0" * 64
    with pytest.raises(ComputeBackendError, match="SHA-256"):
        validate_output_manifest(
            {"files": {"metrics.json": {"size_bytes": 1}}}
        )
    with pytest.raises(ComputeBackendError, match="file-count"):
        validate_output_manifest(
            {
                "files": {
                    "one.bin": {"size_bytes": 1, "checksum_sha256": checksum},
                    "two.bin": {"size_bytes": 1, "checksum_sha256": checksum},
                }
            },
            OutputTransferLimits(max_files=1),
        )
    with pytest.raises(ComputeBackendError, match="total byte limit"):
        validate_output_manifest(
            {
                "files": {
                    "one.bin": {"size_bytes": 6, "checksum_sha256": checksum},
                    "two.bin": {"size_bytes": 5, "checksum_sha256": checksum},
                }
            },
            OutputTransferLimits(max_file_bytes=10, max_total_bytes=10),
        )


def test_artifact_download_command_rejects_remote_shell_metacharacters(tmp_path):
    backend = _backend(tmp_path, TransferRunner([]))

    with pytest.raises(ComputeBackendError, match="unsafe segment"):
        backend.artifact_download_command(
            job_key="training-7-attempt-1",
            remote_relative_path="outputs/result;touch-pwned",
            local_destination=tmp_path / "result",
        )
