import fcntl
import os
import signal
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from test_training_compute import _slurm_config

from al_medlit.training.compute.base import (
    CommandResult,
    ComputeBackendError,
    SubprocessCommandRunner,
)
from al_medlit.training.compute.config import SSHSlurmProfileConfig
from al_medlit.training.compute.slurm import SSHSlurmComputeBackend


def test_timed_command_terminates_its_descendants_and_releases_resource(tmp_path):
    resource = tmp_path / "transport.lock"
    probe = tmp_path / "probe.py"
    probe.write_text("""
import fcntl, os, pathlib, signal, sys, time
from al_medlit.training.compute.base import ComputeBackendError, SubprocessCommandRunner
command = '''
import fcntl, os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
handle = open(sys.argv[1], 'w')
fcntl.flock(handle, fcntl.LOCK_EX)
if os.fork() == 0:
    pathlib.Path(sys.argv[1] + '.ready').touch()
while True:
    time.sleep(0.02)
'''
try:
    SubprocessCommandRunner().run(
        (sys.executable, '-c', command, sys.argv[1]), timeout_seconds=1,
    )
except ComputeBackendError as exc:
    print(str(exc), flush=True)
else:
    raise AssertionError('The transport deadline was ignored')
""")
    process = subprocess.Popen(
        [sys.executable, str(probe), str(resource)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    try:
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
        assert "timed out after 1" in stdout
        assert Path(str(resource) + ".ready").exists()
        with resource.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)


def test_subprocess_runner_preserves_io_and_exit_status_with_deadline():
    result = SubprocessCommandRunner().run(
        (
            sys.executable,
            "-c",
            "import os,sys; print(os.environ['TRANSPORT_TEST']); "
            "print(sys.stdin.read(), file=sys.stderr); sys.exit(7)",
        ),
        environment={"TRANSPORT_TEST": "connected"},
        input_text="payload",
        timeout_seconds=5,
    )
    assert result.returncode == 7
    assert result.stdout == "connected\n"
    assert result.stderr == "payload\n"


def test_slurm_passes_deadlines_to_injected_runner_for_commands_and_transfers(tmp_path):
    calls = []

    class Runner:
        def run(self, argv, *, timeout_seconds):
            calls.append((argv, timeout_seconds))
            return CommandResult(argv, 0, "RUNNING\n", "")

    backend = SSHSlurmComputeBackend(
        replace(_slurm_config(tmp_path), command_timeout_seconds=3, transfer_timeout_seconds=7),
        runner=Runner(),
    )
    backend.poll("123")
    with pytest.raises(ComputeBackendError, match="regular file"):
        backend._download_remote_file(
            job_key="job-1",
            remote_relative_path="outputs/model.bin",
            destination=tmp_path / "model",
        )
    assert [(argv[0], timeout) for argv, timeout in calls] == [("ssh", 3), ("scp", 7)]


def test_every_slurm_transport_has_connection_and_liveness_bounds(tmp_path):
    from al_medlit.training.compute.base import JobBundle

    backend = SSHSlurmComputeBackend(_slurm_config(tmp_path))
    commands = (
        backend.ssh_argv(("squeue",)),
        *backend.upload_commands(JobBundle("job-1", ("python",), tmp_path)),
        backend.artifact_download_command(
            job_key="job-1",
            remote_relative_path="outputs/model.bin",
            local_destination=tmp_path / "model",
        ),
    )
    for argv in commands:
        assert "ConnectTimeout=10" in argv
        assert "ConnectionAttempts=1" in argv
        assert "ServerAliveInterval=15" in argv
        assert "ServerAliveCountMax=3" in argv


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_slurm_rejects_nonfinite_or_nonpositive_command_deadline(tmp_path, value):
    with pytest.raises(ComputeBackendError, match="positive finite"):
        replace(_slurm_config(tmp_path), command_timeout_seconds=value)


def test_profile_preserves_configured_transport_deadlines(tmp_path):
    config = _slurm_config(tmp_path)
    from dataclasses import asdict

    runtime = SSHSlurmProfileConfig.model_validate(
        {
            **asdict(config),
            "command_timeout_seconds": 12,
            "transfer_timeout_seconds": 24,
        }
    ).to_runtime_config()
    assert runtime.command_timeout_seconds == 12
    assert runtime.transfer_timeout_seconds == 24


def test_remote_inference_timeout_rolls_back_and_can_finalize_on_retry(
    db,
    object_storage,
    tmp_path,
    monkeypatch,
):
    from test_inference_candidate_queries import _inference_scope
    from test_inference_execution import (
        DECODER_CONFIG,
        ONE_HOT,
        _finalize,
        _runner_result,
        _sentence_scorer_checkpoint,
    )

    from al_medlit.core.exceptions import ConflictError
    from al_medlit.inference import execution

    threshold = 0.5
    scope = _inference_scope(
        db,
        checkpoint_manifest={"model_type": "evidence_svm"},
        checkpoint_object=_sentence_scorer_checkpoint(
            tmp_path, object_storage, model_type="evidence_svm", threshold=threshold
        ),
        decoder_config=DECODER_CONFIG,
        run_status="running",
        with_window=False,
    )
    scope.run.metrics = {"remote_result_ready": True}
    db.commit()
    run_id = scope.run.id
    ssh = tmp_path / "ssh"
    ssh.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
    ssh.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    backend = SSHSlurmComputeBackend(
        replace(_slurm_config(tmp_path), command_timeout_seconds=0.25),
    )

    with pytest.raises(ConflictError, match="ssh timed out after 0.25 seconds"):
        execution.finalize_remote_inference(
            db,
            object_storage,
            run_id=run_id,
            backend=backend,
            work_root=tmp_path / "work",
        )
    assert not db.in_transaction()
    assert scope.run.status == "running"
    assert scope.run.diagnostics_artifact_id is None
    inference_result = _runner_result(
        scope,
        {"window-0": {ordinal: ONE_HOT["O"] for ordinal in range(4)}},
    )
    inference_result["sentence_score_threshold"] = threshold
    run = _finalize(db, object_storage, tmp_path, scope, inference_result)
    assert run.status == "succeeded"
    assert run.diagnostics_artifact_id is not None


def test_timed_command_returns_when_detached_descendant_holds_output_pipes(tmp_path):
    escaped_pid = tmp_path / "escaped.pid"
    probe = tmp_path / "detached-probe.py"
    probe.write_text("""
import sys
from al_medlit.training.compute.base import ComputeBackendError, SubprocessCommandRunner
command = '''
import os, pathlib, sys, time
if os.fork() == 0:
    os.setsid()
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(30)
'''
try:
    SubprocessCommandRunner().run(
        (sys.executable, '-c', command, sys.argv[1]), timeout_seconds=1,
    )
except ComputeBackendError as exc:
    print(str(exc), flush=True)
else:
    raise AssertionError('The transport deadline was ignored')
""")
    process = subprocess.Popen(
        [sys.executable, str(probe), str(escaped_pid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])},
    )
    try:
        stdout, stderr = process.communicate(timeout=6)
        assert process.returncode == 0, stderr
        assert "timed out after 1" in stdout
        assert escaped_pid.exists(), "The detached pipe holder must actually run"
    finally:
        if escaped_pid.exists():
            try:
                os.kill(int(escaped_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate(timeout=5)
