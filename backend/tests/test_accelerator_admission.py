import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from al_medlit.training.supervision import _terminate_process_group, _watch_process

COMPUTE_ENTRY = """
import contextlib, os, pathlib, sys, time
from al_medlit.core import database, storage
from al_medlit.training import run_execution, supervision
database.register_models = lambda: None
database.SessionLocal = lambda: contextlib.nullcontext(None)
storage.get_object_storage = lambda: None
def execute(*args, **kwargs):
    if os.environ.get('ADMISSION_TEST_RETURN') == '1':
        pathlib.Path(sys.argv[3]).touch()
        return
    if os.fork() == 0:
        pathlib.Path(sys.argv[3] + '.child').touch()
    pathlib.Path(sys.argv[3]).touch()
    while True:
        time.sleep(0.02)
run_execution.execute_training_run = execute
pathlib.Path(sys.argv[3] + '.waiting').touch()
supervision.main()
if os.environ.get('ADMISSION_TEST_RETURN') == '1':
    pathlib.Path(sys.argv[3] + '.teardown').touch()
    while True:
        time.sleep(0.02)
"""


def _launch(tmp_path, name, profile, *, return_from_compute=False):
    marker = tmp_path / name
    return subprocess.Popen(
        [sys.executable, "-c", COMPUTE_ENTRY, "compute", "1", str(marker)],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).parents[1]),
            "AL_MEDLIT_RUNTIME_PROFILE": profile,
            "AL_MEDLIT_LOCAL_ATTEMPT_ROOT": str(tmp_path / "shared-attempts"),
            "ADMISSION_TEST_RETURN": "1" if return_from_compute else "0",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ), marker


def _wait(path, process):
    deadline = time.monotonic() + 20
    while not path.exists():
        if process.poll() is not None:
            raise AssertionError(process.communicate(timeout=5))
        assert time.monotonic() < deadline, f"Process did not reach {path.name}"
        time.sleep(0.02)


def _stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    process.communicate(timeout=5)


def _assert_waiting(marker, process):
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        assert not marker.exists(), "Both accelerator workers entered compute together"
        assert process.poll() is None
        time.sleep(0.02)


def test_different_accelerator_workers_share_one_compute_admission(tmp_path):
    first, first_marker = _launch(tmp_path, "peft-entered", "peft-accelerator")
    second = None
    try:
        _wait(first_marker, first)
        second, second_marker = _launch(tmp_path, "qlora-entered", "qlora-cuda")
        _wait(Path(str(second_marker) + ".waiting"), second)
        _assert_waiting(second_marker, second)
        _stop(first)
        _wait(second_marker, second)
    finally:
        _stop(first)
        if second is not None:
            _stop(second)


def test_accelerator_admission_remains_held_during_interpreter_teardown(tmp_path):
    first, first_marker = _launch(
        tmp_path, "peft-finished", "peft-accelerator", return_from_compute=True,
    )
    second = None
    try:
        _wait(Path(str(first_marker) + ".teardown"), first)
        second, second_marker = _launch(tmp_path, "qlora-entered", "qlora-cuda")
        _wait(Path(str(second_marker) + ".waiting"), second)
        _assert_waiting(second_marker, second)
        _stop(first)
        _wait(second_marker, second)
    finally:
        _stop(first)
        if second is not None:
            _stop(second)


@pytest.mark.parametrize(
    ("stop", "reason"),
    [
        ("cancel", "worker_lease_lost"),
        ("deadline", "execution_deadline_exceeded"),
        ("parent_eof", "task_supervisor_lost"),
    ],
)
def test_waiting_accelerator_compute_stops_without_disturbing_holder(tmp_path, stop, reason):
    holder, holder_marker = _launch(tmp_path, "holder-entered", "peft-accelerator")
    waiter = None
    read_fd, write_fd = os.pipe()
    try:
        _wait(holder_marker, holder)
        waiter, waiter_marker = _launch(tmp_path, "waiter-entered", "qlora-cuda")
        _wait(Path(str(waiter_marker) + ".waiting"), waiter)
        _assert_waiting(waiter_marker, waiter)
        if stop == "parent_eof":
            os.close(write_fd)
            write_fd = None
        outcome = _watch_process(
            waiter,
            parent_fd=read_fd,
            heartbeat=lambda: stop != "cancel",
            max_runtime_seconds=0.2 if stop == "deadline" else 5,
            poll_seconds=0.02,
            grace_seconds=0.1,
        )
        assert outcome.reason == reason
        assert waiter.poll() is not None
        assert holder.poll() is None
        assert not waiter_marker.exists()
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        _stop(holder)
        if waiter is not None:
            _stop(waiter)


def test_cpu_workers_can_enter_compute_concurrently(tmp_path):
    first, first_marker = _launch(tmp_path, "first-cpu", "torch-cpu")
    second = None
    try:
        _wait(first_marker, first)
        second, second_marker = _launch(tmp_path, "second-cpu", "classical-cpu")
        _wait(second_marker, second)
        assert first.poll() is None
        assert second.poll() is None
    finally:
        _stop(first)
        if second is not None:
            _stop(second)


def test_process_group_cleanup_continues_after_denied_liveness_probe(monkeypatch, tmp_path):
    process, marker = _launch(tmp_path, "compute-entered", "peft-accelerator")
    original_killpg = os.killpg
    attempted = []

    def killpg(pid, signum):
        attempted.append(signum)
        if signum == 0:
            raise PermissionError("liveness probes are denied")
        return original_killpg(pid, signum)

    try:
        _wait(marker, process)
        monkeypatch.setattr(os, "killpg", killpg)
        _terminate_process_group(process, 0.1)
        assert process.poll() is not None
        assert signal.SIGKILL in attempted
    finally:
        monkeypatch.setattr(os, "killpg", original_killpg)
        _stop(process)


@pytest.mark.parametrize("denied", [signal.SIGTERM, signal.SIGKILL])
def test_process_group_cleanup_propagates_denied_termination(monkeypatch, denied):
    process = subprocess.Popen(
        [
            sys.executable, "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(30)",
        ],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    original_killpg = os.killpg

    def killpg(pid, signum):
        if signum in {0, denied}:
            raise PermissionError("termination is denied")
        return original_killpg(pid, signum)

    try:
        assert process.stdout.readline() == "ready\n"
        monkeypatch.setattr(os, "killpg", killpg)
        with pytest.raises(PermissionError, match="termination is denied"):
            _terminate_process_group(process, 0.1)
        assert process.poll() is None
    finally:
        monkeypatch.setattr(os, "killpg", original_killpg)
        _stop(process)


def test_watchdog_reaps_compute_that_exits_immediately_on_sigterm():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
    )
    read_fd, write_fd = os.pipe()
    try:
        outcome = _watch_process(
            process,
            parent_fd=read_fd,
            heartbeat=lambda: True,
            max_runtime_seconds=0.2,
            poll_seconds=0.02,
            grace_seconds=0.1,
        )
        assert outcome.reason == "execution_deadline_exceeded"
        assert process.poll() is not None
    finally:
        os.close(read_fd)
        os.close(write_fd)
        _stop(process)
