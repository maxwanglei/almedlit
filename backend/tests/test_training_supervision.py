import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from al_medlit.training.supervision import _watch_process

COMPUTE = """
import fcntl, os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
handle = open(sys.argv[1], 'w')
fcntl.flock(handle, fcntl.LOCK_EX)
if os.fork() == 0:
    pathlib.Path(sys.argv[1] + '.ready').touch()
while True:
    time.sleep(0.05)
"""


def wait_ready(path, process):
    deadline = time.monotonic() + 15
    while not path.exists():
        assert process.poll() is None, "Compute process exited before acquiring its resource"
        assert time.monotonic() < deadline, "Compute process did not acquire its resource"
        time.sleep(0.02)


def assert_resource_released(path):
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        ("cancel", "worker_lease_lost"),
        ("deadline", "execution_deadline_exceeded"),
        ("parent_eof", "task_supervisor_lost"),
        ("database_stall", "worker_lease_check_timed_out"),
    ],
)
def test_watchdog_releases_child_and_grandchild_resource(tmp_path, stop, expected):
    resource = tmp_path / "device.lock"
    read_fd, write_fd = os.pipe()
    release_database = threading.Event()
    process = subprocess.Popen(
        [sys.executable, "-c", COMPUTE, str(resource)],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_ready(Path(str(resource) + ".ready"), process)
        if stop == "parent_eof":
            os.close(write_fd)
            write_fd = None

        def heartbeat():
            if stop == "database_stall":
                release_database.wait(10)
            return stop != "cancel"

        outcome = _watch_process(
            process,
            parent_fd=read_fd,
            heartbeat=heartbeat,
            max_runtime_seconds=0.3 if stop == "deadline" else 5,
            poll_seconds=0.02,
            grace_seconds=0.1,
            database_timeout_seconds=0.2,
        )
        assert outcome.reason == expected
        assert process.poll() is not None
        assert_resource_released(resource)
    finally:
        release_database.set()
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def test_task_cli_closes_compute_on_cancellation_and_process_shutdown(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from test_training_run_execution import MemoryObjectStorage, _training_run_fixture

    from al_medlit.core.database import Base, register_models
    from al_medlit.workflow import schemas, service

    backend = Path(__file__).parents[1]
    database_url = f"sqlite:///{tmp_path / 'training.db'}"
    register_models()
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    environment = {
        **os.environ,
        "AL_MEDLIT_DATABASE_URL": database_url,
        "AL_MEDLIT_WORKER_IMAGE_DIGEST": "a" * 64,
        "AL_MEDLIT_TRAINING_SUPERVISOR_POLL_SECONDS": "0.1",
        "AL_MEDLIT_TRAINING_TERMINATION_GRACE_SECONDS": "0.1",
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(backend), str(backend / "tests")]),
    }
    (tmp_path / "sitecustomize.py").write_text("""
import fcntl, os, pathlib, signal, time
from test_training_run_execution import FakeTrainer, MemoryObjectStorage
from al_medlit.training import trainers
from al_medlit.training.trainers.contracts import TrainerPluginRegistry
from al_medlit.core import storage
class BlockingTrainer(FakeTrainer):
    def train(self, **kwargs):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        with open(os.environ['TRAINING_TEST_RESOURCE'], 'w') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            identity = pathlib.Path(os.environ['TRAINING_TEST_RESOURCE'] + '.watchdog')
            identity.write_text(str(os.getppid()))
            if os.fork() == 0:
                pathlib.Path(os.environ['TRAINING_TEST_RESOURCE'] + '.ready').touch()
            while True:
                time.sleep(0.05)
registry = TrainerPluginRegistry()
registry.register(BlockingTrainer(
    key='sklearn_tfidf', recipe_key='tfidf_logistic_regression',
    runtime_class='classical-cpu', output_name='model.skops'))
trainers.trainer_plugins = registry
storage.get_object_storage = MemoryObjectStorage
""")
    try:
        for stop in ("cancel", "parent_death", "watchdog_sigterm"):
            with Session(engine, autoflush=False) as db:
                scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix=stop)
                run_id, project_id = scope.run.id, scope.project.id
            resource = tmp_path / f"{stop}.lock"
            output = tmp_path / f"{stop}.log"
            with output.open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "from al_medlit.training.tasks import execute_training_run_task; "
                        f"execute_training_run_task({run_id})",
                    ],
                    env={**environment, "TRAINING_TEST_RESOURCE": str(resource)},
                    cwd=backend,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    wait_ready(Path(str(resource) + ".ready"), process)
                    if stop == "cancel":
                        with Session(engine, autoflush=False) as db:
                            service.transition_training_run(
                                db,
                                project_id,
                                run_id,
                                schemas.TrainingRunTransition(status="cancelled"),
                            )
                        process.wait(timeout=15)
                    elif stop == "watchdog_sigterm":
                        watchdog_pid = int(Path(str(resource) + ".watchdog").read_text())
                        os.kill(watchdog_pid, signal.SIGTERM)
                        process.wait(timeout=15)
                    else:
                        process.kill()
                        process.wait(timeout=5)
                    deadline = time.monotonic() + 15
                    while True:
                        try:
                            assert_resource_released(resource)
                            break
                        except BlockingIOError:
                            assert time.monotonic() < deadline, output.read_text()
                            time.sleep(0.05)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)
    finally:
        engine.dispose()


def test_transient_lease_error_retries_without_killing_compute(tmp_path):
    resource = tmp_path / "transient.lock"
    read_fd, write_fd = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-c", COMPUTE, str(resource)],
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    calls = []

    def heartbeat():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("lock timeout during candidate upload")
        return len(calls) == 2

    try:
        wait_ready(Path(str(resource) + ".ready"), process)
        outcome = _watch_process(
            process,
            parent_fd=read_fd,
            heartbeat=heartbeat,
            max_runtime_seconds=5,
            poll_seconds=0.02,
            grace_seconds=0.1,
        )
        assert len(calls) == 3
        assert outcome.reason == "worker_lease_lost"
        assert_resource_released(resource)
    finally:
        os.close(read_fd)
        os.close(write_fd)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
