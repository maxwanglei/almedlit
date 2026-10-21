import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("profile", "concurrency", "expected"),
    [
        ("qlora-cuda", None, 1),
        ("peft-accelerator", None, 1),
        ("classical-cpu", None, 1),
        ("torch-cpu", "3", 3),
    ],
)
def test_worker_cli_executes_celery_with_bounded_reservation(
    tmp_path, profile, concurrency, expected,
):
    environment, command = _worker_command(tmp_path, profile)
    if concurrency is not None:
        environment["AL_MEDLIT_WORKER_CONCURRENCY"] = concurrency
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=15)
    assert completed.returncode == 0, completed.stderr
    arguments = json.loads((tmp_path / "celery-argv.json").read_text())
    assert "--pool=prefork" in arguments
    assert f"--concurrency={expected}" in arguments
    assert "--prefetch-multiplier=1" in arguments
    assert f"--queues={profile}" in arguments


@pytest.mark.parametrize(
    ("profile", "concurrency"),
    [
        ("qlora-cuda", "2"),
        ("peft-accelerator", "2"),
        ("torch-cpu", "0"),
        ("torch-cpu", "invalid"),
    ],
)
def test_worker_cli_rejects_unsafe_or_invalid_concurrency(tmp_path, profile, concurrency):
    environment, command = _worker_command(tmp_path, profile)
    environment["AL_MEDLIT_WORKER_CONCURRENCY"] = concurrency
    completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=15)
    assert completed.returncode != 0
    assert "concurrency" in completed.stderr.lower()
    assert not (tmp_path / "celery-argv.json").exists()


def _worker_command(tmp_path, profile):
    backend = Path(__file__).parents[1]
    celery = tmp_path / "celery"
    celery.write_text(
        f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['WORKER_TEST_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
    )
    celery.chmod(0o755)
    runner = tmp_path / "launch.py"
    runner.write_text("""
from types import SimpleNamespace
from scripts import start_worker
start_worker.collect_runtime_readiness = lambda *a, **k: SimpleNamespace(ready=True)
start_worker.compute_profile_fragment = lambda report: {'ready': report.ready}
start_worker.main()
""")
    environment = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(backend),
        "AL_MEDLIT_RUNTIME_PROFILE": profile,
        "AL_MEDLIT_WORKER_QUEUE": profile,
        "AL_MEDLIT_RUNTIME_REPORT_PATH": str(tmp_path / "runtime.json"),
        "WORKER_TEST_ARGS": str(tmp_path / "celery-argv.json"),
    }
    environment.pop("AL_MEDLIT_WORKER_CONCURRENCY", None)
    return environment, [sys.executable, str(runner)]
