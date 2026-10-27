import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from al_medlit.auth.security import verify_password
from al_medlit.core.config import settings
from al_medlit.core.storage import ObjectStorageError
from al_medlit.project.models import Project
from al_medlit.project.schemas import ProjectCreate
from al_medlit.project.service import create_project
from al_medlit.storage_reclaim.models import OrphanedStorageObject
from al_medlit.storage_reclaim.service import record_orphaned_object
from al_medlit.workspace.models import Workspace
from al_medlit.workspace.service import ensure_default_workspace
from scripts import runtime_preflight, seed_demo, start_worker
from scripts.bootstrap_admin import bootstrap_admin

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("error_code", [
    "XMinioPolicyAlreadyAttached", "XMinioAdminPolicyChangeAlreadyApplied",
])
def test_minio_provisioning_can_attach_an_existing_policy(tmp_path, error_code):
    result = _provision(tmp_path, error_code)
    assert result.returncode == 0, result.stderr


def test_minio_provisioning_still_fails_for_other_errors(tmp_path):
    result = _provision(tmp_path, "AccessDenied")
    assert result.returncode == 1
    assert "AccessDenied" in result.stderr


def _provision(tmp_path, error_code, root_password="root-secret-with-many-characters"):
    mc = tmp_path / "mc"
    mc.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '  *"admin policy attach"*) printf \'%s\\n\' \'{"error":{"Code":"'
        + error_code + '"}}\'; exit 1 ;;\n'
        "esac\n"
    )
    mc.chmod(0o755)
    ca = tmp_path / "ca.crt"
    ca.write_text("test CA")
    return subprocess.run(
        ["sh", str(ROOT / "infra/scripts/provision-minio-app.sh")],
        env=os.environ | {
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "MC_CONFIG_DIR": str(tmp_path / "mc-config"),
            "MINIO_CA_CERT_PATH": str(ca),
            "MINIO_ROOT_USER": "root", "MINIO_ROOT_PASSWORD": root_password,
            "AL_MEDLIT_STORAGE_ACCESS_KEY": "app",
            "AL_MEDLIT_STORAGE_SECRET_KEY": "app-secret-with-many-characters",
        },
        capture_output=True, text=True, check=False,
    )


def test_minio_provisioning_rejects_weak_root_credentials(tmp_path):
    result = _provision(tmp_path, "XMinioPolicyAlreadyAttached", root_password="minioadmin")
    assert result.returncode == 1
    assert "MINIO_ROOT_PASSWORD must contain" in result.stderr


def test_certificate_reuse_rejects_a_mismatched_operator_key(tmp_path):
    script = ROOT / "infra/scripts/generate-minio-certs.sh"
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        subprocess.run(["sh", str(script), str(directory)], check=True)
    (first / "private.key").write_bytes((second / "private.key").read_bytes())
    (first / ".al-medlit-generated").unlink()
    certificate = (first / "public.crt").read_bytes()
    result = subprocess.run(["sh", str(script), str(first)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "rotate them manually" in result.stderr
    assert (first / "public.crt").read_bytes() == certificate
    assert (second / "private.key").stat().st_mode & 0o777 == 0o600
    assert second.stat().st_mode & 0o777 == 0o755
    assert (second / "CAs").stat().st_mode & 0o777 == 0o755


def test_runtime_launcher_uses_the_built_image_digest(tmp_path):
    docker = tmp_path / "docker"
    log = tmp_path / "docker.log"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['DOCKER_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps([sys.argv[1:], "
        "os.environ.get('AL_MEDLIT_CLASSICAL_CPU_IMAGE_DIGEST')]) + '\\n')\n"
        "if sys.argv[1:3] == ['image', 'inspect']:\n"
        "    print('sha256:' + 'a' * 64)\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["sh", str(ROOT / "infra/scripts/start-runtime.sh"), "classical-cpu", "-f", "stack.yml"],
        env=os.environ | {
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "DOCKER_TEST_LOG": str(log),
        }, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][0][-1] == "build"
    assert calls[1][0][-1] == "al-medlit-worker-classical-cpu:local"
    assert calls[2][0][-2:] == ["up", "-d"]
    assert calls[2][1] == "sha256:" + "a" * 64


def test_runtime_probe_records_failed_cleanup(db, testing_session_factory, monkeypatch):
    class Storage:
        def put_bytes(self, *_args, **_kwargs):
            pass

        def get_bytes(self, _key):
            return b"probe"

        def delete(self, _key):
            raise ObjectStorageError("storage temporarily unavailable")

    monkeypatch.setattr(runtime_preflight, "SessionLocal", testing_session_factory)
    monkeypatch.setattr(runtime_preflight, "get_object_storage", Storage)
    with pytest.raises(ObjectStorageError):
        runtime_preflight._probe_storage()
    orphan = db.query(OrphanedStorageObject).one()
    assert orphan.storage_key.startswith("runtime-preflight/")
    assert orphan.origin == "runtime.preflight"


def test_orphan_queue_insert_remains_part_of_the_callers_transaction(db):
    assert db.query(OrphanedStorageObject).count() == 0
    record_orphaned_object(db, "uncommitted-object", origin="audit.rollback", commit=False)
    db.rollback()
    assert db.query(OrphanedStorageObject).count() == 0


def test_bootstrap_password_matches_the_exact_configured_value(db, monkeypatch):
    password = "  strong-configured-password  "
    monkeypatch.setattr(settings, "bootstrap_admin_username", "exact-password-admin")
    monkeypatch.setattr(settings, "bootstrap_admin_password", password)
    user = bootstrap_admin(db)
    assert verify_password(password, user.password_hash)
    assert not verify_password(password.strip(), user.password_hash)


def test_worker_runs_security_checks_before_runtime_probes(monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_RUNTIME_PROFILE", "classical-cpu")
    monkeypatch.setenv("AL_MEDLIT_WORKER_QUEUE", "classical-cpu")

    def reject():
        raise RuntimeError("invalid deployment secrets")

    def unexpected_probe(*_args, **_kwargs):
        pytest.fail("The worker probed the runtime before validating startup")

    monkeypatch.setattr(start_worker, "_run_security_checks", reject)
    monkeypatch.setattr(start_worker, "collect_runtime_readiness", unexpected_probe)
    with pytest.raises(RuntimeError, match="invalid deployment secrets"):
        start_worker.main()


def test_sqlite_application_connections_enforce_foreign_keys(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "from al_medlit.core.database import engine; "
         "print(engine.connect().exec_driver_sql('PRAGMA foreign_keys').scalar())"],
        env=os.environ | {"AL_MEDLIT_DATABASE_URL": f"sqlite:///{tmp_path / 'app.db'}"},
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "1"


def test_seed_failure_rolls_back_service_commits(db, monkeypatch):
    monkeypatch.setattr(seed_demo, "engine", db.get_bind())
    monkeypatch.setattr(seed_demo, "ensure_schema_ready", lambda: None)

    def fail_after_project_commit(session):
        create_project(session, ProjectCreate(name="failed atomic seed"))
        raise RuntimeError("seed failed")

    monkeypatch.setattr(seed_demo, "_seed_demo", fail_after_project_commit)
    with pytest.raises(RuntimeError, match="seed failed"):
        seed_demo.main()
    assert db.query(Project).filter(Project.name == "failed atomic seed").count() == 0


def test_seed_lookup_is_scoped_to_the_default_workspace(db):
    other = Workspace(name="Unrelated tenant", kind="team")
    db.add(other)
    db.flush()
    db.add(Project(name=seed_demo.DEMO_PROJECT_NAME, workspace_id=other.id))
    db.commit()
    assert seed_demo._find_existing_demo_project(db) is None
    default = ensure_default_workspace(db)
    expected = Project(name=seed_demo.DEMO_PROJECT_NAME, workspace_id=default.id)
    db.add(expected)
    db.commit()
    assert seed_demo._find_existing_demo_project(db).id == expected.id


@pytest.mark.parametrize("path", [
    "/api/auth/login", "/api/auth/login/", "/api/invites/secret/",
    "/api/invites/secret/accept/", "/api/account-actions/secret/",
    "/api/workspaces/by-code/secret/join-requests/",
])
def test_nginx_throttles_and_redacts_token_routes_with_optional_slashes(path):
    config = (ROOT / "frontend/nginx/default.conf.template").read_text()
    pattern = re.search(r"location ~ (\^/api/\(\?:auth/[^\n]+) \{", config).group(1)
    assert re.fullmatch(pattern, path)
