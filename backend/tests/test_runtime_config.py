import re
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from al_medlit.core.config import (
    DEFAULT_BOOTSTRAP_ADMIN_PASSWORD,
    DEFAULT_JWT_SECRET,
    settings,
)
from al_medlit.main import create_app

ROOT_DIR = Path(__file__).resolve().parents[2]
REQUIRED_COMPOSE_JWT_LINE = (
    "AL_MEDLIT_JWT_SECRET: ${AL_MEDLIT_JWT_SECRET:?AL_MEDLIT_JWT_SECRET required}"
)
BOOTSTRAP_ADMIN_PASSWORD_LINE = (
    "AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD: ${AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD:-}"
)
ALLOW_SELF_REGISTRATION_LINE = (
    "AL_MEDLIT_ALLOW_SELF_REGISTRATION: ${AL_MEDLIT_ALLOW_SELF_REGISTRATION:-true}"
)
DEPLOYMENT_PROFILE_LINE = "AL_MEDLIT_DEPLOYMENT_PROFILE: ${AL_MEDLIT_DEPLOYMENT_PROFILE:-laptop}"
LOCAL_ATTEMPT_ROOT_LINE = "AL_MEDLIT_LOCAL_ATTEMPT_ROOT: /var/lib/al-medlit/attempts"
LOCAL_ATTEMPT_VOLUME_LINE = "- attempt_data:/var/lib/al-medlit/attempts"
HEAVY_TRAINING_PACKAGES = {
    "bitsandbytes",
    "peft",
    "scikit-learn",
    "torch",
    "transformers",
    "triton",
}


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\\[]", requirement, maxsplit=1)[0].strip().lower()


def test_startup_rejects_default_jwt_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", DEFAULT_JWT_SECRET)

    with pytest.raises(RuntimeError, match="AL_MEDLIT_JWT_SECRET"):
        with TestClient(create_app()):
            pass


@pytest.mark.parametrize("secret", ["", "   "])
def test_runtime_secret_guard_rejects_blank_jwt_secret(secret, monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", secret)

    with pytest.raises(RuntimeError, match="AL_MEDLIT_JWT_SECRET"):
        settings.validate_runtime_secrets()


@pytest.mark.parametrize("secret", ["x", "a" * 31, chr(0x1F40D) * 7])
def test_runtime_secret_guard_rejects_jwt_secrets_under_32_utf8_bytes(secret, monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", secret)

    with pytest.raises(RuntimeError, match="at least 32 UTF-8 bytes"):
        settings.validate_runtime_secrets()


def test_startup_accepts_configured_jwt_secret(monkeypatch):
    monkeypatch.setattr(settings, "jwt_secret", "configured-test-jwt-secret-32-byte-minimum")

    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_runtime_secret_guard_rejects_default_bootstrap_admin_password(monkeypatch):
    monkeypatch.setattr(settings, "bootstrap_admin_password", DEFAULT_BOOTSTRAP_ADMIN_PASSWORD)

    with pytest.raises(RuntimeError, match="AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD"):
        settings.validate_runtime_secrets()


def test_bootstrap_admin_password_must_be_strong_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "bootstrap_admin_password", "short")

    with pytest.raises(RuntimeError, match="AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD"):
        settings.validate_bootstrap_admin_password()


@pytest.mark.parametrize("password", ["a" * 73, chr(0x1F40D) * 19])
def test_bootstrap_admin_password_respects_bcrypt_byte_limit(password, monkeypatch):
    monkeypatch.setattr(settings, "bootstrap_admin_password", password)

    with pytest.raises(RuntimeError, match="at most 72 UTF-8 bytes"):
        settings.validate_bootstrap_admin_password()


def test_compose_requires_jwt_secret_for_backend_python_services():
    compose = (ROOT_DIR / "infra" / "docker-compose.yml").read_text()

    python_service_count = compose.count(
        "dockerfile: infra/docker/backend.Dockerfile"
    ) + compose.count("dockerfile: infra/docker/worker.Dockerfile")
    assert compose.count(REQUIRED_COMPOSE_JWT_LINE) == python_service_count
    assert compose.count(BOOTSTRAP_ADMIN_PASSWORD_LINE) == 1
    assert compose.count(ALLOW_SELF_REGISTRATION_LINE) == 1
    assert compose.count(DEPLOYMENT_PROFILE_LINE) == python_service_count
    assert compose.count(LOCAL_ATTEMPT_ROOT_LINE) == python_service_count
    assert compose.count(LOCAL_ATTEMPT_VOLUME_LINE) == python_service_count
    assert "attempt_data:" in compose


def test_lab_make_target_sets_lab_deployment_profile():
    makefile = (ROOT_DIR / "Makefile").read_text()

    assert "AL_MEDLIT_DEPLOYMENT_PROFILE=lab" in makefile


def test_default_api_install_excludes_training_and_accelerator_packages():
    backend_project = tomllib.loads(
        (ROOT_DIR / "backend" / "pyproject.toml").read_text()
    )
    default_dependencies = {
        _dependency_name(requirement)
        for requirement in backend_project["project"]["dependencies"]
    }
    backend_dockerfile = (
        ROOT_DIR / "infra" / "docker" / "backend.Dockerfile"
    ).read_text()

    assert default_dependencies.isdisjoint(HEAVY_TRAINING_PACKAGES)
    assert "--extra" not in backend_dockerfile
    assert "worker.Dockerfile" not in backend_dockerfile


def test_env_example_documents_registration_and_deployment_profile():
    example = (ROOT_DIR / ".env.example").read_text()

    assert "AL_MEDLIT_ALLOW_SELF_REGISTRATION=true" in example
    assert "AL_MEDLIT_DEPLOYMENT_PROFILE=laptop" in example


def test_backend_makefile_exports_jwt_secret_for_host_dev():
    makefile = (ROOT_DIR / "backend" / "Makefile").read_text()

    assert "AL_MEDLIT_JWT_SECRET ?=" in makefile
    assert "export AL_MEDLIT_JWT_SECRET" in makefile


def test_frontend_proxy_rate_limits_public_auth_and_api_is_loopback_only():
    nginx = (ROOT_DIR / "frontend" / "nginx" / "default.conf.template").read_text()
    rate_limit = (
        ROOT_DIR / "frontend" / "nginx" / "rate-limit.conf.template"
    ).read_text()
    compose = (ROOT_DIR / "infra" / "docker-compose.yml").read_text()

    assert "auth/(?:login|register)" in nginx
    assert "invites/[^/]+/accept" in nginx
    assert "limit_req zone=al_medlit_auth" in nginx
    assert "limit_req_zone $binary_remote_addr zone=al_medlit_auth" in rate_limit
    assert '"127.0.0.1:${AL_MEDLIT_BACKEND_HOST_PORT:-8001}:8000"' in compose
    assert "AL_MEDLIT_AUTH_RATE: ${AL_MEDLIT_AUTH_RATE:-10r/m}" in compose
