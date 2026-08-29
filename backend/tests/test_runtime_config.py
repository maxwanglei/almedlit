import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from al_medlit.core.config import (
    DEFAULT_BOOTSTRAP_ADMIN_PASSWORD,
    DEFAULT_JWT_SECRET,
    Settings,
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
SECURE_MINIO_LINE = 'AL_MEDLIT_STORAGE_SECURE: "true"'
MINIO_CA_PATH_LINE = "AL_MEDLIT_STORAGE_CA_CERT_PATH: /etc/al-medlit/minio-certs/CAs/ca.crt"
MINIO_CERT_MOUNT_LINE = (
    "- ${AL_MEDLIT_MINIO_CERTS_DIR:-./certs/minio}:/etc/al-medlit/minio-certs:ro"
)
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
    assert compose.count(SECURE_MINIO_LINE) == python_service_count
    assert compose.count(MINIO_CA_PATH_LINE) == python_service_count
    assert compose.count(MINIO_CERT_MOUNT_LINE) == python_service_count
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
    # Covers both the invite preview and the accept POST.
    assert "invites/[^/]+(?:/accept)?" in nginx
    assert "limit_req zone=al_medlit_auth" in nginx
    assert "limit_req_zone $binary_remote_addr zone=al_medlit_auth" in rate_limit
    assert '"127.0.0.1:${AL_MEDLIT_BACKEND_HOST_PORT:-8001}:8000"' in compose
    assert "AL_MEDLIT_AUTH_RATE: ${AL_MEDLIT_AUTH_RATE:-10r/m}" in compose


def test_frontend_proxy_sets_security_headers_for_all_responses():
    nginx = (ROOT_DIR / "frontend" / "nginx" / "default.conf.template").read_text()

    assert "add_header Content-Security-Policy" in nginx
    assert "default-src 'self'" in nginx
    assert "script-src 'self'" in nginx
    assert "script-src-attr 'none'" in nginx
    assert "object-src 'none'" in nginx
    assert "frame-ancestors 'none'" in nginx
    assert "connect-src 'self'" in nginx
    assert "'unsafe-eval'" not in nginx
    assert 'add_header X-Frame-Options "DENY" always;' in nginx
    assert 'add_header X-Content-Type-Options "nosniff" always;' in nginx
    assert (
        'add_header Strict-Transport-Security "max-age=31536000" always;'
        in nginx
    )
    assert 'add_header Referrer-Policy "no-referrer" always;' in nginx
    assert "add_header Permissions-Policy" in nginx


STRONG_DB_URL = "postgresql+psycopg2://al_medlit:8f3c1d9e2b7a4506@db:5432/al_medlit"


@pytest.mark.parametrize("secret", ["change-me", "CHANGE-ME", "al_medlit", "minioadmin", "  "])
def test_deployment_guard_rejects_placeholder_storage_secrets(secret, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_secret_key", secret)
    monkeypatch.setattr(settings, "database_url", STRONG_DB_URL)

    with pytest.raises(RuntimeError, match="AL_MEDLIT_STORAGE_SECRET_KEY"):
        settings.validate_deployment_secrets()


@pytest.mark.parametrize("password", ["change-me", "al_medlit", "postgres", "password"])
def test_deployment_guard_rejects_placeholder_database_passwords(password, monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_secret_key", "8f3c1d9e2b7a4506")
    monkeypatch.setattr(
        settings,
        "database_url",
        f"postgresql+psycopg2://al_medlit:{password}@db:5432/al_medlit",
    )

    with pytest.raises(RuntimeError, match="AL_MEDLIT_DATABASE_URL"):
        settings.validate_deployment_secrets()


def test_deployment_guard_accepts_strong_secrets(monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_secret_key", "0c4b8a1f6d2e9370")
    monkeypatch.setattr(settings, "database_url", STRONG_DB_URL)

    settings.validate_deployment_secrets()


def test_runtime_guard_rejects_insecure_minio_transport(monkeypatch):
    monkeypatch.setattr(settings, "storage_backend", "minio")
    monkeypatch.setattr(settings, "storage_secure", False)
    monkeypatch.setattr(settings, "jwt_secret", "configured-test-jwt-secret-32-byte-minimum")

    with pytest.raises(RuntimeError, match="AL_MEDLIT_STORAGE_SECURE"):
        settings.validate_runtime_secrets()


def test_runtime_guard_rejects_insecure_auth_cookie(monkeypatch):
    monkeypatch.setattr(settings, "auth_cookie_secure", False)
    monkeypatch.setattr(settings, "jwt_secret", "configured-test-jwt-secret-32-byte-minimum")

    with pytest.raises(RuntimeError, match="AL_MEDLIT_AUTH_COOKIE_SECURE"):
        settings.validate_runtime_secrets()


def test_auth_cookie_is_secure_by_default():
    assert Settings.model_fields["auth_cookie_secure"].default is True


def test_minio_transport_is_secure_by_default():
    assert Settings.model_fields["storage_secure"].default is True


def test_deployment_guard_skips_local_development_defaults(monkeypatch):
    """SQLite and local object storage are legitimately credential-free."""
    monkeypatch.setattr(settings, "storage_backend", "local")
    monkeypatch.setattr(settings, "storage_secret_key", "al_medlit")
    monkeypatch.setattr(settings, "database_url", "sqlite:///./al_medlit.db")

    settings.validate_deployment_secrets()


def test_compose_publishes_datastores_on_loopback_only():
    compose = (ROOT_DIR / "infra" / "docker-compose.yml").read_text()

    assert '"127.0.0.1:${AL_MEDLIT_POSTGRES_HOST_PORT:-5432}:5432"' in compose
    assert '"127.0.0.1:${AL_MEDLIT_MINIO_HOST_PORT:-9000}:9000"' in compose
    assert '"127.0.0.1:${AL_MEDLIT_MINIO_CONSOLE_HOST_PORT:-9001}:9001"' in compose
    # A bare host:container mapping publishes on every interface.
    for exposed in ('"5432:5432"', '"9000:9000"', '"9001:9001"'):
        assert exposed not in compose


def test_compose_minio_transport_uses_generated_tls_certificates():
    compose = (ROOT_DIR / "infra" / "docker-compose.yml").read_text()

    assert "minio-cert-init:" in compose
    assert 'command: server /data --console-address ":9001" --certs-dir /certs' in compose
    assert "https://localhost:9000/minio/health/ready" in compose
    assert "http://localhost:9000/minio/health/ready" not in compose
    assert "condition: service_completed_successfully" in compose


def test_minio_certificate_generator_is_idempotent_and_verifiable(tmp_path):
    openssl = shutil.which("openssl")
    if openssl is None:
        pytest.skip("OpenSSL is required by the Compose certificate-init image")
    script = ROOT_DIR / "infra" / "scripts" / "generate-minio-certs.sh"
    cert_dir = tmp_path / "certs"

    subprocess.run(["/bin/sh", str(script), str(cert_dir)], check=True)
    certificate = cert_dir / "public.crt"
    ca_certificate = cert_dir / "CAs" / "ca.crt"
    private_key = cert_dir / "private.key"
    generated_marker = cert_dir / ".al-medlit-generated"
    first_certificate = certificate.read_bytes()

    subprocess.run(["/bin/sh", str(script), str(cert_dir)], check=True)
    verification = subprocess.run(
        [openssl, "verify", "-CAfile", str(ca_certificate), str(certificate)],
        check=False,
        capture_output=True,
        text=True,
    )
    certificate_details = subprocess.run(
        [openssl, "x509", "-in", str(certificate), "-noout", "-text"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert verification.returncode == 0, verification.stderr
    assert certificate.read_bytes() == first_certificate
    assert "DNS:minio" in certificate_details.stdout
    assert "DNS:localhost" in certificate_details.stdout
    assert "IP Address:127.0.0.1" in certificate_details.stdout
    assert private_key.stat().st_mode & 0o777 == 0o600
    assert generated_marker.stat().st_mode & 0o777 == 0o600


def test_env_example_ships_no_working_datastore_passwords():
    env_example = (ROOT_DIR / ".env.example").read_text()

    assert "POSTGRES_PASSWORD=\n" in env_example
    assert "MINIO_ROOT_PASSWORD=\n" in env_example
    assert "POSTGRES_PASSWORD=change-me" not in env_example
    assert "MINIO_ROOT_PASSWORD=change-me" not in env_example
