import tempfile
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine.url import make_url

DEFAULT_JWT_SECRET = "change-me-in-production-change-me"
DEFAULT_BOOTSTRAP_ADMIN_PASSWORD = "change-me-now"
FORBIDDEN_BOOTSTRAP_ADMIN_PASSWORDS = frozenset({DEFAULT_BOOTSTRAP_ADMIN_PASSWORD})

# Placeholder and vendor-default credentials that must never reach a deployed
# datastore. These are the values shipped in .env.example and the local
# development defaults below, all of which are public knowledge.
FORBIDDEN_DEPLOYMENT_SECRETS = frozenset(
    {
        "",
        "change-me",
        "changeme",
        "al_medlit",
        "password",
        "postgres",
        "minioadmin",
    }
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AL_MEDLIT_")

    app_name: str = "AL-MedLit"

    # Single canonical stack: PostgreSQL + Redis + MinIO. The deployment
    # profile (laptop vs lab) varies topology, not storage technology.
    database_url: str = "postgresql+psycopg2://al_medlit:al_medlit@localhost:5432/al_medlit"
    echo_sql: bool = False

    # Default to the laptop profile: tasks execute eagerly unless the lab
    # profile overrides these values to point at Redis.
    celery_broker_url: str = "memory://"
    celery_task_always_eager: bool = True
    # Deployment profile for capability infra-filtering: "laptop" | "lab".
    # The lab Docker Compose profile sets AL_MEDLIT_DEPLOYMENT_PROFILE=lab.
    deployment_profile: str = "laptop"
    # Shared durable root for local attempt bundles, process identity, logs,
    # metrics, and recovery state. Compose mounts this path into every Python
    # service that can submit or reconcile work.
    local_attempt_root: str = str(Path(tempfile.gettempdir()) / "al-medlit-attempts")

    # Auth / JWT. jwt_secret MUST be overridden in production via
    # AL_MEDLIT_JWT_SECRET. allow_self_registration can be disabled for
    # sealed PHI deployments.
    jwt_secret: str = DEFAULT_JWT_SECRET
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720
    allow_self_registration: bool = True

    # Optional one-time bootstrap superuser credentials for the explicit
    # scripts.bootstrap_admin command. Migrations must not create loginable
    # accounts from these settings.
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = ""

    storage_endpoint: str = "localhost:9000"
    storage_access_key: str = "al_medlit"
    storage_secret_key: str = "al_medlit"
    storage_bucket: str = "al-medlit"
    storage_secure: bool = False
    # Object storage backend: "minio" (canonical) or "local" (tests/dev).
    storage_backend: Literal["minio", "local"] = "minio"
    storage_local_dir: str = "./object_store"
    # MinIO is S3-compatible. These modes are applied to every object upload;
    # the key identifier is not secret and is required only for SSE-KMS.
    storage_encryption_mode: Literal["none", "sse-s3", "sse-kms"] = "none"
    storage_kms_key_id: str = ""

    # Default thresholds for error-guideline learning.
    error_pattern_min_examples: int = 3
    confident_disagreement_threshold: float = 0.85

    # NCBI E-utilities access for the PubMed/PMC importer. All optional: the
    # importer works without a key (3 req/s); a key raises the limit to 10 req/s.
    ncbi_api_key: str = ""
    ncbi_email: str = ""
    ncbi_tool: str = "al-medlit"
    ncbi_eutils_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ncbi_idconv_base: str = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
    ncbi_request_timeout: float = 30.0

    def validate_runtime_secrets(self) -> None:
        if not self.jwt_secret.strip() or self.jwt_secret == DEFAULT_JWT_SECRET:
            raise RuntimeError(
                "AL_MEDLIT_JWT_SECRET must be set to a non-default value before "
                "starting the API"
            )
        if len(self.jwt_secret.encode("utf-8")) < 32:
            raise RuntimeError("AL_MEDLIT_JWT_SECRET must be at least 32 UTF-8 bytes")
        self.validate_bootstrap_admin_password()

    def validate_bootstrap_admin_password(self, *, require_configured: bool = False) -> None:
        password = self.bootstrap_admin_password.strip()
        if not password:
            if require_configured:
                raise RuntimeError(
                    "AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD must be set before creating "
                    "the bootstrap admin"
                )
            return
        if password in FORBIDDEN_BOOTSTRAP_ADMIN_PASSWORDS:
            raise RuntimeError(
                "AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD must not use a known default value"
            )
        password_bytes = len(password.encode("utf-8"))
        if password_bytes < 12:
            raise RuntimeError(
                "AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD must be at least 12 UTF-8 bytes"
            )
        if password_bytes > 72:
            raise RuntimeError(
                "AL_MEDLIT_BOOTSTRAP_ADMIN_PASSWORD must be at most 72 UTF-8 bytes"
            )

    def validate_deployment_secrets(self) -> None:
        """Reject known placeholder credentials for networked datastores.

        Called from the container entrypoint rather than the FastAPI lifespan:
        the local development defaults (SQLite, local object storage) are
        legitimately weak, and only a networked deployment needs this gate.
        """

        if self.storage_backend == "minio":
            secret = self.storage_secret_key.strip()
            if secret.lower() in FORBIDDEN_DEPLOYMENT_SECRETS:
                raise RuntimeError(
                    "AL_MEDLIT_STORAGE_SECRET_KEY (MINIO_ROOT_PASSWORD) is a known "
                    "placeholder value. Set a strong unique secret before starting "
                    "the API; generate one with: openssl rand -hex 32"
                )

        url = make_url(self.database_url)
        # A URL without a host is a local file database (SQLite) and carries no
        # password to validate.
        if url.host:
            db_password = (url.password or "").strip()
            if db_password.lower() in FORBIDDEN_DEPLOYMENT_SECRETS:
                raise RuntimeError(
                    "The database password in AL_MEDLIT_DATABASE_URL "
                    "(POSTGRES_PASSWORD) is a known placeholder value. Set a "
                    "strong unique password before starting the API; generate one "
                    "with: openssl rand -hex 32"
                )


settings = Settings()
