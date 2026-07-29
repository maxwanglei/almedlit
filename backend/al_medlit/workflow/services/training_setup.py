"""Domain operations for the canonical learning workflow."""

import re
from datetime import UTC, datetime

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.config import settings
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.project.models import Project
from al_medlit.training.runtime_profiles import (
    RUNTIME_PROFILES,
    RuntimeReadinessReport,
    runtime_profile_report_sha256,
    validate_ready_runtime_report,
)
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _normalized_sha256,
    _project,
    _scoped,
)

MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATASET_UPLOAD_ITEMS = 100_000
MAX_JSONL_LINE_BYTES = 1024 * 1024
_PINNED_HUGGING_FACE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_AUTOMATIC_SELECTION_STRATEGIES = {
    "all",
    "random",
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_FEEDBACK_SELECTION_STRATEGIES = {
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_SENSITIVE_TRAINING_CONFIG_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}




def create_training_recipe(
    db: Session, data: schemas.TrainingRecipeCreate, actor: User
) -> models.TrainingRecipe:
    _project(db, data.project_id)
    recipe = models.TrainingRecipe(**data.model_dump(), created_by_user_id=actor.id)
    db.add(recipe)
    _commit(db, "A training recipe with this key already exists in the project")
    db.refresh(recipe)
    return recipe


def list_training_recipes(db: Session, project_id: int) -> list[models.TrainingRecipe]:
    _project(db, project_id)
    return (
        db.query(models.TrainingRecipe)
        .filter(models.TrainingRecipe.project_id == project_id)
        .order_by(models.TrainingRecipe.name, models.TrainingRecipe.id)
        .all()
    )


def create_training_recipe_version(
    db: Session, data: schemas.TrainingRecipeVersionCreate, actor: User
) -> models.TrainingRecipeVersion:
    recipe = _scoped(
        db,
        models.TrainingRecipe,
        data.training_recipe_id,
        data.project_id,
        "Training recipe",
    )
    db.query(models.TrainingRecipe).filter(
        models.TrainingRecipe.id == recipe.id
    ).with_for_update().one()
    content = data.model_dump(exclude={"project_id", "training_recipe_id"})
    version = models.TrainingRecipeVersion(
        **data.model_dump(),
        version_number=_next_version(
            db,
            models.TrainingRecipeVersion,
            models.TrainingRecipeVersion.training_recipe_id,
            recipe.id,
        ),
        content_hash=_canonical_hash(content),
        created_by_user_id=actor.id,
    )
    db.add(version)
    _commit(db, "This training recipe version already exists")
    db.refresh(version)
    return version


def list_training_recipe_versions(
    db: Session, project_id: int, training_recipe_id: int
) -> list[models.TrainingRecipeVersion]:
    _scoped(
        db,
        models.TrainingRecipe,
        training_recipe_id,
        project_id,
        "Training recipe",
    )
    return (
        db.query(models.TrainingRecipeVersion)
        .filter(models.TrainingRecipeVersion.training_recipe_id == training_recipe_id)
        .order_by(models.TrainingRecipeVersion.version_number)
        .all()
    )


def ensure_trusted_training_recipe_version(
    db: Session,
    project_id: int,
    recipe_key: str,
    actor: User,
) -> models.TrainingRecipeVersion:
    """Bind an installed server recipe contract to a project idempotently."""

    _project(db, project_id)
    db.query(Project).filter(Project.id == project_id).with_for_update().one()
    from al_medlit.training.recipe_registry import training_recipes

    descriptor = training_recipes.get(recipe_key)
    recipe = (
        db.query(models.TrainingRecipe)
        .filter(
            models.TrainingRecipe.project_id == project_id,
            models.TrainingRecipe.key == descriptor.key,
        )
        .one_or_none()
    )
    if recipe is None:
        recipe = create_training_recipe(
            db,
            schemas.TrainingRecipeCreate(
                project_id=project_id,
                key=descriptor.key,
                name=descriptor.label,
                description=descriptor.description,
            ),
            actor,
        )

    version_data = schemas.TrainingRecipeVersionCreate(
        project_id=project_id,
        training_recipe_id=recipe.id,
        trainer_plugin_key=descriptor.trainer_key,
        trainer_plugin_version=descriptor.version,
        compatible_task_kinds=[task_kind.value for task_kind in descriptor.supported_task_kinds],
        environment_class=descriptor.environment.runtime_class.value,
        config_schema=descriptor.config_schema,
        default_config={},
        evaluation_defaults={
            "splits": ["test"],
            "selection_split": "validation",
        },
    )
    content_hash = _canonical_hash(
        version_data.model_dump(exclude={"project_id", "training_recipe_id"})
    )
    existing = (
        db.query(models.TrainingRecipeVersion)
        .filter(
            models.TrainingRecipeVersion.training_recipe_id == recipe.id,
            models.TrainingRecipeVersion.content_hash == content_hash,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing
    return create_training_recipe_version(db, version_data, actor)


def create_execution_environment(
    db: Session, data: schemas.ExecutionEnvironmentCreate, actor: User
) -> models.ExecutionEnvironment:
    _project(db, data.project_id)
    if data.environment_class not in RUNTIME_PROFILES:
        raise ValidationError("Execution environment must use a supported runtime profile")
    environment = models.ExecutionEnvironment(
        **data.model_dump(exclude={"image_digest"}),
        image_digest=f"sha256:{_normalized_sha256(data.image_digest)}",
        created_by_user_id=actor.id,
    )
    db.add(environment)
    _commit(db, "An execution environment with this name already exists")
    db.refresh(environment)
    return environment


def list_execution_environments(db: Session, project_id: int) -> list[models.ExecutionEnvironment]:
    _project(db, project_id)
    return (
        db.query(models.ExecutionEnvironment)
        .filter(models.ExecutionEnvironment.project_id == project_id)
        .order_by(models.ExecutionEnvironment.name, models.ExecutionEnvironment.id)
        .all()
    )


def verify_execution_environment(
    db: Session,
    project_id: int,
    environment_id: int,
    data: schemas.EnvironmentVerification,
) -> models.ExecutionEnvironment:
    environment = _scoped(
        db,
        models.ExecutionEnvironment,
        environment_id,
        project_id,
        "Execution environment",
    )
    raw_report = data.verification_report.get(
        "readiness_report",
        data.verification_report,
    )
    try:
        report = RuntimeReadinessReport.model_validate(raw_report)
    except PydanticValidationError as exc:
        raise ValidationError(
            "Environment verification must contain a complete worker readiness report"
        ) from exc
    if report.runtime_profile != environment.environment_class:
        raise ValidationError("Readiness report runtime profile does not match the environment")
    if report.worker_image_digest is None or (
        _normalized_sha256(report.worker_image_digest)
        != _normalized_sha256(environment.image_digest)
    ):
        raise ValidationError("Readiness report image digest does not match the environment")
    if data.status == "available" and not report.ready:
        raise ValidationError("An unavailable worker report cannot enable an environment")
    if data.status == "unavailable" and report.ready:
        raise ValidationError("A ready worker report must be recorded as available")
    descriptor = RUNTIME_PROFILES[environment.environment_class]
    if data.status == "available":
        try:
            validate_ready_runtime_report(report)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
    for import_name, expected_version in environment.package_manifest.items():
        distribution_name = descriptor.distributions.get(import_name, import_name)
        actual_version = report.dependency_versions.get(import_name)
        if actual_version is None:
            matching_import = next(
                (
                    key
                    for key, distribution in descriptor.distributions.items()
                    if distribution == distribution_name
                ),
                None,
            )
            actual_version = (
                report.dependency_versions.get(matching_import)
                if matching_import is not None
                else None
            )
        if str(actual_version) != str(expected_version):
            raise ValidationError(
                f"Worker package {distribution_name!r} does not match the pinned version"
            )
    minimum_memory = int(
        environment.hardware_constraints.get("minimum_device_memory_bytes", 0) or 0
    )
    if minimum_memory and int(report.device_memory_bytes or 0) < minimum_memory:
        raise ValidationError("Worker device memory is below the environment requirement")
    normalized_report = report.model_dump(mode="json")
    environment.status = data.status
    environment.verification_report = {
        "readiness_report": normalized_report,
        "report_sha256": runtime_profile_report_sha256(report),
    }
    environment.verified_at = datetime.now(UTC)
    _commit(db, "Could not update the environment verification")
    db.refresh(environment)
    return environment


def _validated_environment_readiness(
    environment: models.ExecutionEnvironment,
) -> RuntimeReadinessReport:
    if environment.status != "available" or environment.verified_at is None:
        raise ConflictError("Execution environment must pass verification before launch")
    verification = environment.verification_report
    raw_report = verification.get("readiness_report") if isinstance(verification, dict) else None
    if not isinstance(raw_report, dict):
        raise ConflictError("Execution environment has no readiness attestation")
    try:
        report = RuntimeReadinessReport.model_validate(raw_report)
        validate_ready_runtime_report(report)
    except (PydanticValidationError, ValueError) as exc:
        raise ConflictError("Execution environment readiness attestation is invalid") from exc
    if (
        report.runtime_profile != environment.environment_class
        or report.worker_image_digest is None
        or _normalized_sha256(report.worker_image_digest)
        != _normalized_sha256(environment.image_digest or "")
        or verification.get("report_sha256") != runtime_profile_report_sha256(report)
    ):
        raise ConflictError("Execution environment readiness attestation does not match")
    descriptor = RUNTIME_PROFILES[environment.environment_class]
    for import_name, expected_version in environment.package_manifest.items():
        distribution_name = descriptor.distributions.get(import_name, import_name)
        actual_version = report.dependency_versions.get(import_name)
        if actual_version is None:
            matching_import = next(
                (
                    key
                    for key, distribution in descriptor.distributions.items()
                    if distribution == distribution_name
                ),
                None,
            )
            actual_version = (
                report.dependency_versions.get(matching_import)
                if matching_import is not None
                else None
            )
        if str(actual_version) != str(expected_version):
            raise ConflictError(f"Worker package {distribution_name!r} no longer matches its pin")
    minimum_memory = int(
        environment.hardware_constraints.get("minimum_device_memory_bytes", 0) or 0
    )
    if minimum_memory and int(report.device_memory_bytes or 0) < minimum_memory:
        raise ConflictError("Worker device memory is below the environment requirement")
    return report


def _configured_storage_encryption() -> dict[str, str]:
    mode = settings.storage_encryption_mode
    key_id = settings.storage_kms_key_id.strip()
    if settings.storage_backend == "local" and mode != "none":
        raise ValidationError("Local development storage does not support encryption modes")
    if mode == "sse-kms":
        if not key_id:
            raise ValidationError("Configured MinIO SSE-KMS storage requires a key identifier")
        return {"mode": mode, "key_id": key_id}
    if key_id:
        raise ValidationError("A configured storage KMS key identifier requires SSE-KMS mode")
    return {"mode": mode}


def _normalize_storage_encryption(
    raw_policy: dict,
    *,
    backend: str,
) -> dict[str, str]:
    configured = _configured_storage_encryption()
    if not raw_policy:
        requested = configured
    else:
        allowed_keys = {"mode", "key_id"}
        unknown_keys = set(raw_policy) - allowed_keys
        if unknown_keys:
            raise ValidationError(
                "Storage encryption contains unsupported fields: " + ", ".join(sorted(unknown_keys))
            )
        mode = raw_policy.get("mode")
        if mode not in {"none", "sse-s3", "sse-kms"}:
            raise ValidationError("Storage encryption mode must be none, sse-s3, or sse-kms")
        key_id = raw_policy.get("key_id")
        if key_id is not None and (
            not isinstance(key_id, str) or not key_id.strip() or len(key_id.strip()) > 255
        ):
            raise ValidationError("Storage encryption key_id is invalid")
        if mode == "sse-kms" and not key_id:
            raise ValidationError("SSE-KMS storage requires a key_id")
        if mode != "sse-kms" and key_id is not None:
            raise ValidationError("key_id is valid only with SSE-KMS storage")
        requested = (
            {"mode": mode, "key_id": key_id.strip()} if mode == "sse-kms" else {"mode": mode}
        )
    if backend == "local" and requested["mode"] != "none":
        raise ValidationError("Local development storage supports only mode=none")
    if requested != configured:
        raise ValidationError(
            "Storage encryption policy does not match the configured object store"
        )
    return requested


def _validated_storage_policy(
    policy: models.StoragePolicy,
    *,
    project: Project,
) -> dict[str, str]:
    if policy.backend not in {"minio", "local"}:
        raise ValidationError("Storage policy uses an unsupported backend")
    if policy.backend != settings.storage_backend:
        raise ValidationError("Storage policy backend does not match the configured object store")
    if policy.backend == "local" and settings.deployment_profile != "laptop":
        raise ValidationError("Local object storage is development-only")
    if policy.backend == "local" and policy.is_default:
        raise ValidationError("Local development storage cannot be a shared default")
    required_prefix = artifact_service.workspace_blob_prefix(project.workspace_id)
    if policy.artifact_prefix.strip("/") != required_prefix:
        raise ValidationError(f"Storage artifact_prefix must equal {required_prefix!r}")
    if policy.retention_class not in {"indefinite", "resume_14d"}:
        raise ValidationError("Storage policy uses an unsupported retention class")
    if policy.cache_policy:
        raise ValidationError("Training storage cache policies are not supported by this worker")
    return _normalize_storage_encryption(
        policy.encryption,
        backend=policy.backend,
    )


def create_storage_policy(
    db: Session, data: schemas.StoragePolicyCreate, actor: User
) -> models.StoragePolicy:
    project = _project(db, data.project_id)
    if data.backend != settings.storage_backend:
        raise ValidationError("Storage policy backend does not match the configured object store")
    if data.backend == "local" and settings.deployment_profile != "laptop":
        raise ValidationError("Local object storage is development-only")
    if data.backend == "local" and data.is_default:
        raise ValidationError("Local storage cannot be the default shared storage policy")
    required_prefix = artifact_service.workspace_blob_prefix(project.workspace_id)
    normalized_prefix = data.artifact_prefix.strip("/")
    if normalized_prefix != required_prefix:
        raise ValidationError(f"Storage artifact_prefix must equal {required_prefix!r}")
    if data.cache_policy:
        raise ValidationError("Training storage cache policies are not supported by this worker")
    encryption = _normalize_storage_encryption(
        data.encryption,
        backend=data.backend,
    )
    if data.is_default:
        db.query(models.StoragePolicy).filter(
            models.StoragePolicy.project_id == data.project_id,
            models.StoragePolicy.is_default.is_(True),
        ).update({"is_default": False}, synchronize_session=False)
    policy = models.StoragePolicy(
        **data.model_dump(exclude={"artifact_prefix", "encryption", "cache_policy"}),
        artifact_prefix=normalized_prefix,
        encryption=encryption,
        cache_policy={},
        created_by_user_id=actor.id,
    )
    db.add(policy)
    _commit(db, "A storage policy with this name already exists")
    db.refresh(policy)
    return policy


def list_storage_policies(db: Session, project_id: int) -> list[models.StoragePolicy]:
    _project(db, project_id)
    return (
        db.query(models.StoragePolicy)
        .filter(models.StoragePolicy.project_id == project_id)
        .order_by(models.StoragePolicy.is_default.desc(), models.StoragePolicy.name)
        .all()
    )
