"""Domain operations for the canonical learning workflow."""

import re
from datetime import UTC, datetime

from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    NotFoundError,
    ValidationError,
)
from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import BaseModelAsset
from al_medlit.project.models import Project
from al_medlit.workflow import access, models, schemas
from al_medlit.workspace.models import Workspace

from .common import (
    _commit,
    _scoped,
)
from .models import (
    _model_version_lineage_payloads,
)
from .training_runs import (
    get_training_run,
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




def _workspace_projects_with_modules(
    db: Session,
    *,
    workspace_id: int,
    required_modules: set[str],
    project_id: int | None = None,
) -> list[tuple[Project, set[str]]]:
    if db.get(Workspace, workspace_id) is None:
        raise NotFoundError("Workspace not found")
    query = db.query(Project).filter(Project.workspace_id == workspace_id)
    if project_id is not None:
        query = query.filter(Project.id == project_id)
    projects = query.order_by(Project.id.desc()).all()
    eligible: list[tuple[Project, set[str]]] = []
    for project in projects:
        effective_modules = access.effective_project_modules(db, project.id)
        if required_modules.intersection(effective_modules):
            eligible.append((project, effective_modules))
    return eligible


def list_workspace_training_contexts(
    db: Session,
    *,
    workspace_id: int,
    project_id: int | None = None,
    cursor: int | None = None,
    limit: int = 50,
) -> tuple[list[dict], int | None]:
    contexts = _workspace_projects_with_modules(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        required_modules={"train"},
    )
    if cursor is not None:
        contexts = [
            (project, effective_modules)
            for project, effective_modules in contexts
            if project.id < cursor
        ]
    bounded_limit = min(max(limit, 1), 100)
    has_more = len(contexts) > bounded_limit
    contexts = contexts[:bounded_limit]
    items = []
    for project, effective_modules in contexts:
        task_version_count = (
            db.query(func.count(models.TaskVersion.id))
            .filter(models.TaskVersion.project_id == project.id)
            .scalar()
            or 0
        )
        training_dataset_version_count = (
            db.query(func.count(models.TrainingDatasetVersion.id))
            .filter(models.TrainingDatasetVersion.project_id == project.id)
            .scalar()
            or 0
        )
        environment_count = (
            db.query(func.count(models.ExecutionEnvironment.id))
            .filter(models.ExecutionEnvironment.project_id == project.id)
            .scalar()
            or 0
        )
        available_environment_count = (
            db.query(func.count(models.ExecutionEnvironment.id))
            .filter(
                models.ExecutionEnvironment.project_id == project.id,
                models.ExecutionEnvironment.status == "available",
            )
            .scalar()
            or 0
        )
        storage_policy_count = (
            db.query(func.count(models.StoragePolicy.id))
            .filter(models.StoragePolicy.project_id == project.id)
            .scalar()
            or 0
        )
        items.append(
            {
                "project_id": project.id,
                "project_name": project.name,
                "project_description": project.description,
                "training_only": (
                    {"data", "models", "train"}.issubset(effective_modules)
                    and "annotate" not in effective_modules
                ),
                "effective_modules": [
                    module
                    for module in access.PROJECT_MODULES
                    if module in effective_modules
                ],
                "task_version_count": int(task_version_count),
                "training_dataset_version_count": int(training_dataset_version_count),
                "environment_count": int(environment_count),
                "available_environment_count": int(available_environment_count),
                "storage_policy_count": int(storage_policy_count),
            }
        )
    next_cursor = items[-1]["project_id"] if has_more and items else None
    return items, next_cursor


def _training_recipe_run_identity(
    recipe: models.TrainingRecipe,
    recipe_version: models.TrainingRecipeVersion,
    config: dict,
) -> tuple[str, str, dict, str]:
    from al_medlit.training.recipe_registry import training_recipes

    try:
        descriptor = training_recipes.get(recipe.key)
    except NotFoundError:
        return (
            "custom",
            recipe_version.trainer_plugin_key,
            {},
            "custom",
        )
    trainer_key = descriptor.trainer_key
    if trainer_key.startswith("sklearn"):
        framework = "scikit-learn"
    elif trainer_key.startswith("huggingface"):
        framework = "transformers"
    else:
        framework = trainer_key
    base_model_asset_id = config.get("base_model_asset_id")
    base_model = (
        {"asset_id": base_model_asset_id}
        if isinstance(base_model_asset_id, int) and not isinstance(base_model_asset_id, bool)
        else {}
    )
    return (
        descriptor.model_family.value,
        framework,
        base_model,
        descriptor.parameterization.value,
    )


def _recipe_version_ids_for_family(
    db: Session,
    *,
    project_ids: list[int],
    family: str,
) -> list[int]:
    rows = (
        db.query(models.TrainingRecipeVersion, models.TrainingRecipe)
        .join(
            models.TrainingRecipe,
            models.TrainingRecipe.id == models.TrainingRecipeVersion.training_recipe_id,
        )
        .filter(models.TrainingRecipeVersion.project_id.in_(project_ids))
        .all()
    )
    return [
        recipe_version.id
        for recipe_version, recipe in rows
        if _training_recipe_run_identity(recipe, recipe_version, {})[0] == family
    ]


def list_workspace_training_runs(
    db: Session,
    *,
    workspace_id: int,
    project_id: int | None = None,
    cursor: int | None = None,
    limit: int = 50,
    status: str | None = None,
    creator_user_id: int | None = None,
    task_version_id: int | None = None,
    family: str | None = None,
) -> tuple[list[dict], int | None]:
    projects_with_modules = _workspace_projects_with_modules(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        required_modules={"activity", "train"},
    )
    projects = {project.id: project for project, _modules in projects_with_modules}
    if not projects:
        return [], None

    query = db.query(models.TrainingRun).filter(models.TrainingRun.project_id.in_(list(projects)))
    if cursor is not None:
        query = query.filter(models.TrainingRun.id < cursor)
    if status is not None:
        query = query.filter(models.TrainingRun.status == status)
    if creator_user_id is not None:
        query = query.filter(models.TrainingRun.created_by_user_id == creator_user_id)
    if task_version_id is not None:
        query = query.filter(models.TrainingRun.task_version_id == task_version_id)
    if family is not None:
        matching_versions = db.query(models.ModelVersion.id).filter(
            models.ModelVersion.project_id.in_(list(projects)),
            models.ModelVersion.family == family,
        )
        matching_recipe_version_ids = _recipe_version_ids_for_family(
            db,
            project_ids=list(projects),
            family=family,
        )
        query = query.filter(
            or_(
                models.TrainingRun.output_model_version_id.in_(matching_versions),
                models.TrainingRun.parent_model_version_id.in_(matching_versions),
                models.TrainingRun.recipe_version_id.in_(matching_recipe_version_ids),
            )
        )
    bounded_limit = min(max(limit, 1), 100)
    runs = query.order_by(models.TrainingRun.id.desc()).limit(bounded_limit + 1).all()
    has_more = len(runs) > bounded_limit
    runs = runs[:bounded_limit]
    if not runs:
        return [], None

    registered_models = {
        record.id: record
        for record in db.query(models.RegisteredModel)
        .filter(models.RegisteredModel.id.in_({run.registered_model_id for run in runs}))
        .all()
    }
    task_versions = {
        record.id: record
        for record in db.query(models.TaskVersion)
        .filter(models.TaskVersion.id.in_({run.task_version_id for run in runs}))
        .all()
    }
    task_definitions = {
        record.id: record
        for record in db.query(models.TaskDefinition)
        .filter(
            models.TaskDefinition.id.in_(
                {task.task_definition_id for task in task_versions.values()}
            )
        )
        .all()
    }
    training_datasets = {
        record.id: record
        for record in db.query(models.TrainingDatasetVersion)
        .filter(
            models.TrainingDatasetVersion.id.in_({run.training_dataset_version_id for run in runs})
        )
        .all()
    }
    dataset_versions = {
        record.id: record
        for record in db.query(models.DatasetVersion)
        .filter(
            models.DatasetVersion.id.in_(
                {dataset.dataset_version_id for dataset in training_datasets.values()}
            )
        )
        .all()
    }
    recipe_versions = {
        record.id: record
        for record in db.query(models.TrainingRecipeVersion)
        .filter(models.TrainingRecipeVersion.id.in_({run.recipe_version_id for run in runs}))
        .all()
    }
    recipes = {
        record.id: record
        for record in db.query(models.TrainingRecipe)
        .filter(
            models.TrainingRecipe.id.in_(
                {recipe.training_recipe_id for recipe in recipe_versions.values()}
            )
        )
        .all()
    }
    environments = {
        record.id: record
        for record in db.query(models.ExecutionEnvironment)
        .filter(models.ExecutionEnvironment.id.in_({run.environment_id for run in runs}))
        .all()
    }
    storage_policies = {
        record.id: record
        for record in db.query(models.StoragePolicy)
        .filter(models.StoragePolicy.id.in_({run.storage_policy_id for run in runs}))
        .all()
    }
    model_version_ids = {
        version_id
        for run in runs
        for version_id in (
            run.output_model_version_id,
            run.parent_model_version_id,
        )
        if version_id is not None
    }
    model_versions = {
        record.id: record
        for record in db.query(models.ModelVersion)
        .filter(
            models.ModelVersion.id.in_(model_version_ids),
            models.ModelVersion.project_id.in_(list(projects)),
        )
        .all()
    }
    base_model_asset_ids = {
        base_model_asset_id
        for run in runs
        if isinstance(
            base_model_asset_id := run.config.get("base_model_asset_id"),
            int,
        )
        and not isinstance(base_model_asset_id, bool)
    }
    base_model_assets = {
        asset.id: asset
        for asset in db.query(BaseModelAsset)
        .filter(
            BaseModelAsset.id.in_(base_model_asset_ids),
            BaseModelAsset.project_id.in_(list(projects)),
        )
        .all()
    }
    latest_evaluation_by_run: dict[int, models.ModelEvaluation] = {}
    for evaluation in (
        db.query(models.ModelEvaluation)
        .filter(models.ModelEvaluation.training_run_id.in_({run.id for run in runs}))
        .order_by(models.ModelEvaluation.id.desc())
        .all()
    ):
        latest_evaluation_by_run.setdefault(
            evaluation.training_run_id,
            evaluation,
        )
    creators = {
        record.id: record
        for record in db.query(User)
        .filter(
            User.id.in_(
                {run.created_by_user_id for run in runs if run.created_by_user_id is not None}
            )
        )
        .all()
    }

    items = []
    for run in runs:
        project = projects[run.project_id]
        registered_model = registered_models[run.registered_model_id]
        task_version = task_versions[run.task_version_id]
        task_definition = task_definitions[task_version.task_definition_id]
        training_dataset = training_datasets[run.training_dataset_version_id]
        dataset_version = dataset_versions[training_dataset.dataset_version_id]
        recipe_version = recipe_versions[run.recipe_version_id]
        recipe = recipes[recipe_version.training_recipe_id]
        environment = environments[run.environment_id]
        storage_policy = storage_policies[run.storage_policy_id]
        model_version = model_versions.get(
            run.output_model_version_id or run.parent_model_version_id
        )
        if model_version is not None and (
            model_version.project_id != run.project_id
            or model_version.registered_model_id != run.registered_model_id
        ):
            model_version = None
        (
            recipe_family,
            recipe_framework,
            recipe_base_model,
            recipe_training_method,
        ) = _training_recipe_run_identity(
            recipe,
            recipe_version,
            run.config,
        )
        base_model_asset_id = run.config.get("base_model_asset_id")
        base_model_asset = (
            base_model_assets.get(base_model_asset_id)
            if isinstance(base_model_asset_id, int) and not isinstance(base_model_asset_id, bool)
            else None
        )
        if base_model_asset is not None and base_model_asset.project_id == run.project_id:
            recipe_base_model = {
                "asset_id": base_model_asset.id,
                "display_name": base_model_asset.display_name,
                "provider": base_model_asset.provider,
                "source_model_id": base_model_asset.source_model_id,
                "source_identity": (
                    f"{base_model_asset.provider} / {base_model_asset.source_model_id}"
                ),
                "exact_revision": base_model_asset.exact_revision,
            }
        evaluation = latest_evaluation_by_run.get(run.id)
        creator = creators.get(run.created_by_user_id)
        payload = schemas.TrainingRunRead.model_validate(run).model_dump()
        payload.update(
            {
                "project_name": project.name,
                "project_description": project.description,
                "model_name": registered_model.name,
                "family": (model_version.family if model_version else recipe_family),
                "framework": (model_version.framework if model_version else recipe_framework),
                "base_model": (
                    {**recipe_base_model, **model_version.base_model}
                    if model_version
                    else recipe_base_model
                ),
                "training_method": (
                    model_version.training_method if model_version else recipe_training_method
                ),
                "task_name": task_definition.name,
                "task_kind": task_version.task_kind,
                "training_dataset_name": training_dataset.name,
                "dataset_version_number": dataset_version.version_number,
                "recipe_name": recipe.name,
                "runtime_name": environment.name,
                "storage_policy_name": storage_policy.name,
                "evaluation_status": evaluation.status if evaluation else None,
                "evaluation_split": (evaluation.split_name if evaluation else None),
                "evaluation_metrics": (evaluation.metrics if evaluation else {}),
                "creator_username": creator.username if creator else None,
                "creator_display_name": (creator.display_name if creator else None),
            }
        )
        items.append(payload)
    next_cursor = runs[-1].id if has_more else None
    return items, next_cursor


def list_workspace_registered_models(
    db: Session,
    *,
    workspace_id: int,
    project_id: int | None = None,
    cursor: int | None = None,
    limit: int = 50,
    status: str | None = None,
    creator_user_id: int | None = None,
    task_version_id: int | None = None,
    family: str | None = None,
) -> tuple[list[dict], int | None]:
    projects_with_modules = _workspace_projects_with_modules(
        db,
        workspace_id=workspace_id,
        project_id=project_id,
        required_modules={"learning", "models", "train"},
    )
    projects = {project.id: project for project, _modules in projects_with_modules}
    if not projects:
        return [], None

    query = db.query(models.RegisteredModel).filter(
        models.RegisteredModel.project_id.in_(list(projects))
    )
    latest_version_numbers = (
        db.query(
            models.ModelVersion.project_id.label("project_id"),
            models.ModelVersion.registered_model_id.label("registered_model_id"),
            func.max(models.ModelVersion.version_number).label("version_number"),
        )
        .filter(models.ModelVersion.project_id.in_(list(projects)))
        .group_by(
            models.ModelVersion.project_id,
            models.ModelVersion.registered_model_id,
        )
        .subquery()
    )
    if cursor is not None:
        query = query.filter(models.RegisteredModel.id < cursor)
    if status is not None:
        query = query.filter(models.RegisteredModel.lifecycle_status == status)
    if creator_user_id is not None:
        query = query.filter(models.RegisteredModel.created_by_user_id == creator_user_id)
    if task_version_id is not None or family is not None:
        query = query.join(
            latest_version_numbers,
            and_(
                latest_version_numbers.c.registered_model_id == models.RegisteredModel.id,
                latest_version_numbers.c.project_id == models.RegisteredModel.project_id,
            ),
        ).join(
            models.ModelVersion,
            and_(
                models.ModelVersion.project_id == latest_version_numbers.c.project_id,
                models.ModelVersion.registered_model_id
                == latest_version_numbers.c.registered_model_id,
                models.ModelVersion.version_number == latest_version_numbers.c.version_number,
            ),
        )
        if task_version_id is not None:
            query = query.filter(models.ModelVersion.task_version_id == task_version_id)
        if family is not None:
            query = query.filter(models.ModelVersion.family == family)
    bounded_limit = min(max(limit, 1), 100)
    registered_models = (
        query.order_by(models.RegisteredModel.id.desc()).limit(bounded_limit + 1).all()
    )
    has_more = len(registered_models) > bounded_limit
    registered_models = registered_models[:bounded_limit]
    if not registered_models:
        return [], None

    registered_model_ids = {record.id for record in registered_models}
    latest_version_records = (
        db.query(models.ModelVersion)
        .join(
            latest_version_numbers,
            and_(
                models.ModelVersion.project_id == latest_version_numbers.c.project_id,
                models.ModelVersion.registered_model_id
                == latest_version_numbers.c.registered_model_id,
                models.ModelVersion.version_number == latest_version_numbers.c.version_number,
            ),
        )
        .filter(
            models.ModelVersion.registered_model_id.in_(registered_model_ids),
            models.ModelVersion.project_id.in_(list(projects)),
        )
        .all()
    )
    registered_model_projects = {record.id: record.project_id for record in registered_models}
    latest_version_records = [
        version
        for version in latest_version_records
        if registered_model_projects.get(version.registered_model_id) == version.project_id
    ]
    latest_versions = {version.registered_model_id: version for version in latest_version_records}
    latest_versions_by_id = {version.id: version for version in latest_version_records}
    version_payloads, training_datasets = _model_version_lineage_payloads(
        db,
        latest_versions.values(),
    )
    task_versions = {
        record.id: record
        for record in db.query(models.TaskVersion)
        .filter(
            models.TaskVersion.id.in_(
                {version.task_version_id for version in latest_versions.values()}
            ),
            models.TaskVersion.project_id.in_(list(projects)),
        )
        .all()
    }
    task_definitions = {
        record.id: record
        for record in db.query(models.TaskDefinition)
        .filter(
            models.TaskDefinition.id.in_(
                {task.task_definition_id for task in task_versions.values()}
            )
        )
        .all()
    }
    latest_evaluation_by_version: dict[int, models.ModelEvaluation] = {}
    for evaluation in (
        db.query(models.ModelEvaluation)
        .filter(
            models.ModelEvaluation.model_version_id.in_(
                {version.id for version in latest_versions.values()}
            ),
            models.ModelEvaluation.project_id.in_(list(projects)),
        )
        .order_by(models.ModelEvaluation.id.desc())
        .all()
    ):
        version = latest_versions_by_id.get(evaluation.model_version_id)
        if version is not None and version.project_id == evaluation.project_id:
            latest_evaluation_by_version.setdefault(
                evaluation.model_version_id,
                evaluation,
            )
    creators = {
        record.id: record
        for record in db.query(User)
        .filter(
            User.id.in_(
                {
                    model.created_by_user_id
                    for model in registered_models
                    if model.created_by_user_id is not None
                }
            )
        )
        .all()
    }

    items = []
    for registered_model in registered_models:
        project = projects[registered_model.project_id]
        version = latest_versions.get(registered_model.id)
        if version is not None and (
            version.project_id != registered_model.project_id
            or version.registered_model_id != registered_model.id
        ):
            version = None
        task_version = task_versions.get(version.task_version_id) if version else None
        task_definition = (
            task_definitions.get(task_version.task_definition_id) if task_version else None
        )
        training_dataset = (
            training_datasets.get(version.training_dataset_version_id)
            if version and version.training_dataset_version_id is not None
            else None
        )
        evaluation = latest_evaluation_by_version.get(version.id) if version else None
        creator = creators.get(registered_model.created_by_user_id)
        payload = schemas.RegisteredModelRead.model_validate(registered_model).model_dump()
        payload.update(
            {
                "project_name": project.name,
                "project_description": project.description,
                "latest_version": (version_payloads[version.id] if version else None),
                "task_name": task_definition.name if task_definition else None,
                "task_kind": task_version.task_kind if task_version else None,
                "training_dataset_name": (training_dataset.name if training_dataset else None),
                "evaluation_status": evaluation.status if evaluation else None,
                "evaluation_split": (evaluation.split_name if evaluation else None),
                "evaluation_metrics": (evaluation.metrics if evaluation else {}),
                "creator_username": creator.username if creator else None,
                "creator_display_name": (creator.display_name if creator else None),
            }
        )
        items.append(payload)
    next_cursor = registered_models[-1].id if has_more else None
    return items, next_cursor


def transition_training_run(
    db: Session,
    project_id: int,
    training_run_id: int,
    data: schemas.TrainingRunTransition,
) -> models.TrainingRun:
    run = get_training_run(db, project_id, training_run_id)
    allowed = {
        "queued": {"running", "cancelled", "failed"},
        "running": {"succeeded", "failed", "cancelled"},
        "succeeded": set(),
        "failed": set(),
        "cancelled": set(),
    }
    if data.status not in allowed.get(run.status, set()):
        raise ConflictError(f"Cannot transition training run from {run.status} to {data.status}")
    if data.status == "succeeded":
        if data.output_model_version_id is None:
            raise ValidationError("A successful training run requires output_model_version_id")
        output = _scoped(
            db,
            models.ModelVersion,
            data.output_model_version_id,
            project_id,
            "Output model version",
        )
        if output.registered_model_id != run.registered_model_id:
            raise ValidationError("Output model version must belong to the launched model")
    now = datetime.now(UTC)
    if data.status == "running":
        artifact_quota.renew_owner_artifact_reservation(
            db,
            owner_type="training_run",
            owner_id=run.id,
        )
        run.started_at = now
    if data.status in {"succeeded", "failed", "cancelled"}:
        run.completed_at = now
    if data.status == "succeeded":
        artifact_quota.complete_owner_artifact_reservation(
            db,
            owner_type="training_run",
            owner_id=run.id,
        )
    elif data.status in {"failed", "cancelled"}:
        artifact_quota.release_owner_artifact_reservation(
            db,
            owner_type="training_run",
            owner_id=run.id,
            reason=f"Training run ended as {data.status}",
        )
    run.status = data.status
    if data.status != "cancelled":
        run.output_model_version_id = data.output_model_version_id
        run.failure_code = data.failure_code
        run.failure_reason = data.failure_reason
        if data.runtime_snapshot is not None:
            run.runtime_snapshot = data.runtime_snapshot
        if data.storage_snapshot is not None:
            run.storage_snapshot = data.storage_snapshot
    _commit(db, "Could not transition the training run")
    db.refresh(run)
    return run
