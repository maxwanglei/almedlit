"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError
from al_medlit.training.recipe_contracts import RecipeConfigurationValidation
from al_medlit.training.recipe_registry import training_recipes
from al_medlit.training.tasks import enqueue_training_run
from al_medlit.workflow import schemas, service

from .shared import (
    _read,
    _write,
)

router = APIRouter(tags=["workflow"])




@router.get("/training/recipes")
def list_trusted_training_recipes(
    task_kind: str | None = None,
    _current_user: User = Depends(get_current_user),
):
    descriptors = training_recipes.list()
    if task_kind is not None:
        descriptors = tuple(
            descriptor
            for descriptor in descriptors
            if task_kind in {kind.value for kind in descriptor.supported_task_kinds}
        )
    return [descriptor.model_dump(mode="json") for descriptor in descriptors]


@router.post(
    "/training/recipes/{recipe_key}/validate-configuration",
    response_model=RecipeConfigurationValidation,
)
def validate_training_recipe_configuration(
    recipe_key: str,
    payload: schemas.TrainingRecipeConfigurationCreate,
    _current_user: User = Depends(get_current_user),
):
    return training_recipes.validate(recipe_key, payload.config)

@router.post(
    "/training-recipes",
    response_model=schemas.TrainingRecipeRead,
    status_code=status.HTTP_201_CREATED,
)
def create_training_recipe(
    payload: schemas.TrainingRecipeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="train")
    return service.create_training_recipe(db, payload, current_user)


@router.get("/training-recipes", response_model=list[schemas.TrainingRecipeRead])
def list_project_training_recipes(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("train", "activity"),
    )
    return service.list_training_recipes(db, project_id)


@router.post(
    "/training-recipes/versions",
    response_model=schemas.TrainingRecipeVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_training_recipe_version(
    payload: schemas.TrainingRecipeVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="train")
    return service.create_training_recipe_version(db, payload, current_user)


@router.get(
    "/training-recipes/versions",
    response_model=list[schemas.TrainingRecipeVersionRead],
)
def list_training_recipe_versions(
    project_id: int = Query(...),
    training_recipe_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("train", "activity"),
    )
    return service.list_training_recipe_versions(db, project_id, training_recipe_id)


@router.post(
    "/training-recipes/trusted/{recipe_key}",
    response_model=schemas.TrainingRecipeVersionRead,
)
def bind_trusted_training_recipe(
    recipe_key: str,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="train",
    )
    return service.ensure_trusted_training_recipe_version(
        db,
        project_id,
        recipe_key,
        current_user,
    )


@router.post(
    "/environments",
    response_model=schemas.ExecutionEnvironmentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_environment(
    payload: schemas.ExecutionEnvironmentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="admin",
        module="train",
    )
    return service.create_execution_environment(db, payload, current_user)


@router.get("/environments", response_model=list[schemas.ExecutionEnvironmentRead])
def list_environments(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="train",
    )
    return service.list_execution_environments(db, project_id)


@router.post(
    "/environments/{environment_id}/verification",
    response_model=schemas.ExecutionEnvironmentRead,
)
def verify_environment(
    environment_id: int,
    payload: schemas.EnvironmentVerification,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="admin",
        module="train",
    )
    return service.verify_execution_environment(db, project_id, environment_id, payload)


@router.post(
    "/storage/policies",
    response_model=schemas.StoragePolicyRead,
    status_code=status.HTTP_201_CREATED,
)
def create_storage_policy(
    payload: schemas.StoragePolicyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="admin",
        module="train",
    )
    return service.create_storage_policy(db, payload, current_user)


@router.get("/storage/policies", response_model=list[schemas.StoragePolicyRead])
def list_storage_policies(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="train",
    )
    return service.list_storage_policies(db, project_id)


@router.post(
    "/training-runs",
    response_model=schemas.TrainingRunRead,
    status_code=status.HTTP_201_CREATED,
)
def create_training_run(
    payload: schemas.TrainingRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="train",
    )
    training_run = service.create_training_run(db, payload, current_user)
    if training_run.status == "queued":
        enqueue_training_run(
            training_run.id,
            environment_class=training_run.runtime_snapshot["environment_class"],
        )
    return training_run


@router.get("/training-runs", response_model=list[schemas.TrainingRunRead])
def list_training_runs(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("train", "activity"),
    )
    return service.list_training_runs(db, project_id)


@router.get("/training-runs/{training_run_id}", response_model=schemas.TrainingRunRead)
def get_training_run(
    training_run_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("train", "activity"),
    )
    return service.get_training_run(db, project_id, training_run_id)


@router.get(
    "/training-runs/{training_run_id}/evaluations",
    response_model=list[schemas.ModelEvaluationRead],
)
def list_training_run_evaluations(
    training_run_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("train", "activity"),
    )
    return service.list_training_run_evaluations(
        db,
        project_id,
        training_run_id,
    )


@router.get(
    "/evaluations/{evaluation_id}",
    response_model=schemas.ModelEvaluationRead,
)
def get_model_evaluation(
    evaluation_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module=("models", "train", "learning"),
    )
    return service.get_model_evaluation(db, project_id, evaluation_id)


@router.post(
    "/training-runs/{training_run_id}/transition",
    response_model=schemas.TrainingRunRead,
)
def transition_training_run(
    training_run_id: int,
    payload: schemas.TrainingRunCancellation,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="train",
    )
    training_run = service.get_training_run(
        db,
        project_id,
        training_run_id,
    )
    if training_run.created_by_user_id != current_user.id and not current_user.is_superuser:
        raise ForbiddenError("Users may cancel only their own training runs")
    return service.transition_training_run(
        db,
        project_id,
        training_run_id,
        schemas.TrainingRunTransition(status="cancelled"),
    )
