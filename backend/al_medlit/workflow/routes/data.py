"""HTTP routes for the canonical learning workflow."""

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ValidationError
from al_medlit.core.storage import ObjectStorage, get_object_storage
from al_medlit.workflow import schemas, service

from .shared import (
    _read,
    _write,
)

router = APIRouter(tags=["workflow"])




@router.post(
    "/tasks",
    response_model=schemas.TaskDefinitionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_task(
    payload: schemas.TaskDefinitionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="data")
    return service.create_task_definition(db, payload, current_user)


@router.get("/tasks", response_model=list[schemas.TaskDefinitionRead])
def list_tasks(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_task_definitions(db, project_id)


@router.post(
    "/tasks/versions",
    response_model=schemas.TaskVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_task_version(
    payload: schemas.TaskVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(db, current_user, payload.project_id, module="data")
    return service.create_task_version(db, payload, current_user)


@router.get("/tasks/versions", response_model=list[schemas.TaskVersionRead])
def list_task_versions(
    project_id: int = Query(...),
    task_definition_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_task_versions(db, project_id, task_definition_id)


@router.post(
    "/datasets",
    response_model=schemas.DatasetRead,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset(
    payload: schemas.DatasetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_dataset(db, payload, current_user)


@router.get("/datasets", response_model=list[schemas.DatasetRead])
def list_datasets(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_datasets(db, project_id)


@router.post(
    "/datasets/versions",
    response_model=schemas.DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_dataset_version(
    payload: schemas.DatasetVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_dataset_version(db, payload, current_user)


@router.post(
    "/projects/{project_id}/datasets/{dataset_id}/versions/upload",
    response_model=schemas.DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def upload_dataset_version(
    project_id: int,
    dataset_id: int,
    file: UploadFile = File(...),
    source_format: str = Form(
        default="auto",
        pattern=r"^(auto|csv|jsonl|parquet)$",
    ),
    stable_key_field: str | None = Form(default=None, min_length=1, max_length=255),
    group_key_field: str | None = Form(default=None, min_length=1, max_length=255),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_dataset_version_from_upload(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        source_format=source_format,
        file_name=file.filename,
        content_type=file.content_type,
        stream=file.file,
        stable_key_field=stable_key_field,
        group_key_field=group_key_field,
        actor=current_user,
        storage=storage,
    )


@router.post(
    "/projects/{project_id}/datasets/{dataset_id}/versions/project-corpus",
    response_model=schemas.DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def materialize_project_corpus_snapshot(
    project_id: int,
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_project_corpus_dataset_version(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        actor=current_user,
    )


@router.post(
    "/projects/{project_id}/datasets/{dataset_id}/versions/public-registry",
    response_model=schemas.DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def register_public_dataset_version(
    project_id: int,
    dataset_id: int,
    payload: schemas.PublicRegistryDatasetVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_public_registry_dataset_version(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        data=payload,
        actor=current_user,
    )


@router.post(
    "/projects/{project_id}/datasets/{dataset_id}/versions/public-registry-snapshot",
    response_model=schemas.DatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def materialize_public_dataset_snapshot(
    project_id: int,
    dataset_id: int,
    file: UploadFile = File(...),
    registry_dataset_id: str = Form(..., min_length=1, max_length=255),
    exact_revision: str = Form(..., min_length=40, max_length=40),
    source_format: str = Form(..., pattern=r"^(csv|jsonl|parquet)$"),
    config_name: str | None = Form(default=None, min_length=1, max_length=255),
    expected_content_sha256: str | None = Form(
        default=None,
        min_length=64,
        max_length=64,
    ),
    license_identifier: str | None = Form(default=None, max_length=255),
    stable_key_field: str | None = Form(default=None, min_length=1, max_length=255),
    group_key_field: str | None = Form(default=None, min_length=1, max_length=255),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    registry = schemas.PublicRegistryDatasetVersionCreate(
        provider="hugging_face",
        registry_dataset_id=registry_dataset_id,
        exact_revision=exact_revision,
        config_name=config_name,
        source_format=source_format,
        expected_content_sha256=expected_content_sha256,
        license_info=({"identifier": license_identifier} if license_identifier else {}),
    )
    return service.create_public_registry_dataset_version_from_snapshot(
        db,
        project_id=project_id,
        dataset_id=dataset_id,
        registry=registry,
        file_name=file.filename,
        content_type=file.content_type,
        stream=file.file,
        stable_key_field=stable_key_field,
        group_key_field=group_key_field,
        actor=current_user,
        storage=storage,
    )


@router.get("/datasets/versions", response_model=list[schemas.DatasetVersionRead])
def list_dataset_versions(
    project_id: int = Query(...),
    dataset_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_dataset_versions(db, project_id, dataset_id)


@router.get("/datasets/items", response_model=list[schemas.DatasetItemRead])
def list_dataset_items(
    project_id: int = Query(...),
    dataset_version_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_dataset_items(db, project_id, dataset_version_id)


@router.post(
    "/datasets/label-sets",
    response_model=schemas.LabelSetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_label_set(
    payload: schemas.LabelSetVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    if payload.source_kind == "imported":
        raise ValidationError("Imported labels must be extracted from an immutable dataset field")
    return service.create_label_set_version(
        db,
        payload,
        current_user,
        storage=storage,
    )


@router.post(
    "/datasets/label-sets/imported-field",
    response_model=schemas.LabelSetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_imported_label_set(
    payload: schemas.ImportedLabelSetFromFieldCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_imported_label_set_from_field(
        db,
        payload,
        current_user,
        storage=storage,
    )


@router.get("/datasets/label-sets", response_model=list[schemas.LabelSetVersionRead])
def list_label_sets(
    project_id: int = Query(...),
    dataset_version_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_label_set_versions(db, project_id, dataset_version_id)


@router.post(
    "/rounds/{round_id}/label-sets",
    response_model=schemas.LabelSetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_round_label_set(
    round_id: int,
    payload: schemas.RoundLabelSetCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(db, current_user, payload.project_id, module="annotate")
    return service.create_round_label_set(
        db,
        round_id,
        payload,
        current_user,
        storage=storage,
    )


@router.post(
    "/datasets/split-maps",
    response_model=schemas.SplitMapRead,
    status_code=status.HTTP_201_CREATED,
)
def create_split_map(
    payload: schemas.SplitMapCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_split_map(db, payload, current_user)


@router.get("/datasets/split-maps", response_model=list[schemas.SplitMapRead])
def list_split_maps(
    project_id: int = Query(...),
    dataset_version_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_split_maps(db, project_id, dataset_version_id)


@router.post(
    "/datasets/training-versions",
    response_model=schemas.TrainingDatasetVersionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_training_dataset(
    payload: schemas.TrainingDatasetVersionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.create_training_dataset_version(db, payload, current_user)


@router.post(
    "/datasets/training-versions/compose",
    response_model=schemas.TrainingDatasetComposeRead,
    status_code=status.HTTP_201_CREATED,
)
def compose_training_dataset(
    payload: schemas.TrainingDatasetComposeCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    _write(
        db,
        current_user,
        payload.project_id,
        min_role="trainer",
        module="data",
    )
    return service.compose_training_dataset_version(
        db,
        payload,
        current_user,
        storage=storage,
    )


@router.get(
    "/datasets/training-versions",
    response_model=list[schemas.TrainingDatasetVersionRead],
)
def list_training_datasets(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.list_training_dataset_versions(db, project_id)


@router.get(
    "/datasets/training-versions/{training_dataset_version_id}/labels",
    response_model=schemas.ComposedTrainingLabelsRead,
)
def get_composed_training_labels(
    training_dataset_version_id: int,
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _read(
        db,
        current_user,
        project_id,
        min_role="trainer",
        module="data",
    )
    return service.compose_training_dataset_labels(
        db,
        project_id,
        training_dataset_version_id,
    )
