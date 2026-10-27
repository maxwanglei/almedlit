from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_document_resource_access,
    assert_project_member,
    assert_task_assigned,
    has_assignment_bypass,
    lock_document_resource_for_mutation,
    lock_project_member_for_mutation,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError
from al_medlit.core.storage import ObjectStorage, get_object_storage
from al_medlit.inference import service
from al_medlit.inference.execution import execute_inference_run as execute_backend_inference
from al_medlit.inference.models import EvidenceCandidatePrediction, InferenceWindow
from al_medlit.inference.schemas import (
    EvidenceCandidatePredictionRead,
    InferenceCheckpointCreate,
    InferenceCheckpointRead,
    InferenceComputeProfileCreate,
    InferenceComputeProfileRead,
    InferenceRunCreate,
    InferenceRunRead,
    InferenceRunSummaryRead,
    InferenceWindowRead,
    PredictionReviewCreate,
    PredictionReviewRead,
)
from al_medlit.model_artifacts.service import artifact_package_access_restrictions
from al_medlit.training.compute_profiles import get_compute_profile
from al_medlit.training.models import ComputeProfile, ModelCheckpoint
from al_medlit.workspace.capability_dependencies import enforce_capability

router = APIRouter(tags=["inference"])


@router.post(
    "/projects/{project_id}/inference/compute-profiles",
    response_model=InferenceComputeProfileRead,
)
def create_inference_compute_profile(
    project_id: int,
    payload: InferenceComputeProfileCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="inference")
    if payload.backend == "ssh_slurm":
        enforce_capability(db, project_id=project_id, key="hpc_training")
    return service.create_inference_compute_profile(
        db, project_id=project_id, data=payload, actor_user_id=current_user.id,
    )


@router.get(
    "/projects/{project_id}/inference/compute-profiles",
    response_model=list[InferenceComputeProfileRead],
)
def list_inference_compute_profiles(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="inference")
    return db.query(ComputeProfile).filter(
        ComputeProfile.project_id == project_id,
    ).order_by(ComputeProfile.id).all()


@router.post(
    "/projects/{project_id}/inference/checkpoints",
    response_model=InferenceCheckpointRead,
)
def register_inference_checkpoint(
    project_id: int,
    payload: InferenceCheckpointCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    lock_project_member_for_mutation(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="inference")
    return service.register_inference_checkpoint(db, project_id=project_id, data=payload)


@router.get(
    "/projects/{project_id}/inference/checkpoints",
    response_model=list[InferenceCheckpointRead],
)
def list_inference_checkpoints(
    project_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    assert_project_member(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="inference")
    return db.query(ModelCheckpoint).filter(
        ModelCheckpoint.project_id == project_id,
    ).order_by(ModelCheckpoint.id).all()


def _can_view_operational_run_fields(user: User, role: str) -> bool:
    return user.is_superuser or role in {"trainer", "manager", "admin"}


def _require_trainer_run_mutation(
    db: Session,
    current_user: User,
    run,
):
    member = lock_project_member_for_mutation(
        db,
        current_user,
        run.project_id,
        min_role="trainer",
    )
    if (
        member.role == "trainer"
        and run.created_by_user_id != current_user.id
        and not current_user.is_superuser
    ):
        raise ForbiddenError("Trainers may mutate only their own inference runs")
    return member


def _require_checkpoint_execution_access(
    db: Session,
    current_user: User,
    member,
    checkpoint: ModelCheckpoint | None,
    *,
    project_id: int,
) -> None:
    if checkpoint is None or checkpoint.project_id != project_id:
        raise NotFoundError("Model checkpoint not found in project")
    is_manager = current_user.is_superuser or member.role in {"manager", "admin"}
    if (
        checkpoint.package is not None
        and artifact_package_access_restrictions(db, checkpoint.package).requires_manager
        and not is_manager
    ):
        raise ForbiddenError("This model checkpoint is restricted to managers")
    if checkpoint.model_type in {"evidence_lora", "evidence_qlora"}:
        base_asset = service.resolve_peft_base_model_asset(db, checkpoint)
        if (
            artifact_package_access_restrictions(db, base_asset.package).requires_manager
            and not is_manager
        ):
            raise ForbiddenError("This adapter's base model is restricted to managers")


@router.post(
    "/projects/{project_id}/inference/runs",
    response_model=InferenceRunRead,
)
def launch_inference_run(
    project_id: int,
    payload: InferenceRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    member = lock_project_member_for_mutation(
        db,
        current_user,
        project_id,
        min_role="trainer",
    )
    enforce_capability(db, project_id=project_id, key="inference")
    profile = get_compute_profile(db, payload.compute_profile_id)
    if profile.project_id == project_id and profile.backend == "ssh_slurm":
        enforce_capability(db, project_id=project_id, key="hpc_training")
    checkpoint = db.get(ModelCheckpoint, payload.checkpoint_id)
    _require_checkpoint_execution_access(
        db, current_user, member, checkpoint, project_id=project_id,
    )
    run = service.launch_inference_run(
        db,
        project_id=project_id,
        data=payload,
        actor_user_id=current_user.id,
    )
    if (
        member.role == "trainer"
        and run.created_by_user_id != current_user.id
        and not current_user.is_superuser
    ):
        # Project-wide idempotency keys must not let one trainer resume or
        # execute another trainer's existing run by guessing its key.
        raise ForbiddenError("Trainers may execute only their own inference runs")
    _require_checkpoint_execution_access(
        db, current_user, member, run.checkpoint, project_id=project_id,
    )
    return execute_backend_inference(db, storage, run_id=run.id)


@router.get(
    "/projects/{project_id}/inference/runs",
    response_model=list[InferenceRunRead | InferenceRunSummaryRead],
)
def list_inference_runs(
    project_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Project members need run IDs to discover prediction candidates in their
    # assigned document/target scope. Candidate reads remain assignment-filtered.
    member = assert_project_member(db, current_user, project_id)
    enforce_capability(db, project_id=project_id, key="inference")
    return [
        service.inference_run_read_for_access(
            run,
            include_operational_fields=_can_view_operational_run_fields(
                current_user,
                member.role,
            ),
        )
        for run in service.list_inference_runs(db, project_id, limit=limit, offset=offset)
    ]


@router.get("/inference/runs/{run_id}", response_model=InferenceRunRead)
def get_inference_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = service.get_inference_run(db, run_id)
    assert_project_member(db, current_user, run.project_id, min_role="trainer")
    enforce_capability(db, project_id=run.project_id, key="inference")
    return run


@router.post("/inference/runs/{run_id}/cancel", response_model=InferenceRunRead)
def cancel_inference_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = service.get_inference_run(db, run_id)
    _require_trainer_run_mutation(db, current_user, run)
    enforce_capability(db, project_id=run.project_id, key="inference")
    if run.compute_profile.backend == "ssh_slurm":
        enforce_capability(db, project_id=run.project_id, key="hpc_training")
    return service.cancel_inference_run(db, run.id)


@router.post("/inference/runs/{run_id}/execute", response_model=InferenceRunRead)
def execute_inference_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    run = service.get_inference_run(db, run_id)
    member = _require_trainer_run_mutation(db, current_user, run)
    enforce_capability(db, project_id=run.project_id, key="inference")
    if run.compute_profile.backend == "ssh_slurm":
        enforce_capability(db, project_id=run.project_id, key="hpc_training")
    _require_checkpoint_execution_access(
        db, current_user, member, run.checkpoint, project_id=run.project_id,
    )
    return execute_backend_inference(db, storage, run_id=run.id)


@router.get(
    "/inference/runs/{run_id}/windows",
    response_model=list[InferenceWindowRead],
)
def list_inference_windows(
    run_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = service.get_inference_run(db, run_id)
    assert_project_member(db, current_user, run.project_id, min_role="trainer")
    enforce_capability(db, project_id=run.project_id, key="inference")
    return (
        db.query(InferenceWindow)
        .filter(InferenceWindow.run_id == run.id)
        .order_by(
            InferenceWindow.document_id,
            InferenceWindow.target_version_id,
            InferenceWindow.start_sentence_ordinal,
            InferenceWindow.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )


@router.get(
    "/inference/runs/{run_id}/predictions",
    response_model=list[EvidenceCandidatePredictionRead],
)
def list_predictions(
    run_id: int,
    document_id: int | None = Query(None),
    target_version_id: int | None = Query(None),
    status: str | None = Query(None, pattern="^(pending|accepted|modified|rejected)$"),
    limit: int = Query(
        service.DEFAULT_CANDIDATE_PAGE_SIZE,
        ge=1,
        le=service.MAX_CANDIDATE_PAGE_SIZE,
    ),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = service.get_inference_run(db, run_id)
    member = assert_project_member(db, current_user, run.project_id)
    enforce_capability(db, project_id=run.project_id, key="inference")
    candidates = service.list_candidates(
        db,
        run_id=run.id,
        user=current_user,
        document_id=document_id,
        target_version_id=target_version_id,
        status=status,
        limit=limit,
        offset=offset,
    )
    return [
        service.candidate_read_for_user(
            candidate,
            current_user.id,
            include_all_reviews=has_assignment_bypass(current_user, member),
        )
        for candidate in candidates
    ]


@router.get(
    "/inference/predictions/{prediction_id}",
    response_model=EvidenceCandidatePredictionRead,
)
def get_prediction(
    prediction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prediction = db.get(EvidenceCandidatePrediction, prediction_id)
    if prediction is None:
        raise NotFoundError("Evidence prediction not found")
    member = assert_document_resource_access(
        db,
        current_user,
        project_id=prediction.project_id,
        document_id=prediction.document_id,
    )
    service.assert_candidate_visible(db, current_user, member, prediction)
    assignment = assert_task_assigned(
        db,
        current_user,
        member,
        project_id=prediction.project_id,
        document_id=prediction.document_id,
        annotation_type="evidence_block",
        target_version_id=prediction.target_version_id,
        structure_version_id=prediction.structure_version_id,
    )
    enforce_capability(db, project_id=prediction.project_id, key="inference")
    return service.candidate_read_for_user(
        prediction,
        current_user.id,
        include_all_reviews=has_assignment_bypass(current_user, member),
        assignment_id=assignment.id if assignment is not None else None,
    )


@router.post(
    "/inference/predictions/{prediction_id}/review",
    response_model=PredictionReviewRead,
)
def review_prediction(
    prediction_id: int,
    payload: PredictionReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prediction = db.get(EvidenceCandidatePrediction, prediction_id)
    if prediction is None:
        raise NotFoundError("Evidence prediction not found")
    member = lock_document_resource_for_mutation(
        db,
        current_user,
        project_id=prediction.project_id,
        document_id=prediction.document_id,
        lock_assignment=False,
    )
    service.assert_candidate_visible(db, current_user, member, prediction)
    assignment = assert_task_assigned(
        db,
        current_user,
        member,
        project_id=prediction.project_id,
        document_id=prediction.document_id,
        annotation_type="evidence_block",
        target_version_id=prediction.target_version_id,
        structure_version_id=prediction.structure_version_id,
        require_mutable_assignment=True,
    )
    enforce_capability(db, project_id=prediction.project_id, key="inference")
    return service.review_prediction(
        db,
        prediction_id=prediction.id,
        data=payload,
        actor=current_user,
        required_assignment_id=(
            assignment.id
            if assignment is not None and not has_assignment_bypass(current_user, member)
            else None
        ),
    )
