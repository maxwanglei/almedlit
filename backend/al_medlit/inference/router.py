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
    InferenceRunCreate,
    InferenceRunRead,
    InferenceRunSummaryRead,
    InferenceWindowRead,
    PredictionReviewCreate,
    PredictionReviewRead,
)
from al_medlit.training.compute_profiles import (
    build_compute_backend,
    get_compute_profile,
)
from al_medlit.training.models import ModelCheckpoint
from al_medlit.workspace.capability_dependencies import enforce_capability

router = APIRouter(tags=["inference"])


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
    if checkpoint is not None and checkpoint.model_type in {
        "evidence_lora",
        "evidence_qlora",
    }:
        base_asset = service.resolve_peft_base_model_asset(db, checkpoint)
        if (
            member.role == "trainer"
            and base_asset.access_mode == "manager_only"
            and not current_user.is_superuser
        ):
            raise ForbiddenError("This adapter's base model is restricted to managers")
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
    return execute_backend_inference(db, storage, run_id=run.id)


@router.get(
    "/projects/{project_id}/inference/runs",
    response_model=list[InferenceRunRead | InferenceRunSummaryRead],
)
def list_inference_runs(
    project_id: int,
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
        for run in service.list_inference_runs(db, project_id)
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
    return service.cancel_inference_run_with_backend(
        db,
        run_id=run.id,
        backend=build_compute_backend(run.compute_profile),
    )


@router.post("/inference/runs/{run_id}/execute", response_model=InferenceRunRead)
def execute_inference_run(
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: ObjectStorage = Depends(get_object_storage),
):
    run = service.get_inference_run(db, run_id)
    _require_trainer_run_mutation(db, current_user, run)
    enforce_capability(db, project_id=run.project_id, key="inference")
    if run.compute_profile.backend == "ssh_slurm":
        enforce_capability(db, project_id=run.project_id, key="hpc_training")
    return execute_backend_inference(db, storage, run_id=run.id)


@router.get(
    "/inference/runs/{run_id}/windows",
    response_model=list[InferenceWindowRead],
)
def list_inference_windows(
    run_id: int,
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
        )
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
