import gzip
from datetime import UTC, datetime
from io import BytesIO
from uuid import uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import resource_access_clause
from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.corpus.models import Document, DocumentSentence
from al_medlit.evidence import service as evidence_service
from al_medlit.evidence.models import EvidenceTargetVersion
from al_medlit.evidence.schemas import EvidenceBlockPayloadV1
from al_medlit.inference.decoder import DecoderResult
from al_medlit.inference.models import (
    EvidenceCandidatePrediction,
    EvidencePredictionReview,
    InferenceRun,
    InferenceWindow,
)
from al_medlit.inference.schemas import (
    EvidenceCandidatePredictionRead,
    InferenceRunCreate,
    InferenceRunRead,
    InferenceRunSummaryRead,
    PredictionReviewCreate,
    PredictionReviewRead,
)
from al_medlit.lineage.models import CorpusSnapshot
from al_medlit.lineage.service import (
    add_lineage_edge,
    canonical_json_bytes,
    register_stored_artifact,
)
from al_medlit.model_artifacts.models import ArtifactPackageRetention, BaseModelAsset
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.training.compute.base import ComputeBackend, JobBundle, JobState
from al_medlit.training.models import ComputeProfile, ModelCheckpoint
from al_medlit.training.preflight import quantization_profile_ready
from al_medlit.training.windowing import (
    EvidenceBlockWindowBuilder,
    TargetCondition,
    WindowBuilderConfig,
    WindowSentenceInput,
)

OPEN_RUN_STATUSES = {"queued", "submitted", "running"}
TERMINAL_RUN_STATUSES = {"succeeded", "failed", "cancelled"}


def _find_inference_run_by_idempotency_key(
    db: Session,
    *,
    project_id: int,
    idempotency_key: str,
) -> InferenceRun | None:
    return (
        db.query(InferenceRun)
        .filter(
            InferenceRun.project_id == project_id,
            InferenceRun.idempotency_key == idempotency_key,
        )
        .one_or_none()
    )


def launch_inference_run(
    db: Session,
    *,
    project_id: int,
    data: InferenceRunCreate,
    actor_user_id: int | None,
) -> InferenceRun:
    existing = _find_inference_run_by_idempotency_key(
        db,
        project_id=project_id,
        idempotency_key=data.idempotency_key,
    )
    if existing is not None:
        return existing
    snapshot = db.get(CorpusSnapshot, data.corpus_snapshot_id)
    if snapshot is None or snapshot.project_id != project_id:
        raise NotFoundError("Corpus snapshot not found in project")
    checkpoint = db.get(ModelCheckpoint, data.checkpoint_id)
    if checkpoint is None or checkpoint.project_id != project_id:
        raise NotFoundError("Model checkpoint not found in project")
    synthetic_smoke = bool(checkpoint.manifest.get("synthetic_mode"))
    if checkpoint.readiness != "ready" and not synthetic_smoke:
        raise ValidationError("Model checkpoint is not ready for inference")
    if (
        not synthetic_smoke
        and checkpoint.package is not None
        and (checkpoint.package.readiness != "ready" or not checkpoint.package.deployable)
    ):
        raise ValidationError("Model package is not ready for inference")
    if not synthetic_smoke and checkpoint.package is not None:
        retention = (
            db.query(ArtifactPackageRetention)
            .filter(ArtifactPackageRetention.package_id == checkpoint.package.id)
            .with_for_update()
            .one_or_none()
        )
        if (
            retention is None
            or retention.archived_at is not None
            or retention.purged_at is not None
        ):
            raise ValidationError("Model package is archived or unavailable")
    profile = db.get(ComputeProfile, data.compute_profile_id)
    if profile is None or profile.project_id != project_id or profile.status != "active":
        raise ValidationError("Compute profile is not active in project")
    if checkpoint.model_type in {"evidence_lora", "evidence_qlora"}:
        resolve_peft_base_model_asset(db, checkpoint)
        if checkpoint.model_type == "evidence_qlora" and not quantization_profile_ready(profile):
            raise ValidationError(
                "QLoRA inference requires a compute profile with verified CUDA 4-bit support"
            )
    requested_targets = set(data.target_version_ids)
    trained_targets = set(checkpoint.trained_target_version_ids)
    if checkpoint.training_mode == "conditioned":
        if not requested_targets.issubset(trained_targets):
            raise ValidationError("Conditioned checkpoint was not trained for every target")
    elif requested_targets != trained_targets or len(requested_targets) != 1:
        raise ValidationError("Per-target checkpoint accepts exactly its trained target")
    for target_id in requested_targets:
        target_version = db.get(EvidenceTargetVersion, target_id)
        if target_version is None or target_version.target.project_id != project_id:
            raise ValidationError(f"Target version {target_id} is not in project")
    if data.window_config.max_tokens > checkpoint.max_context_tokens:
        raise ValidationError("Inference context exceeds checkpoint encoder capacity")

    run = InferenceRun(
        project_id=project_id,
        corpus_snapshot_id=snapshot.id,
        checkpoint_id=checkpoint.id,
        compute_profile_id=profile.id,
        name=data.name,
        target_version_ids=sorted(requested_targets),
        window_config=data.window_config.model_dump(),
        decoder_config=data.decoder_config.model_dump(),
        status="queued",
        idempotency_key=data.idempotency_key,
        created_by_user_id=actor_user_id,
    )
    try:
        with db.begin_nested():
            db.add(run)
            db.flush()
    except IntegrityError:
        # The project/idempotency-key constraint is the serialization point.
        # Keep the surrounding request transaction usable and return the run
        # committed by the concurrent winner. If no winner exists, the
        # integrity error came from a different constraint and must surface.
        existing = _find_inference_run_by_idempotency_key(
            db,
            project_id=project_id,
            idempotency_key=data.idempotency_key,
        )
        if existing is not None:
            return existing
        raise
    db.commit()
    db.refresh(run)
    return run


def resolve_peft_base_model_asset(
    db: Session,
    checkpoint: ModelCheckpoint,
) -> BaseModelAsset:
    package = checkpoint.package
    if package is None or package.retention is None:
        raise ValidationError("PEFT checkpoint is missing its immutable adapter package")
    if package.retention.purged_at is not None:
        raise ValidationError("PEFT adapter package payload has been purged")
    references = [
        reference
        for reference in package.outgoing_references
        if reference.relationship_type in {"uses_base_model", "base_model"}
    ]
    if len(references) != 1:
        raise ValidationError("PEFT adapter must reference exactly one base-model package")
    base_package = references[0].target_package
    base_retention = (
        db.query(ArtifactPackageRetention)
        .filter(ArtifactPackageRetention.package_id == base_package.id)
        .with_for_update()
        .one_or_none()
    )
    asset = (
        db.query(BaseModelAsset).filter(BaseModelAsset.package_id == base_package.id).one_or_none()
    )
    if (
        asset is None
        or asset.project_id != checkpoint.project_id
        or asset.state is None
        or asset.state.readiness != "ready"
        or asset.state.archived_at is not None
        or base_package.readiness != "ready"
        or base_retention is None
        or base_retention.archived_at is not None
        or base_retention.purged_at is not None
    ):
        raise ValidationError("The immutable PEFT base model is not ready")
    manifest_base = (
        (checkpoint.manifest.get("runtime") or {}).get("adapter_manifest", {}).get("base_model", {})
    )
    if manifest_base and (
        int(manifest_base.get("asset_id", -1)) != asset.id
        or int(manifest_base.get("package_id", -1)) != base_package.id
        or manifest_base.get("manifest_digest") != base_package.manifest_digest
        or manifest_base.get("exact_revision") != asset.exact_revision
    ):
        raise ValidationError("PEFT checkpoint base-model lineage does not match the catalog")
    return asset


def materialize_inference_windows(
    db: Session,
    *,
    run_id: int,
    token_counter,
) -> list[InferenceWindow]:
    run = get_inference_run(db, run_id)
    config = WindowBuilderConfig(
        max_tokens=run.window_config["max_tokens"],
        overlap_tokens=run.window_config["overlap_tokens"],
        target_conditioning=run.checkpoint.training_mode == "conditioned",
        require_reviewed_gold=False,
    )
    builder = EvidenceBlockWindowBuilder(token_counter, config)
    existing_keys = {window.stable_key for window in run.windows}
    created: list[InferenceWindow] = []
    for snapshot_document in run.corpus_snapshot.documents:
        document = db.get(Document, snapshot_document.document_id)
        sentence_models = (
            db.query(DocumentSentence)
            .filter(DocumentSentence.structure_version_id == snapshot_document.structure_version_id)
            .order_by(DocumentSentence.ordinal)
            .all()
        )
        sentence_inputs = [
            WindowSentenceInput(
                id=sentence.id,
                ordinal=sentence.ordinal,
                paragraph_ordinal=sentence.paragraph.ordinal,
                section_path=tuple(sentence.section.path or []),
                text=document.text[sentence.start_offset : sentence.end_offset],
                start_char=sentence.start_offset,
                end_char=sentence.end_offset,
            )
            for sentence in sentence_models
        ]
        for target_id in run.target_version_ids:
            target_version = db.get(EvidenceTargetVersion, target_id)
            result = builder.build(
                document_id=document.id,
                structure_version_id=snapshot_document.structure_version_id,
                target=TargetCondition(
                    id=target_version.id,
                    key=target_version.target.key,
                    name=target_version.target.name,
                    text=target_version.text,
                ),
                sentences=sentence_inputs,
            )
            for window in result.windows:
                if window.id in existing_keys:
                    continue
                model = InferenceWindow(
                    run_id=run.id,
                    document_id=document.id,
                    structure_version_id=snapshot_document.structure_version_id,
                    target_version_id=target_id,
                    stable_key=window.id,
                    start_sentence_ordinal=window.start_sentence_ordinal,
                    end_sentence_ordinal=window.end_sentence_ordinal,
                    token_count=window.token_count,
                    status="pending",
                )
                db.add(model)
                created.append(model)
                existing_keys.add(window.id)
    db.commit()
    for window in created:
        db.refresh(window)
    return list(run.windows)


def persist_decoder_result(
    db: Session,
    *,
    run_id: int,
    document_id: int,
    structure_version_id: int,
    target_version_id: int,
    result: DecoderResult,
    source_window_ids: list[int],
    diagnostics_artifact_id: int | None = None,
) -> list[EvidenceCandidatePrediction]:
    run = get_inference_run(db, run_id)
    if target_version_id not in run.target_version_ids:
        raise ValidationError("Target is not part of this inference run")
    snapshot_document = next(
        (item for item in run.corpus_snapshot.documents if item.document_id == document_id),
        None,
    )
    if snapshot_document is None or snapshot_document.structure_version_id != structure_version_id:
        raise ValidationError("Document structure is not pinned by this inference run")
    windows = (
        db.query(InferenceWindow)
        .filter(InferenceWindow.id.in_(source_window_ids), InferenceWindow.run_id == run.id)
        .all()
    )
    if len(windows) != len(set(source_window_ids)):
        raise ValidationError("Source windows must belong to the inference run")

    counts = {str(sentence.ordinal): sentence.contribution_count for sentence in result.sentences}
    for window in windows:
        window.sentence_contribution_counts = {
            ordinal: count
            for ordinal, count in counts.items()
            if window.start_sentence_ordinal <= int(ordinal) <= window.end_sentence_ordinal
        }
        window.status = "completed"

    candidates: list[EvidenceCandidatePrediction] = []
    for block in result.blocks:
        existing = (
            db.query(EvidenceCandidatePrediction)
            .filter(
                EvidenceCandidatePrediction.run_id == run.id,
                EvidenceCandidatePrediction.document_id == document_id,
                EvidenceCandidatePrediction.target_version_id == target_version_id,
                EvidenceCandidatePrediction.start_sentence_ordinal == block.start_ordinal,
                EvidenceCandidatePrediction.end_sentence_ordinal == block.end_ordinal,
            )
            .first()
        )
        if existing is not None:
            candidates.append(existing)
            continue
        candidate = EvidenceCandidatePrediction(
            project_id=run.project_id,
            run_id=run.id,
            checkpoint_id=run.checkpoint_id,
            document_id=document_id,
            structure_version_id=structure_version_id,
            target_version_id=target_version_id,
            start_sentence_id=block.start_sentence_id,
            end_sentence_id=block.end_sentence_id,
            start_sentence_ordinal=block.start_ordinal,
            end_sentence_ordinal=block.end_ordinal,
            start_char=block.start_char,
            end_char=block.end_char,
            block_confidence=block.confidence,
            boundary_confidence={
                "start": block.start_confidence,
                "end": block.end_confidence,
            },
            uncertainty=block.uncertainty,
            decoder_version=result.decoder_version,
            source_window_ids=sorted(set(source_window_ids)),
            diagnostics_artifact_id=diagnostics_artifact_id,
            metadata_={"sentence_ordinals": list(block.sentence_ordinals)},
        )
        db.add(candidate)
        candidates.append(candidate)
    db.commit()
    for candidate in candidates:
        db.refresh(candidate)
    return candidates


def store_inference_diagnostics(
    db: Session,
    storage: ObjectStorage,
    *,
    run_id: int,
    diagnostics: dict,
    actor_user_id: int | None,
) -> int:
    run = get_inference_run(db, run_id)
    compressed = gzip.compress(canonical_json_bytes(diagnostics), mtime=0)
    key = f"projects/{run.project_id}/inference/{run.id}/{uuid4().hex}-diagnostics.json.gz"
    stored = storage.put_stream(
        key,
        BytesIO(compressed),
        length=len(compressed),
        content_type="application/gzip",
    )
    artifact = register_stored_artifact(
        db,
        project_id=run.project_id,
        artifact_type="inference_diagnostics",
        stored=stored,
        manifest={
            "schema_version": "inference-diagnostics-v1",
            "run_id": run.id,
            "checkpoint_id": run.checkpoint_id,
            "corpus_snapshot_id": run.corpus_snapshot_id,
            "compression": "gzip",
        },
        created_by_user_id=actor_user_id,
        schema_version="inference-diagnostics-v1",
    )
    add_lineage_edge(
        db,
        upstream_artifact_id=run.checkpoint.artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="produced_diagnostics",
    )
    add_lineage_edge(
        db,
        upstream_artifact_id=run.corpus_snapshot.artifact_id,
        downstream_artifact_id=artifact.id,
        relationship_type="inferred_over",
    )
    run.diagnostics_artifact_id = artifact.id
    db.commit()
    return artifact.id


def submit_inference_run(
    db: Session,
    *,
    run_id: int,
    bundle: JobBundle,
    backend: ComputeBackend,
) -> InferenceRun:
    run = get_inference_run(db, run_id)
    if run.external_job_id is not None:
        return run
    if run.status != "queued":
        raise ConflictError(f"Inference run cannot be submitted from status '{run.status}'")
    submission = backend.submit(bundle)
    run.external_job_id = submission.external_job_id
    run.status = submission.status
    now = datetime.now(UTC)
    if submission.status == "running":
        run.started_at = now
    if submission.status in TERMINAL_RUN_STATUSES:
        run.completed_at = now
    run.metrics = {**run.metrics, "compute_submission": submission.metadata}
    db.commit()
    db.refresh(run)
    return run


def reconcile_inference_run(
    db: Session,
    *,
    run_id: int,
    backend: ComputeBackend,
) -> InferenceRun:
    run = get_inference_run(db, run_id)
    if run.external_job_id is None or run.status not in OPEN_RUN_STATUSES:
        return run
    _apply_run_state(run, backend.poll(run.external_job_id))
    db.commit()
    db.refresh(run)
    return run


def cancel_inference_run(db: Session, run_id: int) -> InferenceRun:
    run = get_inference_run(db, run_id)
    if run.status in TERMINAL_RUN_STATUSES:
        return run
    run.status = "cancelled"
    run.completed_at = datetime.now(UTC)
    db.commit()
    db.refresh(run)
    return run


def cancel_inference_run_with_backend(
    db: Session,
    *,
    run_id: int,
    backend: ComputeBackend,
) -> InferenceRun:
    run = get_inference_run(db, run_id)
    if run.status not in OPEN_RUN_STATUSES:
        return run
    state = (
        backend.cancel(run.external_job_id)
        if run.external_job_id is not None
        else JobState(status="cancelled", raw_state="CANCELLED_BEFORE_SUBMIT")
    )
    _apply_run_state(run, state)
    db.commit()
    db.refresh(run)
    return run


def get_inference_run(db: Session, run_id: int) -> InferenceRun:
    run = db.get(InferenceRun, run_id)
    if run is None:
        raise NotFoundError("Inference run not found")
    return run


def list_inference_runs(db: Session, project_id: int) -> list[InferenceRun]:
    return (
        db.query(InferenceRun)
        .filter(InferenceRun.project_id == project_id)
        .order_by(InferenceRun.created_at.desc())
        .all()
    )


def inference_run_read_for_access(
    run: InferenceRun,
    *,
    include_operational_fields: bool,
) -> InferenceRunRead | InferenceRunSummaryRead:
    if include_operational_fields:
        return InferenceRunRead.model_validate(run)
    return InferenceRunSummaryRead.model_validate(run)


def list_candidates(
    db: Session,
    *,
    run_id: int,
    user: User,
    document_id: int | None = None,
    target_version_id: int | None = None,
    status: str | None = None,
) -> list[EvidenceCandidatePrediction]:
    query = (
        db.query(EvidenceCandidatePrediction)
        .join(Project, Project.id == EvidenceCandidatePrediction.project_id)
        .join(Document, Document.id == EvidenceCandidatePrediction.document_id)
        .filter(
            EvidenceCandidatePrediction.run_id == run_id,
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=EvidenceCandidatePrediction.project_id,
                document_id=EvidenceCandidatePrediction.document_id,
                annotation_type="evidence_block",
                target_version_id=EvidenceCandidatePrediction.target_version_id,
                structure_version_id=EvidenceCandidatePrediction.structure_version_id,
            ),
        )
    )
    if document_id is not None:
        query = query.filter(EvidenceCandidatePrediction.document_id == document_id)
    if target_version_id is not None:
        query = query.filter(EvidenceCandidatePrediction.target_version_id == target_version_id)
    candidates = query.order_by(
        EvidenceCandidatePrediction.document_id,
        EvidenceCandidatePrediction.target_version_id,
        EvidenceCandidatePrediction.start_sentence_ordinal,
    ).all()
    visible: list[EvidenceCandidatePrediction] = []
    for candidate in candidates:
        assignment_id = _latest_candidate_assignment_id(db, candidate, user.id)
        candidate._review_assignment_id = assignment_id
        set_candidate_review_status(
            candidate,
            user.id,
            assignment_id=assignment_id,
        )
        if status is None or candidate.review_status == status:
            visible.append(candidate)
    return visible


def _latest_candidate_assignment_id(
    db: Session,
    candidate: EvidenceCandidatePrediction,
    reviewer_user_id: int,
) -> int | None:
    return (
        db.query(TaskAssignment.id)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == candidate.project_id,
            TaskAssignment.document_id == candidate.document_id,
            TaskAssignment.assignee_user_id == reviewer_user_id,
            TaskAssignment.target_version_id == candidate.target_version_id,
            TaskAssignment.structure_version_id == candidate.structure_version_id,
            ProjectTask.project_id == candidate.project_id,
            ProjectTask.annotation_type == "evidence_block",
            ProjectTask.enabled.is_(True),
        )
        .order_by(TaskAssignment.id.desc())
        .limit(1)
        .scalar()
    )


_REVIEW_ASSIGNMENT_UNSET = object()


def set_candidate_review_status(
    candidate: EvidenceCandidatePrediction,
    reviewer_user_id: int,
    *,
    assignment_id: int | None | object = _REVIEW_ASSIGNMENT_UNSET,
) -> EvidenceCandidatePrediction:
    """Attach the request actor's decision without conflating co-annotators."""

    if assignment_id is _REVIEW_ASSIGNMENT_UNSET:
        assignment_id = getattr(
            candidate,
            "_review_assignment_id",
            _REVIEW_ASSIGNMENT_UNSET,
        )

    current = next(
        (
            review
            for review in reversed(candidate.reviews)
            if review.reviewer_user_id == reviewer_user_id
            and (assignment_id is _REVIEW_ASSIGNMENT_UNSET or review.assignment_id == assignment_id)
        ),
        None,
    )
    candidate.review_status = (
        {
            "accept": "accepted",
            "modify": "modified",
            "reject": "rejected",
        }[current.action]
        if current is not None
        else "pending"
    )
    return candidate


def candidate_read_for_user(
    candidate: EvidenceCandidatePrediction,
    reviewer_user_id: int,
    *,
    include_all_reviews: bool,
    assignment_id: int | None | object = _REVIEW_ASSIGNMENT_UNSET,
) -> EvidenceCandidatePredictionRead:
    """Serialize a prediction without exposing co-annotators' decisions."""

    if assignment_id is _REVIEW_ASSIGNMENT_UNSET:
        assignment_id = getattr(
            candidate,
            "_review_assignment_id",
            _REVIEW_ASSIGNMENT_UNSET,
        )
    set_candidate_review_status(
        candidate,
        reviewer_user_id,
        assignment_id=assignment_id,
    )
    result = EvidenceCandidatePredictionRead.model_validate(candidate)
    if not include_all_reviews:
        result.reviews = [
            PredictionReviewRead.model_validate(review)
            for review in candidate.reviews
            if review.reviewer_user_id == reviewer_user_id
            and (assignment_id is _REVIEW_ASSIGNMENT_UNSET or review.assignment_id == assignment_id)
        ]
        # Candidate.status and review.revision are aggregate/global fields. For
        # a restricted annotator, expose only their own decision and a local
        # revision so prior co-reviewer activity cannot be inferred.
        result.status = result.review_status
        for review in result.reviews:
            review.revision = 1
    return result


def review_prediction(
    db: Session,
    *,
    prediction_id: int,
    data: PredictionReviewCreate,
    actor: User,
    required_assignment_id: int | None = None,
) -> EvidencePredictionReview:
    candidate = db.get(EvidenceCandidatePrediction, prediction_id)
    if candidate is None:
        raise NotFoundError("Evidence prediction not found")
    evidence_service._lock_evidence_scope(db, candidate.target_version_id)
    assignment = (
        db.query(TaskAssignment)
        .join(ProjectTask, ProjectTask.id == TaskAssignment.task_id)
        .filter(
            TaskAssignment.project_id == candidate.project_id,
            TaskAssignment.document_id == candidate.document_id,
            TaskAssignment.assignee_user_id == actor.id,
            TaskAssignment.target_version_id == candidate.target_version_id,
            TaskAssignment.structure_version_id == candidate.structure_version_id,
            ProjectTask.project_id == candidate.project_id,
            ProjectTask.annotation_type == "evidence_block",
            ProjectTask.enabled.is_(True),
        )
        .filter(
            TaskAssignment.id == required_assignment_id
            if required_assignment_id is not None
            else True
        )
        .order_by(TaskAssignment.id.desc())
        .populate_existing()
        .with_for_update(of=TaskAssignment)
        .first()
    )
    if assignment is None and required_assignment_id is not None:
        raise ConflictError("Evidence assignment is no longer assigned to this reviewer")
    if (
        assignment is not None
        and assignment.status not in evidence_service.MUTABLE_ASSIGNMENT_STATUSES
    ):
        raise ConflictError("Evidence assignment is finalized and read-only")
    prediction = (
        db.query(EvidenceCandidatePrediction)
        .filter(EvidenceCandidatePrediction.id == prediction_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if prediction is None:
        raise NotFoundError("Evidence prediction not found")
    existing_review = db.query(EvidencePredictionReview.id).filter(
        EvidencePredictionReview.prediction_id == prediction.id,
        EvidencePredictionReview.reviewer_user_id == actor.id,
    )
    existing_review = (
        existing_review.filter(EvidencePredictionReview.assignment_id == assignment.id)
        if assignment is not None
        else existing_review.filter(EvidencePredictionReview.assignment_id.is_(None))
    )
    if existing_review.first() is not None:
        raise ConflictError("You already reviewed this evidence prediction")

    resulting_annotation: Annotation | None = None
    selected_boundaries = None
    if data.action in {"accept", "modify"}:
        start_sentence_id = (
            data.start_sentence_id if data.action == "modify" else prediction.start_sentence_id
        )
        end_sentence_id = (
            data.end_sentence_id if data.action == "modify" else prediction.end_sentence_id
        )
        payload = EvidenceBlockPayloadV1(
            structure_version_id=prediction.structure_version_id,
            target_version_id=prediction.target_version_id,
            start_sentence_id=start_sentence_id,
            end_sentence_id=end_sentence_id,
            labels=data.labels,
            note=data.note,
        )
        resulting_annotation = evidence_service.create_evidence_block(
            db,
            {
                "project_id": prediction.project_id,
                "document_id": prediction.document_id,
                "annotator_user_id": actor.id,
                "annotator_id": actor.username,
                "guideline_version_id": (
                    assignment.guideline_version_id if assignment is not None else None
                ),
                "evidence": {
                    "prediction_id": prediction.id,
                    "inference_run_id": prediction.run_id,
                    "model_checkpoint_id": prediction.checkpoint_id,
                },
                "attributes": {"created_from_prediction": True},
            },
            payload,
            actor_user_id=actor.id,
            required_assignment_id=required_assignment_id,
            commit=False,
        )
        selected_boundaries = {
            "start_sentence_id": resulting_annotation.evidence_block.start_sentence_id,
            "end_sentence_id": resulting_annotation.evidence_block.end_sentence_id,
            "start_sentence_ordinal": (resulting_annotation.evidence_block.start_sentence_ordinal),
            "end_sentence_ordinal": resulting_annotation.evidence_block.end_sentence_ordinal,
        }

    revision = len(prediction.reviews) + 1
    review = EvidencePredictionReview(
        prediction_id=prediction.id,
        assignment_id=assignment.id if assignment is not None else None,
        guideline_version_id=(assignment.guideline_version_id if assignment is not None else None),
        reviewer_user_id=actor.id,
        action=data.action,
        revision=revision,
        resulting_annotation_id=(
            resulting_annotation.id if resulting_annotation is not None else None
        ),
        selected_boundaries=selected_boundaries,
        note=data.note,
        metadata_=data.metadata_,
    )
    prediction.status = {
        "accept": "accepted",
        "modify": "modified",
        "reject": "rejected",
    }[data.action]
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def _apply_run_state(run: InferenceRun, state: JobState) -> None:
    now = datetime.now(UTC)
    if state.status != "unknown":
        run.status = state.status
    if state.status == "running" and run.started_at is None:
        run.started_at = now
    if state.status in TERMINAL_RUN_STATUSES:
        run.completed_at = now
    if state.status == "failed":
        run.failure_reason = state.reason or state.raw_state
    run.metrics = {
        **run.metrics,
        "scheduler_state": state.raw_state,
        "exit_code": state.exit_code,
    }
