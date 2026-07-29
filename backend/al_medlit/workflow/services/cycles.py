"""Domain operations for the canonical learning workflow."""

import re

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ConflictError,
    ValidationError,
)
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_sequence,
    _next_version,
    _optional_package,
    _project,
    _require_actor_role,
    _scoped,
    _validate_task_output,
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




def create_learning_cycle(
    db: Session, data: schemas.LearningCycleCreate, actor: User
) -> models.LearningCycle:
    task = _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    dataset = _scoped(
        db,
        models.DatasetVersion,
        data.source_dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    if data.parent_cycle_id is not None:
        parent = _scoped(
            db,
            models.LearningCycle,
            data.parent_cycle_id,
            data.project_id,
            "Parent learning cycle",
        )
        parent_task = _scoped(
            db,
            models.TaskVersion,
            parent.task_version_id,
            data.project_id,
            "Parent task version",
        )
        parent_dataset = _scoped(
            db,
            models.DatasetVersion,
            parent.source_dataset_version_id,
            data.project_id,
            "Parent dataset version",
        )
        if (
            parent_task.task_definition_id != task.task_definition_id
            or parent_dataset.dataset_id != dataset.dataset_id
        ):
            raise ValidationError(
                "Parent cycle must use versions of the same task and dataset identities"
            )
    if data.baseline_model_version_id is not None:
        baseline = _scoped(
            db,
            models.ModelVersion,
            data.baseline_model_version_id,
            data.project_id,
            "Baseline model version",
        )
        if baseline.task_version_id != task.id:
            raise ValidationError("Baseline model must use the cycle task version")
    cycle = models.LearningCycle(
        project_id=data.project_id,
        parent_cycle_id=data.parent_cycle_id,
        name=data.name,
        sequence=_next_sequence(db, models.LearningCycle, data.project_id),
        goal=data.goal,
        task_version_id=task.id,
        source_dataset_version_id=dataset.id,
        baseline_model_version_id=data.baseline_model_version_id,
        feedback_sources=[source.model_dump(mode="json") for source in data.feedback_sources],
        metadata_=data.metadata,
        created_by_user_id=actor.id,
    )
    db.add(cycle)
    _commit(db, "Could not create the learning cycle")
    db.refresh(cycle)
    return cycle


def list_learning_cycles(db: Session, project_id: int) -> list[models.LearningCycle]:
    _project(db, project_id)
    return (
        db.query(models.LearningCycle)
        .filter(models.LearningCycle.project_id == project_id)
        .order_by(models.LearningCycle.sequence)
        .all()
    )


_CYCLE_TRANSITIONS = {
    "planned": {"active", "cancelled"},
    "active": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}
_CYCLE_STAGES = (
    "pool",
    "select",
    "annotate",
    "resolve",
    "reflect",
    "publish_guideline",
    "train",
    "evaluate",
    "complete",
)


def transition_learning_cycle(
    db: Session,
    project_id: int,
    cycle_id: int,
    data: schemas.CycleTransition,
) -> models.LearningCycle:
    cycle = _scoped(db, models.LearningCycle, cycle_id, project_id, "Learning cycle")
    if data.status != cycle.status and data.status not in _CYCLE_TRANSITIONS.get(
        cycle.status, set()
    ):
        raise ConflictError(
            f"Cannot transition learning cycle from {cycle.status} to {data.status}"
        )
    current_stage_index = _CYCLE_STAGES.index(cycle.current_stage)
    next_stage_index = _CYCLE_STAGES.index(data.current_stage)
    if data.status == "cancelled":
        if data.current_stage != cycle.current_stage:
            raise ValidationError("Cancelling a cycle cannot change its current stage")
    elif data.status == "completed":
        if data.current_stage != "complete" or cycle.current_stage != "evaluate":
            raise ValidationError("A cycle may complete only from evaluate into the complete stage")
    elif next_stage_index not in {current_stage_index, current_stage_index + 1}:
        raise ValidationError("Learning cycle stages must advance one canonical step at a time")
    if data.status != "completed" and data.current_stage == "complete":
        raise ValidationError("The complete stage requires completed cycle status")
    if cycle.status == "planned" and data.status == "planned" and next_stage_index != 0:
        raise ValidationError("A planned cycle must remain at the pool stage")
    if data.output_training_dataset_version_id is not None:
        output_training_dataset = _scoped(
            db,
            models.TrainingDatasetVersion,
            data.output_training_dataset_version_id,
            project_id,
            "Output training dataset version",
        )
        if (
            output_training_dataset.task_version_id != cycle.task_version_id
            or output_training_dataset.dataset_version_id != cycle.source_dataset_version_id
        ):
            raise ValidationError(
                "Cycle output training data must use the pinned task and dataset versions"
            )
    if data.output_model_version_id is not None:
        output_model = _scoped(
            db,
            models.ModelVersion,
            data.output_model_version_id,
            project_id,
            "Output model version",
        )
        if output_model.task_version_id != cycle.task_version_id:
            raise ValidationError("Cycle output model must use the pinned task version")
        if (
            output_model.training_dataset_version_id is not None
            and data.output_training_dataset_version_id is not None
            and output_model.training_dataset_version_id != data.output_training_dataset_version_id
        ):
            raise ValidationError(
                "Cycle output model must use the recorded output training dataset"
            )
    if data.status != "completed" and (
        data.output_training_dataset_version_id is not None
        or data.output_model_version_id is not None
    ):
        raise ValidationError("Cycle outputs may be attached only on completion")
    if data.status == "completed" and (
        data.output_training_dataset_version_id is None and data.output_model_version_id is None
    ):
        raise ValidationError("A completed learning cycle must record at least one output")
    cycle.status = data.status
    cycle.current_stage = data.current_stage
    cycle.output_training_dataset_version_id = data.output_training_dataset_version_id
    cycle.output_model_version_id = data.output_model_version_id
    _commit(db, "Could not transition the learning cycle")
    db.refresh(cycle)
    return cycle


def _cycle_declares_feedback_source(
    cycle: models.LearningCycle,
    data: schemas.FeedbackRunCreate,
) -> bool:
    if data.producer_type == "registered_model":
        return (
            data.model_version_id is not None
            and cycle.baseline_model_version_id == data.model_version_id
        )
    for source in cycle.feedback_sources or []:
        if source.get("producer_type") != data.producer_type:
            continue
        if data.producer_type != "external_llm":
            return True
        if (
            source.get("provider") == data.provider
            and source.get("external_model_id") == data.external_model_id
            and source.get("exact_revision") == data.exact_revision
        ):
            return True
    return False


def create_feedback_run(
    db: Session, data: schemas.FeedbackRunCreate, actor: User
) -> models.FeedbackRun:
    if data.producer_type == "external_llm":
        _require_actor_role(
            db,
            project_id=data.project_id,
            actor=actor,
            minimum_role="manager",
        )
    dataset = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    task = _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    if data.cycle_id is not None:
        cycle = _scoped(db, models.LearningCycle, data.cycle_id, data.project_id, "Learning cycle")
        if cycle.source_dataset_version_id != dataset.id or cycle.task_version_id != task.id:
            raise ValidationError("Feedback run must match its learning cycle task and dataset")
        if not _cycle_declares_feedback_source(cycle, data):
            raise ValidationError("Feedback run producer is not declared by its learning cycle")
    if data.model_version_id is not None:
        model_version = _scoped(
            db,
            models.ModelVersion,
            data.model_version_id,
            data.project_id,
            "Model version",
        )
        if model_version.task_version_id != task.id:
            raise ValidationError("Feedback model must use the feedback task version")
    run = models.FeedbackRun(**data.model_dump(), created_by_user_id=actor.id)
    db.add(run)
    _commit(db, "Could not create the feedback run")
    db.refresh(run)
    return run


def list_feedback_runs(db: Session, project_id: int) -> list[models.FeedbackRun]:
    _project(db, project_id)
    return (
        db.query(models.FeedbackRun)
        .filter(models.FeedbackRun.project_id == project_id)
        .order_by(models.FeedbackRun.created_at.desc(), models.FeedbackRun.id.desc())
        .all()
    )


def create_feedback_set(
    db: Session, data: schemas.FeedbackSetCreate, actor: User
) -> models.FeedbackSetVersion:
    run = _scoped(db, models.FeedbackRun, data.feedback_run_id, data.project_id, "Feedback run")
    task_version = _scoped(
        db,
        models.TaskVersion,
        run.task_version_id,
        data.project_id,
        "Task version",
    )
    db.query(models.FeedbackRun).filter(models.FeedbackRun.id == run.id).with_for_update().one()
    _optional_package(db, data.project_id, data.artifact_package_id)
    seen: set[tuple[int, str]] = set()
    candidate_payloads = []
    for candidate in data.candidates:
        item = _scoped(
            db,
            models.DatasetItem,
            candidate.dataset_item_id,
            data.project_id,
            "Dataset item",
        )
        if item.dataset_version_id != run.dataset_version_id:
            raise ValidationError("Feedback candidates must belong to the run dataset version")
        identity = (candidate.dataset_item_id, candidate.candidate_key)
        if identity in seen:
            raise ValidationError("Feedback candidate item/key pairs must be unique")
        seen.add(identity)
        _validate_task_output(
            task_version,
            candidate.output,
            label=f"Feedback candidate for dataset item {item.stable_key!r}",
        )
        payload = candidate.model_dump()
        payload["content_hash"] = _canonical_hash(payload)
        candidate_payloads.append(payload)
    content = {
        "feedback_run_id": run.id,
        "output_schema": data.output_schema,
        "candidates": candidate_payloads,
    }
    feedback_set = models.FeedbackSetVersion(
        project_id=data.project_id,
        feedback_run_id=run.id,
        dataset_version_id=run.dataset_version_id,
        task_version_id=run.task_version_id,
        version_number=_next_version(
            db,
            models.FeedbackSetVersion,
            models.FeedbackSetVersion.feedback_run_id,
            run.id,
        ),
        output_schema=data.output_schema,
        candidate_count=len(candidate_payloads),
        content_hash=_canonical_hash(content),
        artifact_package_id=data.artifact_package_id,
        created_by_user_id=actor.id,
    )
    db.add(feedback_set)
    db.flush()
    for payload in candidate_payloads:
        db.add(
            models.FeedbackCandidate(
                project_id=data.project_id,
                feedback_set_version_id=feedback_set.id,
                **payload,
            )
        )
    run.status = "completed"
    _commit(db, "Could not finalize the feedback set")
    db.refresh(feedback_set)
    return feedback_set


def list_feedback_sets(
    db: Session, project_id: int, feedback_run_id: int | None = None
) -> list[models.FeedbackSetVersion]:
    _project(db, project_id)
    query = db.query(models.FeedbackSetVersion).filter(
        models.FeedbackSetVersion.project_id == project_id
    )
    if feedback_run_id is not None:
        _scoped(db, models.FeedbackRun, feedback_run_id, project_id, "Feedback run")
        query = query.filter(models.FeedbackSetVersion.feedback_run_id == feedback_run_id)
    return query.order_by(
        models.FeedbackSetVersion.created_at.desc(),
        models.FeedbackSetVersion.id.desc(),
    ).all()


def list_feedback_candidates(
    db: Session, project_id: int, feedback_set_version_id: int
) -> list[models.FeedbackCandidate]:
    _scoped(
        db,
        models.FeedbackSetVersion,
        feedback_set_version_id,
        project_id,
        "Feedback set version",
    )
    return (
        db.query(models.FeedbackCandidate)
        .filter(models.FeedbackCandidate.feedback_set_version_id == feedback_set_version_id)
        .order_by(models.FeedbackCandidate.dataset_item_id, models.FeedbackCandidate.id)
        .all()
    )


def create_selection_run(
    db: Session, data: schemas.SelectionRunCreate, actor: User
) -> models.SelectionRun:
    dataset = _scoped(
        db,
        models.DatasetVersion,
        data.dataset_version_id,
        data.project_id,
        "Dataset version",
    )
    task = _scoped(db, models.TaskVersion, data.task_version_id, data.project_id, "Task version")
    if data.cycle_id is not None:
        cycle = _scoped(db, models.LearningCycle, data.cycle_id, data.project_id, "Learning cycle")
        if cycle.source_dataset_version_id != dataset.id or cycle.task_version_id != task.id:
            raise ValidationError("Selection run must match its learning cycle task and dataset")
    if data.feedback_set_version_id is not None:
        feedback_set = _scoped(
            db,
            models.FeedbackSetVersion,
            data.feedback_set_version_id,
            data.project_id,
            "Feedback set version",
        )
        if feedback_set.dataset_version_id != dataset.id or feedback_set.task_version_id != task.id:
            raise ValidationError("Selection feedback must use the selected task and dataset")
    if data.split_map_id is None:
        raise ValidationError("Active-learning selection requires a governed split map")
    split_map = _scoped(db, models.SplitMap, data.split_map_id, data.project_id, "Split map")
    if split_map.dataset_version_id != dataset.id:
        raise ValidationError("Selection split map must use the selected dataset")
    run = models.SelectionRun(**data.model_dump(), created_by_user_id=actor.id)
    db.add(run)
    _commit(db, "Could not create the selection run")
    db.refresh(run)
    return run


def list_selection_runs(db: Session, project_id: int) -> list[models.SelectionRun]:
    _project(db, project_id)
    return (
        db.query(models.SelectionRun)
        .filter(models.SelectionRun.project_id == project_id)
        .order_by(models.SelectionRun.created_at.desc(), models.SelectionRun.id.desc())
        .all()
    )
