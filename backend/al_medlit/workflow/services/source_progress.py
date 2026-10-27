"""Distinct source-item progress from finalized annotation submissions."""

from collections import defaultdict

from sqlalchemy.orm import Session

from al_medlit.workflow import models
from al_medlit.workflow.services.common import _scoped


def annotation_progress(
    db: Session, project_id: int, task_version_id: int | None = None,
) -> list[dict]:
    if task_version_id is not None:
        _scoped(db, models.TaskVersion, task_version_id, project_id, "Task version")
    versions = (
        db.query(models.DatasetVersion)
        .filter(models.DatasetVersion.project_id == project_id)
        .order_by(models.DatasetVersion.dataset_id, models.DatasetVersion.version_number)
        .all()
    )
    submitted_decisions: dict[tuple[int, int], set[int]] = defaultdict(set)
    submissions = (
        db.query(models.RoundSubmission)
        .join(
            models.AnnotationRound,
            models.RoundSubmission.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(
            models.RoundSubmission.project_id == project_id,
            models.AnnotationRound.project_id == project_id,
            models.AnnotationRound.status.in_(("open", "closed")),
        )
    )
    if task_version_id is not None:
        submissions = submissions.filter(models.AnnotationRound.task_version_id == task_version_id)
    for submission in submissions:
        submitted_decisions[(submission.annotation_round_id, submission.annotator_user_id)].update(
            submission.decision_ids
        )

    items_by_version: dict[int, set[int]] = defaultdict(set)
    if submitted_decisions:
        decisions = (
            db.query(
                models.RoundAnnotationDecision.id,
                models.RoundAnnotationDecision.annotator_user_id,
                models.RoundItem.annotation_round_id,
                models.DatasetItem.id,
                models.DatasetItem.dataset_version_id,
            )
            .join(
                models.RoundItem,
                models.RoundAnnotationDecision.round_item_id == models.RoundItem.id,
            )
            .join(models.DatasetItem, models.RoundItem.dataset_item_id == models.DatasetItem.id)
            .filter(
                models.RoundAnnotationDecision.project_id == project_id,
                models.RoundItem.project_id == project_id,
                models.DatasetItem.project_id == project_id,
                models.RoundAnnotationDecision.decision_kind.in_(
                    ("annotation", "correction", "adjudication")
                ),
            )
            .all()
        )
        for decision_id, actor_id, round_id, item_id, version_id in decisions:
            if decision_id in submitted_decisions.get((round_id, actor_id), ()):
                items_by_version[version_id].add(item_id)
    return [
        {
            "dataset_id": version.dataset_id,
            "dataset_version_id": version.id,
            "total": version.item_count,
            "submitted": len(items_by_version[version.id]),
        }
        for version in versions
    ]
