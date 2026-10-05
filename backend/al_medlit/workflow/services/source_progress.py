"""Distinct source-item progress from finalized annotation submissions."""

from collections import defaultdict

from sqlalchemy.orm import Session

from al_medlit.workflow import models


def annotation_progress(db: Session, project_id: int) -> list[dict]:
    versions = (
        db.query(models.DatasetVersion)
        .filter(models.DatasetVersion.project_id == project_id)
        .order_by(models.DatasetVersion.dataset_id, models.DatasetVersion.version_number)
        .all()
    )
    submitted_decisions: dict[tuple[int, int], set[int]] = defaultdict(set)
    for submission in (
        db.query(models.RoundSubmission)
        .join(
            models.AnnotationRound,
            models.RoundSubmission.annotation_round_id == models.AnnotationRound.id,
        )
        .filter(
            models.RoundSubmission.project_id == project_id,
            models.AnnotationRound.project_id == project_id,
        )
        .all()
    ):
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
