"""Shared transactional persistence for immutable feedback-set versions."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import insert
from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ConflictError
from al_medlit.workflow import models

from .common import _next_version

FEEDBACK_CANDIDATE_INSERT_BATCH_SIZE = 1000


def persist_feedback_set_version(
    db: Session,
    *,
    run: models.FeedbackRun,
    output_schema: dict,
    candidate_payloads: Iterable[dict],
    candidate_count: int,
    content_hash: str,
    artifact_package_id: int | None,
    created_by_user_id: int | None,
) -> models.FeedbackSetVersion:
    """Stage a set and its candidates in the caller's current transaction."""

    feedback_set = models.FeedbackSetVersion(
        project_id=run.project_id,
        feedback_run_id=run.id,
        dataset_version_id=run.dataset_version_id,
        task_version_id=run.task_version_id,
        version_number=_next_version(
            db,
            models.FeedbackSetVersion,
            models.FeedbackSetVersion.feedback_run_id,
            run.id,
        ),
        output_schema=output_schema,
        candidate_count=candidate_count,
        content_hash=content_hash,
        artifact_package_id=artifact_package_id,
        created_by_user_id=created_by_user_id,
    )
    db.add(feedback_set)
    db.flush()

    inserted = 0
    insert_rows: list[dict] = []
    created_at = datetime.now(UTC)
    for payload in candidate_payloads:
        insert_rows.append(
            {
                "project_id": run.project_id,
                "feedback_set_version_id": feedback_set.id,
                "created_at": created_at,
                "updated_at": created_at,
                **payload,
            }
        )
        if len(insert_rows) >= FEEDBACK_CANDIDATE_INSERT_BATCH_SIZE:
            db.execute(insert(models.FeedbackCandidate), insert_rows)
            inserted += len(insert_rows)
            insert_rows.clear()
    if insert_rows:
        db.execute(insert(models.FeedbackCandidate), insert_rows)
        inserted += len(insert_rows)
    if inserted != candidate_count:
        raise ConflictError("Feedback candidate count changed during finalization")
    return feedback_set
