"""Prediction endpoints with explicit inference access, independent of AL."""

import csv
import io
import json
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ConflictError
from al_medlit.project.models import Project
from al_medlit.training.tasks import enqueue_feedback_scoring
from al_medlit.workflow import access, models
from al_medlit.workflow import prediction_schemas as schemas
from al_medlit.workflow.services import predictions as service
from al_medlit.workflow.services.feedback_scoring import request_feedback_run_materialization

router = APIRouter(tags=["predictions"])


def _authorize(
    db: Session, user: User, project_id: int, *, write: bool = False, review: bool = False
) -> None:
    authorize = access.authorize_project_write if write else access.authorize_project_read
    authorize(db, user, project_id, min_role="manager" if review else "trainer", module="data")
    # Tuples mean ANY module; these must both be present.
    access.enforce_project_module(db, project_id, "models")
    project = db.get(Project, project_id)
    access.authorize_workspace_capability(
        db, user, project.workspace_id, capability="inference", min_role="trainer"
    )
    if review:
        access.enforce_project_module(db, project_id, "annotate")


def _dispatch(db: Session, user: User, project_id: int, run_id: int) -> dict:
    dispatch = request_feedback_run_materialization(
        db, project_id=project_id, feedback_run_id=run_id, actor=user
    )
    if dispatch.should_enqueue:
        try:
            enqueue_feedback_scoring(dispatch.run.id)
        except Exception:
            # The durable queued run can safely be redispatched using Retry.
            db.expire_all()
            db.query(models.FeedbackRun).filter(
                models.FeedbackRun.id == run_id, models.FeedbackRun.status == "queued"
            ).update(
                {
                    "status": "failed",
                    "failure_code": "dispatch_failed",
                    "failure_reason": "The worker could not be reached. Retry this prediction run.",
                },
                synchronize_session=False,
            )
            db.commit()
        db.expire_all()
    return service.prediction_read(db, service.get_prediction_run(db, project_id, run_id))


@router.post(
    "/projects/{project_id}/prediction-runs",
    response_model=schemas.PredictionRunRead,
    status_code=202,
)
def create_prediction_run(
    project_id: int,
    payload: schemas.PredictionRunCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id, write=True)
    run = service.create_prediction_run(db, project_id, payload, current_user)
    if run.status != "planned":
        return service.prediction_read(db, run)
    return _dispatch(db, current_user, project_id, run.id)


@router.get(
    "/projects/{project_id}/prediction-runs", response_model=list[schemas.PredictionRunRead]
)
def list_prediction_runs(
    project_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    _authorize(db, current_user, project_id)
    return [
        service.prediction_read(db, run) for run in service.list_prediction_runs(db, project_id)
    ]


@router.get(
    "/projects/{project_id}/prediction-runs/{run_id}", response_model=schemas.PredictionRunRead
)
def get_prediction_run(
    project_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id)
    return service.prediction_read(db, service.get_prediction_run(db, project_id, run_id))


@router.post(
    "/projects/{project_id}/prediction-runs/{run_id}/retry",
    response_model=schemas.PredictionRunRead,
    status_code=202,
)
def retry_prediction_run(
    project_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id, write=True)
    run = service.get_prediction_run(db, project_id, run_id)
    if run.status not in {"failed", "planned", "queued"}:
        raise ConflictError("Only failed or queued prediction runs can be retried")
    return _dispatch(db, current_user, project_id, run.id)


@router.get(
    "/projects/{project_id}/prediction-runs/{run_id}/results",
    response_model=schemas.PredictionResultPage,
)
def prediction_results(
    project_id: int,
    run_id: int,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id)
    run = service.get_prediction_run(db, project_id, run_id)
    output = service._output(db, run)
    return {
        "items": service.result_rows(db, run, offset=offset, limit=limit),
        "total": output.candidate_count,
        "offset": offset,
        "limit": limit,
    }


def _csv_cell(value):
    text = (
        json.dumps(value, ensure_ascii=False)
        if isinstance(value, (dict, list))
        else str(value if value is not None else "")
    )
    # Downloaded text is untrusted spreadsheet content.
    return "'" + text if text.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else text


@router.get("/projects/{project_id}/prediction-runs/{run_id}/download")
def download_predictions(
    project_id: int,
    run_id: int,
    format: Literal["csv", "jsonl"] = "csv",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id)
    run = service.get_prediction_run(db, project_id, run_id)
    rows = service.result_rows(db, run)
    if format == "jsonl":
        content = "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows
        )
        content_type = "application/x-ndjson"
    else:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(schemas.PredictionResult.model_fields))
        writer.writeheader()
        writer.writerows({key: _csv_cell(value) for key, value in row.items()} for row in rows)
        content, content_type = buffer.getvalue(), "text/csv"
    return Response(
        content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="predictions-{run.id}.{format}"',
            "Cache-Control": "no-store",
        },
    )


@router.post(
    "/projects/{project_id}/prediction-runs/{run_id}/review-round",
    response_model=schemas.PredictionReviewRead,
    status_code=201,
)
def create_prediction_review(
    project_id: int,
    run_id: int,
    payload: schemas.PredictionReviewCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _authorize(db, current_user, project_id, write=True, review=True)
    return service.create_review_round(db, project_id, run_id, payload, current_user)
