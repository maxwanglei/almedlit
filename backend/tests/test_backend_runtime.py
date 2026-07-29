import logging
from datetime import UTC, timedelta

import pytest
from sqlalchemy import DateTime
from sqlalchemy.exc import IntegrityError

from al_medlit.annotation.models import AnnotationCorrection
from al_medlit.annotation.schemas import AnnotationCorrectionCreate, AnnotationCreate
from al_medlit.annotation.service import create_annotation, create_correction
from al_medlit.co_learning.error_guideline_learning.events import register_event_handlers
from al_medlit.co_learning.error_guideline_learning.models import ErrorPattern
from al_medlit.core.events import event_bus
from al_medlit.core.exceptions import ConflictError
from al_medlit.core.models import utc_now
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.corpus.service import create_document
from al_medlit.main import create_app
from al_medlit.project.models import Project
from al_medlit.project.schemas import ProjectCreate, ProjectTaskCreate, TaskAssignmentCreate
from al_medlit.project.service import (
    create_project,
    create_project_task,
    create_task_assignment,
)


def test_timestamp_columns_use_timezone_aware_utc_defaults():
    created_column = Project.__table__.c.created_at
    updated_column = Project.__table__.c.updated_at

    assert isinstance(created_column.type, DateTime)
    assert created_column.type.timezone is True
    assert isinstance(updated_column.type, DateTime)
    assert updated_column.type.timezone is True
    assert created_column.default is not None
    assert updated_column.default is not None
    assert updated_column.onupdate is not None

    timestamp = utc_now()

    assert timestamp.tzinfo is UTC
    assert timestamp.utcoffset() == timedelta(0)


def test_create_project_route_uses_test_database(client):
    response = client.post(
        "/api/projects",
        json={
            "name": "smoke-project",
            "description": "runtime smoke test",
            "annotation_schema": {"labels": {}},
            "settings": {},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "smoke-project"

    list_response = client.get("/api/projects")
    assert list_response.status_code == 200
    assert any(project["id"] == payload["id"] for project in list_response.json())


def test_evidence_release_routers_are_registered() -> None:
    paths = set(create_app().openapi()["paths"])

    assert {
        "/api/documents/{document_id}/structure",
        "/api/projects/{project_id}/evidence-targets",
        "/api/annotations/evidence-blocks/commands",
        "/api/projects/{project_id}/corpus-snapshots",
        "/api/projects/{project_id}/exports",
        "/api/training-runs",
        "/api/training-runs/{training_run_id}/transition",
        "/api/projects/{project_id}/inference/runs",
        "/api/inference/predictions/{prediction_id}/review",
    } <= paths


def test_register_event_handlers_is_idempotent():
    register_event_handlers()
    register_event_handlers()

    assert event_bus.subscriber_count("annotation.correction.created") == 1


def test_create_correction_publishes_single_error_pattern(db, testing_session_factory):
    register_event_handlers()
    register_event_handlers()

    project = create_project(
        db,
        ProjectCreate(
            name="boundary-project",
            annotation_schema={
                "annotation_types": ["entity"],
                "labels": [{"name": "Drug", "color": "#7aa2f7"}],
            },
            settings={},
        ),
    )
    assert project.annotation_schema == {
        "labels": {
            "entity": [
                {
                    "name": "Drug",
                    "color": "#7aa2f7",
                    "description": None,
                }
            ]
        }
    }
    document = create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text="Patients receiving low-dose aspirin therapy were excluded.",
        ),
    )
    original_annotation = create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=document.id,
            annotation_type="entity",
            label="Drug",
            start_offset=19,
            end_offset=43,
            text_span="low-dose aspirin therapy",
            source="model",
            status="rejected",
        ),
    )
    corrected_annotation = create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=document.id,
            annotation_type="entity",
            label="Drug",
            start_offset=28,
            end_offset=35,
            text_span="aspirin",
            source="human",
            status="gold",
        ),
    )

    create_correction(
        db,
        AnnotationCorrectionCreate(
            project_id=project.id,
            document_id=document.id,
            original_annotation_id=original_annotation.id,
            corrected_annotation_id=corrected_annotation.id,
            correction_source="adjudication",
            error_type="boundary_error",
        ),
        created_by_user_id=None,
    )

    verification_session = testing_session_factory()
    try:
        patterns = (
            verification_session.query(ErrorPattern)
            .filter(ErrorPattern.project_id == project.id)
            .all()
        )
    finally:
        verification_session.close()

    assert len(patterns) == 1
    assert patterns[0].error_type == "boundary_error"
    assert patterns[0].example_count == 1


def test_create_correction_is_returned_when_event_handler_fails(
    db,
    testing_session_factory,
    monkeypatch,
    caplog,
):
    def failing_handler(payload: dict) -> None:
        raise RuntimeError(f"handler failed for correction {payload['correction_id']}")

    monkeypatch.setitem(
        event_bus.handlers,
        "annotation.correction.created",
        [failing_handler],
    )
    caplog.set_level(logging.ERROR, logger="al_medlit.core.events")

    project = create_project(
        db,
        ProjectCreate(
            name="correction-event-failure-project",
            annotation_schema={"labels": {}},
            settings={},
        ),
    )
    document = create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text="Patients receiving aspirin improved.",
        ),
    )

    correction = create_correction(
        db,
        AnnotationCorrectionCreate(
            project_id=project.id,
            document_id=document.id,
            error_type="handler_failure",
        ),
        created_by_user_id=None,
    )

    verification_session = testing_session_factory()
    try:
        persisted = verification_session.get(AnnotationCorrection, correction.id)
    finally:
        verification_session.close()

    assert correction.id is not None
    assert persisted is not None
    assert persisted.error_type == "handler_failure"
    assert "failed while handling annotation.correction.created" in caplog.text


def _duplicate_integrity_error() -> IntegrityError:
    return IntegrityError("INSERT", {}, Exception("duplicate key"))


def test_create_project_maps_unique_race_to_conflict(db, monkeypatch):
    def raise_integrity_error() -> None:
        raise _duplicate_integrity_error()

    monkeypatch.setattr(db, "commit", raise_integrity_error)

    with pytest.raises(ConflictError, match="Project with name 'racy-project' already exists"):
        create_project(
            db,
            ProjectCreate(
                name="racy-project",
                annotation_schema={"labels": {}},
                settings={},
            ),
        )


def test_create_project_task_maps_unique_race_to_conflict(db, monkeypatch):
    project = create_project(
        db,
        ProjectCreate(
            name="racy-task-project",
            annotation_schema={"labels": {}},
            settings={},
        ),
    )

    def raise_integrity_error(*_args, **_kwargs) -> None:
        raise _duplicate_integrity_error()

    monkeypatch.setattr(db, "flush", raise_integrity_error)

    with pytest.raises(
        ConflictError,
        match="Project task already exists for annotation_type 'entity'",
    ):
        create_project_task(
            db,
            project.id,
            ProjectTaskCreate(annotation_type="entity"),
        )


def test_create_task_assignment_maps_unique_race_to_conflict(db, monkeypatch):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.workspace import service as workspace_service

    project = create_project(
        db,
        ProjectCreate(
            name="racy-assignment-project",
            annotation_schema={"labels": {}},
            settings={},
        ),
    )
    task = create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="entity"),
    )
    document = create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            text="Patients receiving aspirin improved.",
        ),
    )
    user = auth_service.register_user(db, UserCreate(username="curator1", password="pw"))
    workspace_service.add_member(db, project.workspace_id, user.id, role="annotator")

    def raise_integrity_error() -> None:
        raise _duplicate_integrity_error()

    monkeypatch.setattr(db, "commit", raise_integrity_error)

    with pytest.raises(
        ConflictError,
        match="Task assignment already exists for this task, document, and annotator",
    ):
        create_task_assignment(
            db,
            project.id,
            TaskAssignmentCreate(
                task_id=task.id,
                document_id=document.id,
                annotator_id="curator1",
            ),
        )
