"""Opt-in HTTP races for trainer authorization followed by project writes.

Run with ``AL_MEDLIT_RUN_POSTGRES_TESTS=1 pytest -q
tests/test_trainer_route_postgres_contention.py`` against real PostgreSQL.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from test_system_administration_postgres_contention import (
    _new_user,
    _set_transaction_timeouts,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)
from test_training_preparation_postgres_contention import _scope as _preparation_scope
from test_training_run_execution import MemoryObjectStorage, _completed_tfidf_model

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.auth.tenancy import lock_document_resource_for_mutation
from al_medlit.core.database import get_db
from al_medlit.core.storage import get_object_storage
from al_medlit.corpus.models import Document
from al_medlit.main import create_app
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.routes import predictions as prediction_routes
from al_medlit.workflow.services import labels, predictions, preparation
from al_medlit.workspace.models import Workspace, WorkspaceMember

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _route_actors(db, project_id):
    workspace_id = db.get(Project, project_id).workspace_id
    db.get(Workspace, workspace_id).capability_preset = "full"
    user_ids = []
    for index, role in enumerate(("trainer", "trainer", "annotator")):
        user = _new_user(db, f"trainer-route-{index}")
        db.add(WorkspaceMember(workspace_id=workspace_id, user_id=user.id, role=role))
        user_ids.append(user.id)
    db.commit()
    return user_ids


@contextmanager
def _postgres_client(session_factory, storage):
    app = create_app()

    def override_get_db():
        with session_factory() as db:
            _set_transaction_timeouts(db)
            yield db

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_object_storage] = lambda: storage
    with TestClient(app, base_url="https://testserver") as client:
        yield client


def _headers(user_id):
    return {"Authorization": f"Bearer {create_access_token(str(user_id))}"}


def _synchronize_service_entry(monkeypatch, module, name):
    original = getattr(module, name)
    barrier = threading.Barrier(2)
    entered = []

    def synchronized(*args, **kwargs):
        # The route has completed real token, membership, role, and module
        # authorization. Neither service has acquired its project write lock.
        # Before the fix, both transactions already hold FOR SHARE here and
        # deadlock when their services upgrade it. After the fix, the services
        # may serialize normally on the first exclusive project lock.
        entered.append(threading.get_ident())
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, synchronized)
    return entered


def _concurrent_posts(client, url, payloads, actor_ids):
    with ThreadPoolExecutor(max_workers=2) as executor:
        requests = [
            executor.submit(client.post, url, json=payload, headers=_headers(actor_id))
            for payload, actor_id in zip(payloads, actor_ids, strict=True)
        ]
        return [request.result(timeout=15) for request in requests]


@pytest.mark.parametrize("operation", ["compose", "prepare", "legacy", "bind", "prediction"])
def test_distinct_trainers_can_write_same_project_without_lock_upgrade_deadlock(
    postgres_session_factory, monkeypatch, operation
):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    storage = MemoryObjectStorage()
    with postgres_session_factory() as db:
        if operation == "prediction":
            scope = _completed_tfidf_model(db, storage, suffix=f"trainer-route-{uuid4().hex[:8]}")
            project_id = scope.project.id
            payload = {
                "name": "Trainer predictions",
                "dataset_version_id": scope.dataset.id,
                "model_version_id": scope.model_version.id,
            }
            payloads = [dict(payload, request_key=f"trainer-{index}") for index in range(2)]
            url = f"/api/projects/{project_id}/prediction-runs"
            target, function = predictions, "create_prediction_run"
            expected_status, model = 202, models.FeedbackRun
        else:
            project_id, creator_id, request = _preparation_scope(db, legacy_input=True)
            source_id = request.sources[0].dataset_version_id
            if operation == "compose":
                payload = {
                    "project_id": project_id,
                    "dataset_version_id": source_id,
                    "task_version_id": request.task_version_id,
                    "input_field": "text",
                    "label_field": "label",
                    "train_percent": 80,
                    "validation_percent": 10,
                }
                payloads = [dict(payload, name=f"Trainer examples {index}") for index in range(2)]
                url = "/api/datasets/training-versions/compose"
                target, function = service, "compose_training_dataset_version"
            elif operation == "prepare":
                payloads = [
                    request.model_copy(
                        update={
                            "name": f"Trainer examples {index}",
                            "idempotency_key": f"trainer-{index}",
                        }
                    ).model_dump(mode="json")
                    for index in range(2)
                ]
                url = f"/api/projects/{project_id}/training-datasets/prepare"
                target, function = preparation, "prepare_training_dataset"
            elif operation == "legacy":
                actor = db.get(User, creator_id)
                label_set = labels.create_label_set_version(
                    db,
                    schemas.LabelSetVersionCreate(
                        project_id=project_id,
                        dataset_version_id=source_id,
                        task_version_id=request.task_version_id,
                        name="Imported trainer labels",
                        source_kind="imported",
                        labels={f"paper-{index}": "yes" for index in range(6)},
                    ),
                    actor,
                )
                split = labels.create_split_map(
                    db,
                    schemas.SplitMapCreate(
                        project_id=project_id,
                        dataset_version_id=source_id,
                        name="Trainer split",
                        strategy="manual",
                        assignments={
                            f"paper-{index}": "train"
                            if index < 4
                            else ("validation" if index == 4 else "test")
                            for index in range(6)
                        },
                    ),
                    actor,
                )
                payload = {
                    "project_id": project_id,
                    "dataset_version_id": source_id,
                    "task_version_id": request.task_version_id,
                    "label_set_version_ids": [label_set.id],
                    "split_map_id": split.id,
                    "preprocessing": {"input_field": "text"},
                }
                payloads = [dict(payload, name=f"Trainer examples {index}") for index in range(2)]
                url = "/api/datasets/training-versions"
                target, function = service, "create_training_dataset_version"
            else:
                payloads = [None, None]
                url = (
                    "/api/training-recipes/trusted/tfidf_logistic_regression"
                    f"?project_id={project_id}"
                )
                target, function = service, "ensure_trusted_training_recipe_version"
            expected_status = 200 if operation == "bind" else 201
            model = (
                models.TrainingRecipeVersion
                if operation == "bind"
                else models.TrainingDatasetVersion
            )
        actor_ids = _route_actors(db, project_id)

    monkeypatch.setattr(prediction_routes, "enqueue_feedback_scoring", lambda _run_id: None)
    entered = _synchronize_service_entry(monkeypatch, target, function)
    with _postgres_client(postgres_session_factory, storage) as client:
        responses = _concurrent_posts(client, url, payloads, actor_ids[:2])
        assert [response.status_code for response in responses] == [expected_status] * 2, [
            response.text for response in responses
        ]
        assert len(entered) == 2

        # The lock fix must preserve the trainer role requirement. A rejected
        # annotator must never enter the service or wait at its race barrier.
        rejected = client.post(url, json=payloads[0], headers=_headers(actor_ids[2]))
        assert rejected.status_code == 403, rejected.text
        assert len(entered) == 2

    with postgres_session_factory() as db:
        expected_count = 1 if operation == "bind" else 2
        assert db.query(model).filter_by(project_id=project_id).count() == expected_count
    if operation == "bind":
        assert responses[0].json()["id"] == responses[1].json()["id"]


def test_document_authorization_keeps_compatible_project_locks_until_transaction_ends(
    postgres_session_factory,
):
    with postgres_session_factory() as db:
        creator = _new_user(db, "document-lock-creator")
        workspace = Workspace(name=f"Document locks {uuid4().hex}", created_by=creator.id)
        db.add(workspace)
        db.flush()
        project = Project(workspace_id=workspace.id, name="Document locks")
        db.add(project)
        db.flush()
        document = Document(project_id=project.id, text="Restricted document mutations")
        db.add(document)
        actor_ids = []
        for index in range(2):
            actor = _new_user(db, f"document-lock-annotator-{index}")
            db.add(WorkspaceMember(workspace_id=workspace.id, user_id=actor.id, role="annotator"))
            actor_ids.append(actor.id)
        db.commit()
        project_id, document_id = project.id, document.id

    with (
        postgres_session_factory() as left,
        postgres_session_factory() as right,
        postgres_session_factory() as assignment_writer,
    ):
        for db, actor_id in zip((left, right), actor_ids, strict=True):
            _set_transaction_timeouts(db)
            lock_document_resource_for_mutation(
                db,
                db.get(User, actor_id),
                project_id=project_id,
                document_id=document_id,
            )

        # Both restricted actors can authorize concurrently. Each retains a
        # compatible shared lock that excludes project/assignment-mode changes
        # for the remainder of its transaction, even after the other finishes.
        for completed in (left, right):
            with pytest.raises(OperationalError) as blocked:
                (
                    assignment_writer.query(Project)
                    .filter(Project.id == project_id)
                    .with_for_update(nowait=True)
                    .one()
                )
            assert blocked.value.orig.pgcode == "55P03"
            assignment_writer.rollback()
            completed.rollback()

        assert (
            assignment_writer.query(Project)
            .filter(Project.id == project_id)
            .with_for_update(nowait=True)
            .one()
            .id
        ) == project_id
