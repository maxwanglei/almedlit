import io
import json
import logging
import os
import shutil
import struct
import subprocess
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import event
from test_inference_candidate_queries import _inference_scope

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.archive import (
    ArchiveExtractionError,
    ArchiveExtractionLimits,
    extract_zip_bounded,
)
from al_medlit.core.config import Settings
from al_medlit.core.exceptions import ConflictError
from al_medlit.core.storage import ObjectNotFoundError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.schemas import DocumentCreate, DocumentSummaryRead
from al_medlit.inference import service as inference_service
from al_medlit.inference.models import InferenceRun
from al_medlit.inference.schemas import InferenceRunCreate
from al_medlit.model_artifacts.models import ArtifactBlob
from al_medlit.model_artifacts.service import blob_storage_key, sweep_unregistered_artifact_objects
from al_medlit.project.models import Project
from al_medlit.storage_reclaim.models import OrphanedStorageObject
from al_medlit.storage_reclaim.service import reclaim_orphaned_objects, record_orphaned_object
from al_medlit.submission.models import AnnotationSubmission
from al_medlit.workspace.models import Workspace, WorkspaceMember


def _project(db):
    workspace = Workspace(name="Platform audit", kind="team", join_code="private-code")
    db.add(workspace)
    db.flush()
    project = Project(name="Audit project", workspace_id=workspace.id)
    db.add(project)
    db.commit()
    return project


def test_document_summaries_skip_content_and_paginate(db):
    project = _project(db)
    for index in range(3):
        corpus_service.create_document(
            db,
            DocumentCreate(project_id=project.id, title=str(index), text="Large text. " * 100),
        )
    project_id = project.id
    db.expire_all()
    statements = []

    def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", observe)
    try:
        first = corpus_service.list_documents(db, project_id, summary=True, limit=2)
        serialized = [DocumentSummaryRead.model_validate(item).model_dump() for item in first]
        second = corpus_service.list_documents(db, project_id, summary=True, limit=2, offset=2)
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", observe)
    assert [item["title"] for item in serialized] == ["2", "1"]
    assert [item.title for item in second] == ["0"]
    assert all("text" not in item and "sentences" not in item for item in serialized)
    assert len(statements) == 2
    assert not any("documents.text" in statement for statement in statements)


def test_detailed_document_pages_batch_load_sentences(db):
    project = _project(db)
    for _index in range(5):
        corpus_service.create_document(db, DocumentCreate(project_id=project.id, text="One. Two."))
    project_id = project.id
    db.expire_all()
    statements = []

    def observe(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", observe)
    try:
        documents = corpus_service.list_documents(db, project_id)
        assert all(item.sentences for item in documents)
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", observe)
    assert len(statements) == 3


def test_percent_encoded_question_mark_cannot_make_a_route_public(client):
    @client._client.app.get("/api/invites/example?private/resource")
    def future_private_route():
        return {"secret": True}

    response = client._client.get("/api/invites/example%3Fprivate/resource")
    assert response.status_code == 401


def test_zip_comment_cannot_choose_a_different_directory(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a" * 200, b"secret")
    raw = buffer.getvalue()
    end = struct.Struct("<4s4H2LH")
    record = end.unpack(raw[-end.size :])
    directory = raw[record[6] : record[6] + record[5]]
    fake = end.pack(b"PK\x05\x06", 0, 0, 1, 1, len(directory), record[6], 1)
    comment = directory + fake
    outer = end.pack(b"PK\x05\x06", 0, 0, 0, 0, 0, 0, len(comment))
    source = tmp_path / "ambiguous.zip"
    source.write_bytes(raw[: -end.size] + outer + comment)
    with zipfile.ZipFile(source) as archive:
        assert len(archive.infolist()) == 1
    with pytest.raises(ArchiveExtractionError):
        extract_zip_bounded(
            source,
            tmp_path / "out",
            limits=ArchiveExtractionLimits(max_central_directory_bytes=100),
        )
    assert not list((tmp_path / "out").iterdir())


def test_reclaim_preserves_a_submission_after_ambiguous_commit(db, object_storage):
    project = _project(db)
    document = corpus_service.create_document(
        db, DocumentCreate(project_id=project.id, text="Saved.")
    )
    key = "submissions/live.json"
    object_storage.put_bytes(key, b"saved")
    db.add(
        AnnotationSubmission(
            project_id=project.id,
            document_id=document.id,
            storage_key=key,
            file_name="live.json",
            content_type="application/json",
            size_bytes=5,
            checksum_sha256="a" * 64,
        )
    )
    db.commit()
    record_orphaned_object(db, key, origin="submission.create_rollback")
    result = reclaim_orphaned_objects(
        db, object_storage, now=datetime.now(UTC) + timedelta(hours=1)
    )
    db.commit()
    assert result.reclaimed_count == 0
    assert object_storage.get_bytes(key) == b"saved"
    assert db.query(OrphanedStorageObject).count() == 0


def test_artifact_sweep_finds_rolled_back_publication_and_preserves_live_objects(
    db, object_storage
):
    project = _project(db)
    stale = blob_storage_key(project.workspace_id, "a" * 64)
    live = blob_storage_key(project.workspace_id, "b" * 64)
    recent = blob_storage_key(project.workspace_id, "c" * 64)
    for key in (stale, live, recent):
        object_storage.put_bytes(key, b"saved")
    older = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    for key in (stale, live):
        os.utime(object_storage.root / key, (older, older))
    db.add(
        ArtifactBlob(
            workspace_id=project.workspace_id,
            checksum_sha256="b" * 64,
            size_bytes=5,
            content_type="application/octet-stream",
            storage_key=live,
        )
    )
    db.commit()
    assert sweep_unregistered_artifact_objects(db, object_storage) == 1
    db.commit()
    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(stale)
    assert object_storage.get_bytes(live) == b"saved"
    assert object_storage.get_bytes(recent) == b"saved"


def test_logout_revokes_a_copied_bearer_token(client, auth_user):
    headers = {"Authorization": f"Bearer {auth_user['access_token']}"}
    assert client.get("/api/auth/me", headers=headers).status_code == 200
    assert client.post("/api/auth/logout", headers=headers).status_code == 200
    assert client.get("/api/auth/me", headers=headers).status_code == 401


def test_invite_tokens_can_be_sent_only_in_the_body(client, db):
    from al_medlit.workspace import service as workspace_service

    project = _project(db)
    owner = User(username="inviter", password_hash="unused")
    db.add(owner)
    db.flush()
    db.add(WorkspaceMember(workspace_id=project.workspace_id, user_id=owner.id, role="admin"))
    db.flush()
    invite = workspace_service.create_invite(
        db,
        project.workspace_id,
        created_by=owner.id,
        role="annotator",
        expires_minutes=60,
    )
    db.commit()
    response = client._client.post("/api/invites/preview", json={"token": invite.token})
    assert response.status_code == 200
    assert response.json()["workspace_name"] == "Platform audit"


def test_account_action_links_use_a_fragment_and_body_api(client, db):
    from al_medlit.administration import service as admin_service
    from al_medlit.administration.schemas import AdminUserCreate

    admin = User(username="system-owner", password_hash="unused", is_superuser=True)
    db.add(admin)
    db.commit()
    created = admin_service.create_inactive_user(
        db,
        actor_user_id=admin.id,
        data=AdminUserCreate(username="activate-me"),
    )
    db.commit()
    assert created.action.url.startswith("/account-actions#token=")
    token = created.action.url.split("#token=", 1)[1]
    preview = client._client.post("/api/account-actions/preview", json={"token": token})
    assert preview.status_code == 200
    complete = client._client.post(
        "/api/account-actions/complete",
        json={"token": token, "password": "new-password-12345"},
    )
    assert complete.status_code == 200


def test_legacy_action_token_access_logs_are_redacted():
    from al_medlit.auth.logging import ActionTokenAccessFilter

    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %s',
        ("client", "GET", "/api/account-actions/private-reset-token", "1.1", 200),
        None,
    )
    assert ActionTokenAccessFilter().filter(record)
    assert "private-reset-token" not in record.getMessage()
    assert "[redacted]" in record.getMessage()


def test_workspace_join_codes_are_visible_only_to_administrators(client, db):
    project = _project(db)
    user = User(username="member", password_hash="unused")
    db.add(user)
    db.flush()
    member = WorkspaceMember(workspace_id=project.workspace_id, user_id=user.id, role="annotator")
    db.add(member)
    db.commit()
    headers = {"Authorization": f"Bearer {create_access_token(user.id)}"}
    response = client.get("/api/workspaces", headers=headers)
    assert response.status_code == 200
    assert response.json()[0]["join_code"] is None
    member.role = "admin"
    db.commit()
    assert client.get("/api/workspaces", headers=headers).json()[0]["join_code"] == "private-code"


def test_inference_idempotency_rejects_a_changed_request(db):
    scope = _inference_scope(db)
    data = InferenceRunCreate(
        name=scope.run.name,
        corpus_snapshot_id=scope.run.corpus_snapshot_id,
        checkpoint_id=scope.run.checkpoint_id,
        compute_profile_id=scope.run.compute_profile_id,
        target_version_ids=scope.run.target_version_ids,
        idempotency_key=scope.run.idempotency_key,
    )
    scope.run.window_config = data.window_config.model_dump()
    scope.run.decoder_config = data.decoder_config.model_dump()
    db.commit()
    assert (
        inference_service.launch_inference_run(
            db,
            project_id=scope.project.id,
            data=data,
            actor_user_id=scope.user.id,
        ).id
        == scope.run.id
    )
    with pytest.raises(ConflictError, match="different parameters"):
        inference_service.launch_inference_run(
            db,
            project_id=scope.project.id,
            data=data.model_copy(update={"name": "Changed"}),
            actor_user_id=scope.user.id,
        )


@pytest.mark.parametrize("secret", ["x", "a" * 40, "short-secret"])
def test_nondefault_weak_storage_secrets_are_rejected(secret):
    configured = Settings(
        database_url="sqlite://",
        storage_backend="minio",
        storage_secret_key=secret,
    )
    with pytest.raises(RuntimeError, match="AL_MEDLIT_STORAGE_SECRET_KEY"):
        configured.validate_deployment_secrets()


def test_inference_run_pages_have_a_stable_total_order(db):
    scope = _inference_scope(db)
    for index in range(3):
        db.add(
            InferenceRun(
                project_id=scope.project.id,
                corpus_snapshot_id=scope.run.corpus_snapshot_id,
                checkpoint_id=scope.run.checkpoint_id,
                compute_profile_id=scope.run.compute_profile_id,
                name=f"Run {index}",
                target_version_ids=scope.run.target_version_ids,
                window_config={},
                decoder_config={},
                status="queued",
                idempotency_key=f"page-key-{index}",
                created_at=scope.run.created_at,
            )
        )
    db.commit()
    first = inference_service.list_inference_runs(db, scope.project.id, limit=2)
    second = inference_service.list_inference_runs(db, scope.project.id, limit=2, offset=2)
    assert len(first) == len(second) == 2
    assert not {item.id for item in first}.intersection(item.id for item in second)
    assert [item.id for item in first + second] == sorted(
        [item.id for item in first + second],
        reverse=True,
    )


def test_login_normalizes_username_before_authentication(client, db):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate

    auth_service.register_user(db, UserCreate(username="normalized", password="password-12345"))
    db.commit()
    response = client.post(
        "/api/auth/login",
        json={"username": "  normalized  ", "password": "password-12345"},
    )
    assert response.status_code == 200


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker Compose CLI is unavailable")
def test_compose_uses_a_separate_storage_account_and_waits_for_api_health():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env.example",
            "-f",
            "infra/docker-compose.yml",
            "--profile",
            "lab",
            "config",
            "--format",
            "json",
        ],
        cwd=root,
        env=os.environ
        | {
            "POSTGRES_PASSWORD": "compose-test-database-password",
            "MINIO_ROOT_PASSWORD": "compose-test-root-password",
            "AL_MEDLIT_STORAGE_ACCESS_KEY": "al-medlit-app",
            "AL_MEDLIT_STORAGE_BUCKET": "platform-audit-bucket",
            "AL_MEDLIT_STORAGE_SECRET_KEY": "compose-test-application-password",
            "AL_MEDLIT_JWT_SECRET": "compose-test-jwt-secret-at-least-32-bytes",
            "AL_MEDLIT_DEPLOYMENT_PROFILE": "lab",
            "AL_MEDLIT_CELERY_BROKER_URL": "redis://redis:6379/0",
            "AL_MEDLIT_CELERY_TASK_ALWAYS_EAGER": "false",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    services = json.loads(result.stdout)["services"]
    backend = services["backend"]
    assert "healthcheck" in backend
    assert backend["environment"]["AL_MEDLIT_STORAGE_ACCESS_KEY"] == "al-medlit-app"
    assert backend["environment"]["AL_MEDLIT_STORAGE_BUCKET"] == "platform-audit-bucket"
    assert services["minio-app-init"]["environment"]["AL_MEDLIT_STORAGE_BUCKET"] == (
        "platform-audit-bucket"
    )
    assert backend["environment"]["AL_MEDLIT_STORAGE_SECRET_KEY"] != "compose-test-root-password"
    assert backend["environment"]["AL_MEDLIT_CELERY_TASK_ALWAYS_EAGER"] == "false"
    assert backend["depends_on"]["minio-app-init"]["condition"] == "service_completed_successfully"
    for name in ("worker-control", "scheduler"):
        assert services[name]["depends_on"]["backend"]["condition"] == "service_healthy"
