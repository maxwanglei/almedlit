import json
from datetime import UTC, datetime, timedelta

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ConflictError
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactStorageReservation,
    WorkspaceArtifactQuota,
)
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate
from al_medlit.model_artifacts.service import (
    PackageFileUpload,
    approve_artifact_package_purge,
    garbage_collect_artifact_blobs,
    publish_artifact_package,
    request_artifact_package_archive,
    update_artifact_package_retention,
)
from al_medlit.project.models import Project
from al_medlit.workspace import service as workspace_service


def _user(db, username: str) -> User:
    user = User(
        username=username,
        password_hash="test-only",
        display_name=username,
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user


def _headers(user: User) -> dict[str, str]:
    return {"Authorization": f"Bearer {create_access_token(str(user.id))}"}


def _workspace_projects(db):
    manager = _user(db, "quota-manager")
    trainer = _user(db, "quota-trainer")
    annotator = _user(db, "quota-annotator")
    workspace = workspace_service.create_team_workspace(
        db,
        manager,
        name="Quota workspace",
    )
    workspace_service.add_member(db, workspace.id, trainer.id, role="trainer")
    workspace_service.add_member(db, workspace.id, annotator.id, role="annotator")
    first = Project(name="Quota project one", workspace_id=workspace.id)
    second = Project(name="Quota project two", workspace_id=workspace.id)
    db.add_all([first, second])
    db.commit()
    return manager, trainer, annotator, workspace, first, second


def _package_spec(display_name: str) -> ArtifactPackageCreate:
    return ArtifactPackageCreate(
        package_kind="trained_model",
        package_format="safetensors",
        display_name=display_name,
        model_family="deep_learning",
        model_type="quota-test-transformer",
        readiness="ready",
        deployable=True,
        task_contract={"task": "classification", "version": "1"},
        runtime={"transformers": "test"},
    )


def _package_files() -> list[PackageFileUpload]:
    return [
        PackageFileUpload(
            "config.json",
            json.dumps({"architectures": ["QuotaModel"]}).encode(),
            role="model_config",
            content_type="application/json",
        ),
        PackageFileUpload(
            "model.safetensors",
            b"workspace-deduplicated-model-weights",
            role="model_weights",
        ),
    ]


def test_quota_reservations_are_atomic_releasable_and_expirable(db):
    manager, _trainer, _annotator, workspace, first, second = _workspace_projects(db)
    quota.set_workspace_artifact_quota(
        db,
        workspace_id=workspace.id,
        limit_bytes=100,
        reservation_ttl_seconds=60,
        actor_user_id=manager.id,
    )
    first_reservation = quota.reserve_artifact_bytes(
        db,
        project_id=first.id,
        owner_type="test_job",
        owner_id=1,
        idempotency_key="first-reservation",
        requested_bytes=80,
        actor_user_id=manager.id,
    )
    db.commit()

    with pytest.raises(ConflictError, match="20 bytes available"):
        quota.reserve_artifact_bytes(
            db,
            project_id=second.id,
            owner_type="test_job",
            owner_id=2,
            idempotency_key="over-limit",
            requested_bytes=30,
            actor_user_id=manager.id,
        )
    db.rollback()

    quota.release_artifact_reservation(
        db,
        reservation_id=first_reservation.id,
        reason="Interrupted before execution",
    )
    db.commit()
    quota_row = (
        db.query(WorkspaceArtifactQuota)
        .filter(WorkspaceArtifactQuota.workspace_id == workspace.id)
        .one()
    )
    assert quota_row.reserved_bytes == 0
    assert db.get(ArtifactStorageReservation, first_reservation.id).status == "released"

    started_at = datetime.now(UTC)
    expiring = quota.reserve_artifact_bytes(
        db,
        project_id=first.id,
        owner_type="test_job",
        owner_id=3,
        idempotency_key="expiring-reservation",
        requested_bytes=90,
        actor_user_id=manager.id,
        now=started_at,
    )
    db.commit()
    result = quota.expire_artifact_reservations(
        db,
        workspace_id=workspace.id,
        now=started_at + timedelta(seconds=61),
    )
    db.commit()
    assert result.expired_count == 1
    assert result.released_bytes == 90
    assert db.get(ArtifactStorageReservation, expiring.id).status == "expired"
    snapshot = quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workspace.id,
    )
    assert snapshot.reserved_bytes == 0
    assert snapshot.available_bytes == 100


def test_publication_commits_only_workspace_unique_blob_bytes(
    db,
    object_storage,
):
    manager, _trainer, _annotator, workspace, first, second = _workspace_projects(db)
    first_reservation = quota.reserve_artifact_bytes(
        db,
        project_id=first.id,
        owner_type="test_job",
        owner_id=10,
        idempotency_key="dedup-first",
        requested_bytes=10_000,
        actor_user_id=manager.id,
    )
    first_package = publish_artifact_package(
        db,
        object_storage,
        project_id=first.id,
        data=_package_spec("First"),
        files=_package_files(),
        actor_user_id=manager.id,
        reservation_id=first_reservation.id,
    )
    db.commit()
    first_committed = db.get(
        ArtifactStorageReservation,
        first_reservation.id,
    ).committed_bytes
    assert first_committed > sum(item.size_bytes for item in first_package.files)

    second_reservation = quota.reserve_artifact_bytes(
        db,
        project_id=second.id,
        owner_type="test_job",
        owner_id=11,
        idempotency_key="dedup-second",
        requested_bytes=10_000,
        actor_user_id=manager.id,
    )
    second_package = publish_artifact_package(
        db,
        object_storage,
        project_id=second.id,
        data=_package_spec("Second"),
        files=_package_files(),
        actor_user_id=manager.id,
        reservation_id=second_reservation.id,
    )
    db.commit()
    second_committed = db.get(
        ArtifactStorageReservation,
        second_reservation.id,
    ).committed_bytes

    first_weights = next(
        item for item in first_package.files if item.relative_path == "model.safetensors"
    )
    second_weights = next(
        item for item in second_package.files if item.relative_path == "model.safetensors"
    )
    assert first_weights.blob_id == second_weights.blob_id
    assert second_committed == second_package.manifest_blob.size_bytes
    assert second_committed < first_committed
    snapshot = quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workspace.id,
    )
    physical_bytes = sum(
        blob.size_bytes
        for blob in db.query(ArtifactBlob)
        .filter(
            ArtifactBlob.workspace_id == workspace.id,
            ArtifactBlob.status == "ready",
        )
        .all()
    )
    assert snapshot.used_bytes == physical_bytes
    assert snapshot.physical_bytes == physical_bytes
    assert snapshot.accounting_consistent is True


def test_legal_hold_blocks_cleanup_and_gc_returns_quota_bytes(
    db,
    object_storage,
):
    manager, _trainer, _annotator, workspace, project, _second = _workspace_projects(db)
    reservation = quota.reserve_artifact_bytes(
        db,
        project_id=project.id,
        owner_type="test_job",
        owner_id=20,
        idempotency_key="held-package",
        requested_bytes=10_000,
        actor_user_id=manager.id,
    )
    package = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_package_spec("Held"),
        files=_package_files(),
        actor_user_id=manager.id,
        reservation_id=reservation.id,
    )
    update_artifact_package_retention(
        db,
        package=package,
        actor_user_id=manager.id,
        legal_hold=True,
    )
    db.commit()
    used_before = quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workspace.id,
    ).used_bytes

    with pytest.raises(ConflictError, match="legal hold"):
        request_artifact_package_archive(
            db,
            package=package,
            actor_user_id=manager.id,
        )
    db.rollback()
    no_cleanup = garbage_collect_artifact_blobs(
        db,
        object_storage,
    )
    db.commit()
    assert no_cleanup.deleted_blob_count == 0
    assert (
        quota.workspace_artifact_quota_snapshot(
            db,
            workspace_id=workspace.id,
        ).used_bytes
        == used_before
    )

    update_artifact_package_retention(
        db,
        package=package,
        actor_user_id=manager.id,
        legal_hold=False,
    )
    request_artifact_package_archive(
        db,
        package=package,
        actor_user_id=manager.id,
        reason="Retention period completed",
        now=datetime.now(UTC) - timedelta(days=8),
    )
    approve_artifact_package_purge(
        db,
        package=package,
        actor_user_id=manager.id,
        reason="Approved quota cleanup",
    )
    db.commit()
    cleanup = garbage_collect_artifact_blobs(
        db,
        object_storage,
    )
    db.commit()
    assert cleanup.reclaimed_bytes == used_before
    snapshot = quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workspace.id,
    )
    assert snapshot.used_bytes == 0
    assert snapshot.physical_bytes == 0
    assert snapshot.accounting_consistent is True


def test_quota_api_hides_mutation_from_non_managers(db, client):
    manager, trainer, annotator, workspace, _first, _second = _workspace_projects(db)
    path = f"/api/workspaces/{workspace.id}/artifact-quota"

    assert client.get(path, headers=_headers(annotator)).status_code == 403
    visible = client.get(path, headers=_headers(trainer))
    assert visible.status_code == 200
    assert visible.json()["limit_bytes"] is None

    forbidden = client.patch(
        path,
        json={"limit_bytes": 2048},
        headers=_headers(trainer),
    )
    assert forbidden.status_code == 403
    updated = client.patch(
        path,
        json={"limit_bytes": 2048, "reservation_ttl_seconds": 120},
        headers=_headers(manager),
    )
    assert updated.status_code == 200
    assert updated.json()["limit_bytes"] == 2048
    assert updated.json()["reservation_ttl_seconds"] == 120
