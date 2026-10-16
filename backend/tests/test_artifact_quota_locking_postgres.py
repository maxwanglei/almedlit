"""PostgreSQL races between unlocked artifact quota reads and locked mutations."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import func, text
from sqlalchemy.orm import Session, sessionmaker
from test_artifact_quotas import _package_files, _package_spec
from test_system_administration_postgres_contention import (
    _LOCK_TIMEOUT_SECONDS,
    _THREAD_TIMEOUT_SECONDS,
    _new_user,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.model_artifacts import quota as artifact_quota
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactStorageReservation,
    WorkspaceArtifactQuota,
)
from al_medlit.model_artifacts.service import (
    approve_artifact_package_purge,
    garbage_collect_artifact_blobs,
    publish_artifact_package,
    request_artifact_package_archive,
    verify_artifact_package,
)
from al_medlit.project.models import Project
from al_medlit.workspace import service as workspace_service

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]

_RESERVATION_BYTES = 30


def _quota_workspace(session_factory: sessionmaker[Session]) -> tuple[int, int, int]:
    with session_factory() as db:
        manager = _new_user(db, "quota-lock-manager")
        workspace = workspace_service.create_team_workspace(
            db,
            manager,
            name=f"Quota locks {uuid4().hex[:8]}",
        )
        project = Project(name="Quota lock project", workspace_id=workspace.id)
        db.add(project)
        db.commit()
        return manager.id, workspace.id, project.id


def _reserve(
    db: Session,
    *,
    project_id: int,
    owner_type: str,
    owner_id: int,
) -> int:
    return artifact_quota.reserve_artifact_bytes(
        db,
        project_id=project_id,
        owner_type=owner_type,
        owner_id=owner_id,
        idempotency_key=f"quota-lock-{uuid4().hex}",
        requested_bytes=_RESERVATION_BYTES,
        actor_user_id=None,
    ).id


def _quota_row(db: Session, workspace_id: int) -> WorkspaceArtifactQuota:
    return (
        db.query(WorkspaceArtifactQuota)
        .filter(WorkspaceArtifactQuota.workspace_id == workspace_id)
        .one()
    )


def _two_reservations(
    session_factory: sessionmaker[Session],
) -> tuple[int, str, int]:
    _manager_id, workspace_id, project_id = _quota_workspace(session_factory)
    owner_type = f"quota-lock-{uuid4().hex[:8]}"
    with session_factory() as db:
        first_id = _reserve(db, project_id=project_id, owner_type=owner_type, owner_id=1)
        _reserve(db, project_id=project_id, owner_type=owner_type, owner_id=2)
        db.commit()
    return workspace_id, owner_type, first_id


def _reservation_outcome(
    session_factory: sessionmaker[Session],
    *,
    workspace_id: int,
    reservation_id: int,
) -> tuple[str, int]:
    with session_factory() as db:
        return (
            db.get(ArtifactStorageReservation, reservation_id).status,
            _quota_row(db, workspace_id).reserved_bytes,
        )


def _wait_for_lock_wait(session_factory: sessionmaker[Session], backend_pid: int) -> None:
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    with session_factory() as monitor:
        while time.monotonic() < deadline:
            wait_event_type = monitor.execute(
                text("SELECT wait_event_type FROM pg_stat_activity WHERE pid = :pid"),
                {"pid": backend_pid},
            ).scalar()
            # pg_stat_activity is a per-transaction snapshot.
            monitor.rollback()
            if wait_event_type == "Lock":
                return
            time.sleep(0.01)
    raise AssertionError(f"Backend {backend_pid} never waited on a lock")


@pytest.mark.parametrize(
    ("stale_mutation", "concurrent_mutation", "concurrent_status"),
    [
        ("release", "complete_owner", "committed"),
        ("complete_owner", "release", "released"),
    ],
)
def test_reservation_mutation_sees_status_committed_after_its_unlocked_read(
    postgres_session_factory: sessionmaker[Session],
    monkeypatch,
    stale_mutation: str,
    concurrent_mutation: str,
    concurrent_status: str,
) -> None:
    workspace_id, owner_type, first_id = _two_reservations(postgres_session_factory)
    mutations: dict[str, Callable[[Session], ArtifactStorageReservation | None]] = {
        "release": lambda db: artifact_quota.release_artifact_reservation(
            db,
            reservation_id=first_id,
            reason="Training run ended as cancelled",
        ),
        "complete_owner": lambda db: artifact_quota.complete_owner_artifact_reservation(
            db,
            owner_type=owner_type,
            owner_id=1,
        ),
    }
    original_locked_quota = artifact_quota._locked_quota

    def locked_quota_after_concurrent_commit(
        db: Session,
        workspace_id: int,
    ) -> WorkspaceArtifactQuota:
        # The caller has read the reservation without a lock. A competing
        # terminal transition commits before the caller acquires the quota.
        if db.info.pop("commit_concurrent_mutation", False):
            with postgres_session_factory() as other:
                mutations[concurrent_mutation](other)
                other.commit()
        return original_locked_quota(db, workspace_id)

    monkeypatch.setattr(artifact_quota, "_locked_quota", locked_quota_after_concurrent_commit)
    with postgres_session_factory() as db:
        db.info["commit_concurrent_mutation"] = True
        mutations[stale_mutation](db)
        db.commit()

    assert _reservation_outcome(
        postgres_session_factory,
        workspace_id=workspace_id,
        reservation_id=first_id,
    ) == (concurrent_status, _RESERVATION_BYTES)


def test_reservation_release_waiting_on_completion_sees_committed_status(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    workspace_id, owner_type, first_id = _two_reservations(postgres_session_factory)
    started = threading.Event()
    results: dict[str, Any] = {}

    def release() -> None:
        with postgres_session_factory() as db:
            try:
                db.execute(text(f"SET lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
                db.execute(text(f"SET statement_timeout = '{_THREAD_TIMEOUT_SECONDS - 2}s'"))
                results["pid"] = db.execute(text("SELECT pg_backend_pid()")).scalar_one()
                started.set()
                reservation = artifact_quota.release_artifact_reservation(
                    db,
                    reservation_id=first_id,
                    reason="Released by a workspace manager",
                )
                results["release"] = ("ok", reservation.status)
                db.commit()
            except Exception as exc:  # noqa: BLE001 - assert concurrent failures below
                db.rollback()
                results["release"] = ("error", exc)

    releaser = threading.Thread(target=release, daemon=True)
    with postgres_session_factory() as completion:
        completion.execute(text(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT_SECONDS}s'"))
        artifact_quota.complete_owner_artifact_reservation(
            completion,
            owner_type=owner_type,
            owner_id=1,
        )
        releaser.start()
        try:
            assert started.wait(timeout=_LOCK_TIMEOUT_SECONDS), results
            _wait_for_lock_wait(postgres_session_factory, results["pid"])
            completion.commit()
        finally:
            releaser.join(timeout=_THREAD_TIMEOUT_SECONDS)

    assert not releaser.is_alive(), results
    assert results["release"] == ("ok", "committed"), results
    assert _reservation_outcome(
        postgres_session_factory,
        workspace_id=workspace_id,
        reservation_id=first_id,
    ) == ("committed", _RESERVATION_BYTES)


def test_publication_after_blob_read_rehydrates_concurrently_purged_blobs(
    postgres_session_factory: sessionmaker[Session],
    object_storage,
) -> None:
    manager_id, workspace_id, project_id = _quota_workspace(postgres_session_factory)
    with postgres_session_factory() as db:
        original_id = publish_artifact_package(
            db,
            object_storage,
            project_id=project_id,
            data=_package_spec("Purged original"),
            files=_package_files(),
            actor_user_id=manager_id,
        ).id
        db.commit()

    with postgres_session_factory() as db:
        # Hold the blob rows in the identity map, as staging a base model does
        # for the rest of its training transaction.
        verified = verify_artifact_package(
            db,
            object_storage,
            project_id=project_id,
            package_id=original_id,
        )
        with postgres_session_factory() as purger:
            original = purger.get(ArtifactPackage, original_id)
            request_artifact_package_archive(
                purger,
                package=original,
                actor_user_id=manager_id,
                reason="Retention period completed",
                now=datetime.now(UTC) - timedelta(days=8),
            )
            approve_artifact_package_purge(
                purger,
                package=original,
                actor_user_id=manager_id,
                reason="Approved quota cleanup",
            )
            purger.commit()
            garbage_collect_artifact_blobs(purger, object_storage, workspace_id=workspace_id)
            purger.commit()
        republished_id = publish_artifact_package(
            db,
            object_storage,
            project_id=project_id,
            data=_package_spec("Republished"),
            files=_package_files(),
            actor_user_id=manager_id,
        ).id
        db.commit()
        assert verified.id == original_id

    with postgres_session_factory() as db:
        republished = db.get(ArtifactPackage, republished_id)
        file_blobs = {item.blob for item in republished.files}
        physical_bytes = db.scalar(
            db.query(func.coalesce(func.sum(ArtifactBlob.size_bytes), 0))
            .filter(
                ArtifactBlob.workspace_id == workspace_id,
                ArtifactBlob.status == "ready",
            )
            .statement
        )
        expected_bytes = (
            sum(blob.size_bytes for blob in file_blobs) + republished.manifest_blob.size_bytes
        )
        assert (
            {blob.status for blob in file_blobs},
            _quota_row(db, workspace_id).used_bytes,
            physical_bytes,
        ) == ({"ready"}, expected_bytes, expected_bytes)
