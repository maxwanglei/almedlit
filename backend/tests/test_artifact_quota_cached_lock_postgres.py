from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session, sessionmaker
from test_artifact_quota_locking_postgres import (
    _RESERVATION_BYTES,
    _quota_row,
    _quota_workspace,
    _reserve,
    _two_reservations,
)
from test_artifact_quotas import _package_files, _package_spec
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.core.exceptions import ConflictError
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts.models import ArtifactStorageReservation
from al_medlit.model_artifacts.service import publish_artifact_package

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def test_release_refreshes_preloaded_quota_after_another_reservation_was_released(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    workspace_id, owner_type, first_id = _two_reservations(postgres_session_factory)
    with postgres_session_factory() as db:
        cached_quota = _quota_row(db, workspace_id)
        assert cached_quota.reserved_bytes == 2 * _RESERVATION_BYTES
        with postgres_session_factory() as other:
            quota.release_artifact_reservation(other, reservation_id=first_id, reason="Cancelled")
            other.commit()
        quota.complete_owner_artifact_reservation(db, owner_type=owner_type, owner_id=2)
        db.commit()

    with postgres_session_factory() as db:
        assert _quota_row(db, workspace_id).reserved_bytes == 0


def test_reservation_limit_uses_usage_committed_after_a_quota_was_preloaded(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    manager_id, workspace_id, project_id = _quota_workspace(postgres_session_factory)
    with postgres_session_factory() as db:
        quota.set_workspace_artifact_quota(
            db,
            workspace_id=workspace_id,
            limit_bytes=_RESERVATION_BYTES,
            reservation_ttl_seconds=60,
            actor_user_id=manager_id,
        )
        db.commit()

    with postgres_session_factory() as db:
        cached_quota = _quota_row(db, workspace_id)
        assert cached_quota.reserved_bytes == 0
        with postgres_session_factory() as other:
            _reserve(other, project_id=project_id, owner_type="quota-cached", owner_id=1)
            other.commit()
        with pytest.raises(ConflictError, match="quota exceeded"):
            _reserve(db, project_id=project_id, owner_type="quota-cached", owner_id=2)
        db.rollback()

    with postgres_session_factory() as db:
        assert _quota_row(db, workspace_id).reserved_bytes == _RESERVATION_BYTES


def test_publication_rejects_a_preloaded_reservation_that_was_released(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    workspace_id, _owner_type, first_id = _two_reservations(postgres_session_factory)
    with postgres_session_factory() as db:
        cached_reservation = db.get(ArtifactStorageReservation, first_id)
        assert cached_reservation.status == "active"
        with postgres_session_factory() as other:
            quota.release_artifact_reservation(other, reservation_id=first_id, reason="Cancelled")
            other.commit()
        with pytest.raises(ConflictError, match="cannot be committed from status 'released'"):
            quota.admit_artifact_publication(
                db,
                workspace_id=workspace_id,
                project_id=cached_reservation.project_id,
                candidates=[("a" * 64, 10)],
                reservation_id=first_id,
            )
        db.rollback()

    with postgres_session_factory() as db:
        assert db.get(ArtifactStorageReservation, first_id).status == "released"
        current_quota = _quota_row(db, workspace_id)
        assert (current_quota.used_bytes, current_quota.reserved_bytes) == (0, _RESERVATION_BYTES)


@pytest.mark.parametrize("operation", ["reserve", "publish"])
def test_reservation_refresh_preserves_expiration_from_the_current_transaction(
    postgres_session_factory: sessionmaker[Session],
    operation: str,
) -> None:
    manager_id, workspace_id, project_id = _quota_workspace(postgres_session_factory)
    created_at = datetime.now(UTC) - timedelta(days=8)
    owner_type = f"quota-expired-{uuid4().hex[:8]}"
    with postgres_session_factory() as db:
        reservation = quota.reserve_artifact_bytes(
            db,
            project_id=project_id,
            owner_type=owner_type,
            owner_id=1,
            idempotency_key="quota-expired",
            requested_bytes=_RESERVATION_BYTES,
            actor_user_id=manager_id,
            now=created_at,
        )
        reservation_id = reservation.id
        db.commit()

    with postgres_session_factory() as db:
        if operation == "reserve":
            retried = quota.reserve_artifact_bytes(
                db,
                project_id=project_id,
                owner_type=owner_type,
                owner_id=1,
                idempotency_key="quota-expired",
                requested_bytes=_RESERVATION_BYTES,
                actor_user_id=manager_id,
            )
            assert retried.status == "expired"
            db.commit()
        else:
            with pytest.raises(ConflictError, match="cannot be committed from status 'expired'"):
                quota.admit_artifact_publication(
                    db,
                    workspace_id=workspace_id,
                    project_id=project_id,
                    candidates=[("a" * 64, 10)],
                    reservation_id=reservation_id,
                )
            db.rollback()


def test_idempotent_reservation_returns_its_committed_terminal_status(
    postgres_session_factory: sessionmaker[Session],
) -> None:
    _manager_id, _workspace_id, project_id = _quota_workspace(postgres_session_factory)
    with postgres_session_factory() as db:
        reservation = quota.reserve_artifact_bytes(
            db,
            project_id=project_id,
            owner_type="quota-retry",
            owner_id=1,
            idempotency_key="quota-retry",
            requested_bytes=_RESERVATION_BYTES,
            actor_user_id=None,
        )
        reservation_id = reservation.id
        db.commit()

    with postgres_session_factory() as db:
        cached_reservation = db.get(ArtifactStorageReservation, reservation_id)
        assert cached_reservation.status == "active"
        with postgres_session_factory() as other:
            quota.release_artifact_reservation(
                other, reservation_id=reservation_id, reason="Cancelled"
            )
            other.commit()
        retried = quota.reserve_artifact_bytes(
            db,
            project_id=project_id,
            owner_type="quota-retry",
            owner_id=1,
            idempotency_key="quota-retry",
            requested_bytes=_RESERVATION_BYTES,
            actor_user_id=None,
        )
        assert (retried.id, retried.status) == (reservation_id, "released")


def test_link_rejects_replacing_a_package_link_committed_after_its_reservation_was_loaded(
    postgres_session_factory: sessionmaker[Session],
    object_storage,
) -> None:
    workspace_id, owner_type, reservation_id = _two_reservations(postgres_session_factory)
    with postgres_session_factory() as db:
        reservation = quota.complete_owner_artifact_reservation(
            db, owner_type=owner_type, owner_id=1
        )
        project_id = reservation.project_id
        package_ids = [
            publish_artifact_package(
                db,
                object_storage,
                project_id=project_id,
                data=_package_spec(name),
                files=_package_files(),
                actor_user_id=None,
            ).id
            for name in ("First link", "Second link")
        ]
        db.commit()

    with postgres_session_factory() as db:
        cached_reservation = db.get(ArtifactStorageReservation, reservation_id)
        assert cached_reservation.artifact_package_id is None
        with postgres_session_factory() as other:
            quota.link_reservation_artifact_package(
                other, reservation_id=reservation_id, artifact_package_id=package_ids[0]
            )
            other.commit()
        with pytest.raises(ConflictError, match="already linked to another package"):
            quota.link_reservation_artifact_package(
                db, reservation_id=reservation_id, artifact_package_id=package_ids[1]
            )
        db.rollback()

    with postgres_session_factory() as db:
        reservation = db.get(ArtifactStorageReservation, reservation_id)
        assert reservation.artifact_package_id == package_ids[0]
        assert _quota_row(db, workspace_id).reserved_bytes == _RESERVATION_BYTES


@pytest.mark.parametrize("operation", ["reserve", "publish"])
def test_idempotent_retry_preserves_expiration_before_another_quota_mutation(
    postgres_session_factory: sessionmaker[Session],
    operation: str,
) -> None:
    manager_id, workspace_id, project_id = _quota_workspace(postgres_session_factory)
    owner_type = f"quota-composed-{uuid4().hex[:8]}"
    now = datetime.now(UTC)
    with postgres_session_factory() as db:

        def reserve(owner_id: int, created_at: datetime) -> ArtifactStorageReservation:
            return quota.reserve_artifact_bytes(
                db,
                project_id=project_id,
                owner_type=owner_type,
                owner_id=owner_id,
                idempotency_key=f"{owner_type}-{owner_id}",
                requested_bytes=_RESERVATION_BYTES,
                actor_user_id=manager_id,
                now=created_at,
            )

        completed = reserve(1, now)
        quota.complete_owner_artifact_reservation(db, owner_type=owner_type, owner_id=1)
        active = reserve(2, now)
        overdue = reserve(3, now - timedelta(days=8))
        db.commit()

        if operation == "reserve":
            assert reserve(3, now).status == "expired"
        else:
            assert quota.admit_artifact_publication(
                db,
                workspace_id=workspace_id,
                project_id=project_id,
                candidates=[],
                reservation_id=completed.id,
            ) == 0
        quota.release_artifact_reservation(db, reservation_id=active.id, reason="Cancelled")
        db.commit()
        assert overdue.status == "expired"

    with postgres_session_factory() as db:
        assert _quota_row(db, workspace_id).reserved_bytes == 0
