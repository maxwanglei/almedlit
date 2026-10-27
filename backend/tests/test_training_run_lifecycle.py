from datetime import UTC, datetime, timedelta

import pytest
from test_training_run_execution import MemoryObjectStorage, _training_run_fixture

from al_medlit.core.exceptions import ConflictError
from al_medlit.model_artifacts.models import ArtifactStorageReservation
from al_medlit.training.run_execution import execute_training_run, reconcile_stale_training_runs
from al_medlit.workflow import models


@pytest.fixture(autouse=True)
def worker_image(monkeypatch):
    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)


@pytest.mark.parametrize("reservation_state", ["expired", "released"])
def test_failed_claim_closes_queued_run(db, reservation_state):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix=reservation_state)
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    if reservation_state == "expired":
        reservation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    else:
        from al_medlit.model_artifacts.quota import release_artifact_reservation

        release_artifact_reservation(db, reservation_id=reservation.id, reason="launch abandoned")
    db.commit()

    with pytest.raises(ConflictError):
        execute_training_run(db, storage, training_run_id=scope.run.id)

    db.expire_all()
    run = db.get(models.TrainingRun, scope.run.id)
    assert run.status == "failed"
    assert run.completed_at is not None
    assert db.get(ArtifactStorageReservation, reservation.id).status in {"expired", "released"}


def test_reconciler_closes_old_queued_run_and_releases_reservation(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="never-delivered")
    scope.run.created_at = datetime.now(UTC) - timedelta(days=8)
    db.commit()

    recovered = reconcile_stale_training_runs(db)

    assert recovered == [scope.run.id]
    db.expire_all()
    run = db.get(models.TrainingRun, scope.run.id)
    assert run.status == "failed"
    assert run.failure_code == "queue_wait_expired"
    assert db.get(ArtifactStorageReservation, run.artifact_reservation_id).status == "released"


def test_missing_reservation_closes_queued_run(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="missing")
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    from al_medlit.model_artifacts.quota import release_artifact_reservation

    release_artifact_reservation(db, reservation_id=reservation.id, reason="launch abandoned")
    scope.run.artifact_reservation_id = None
    db.flush()
    from sqlalchemy import delete

    db.execute(
        delete(ArtifactStorageReservation).where(ArtifactStorageReservation.id == reservation.id)
    )
    db.commit()

    with pytest.raises(ConflictError):
        execute_training_run(db, storage, training_run_id=scope.run.id)

    db.expire_all()
    assert db.get(models.TrainingRun, scope.run.id).status == "failed"


def test_queued_expired_reservation_recovers_before_queue_deadline(db):
    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="queued-expired")
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    reservation.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    db.commit()
    assert reconcile_stale_training_runs(db) == [scope.run.id]
    assert (
        db.get(models.TrainingRun, scope.run.id).failure_code == "artifact_reservation_unavailable"
    )


def test_recovery_limit_does_not_hide_stale_runs_behind_healthy_run(db):
    from al_medlit.training.run_execution import _claim_training_run

    storage = MemoryObjectStorage()
    healthy = _training_run_fixture(db, storage=storage, suffix="healthy-prefix")
    _claim_training_run(db, healthy.run.id)
    stale = _training_run_fixture(db, storage=storage, suffix="stale-suffix")
    _claim_training_run(db, stale.run.id)
    stale.run.runtime_snapshot = {
        **stale.run.runtime_snapshot,
        "worker_heartbeat_at": (datetime.now(UTC) - timedelta(minutes=10)).isoformat(),
    }
    db.commit()

    assert reconcile_stale_training_runs(db, limit=1) == [stale.run.id]
    assert db.get(models.TrainingRun, healthy.run.id).status == "running"


def test_foreign_attempt_cannot_resume_renew_or_fail_owned_run(db):
    from al_medlit.training.run_execution import (
        _claim_training_run,
        _mark_failed,
        heartbeat_training_run,
    )

    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="owned-attempt")
    run, claimed = _claim_training_run(db, scope.run.id, execution_token="owner")
    assert claimed
    heartbeat = run.runtime_snapshot["worker_heartbeat_at"]
    assert not heartbeat_training_run(db, run.id, execution_token="foreign")
    _mark_failed(db, run.id, RuntimeError("foreign"), execution_token="foreign")
    execute_training_run(db, storage, training_run_id=run.id, execution_token="foreign")
    duplicate, claimed = _claim_training_run(db, run.id, execution_token="duplicate")
    assert not claimed
    assert duplicate.status == "running"
    assert duplicate.runtime_snapshot["worker_execution_token"] == "owner"
    assert duplicate.runtime_snapshot["worker_heartbeat_at"] == heartbeat


def test_late_failure_preserves_cancelled_run(db):
    from al_medlit.training.run_execution import _claim_training_run, _mark_failed
    from al_medlit.workflow import schemas, service

    storage = MemoryObjectStorage()
    scope = _training_run_fixture(db, storage=storage, suffix="cancelled-attempt")
    _claim_training_run(db, scope.run.id, execution_token="owner")
    service.transition_training_run(
        db, scope.project.id, scope.run.id, schemas.TrainingRunTransition(status="cancelled")
    )
    _mark_failed(db, scope.run.id, RuntimeError("late failure"), execution_token="owner")
    assert db.get(models.TrainingRun, scope.run.id).status == "cancelled"


def test_dispatch_acknowledgement_failure_preserves_claim_at_transition_boundary(db, monkeypatch):
    from types import SimpleNamespace

    from al_medlit.training.run_execution import _claim_training_run
    from al_medlit.workflow import service
    from al_medlit.workflow.routes import training as routes

    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix="dispatch-race")
    monkeypatch.setattr(routes, "_write", lambda *args, **kwargs: None)
    monkeypatch.setattr(service, "create_training_run", lambda *args, **kwargs: scope.run)

    def accepted_without_acknowledgement(*args, **kwargs):
        raise RuntimeError("broker accepted delivery but acknowledgement was lost")

    monkeypatch.setattr(routes, "enqueue_training_run", accepted_without_acknowledgement)
    real_lock = service.lock_training_run_for_mutation
    claimed = False

    def worker_claim_before_recovery_lock(*args, **kwargs):
        nonlocal claimed
        if not claimed:
            claimed = True
            _claim_training_run(db, scope.run.id, execution_token="worker")
        return real_lock(*args, **kwargs)

    monkeypatch.setattr(
        service, "lock_training_run_for_mutation", worker_claim_before_recovery_lock
    )
    returned = routes.create_training_run(
        SimpleNamespace(project_id=scope.project.id),
        current_user=object(),
        db=db,
    )
    assert claimed
    assert returned.status == "running"
    reservation = db.get(ArtifactStorageReservation, returned.artifact_reservation_id)
    assert reservation.status == "active"
    assert returned.runtime_snapshot["worker_execution_token"] == "worker"
