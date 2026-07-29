from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactStorageReservation,
    WorkspaceArtifactQuota,
)
from al_medlit.project.models import Project
from al_medlit.workspace.models import Workspace

DEFAULT_TRAINING_RESERVATION_BYTES = 1024 * 1024 * 1024
DEFAULT_RESERVATION_TTL_SECONDS = 7 * 24 * 60 * 60
_ACTIVE_STATUS = "active"
_TERMINAL_STATUSES = {"committed", "released", "expired"}


@dataclass(frozen=True, slots=True)
class WorkspaceArtifactQuotaSnapshot:
    workspace_id: int
    limit_bytes: int | None
    used_bytes: int
    reserved_bytes: int
    available_bytes: int | None
    reservation_ttl_seconds: int
    active_reservation_count: int
    next_expiration_at: datetime | None
    physical_bytes: int
    accounting_consistent: bool
    updated_at: datetime | None


@dataclass(frozen=True, slots=True)
class ReservationExpiryResult:
    expired_count: int
    released_bytes: int


def _physical_usage(db: Session, workspace_id: int) -> int:
    return int(
        db.query(func.coalesce(func.sum(ArtifactBlob.size_bytes), 0))
        .filter(
            ArtifactBlob.workspace_id == workspace_id,
            ArtifactBlob.status == "ready",
        )
        .scalar()
        or 0
    )


def _require_workspace(db: Session, workspace_id: int) -> Workspace:
    workspace = db.get(Workspace, workspace_id)
    if workspace is None:
        raise NotFoundError("Workspace not found")
    return workspace


def _project_workspace_id(db: Session, project_id: int) -> int:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project.workspace_id


def _locked_quota(db: Session, workspace_id: int) -> WorkspaceArtifactQuota:
    _require_workspace(db, workspace_id)
    quota = (
        db.query(WorkspaceArtifactQuota)
        .filter(WorkspaceArtifactQuota.workspace_id == workspace_id)
        .with_for_update()
        .one_or_none()
    )
    if quota is not None:
        return quota

    candidate = WorkspaceArtifactQuota(
        workspace_id=workspace_id,
        limit_bytes=None,
        used_bytes=_physical_usage(db, workspace_id),
        reserved_bytes=0,
        reservation_ttl_seconds=DEFAULT_RESERVATION_TTL_SECONDS,
    )
    try:
        with db.begin_nested():
            db.add(candidate)
            db.flush()
        return candidate
    except IntegrityError:
        return (
            db.query(WorkspaceArtifactQuota)
            .filter(WorkspaceArtifactQuota.workspace_id == workspace_id)
            .with_for_update()
            .one()
        )


def lock_workspace_artifact_quota(
    db: Session,
    *,
    workspace_id: int,
) -> WorkspaceArtifactQuota:
    return _locked_quota(db, workspace_id)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _expire_locked(
    db: Session,
    quota: WorkspaceArtifactQuota,
    *,
    now: datetime,
) -> ReservationExpiryResult:
    expired = (
        db.query(ArtifactStorageReservation)
        .filter(
            ArtifactStorageReservation.workspace_id == quota.workspace_id,
            ArtifactStorageReservation.status == _ACTIVE_STATUS,
            ArtifactStorageReservation.expires_at <= now,
        )
        .with_for_update()
        .all()
    )
    released_bytes = sum(item.reserved_bytes for item in expired)
    if released_bytes > quota.reserved_bytes:
        raise ConflictError("Workspace artifact reservation accounting is inconsistent")
    for reservation in expired:
        reservation.status = "expired"
        reservation.released_at = now
        reservation.release_reason = "Reservation lease expired"
    quota.reserved_bytes -= released_bytes
    return ReservationExpiryResult(
        expired_count=len(expired),
        released_bytes=released_bytes,
    )


def expire_artifact_reservations(
    db: Session,
    *,
    workspace_id: int,
    now: datetime | None = None,
) -> ReservationExpiryResult:
    quota = _locked_quota(db, workspace_id)
    result = _expire_locked(db, quota, now=now or datetime.now(UTC))
    db.flush()
    return result


def set_workspace_artifact_quota(
    db: Session,
    *,
    workspace_id: int,
    limit_bytes: int | None,
    reservation_ttl_seconds: int | None,
    actor_user_id: int | None,
) -> WorkspaceArtifactQuota:
    if limit_bytes is not None and limit_bytes < 0:
        raise ValidationError("Artifact quota limit_bytes cannot be negative")
    if reservation_ttl_seconds is not None and reservation_ttl_seconds <= 0:
        raise ValidationError("Artifact reservation TTL must be positive")
    quota = _locked_quota(db, workspace_id)
    _expire_locked(db, quota, now=datetime.now(UTC))
    if limit_bytes is not None and limit_bytes < quota.used_bytes + quota.reserved_bytes:
        raise ConflictError(
            "Artifact quota cannot be lower than current physical usage plus reservations"
        )
    quota.limit_bytes = limit_bytes
    if reservation_ttl_seconds is not None:
        quota.reservation_ttl_seconds = reservation_ttl_seconds
    quota.updated_by_user_id = actor_user_id
    db.flush()
    return quota


def reserve_artifact_bytes(
    db: Session,
    *,
    project_id: int,
    owner_type: str,
    owner_id: int,
    idempotency_key: str,
    requested_bytes: int | None,
    actor_user_id: int | None,
    now: datetime | None = None,
) -> ArtifactStorageReservation:
    if not owner_type or len(owner_type) > 40:
        raise ValidationError("Artifact reservation owner_type is invalid")
    if owner_id <= 0:
        raise ValidationError("Artifact reservation owner_id must be positive")
    normalized_key = idempotency_key.strip()
    if not normalized_key or len(normalized_key) > 200:
        raise ValidationError("Artifact reservation idempotency_key is invalid")
    amount = requested_bytes or DEFAULT_TRAINING_RESERVATION_BYTES
    if amount <= 0:
        raise ValidationError("Artifact reservation bytes must be positive")

    workspace_id = _project_workspace_id(db, project_id)
    quota = _locked_quota(db, workspace_id)
    effective_now = now or datetime.now(UTC)
    _expire_locked(db, quota, now=effective_now)
    existing = (
        db.query(ArtifactStorageReservation)
        .filter(
            ArtifactStorageReservation.workspace_id == workspace_id,
            ArtifactStorageReservation.idempotency_key == normalized_key,
        )
        .with_for_update()
        .one_or_none()
    )
    if existing is not None:
        if (
            existing.project_id != project_id
            or existing.owner_type != owner_type
            or existing.owner_id != owner_id
            or existing.reserved_bytes != amount
        ):
            raise ConflictError(
                "Artifact reservation idempotency key was reused with different inputs"
            )
        return existing

    projected = quota.used_bytes + quota.reserved_bytes + amount
    if quota.limit_bytes is not None and projected > quota.limit_bytes:
        available = max(0, quota.limit_bytes - quota.used_bytes - quota.reserved_bytes)
        raise ConflictError(
            f"Workspace artifact quota exceeded: requested {amount} bytes, "
            f"{available} bytes available"
        )
    reservation = ArtifactStorageReservation(
        workspace_id=workspace_id,
        project_id=project_id,
        owner_type=owner_type,
        owner_id=owner_id,
        idempotency_key=normalized_key,
        reserved_bytes=amount,
        committed_bytes=0,
        status=_ACTIVE_STATUS,
        expires_at=effective_now
        + timedelta(seconds=quota.reservation_ttl_seconds),
        created_by_user_id=actor_user_id,
    )
    db.add(reservation)
    quota.reserved_bytes += amount
    db.flush()
    return reservation


def _locked_reservation(
    db: Session,
    reservation_id: int,
) -> tuple[WorkspaceArtifactQuota, ArtifactStorageReservation]:
    reservation = db.get(ArtifactStorageReservation, reservation_id)
    if reservation is None:
        raise NotFoundError("Artifact storage reservation not found")
    quota = _locked_quota(db, reservation.workspace_id)
    reservation = (
        db.query(ArtifactStorageReservation)
        .filter(ArtifactStorageReservation.id == reservation_id)
        .with_for_update()
        .one()
    )
    return quota, reservation


def renew_artifact_reservation(
    db: Session,
    *,
    reservation_id: int,
    now: datetime | None = None,
) -> ArtifactStorageReservation:
    quota, reservation = _locked_reservation(db, reservation_id)
    effective_now = now or datetime.now(UTC)
    _expire_locked(db, quota, now=effective_now)
    if reservation.status != _ACTIVE_STATUS:
        raise ConflictError(
            f"Artifact reservation cannot be renewed from status '{reservation.status}'"
        )
    reservation.expires_at = effective_now + timedelta(
        seconds=quota.reservation_ttl_seconds
    )
    db.flush()
    return reservation


def release_artifact_reservation(
    db: Session,
    *,
    reservation_id: int,
    reason: str,
    now: datetime | None = None,
) -> ArtifactStorageReservation:
    normalized_reason = reason.strip()
    if not normalized_reason or len(normalized_reason) > 500:
        raise ValidationError("Artifact reservation release reason is invalid")
    quota, reservation = _locked_reservation(db, reservation_id)
    if reservation.status in _TERMINAL_STATUSES:
        return reservation
    if reservation.reserved_bytes > quota.reserved_bytes:
        raise ConflictError("Workspace artifact reservation accounting is inconsistent")
    quota.reserved_bytes -= reservation.reserved_bytes
    reservation.status = "released"
    reservation.released_at = now or datetime.now(UTC)
    reservation.release_reason = normalized_reason
    db.flush()
    return reservation


def release_owner_artifact_reservation(
    db: Session,
    *,
    owner_type: str,
    owner_id: int,
    reason: str,
) -> ArtifactStorageReservation | None:
    reservation = (
        db.query(ArtifactStorageReservation)
        .filter(
            ArtifactStorageReservation.owner_type == owner_type,
            ArtifactStorageReservation.owner_id == owner_id,
        )
        .one_or_none()
    )
    if reservation is None:
        return None
    return release_artifact_reservation(
        db,
        reservation_id=reservation.id,
        reason=reason,
    )


def renew_owner_artifact_reservation(
    db: Session,
    *,
    owner_type: str,
    owner_id: int,
) -> ArtifactStorageReservation | None:
    reservation = (
        db.query(ArtifactStorageReservation)
        .filter(
            ArtifactStorageReservation.owner_type == owner_type,
            ArtifactStorageReservation.owner_id == owner_id,
        )
        .one_or_none()
    )
    if reservation is None or reservation.status != _ACTIVE_STATUS:
        return reservation
    return renew_artifact_reservation(db, reservation_id=reservation.id)


def _candidate_sizes(
    candidates: Iterable[tuple[str, int]],
) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for checksum_sha256, size_bytes in candidates:
        if size_bytes < 0:
            raise ValidationError("Artifact candidate size cannot be negative")
        existing_size = sizes.get(checksum_sha256)
        if existing_size is not None and existing_size != size_bytes:
            raise ConflictError("Artifact checksum was paired with inconsistent sizes")
        sizes[checksum_sha256] = size_bytes
    return sizes


def admit_artifact_publication(
    db: Session,
    *,
    workspace_id: int,
    project_id: int,
    candidates: Iterable[tuple[str, int]],
    reservation_id: int | None = None,
) -> int:
    """Atomically convert a reservation into workspace-unique physical usage."""

    if _project_workspace_id(db, project_id) != workspace_id:
        raise ValidationError("Artifact publication project is not in the workspace")
    sizes = _candidate_sizes(candidates)
    quota = _locked_quota(db, workspace_id)
    effective_now = datetime.now(UTC)
    _expire_locked(db, quota, now=effective_now)
    existing_by_checksum = {
        blob.checksum_sha256: blob
        for blob in db.query(ArtifactBlob)
        .filter(
            ArtifactBlob.workspace_id == workspace_id,
            ArtifactBlob.checksum_sha256.in_(sizes),
        )
        .with_for_update()
        .all()
    }
    physical_delta = 0
    for checksum_sha256, size_bytes in sizes.items():
        blob = existing_by_checksum.get(checksum_sha256)
        if blob is not None and blob.size_bytes != size_bytes:
            raise ConflictError("Registered artifact blob size is inconsistent")
        if blob is None or blob.status == "purged":
            physical_delta += size_bytes

    reservation = None
    own_reserved_bytes = 0
    if reservation_id is not None:
        reservation = (
            db.query(ArtifactStorageReservation)
            .filter(ArtifactStorageReservation.id == reservation_id)
            .with_for_update()
            .one_or_none()
        )
        if (
            reservation is None
            or reservation.workspace_id != workspace_id
            or reservation.project_id != project_id
        ):
            raise NotFoundError("Artifact storage reservation not found for publication")
        if reservation.status == "committed" and physical_delta == 0:
            return 0
        if reservation.status != _ACTIVE_STATUS:
            raise ConflictError(
                f"Artifact reservation cannot be committed from status "
                f"'{reservation.status}'"
            )
        own_reserved_bytes = reservation.reserved_bytes

    projected = (
        quota.used_bytes
        + quota.reserved_bytes
        - own_reserved_bytes
        + physical_delta
    )
    if quota.limit_bytes is not None and projected > quota.limit_bytes:
        available = max(
            0,
            quota.limit_bytes
            - quota.used_bytes
            - quota.reserved_bytes
            + own_reserved_bytes,
        )
        raise ConflictError(
            "Workspace artifact quota exceeded during publication: "
            f"{physical_delta} physical bytes required, {available} bytes available"
        )

    quota.used_bytes += physical_delta
    if reservation is not None:
        if reservation.reserved_bytes > quota.reserved_bytes:
            raise ConflictError("Workspace artifact reservation accounting is inconsistent")
        quota.reserved_bytes -= reservation.reserved_bytes
        reservation.status = "committed"
        reservation.committed_bytes = physical_delta
        reservation.committed_at = effective_now
    db.flush()
    return physical_delta


def link_reservation_artifact_package(
    db: Session,
    *,
    reservation_id: int,
    artifact_package_id: int,
) -> ArtifactStorageReservation:
    reservation = (
        db.query(ArtifactStorageReservation)
        .filter(ArtifactStorageReservation.id == reservation_id)
        .with_for_update()
        .one_or_none()
    )
    if reservation is None:
        raise NotFoundError("Artifact storage reservation not found")
    if reservation.status != "committed":
        raise ConflictError("Only committed reservations can link an artifact package")
    if (
        reservation.artifact_package_id is not None
        and reservation.artifact_package_id != artifact_package_id
    ):
        raise ConflictError("Artifact reservation is already linked to another package")
    reservation.artifact_package_id = artifact_package_id
    db.flush()
    return reservation


def complete_owner_artifact_reservation(
    db: Session,
    *,
    owner_type: str,
    owner_id: int,
) -> ArtifactStorageReservation | None:
    reservation = (
        db.query(ArtifactStorageReservation)
        .filter(
            ArtifactStorageReservation.owner_type == owner_type,
            ArtifactStorageReservation.owner_id == owner_id,
        )
        .one_or_none()
    )
    if reservation is None or reservation.status != _ACTIVE_STATUS:
        return reservation
    quota, reservation = _locked_reservation(db, reservation.id)
    if reservation.reserved_bytes > quota.reserved_bytes:
        raise ConflictError("Workspace artifact reservation accounting is inconsistent")
    quota.reserved_bytes -= reservation.reserved_bytes
    reservation.status = "committed"
    reservation.committed_bytes = 0
    reservation.committed_at = datetime.now(UTC)
    db.flush()
    return reservation


def record_reclaimed_artifact_bytes(
    db: Session,
    *,
    workspace_id: int,
    reclaimed_bytes: int,
) -> WorkspaceArtifactQuota:
    if reclaimed_bytes < 0:
        raise ValidationError("Reclaimed artifact bytes cannot be negative")
    quota = _locked_quota(db, workspace_id)
    quota.used_bytes = max(0, quota.used_bytes - reclaimed_bytes)
    db.flush()
    return quota


def workspace_artifact_quota_snapshot(
    db: Session,
    *,
    workspace_id: int,
) -> WorkspaceArtifactQuotaSnapshot:
    _require_workspace(db, workspace_id)
    quota = (
        db.query(WorkspaceArtifactQuota)
        .filter(WorkspaceArtifactQuota.workspace_id == workspace_id)
        .one_or_none()
    )
    physical_bytes = _physical_usage(db, workspace_id)
    if quota is None:
        limit_bytes = None
        used_bytes = physical_bytes
        reserved_bytes = 0
        ttl_seconds = DEFAULT_RESERVATION_TTL_SECONDS
        updated_at = None
    else:
        limit_bytes = quota.limit_bytes
        used_bytes = quota.used_bytes
        reserved_bytes = quota.reserved_bytes
        ttl_seconds = quota.reservation_ttl_seconds
        updated_at = quota.updated_at
    active_query = db.query(ArtifactStorageReservation).filter(
        ArtifactStorageReservation.workspace_id == workspace_id,
        ArtifactStorageReservation.status == _ACTIVE_STATUS,
    )
    active_count = active_query.count()
    next_expiration = active_query.with_entities(
        func.min(ArtifactStorageReservation.expires_at)
    ).scalar()
    available = (
        None
        if limit_bytes is None
        else max(0, limit_bytes - used_bytes - reserved_bytes)
    )
    return WorkspaceArtifactQuotaSnapshot(
        workspace_id=workspace_id,
        limit_bytes=limit_bytes,
        used_bytes=used_bytes,
        reserved_bytes=reserved_bytes,
        available_bytes=available,
        reservation_ttl_seconds=ttl_seconds,
        active_reservation_count=active_count,
        next_expiration_at=next_expiration,
        physical_bytes=physical_bytes,
        accounting_consistent=physical_bytes == used_bytes,
        updated_at=updated_at,
    )


def list_workspace_artifact_reservations(
    db: Session,
    *,
    workspace_id: int,
    status: str | None = None,
) -> list[ArtifactStorageReservation]:
    _require_workspace(db, workspace_id)
    query = db.query(ArtifactStorageReservation).filter(
        ArtifactStorageReservation.workspace_id == workspace_id
    )
    if status is not None:
        query = query.filter(ArtifactStorageReservation.status == status)
    return query.order_by(
        ArtifactStorageReservation.created_at.desc(),
        ArtifactStorageReservation.id.desc(),
    ).all()
