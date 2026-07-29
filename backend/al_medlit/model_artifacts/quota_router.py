from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from al_medlit.core.database import get_db
from al_medlit.core.exceptions import NotFoundError
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts.models import ArtifactStorageReservation
from al_medlit.model_artifacts.schemas import (
    ArtifactStorageReservationRead,
    ArtifactStorageReservationRelease,
    WorkspaceArtifactQuotaRead,
    WorkspaceArtifactQuotaUpdate,
)
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.dependencies import require_role
from al_medlit.workspace.models import WorkspaceMember

router = APIRouter(prefix="/workspaces", tags=["artifact-quotas"])


def _snapshot(db: Session, workspace_id: int) -> WorkspaceArtifactQuotaRead:
    return WorkspaceArtifactQuotaRead.model_validate(
        quota.workspace_artifact_quota_snapshot(
            db,
            workspace_id=workspace_id,
        ),
        from_attributes=True,
    )


@router.get(
    "/{workspace_id}/artifact-quota",
    response_model=WorkspaceArtifactQuotaRead,
)
def get_workspace_artifact_quota(
    workspace_id: int,
    _trainer: WorkspaceMember = Depends(require_role("trainer")),
    db: Session = Depends(get_db),
):
    return _snapshot(db, workspace_id)


@router.patch(
    "/{workspace_id}/artifact-quota",
    response_model=WorkspaceArtifactQuotaRead,
)
def update_workspace_artifact_quota(
    workspace_id: int,
    payload: WorkspaceArtifactQuotaUpdate,
    manager: WorkspaceMember = Depends(require_role("manager")),
    db: Session = Depends(get_db),
):
    workspace_service.lock_workspace_for_update(db, workspace_id)
    workspace_service.require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=manager.user_id,
        minimum_role="manager",
    )
    current = quota.workspace_artifact_quota_snapshot(
        db,
        workspace_id=workspace_id,
    )
    quota.set_workspace_artifact_quota(
        db,
        workspace_id=workspace_id,
        limit_bytes=(
            payload.limit_bytes
            if "limit_bytes" in payload.model_fields_set
            else current.limit_bytes
        ),
        reservation_ttl_seconds=payload.reservation_ttl_seconds,
        actor_user_id=manager.user_id,
    )
    db.commit()
    return _snapshot(db, workspace_id)


@router.get(
    "/{workspace_id}/artifact-reservations",
    response_model=list[ArtifactStorageReservationRead],
)
def list_artifact_reservations(
    workspace_id: int,
    reservation_status: str | None = Query(default=None, alias="status"),
    _manager: WorkspaceMember = Depends(require_role("manager")),
    db: Session = Depends(get_db),
):
    return quota.list_workspace_artifact_reservations(
        db,
        workspace_id=workspace_id,
        status=reservation_status,
    )


@router.post(
    "/{workspace_id}/artifact-reservations/{reservation_id}/release",
    response_model=ArtifactStorageReservationRead,
)
def release_artifact_reservation(
    workspace_id: int,
    reservation_id: int,
    payload: ArtifactStorageReservationRelease,
    manager: WorkspaceMember = Depends(require_role("manager")),
    db: Session = Depends(get_db),
):
    workspace_service.lock_workspace_for_update(db, workspace_id)
    workspace_service.require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=manager.user_id,
        minimum_role="manager",
    )
    reservation = db.get(ArtifactStorageReservation, reservation_id)
    if reservation is None or reservation.workspace_id != workspace_id:
        raise NotFoundError("Artifact storage reservation not found")
    reservation = quota.release_artifact_reservation(
        db,
        reservation_id=reservation.id,
        reason=payload.reason,
    )
    db.commit()
    db.refresh(reservation)
    return reservation


@router.post(
    "/{workspace_id}/artifact-reservations/expire",
    response_model=WorkspaceArtifactQuotaRead,
)
def expire_artifact_reservations(
    workspace_id: int,
    manager: WorkspaceMember = Depends(require_role("manager")),
    db: Session = Depends(get_db),
):
    workspace_service.lock_workspace_for_update(db, workspace_id)
    workspace_service.require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=manager.user_id,
        minimum_role="manager",
    )
    quota.expire_artifact_reservations(db, workspace_id=workspace_id)
    db.commit()
    return _snapshot(db, workspace_id)
