from sqlalchemy.orm import Session

from al_medlit.core import capabilities as cap
from al_medlit.core.config import settings
from al_medlit.core.exceptions import NotFoundError, ValidationError
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.models import Workspace


def effective_capabilities_for_workspace(db: Session, workspace_id: int) -> set[str]:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise NotFoundError("Workspace not found")
    selected = cap.preset_keys(ws.capability_preset, ws.capability_overrides or [])
    return cap.effective_capabilities(
        selected,
        profile=settings.deployment_profile,
        workspace_kind=ws.kind,
    )


def set_capability(
    db: Session,
    workspace_id: int,
    *,
    preset: str,
    overrides: list[str] | None = None,
    actor_user_id: int,
) -> Workspace:
    # Capability updates are workspace mutations too. Serialize them with
    # membership changes and re-check the administrator after the lock, rather than
    # trusting a potentially stale FastAPI dependency result.
    ws = workspace_service.lock_workspace_for_update(db, workspace_id)
    workspace_service.require_actor_role_after_workspace_lock(
        db,
        workspace_id,
        actor_user_id=actor_user_id,
        minimum_role="admin",
    )

    valid_presets = set(cap.PRESETS) | {"custom"}
    if preset not in valid_presets:
        raise ValidationError(f"Unknown preset '{preset}'")
    if preset == "custom":
        unknown = set(overrides or []) - set(cap.CAPABILITIES)
        if unknown:
            raise ValidationError(f"Unknown capability keys: {sorted(unknown)}")

    ws.capability_preset = preset
    ws.capability_overrides = list(overrides or [])
    db.flush()
    return ws
