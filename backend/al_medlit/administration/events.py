from sqlalchemy.orm import Session

from al_medlit.administration.models import AdminAuditEvent


def record_admin_event(
    db: Session,
    *,
    event_type: str,
    actor_user_id: int | None = None,
    target_user_id: int | None = None,
    workspace_id: int | None = None,
    details: dict | None = None,
) -> AdminAuditEvent:
    """Stage an immutable audit event in the caller's transaction."""

    audit_event = AdminAuditEvent(
        event_type=event_type,
        actor_user_id=actor_user_id,
        target_user_id=target_user_id,
        workspace_id=workspace_id,
        details=dict(details or {}),
    )
    db.add(audit_event)
    return audit_event
