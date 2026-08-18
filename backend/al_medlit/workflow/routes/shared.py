"""HTTP routes for the canonical learning workflow."""

from collections.abc import Iterable

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.workflow import access


def _read(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = None,
    module: str | tuple[str, ...],
) -> None:
    access.authorize_project_read(
        db,
        user,
        project_id,
        min_role=min_role,
        module=module,
    )


def _write(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = "manager",
    module: str | tuple[str, ...],
    related_user_ids: Iterable[int] = (),
) -> None:
    access.authorize_project_write(
        db,
        user,
        project_id,
        min_role=min_role,
        module=module,
        related_user_ids=related_user_ids,
    )


def _workspace_training_read(
    db: Session,
    user: User,
    workspace_id: int,
) -> None:
    access.authorize_workspace_capability(
        db,
        user,
        workspace_id,
        capability="training",
        min_role="trainer",
    )


def _workspace_annotation_read(
    db: Session,
    user: User,
    workspace_id: int,
) -> None:
    access.authorize_workspace_capability(
        db,
        user,
        workspace_id,
        capability="annotation",
    )
