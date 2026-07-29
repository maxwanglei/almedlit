from collections.abc import Callable

from fastapi import Depends, Query
from sqlalchemy.orm import Session

from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError
from al_medlit.project.models import Project
from al_medlit.workspace.capability_service import effective_capabilities_for_workspace


def _workspace_id_for_project(db: Session, project_id: int) -> int:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    if project.workspace_id is None:
        raise NotFoundError("Project is not attached to a workspace")
    return project.workspace_id


def enforce_capability(db: Session, *, project_id: int, key: str) -> None:
    workspace_id = _workspace_id_for_project(db, project_id)
    effective = effective_capabilities_for_workspace(db, workspace_id)
    if key not in effective:
        raise ForbiddenError(f"Capability '{key}' is not enabled for this workspace")


def require_capability(key: str) -> Callable[..., None]:
    def _dependency(
        project_id: int = Query(...),
        db: Session = Depends(get_db),
    ) -> None:
        enforce_capability(db, project_id=project_id, key=key)

    return _dependency
