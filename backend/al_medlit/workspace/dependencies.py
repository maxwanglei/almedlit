from collections.abc import Callable

from fastapi import Depends
from sqlalchemy.orm import Session

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, NotFoundError
from al_medlit.workspace import service
from al_medlit.workspace.models import Workspace, WorkspaceMember

ROLE_RANK = {
    "annotator": 0,
    "trainer": 1,
    "manager": 2,
    "admin": 3,
}
ROLE_ORDER = ROLE_RANK


def _role_satisfies(actual: str, required: tuple[str, ...]) -> bool:
    if not required:
        return True
    actual_rank = ROLE_RANK.get(actual, -1)
    return any(actual_rank >= ROLE_RANK.get(role, 99) for role in required)


def require_role(*roles: str) -> Callable[..., WorkspaceMember]:
    def _dependency(
        workspace_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> WorkspaceMember:
        if db.get(Workspace, workspace_id) is None:
            raise NotFoundError("Workspace not found")
        if current_user.is_superuser:
            return WorkspaceMember(
                workspace_id=workspace_id,
                user_id=current_user.id,
                role="admin",
            )
        member = service.get_member(db, workspace_id, current_user.id)
        if member is None:
            raise ForbiddenError("You are not a member of this workspace")
        if not _role_satisfies(member.role, roles):
            raise ForbiddenError("Insufficient workspace role")
        return member

    return _dependency
