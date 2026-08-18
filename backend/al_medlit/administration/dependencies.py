from fastapi import Depends

from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.core.exceptions import ForbiddenError


def require_active_superuser(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active or not current_user.is_superuser:
        raise ForbiddenError("Active system administrator access is required")
    return current_user
