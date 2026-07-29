from __future__ import annotations

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.security import hash_password
from al_medlit.core.config import settings
from al_medlit.core.database import SessionLocal, ensure_schema_ready
from al_medlit.workspace import service as workspace_service


def bootstrap_admin(db: Session) -> User:
    settings.validate_bootstrap_admin_password(require_configured=True)
    username = settings.bootstrap_admin_username.strip()
    if not username:
        raise RuntimeError("AL_MEDLIT_BOOTSTRAP_ADMIN_USERNAME must not be blank")
    password = settings.bootstrap_admin_password.strip()

    user = db.query(User).filter(User.username == username).first()
    if user is not None and not user.is_superuser:
        raise RuntimeError(
            f"Refusing to bootstrap {username!r}: that username belongs to an "
            "existing non-superuser account"
        )

    password_hash = hash_password(password)
    if user is None:
        user = User(
            username=username,
            email=None,
            password_hash=password_hash,
            display_name=username,
            is_active=True,
            is_superuser=True,
        )
        db.add(user)
        db.flush()
    else:
        user.password_hash = password_hash
        user.display_name = user.display_name or username
        user.is_active = True
        user.is_superuser = True
        db.flush()

    default_workspace = workspace_service.ensure_default_workspace(db)
    member = workspace_service.get_member(db, default_workspace.id, user.id)
    if member is None:
        workspace_service.add_member(db, default_workspace.id, user.id, role="admin")
    elif member.role != "admin":
        member.role = "admin"
        db.flush()

    db.commit()
    db.refresh(user)
    return user


def main() -> None:
    ensure_schema_ready()
    with SessionLocal() as db:
        user = bootstrap_admin(db)
    print(f"Bootstrap admin {user.username!r} is ready.")


if __name__ == "__main__":
    main()
