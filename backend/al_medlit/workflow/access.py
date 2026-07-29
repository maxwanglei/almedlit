"""Canonical project authorization and module availability."""

from __future__ import annotations

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    assert_project_member,
    assert_workspace_member,
    lock_project_member_for_mutation,
)
from al_medlit.core.exceptions import ForbiddenError, NotFoundError, ValidationError
from al_medlit.project.models import Project
from al_medlit.workspace.capability_service import effective_capabilities_for_workspace
from al_medlit.workspace.models import WorkspaceMember

PROJECT_MODULES = (
    "data",
    "annotate",
    "learning",
    "train",
    "models",
    "guidelines",
    "activity",
)

_WORKSPACE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "data": ("annotation", "training", "active_learning", "co_learning", "lineage"),
    "annotate": ("annotation",),
    "learning": ("active_learning", "co_learning"),
    "train": ("training",),
    "models": ("training", "inference", "lineage"),
    "guidelines": ("co_learning",),
    "activity": ("lineage",),
}
_MODULE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "data": (),
    "annotate": ("data",),
    "learning": ("data", "annotate"),
    "train": ("data", "models"),
    "models": (),
    "guidelines": ("data", "annotate"),
    "activity": (),
}


def _project(db: Session, project_id: int) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise NotFoundError("Project not found")
    return project


def selected_project_modules(project: Project) -> set[str]:
    configured = project.settings.get("modules")
    if configured is None:
        return set(PROJECT_MODULES)
    if not isinstance(configured, list):
        return set()
    return close_project_module_dependencies(
        {item for item in configured if item in PROJECT_MODULES}
    )


def close_project_module_dependencies(selected: set[str]) -> set[str]:
    result = set(selected)
    changed = True
    while changed:
        changed = False
        for module in tuple(result):
            for dependency in _MODULE_DEPENDENCIES[module]:
                if dependency not in result:
                    result.add(dependency)
                    changed = True
    return result


def effective_project_modules(db: Session, project_id: int) -> set[str]:
    project = _project(db, project_id)
    selected = selected_project_modules(project)
    workspace_capabilities = effective_capabilities_for_workspace(
        db,
        project.workspace_id,
    )
    available = {
        module
        for module in selected
        if any(
            capability in workspace_capabilities
            for capability in _WORKSPACE_REQUIREMENTS[module]
        )
    }
    return {
        module
        for module in available
        if all(dependency in available for dependency in _MODULE_DEPENDENCIES[module])
    }


def module_configuration(db: Session, project_id: int) -> dict:
    project = _project(db, project_id)
    workspace_capabilities = effective_capabilities_for_workspace(
        db,
        project.workspace_id,
    )
    selected = selected_project_modules(project)
    effective = effective_project_modules(db, project_id)
    return {
        "project_id": project.id,
        "selected": [module for module in PROJECT_MODULES if module in selected],
        "effective": [module for module in PROJECT_MODULES if module in effective],
        "workspace_capabilities": sorted(workspace_capabilities),
    }


def update_project_modules(
    db: Session,
    project_id: int,
    selected: list[str],
) -> dict:
    project = _project(db, project_id)
    if len(selected) != len(set(selected)):
        raise ValidationError("Project modules must be unique")
    unknown = sorted(set(selected) - set(PROJECT_MODULES))
    if unknown:
        raise ValidationError(f"Unknown project modules: {', '.join(unknown)}")
    selected_set = close_project_module_dependencies(set(selected))
    settings = dict(project.settings)
    settings["modules"] = [
        module for module in PROJECT_MODULES if module in selected_set
    ]
    project.settings = settings
    db.commit()
    db.refresh(project)
    return module_configuration(db, project_id)


def enforce_project_module(
    db: Session,
    project_id: int,
    module: str | tuple[str, ...],
) -> None:
    requested = (module,) if isinstance(module, str) else module
    unknown = set(requested) - set(PROJECT_MODULES)
    if unknown:
        raise RuntimeError(f"Unknown platform modules: {sorted(unknown)}")
    if not set(requested).intersection(effective_project_modules(db, project_id)):
        raise ForbiddenError(
            "None of the required project modules are enabled or available: "
            + ", ".join(requested)
        )


def authorize_project_read(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = None,
    module: str | tuple[str, ...],
) -> None:
    """Apply the canonical membership, role, capability, and module policy."""

    assert_project_member(db, user, project_id, min_role=min_role)
    enforce_project_module(db, project_id, module)


def authorize_project_write(
    db: Session,
    user: User,
    project_id: int,
    *,
    min_role: str | None = "manager",
    module: str | tuple[str, ...],
) -> WorkspaceMember:
    """Authorize a mutation while holding the actor's membership lock."""

    member = lock_project_member_for_mutation(
        db,
        user,
        project_id,
        min_role=min_role,
    )
    enforce_project_module(db, project_id, module)
    return member


def authorize_workspace_capability(
    db: Session,
    user: User,
    workspace_id: int,
    *,
    capability: str,
    min_role: str | None = None,
) -> None:
    """Apply the same workspace membership and capability policy to aggregates."""

    assert_workspace_member(db, user, workspace_id, min_role=min_role)
    if capability not in effective_capabilities_for_workspace(db, workspace_id):
        raise ForbiddenError(
            f"Capability '{capability}' is not enabled for this workspace"
        )
