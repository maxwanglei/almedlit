"""Resolving the default workspace must preserve administrator capability policy."""

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ForbiddenError
from al_medlit.project import service as project_service
from al_medlit.project.schemas import ProjectCreate
from al_medlit.workspace import service
from al_medlit.workspace.capability_dependencies import enforce_capability


@pytest.fixture
def default_workspace(db):
    admin = User(username="default-admin", password_hash="test-only", is_active=True)
    manager = User(username="default-manager", password_hash="test-only", is_active=True)
    db.add_all([admin, manager])
    db.flush()
    workspace = service.ensure_default_workspace(db)
    service.add_member(db, workspace.id, admin.id, role="admin")
    service.add_member(db, workspace.id, manager.id, role="manager")
    db.commit()
    return (
        workspace,
        {"Authorization": f"Bearer {create_access_token(str(admin.id))}"},
        {"Authorization": f"Bearer {create_access_token(str(manager.id))}"},
    )


@pytest.mark.parametrize("workspace_reference", ["omitted", "null", "explicit"])
@pytest.mark.parametrize("policy", [
    {"preset": "annotate", "overrides": []},
    {"preset": "custom", "overrides": ["annotation", "export"]},
])
def test_manager_project_creation_preserves_default_workspace_restrictions(
    client, db, default_workspace, workspace_reference, policy,
):
    workspace, admin_headers, manager_headers = default_workspace
    capability_url = f"/api/workspaces/{workspace.id}/capability"
    configured = client.patch(capability_url, headers=admin_headers, json=policy)
    assert configured.status_code == 200, configured.text
    forbidden = client.patch(
        capability_url, headers=manager_headers, json={"preset": "full", "overrides": []},
    )
    assert forbidden.status_code == 403

    payload = {"name": "Manager project"}
    if workspace_reference != "omitted":
        payload["workspace_id"] = workspace.id if workspace_reference == "explicit" else None
    created = client.post("/api/projects", headers=manager_headers, json=payload)
    assert created.status_code == 200, created.text
    assert created.json()["workspace_id"] == workspace.id

    actual = client.get(f"/api/workspaces/{workspace.id}/capabilities", headers=manager_headers)
    assert actual.status_code == 200, actual.text
    assert actual.json() == configured.json()
    db.refresh(workspace)
    assert workspace.capability_preset == policy["preset"]
    assert workspace.capability_overrides == policy["overrides"]
    with pytest.raises(ForbiddenError):
        enforce_capability(db, project_id=created.json()["id"], key="training")


def test_project_service_fallback_preserves_existing_default_policy(db, default_workspace):
    workspace, _admin, _manager = default_workspace
    workspace.capability_preset = "custom"
    workspace.capability_overrides = ["annotation"]
    db.commit()
    project = project_service.create_project(db, ProjectCreate(name="Service project"))
    assert project.workspace_id == workspace.id
    db.refresh(workspace)
    assert workspace.capability_preset == "custom"
    assert workspace.capability_overrides == ["annotation"]


def test_new_default_workspace_initializes_full_capabilities_once(db):
    workspace = service.ensure_default_workspace(db)
    assert workspace.capability_preset == "full"
    assert workspace.capability_overrides == []
    workspace.capability_preset = "annotate"
    db.commit()
    again = service.ensure_default_workspace(db)
    assert again.id == workspace.id
    assert again.capability_preset == "annotate"
