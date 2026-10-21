import pytest
from test_artifact_quotas import _headers, _package_files, _package_spec, _workspace_projects
from test_login_hardening import _invite_login, _login, _make_user, _open_invite

from al_medlit.core.config import Settings, settings
from al_medlit.model_artifacts import quota
from al_medlit.model_artifacts.service import publish_artifact_package


@pytest.mark.parametrize("endpoint", ["login", "invite"])
def test_valid_owner_can_login_during_continuous_bad_attempts(client, db, monkeypatch, endpoint):
    monkeypatch.setattr(settings, "login_failure_threshold", 2)
    user = _make_user(db, "sole-admin-under-attack", is_superuser=True)
    invite = _open_invite(db)
    for _ in range(2):
        assert _login(client, user.username, "wrong").status_code == 401
    assert _login(client, user.username, "wrong").status_code == 429

    response = (
        _login(client, user.username, "a-secure-password")
        if endpoint == "login"
        else _invite_login(client, invite.token, user.username, "a-secure-password")
    )

    assert response.status_code == 200, response.text


def test_failed_login_status_does_not_reveal_account_existence(client, db, monkeypatch):
    monkeypatch.setattr(settings, "login_failure_threshold", 2)
    _make_user(db, "existing-account")
    db.commit()

    known = [_login(client, "existing-account", "wrong").status_code for _ in range(4)]
    unknown = [_login(client, "missing-account", "wrong").status_code for _ in range(4)]

    assert known == unknown == [401, 401, 429, 429]


@pytest.mark.parametrize(
    ("license_info", "trainer_status", "manager_status"),
    [
        ({"base_model": {"access_mode": "execution_only"}}, 403, 403),
        ({"base_model": {"access_mode": "manager_only"}}, 404, 200),
        ({"access_mode": "downloadable", "base_model": {"access_mode": "execution_only"}},
         403, 403),
        ({"base_model": {"base_model": {"access_mode": "execution_only"}}}, 403, 403),
        ({"access_mode": "manager_only", "base_model": {"access_mode": "execution_only"}},
         404, 403),
    ],
)
def test_derived_package_preserves_base_download_restrictions(
    db, client, object_storage, license_info, trainer_status, manager_status
):
    manager, trainer, _annotator, workspace, project, _second = _workspace_projects(db)
    workspace.capability_preset = "full"
    db.commit()
    package = publish_artifact_package(
        db, object_storage, project_id=project.id,
        data=_package_spec("Derived model").model_copy(update={"license_info": license_info}),
        files=_package_files(), actor_user_id=manager.id,
    )
    db.commit()
    path = f"/api/artifact-packages/{package.id}/files/{package.files[0].id}/download"

    assert client.get(path, headers=_headers(trainer)).status_code == trainer_status
    assert client.get(path, headers=_headers(manager)).status_code == manager_status


def test_public_signup_is_disabled_by_default(client, db, monkeypatch):
    monkeypatch.delenv("AL_MEDLIT_ALLOW_SELF_REGISTRATION", raising=False)
    monkeypatch.setattr(settings, "allow_self_registration", Settings().allow_self_registration)
    response = client.post(
        "/api/auth/register",
        json={"username": "unapproved-signup", "password": "a-secure-password"},
    )
    assert response.status_code == 403


def test_new_workspace_receives_a_finite_artifact_budget(db):
    _manager, _trainer, _annotator, workspace, _first, _second = _workspace_projects(db)
    snapshot = quota.workspace_artifact_quota_snapshot(db, workspace_id=workspace.id)
    assert snapshot.limit_bytes is not None
    assert 0 < snapshot.limit_bytes <= 64 * 1024**3


@pytest.mark.parametrize("requested", [None, 2048])
def test_workspace_admin_cannot_expand_instance_storage_allocation(db, client, requested):
    manager, _trainer, _annotator, workspace, _first, _second = _workspace_projects(db)
    quota.set_workspace_artifact_quota(
        db, workspace_id=workspace.id, limit_bytes=1024,
        reservation_ttl_seconds=None, actor_user_id=manager.id,
    )
    db.commit()

    response = client.patch(
        f"/api/workspaces/{workspace.id}/artifact-quota",
        json={"limit_bytes": requested}, headers=_headers(manager),
    )

    assert response.status_code == 403


def test_only_instance_admin_can_raise_storage_cap(db, client):
    manager, _trainer, _annotator, workspace, _first, _second = _workspace_projects(db)
    manager.is_superuser = True
    quota.set_workspace_artifact_quota(
        db, workspace_id=workspace.id, limit_bytes=1024,
        reservation_ttl_seconds=None, actor_user_id=manager.id,
    )
    db.commit()
    response = client.patch(
        f"/api/workspaces/{workspace.id}/artifact-quota",
        json={"limit_bytes": None}, headers=_headers(manager),
    )
    assert response.status_code == 200
    assert response.json()["limit_bytes"] is None


def test_default_budget_is_enforced_before_artifact_reservation(db, monkeypatch):
    from al_medlit.core.exceptions import ConflictError

    monkeypatch.setattr(settings, "default_workspace_artifact_quota_bytes", 1024)
    manager, _trainer, _annotator, workspace, first, _second = _workspace_projects(db)
    with pytest.raises(ConflictError, match="quota"):
        quota.reserve_artifact_bytes(
            db, project_id=first.id, owner_type="test", owner_id=1,
            idempotency_key="above-default", requested_bytes=1025,
            actor_user_id=manager.id,
        )
    assert quota.workspace_artifact_quota_snapshot(
        db, workspace_id=workspace.id,
    ).reserved_bytes == 0


def test_registration_can_be_explicitly_enabled(client, monkeypatch):
    monkeypatch.setattr(settings, "allow_self_registration", True)
    response = client.post(
        "/api/auth/register",
        json={"username": "approved-public-signup", "password": "a-secure-password"},
    )
    assert response.status_code == 200
