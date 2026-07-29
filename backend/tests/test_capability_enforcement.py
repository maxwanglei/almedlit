def _make_user(db, username):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate

    user = auth_service.register_user(db, UserCreate(username=username, password="pw"))
    db.flush()
    return user


def _project_in_workspace(db, ws_id):
    from al_medlit.project.models import Project

    project = Project(name=f"proj-{ws_id}", workspace_id=ws_id)
    db.add(project)
    db.flush()
    return project


def test_effective_for_individual_annotate_workspace(db):
    from al_medlit.workspace import capability_service, service

    user = _make_user(db, "capuser1")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    eff = capability_service.effective_capabilities_for_workspace(db, ws.id)
    assert eff == {"annotation"}


def test_set_capability_changes_effective(db):
    from al_medlit.workspace import capability_service, service

    user = _make_user(db, "capuser2")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    capability_service.set_capability(
        db,
        ws.id,
        preset="annotate_train",
        actor_user_id=user.id,
    )
    db.flush()
    eff = capability_service.effective_capabilities_for_workspace(db, ws.id)
    assert eff == {"annotation", "lineage", "export", "training", "inference"}


def test_set_capability_rejects_unknown_preset(db):
    import pytest

    from al_medlit.core.exceptions import ValidationError
    from al_medlit.workspace import capability_service, service

    user = _make_user(db, "capuser3")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    with pytest.raises(ValidationError):
        capability_service.set_capability(
            db,
            ws.id,
            preset="nope",
            actor_user_id=user.id,
        )


def test_set_capability_revalidates_admin_after_workspace_lock(db, monkeypatch):
    import pytest
    from sqlalchemy.orm import Query

    from al_medlit.core.exceptions import ForbiddenError
    from al_medlit.workspace import capability_service, service
    from al_medlit.workspace.models import WorkspaceMember

    owner = _make_user(db, "cap-fresh-owner")
    manager = _make_user(db, "cap-fresh-admin")
    workspace = service.create_team_workspace(db, owner, name="Capability Fresh Auth")
    service.add_member(db, workspace.id, manager.id, role="admin")
    db.flush()

    lock_calls = []
    original_with_for_update = Query.with_for_update

    def recording_with_for_update(query, *args, **kwargs):
        lock_calls.append(query)
        return original_with_for_update(query, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", recording_with_for_update)
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace.id,
        WorkspaceMember.user_id == manager.id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)

    with pytest.raises(ForbiddenError, match="Insufficient workspace role"):
        capability_service.set_capability(
            db,
            workspace.id,
            preset="full",
            actor_user_id=manager.id,
        )
    assert len(lock_calls) == 1


def test_enforce_capability_allows_enabled(db):
    from al_medlit.workspace import capability_service, service
    from al_medlit.workspace.capability_dependencies import enforce_capability

    user = _make_user(db, "enf1")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    capability_service.set_capability(
        db,
        ws.id,
        preset="annotate_train",
        actor_user_id=user.id,
    )
    db.flush()
    project = _project_in_workspace(db, ws.id)
    enforce_capability(db, project_id=project.id, key="training")


def test_enforce_capability_blocks_disabled(db):
    import pytest

    from al_medlit.core.exceptions import ForbiddenError
    from al_medlit.workspace import service
    from al_medlit.workspace.capability_dependencies import enforce_capability

    user = _make_user(db, "enf2")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    project = _project_in_workspace(db, ws.id)
    with pytest.raises(ForbiddenError):
        enforce_capability(db, project_id=project.id, key="training")


def test_capabilities_api_read_and_update(auth_client, auth_user, db):
    ws_id = auth_user["workspace_id"]

    read = auth_client.get(f"/api/workspaces/{ws_id}/capabilities")
    assert read.status_code == 200
    assert read.json()["preset"] == "annotate"
    assert read.json()["effective"] == ["annotation"]

    upd = auth_client.patch(
        f"/api/workspaces/{ws_id}/capability",
        json={"preset": "annotate_train", "overrides": []},
    )
    assert upd.status_code == 200

    read2 = auth_client.get(f"/api/workspaces/{ws_id}/capabilities")
    assert set(read2.json()["effective"]) == {
        "annotation",
        "lineage",
        "export",
        "training",
        "inference",
    }


def test_colearning_list_blocked_when_capability_off(auth_client, auth_user, db):
    from al_medlit.project.models import Project

    project = Project(name="colearn-proj", workspace_id=auth_user["workspace_id"])
    db.add(project)
    db.commit()

    resp = auth_client.get(
        f"/api/co-learning/error-guideline/patterns?project_id={project.id}"
    )
    assert resp.status_code == 403


def test_colearning_list_allowed_when_capability_on(auth_client, auth_user, db):
    from al_medlit.project.models import Project
    from al_medlit.workspace import capability_service

    capability_service.set_capability(
        db,
        auth_user["workspace_id"],
        preset="full",
        actor_user_id=auth_user["id"],
    )
    db.commit()
    project = Project(name="colearn-proj2", workspace_id=auth_user["workspace_id"])
    db.add(project)
    db.commit()

    resp = auth_client.get(
        f"/api/co-learning/error-guideline/patterns?project_id={project.id}"
    )
    assert resp.status_code == 200
