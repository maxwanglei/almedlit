import pytest


def _make_user(db, username):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate

    user = auth_service.register_user(db, UserCreate(username=username, password="pw"))
    db.flush()
    return user


def test_workspace_models_persist(db):
    from al_medlit.auth.models import User
    from al_medlit.workspace.models import Workspace, WorkspaceMember

    user = User(username="bob", password_hash="x")
    db.add(user)
    db.flush()

    ws = Workspace(name="Bob's space", kind="individual", created_by=user.id)
    db.add(ws)
    db.flush()

    member = WorkspaceMember(workspace_id=ws.id, user_id=user.id, role="admin")
    db.add(member)
    db.flush()

    assert ws.id is not None
    assert member.role == "admin"
    assert ws.kind == "individual"


def test_create_personal_workspace_makes_admin_member(db):
    from al_medlit.workspace import service

    user = _make_user(db, "ivy")
    ws = service.create_personal_workspace(db, user)
    db.flush()
    assert ws.kind == "individual"
    member = service.get_member(db, ws.id, user.id)
    assert member is not None
    assert member.role == "admin"


def test_create_personal_workspace_allows_shared_display_name_across_users(db):
    """Display names are tenant labels, not a global namespace."""
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.workspace import service

    u1 = auth_service.register_user(
        db, UserCreate(username="alice", password="pw", display_name="Sam")
    )
    u2 = auth_service.register_user(
        db, UserCreate(username="bob", password="pw", display_name="Sam")
    )
    db.flush()

    ws1 = service.create_personal_workspace(db, u1)
    ws2 = service.create_personal_workspace(db, u2)
    db.flush()

    assert ws1.name == "Sam's Workspace"
    assert ws2.name == "Sam's Workspace"
    # Both users still own an admin membership in their own workspace.
    assert service.get_member(db, ws2.id, u2.id).role == "admin"


def test_create_team_workspace_has_join_code(db):
    from al_medlit.workspace import service

    user = _make_user(db, "jack")
    ws = service.create_team_workspace(db, user, name="Team A")
    db.flush()
    assert ws.kind == "team"
    assert ws.join_code
    assert service.get_member(db, ws.id, user.id).role == "admin"


def test_register_team_workspace_always_makes_creator_admin(client, db):
    from al_medlit.auth import service as auth_service
    from al_medlit.workspace import service as workspace_service

    resp = client.post(
        "/api/auth/register",
        json={
            "username": "role-register",
            "password": "strong-password",
            "display_name": "Role Register",
            "workspace_kind": "team",
            "workspace_name": "Role Register Team",
            "role": "manager",
        },
    )

    assert resp.status_code == 200
    user = auth_service.get_user_by_username(db, "role-register")
    assert user is not None
    workspaces = workspace_service.list_workspaces_for_user(db, user)
    assert len(workspaces) == 1
    assert workspaces[0].kind == "team"
    assert workspaces[0].name == "Role Register Team"
    member = workspace_service.get_member(db, workspaces[0].id, user.id)
    assert member is not None
    assert member.role == "admin"


def test_team_names_are_creator_scoped_but_same_owner_duplicates_conflict(client, db):
    from al_medlit.auth import service as auth_service
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.workspace import service as workspace_service

    first = client.post(
        "/api/auth/register",
        json={
            "username": "team-name-owner",
            "password": "strong-password",
            "workspace_kind": "team",
            "workspace_name": "Shared Team Name",
        },
    )
    assert first.status_code == 200, first.text

    second_owner = client.post(
        "/api/auth/register",
        json={
            "username": "second-team-name-owner",
            "password": "strong-password",
            "workspace_kind": "team",
            "workspace_name": "Shared Team Name",
        },
    )
    assert second_owner.status_code == 200, second_owner.text
    first_user = auth_service.get_user_by_username(db, "team-name-owner")
    second_user = auth_service.get_user_by_username(db, "second-team-name-owner")
    assert first_user is not None
    assert second_user is not None
    assert [ws.name for ws in workspace_service.list_workspaces_for_user(db, second_user)] == [
        "Shared Team Name"
    ]

    with pytest.raises(ConflictError, match="already have"):
        workspace_service.create_team_workspace(db, first_user, "Shared Team Name")


def test_project_names_are_unique_within_each_workspace(db):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import ProjectCreate, ProjectUpdate
    from al_medlit.workspace import service as workspace_service

    owner_a = _make_user(db, "project-name-owner-a")
    owner_b = _make_user(db, "project-name-owner-b")
    workspace_a = workspace_service.create_team_workspace(
        db,
        owner_a,
        "Shared Workspace Label",
    )
    workspace_b = workspace_service.create_team_workspace(
        db,
        owner_b,
        "Shared Workspace Label",
    )

    project_service.create_project(
        db,
        ProjectCreate(name="Shared Project", workspace_id=workspace_a.id),
    )
    project_b = project_service.create_project(
        db,
        ProjectCreate(name="Other Project", workspace_id=workspace_b.id),
    )
    renamed = project_service.update_project(
        db,
        project_b.id,
        ProjectUpdate(name="Shared Project"),
    )
    assert renamed.name == "Shared Project"

    with pytest.raises(ConflictError, match="already exists"):
        project_service.create_project(
            db,
            ProjectCreate(name="Shared Project", workspace_id=workspace_a.id),
        )


def test_inactive_members_cannot_receive_new_or_reassigned_work(db):
    from al_medlit.core.exceptions import ValidationError
    from al_medlit.corpus.models import Document
    from al_medlit.project import service as project_service
    from al_medlit.project.models import Project, ProjectTask
    from al_medlit.project.schemas import TaskAssignmentCreate, TaskAssignmentUpdate
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "inactive-assignment-owner")
    active_user = _make_user(db, "active-assignee")
    inactive_user = _make_user(db, "inactive-assignee")
    workspace = workspace_service.create_team_workspace(
        db,
        owner,
        "Inactive Assignment Team",
    )
    workspace_service.add_member(db, workspace.id, active_user.id, role="annotator")
    workspace_service.add_member(db, workspace.id, inactive_user.id, role="annotator")
    inactive_user.is_active = False
    project = Project(name="inactive-assignment-project", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    task = ProjectTask(
        project_id=project.id,
        annotation_type="entity",
        display_name="Entities",
        enabled=True,
    )
    document = Document(project_id=project.id, title="Inactive", text="Text")
    db.add_all([task, document])
    db.flush()

    with pytest.raises(ValidationError, match="must be active"):
        project_service.create_task_assignment(
            db,
            project.id,
            TaskAssignmentCreate(
                task_id=task.id,
                document_id=document.id,
                assignee_user_id=inactive_user.id,
            ),
            assigned_by_user=owner,
        )

    assignment = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=document.id,
            assignee_user_id=active_user.id,
        ),
        assigned_by_user=owner,
    )
    with pytest.raises(ValidationError, match="must be active"):
        project_service.update_task_assignment(
            db,
            project.id,
            assignment.id,
            TaskAssignmentUpdate(assignee_user_id=inactive_user.id),
        )


def test_register_keeps_legacy_individual_admin_default(client, db):
    from al_medlit.auth import service as auth_service
    from al_medlit.workspace import service as workspace_service

    resp = client.post(
        "/api/auth/register",
        json={"username": "legacy-register", "password": "strong-password"},
    )

    assert resp.status_code == 200
    user = auth_service.get_user_by_username(db, "legacy-register")
    assert user is not None
    workspaces = workspace_service.list_workspaces_for_user(db, user)
    assert len(workspaces) == 1
    assert workspaces[0].kind == "individual"
    member = workspace_service.get_member(db, workspaces[0].id, user.id)
    assert member is not None
    assert member.role == "admin"


def test_personal_submission_creates_missing_self_assignment(auth_client, auth_user, db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate

    project_response = auth_client.post(
        "/api/projects",
        json={
            "name": "personal-submission-self-assignment",
            "workspace_id": auth_user["workspace_id"],
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Finding", "color": "#2563eb"}],
                }
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project_id = project_response.json()["id"]

    document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project_id,
            title="Legacy personal document",
            text="Aspirin reduced fever.",
        ),
    )

    assignments_before = auth_client.get(
        f"/api/projects/{project_id}/assignments",
        params={"scope": "mine"},
    )
    assert assignments_before.status_code == 200
    assert assignments_before.json() == []

    submission_response = auth_client.post(
        f"/api/projects/{project_id}/documents/{document.id}/submissions",
        json={"kind": "submission"},
    )
    assert submission_response.status_code == 200

    assignments_after = auth_client.get(
        f"/api/projects/{project_id}/assignments",
        params={"scope": "mine"},
    )
    assert assignments_after.status_code == 200
    assignment_body = assignments_after.json()
    assert len(assignment_body) == 1
    assert assignment_body[0]["document_id"] == document.id
    assert assignment_body[0]["annotator_id"] == auth_user["username"]
    assert assignment_body[0]["status"] == "submitted"
    assert assignment_body[0]["metadata_"]["source"] == "personal_workspace"

    from al_medlit.auth.models import User
    from al_medlit.corpus import service as corpus_service
    from al_medlit.guideline import service as guideline_service
    from al_medlit.guideline.schemas import GuidelineVersionCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.models import TaskAssignment

    guideline = guideline_service.create_guideline_version(
        db,
        GuidelineVersionCreate(
            project_id=project_id,
            version_label="v2",
            markdown="A new local annotation round.",
        ),
    )
    structure = corpus_service.rebuild_document_structure(db, document.id)
    user = db.get(User, auth_user["id"])
    created = project_service.ensure_personal_project_self_assignments(
        db,
        project_id,
        user,
        document_ids=[document.id],
    )
    assert len(created) == 1
    assert created[0].guideline_version_id == guideline.id
    assert created[0].structure_version_id == structure.id
    assert created[0].assignment_scope_key == (
        f"document:structure:{structure.id}:guideline:{guideline.id}"
    )
    assert (
        project_service.ensure_personal_project_self_assignments(
            db,
            project_id,
            user,
            document_ids=[document.id],
        )
        == []
    )
    assert (
        db.query(TaskAssignment)
        .filter(
            TaskAssignment.project_id == project_id,
            TaskAssignment.document_id == document.id,
            TaskAssignment.assignee_user_id == user.id,
        )
        .count()
        == 2
    )


def test_personal_owner_can_reopen_edit_and_resubmit_while_snapshots_remain(
    auth_client,
    auth_user,
):
    project = auth_client.post(
        "/api/projects",
        json={
            "name": "personal-reopen-paper-task",
            "workspace_id": auth_user["workspace_id"],
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Clinical findings",
                    "labels": [{"name": "Finding", "color": "#2563eb"}],
                }
            ],
        },
    ).json()
    document = auth_client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Paper to correct",
            "text": "Aspirin reduced fever.",
        },
    ).json()
    assignment_url = f"/api/projects/{project['id']}/assignments"
    assignment = auth_client.get(
        assignment_url,
        params={"scope": "mine"},
    ).json()[0]

    first_submission = auth_client.post(
        f"/api/projects/{project['id']}/documents/{document['id']}/submissions",
        json={"kind": "submission", "assignment_id": assignment["id"]},
    )
    assert first_submission.status_code == 200, first_submission.text
    assert first_submission.json()["annotation_count"] == 0

    reopened = auth_client.post(
        f"{assignment_url}/{assignment['id']}/reopen",
    )
    assert reopened.status_code == 200, reopened.text
    reopened_body = reopened.json()
    assert reopened_body["id"] == assignment["id"]
    assert reopened_body["status"] == "in_progress"
    assert reopened_body["metadata_"]["reopen_history"][0]["from_status"] == "submitted"
    assert reopened_body["metadata_"]["reopen_history"][0]["actor_user_id"] == auth_user["id"]

    duplicate_reopen = auth_client.post(
        f"{assignment_url}/{assignment['id']}/reopen",
    )
    assert duplicate_reopen.status_code == 409

    created_annotation = auth_client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Finding",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
        },
    )
    assert created_annotation.status_code == 200, created_annotation.text

    second_submission = auth_client.post(
        f"/api/projects/{project['id']}/documents/{document['id']}/submissions",
        json={"kind": "submission", "assignment_id": assignment["id"]},
    )
    assert second_submission.status_code == 200, second_submission.text
    assert second_submission.json()["annotation_count"] == 1

    submissions = auth_client.get(
        f"/api/projects/{project['id']}/submissions",
        params={"document_id": document["id"], "scope": "mine"},
    )
    assert submissions.status_code == 200
    assert len(submissions.json()) == 2


def test_team_assignment_cannot_use_personal_reopen(auth_client, auth_user):
    workspace = auth_client.post("/api/workspaces", json={"name": "Reopen Team"}).json()
    project = auth_client.post(
        "/api/projects",
        json={
            "name": "team-reopen-rejected",
            "workspace_id": workspace["id"],
            "tasks": [{"annotation_type": "entity", "labels": []}],
        },
    ).json()
    document = auth_client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Team paper",
            "text": "Team annotation.",
        },
    ).json()
    assignment = auth_client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": project["tasks"][0]["id"],
            "document_id": document["id"],
            "assignee_user_id": auth_user["id"],
            "status": "submitted",
        },
    ).json()

    response = auth_client.post(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}/reopen",
    )
    assert response.status_code == 403
    assert "personal workspace" in response.json()["detail"]


def test_personal_version_changes_provision_fresh_self_assignment_rounds(
    auth_client,
    auth_user,
):
    project = auth_client.post(
        "/api/projects",
        json={
            "name": "personal-version-round-hooks",
            "workspace_id": auth_user["workspace_id"],
            "tasks": [{"annotation_type": "entity", "labels": []}],
        },
    ).json()
    document = auth_client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Personal rounds",
            "text": "Aspirin improved outcomes.",
        },
    ).json()
    assignment_url = f"/api/projects/{project['id']}/assignments"
    assignments = auth_client.get(
        assignment_url,
        params={"scope": "mine"},
    ).json()
    assert len(assignments) == 1
    first = assignments[0]
    assert auth_client.patch(
        f"{assignment_url}/{first['id']}",
        json={"status": "submitted"},
    ).status_code == 200

    guideline = auth_client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v2",
            "markdown": "Second local round.",
        },
    )
    assert guideline.status_code == 200, guideline.text
    assignments = auth_client.get(
        assignment_url,
        params={"scope": "mine"},
    ).json()
    assert len(assignments) == 2
    second = max(assignments, key=lambda item: item["id"])
    assert second["guideline_version_id"] == guideline.json()["id"]
    assert auth_client.patch(
        f"{assignment_url}/{second['id']}",
        json={"status": "submitted"},
    ).status_code == 200

    rebuilt = auth_client.post(
        f"/api/documents/{document['id']}/structure/rebuild",
        json={"activate": True},
    )
    assert rebuilt.status_code == 200, rebuilt.text
    assignments = auth_client.get(
        assignment_url,
        params={"scope": "mine"},
    ).json()
    assert len(assignments) == 3
    third = max(assignments, key=lambda item: item["id"])
    assert third["structure_version_id"] == rebuilt.json()["structure_version"][
        "id"
    ]
    assert third["guideline_version_id"] == guideline.json()["id"]


def test_personal_evidence_provisioning_keeps_one_open_logical_target_round(
    auth_client,
    auth_user,
):
    project = auth_client.post(
        "/api/projects",
        json={
            "name": "personal-logical-evidence-rounds",
            "workspace_id": auth_user["workspace_id"],
            "tasks": [{"annotation_type": "evidence_block", "labels": []}],
        },
    ).json()
    document = auth_client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Logical evidence rounds",
            "text": "Alpha. Beta.",
        },
    ).json()
    task_id = project["tasks"][0]["id"]
    target_a = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets",
        json={
            "task_id": task_id,
            "key": "personal-logical-a",
            "name": "Personal logical A",
            "initial_version": {"text": "A v1"},
        },
    ).json()
    version_a1 = target_a["versions"][0]
    activated_a1 = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets/{target_a['id']}/activate",
        json={"version_id": version_a1["id"]},
    )
    assert activated_a1.status_code == 200, activated_a1.text
    assignment_url = f"/api/projects/{project['id']}/assignments"
    first_rounds = auth_client.get(
        assignment_url,
        params={"scope": "mine", "document_id": document["id"]},
    ).json()
    assert [item["target_version_id"] for item in first_rounds] == [version_a1["id"]]

    version_a2 = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets/{target_a['id']}/versions",
        json={"text": "A v2"},
    ).json()
    activated_a2 = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets/{target_a['id']}/activate",
        json={"version_id": version_a2["id"]},
    )
    assert activated_a2.status_code == 200, activated_a2.text
    # A new immutable version of target A does not create a second writable
    # assignment while A v1 is still open.
    assert len(
        auth_client.get(
            assignment_url,
            params={"scope": "mine", "document_id": document["id"]},
        ).json()
    ) == 1

    target_b = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets",
        json={
            "task_id": task_id,
            "key": "personal-logical-b",
            "name": "Personal logical B",
            "initial_version": {"text": "B v1"},
        },
    ).json()
    version_b1 = target_b["versions"][0]
    activated_b1 = auth_client.post(
        f"/api/projects/{project['id']}/evidence-targets/{target_b['id']}/activate",
        json={"version_id": version_b1["id"]},
    )
    assert activated_b1.status_code == 200, activated_b1.text
    target_versions = {
        item["target_version_id"]
        for item in auth_client.get(
            assignment_url,
            params={"scope": "mine", "document_id": document["id"]},
        ).json()
    }
    assert target_versions == {version_a1["id"], version_b1["id"]}


def test_personal_submission_unblocks_version_activated_during_open_round(
    auth_client,
    auth_user,
):
    project = auth_client.post(
        "/api/projects",
        json={
            "name": "personal-mid-round-version-change",
            "workspace_id": auth_user["workspace_id"],
            "tasks": [{"annotation_type": "entity", "labels": []}],
        },
    ).json()
    document = auth_client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Mid-round change",
            "text": "Aspirin improved outcomes.",
        },
    ).json()
    assignment_url = f"/api/projects/{project['id']}/assignments"
    first = auth_client.get(assignment_url, params={"scope": "mine"}).json()
    assert len(first) == 1

    guideline = auth_client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "mid-round-v2",
            "markdown": "Activated while v1 remains open.",
        },
    )
    assert guideline.status_code == 200, guideline.text
    # The active v1 assignment remains the only writable round until submit.
    assert len(
        auth_client.get(assignment_url, params={"scope": "mine"}).json()
    ) == 1

    submitted = auth_client.post(
        f"/api/projects/{project['id']}/documents/{document['id']}/submissions",
        json={"kind": "submission"},
    )
    assert submitted.status_code == 200, submitted.text
    assignments = auth_client.get(
        assignment_url,
        params={"scope": "mine"},
    ).json()
    assert len(assignments) == 2
    newest = max(assignments, key=lambda item: item["id"])
    assert newest["status"] == "assigned"
    assert newest["guideline_version_id"] == guideline.json()["id"]

    writable = auth_client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Finding",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
        },
    )
    assert writable.status_code == 200, writable.text
    assert writable.json()["guideline_version_id"] == guideline.json()["id"]


def test_change_role_updates_membership(db):
    from al_medlit.workspace import service

    owner = _make_user(db, "kate")
    member_user = _make_user(db, "leo")
    ws = service.create_team_workspace(db, owner, name="Team B")
    db.flush()
    service.add_member(db, ws.id, member_user.id, role="annotator")
    db.flush()
    service.change_role(
        db,
        ws.id,
        member_user.id,
        role="trainer",
        actor_user_id=owner.id,
    )
    db.flush()
    assert service.get_member(db, ws.id, member_user.id).role == "trainer"


def test_team_workspace_cannot_lose_its_last_admin(db):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.workspace import service

    owner = _make_user(db, "last-admin")
    ws = service.create_team_workspace(db, owner, name="Last Admin Team")

    with pytest.raises(ConflictError, match="at least one admin"):
        service.change_role(
            db,
            ws.id,
            owner.id,
            role="manager",
            actor_user_id=owner.id,
        )
    with pytest.raises(ConflictError, match="at least one admin"):
        service.remove_member(db, ws.id, owner.id, actor_user_id=owner.id)


def test_inactive_admin_membership_does_not_count_toward_last_active_admin(db):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.workspace import service

    owner = _make_user(db, "last-active-admin")
    inactive_admin = _make_user(db, "inactive-preserved-admin")
    workspace = service.create_team_workspace(db, owner, name="Active Admin Team")
    service.add_member(db, workspace.id, inactive_admin.id, role="admin")
    inactive_admin.is_active = False
    db.flush()

    with pytest.raises(ConflictError, match="at least one admin"):
        service.change_role(
            db,
            workspace.id,
            owner.id,
            role="manager",
            actor_user_id=owner.id,
        )
    with pytest.raises(ConflictError, match="at least one admin"):
        service.remove_member(
            db,
            workspace.id,
            owner.id,
            actor_user_id=owner.id,
        )

    # Cleaning up an inactive preserved membership is safe while an active
    # administrator remains.
    service.remove_member(
        db,
        workspace.id,
        inactive_admin.id,
        actor_user_id=owner.id,
    )
    assert service.get_member(db, workspace.id, inactive_admin.id) is None
    assert service.get_member(db, workspace.id, owner.id).role == "admin"


def test_shared_workspace_transitions_request_database_row_locks(db, monkeypatch):
    from sqlalchemy.orm import Query

    from al_medlit.auth.models import User
    from al_medlit.workspace import service
    from al_medlit.workspace.models import Workspace, WorkspaceInvite

    lock_calls = []
    original_with_for_update = Query.with_for_update

    def recording_with_for_update(query, *args, **kwargs):
        lock_calls.append(query)
        return original_with_for_update(query, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", recording_with_for_update)

    owner = _make_user(db, "locking-owner")
    second_admin = _make_user(db, "locking-admin")
    invitee = _make_user(db, "locking-invitee")
    applicant = _make_user(db, "locking-applicant")
    workspace = service.create_team_workspace(db, owner, name="Locking Team")
    service.add_member(db, workspace.id, second_admin.id, role="admin")

    invite = service.create_invite(
        db,
        workspace.id,
        created_by=owner.id,
        role="annotator",
        expires_minutes=None,
    )
    lock_calls.clear()
    service.accept_invite(db, invite, invitee)
    # Acceptance locks the accepting user before workspace and invite. Inviter
    # authority is refreshed normally after those locks, avoiding an inverse
    # Workspace -> User edge against account deactivation's User -> Workspace order.
    assert [query.column_descriptions[0]["entity"] for query in lock_calls] == [
        User,
        Workspace,
        WorkspaceInvite,
    ]

    join_request = service.create_join_request(
        db,
        workspace.id,
        applicant.id,
        message=None,
    )
    lock_calls.clear()
    service.decide_join_request(
        db,
        join_request.id,
        approve=True,
        decided_by=owner.id,
    )
    # Decision authorization is serialized on the workspace before the
    # individual request row is locked and transitioned.
    assert len(lock_calls) == 2

    lock_calls.clear()
    service.change_role(
        db,
        workspace.id,
        second_admin.id,
        role="manager",
        actor_user_id=owner.id,
    )
    # The workspace lock is followed by a refreshed target membership lock.
    assert len(lock_calls) == 2


def test_workspace_mutations_revalidate_actor_and_stale_target(db):
    from al_medlit.core.exceptions import ConflictError, ForbiddenError
    from al_medlit.workspace import service
    from al_medlit.workspace.models import WorkspaceMember

    owner = _make_user(db, "fresh-auth-owner")
    stale_actor = _make_user(db, "fresh-auth-actor")
    target = _make_user(db, "fresh-auth-target")
    superuser = _make_user(db, "fresh-auth-root")
    superuser.is_superuser = True
    workspace = service.create_team_workspace(db, owner, name="Fresh Auth Team")
    service.add_member(db, workspace.id, stale_actor.id, role="admin")
    stale_target = service.add_member(db, workspace.id, target.id, role="annotator")
    db.flush()

    # Simulate the route dependency retaining an admin membership while a
    # preceding, serialized request has already demoted that actor in the DB.
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace.id,
        WorkspaceMember.user_id == stale_actor.id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    with pytest.raises(ForbiddenError, match="Insufficient workspace role"):
        service.change_role(
            db,
            workspace.id,
            target.id,
            role="trainer",
            actor_user_id=stale_actor.id,
        )

    # A superuser needs no membership, and the target query must refresh a
    # cached identity before enforcing the last-admin invariant.
    assert stale_target.role == "annotator"
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace.id,
        WorkspaceMember.user_id == owner.id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace.id,
        WorkspaceMember.user_id == target.id,
    ).update({WorkspaceMember.role: "admin"}, synchronize_session=False)
    with pytest.raises(ConflictError, match="at least one admin"):
        service.remove_member(
            db,
            workspace.id,
            target.id,
            actor_user_id=superuser.id,
        )


@pytest.mark.parametrize(
    ("inviter_role", "invite_role", "authority_change"),
    [
        ("admin", "annotator", "demoted"),
        ("admin", "admin", "grant_ceiling_lowered"),
        ("admin", "annotator", "removed"),
        ("admin", "annotator", "deactivated"),
    ],
)
def test_accept_invite_revalidates_stale_inviter_authority(
    db,
    inviter_role,
    invite_role,
    authority_change,
):
    from al_medlit.auth.models import User
    from al_medlit.core.exceptions import ForbiddenError
    from al_medlit.workspace import service

    owner = _make_user(db, f"invite-authority-owner-{authority_change}")
    inviter = _make_user(db, f"invite-authority-actor-{authority_change}")
    invitee = _make_user(db, f"invite-authority-target-{authority_change}")
    workspace = service.create_team_workspace(
        db,
        owner,
        name=f"Invite Authority {authority_change}",
    )
    service.add_member(db, workspace.id, inviter.id, role=inviter_role)
    invite = service.create_invite(
        db,
        workspace.id,
        created_by=inviter.id,
        role=invite_role,
        expires_minutes=None,
    )

    if authority_change == "demoted":
        service.change_role(
            db,
            workspace.id,
            inviter.id,
            role="manager",
            actor_user_id=owner.id,
        )
    elif authority_change == "grant_ceiling_lowered":
        service.change_role(
            db,
            workspace.id,
            inviter.id,
            role="manager",
            actor_user_id=owner.id,
        )
    elif authority_change == "removed":
        service.remove_member(
            db,
            workspace.id,
            inviter.id,
            actor_user_id=owner.id,
        )
    else:
        # Leave the already-loaded inviter stale to verify acceptance refreshes
        # the database row after it acquires the invite lock.
        db.query(User).filter(User.id == inviter.id).update(
            {User.is_active: False},
            synchronize_session=False,
        )

    with pytest.raises(ForbiddenError, match="no longer authorized"):
        service.accept_invite(db, invite, invitee)

    assert invite.accepted_at is None
    assert service.get_member(db, workspace.id, invitee.id) is None


def test_accept_invite_revalidates_stale_accepting_user_status(db):
    from al_medlit.auth.models import User
    from al_medlit.core.exceptions import ForbiddenError
    from al_medlit.workspace import service

    owner = _make_user(db, "inactive-acceptance-owner")
    invitee = _make_user(db, "inactive-acceptance-target")
    workspace = service.create_team_workspace(
        db,
        owner,
        name="Inactive Acceptance Team",
    )
    invite = service.create_invite(
        db,
        workspace.id,
        created_by=owner.id,
        role="annotator",
        expires_minutes=None,
    )
    db.query(User).filter(User.id == invitee.id).update(
        {User.is_active: False},
        synchronize_session=False,
    )

    with pytest.raises(ForbiddenError, match="inactive user"):
        service.accept_invite(db, invite, invitee)

    assert invite.accepted_at is None
    assert invite.accepted_by is None
    assert service.get_member(db, workspace.id, invitee.id) is None


def test_active_superuser_invite_does_not_require_workspace_membership(db):
    from al_medlit.workspace import service

    owner = _make_user(db, "superuser-invite-owner")
    superuser = _make_user(db, "superuser-invite-actor")
    superuser.is_superuser = True
    invitee = _make_user(db, "superuser-invite-target")
    workspace = service.create_team_workspace(db, owner, name="Superuser Invite Team")

    invite = service.create_invite(
        db,
        workspace.id,
        created_by=superuser.id,
        role="manager",
        expires_minutes=None,
    )
    member = service.accept_invite(db, invite, invitee)

    assert member.role == "manager"
    assert service.get_member(db, workspace.id, superuser.id) is None


def test_remove_member_withdraws_mutable_assignments_and_preserves_history(db):
    from sqlalchemy.orm import Query

    from al_medlit.core.exceptions import ConflictError, ValidationError
    from al_medlit.corpus.models import Document
    from al_medlit.project import service as project_service
    from al_medlit.project.models import Project, ProjectTask, TaskAssignment
    from al_medlit.project.schemas import TaskAssignmentCreate, TaskAssignmentUpdate
    from al_medlit.workspace import service
    from al_medlit.workspace.models import Workspace, WorkspaceMember

    owner = _make_user(db, "offboarding-owner")
    departing = _make_user(db, "offboarding-departing")
    replacement = _make_user(db, "offboarding-replacement")
    workspace = service.create_team_workspace(db, owner, name="Offboarding Team")
    service.add_member(db, workspace.id, departing.id, role="annotator")
    service.add_member(db, workspace.id, replacement.id, role="annotator")

    projects = [
        Project(name=f"offboarding-project-{index}", workspace_id=workspace.id)
        for index in range(2)
    ]
    db.add_all(projects)
    db.flush()
    tasks = [
        ProjectTask(
            project_id=project.id,
            annotation_type="entity",
            display_name="Entities",
            enabled=True,
        )
        for project in projects
    ]
    db.add_all(tasks)
    db.flush()

    statuses = (
        "assigned",
        "in_progress",
        "blocked",
        "submitted",
        "adjudication_ready",
        "adjudicated",
        "completed",
    )
    assignments = []
    documents = []
    for index, assignment_status in enumerate(statuses):
        project_index = index % len(projects)
        document = Document(
            project_id=projects[project_index].id,
            title=f"Offboarding document {index}",
            text="Audit-preserved annotation text.",
        )
        db.add(document)
        db.flush()
        assignment = TaskAssignment(
            project_id=projects[project_index].id,
            task_id=tasks[project_index].id,
            document_id=document.id,
            assignee_user_id=departing.id,
            annotator_id=departing.username,
            assignment_scope_key="document",
            status=assignment_status,
            assigned_by_user_id=owner.id,
            assigned_by=owner.username,
            metadata_={"audit_seed": assignment_status},
        )
        db.add(assignment)
        assignments.append(assignment)
        documents.append(document)
    db.flush()

    locked_entities = []
    original_with_for_update = Query.with_for_update

    def recording_with_for_update(query, *args, **kwargs):
        locked_entities.append(query.column_descriptions[0].get("entity"))
        return original_with_for_update(query, *args, **kwargs)

    # The workspace boundary is canonical, followed by the membership and all
    # mutable assignments in deterministic primary-key order.
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(Query, "with_for_update", recording_with_for_update)
        service.remove_member(
            db,
            workspace.id,
            departing.id,
            actor_user_id=owner.id,
        )
    assert locked_entities[:3] == [Workspace, WorkspaceMember, TaskAssignment]
    assert service.get_member(db, workspace.id, departing.id) is None

    for assignment, prior_status in zip(assignments, statuses, strict=True):
        db.refresh(assignment)
        if prior_status in {"assigned", "in_progress", "blocked"}:
            assert assignment.status == "withdrawn"
            assert assignment.metadata_["audit_seed"] == prior_status
            offboarding = assignment.metadata_["workspace_offboarding"]
            assert offboarding["prior_status"] == prior_status
            assert offboarding["reason"] == "workspace_member_removed"
            assert offboarding["workspace_id"] == workspace.id
            assert offboarding["removed_user_id"] == departing.id
            assert offboarding["removed_by_user_id"] == owner.id
            assert offboarding["withdrawn_at"]
        else:
            assert assignment.status == prior_status
            assert assignment.metadata_ == {"audit_seed": prior_status}

    withdrawn = assignments[0]
    with pytest.raises(ConflictError, match="immutable audit records"):
        project_service.update_task_assignment(
            db,
            withdrawn.project_id,
            withdrawn.id,
            TaskAssignmentUpdate(status="assigned"),
        )
    with pytest.raises(ConflictError, match="immutable audit records"):
        project_service.update_task_assignment(
            db,
            withdrawn.project_id,
            withdrawn.id,
            TaskAssignmentUpdate(metadata_={"workspace_offboarding": {}}),
        )

    # A different active member can take a fresh writable round while the
    # departed annotator's row and metadata remain available for audit.
    with pytest.raises(ValidationError, match="reserved for workspace offboarding"):
        project_service.create_task_assignment(
            db,
            projects[0].id,
            TaskAssignmentCreate(
                task_id=tasks[0].id,
                document_id=documents[0].id,
                assignee_user_id=replacement.id,
                status="withdrawn",
            ),
            assigned_by_user=owner,
        )
    replacement_assignment = project_service.create_task_assignment(
        db,
        projects[0].id,
        TaskAssignmentCreate(
            task_id=tasks[0].id,
            document_id=documents[0].id,
            assignee_user_id=replacement.id,
        ),
        assigned_by_user=owner,
    )
    assert replacement_assignment.status == "assigned"
    assert replacement_assignment.assignee_user_id == replacement.id
    db.refresh(withdrawn)
    assert withdrawn.status == "withdrawn"


def test_assignment_mutations_lock_workspace_before_project_scope(db, monkeypatch):
    from sqlalchemy.orm import Query

    from al_medlit.corpus.models import Document
    from al_medlit.project import service as project_service
    from al_medlit.project.models import Project, ProjectTask
    from al_medlit.project.schemas import TaskAssignmentCreate, TaskAssignmentUpdate
    from al_medlit.workspace import service as workspace_service
    from al_medlit.workspace.models import Workspace

    owner = _make_user(db, "assignment-lock-owner")
    first_assignee = _make_user(db, "assignment-lock-first")
    next_assignee = _make_user(db, "assignment-lock-next")
    workspace = workspace_service.create_team_workspace(
        db,
        owner,
        name="Assignment Lock Team",
    )
    workspace_service.add_member(
        db,
        workspace.id,
        first_assignee.id,
        role="annotator",
    )
    workspace_service.add_member(
        db,
        workspace.id,
        next_assignee.id,
        role="annotator",
    )
    project = Project(name="assignment-lock-project", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    task = ProjectTask(
        project_id=project.id,
        annotation_type="entity",
        display_name="Entities",
        enabled=True,
    )
    document = Document(project_id=project.id, title="Lock order", text="Text")
    db.add_all([task, document])
    db.flush()

    locked_entities = []
    original_with_for_update = Query.with_for_update

    def recording_with_for_update(query, *args, **kwargs):
        locked_entities.append(query.column_descriptions[0].get("entity"))
        return original_with_for_update(query, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", recording_with_for_update)
    assignment = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=document.id,
            assignee_user_id=first_assignee.id,
        ),
        assigned_by_user=owner,
    )
    assert locked_entities[:4] == [Workspace, Project, ProjectTask, Document]

    locked_entities.clear()
    project_service.update_task_assignment(
        db,
        project.id,
        assignment.id,
        TaskAssignmentUpdate(assignee_user_id=next_assignee.id),
    )
    assert locked_entities[:5] == [
        Workspace,
        Project,
        ProjectTask,
        Document,
        type(assignment),
    ]


def test_individual_workspace_rejects_members_invites_and_join_requests(
    auth_client,
    auth_user,
    db,
):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.workspace import service

    guest = _make_user(db, "individual-guest")
    db.commit()

    with pytest.raises(ConflictError, match="Individual workspaces"):
        service.add_member(
            db,
            auth_user["workspace_id"],
            guest.id,
            role="annotator",
        )
    with pytest.raises(ConflictError, match="Individual workspaces"):
        service.create_join_request(
            db,
            auth_user["workspace_id"],
            guest.id,
            message=None,
        )

    invite = auth_client.post(
        f"/api/workspaces/{auth_user['workspace_id']}/invites",
        json={"role": "annotator"},
    )
    assert invite.status_code == 409
    assert "Individual workspaces" in invite.json()["detail"]

    for operation in (
        lambda: service.change_role(
            db,
            auth_user["workspace_id"],
            auth_user["id"],
            role="admin",
            actor_user_id=auth_user["id"],
        ),
        lambda: service.remove_member(
            db,
            auth_user["workspace_id"],
            auth_user["id"],
            actor_user_id=auth_user["id"],
        ),
        lambda: service.rotate_join_code(
            db,
            auth_user["workspace_id"],
            actor_user_id=auth_user["id"],
        ),
    ):
        with pytest.raises(ConflictError, match="Individual workspaces"):
            operation()


def test_require_role_allows_sufficient_role(db):
    from al_medlit.workspace import service
    from al_medlit.workspace.dependencies import require_role

    owner = _make_user(db, "mona")
    ws = service.create_team_workspace(db, owner, name="Team C")
    db.flush()

    dep = require_role("manager")
    member = dep(workspace_id=ws.id, current_user=owner, db=db)
    assert member.role == "admin"


def test_require_role_rejects_insufficient_role(db):
    from al_medlit.core.exceptions import ForbiddenError
    from al_medlit.workspace import service
    from al_medlit.workspace.dependencies import require_role

    owner = _make_user(db, "nina")
    annotator = _make_user(db, "omar")
    ws = service.create_team_workspace(db, owner, name="Team D")
    db.flush()
    service.add_member(db, ws.id, annotator.id, role="annotator")
    db.flush()

    dep = require_role("manager")
    with pytest.raises(ForbiddenError):
        dep(workspace_id=ws.id, current_user=annotator, db=db)


def test_require_role_superuser_bypass(db):
    from al_medlit.auth.models import User
    from al_medlit.workspace import service
    from al_medlit.workspace.dependencies import require_role

    owner = _make_user(db, "pia")
    ws = service.create_team_workspace(db, owner, name="Team E")
    db.flush()

    root = User(username="root", password_hash="x", is_superuser=True)
    db.add(root)
    db.flush()

    dep = require_role("admin")
    member = dep(workspace_id=ws.id, current_user=root, db=db)
    assert member.role == "admin"


def test_create_and_list_workspaces_via_api(auth_client):
    resp = auth_client.post("/api/workspaces", json={"name": "My Team"})
    assert resp.status_code == 200
    ws = resp.json()
    assert ws["kind"] == "team"
    assert ws["join_code"]

    listing = auth_client.get("/api/workspaces")
    assert listing.status_code == 200
    names = [w["name"] for w in listing.json()]
    assert "My Team" in names


def test_workspace_governance_rotates_team_join_code(auth_client):
    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Rotating Join Code Team"},
    ).json()
    old_code = workspace["join_code"]

    governance = auth_client.get(
        f"/api/workspaces/{workspace['id']}/governance"
    )
    rotated = auth_client.post(
        f"/api/workspaces/{workspace['id']}/join-code/rotate"
    )

    assert governance.status_code == 200
    assert governance.json() == {
        "workspace_id": workspace["id"],
        "workspace_kind": "team",
        "join_code": old_code,
        "default_invite_expiry_minutes": 10_080,
    }
    assert rotated.status_code == 200
    assert rotated.json()["workspace_id"] == workspace["id"]
    assert rotated.json()["workspace_kind"] == "team"
    assert rotated.json()["join_code"]
    assert rotated.json()["join_code"] != old_code


def test_rotated_join_code_invalidates_the_previous_code(auth_client, db):
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.auth.security import create_access_token
    from al_medlit.auth.service import register_user

    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Invalidated Join Code Team"},
    ).json()
    old_code = workspace["join_code"]
    new_code = auth_client.post(
        f"/api/workspaces/{workspace['id']}/join-code/rotate"
    ).json()["join_code"]
    applicant = register_user(
        db,
        UserCreate(username="rotation-applicant", password="pw"),
    )
    db.commit()
    headers = {
        "Authorization": f"Bearer {create_access_token(applicant.id)}",
    }

    stale = auth_client.post(
        f"/api/workspaces/by-code/{old_code}/join-requests",
        json={},
        headers=headers,
    )
    current = auth_client.post(
        f"/api/workspaces/by-code/{new_code}/join-requests",
        json={},
        headers=headers,
    )

    assert stale.status_code == 404
    assert current.status_code == 200
    assert current.json()["workspace_id"] == workspace["id"]


def test_workspace_governance_describes_individual_without_team_controls(
    auth_client,
    auth_user,
):
    governance = auth_client.get(
        f"/api/workspaces/{auth_user['workspace_id']}/governance"
    )
    invites = auth_client.get(
        f"/api/workspaces/{auth_user['workspace_id']}/invites"
    )
    rotate = auth_client.post(
        f"/api/workspaces/{auth_user['workspace_id']}/join-code/rotate"
    )

    assert governance.status_code == 200
    assert governance.json()["workspace_kind"] == "individual"
    assert governance.json()["join_code"] is None
    assert invites.status_code == 409
    assert rotate.status_code == 409


def test_member_role_management_via_api(auth_client, db):
    ws = auth_client.post("/api/workspaces", json={"name": "RoleTeam"}).json()

    other = _make_user(db, "quinn")
    db.commit()
    from al_medlit.workspace import service

    service.add_member(db, ws["id"], other.id, role="annotator")
    db.commit()

    resp = auth_client.patch(
        f"/api/workspaces/{ws['id']}/members/{other.id}",
        json={"role": "trainer"},
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "trainer"


def test_workspace_governance_mutations_append_admin_events(auth_client, db):
    from al_medlit.administration.models import AdminAuditEvent
    from al_medlit.workspace import service

    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Governance Audit Team"},
    ).json()
    other = _make_user(db, "governance-audit-member")
    service.add_member(db, workspace["id"], other.id, role="annotator")
    db.commit()

    role_change = auth_client.patch(
        f"/api/workspaces/{workspace['id']}/members/{other.id}",
        json={"role": "trainer"},
    )
    invite = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    ).json()
    rotate = auth_client.post(
        f"/api/workspaces/{workspace['id']}/join-code/rotate"
    )
    revoke = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/invites/{invite['id']}"
    )
    remove = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/members/{other.id}"
    )

    assert role_change.status_code == 200
    assert rotate.status_code == 200
    assert revoke.status_code == 204
    assert remove.status_code == 204
    db.expire_all()
    events = (
        db.query(AdminAuditEvent)
        .filter(AdminAuditEvent.workspace_id == workspace["id"])
        .order_by(AdminAuditEvent.id)
        .all()
    )
    assert [event.event_type for event in events] == [
        "workspace.member_role_changed",
        "workspace.invite_created",
        "workspace.join_code_rotated",
        "workspace.invite_revoked",
        "workspace.member_removed",
    ]
    assert all("token" not in event.details for event in events)
    assert all("join_code" not in event.details for event in events)


def test_invite_create_and_accept_new_user(auth_client):
    ws = auth_client.post("/api/workspaces", json={"name": "InviteTeam"}).json()
    inv = auth_client.post(
        f"/api/workspaces/{ws['id']}/invites",
        json={"role": "annotator"},
    )
    assert inv.status_code == 200
    token = inv.json()["token"]

    weak_password = auth_client.post(
        f"/api/invites/{token}/accept",
        json={"username": "invitee", "password": "short"},
        headers={"Authorization": ""},
    )
    assert weak_password.status_code == 422

    accept = auth_client.post(
        f"/api/invites/{token}/accept",
        json={
            "username": "invitee",
            "password": "strong-password",
            "display_name": "Invitee",
        },
        headers={"Authorization": ""},
    )
    assert accept.status_code == 200
    access_token = accept.json()["access_token"]
    assert access_token

    repeated = auth_client.post(
        f"/api/invites/{token}/accept",
        json={},
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert repeated.status_code == 404


def test_invite_uses_instance_default_expiry_and_validates_explicit_bounds(
    auth_client,
    db,
):
    from datetime import UTC, datetime, timedelta

    from al_medlit.administration.models import InstancePolicy

    db.add(
        InstancePolicy(
            id=1,
            allow_self_registration=None,
            default_invite_expiry_minutes=1_440,
            account_action_expiry_minutes=60,
        )
    )
    db.commit()
    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Policy Expiry Team"},
    ).json()

    before = datetime.now(UTC)
    defaulted = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    )
    after = datetime.now(UTC)

    assert defaulted.status_code == 200, defaulted.text
    body = defaulted.json()
    assert isinstance(body["id"], int)
    expires_at = datetime.fromisoformat(body["expires_at"])
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    assert (
        before + timedelta(minutes=1_440)
        <= expires_at
        <= after + timedelta(minutes=1_440)
    )

    for invalid_expiry in (59, 43_201):
        response = auth_client.post(
            f"/api/workspaces/{workspace['id']}/invites",
            json={"role": "annotator", "expires_minutes": invalid_expiry},
        )
        assert response.status_code == 422


def test_open_invites_can_be_listed_without_tokens_and_revoked(auth_client, auth_user, db):
    from al_medlit.workspace.models import WorkspaceInvite

    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Revocable Invite Team"},
    ).json()
    created = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "trainer", "expires_minutes": 1_440},
    ).json()

    listing = auth_client.get(f"/api/workspaces/{workspace['id']}/invites")

    assert listing.status_code == 200
    assert listing.json() == [
        {
            "id": created["id"],
            "workspace_id": workspace["id"],
            "role": "trainer",
            "created_by": auth_user["id"],
            "created_by_username": auth_user["username"],
            "expires_at": created["expires_at"],
            "created_at": listing.json()[0]["created_at"],
        }
    ]
    assert "token" not in listing.json()[0]

    revoked = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/invites/{created['id']}"
    )
    repeated = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/invites/{created['id']}"
    )

    assert revoked.status_code == 204
    assert repeated.status_code == 204
    stored = db.get(WorkspaceInvite, created["id"])
    db.refresh(stored)
    assert stored.revoked_at is not None
    assert stored.revoked_by == auth_user["id"]
    assert auth_client.get(f"/api/workspaces/{workspace['id']}/invites").json() == []
    assert (
        auth_client.get(
            f"/api/invites/{created['token']}",
            headers={"Authorization": ""},
        ).status_code
        == 404
    )
    assert (
        auth_client.post(
            f"/api/invites/{created['token']}/accept",
            json={"username": "revoked-invitee", "password": "strong-password"},
            headers={"Authorization": ""},
        ).status_code
        == 404
    )


def test_open_invite_listing_excludes_expired_and_consumed_invites(auth_client, auth_user, db):
    from datetime import UTC, datetime, timedelta

    from al_medlit.workspace.models import WorkspaceInvite

    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Only Open Invites Team"},
    ).json()
    active = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    ).json()
    expired = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "trainer"},
    ).json()
    consumed = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "manager"},
    ).json()
    expired_row = db.get(WorkspaceInvite, expired["id"])
    consumed_row = db.get(WorkspaceInvite, consumed["id"])
    expired_row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    consumed_row.accepted_at = datetime.now(UTC)
    consumed_row.accepted_by = auth_user["id"]
    db.commit()

    listing = auth_client.get(f"/api/workspaces/{workspace['id']}/invites")
    revoke_consumed = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/invites/{consumed['id']}"
    )

    assert listing.status_code == 200
    assert [invite["id"] for invite in listing.json()] == [active["id"]]
    assert revoke_consumed.status_code == 409


def test_invite_preview_describes_workspace_without_authentication(auth_client):
    ws = auth_client.post("/api/workspaces", json={"name": "PreviewTeam"}).json()
    token = auth_client.post(
        f"/api/workspaces/{ws['id']}/invites",
        json={"role": "trainer"},
    ).json()["token"]

    preview = auth_client.get(f"/api/invites/{token}", headers={"Authorization": ""})

    assert preview.status_code == 200
    body = preview.json()
    assert body["workspace_name"] == "PreviewTeam"
    assert body["workspace_kind"] == "team"
    assert body["role"] == "trainer"
    # The token holder learns nothing beyond what they need to decide.
    assert "workspace_id" not in body
    assert "join_code" not in body


def test_invite_preview_rejects_unknown_and_consumed_tokens(auth_client):
    unknown = auth_client.get("/api/invites/not-a-real-token", headers={"Authorization": ""})
    assert unknown.status_code == 404

    ws = auth_client.post("/api/workspaces", json={"name": "ConsumedTeam"}).json()
    token = auth_client.post(
        f"/api/workspaces/{ws['id']}/invites",
        json={"role": "annotator"},
    ).json()["token"]

    accepted = auth_client.post(
        f"/api/invites/{token}/accept",
        json={"username": "preview-invitee", "password": "strong-password"},
        headers={"Authorization": ""},
    )
    assert accepted.status_code == 200

    assert (
        auth_client.get(f"/api/invites/{token}", headers={"Authorization": ""}).status_code
        == 404
    )


def test_invite_preview_rejects_expired_tokens(auth_client, db):
    from datetime import UTC, datetime, timedelta

    from al_medlit.workspace.models import WorkspaceInvite

    ws = auth_client.post("/api/workspaces", json={"name": "ExpiredTeam"}).json()
    token = auth_client.post(
        f"/api/workspaces/{ws['id']}/invites",
        json={"role": "annotator", "expires_minutes": 60},
    ).json()["token"]

    invite = db.query(WorkspaceInvite).filter(WorkspaceInvite.token == token).one()
    invite.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    db.commit()

    assert (
        auth_client.get(f"/api/invites/{token}", headers={"Authorization": ""}).status_code
        == 404
    )


def test_invite_preview_rejects_stale_creator_authority(auth_client, db):
    from al_medlit.auth.models import User

    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": "Stale Invite Authority Team"},
    ).json()
    token = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    ).json()["token"]
    db.query(User).filter(User.username == "tester").update(
        {User.is_active: False},
        synchronize_session=False,
    )
    db.commit()

    preview = auth_client.get(
        f"/api/invites/{token}",
        headers={"Authorization": ""},
    )

    assert preview.status_code == 403
    assert "no longer authorized" in preview.json()["detail"]


def test_invite_accept_supports_an_existing_signed_in_user(auth_client, db):
    """The bearer path is how an existing account redeems an invite."""
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.auth.security import create_access_token
    from al_medlit.auth.service import register_user

    ws = auth_client.post("/api/workspaces", json={"name": "ExistingUserTeam"}).json()
    token = auth_client.post(
        f"/api/workspaces/{ws['id']}/invites",
        json={"role": "annotator"},
    ).json()["token"]

    existing = register_user(
        db,
        UserCreate(username="already-registered", password="strong-password"),
    )
    db.commit()

    accept = auth_client.post(
        f"/api/invites/{token}/accept",
        json={},
        headers={"Authorization": f"Bearer {create_access_token(existing.id)}"},
    )

    assert accept.status_code == 200
    members = auth_client.get(f"/api/workspaces/{ws['id']}/members").json()
    assert any(member["username"] == "already-registered" for member in members)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("username", "x" * 121),
        ("display_name", "x" * 121),
    ],
)
def test_invite_accept_rejects_user_fields_over_database_limits(
    auth_client,
    field,
    value,
):
    workspace = auth_client.post(
        "/api/workspaces",
        json={"name": f"Invite Limits {field}"},
    ).json()
    invite = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    )
    assert invite.status_code == 200
    token = invite.json()["token"]
    payload = {
        "username": f"valid-{field}",
        "password": "strong-password",
        field: value,
    }

    response = auth_client.post(
        f"/api/invites/{token}/accept",
        json=payload,
        headers={"Authorization": ""},
    )

    assert response.status_code == 422


def test_manager_cannot_create_workspace_invites(auth_client, db):
    from al_medlit.auth.security import create_access_token
    from al_medlit.workspace import service

    workspace = auth_client.post("/api/workspaces", json={"name": "Ranked Invites"}).json()
    open_invite = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "annotator"},
    ).json()
    manager = _make_user(db, "invite-manager")
    service.add_member(db, workspace["id"], manager.id, role="manager")
    db.commit()
    headers = {"Authorization": f"Bearer {create_access_token(manager.id)}"}

    elevated = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "admin"},
        headers=headers,
    )
    peer = auth_client.post(
        f"/api/workspaces/{workspace['id']}/invites",
        json={"role": "manager"},
        headers=headers,
    )
    listing = auth_client.get(
        f"/api/workspaces/{workspace['id']}/invites",
        headers=headers,
    )
    revoke = auth_client.delete(
        f"/api/workspaces/{workspace['id']}/invites/{open_invite['id']}",
        headers=headers,
    )
    rotate = auth_client.post(
        f"/api/workspaces/{workspace['id']}/join-code/rotate",
        headers=headers,
    )
    governance = auth_client.get(
        f"/api/workspaces/{workspace['id']}/governance",
        headers=headers,
    )
    capability = auth_client.patch(
        f"/api/workspaces/{workspace['id']}/capability",
        json={"preset": "train", "overrides": []},
        headers=headers,
    )
    join_requests = auth_client.get(
        f"/api/workspaces/{workspace['id']}/join-requests",
        headers=headers,
    )

    assert elevated.status_code == 403
    assert peer.status_code == 403
    assert listing.status_code == 403
    assert revoke.status_code == 403
    assert rotate.status_code == 403
    assert governance.status_code == 403
    assert capability.status_code == 403
    assert join_requests.status_code == 403


def test_apply_to_join_then_approve(client, db):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate
    from al_medlit.auth.security import create_access_token
    from al_medlit.workspace import service

    owner = auth_service.register_user(db, UserCreate(username="owner2", password="pw"))
    ws = service.create_team_workspace(db, owner, name="ApplyTeam")
    applicant = auth_service.register_user(
        db,
        UserCreate(
            username="applicant",
            password="pw",
            display_name="Applicant Name",
            email="applicant@example.test",
        ),
    )
    db.commit()

    owner_token = create_access_token(str(owner.id))
    applicant_token = create_access_token(str(applicant.id))

    apply = client.post(
        f"/api/workspaces/by-code/{ws.join_code}/join-requests",
        json={"message": "please"},
        headers={"Authorization": f"Bearer {applicant_token}"},
    )
    assert apply.status_code == 200
    request_body = apply.json()
    req_id = request_body["id"]
    assert request_body["username"] == "applicant"
    assert request_body["display_name"] == "Applicant Name"
    assert request_body["email"] == "applicant@example.test"
    assert request_body["created_at"]

    pending = client.get(
        f"/api/workspaces/{ws.id}/join-requests",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert pending.status_code == 200
    assert pending.json()[0]["username"] == "applicant"
    assert pending.json()[0]["display_name"] == "Applicant Name"
    assert pending.json()[0]["email"] == "applicant@example.test"

    approve = client.post(
        f"/api/join-requests/{req_id}/approve",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert approve.status_code == 200

    assert service.get_member(db, ws.id, applicant.id) is not None


def test_create_join_request_is_idempotent_and_uniquely_indexed(db):
    from sqlalchemy import inspect

    from al_medlit.workspace import service
    from al_medlit.workspace.models import WorkspaceJoinRequest

    owner = _make_user(db, "join-owner")
    applicant = _make_user(db, "join-applicant")
    ws = service.create_team_workspace(db, owner, name="Unique Join Team")

    first = service.create_join_request(db, ws.id, applicant.id, message="first")
    second = service.create_join_request(db, ws.id, applicant.id, message="second")

    assert second.id == first.id
    assert (
        db.query(WorkspaceJoinRequest)
        .filter(
            WorkspaceJoinRequest.workspace_id == ws.id,
            WorkspaceJoinRequest.user_id == applicant.id,
            WorkspaceJoinRequest.status == "pending",
        )
        .count()
        == 1
    )
    indexes = {
        index["name"]: index for index in inspect(db.bind).get_indexes("workspace_join_requests")
    }
    assert indexes["uq_workspace_join_requests_pending_user"]["unique"] == 1


def test_ensure_default_workspace_handles_integrity_error(db):
    from unittest.mock import patch

    from sqlalchemy.exc import IntegrityError

    from al_medlit.workspace import service
    from al_medlit.workspace.models import Workspace

    db.query(Workspace).filter(Workspace.name == "Default", Workspace.kind == "team").delete()
    db.flush()

    original_flush = db.flush
    calls = 0

    def mock_flush(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise IntegrityError("mock uniqueness violation", params=None, orig=None)
        return original_flush(*args, **kwargs)

    pre_existing = Workspace(
        name="Default",
        kind="team",
        created_by=None,
        capability_preset="full",
        capability_overrides=[],
    )
    db.add(pre_existing)
    db.flush()
    db.expunge(pre_existing)

    with patch.object(db, "flush", side_effect=mock_flush):
        ws = service.ensure_default_workspace(db)
        assert ws.name == "Default"
        assert ws.kind == "team"
