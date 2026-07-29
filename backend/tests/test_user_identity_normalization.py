def _make_user(db, username: str):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate

    user = auth_service.register_user(db, UserCreate(username=username, password="pw"))
    db.flush()
    return user


def _headers_for(user) -> dict[str, str]:
    from al_medlit.auth.security import create_access_token

    return {"Authorization": f"Bearer {create_access_token(str(user.id))}"}


def test_assignment_visibility_uses_user_ids_and_roles(client, db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import (
        ProjectCreate,
        ProjectTaskCreate,
        TaskAssignmentCreate,
    )
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "identity-owner")
    alice = _make_user(db, "identity-alice")
    bob = _make_user(db, "identity-bob")
    workspace = workspace_service.create_team_workspace(db, owner, name="Identity Team")
    workspace_service.add_member(db, workspace.id, alice.id, role="annotator")
    workspace_service.add_member(db, workspace.id, bob.id, role="annotator")
    project = project_service.create_project(
        db,
        ProjectCreate(name="identity-project", workspace_id=workspace.id),
    )
    task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="entity"),
    )
    alice_doc = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, title="Alice", text="Alpha"),
    )
    bob_doc = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, title="Bob", text="Beta"),
    )
    alice_assignment = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=alice_doc.id,
            assignee_user_id=alice.id,
        ),
    )
    bob_assignment = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=bob_doc.id,
            assignee_user_id=bob.id,
        ),
    )
    db.commit()

    alice_assignments = client.get(
        f"/api/projects/{project.id}/assignments",
        headers=_headers_for(alice),
    )
    assert alice_assignments.status_code == 200
    assert [row["id"] for row in alice_assignments.json()] == [alice_assignment.id]
    assert alice_assignments.json()[0]["assignee_user_id"] == alice.id

    bob_assignments = client.get(
        f"/api/projects/{project.id}/assignments",
        headers=_headers_for(bob),
    )
    assert bob_assignments.status_code == 200
    assert [row["id"] for row in bob_assignments.json()] == [bob_assignment.id]

    owner_assignments = client.get(
        f"/api/projects/{project.id}/assignments",
        headers=_headers_for(owner),
    )
    assert owner_assignments.status_code == 200
    assert {row["id"] for row in owner_assignments.json()} == {
        alice_assignment.id,
        bob_assignment.id,
    }

    forbidden_all = client.get(
        f"/api/projects/{project.id}/assignments",
        params={"scope": "all"},
        headers=_headers_for(alice),
    )
    assert forbidden_all.status_code == 403

    alice_documents = client.get(
        "/api/documents",
        params={"project_id": project.id},
        headers=_headers_for(alice),
    )
    assert alice_documents.status_code == 200
    assert [row["id"] for row in alice_documents.json()] == [alice_doc.id]


def test_annotator_workbench_requires_assigned_document(client, db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import (
        ProjectCreate,
        ProjectTaskCreate,
        TaskAssignmentCreate,
    )
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "workbench-owner")
    alice = _make_user(db, "workbench-alice")
    workspace = workspace_service.create_team_workspace(db, owner, name="Workbench Team")
    workspace_service.add_member(db, workspace.id, alice.id, role="annotator")
    project = project_service.create_project(
        db,
        ProjectCreate(name="workbench-project", workspace_id=workspace.id),
    )
    task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="entity"),
    )
    assigned_doc = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, title="Assigned", text="Alpha"),
    )
    unassigned_doc = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, title="Unassigned", text="Beta"),
    )
    project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=assigned_doc.id,
            assignee_user_id=alice.id,
        ),
    )
    db.commit()

    allowed = client.get(
        f"/api/annotation-workbench/documents/{assigned_doc.id}",
        headers=_headers_for(alice),
    )
    assert allowed.status_code == 200
    assert [row["assignee_user_id"] for row in allowed.json()["assignments"]] == [alice.id]

    blocked = client.get(
        f"/api/annotation-workbench/documents/{unassigned_doc.id}",
        headers=_headers_for(alice),
    )
    assert blocked.status_code == 403

    manager_allowed = client.get(
        f"/api/annotation-workbench/documents/{unassigned_doc.id}",
        headers=_headers_for(owner),
    )
    assert manager_allowed.status_code == 200


def test_assignment_create_rejects_user_outside_project_workspace(client, db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import ProjectCreate, ProjectTaskCreate
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "outside-owner")
    outsider = _make_user(db, "outside-user")
    workspace = workspace_service.create_team_workspace(db, owner, name="Outside Team")
    project = project_service.create_project(
        db,
        ProjectCreate(name="outside-project", workspace_id=workspace.id),
    )
    task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="entity"),
    )
    document = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, title="Doc", text="Alpha"),
    )
    db.commit()

    response = client.post(
        f"/api/projects/{project.id}/assignments",
        json={
            "task_id": task.id,
            "document_id": document.id,
            "assignee_user_id": outsider.id,
        },
        headers=_headers_for(owner),
    )
    assert response.status_code == 422
    assert "project workspace" in response.json()["detail"]
