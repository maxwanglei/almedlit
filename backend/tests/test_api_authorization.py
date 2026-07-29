from dataclasses import dataclass


@dataclass(frozen=True)
class AuthHeaders:
    member_a: dict[str, str]
    owner_a: dict[str, str]
    owner_b: dict[str, str]
    superuser: dict[str, str]


@dataclass(frozen=True)
class TenantResources:
    workspace_a_id: int
    workspace_b_id: int
    project_a_id: int
    project_b_id: int
    document_a_id: int
    document_b_id: int
    guideline_a_id: int
    annotation_a_id: int
    annotation_b_id: int
    task_a_id: int


def _make_user(db, username: str, *, is_superuser: bool = False):
    from al_medlit.auth import service as auth_service
    from al_medlit.auth.schemas import UserCreate

    user = auth_service.register_user(db, UserCreate(username=username, password="pw"))
    user.is_superuser = is_superuser
    db.flush()
    return user


def _headers_for(user) -> dict[str, str]:
    from al_medlit.auth.security import create_access_token

    return {"Authorization": f"Bearer {create_access_token(str(user.id))}"}


def _create_tenant_fixture(db) -> tuple[AuthHeaders, TenantResources]:
    from al_medlit.annotation import service as annotation_service
    from al_medlit.annotation.schemas import AnnotationCreate
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.guideline import service as guideline_service
    from al_medlit.guideline.schemas import GuidelineVersionCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import (
        ProjectCreate,
        ProjectTaskCreate,
        TaskAssignmentCreate,
    )
    from al_medlit.workspace import capability_service
    from al_medlit.workspace import service as workspace_service

    owner_a = _make_user(db, "authz-owner-a")
    member_a = _make_user(db, "authz-member-a")
    owner_b = _make_user(db, "authz-owner-b")
    superuser = _make_user(db, "authz-root", is_superuser=True)

    workspace_a = workspace_service.create_team_workspace(db, owner_a, name="Authz A")
    workspace_service.add_member(db, workspace_a.id, member_a.id, role="annotator")
    workspace_b = workspace_service.create_team_workspace(db, owner_b, name="Authz B")
    capability_service.set_capability(
        db,
        workspace_b.id,
        preset="full",
        actor_user_id=owner_b.id,
    )

    project_a = project_service.create_project(
        db,
        ProjectCreate(name="authz-project-a", workspace_id=workspace_a.id),
    )
    project_b = project_service.create_project(
        db,
        ProjectCreate(name="authz-project-b", workspace_id=workspace_b.id),
    )
    document_a = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project_a.id, title="A", text="Alpha beta."),
    )
    document_b = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project_b.id, title="B", text="Gamma delta."),
    )
    guideline_a = guideline_service.create_guideline_version(
        db,
        GuidelineVersionCreate(project_id=project_a.id, markdown="Use labels."),
    )
    annotation_a = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project_a.id,
            document_id=document_a.id,
            annotation_type="entity",
            label="Condition",
            start_offset=0,
            end_offset=5,
            text_span="Alpha",
            annotator_user_id=member_a.id,
            annotator_id="authz-member-a",
            guideline_version_id=guideline_a.id,
        ),
    )
    annotation_b = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project_b.id,
            document_id=document_b.id,
            annotation_type="entity",
            label="Condition",
            start_offset=0,
            end_offset=5,
            text_span="Gamma",
            annotator_user_id=owner_b.id,
            annotator_id="authz-owner-b",
        ),
    )
    task_a = project_service.create_project_task(
        db,
        project_a.id,
        ProjectTaskCreate(annotation_type="entity", labels=[]),
    )
    project_service.create_task_assignment(
        db,
        project_a.id,
        TaskAssignmentCreate(
            task_id=task_a.id,
            document_id=document_a.id,
            assignee_user_id=member_a.id,
        ),
        assigned_by_user=owner_a,
    )
    db.commit()

    headers = AuthHeaders(
        member_a=_headers_for(member_a),
        owner_a=_headers_for(owner_a),
        owner_b=_headers_for(owner_b),
        superuser=_headers_for(superuser),
    )
    resources = TenantResources(
        workspace_a_id=workspace_a.id,
        workspace_b_id=workspace_b.id,
        project_a_id=project_a.id,
        project_b_id=project_b.id,
        document_a_id=document_a.id,
        document_b_id=document_b.id,
        guideline_a_id=guideline_a.id,
        annotation_a_id=annotation_a.id,
        annotation_b_id=annotation_b.id,
        task_a_id=task_a.id,
    )
    return headers, resources


def test_global_auth_guard_blocks_protected_routes(client):
    missing = client.get("/api/projects", headers={"Authorization": ""})
    assert missing.status_code == 401

    invalid = client.get("/api/projects", headers={"Authorization": "Bearer bad-token"})
    assert invalid.status_code == 401

    health = client.get("/health")
    assert health.status_code == 200

    register = client.post(
        "/api/auth/register",
        json={"username": "authn-newbie", "password": "strong-password"},
    )
    assert register.status_code == 200

    login = client.post(
        "/api/auth/login",
        json={"username": "authn-newbie", "password": "strong-password"},
    )
    assert login.status_code == 200

    assert client.get("/api/documents", headers={"Authorization": ""}).status_code == 401
    assert (
        client.get(
            "/api/guidelines",
            params={"project_id": 1},
            headers={"Authorization": ""},
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/annotations",
            json={},
            headers={"Authorization": ""},
        ).status_code
        == 401
    )


def test_cross_tenant_resource_access_is_forbidden_and_lists_are_scoped(client, db):
    headers, resources = _create_tenant_fixture(db)

    assert (
        client.get(
            f"/api/projects/{resources.project_a_id}",
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/projects/{resources.project_b_id}",
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/projects/{resources.project_b_id}",
            headers=headers.superuser,
        ).status_code
        == 200
    )

    assert (
        client.get(
            f"/api/documents/{resources.document_a_id}",
            headers=headers.member_a,
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/api/documents/{resources.document_b_id}",
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/guidelines",
            params={"project_id": resources.project_b_id},
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/annotations/{resources.annotation_b_id}",
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/projects/{resources.project_b_id}/iaa",
            params={"annotation_type": "entity"},
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/projects/{resources.project_b_id}/import/pubmed/preview",
            json={"pmids": ["11111"]},
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/projects/{resources.project_b_id}/import/pubmed",
            json={"pmids": ["11111"], "include_abstract_only": True},
            headers=headers.member_a,
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/api/co-learning/error-guideline/patterns",
            params={"project_id": resources.project_b_id},
            headers=headers.member_a,
        ).status_code
        == 403
    )

    projects = client.get("/api/projects", headers=headers.member_a)
    assert projects.status_code == 403
    my_work_projects = client.get(
        "/api/projects/my-work",
        params={"workspace_id": resources.workspace_a_id},
        headers=headers.member_a,
    )
    assert my_work_projects.status_code == 200
    assert [project["id"] for project in my_work_projects.json()] == [
        resources.project_a_id
    ]
    assert my_work_projects.json()[0]["settings"] == {}

    documents = client.get("/api/documents", headers=headers.member_a).json()
    assert [document["id"] for document in documents] == [resources.document_a_id]

    annotations = client.get("/api/annotations", headers=headers.member_a).json()
    assert [annotation["id"] for annotation in annotations] == [resources.annotation_a_id]


def test_project_management_mutations_require_manager_role(client, db):
    headers, resources = _create_tenant_fixture(db)

    create_project = client.post(
        "/api/projects",
        json={"name": "authz-member-created", "workspace_id": resources.workspace_a_id},
        headers=headers.member_a,
    )
    assert create_project.status_code == 403

    update_project = client.patch(
        f"/api/projects/{resources.project_a_id}",
        json={"description": "member cannot update"},
        headers=headers.member_a,
    )
    assert update_project.status_code == 403

    create_task = client.post(
        f"/api/projects/{resources.project_a_id}/tasks",
        json={"annotation_type": "sentence_label", "labels": []},
        headers=headers.member_a,
    )
    assert create_task.status_code == 403

    create_assignment = client.post(
        f"/api/projects/{resources.project_a_id}/assignments",
        json={
            "task_id": resources.task_a_id,
            "document_id": resources.document_a_id,
            "annotator_id": "authz-member-a",
        },
        headers=headers.member_a,
    )
    assert create_assignment.status_code == 403

    create_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": resources.project_a_id,
            "version_label": "unauthorized",
            "markdown": "An annotator must not activate project policy.",
        },
        headers=headers.member_a,
    )
    assert create_guideline.status_code == 403

    owner_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": resources.project_a_id,
            "version_label": "owner-authored",
            "markdown": "Manager-controlled policy.",
            "author_id": "forged-author",
        },
        headers=headers.owner_a,
    )
    assert owner_guideline.status_code == 200, owner_guideline.text
    assert owner_guideline.json()["author_id"] == "authz-owner-a"

    allowed_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": resources.project_a_id,
            "document_id": resources.document_a_id,
            "annotation_type": "entity",
            "label": "Condition",
            "start_offset": 6,
            "end_offset": 10,
            "text_span": "beta",
        },
        headers=headers.member_a,
    )
    assert allowed_annotation.status_code == 200

    allowed_update = client.patch(
        f"/api/projects/{resources.project_a_id}",
        json={"description": "owner can update"},
        headers=headers.owner_a,
    )
    assert allowed_update.status_code == 200
