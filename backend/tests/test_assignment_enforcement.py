"""Regression coverage for hybrid/private assignment enforcement.

Team projects without assignments remain open.  Once a project has any task
assignment, annotators and trainers may only see documents assigned to them
through an enabled task, may only use the exact task types assigned to them,
and may only see or mutate their own annotations, corrections, and
submissions.  Managers, admins, and superusers retain project-wide access.

Every request in this module supplies an explicit bearer token so the legacy
test client's implicit admin authentication cannot mask authorization bugs.
"""

from dataclasses import dataclass

import pytest

from al_medlit.core.exceptions import ConflictError


@dataclass(frozen=True)
class AssignmentFixture:
    workspace_id: int
    project_id: int
    open_project_id: int
    shared_document_id: int
    bob_document_id: int
    unassigned_document_id: int
    disabled_document_id: int
    open_document_id: int
    entity_task_id: int
    relation_task_id: int
    doc_label_task_id: int
    disabled_task_id: int
    open_entity_task_id: int
    alice_entity_assignment_id: int
    alice_user_id: int
    bob_user_id: int
    trainer_user_id: int
    manager_user_id: int
    owner_user_id: int
    alice_annotation_ids: tuple[int, int]
    bob_annotation_id: int
    bob_private_annotation_id: int
    automated_annotation_id: int
    alice_headers: dict[str, str]
    bob_headers: dict[str, str]
    trainer_headers: dict[str, str]
    manager_headers: dict[str, str]
    owner_headers: dict[str, str]
    superuser_headers: dict[str, str]


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


def _create_assignment_fixture(db) -> AssignmentFixture:
    from al_medlit.annotation import service as annotation_service
    from al_medlit.annotation.schemas import AnnotationCreate
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import (
        ProjectCreate,
        ProjectTaskCreate,
        TaskAssignmentCreate,
    )
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "assign-owner")
    manager = _make_user(db, "assign-manager")
    alice = _make_user(db, "assign-alice")
    bob = _make_user(db, "assign-bob")
    trainer = _make_user(db, "assign-trainer")
    superuser = _make_user(db, "assign-root", is_superuser=True)

    workspace = workspace_service.create_team_workspace(db, owner, name="Assign WS")
    workspace_service.add_member(db, workspace.id, manager.id, role="manager")
    workspace_service.add_member(db, workspace.id, alice.id, role="annotator")
    workspace_service.add_member(db, workspace.id, bob.id, role="annotator")
    workspace_service.add_member(db, workspace.id, trainer.id, role="trainer")

    project = project_service.create_project(
        db,
        ProjectCreate(name="assign-project", workspace_id=workspace.id),
    )
    shared_document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            title="Shared assignment",
            text="Alpha beta gamma delta.",
        ),
    )
    bob_document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            title="Bob only",
            text="Bob private.",
        ),
    )
    unassigned_document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            title="Unassigned",
            text="Secret unassigned.",
        ),
    )
    disabled_document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=project.id,
            title="Disabled task only",
            text="Disabled only.",
        ),
    )

    entity_task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="entity", labels=[]),
    )
    relation_task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="relation", labels=[]),
    )
    doc_label_task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="doc_label", labels=[]),
    )
    disabled_task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="sentence_label", labels=[]),
    )

    alice_entity_assignment = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=entity_task.id,
            document_id=shared_document.id,
            assignee_user_id=alice.id,
        ),
        assigned_by_user=owner,
    )
    project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=entity_task.id,
            document_id=shared_document.id,
            assignee_user_id=bob.id,
        ),
        assigned_by_user=owner,
    )
    project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=entity_task.id,
            document_id=bob_document.id,
            assignee_user_id=bob.id,
        ),
        assigned_by_user=owner,
    )
    # Create a valid assignment first, then disable the task.  A stale
    # assignment to a disabled task must not grant document or task access.
    project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=disabled_task.id,
            document_id=disabled_document.id,
            assignee_user_id=alice.id,
        ),
        assigned_by_user=owner,
    )
    disabled_task.enabled = False
    db.commit()

    alice_first = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=shared_document.id,
            annotation_type="entity",
            label="AliceAlpha",
            start_offset=0,
            end_offset=5,
            text_span="Alpha",
            annotator_user_id=alice.id,
            annotator_id=alice.username,
        ),
    )
    alice_second = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=shared_document.id,
            annotation_type="entity",
            label="AliceBeta",
            start_offset=6,
            end_offset=10,
            text_span="beta",
            annotator_user_id=alice.id,
            annotator_id=alice.username,
        ),
    )
    bob_annotation = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=shared_document.id,
            annotation_type="entity",
            label="BobGamma",
            start_offset=11,
            end_offset=16,
            text_span="gamma",
            annotator_user_id=bob.id,
            annotator_id=bob.username,
        ),
    )
    automated_annotation = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=shared_document.id,
            annotation_type="entity",
            label="ModelDelta",
            start_offset=17,
            end_offset=22,
            text_span="delta",
            source="model",
        ),
    )
    bob_private_annotation = annotation_service.create_annotation(
        db,
        AnnotationCreate(
            project_id=project.id,
            document_id=bob_document.id,
            annotation_type="entity",
            label="BobPrivate",
            start_offset=0,
            end_offset=3,
            text_span="Bob",
            annotator_user_id=bob.id,
            annotator_id=bob.username,
        ),
    )

    open_project = project_service.create_project(
        db,
        ProjectCreate(name="open-assignment-project", workspace_id=workspace.id),
    )
    open_document = corpus_service.create_document(
        db,
        DocumentCreate(
            project_id=open_project.id,
            title="Open project document",
            text="Open access.",
        ),
    )
    open_entity_task = project_service.create_project_task(
        db,
        open_project.id,
        ProjectTaskCreate(annotation_type="entity", labels=[]),
    )
    db.commit()

    return AssignmentFixture(
        workspace_id=workspace.id,
        project_id=project.id,
        open_project_id=open_project.id,
        shared_document_id=shared_document.id,
        bob_document_id=bob_document.id,
        unassigned_document_id=unassigned_document.id,
        disabled_document_id=disabled_document.id,
        open_document_id=open_document.id,
        entity_task_id=entity_task.id,
        relation_task_id=relation_task.id,
        doc_label_task_id=doc_label_task.id,
        disabled_task_id=disabled_task.id,
        open_entity_task_id=open_entity_task.id,
        alice_entity_assignment_id=alice_entity_assignment.id,
        alice_user_id=alice.id,
        bob_user_id=bob.id,
        trainer_user_id=trainer.id,
        manager_user_id=manager.id,
        owner_user_id=owner.id,
        alice_annotation_ids=(alice_first.id, alice_second.id),
        bob_annotation_id=bob_annotation.id,
        bob_private_annotation_id=bob_private_annotation.id,
        automated_annotation_id=automated_annotation.id,
        alice_headers=_headers_for(alice),
        bob_headers=_headers_for(bob),
        trainer_headers=_headers_for(trainer),
        manager_headers=_headers_for(manager),
        owner_headers=_headers_for(owner),
        superuser_headers=_headers_for(superuser),
    )


def _entity_payload(
    fixture: AssignmentFixture,
    document_id: int,
    *,
    start_offset: int = 0,
    end_offset: int = 5,
    text_span: str = "Alpha",
    label: str = "Condition",
) -> dict:
    return {
        "project_id": fixture.project_id,
        "document_id": document_id,
        "annotation_type": "entity",
        "label": label,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "text_span": text_span,
    }


def _response_ids(response) -> set[int]:
    assert response.status_code == 200, response.text
    return {item["id"] for item in response.json()}


def test_document_get_and_collections_enforce_hybrid_assignment_scope(client, db):
    fixture = _create_assignment_fixture(db)

    assert (
        client.get(
            f"/api/documents/{fixture.shared_document_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 200
    )
    for document_id in (
        fixture.bob_document_id,
        fixture.unassigned_document_id,
        fixture.disabled_document_id,
    ):
        assert (
            client.get(
                f"/api/documents/{document_id}",
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )

    project_default = client.get(
        "/api/documents",
        params={"project_id": fixture.project_id},
        headers=fixture.alice_headers,
    )
    assert _response_ids(project_default) == {fixture.shared_document_id}

    project_mine = client.get(
        "/api/documents",
        params={"project_id": fixture.project_id, "scope": "mine"},
        headers=fixture.alice_headers,
    )
    assert _response_ids(project_mine) == {fixture.shared_document_id}

    # Unscoped collections must apply each project's policy independently:
    # Alice's assigned document plus every document in the open project.
    unscoped = client.get("/api/documents", headers=fixture.alice_headers)
    assert _response_ids(unscoped) == {
        fixture.shared_document_id,
        fixture.open_document_id,
    }
    unscoped_mine = client.get(
        "/api/documents",
        params={"scope": "mine"},
        headers=fixture.alice_headers,
    )
    assert _response_ids(unscoped_mine) == {
        fixture.shared_document_id,
        fixture.open_document_id,
    }

    for params in (
        {"project_id": fixture.project_id, "scope": "all"},
        {"scope": "all"},
    ):
        assert (
            client.get(
                "/api/documents",
                params=params,
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )

    assert (
        client.get(
            f"/api/documents/{fixture.unassigned_document_id}",
            headers=fixture.trainer_headers,
        ).status_code
        == 403
    )


def test_workbench_returns_only_current_users_editing_layer(client, db):
    fixture = _create_assignment_fixture(db)

    response = client.get(
        f"/api/annotation-workbench/documents/{fixture.shared_document_id}",
        headers=fixture.alice_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert {task["id"] for task in payload["tasks"]} == {fixture.entity_task_id}
    assert {item["id"] for item in payload["annotations"]} == set(fixture.alice_annotation_ids)
    assert {item["id"] for item in payload["assignments"]} == {fixture.alice_entity_assignment_id}

    assert (
        client.get(
            f"/api/annotation-workbench/documents/{fixture.bob_document_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/annotation-workbench/documents/{fixture.disabled_document_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )

    manager = client.get(
        f"/api/annotation-workbench/documents/{fixture.shared_document_id}",
        headers=fixture.manager_headers,
    )
    assert manager.status_code == 200, manager.text
    manager_payload = manager.json()
    assert {task["id"] for task in manager_payload["tasks"]} == {
        fixture.entity_task_id,
        fixture.relation_task_id,
        fixture.doc_label_task_id,
    }
    assert manager_payload["annotations"] == []
    assert manager_payload["assignments"] == []
    assert manager_payload["correction_locked_annotation_ids"] == []


def test_reassignment_cannot_give_annotator_two_open_version_rounds(client):
    project = client.post(
        "/api/projects",
        json={
            "name": "reassignment-open-round-invariant",
            "tasks": [{"annotation_type": "entity"}],
        },
    ).json()
    document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Versioned assignment document",
            "text": "A sentence for annotation.",
        },
    ).json()
    task_id = project["tasks"][0]["id"]
    alice_assignment = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": task_id,
            "document_id": document["id"],
            "annotator_id": "round-alice",
        },
    ).json()

    rebuilt = client.post(
        f"/api/documents/{document['id']}/structure/rebuild",
        json={"activate": True},
    )
    assert rebuilt.status_code == 200, rebuilt.text
    bob_assignment = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": task_id,
            "document_id": document["id"],
            "annotator_id": "round-bob",
        },
    ).json()
    assert bob_assignment["structure_version_id"] != alice_assignment["structure_version_id"]

    response = client.patch(
        f"/api/projects/{project['id']}/assignments/{bob_assignment['id']}",
        json={"assignee_user_id": alice_assignment["assignee_user_id"]},
    )
    assert response.status_code == 409
    assert "current assignment round" in response.json()["detail"]


def test_evidence_open_rounds_use_logical_target_identity(db):
    from al_medlit.corpus import service as corpus_service
    from al_medlit.corpus.schemas import DocumentCreate
    from al_medlit.evidence import service as evidence_service
    from al_medlit.evidence.schemas import (
        EvidenceTargetCreate,
        EvidenceTargetVersionCreate,
    )
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import (
        ProjectCreate,
        ProjectTaskCreate,
        TaskAssignmentCreate,
        TaskAssignmentUpdate,
    )
    from al_medlit.workspace import service as workspace_service

    owner = _make_user(db, "logical-target-owner")
    alice = _make_user(db, "logical-target-alice")
    bob = _make_user(db, "logical-target-bob")
    workspace = workspace_service.create_team_workspace(
        db,
        owner,
        name="Logical Target Rounds",
    )
    workspace_service.add_member(db, workspace.id, alice.id, role="annotator")
    workspace_service.add_member(db, workspace.id, bob.id, role="annotator")
    project = project_service.create_project(
        db,
        ProjectCreate(name="logical-target-rounds", workspace_id=workspace.id),
    )
    document = corpus_service.create_document(
        db,
        DocumentCreate(project_id=project.id, text="Alpha. Beta."),
    )
    task = project_service.create_project_task(
        db,
        project.id,
        ProjectTaskCreate(annotation_type="evidence_block", labels=[]),
    )

    target_a = evidence_service.create_target(
        db,
        project.id,
        EvidenceTargetCreate(
            task_id=task.id,
            key="logical-a",
            name="Logical A",
            initial_version=EvidenceTargetVersionCreate(text="A v1"),
        ),
        actor_user_id=owner.id,
    )
    version_a1 = target_a.versions[0]
    evidence_service.activate_target(db, project.id, target_a.id, version_a1.id)
    alice_a1 = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=document.id,
            assignee_user_id=alice.id,
            target_version_id=version_a1.id,
        ),
        assigned_by_user=owner,
    )

    version_a2 = evidence_service.create_target_version(
        db,
        project.id,
        target_a.id,
        EvidenceTargetVersionCreate(text="A v2"),
        actor_user_id=owner.id,
    )
    evidence_service.activate_target(db, project.id, target_a.id, version_a2.id)
    with pytest.raises(ConflictError, match="current assignment round"):
        project_service.create_task_assignment(
            db,
            project.id,
            TaskAssignmentCreate(
                task_id=task.id,
                document_id=document.id,
                assignee_user_id=alice.id,
                target_version_id=version_a2.id,
            ),
            assigned_by_user=owner,
        )

    bob_a2 = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=document.id,
            assignee_user_id=bob.id,
            target_version_id=version_a2.id,
        ),
        assigned_by_user=owner,
    )
    with pytest.raises(ConflictError, match="current assignment round"):
        project_service.update_task_assignment(
            db,
            project.id,
            bob_a2.id,
            TaskAssignmentUpdate(assignee_user_id=alice.id),
        )

    target_b = evidence_service.create_target(
        db,
        project.id,
        EvidenceTargetCreate(
            task_id=task.id,
            key="logical-b",
            name="Logical B",
            initial_version=EvidenceTargetVersionCreate(text="B v1"),
        ),
        actor_user_id=owner.id,
    )
    version_b1 = target_b.versions[0]
    evidence_service.activate_target(db, project.id, target_b.id, version_b1.id)
    alice_b1 = project_service.create_task_assignment(
        db,
        project.id,
        TaskAssignmentCreate(
            task_id=task.id,
            document_id=document.id,
            assignee_user_id=alice.id,
            target_version_id=version_b1.id,
        ),
        assigned_by_user=owner,
    )
    assert alice_a1.target_version.target_id == target_a.id
    assert alice_b1.target_version.target_id == target_b.id


def test_annotation_collections_filter_documents_tasks_and_ownership(client, db):
    fixture = _create_assignment_fixture(db)
    expected_alice = set(fixture.alice_annotation_ids)

    for params in (
        {},
        {"project_id": fixture.project_id},
        {"document_id": fixture.shared_document_id},
        {"project_id": fixture.project_id, "annotator_id": "assign-alice"},
    ):
        response = client.get(
            "/api/annotations",
            params=params,
            headers=fixture.alice_headers,
        )
        assert _response_ids(response) == expected_alice

    for document_id in (fixture.bob_document_id, fixture.unassigned_document_id):
        assert (
            client.get(
                "/api/annotations",
                params={"document_id": document_id},
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )

    targeted_other_user = client.get(
        "/api/annotations",
        params={
            "project_id": fixture.project_id,
            "annotator_id": "assign-bob",
        },
        headers=fixture.alice_headers,
    )
    assert targeted_other_user.status_code == 403

    for assignment_filter in (
        {"assignee_user_id": fixture.bob_user_id},
        {"annotator_id": "assign-bob"},
    ):
        targeted_assignments = client.get(
            f"/api/projects/{fixture.project_id}/assignments",
            params=assignment_filter,
            headers=fixture.alice_headers,
        )
        assert targeted_assignments.status_code == 403

    trainer = client.get(
        "/api/annotations",
        params={"project_id": fixture.project_id},
        headers=fixture.trainer_headers,
    )
    assert trainer.status_code == 200
    assert trainer.json() == []

    manager = client.get(
        "/api/annotations",
        params={"project_id": fixture.project_id},
        headers=fixture.manager_headers,
    )
    assert _response_ids(manager) == {
        *fixture.alice_annotation_ids,
        fixture.bob_annotation_id,
        fixture.bob_private_annotation_id,
        fixture.automated_annotation_id,
    }


def test_annotation_create_requires_exact_enabled_task_assignment(client, db):
    fixture = _create_assignment_fixture(db)

    entity = client.post(
        "/api/annotations",
        json={
            **_entity_payload(
                fixture,
                fixture.shared_document_id,
                start_offset=17,
                end_offset=22,
                text_span="delta",
                label="AliceNewEntity",
            ),
            # Identity fields are accepted for backwards-compatible wire
            # parsing but must always be overwritten from the token.
            "annotator_user_id": fixture.bob_user_id,
            "annotator_id": "assign-bob",
        },
        headers=fixture.alice_headers,
    )
    assert entity.status_code == 200, entity.text
    assert entity.json()["annotator_user_id"] == fixture.alice_user_id
    assert entity.json()["annotator_id"] == "assign-alice"

    relation = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "annotation_type": "relation",
            "label": "Related",
            "head_annotation_id": fixture.alice_annotation_ids[0],
            "tail_annotation_id": fixture.alice_annotation_ids[1],
        },
        headers=fixture.alice_headers,
    )
    assert relation.status_code == 403

    document_label = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "annotation_type": "doc_label",
            "label": "Positive",
        },
        headers=fixture.alice_headers,
    )
    assert document_label.status_code == 403

    no_configured_task = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "annotation_type": "passage_label",
            "label": "Relevant",
            "start_offset": 0,
            "end_offset": 5,
            "text_span": "Alpha",
        },
        headers=fixture.alice_headers,
    )
    assert no_configured_task.status_code == 403

    stale_disabled_assignment = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.disabled_document_id,
            "annotation_type": "sentence_label",
            "label": "Relevant",
            "start_offset": 0,
            "end_offset": 8,
            "text_span": "Disabled",
        },
        headers=fixture.alice_headers,
    )
    assert stale_disabled_assignment.status_code == 403


def test_annotation_writes_revalidate_a_cached_manager_role(
    client,
    db,
    monkeypatch,
):
    from al_medlit.annotation import router as annotation_router
    from al_medlit.annotation.models import Annotation, AnnotationCorrection
    from al_medlit.workspace.models import WorkspaceMember

    fixture = _create_assignment_fixture(db)
    original = client.post(
        "/api/annotations",
        json=_entity_payload(
            fixture,
            fixture.unassigned_document_id,
            start_offset=0,
            end_offset=6,
            text_span="Secret",
            label="ManagerOriginal",
        ),
        headers=fixture.manager_headers,
    )
    assert original.status_code == 200, original.text
    original_id = original.json()["id"]

    real_lock = annotation_router.lock_document_resource_for_mutation
    recheck_calls: list[int] = []

    def demote_before_locked_recheck(session, user, **kwargs):
        recheck_calls.append(user.id)
        session.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == fixture.workspace_id,
            WorkspaceMember.user_id == fixture.manager_user_id,
        ).update(
            {WorkspaceMember.role: "annotator"},
            synchronize_session=False,
        )
        return real_lock(session, user, **kwargs)

    monkeypatch.setattr(
        annotation_router,
        "lock_document_resource_for_mutation",
        demote_before_locked_recheck,
    )

    create = client.post(
        "/api/annotations",
        json=_entity_payload(
            fixture,
            fixture.unassigned_document_id,
            start_offset=7,
            end_offset=17,
            text_span="unassigned",
            label="StaleManagerCreate",
        ),
        headers=fixture.manager_headers,
    )
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == fixture.workspace_id,
        WorkspaceMember.user_id == fixture.manager_user_id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    db.commit()
    update = client.patch(
        f"/api/annotations/{original_id}",
        json={"label": "StaleManagerUpdate"},
        headers=fixture.manager_headers,
    )
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == fixture.workspace_id,
        WorkspaceMember.user_id == fixture.manager_user_id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    db.commit()
    delete = client.delete(
        f"/api/annotations/{original_id}",
        headers=fixture.manager_headers,
    )
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == fixture.workspace_id,
        WorkspaceMember.user_id == fixture.manager_user_id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    db.commit()
    correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.unassigned_document_id,
            "original_annotation_id": original_id,
            "correction_note": "stale manager must not write",
        },
        headers=fixture.manager_headers,
    )

    assert [response.status_code for response in (create, update, delete, correction)] == [
        403,
        403,
        403,
        403,
    ]
    assert recheck_calls == [fixture.manager_user_id] * 4
    db.expire_all()
    assert db.get(Annotation, original_id).label == "ManagerOriginal"
    assert db.query(Annotation).filter(Annotation.label == "StaleManagerCreate").first() is None
    assert db.query(AnnotationCorrection).count() == 0


def test_other_domain_writes_revalidate_cached_manager_role(
    client,
    db,
    monkeypatch,
):
    from al_medlit.evidence import router as evidence_router
    from al_medlit.submission import router as submission_router
    from al_medlit.workspace.models import WorkspaceMember

    fixture = _create_assignment_fixture(db)
    recheck_calls: list[str] = []

    def demote(session):
        session.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == fixture.workspace_id,
            WorkspaceMember.user_id == fixture.manager_user_id,
        ).update(
            {WorkspaceMember.role: "annotator"},
            synchronize_session=False,
        )

    real_project_lock = evidence_router.lock_project_member_for_mutation

    def demote_before_project_recheck(session, user, project_id, **kwargs):
        recheck_calls.append("project")
        demote(session)
        return real_project_lock(session, user, project_id, **kwargs)

    real_document_lock = submission_router.lock_document_resource_for_mutation

    def demote_before_document_recheck(session, user, **kwargs):
        recheck_calls.append("document")
        demote(session)
        return real_document_lock(session, user, **kwargs)

    monkeypatch.setattr(
        evidence_router,
        "lock_project_member_for_mutation",
        demote_before_project_recheck,
    )
    monkeypatch.setattr(
        submission_router,
        "lock_document_resource_for_mutation",
        demote_before_document_recheck,
    )

    target = client.post(
        f"/api/projects/{fixture.project_id}/evidence-targets",
        json={
            "task_id": fixture.entity_task_id,
            "key": "stale-manager-target",
            "name": "Stale manager target",
            "initial_version": {"text": "Target"},
        },
        headers=fixture.manager_headers,
    )
    db.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == fixture.workspace_id,
        WorkspaceMember.user_id == fixture.manager_user_id,
    ).update({WorkspaceMember.role: "manager"}, synchronize_session=False)
    db.commit()
    submission = client.post(
        f"/api/projects/{fixture.project_id}/documents/"
        f"{fixture.unassigned_document_id}/submissions",
        json={"kind": "re_export"},
        headers=fixture.manager_headers,
    )

    assert target.status_code == 403
    assert submission.status_code == 403
    assert recheck_calls == ["project", "document"]


def test_open_project_write_revalidates_a_cached_removed_member(
    client,
    db,
    monkeypatch,
):
    from al_medlit.annotation import router as annotation_router
    from al_medlit.workspace.models import WorkspaceMember

    fixture = _create_assignment_fixture(db)
    real_lock = annotation_router.lock_document_resource_for_mutation
    recheck_calls = 0

    def remove_before_locked_recheck(session, user, **kwargs):
        nonlocal recheck_calls
        recheck_calls += 1
        session.query(WorkspaceMember).filter(
            WorkspaceMember.workspace_id == fixture.workspace_id,
            WorkspaceMember.user_id == fixture.alice_user_id,
        ).delete(synchronize_session=False)
        return real_lock(session, user, **kwargs)

    monkeypatch.setattr(
        annotation_router,
        "lock_document_resource_for_mutation",
        remove_before_locked_recheck,
    )

    response = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.open_project_id,
            "document_id": fixture.open_document_id,
            "annotation_type": "entity",
            "label": "RemovedMember",
            "start_offset": 0,
            "end_offset": 4,
            "text_span": "Open",
        },
        headers=fixture.alice_headers,
    )

    assert response.status_code == 403
    assert recheck_calls == 1


def test_coassigned_annotators_cannot_read_update_or_delete_each_others_work(client, db):
    fixture = _create_assignment_fixture(db)

    for annotation_id, headers in (
        (fixture.bob_annotation_id, fixture.alice_headers),
        (fixture.alice_annotation_ids[0], fixture.bob_headers),
        (fixture.automated_annotation_id, fixture.alice_headers),
    ):
        assert (
            client.get(
                f"/api/annotations/{annotation_id}",
                headers=headers,
            ).status_code
            == 403
        )
        assert (
            client.patch(
                f"/api/annotations/{annotation_id}",
                json={"label": "Hijacked"},
                headers=headers,
            ).status_code
            == 403
        )
        assert (
            client.delete(
                f"/api/annotations/{annotation_id}",
                headers=headers,
            ).status_code
            == 403
        )

    for annotation_id in fixture.alice_annotation_ids:
        assert (
            client.get(
                f"/api/annotations/{annotation_id}",
                headers=fixture.alice_headers,
            ).status_code
            == 200
        )

    assert (
        client.get(
            f"/api/annotations/{fixture.bob_annotation_id}",
            headers=fixture.manager_headers,
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"/api/annotations/{fixture.bob_annotation_id}",
            json={"label": "Manager rewrite"},
            headers=fixture.manager_headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(
            f"/api/annotations/{fixture.bob_annotation_id}",
            headers=fixture.manager_headers,
        ).status_code
        == 403
    )


def test_annotation_patch_rejects_explicit_null_for_required_fields(client, db):
    fixture = _create_assignment_fixture(db)

    for field in ("label", "status", "evidence", "attributes"):
        response = client.patch(
            f"/api/annotations/{fixture.alice_annotation_ids[0]}",
            json={field: None},
            headers=fixture.alice_headers,
        )
        assert response.status_code == 422, response.text


def test_relation_create_and_update_reject_other_annotators_references(client, db):
    from al_medlit.auth.models import User
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import TaskAssignmentCreate

    fixture = _create_assignment_fixture(db)
    owner = db.get(User, fixture.owner_user_id)
    project_service.create_task_assignment(
        db,
        fixture.project_id,
        TaskAssignmentCreate(
            task_id=fixture.relation_task_id,
            document_id=fixture.shared_document_id,
            assignee_user_id=fixture.alice_user_id,
        ),
        assigned_by_user=owner,
    )

    private_relation = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "annotation_type": "relation",
            "label": "PrivateRefs",
            "head_annotation_id": fixture.alice_annotation_ids[0],
            "tail_annotation_id": fixture.alice_annotation_ids[1],
        },
        headers=fixture.alice_headers,
    )
    assert private_relation.status_code == 200, private_relation.text

    for foreign_field in ("head_annotation_id", "tail_annotation_id"):
        relation_payload = {
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "annotation_type": "relation",
            "label": f"Foreign{foreign_field}",
            "head_annotation_id": fixture.alice_annotation_ids[0],
            "tail_annotation_id": fixture.alice_annotation_ids[1],
        }
        relation_payload[foreign_field] = fixture.bob_annotation_id
        foreign_reference = client.post(
            "/api/annotations",
            json=relation_payload,
            headers=fixture.alice_headers,
        )
        assert foreign_reference.status_code == 422
        assert foreign_reference.json()["detail"] == (
            "One or more referenced annotations are unavailable"
        )

    for foreign_field in ("head_annotation_id", "tail_annotation_id"):
        update_to_foreign_reference = client.patch(
            f"/api/annotations/{private_relation.json()['id']}",
            json={foreign_field: fixture.bob_annotation_id},
            headers=fixture.alice_headers,
        )
        assert update_to_foreign_reference.status_code == 422
        assert update_to_foreign_reference.json()["detail"] == (
            "One or more referenced annotations are unavailable"
        )


def test_corrections_are_creator_private_and_references_are_private(client, db):
    from al_medlit.annotation.models import AnnotationCorrection

    fixture = _create_assignment_fixture(db)
    alice_correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "original_annotation_id": fixture.alice_annotation_ids[0],
            "correction_note": "Alice correction",
            "correction_source": "model",
            # Ownership is server-derived and cannot be spoofed.
            "created_by_user_id": fixture.bob_user_id,
        },
        headers=fixture.alice_headers,
    )
    assert alice_correction.status_code == 200, alice_correction.text
    assert alice_correction.json()["created_by_user_id"] == fixture.alice_user_id
    assert alice_correction.json()["correction_source"] == "human"

    bob_correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "original_annotation_id": fixture.bob_annotation_id,
            "correction_note": "Bob correction",
        },
        headers=fixture.bob_headers,
    )
    assert bob_correction.status_code == 200, bob_correction.text
    assert bob_correction.json()["created_by_user_id"] == fixture.bob_user_id

    for foreign_field in ("original_annotation_id", "corrected_annotation_id"):
        foreign_reference = client.post(
            "/api/annotations/corrections",
            json={
                "project_id": fixture.project_id,
                "document_id": fixture.shared_document_id,
                foreign_field: fixture.bob_annotation_id,
                "correction_note": "Alice must not reference Bob's annotation",
            },
            headers=fixture.alice_headers,
        )
        assert foreign_reference.status_code == 422
        assert foreign_reference.json()["detail"] == (
            "One or more referenced annotations are unavailable"
        )

    # Simulate a legacy/ambiguous correction that could not be attributed
    # during migration.  It is intentionally manager-only.
    legacy = AnnotationCorrection(
        project_id=fixture.project_id,
        document_id=fixture.shared_document_id,
        original_annotation_id=None,
        corrected_annotation_id=None,
        created_by_user_id=None,
        correction_note="Ambiguous legacy correction",
        metadata_={},
    )
    db.add(legacy)
    db.commit()

    alice_rows = client.get(
        "/api/annotations/corrections",
        params={"project_id": fixture.project_id},
        headers=fixture.alice_headers,
    )
    assert _response_ids(alice_rows) == {alice_correction.json()["id"]}

    alice_unscoped = client.get(
        "/api/annotations/corrections",
        headers=fixture.alice_headers,
    )
    assert _response_ids(alice_unscoped) == {alice_correction.json()["id"]}

    bob_rows = client.get(
        "/api/annotations/corrections",
        params={"project_id": fixture.project_id},
        headers=fixture.bob_headers,
    )
    assert _response_ids(bob_rows) == {bob_correction.json()["id"]}

    manager_rows = client.get(
        "/api/annotations/corrections",
        params={"project_id": fixture.project_id},
        headers=fixture.manager_headers,
    )
    assert _response_ids(manager_rows) == {
        alice_correction.json()["id"],
        bob_correction.json()["id"],
        legacy.id,
    }


def test_submission_lists_get_and_download_require_ownership_and_current_access(
    client,
    db,
):
    from al_medlit.project.models import TaskAssignment

    fixture = _create_assignment_fixture(db)
    alice_submission = client.post(
        f"/api/projects/{fixture.project_id}/documents/{fixture.shared_document_id}/submissions",
        json={
            # Submission identity, like annotation identity, is token-derived.
            "annotator_user_id": fixture.bob_user_id,
            "annotator_id": "assign-bob",
        },
        headers=fixture.alice_headers,
    )
    assert alice_submission.status_code == 200, alice_submission.text
    alice_submission_id = alice_submission.json()["id"]
    assert alice_submission.json()["annotator_user_id"] == fixture.alice_user_id
    assert alice_submission.json()["annotator_id"] == "assign-alice"

    bob_submission = client.post(
        f"/api/projects/{fixture.project_id}/documents/{fixture.shared_document_id}/submissions",
        json={},
        headers=fixture.bob_headers,
    )
    assert bob_submission.status_code == 200, bob_submission.text
    bob_submission_id = bob_submission.json()["id"]

    alice_list = client.get(
        f"/api/projects/{fixture.project_id}/submissions",
        headers=fixture.alice_headers,
    )
    assert _response_ids(alice_list) == {alice_submission_id}
    alice_mine = client.get(
        f"/api/projects/{fixture.project_id}/submissions",
        params={"scope": "mine"},
        headers=fixture.alice_headers,
    )
    assert _response_ids(alice_mine) == {alice_submission_id}
    assert (
        client.get(
            f"/api/submissions/{alice_submission_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 200
    )
    download = client.get(
        f"/api/submissions/{alice_submission_id}/download",
        headers=fixture.alice_headers,
    )
    assert download.status_code == 200, download.text
    snapshot = download.json()
    assert snapshot["document"]["id"] == fixture.shared_document_id
    assert {item["id"] for item in snapshot["annotations"]} == set(fixture.alice_annotation_ids)

    for path in (
        f"/api/submissions/{bob_submission_id}",
        f"/api/submissions/{bob_submission_id}/download",
    ):
        assert client.get(path, headers=fixture.alice_headers).status_code == 403

    for params in (
        {"scope": "all"},
        {"annotator_user_id": fixture.bob_user_id},
        {"annotator_id": "assign-bob"},
    ):
        assert (
            client.get(
                f"/api/projects/{fixture.project_id}/submissions",
                params=params,
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )

    manager_list = client.get(
        f"/api/projects/{fixture.project_id}/submissions",
        params={"scope": "all"},
        headers=fixture.manager_headers,
    )
    assert _response_ids(manager_list) == {alice_submission_id, bob_submission_id}
    assert (
        client.get(
            f"/api/submissions/{alice_submission_id}/download",
            headers=fixture.manager_headers,
        ).status_code
        == 200
    )

    # Revoking Alice's only enabled assignment to the document immediately
    # removes access even to her previously-created submission snapshot.
    db.query(TaskAssignment).filter(TaskAssignment.id == fixture.alice_entity_assignment_id).delete(
        synchronize_session=False
    )
    db.commit()

    revoked_list = client.get(
        f"/api/projects/{fixture.project_id}/submissions",
        headers=fixture.alice_headers,
    )
    assert revoked_list.status_code == 200
    assert revoked_list.json() == []
    assert (
        client.get(
            f"/api/projects/{fixture.project_id}/submissions",
            params={"document_id": fixture.shared_document_id},
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    for path in (
        f"/api/submissions/{alice_submission_id}",
        f"/api/submissions/{alice_submission_id}/download",
    ):
        assert client.get(path, headers=fixture.alice_headers).status_code == 403
    assert (
        client.post(
            f"/api/projects/{fixture.project_id}"
            f"/documents/{fixture.shared_document_id}/submissions",
            json={},
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )


def test_submission_creation_rejects_unassigned_document(client, db):
    fixture = _create_assignment_fixture(db)

    for document_id in (fixture.bob_document_id, fixture.unassigned_document_id):
        assert (
            client.get(
                f"/api/projects/{fixture.project_id}/submissions",
                params={"document_id": document_id},
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )
        assert (
            client.post(
                f"/api/projects/{fixture.project_id}/documents/{document_id}/submissions",
                json={},
                headers=fixture.alice_headers,
            ).status_code
            == 403
        )


def test_zero_assignment_project_is_open_then_first_assignment_restricts_it(client, db):
    from al_medlit.auth.models import User
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import TaskAssignmentCreate

    fixture = _create_assignment_fixture(db)

    assert (
        client.get(
            f"/api/documents/{fixture.open_document_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 200
    )
    open_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.open_project_id,
            "document_id": fixture.open_document_id,
            "annotation_type": "entity",
            "label": "OpenAlice",
            "start_offset": 0,
            "end_offset": 4,
            "text_span": "Open",
        },
        headers=fixture.alice_headers,
    )
    assert open_annotation.status_code == 200, open_annotation.text

    # Private ownership still applies while the project is open.
    bob_open_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": fixture.open_project_id,
            "document_id": fixture.open_document_id,
            "annotation_type": "entity",
            "label": "OpenBob",
            "start_offset": 5,
            "end_offset": 11,
            "text_span": "access",
        },
        headers=fixture.bob_headers,
    )
    assert bob_open_annotation.status_code == 200, bob_open_annotation.text
    alice_rows = client.get(
        "/api/annotations",
        params={"project_id": fixture.open_project_id},
        headers=fixture.alice_headers,
    )
    assert _response_ids(alice_rows) == {open_annotation.json()["id"]}

    owner = db.get(User, fixture.owner_user_id)
    project_service.create_task_assignment(
        db,
        fixture.open_project_id,
        TaskAssignmentCreate(
            task_id=fixture.open_entity_task_id,
            document_id=fixture.open_document_id,
            assignee_user_id=fixture.bob_user_id,
        ),
        assigned_by_user=owner,
    )

    assert (
        client.get(
            f"/api/documents/{fixture.open_document_id}",
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    after_assignment = client.get(
        "/api/annotations",
        params={"project_id": fixture.open_project_id},
        headers=fixture.alice_headers,
    )
    assert after_assignment.status_code == 200
    assert after_assignment.json() == []
    assert (
        client.post(
            "/api/annotations",
            json={
                "project_id": fixture.open_project_id,
                "document_id": fixture.open_document_id,
                "annotation_type": "entity",
                "label": "NoLongerOpen",
                "start_offset": 0,
                "end_offset": 4,
                "text_span": "Open",
            },
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    assert (
        client.get(
            f"/api/documents/{fixture.open_document_id}",
            headers=fixture.bob_headers,
        ).status_code
        == 200
    )


def test_assignment_with_persisted_work_cannot_be_revoked_by_deletion(
    client,
    db,
):
    from al_medlit.auth.models import User
    from al_medlit.project import service as project_service
    from al_medlit.project.schemas import TaskAssignmentCreate

    fixture = _create_assignment_fixture(db)
    correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "original_annotation_id": fixture.alice_annotation_ids[0],
            "correction_note": "Alice-owned entity correction",
        },
        headers=fixture.alice_headers,
    )
    assert correction.status_code == 200, correction.text
    correction_id = correction.json()["id"]
    manager_correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "original_annotation_id": fixture.alice_annotation_ids[1],
            "correction_note": "Manager-owned lock on Alice work",
        },
        headers=fixture.manager_headers,
    )
    assert manager_correction.status_code == 200, manager_correction.text
    historical_submission = client.post(
        f"/api/projects/{fixture.project_id}/documents/{fixture.shared_document_id}/submissions",
        json={},
        headers=fixture.alice_headers,
    )
    assert historical_submission.status_code == 200, historical_submission.text
    historical_submission_id = historical_submission.json()["id"]

    before = client.get(
        f"/api/annotation-workbench/documents/{fixture.shared_document_id}",
        headers=fixture.alice_headers,
    )
    assert set(before.json()["correction_locked_annotation_ids"]) == set(
        fixture.alice_annotation_ids
    )
    assert _response_ids(
        client.get(
            "/api/annotations/corrections",
            params={"project_id": fixture.project_id},
            headers=fixture.alice_headers,
        )
    ) == {correction_id}

    owner = db.get(User, fixture.owner_user_id)
    project_service.create_task_assignment(
        db,
        fixture.project_id,
        TaskAssignmentCreate(
            task_id=fixture.relation_task_id,
            document_id=fixture.shared_document_id,
            assignee_user_id=fixture.alice_user_id,
        ),
        assigned_by_user=owner,
    )
    rejected = client.delete(
        f"/api/projects/{fixture.project_id}/assignments/{fixture.alice_entity_assignment_id}",
        headers=fixture.manager_headers,
    )
    assert rejected.status_code == 409
    assert "submissions or work exist" in rejected.json()["detail"]

    # The failed revocation is atomic: the assignment and every immutable pin
    # remain in place, so historical records do not become detached.
    after = client.get(
        f"/api/annotation-workbench/documents/{fixture.shared_document_id}",
        headers=fixture.alice_headers,
    )
    assert after.status_code == 200, after.text
    assert {task["id"] for task in after.json()["tasks"]} == {
        fixture.entity_task_id,
        fixture.relation_task_id,
    }
    assert {item["id"] for item in after.json()["annotations"]} == set(fixture.alice_annotation_ids)
    assert set(after.json()["correction_locked_annotation_ids"]) == set(
        fixture.alice_annotation_ids
    )
    assert (
        client.get(
            f"/api/submissions/{historical_submission_id}/download",
            headers=fixture.alice_headers,
        ).status_code
        == 200
    )


def test_manager_admin_and_superuser_bypass_assignment_and_private_ownership(
    client,
    db,
):
    fixture = _create_assignment_fixture(db)

    for headers in (
        fixture.manager_headers,
        fixture.owner_headers,
        fixture.superuser_headers,
    ):
        assert (
            client.get(
                f"/api/documents/{fixture.unassigned_document_id}",
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"/api/annotations/{fixture.bob_annotation_id}",
                headers=headers,
            ).status_code
            == 200
        )
        rows = client.get(
            "/api/annotations",
            params={"project_id": fixture.project_id},
            headers=headers,
        )
        assert fixture.bob_annotation_id in _response_ids(rows)

    manager_annotation = client.post(
        "/api/annotations",
        json=_entity_payload(
            fixture,
            fixture.unassigned_document_id,
            start_offset=0,
            end_offset=6,
            text_span="Secret",
            label="ManagerBypass",
        ),
        headers=fixture.manager_headers,
    )
    assert manager_annotation.status_code == 200, manager_annotation.text

    manager_submission = client.post(
        f"/api/projects/{fixture.project_id}"
        f"/documents/{fixture.unassigned_document_id}/submissions",
        json={"kind": "re_export"},
        headers=fixture.manager_headers,
    )
    assert manager_submission.status_code == 200, manager_submission.text
    submission_id = manager_submission.json()["id"]
    for headers in (
        fixture.manager_headers,
        fixture.owner_headers,
        fixture.superuser_headers,
    ):
        assert (
            client.get(
                f"/api/submissions/{submission_id}",
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.get(
                f"/api/submissions/{submission_id}/download",
                headers=headers,
            ).status_code
            == 200
        )


def test_cross_annotator_derived_surfaces_require_manager_role(client, db):
    from al_medlit.workspace import capability_service

    fixture = _create_assignment_fixture(db)
    capability_service.set_capability(
        db,
        fixture.workspace_id,
        preset="full",
        actor_user_id=fixture.owner_user_id,
    )
    db.commit()
    correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": fixture.project_id,
            "document_id": fixture.shared_document_id,
            "original_annotation_id": fixture.alice_annotation_ids[0],
            "error_type": "boundary_error",
        },
        headers=fixture.alice_headers,
    )
    assert correction.status_code == 200, correction.text

    iaa_path = f"/api/projects/{fixture.project_id}/iaa"
    iaa_params = {"annotation_type": "entity"}
    assert (
        client.get(
            iaa_path,
            params=iaa_params,
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    assert (
        client.get(
            iaa_path,
            params=iaa_params,
            headers=fixture.manager_headers,
        ).status_code
        == 200
    )

    patterns_path = "/api/co-learning/error-guideline/patterns"
    patterns_params = {"project_id": fixture.project_id}
    assert (
        client.get(
            patterns_path,
            params=patterns_params,
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    assert (
        client.get(
            patterns_path,
            params=patterns_params,
            headers=fixture.manager_headers,
        ).status_code
        == 200
    )

    correction_pattern_path = (
        f"/api/co-learning/error-guideline/patterns/from-correction/{correction.json()['id']}"
    )
    assert (
        client.post(
            correction_pattern_path,
            headers=fixture.alice_headers,
        ).status_code
        == 403
    )
    assert (
        client.post(
            correction_pattern_path,
            headers=fixture.manager_headers,
        ).status_code
        == 200
    )
