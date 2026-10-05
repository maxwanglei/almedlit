"""Project summaries count saved tasks, independently of their annotation rounds."""

from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.auth.security import create_access_token
from al_medlit.project import service as project_service
from al_medlit.project.models import Project
from al_medlit.workflow import models, schemas, service


def _headers(user):
    return {"Authorization": f"Bearer {create_access_token(str(user.id))}"}


def _add_ner_and_round(db, scope, data):
    service.create_task_definition(
        db,
        schemas.TaskDefinitionCreate(
            project_id=scope.project.id,
            key="named-entities",
            name="Named entities",
        ),
        scope.manager,
    )
    service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=scope.project.id,
            task_definition_id=data.task.id,
            task_kind="classification",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        ),
        scope.manager,
    )
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=scope.project.id,
            name="Classification only",
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[scope.annotator.id],
        ),
        scope.manager,
    )
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")


def test_project_responses_count_all_tasks_without_counting_versions_as_tasks(
    client, db, workflow_scope, workflow_data,
):
    scope, data = workflow_scope, workflow_data
    _add_ner_and_round(db, scope, data)
    manager_headers = _headers(scope.manager)

    responses = [
        client.get(f"/api/projects/{scope.project.id}", headers=manager_headers),
        client.patch(
            f"/api/projects/{scope.project.id}",
            json={"description": "Two tasks, one active round"},
            headers=manager_headers,
        ),
    ]
    for response in responses:
        assert response.status_code == 200, response.text
        assert response.json()["tasks"] == []
        assert response.json()["workflow_task_count"] == 2
        assert response.json()["workflow_round_count"] == 1

    listed = client.get(
        "/api/projects", params={"workspace_id": scope.workspace.id},
        headers=manager_headers,
    )
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["workflow_task_count"] == 2
    assert listed.json()[0]["workflow_round_count"] == 1

    assigned = client.get(
        "/api/projects/my-work", params={"workspace_id": scope.workspace.id},
        headers=_headers(scope.annotator),
    )
    assert assigned.status_code == 200, assigned.text
    assert assigned.json()[0]["workflow_task_count"] == 2
    assert assigned.json()[0]["workflow_round_count"] == 1
    assert assigned.json()[0]["settings"] == {}


def test_project_counts_remain_scoped_to_authorized_projects(
    client, db, workflow_scope, workflow_data,
):
    scope = workflow_scope
    _add_ner_and_round(db, scope, workflow_data)
    empty = Project(name="No tasks", workspace_id=scope.workspace.id)
    db.add(empty)
    db.commit()

    summaries = project_service.read_projects(db, [scope.project, empty])
    assert [(item.workflow_task_count, item.workflow_round_count) for item in summaries] == [
        (2, 1), (0, 0),
    ]
    assert project_service.read_projects(db, []) == []
    assert db.query(models.TaskDefinition).count() == 2

    for path in ("/api/projects", f"/api/projects/{scope.project.id}"):
        rejected = client.get(path, headers=_headers(scope.outsider))
        assert rejected.status_code == 403
        assert "workflow_task_count" not in rejected.json()


def test_new_project_has_explicit_zero_workflow_counts(client):
    response = client.post("/api/projects", json={"name": "New empty project"})
    assert response.status_code == 200, response.text
    assert response.json()["workflow_task_count"] == 0
    assert response.json()["workflow_round_count"] == 0
