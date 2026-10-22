"""Run the browser fixture's policy cases against the real backend routes."""

import json
import runpy
from pathlib import Path

import pytest
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.auth.security import create_access_token
from al_medlit.corpus.models import Document
from al_medlit.project.models import ProjectTask, TaskAssignment
from al_medlit.workflow import schemas, service
from al_medlit.workspace import service as workspace_service
from al_medlit.workspace.models import WorkspaceMember

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "frontend/e2e/fixtures"
CASES = json.loads((FIXTURES / "backendAccessCases.json").read_text())
EXPORTER = runpy.run_path(str(ROOT / "scripts/generate_e2e_contract.py"))


def test_browser_contract_snapshot_matches_backend():
    actual = json.loads(json.dumps(EXPORTER["build_contract"]()))
    assert json.loads((FIXTURES / "backendContract.json").read_text()) == actual


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_browser_access_case_matches_real_route(client, db, workflow_scope, workflow_data, case):
    scope, data = workflow_scope, workflow_data
    annotation_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=scope.project.id,
            name="Fixture contract round",
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            annotator_user_ids=[scope.annotator.id],
            assistance_policy="blind",
            reannotation_mode="full_dataset",
        ),
        scope.manager,
    )
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    scope.workspace.capability_preset = case["preset"]
    scope.workspace.capability_overrides = []
    scope.project.settings = {} if case["modules"] is None else {"modules": case["modules"]}
    actor = scope.manager if case["role"] == "personal" else getattr(scope, case["role"])
    membership = db.query(WorkspaceMember).filter_by(
        workspace_id=scope.workspace.id, user_id=actor.id,
    ).one()
    membership.role = "admin" if case["role"] == "personal" else case["role"]
    other_workspace_id = 0
    if "{other_workspace}" in case["path"]:
        other_workspace = workspace_service.create_team_workspace(
            db, scope.outsider, name="Other workspace",
        )
        workspace_service.add_member(db, other_workspace.id, actor.id, role="annotator")
        other_workspace_id = other_workspace.id
    db.commit()
    path = case["path"].format(
        project=scope.project.id, workspace=scope.workspace.id, round=annotation_round.id,
        other_workspace=other_workspace_id,
    )
    response = client.get(
        "/api" + path,
        headers={"Authorization": f"Bearer {create_access_token(str(actor.id))}"},
    )
    assert response.status_code == case["status"], response.text
    if "expected_body" in case:
        assert response.json() == case["expected_body"]


@pytest.mark.parametrize("modules", [["annotate"], []])
def test_created_project_modules_follow_saved_settings(client, workflow_scope, modules):
    scope = workflow_scope
    headers = {"Authorization": f"Bearer {create_access_token(str(scope.manager.id))}"}
    created = client.post("/api/projects", headers=headers, json={
        "workspace_id": scope.workspace.id, "name": "Configured project",
        "settings": {"modules": modules},
    })
    assert created.status_code == 200, created.text
    response = client.get(f"/api/projects/{created.json()['id']}/modules", headers=headers)
    assert response.status_code == 200, response.text
    expected = ["data", "annotate"] if modules else []
    assert response.json()["selected"] == expected
    assert response.json()["effective"] == expected


def test_unassigned_submission_denies_another_actors_assignment(client, db, workflow_scope):
    scope = workflow_scope
    document = Document(project_id=scope.project.id, external_id="fixture-owner-probe",
                        title="Assigned elsewhere", text="Evidence text.", source="test")
    db.add(document)
    task = ProjectTask(project_id=scope.project.id, annotation_type="document_classification",
                       display_name="Screening", labels=["include"])
    db.add(task)
    db.flush()
    db.add(TaskAssignment(id=51, project_id=scope.project.id, task_id=task.id,
                         document_id=document.id, assignee_user_id=scope.trainer.id,
                         annotator_id=scope.trainer.username, status="in_progress"))
    db.commit()
    response = client.post(
        f"/api/projects/{scope.project.id}/documents/{document.id}/submissions",
        headers={"Authorization": f"Bearer {create_access_token(str(scope.annotator.id))}"},
        json={"assignment_id": 51},
    )
    assert response.status_code == 403, response.text
    assert "not assigned" in response.json()["detail"]
