"""Incremental rounds retain source-key coverage within their task and collection."""

import pytest
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.auth.security import create_access_token
from al_medlit.workflow import schemas, service


def create_round(client, scope, version, task, mode="unassigned"):
    return client.post(
        "/api/rounds",
        headers={"Authorization": f"Bearer {create_access_token(str(scope.manager.id))}"},
        json={
            "project_id": scope.project.id,
            "name": "Incremental papers",
            "dataset_version_id": version.id,
            "task_version_id": task.id,
            "assistance_policy": "blind",
            "reannotation_mode": mode,
            "annotator_user_ids": [scope.annotator.id],
        },
    )


def version_with_keys(db, scope, dataset_id, keys):
    return service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=scope.project.id,
            dataset_id=dataset_id,
            source_revision="new-papers",
            source_format="jsonl",
            items=[
                schemas.DatasetItemCreate(stable_key=key, payload={"text": key}) for key in keys
            ],
        ),
        scope.manager,
    )


@pytest.mark.parametrize("status", ["draft", "open", "closed", "cancelled"])
def test_incremental_round_only_assigns_new_keys(client, db, workflow_scope, workflow_data, status):
    scope, data = workflow_scope, workflow_data
    original = create_round(client, scope, data.dataset_version, data.task_version, "full_dataset")
    assert original.status_code == 201, original.text
    round_id = original.json()["id"]
    if status in ("open", "closed"):
        service.transition_annotation_round(db, scope.project.id, round_id, "open")
    if status == "closed":
        items = service.list_round_items(db, scope.project.id, round_id)
        decisions = [
            service.create_annotation_decision(
                db,
                schemas.AnnotationDecisionCreate(
                    project_id=scope.project.id, round_item_id=item.id, output={"label": "positive"}
                ),
                scope.annotator,
            )
            for item in items
        ]
        service.create_round_submission(
            db,
            schemas.RoundSubmissionCreate(
                project_id=scope.project.id,
                annotation_round_id=round_id,
                decision_ids=[decision.id for decision in decisions],
            ),
            scope.annotator,
        )
    if status in ("closed", "cancelled"):
        service.transition_annotation_round(db, scope.project.id, round_id, status)
    newer = version_with_keys(
        db, scope, data.dataset.id, ["train-1", "pool-1", "test-1", "added-pmid"]
    )
    response = create_round(client, scope, newer, data.task_version)
    assert response.status_code == 201, response.text
    selected = service.list_round_items(db, scope.project.id, response.json()["id"])
    items = {
        item.id: item.stable_key
        for item in service.list_dataset_items(db, scope.project.id, newer.id)
    }
    assert {items[item.dataset_item_id] for item in selected} == (
        {"train-1", "pool-1", "test-1", "added-pmid"} if status == "cancelled" else {"added-pmid"}
    )
    assert len(service.list_round_items(db, scope.project.id, round_id)) == 3


def test_incremental_coverage_never_crosses_collection_or_task_version(
    client, db, workflow_scope, workflow_data
):
    scope, data = workflow_scope, workflow_data
    original = create_round(client, scope, data.dataset_version, data.task_version, "full_dataset")
    assert original.status_code == 201, original.text
    other_dataset = service.create_dataset(
        db,
        schemas.DatasetCreate(
            project_id=scope.project.id, name="Other collection", source_type="generated"
        ),
        scope.manager,
    )
    other_version = version_with_keys(db, scope, other_dataset.id, ["train-1"])
    other_task = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=scope.project.id,
            task_definition_id=data.task_version.task_definition_id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"type": "string"},
        ),
        scope.manager,
    )
    for version, task in [(other_version, data.task_version), (data.dataset_version, other_task)]:
        response = create_round(client, scope, version, task)
        assert response.status_code == 201, response.text
        assert service.list_round_items(db, scope.project.id, response.json()["id"])


def test_empty_or_fully_covered_source_does_not_leave_a_draft(
    client, db, workflow_scope, workflow_data
):
    scope, data = workflow_scope, workflow_data
    original = create_round(client, scope, data.dataset_version, data.task_version, "full_dataset")
    assert original.status_code == 201, original.text
    empty = version_with_keys(db, scope, data.dataset.id, [])
    before = len(service.list_annotation_rounds(db, scope.project.id))
    for version, mode in [(empty, "full_dataset"), (data.dataset_version, "unassigned")]:
        response = create_round(client, scope, version, data.task_version, mode)
        assert response.status_code == 422, response.text
        assert len(service.list_annotation_rounds(db, scope.project.id)) == before
