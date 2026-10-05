"""Source progress counts submitted examples, not draft decisions or label copies."""

from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.auth.security import create_access_token
from al_medlit.workflow import schemas, service
from al_medlit.workflow.services.source_progress import annotation_progress


def _round(db, scope, data, name, actor):
    result = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=scope.project.id,
            name=name,
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[actor.id],
        ),
        scope.manager,
    )
    service.transition_annotation_round(db, scope.project.id, result.id, "open")
    return result, service.list_round_items(db, scope.project.id, result.id)


def _decision(db, scope, item, actor):
    return service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=scope.project.id,
            round_item_id=item.id,
            output={"label": "positive"},
        ),
        actor,
    )


def _submit(db, scope, annotation_round, decision, actor):
    return service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=scope.project.id,
            annotation_round_id=annotation_round.id,
            decision_ids=[decision.id],
        ),
        actor,
    )


def test_progress_counts_distinct_submitted_items_across_rounds_and_annotators(
    client,
    db,
    workflow_scope,
    workflow_data,
):
    scope, data = workflow_scope, workflow_data
    assert annotation_progress(db, scope.project.id) == [
        {
            "dataset_id": data.dataset.id,
            "dataset_version_id": data.dataset_version.id,
            "total": 3,
            "submitted": 0,
        }
    ]
    first_round, first_items = _round(db, scope, data, "Initial annotation", scope.annotator)
    submitted = _decision(db, scope, first_items[0], scope.annotator)
    pending = _decision(db, scope, first_items[1], scope.annotator)
    _submit(db, scope, first_round, submitted, scope.annotator)
    second_round, second_items = _round(db, scope, data, "Another reviewer", scope.manager)
    repeated = _decision(db, scope, second_items[0], scope.manager)
    _submit(db, scope, second_round, repeated, scope.manager)
    assert annotation_progress(db, scope.project.id)[0]["submitted"] == 1
    _submit(db, scope, first_round, pending, scope.annotator)
    response = client.get(
        f"/api/projects/{scope.project.id}/datasets/annotation-progress",
        headers={"Authorization": f"Bearer {create_access_token(str(scope.trainer.id))}"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == [
        {
            "dataset_id": data.dataset.id,
            "dataset_version_id": data.dataset_version.id,
            "total": 3,
            "submitted": 2,
        }
    ]
    assert first_round.status == "open"
    assert annotation_progress(db, scope.project.id + 999) == []


def test_progress_endpoint_requires_project_trainer(client, workflow_scope):
    scope = workflow_scope
    for actor in (scope.annotator, scope.outsider):
        response = client.get(
            f"/api/projects/{scope.project.id}/datasets/annotation-progress",
            headers={"Authorization": f"Bearer {create_access_token(str(actor.id))}"},
        )
        assert response.status_code == 403
