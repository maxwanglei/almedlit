"""Guideline pilots compare the latest finalized decision for each author and item."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from test_paper_entity_rounds import paper_tasks as paper_tasks
from test_workflow import _headers, _user
from test_workflow import workflow_scope as workflow_scope

from al_medlit.workflow import models, schemas, service
from al_medlit.workspace import service as workspace_service


@pytest.fixture
def pilot(db, workflow_scope, paper_tasks):
    scope = workflow_scope
    second = _user(db, "second-guideline-annotator")
    workspace_service.add_member(db, scope.workspace.id, second.id, role="annotator")
    actors = [scope.annotator, second]
    guideline = service.create_guideline(db, schemas.GuidelineCreate(
        project_id=scope.project.id,
        task_definition_id=paper_tasks.classification.task_definition_id,
        name="Paper relevance guidelines",
    ), scope.manager)
    revision = service.create_guideline_revision(db, schemas.GuidelineRevisionCreate(
        project_id=scope.project.id, guideline_id=guideline.id,
        task_version_id=paper_tasks.classification.id,
        markdown="Classify each paper independently.",
    ), scope.manager)
    service.transition_guideline_revision(
        db, scope.project.id, revision.id, schemas.GuidelineRevisionTransition(status="pilot"),
        scope.manager,
    )
    annotation_round = service.create_annotation_round(db, schemas.AnnotationRoundCreate(
        project_id=scope.project.id, name="Incremental paper pilot",
        dataset_version_id=paper_tasks.version.id, task_version_id=paper_tasks.classification.id,
        guideline_revision_id=revision.id, assistance_policy="blind",
        reannotation_mode="full_dataset", annotator_user_ids=[actor.id for actor in actors],
    ), scope.manager)
    impact = service.create_guideline_impact(db, schemas.GuidelineImpactCreate(
        project_id=scope.project.id, guideline_revision_id=revision.id,
        pilot_round_id=annotation_round.id,
    ), scope.manager)
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    items = service.list_round_items(db, scope.project.id, annotation_round.id)
    assert len(items) == 2
    return SimpleNamespace(
        project=scope.project, manager=scope.manager, actors=actors, revision=revision,
        round=annotation_round, impact=impact, items=items,
    )


def _decision(db, pilot, actor, item, output="relevant", prior=None):
    return service.create_annotation_decision(db, schemas.AnnotationDecisionCreate(
        project_id=pilot.project.id, round_item_id=item.id, output=output,
        supersedes_decision_id=prior.id if prior is not None else None,
    ), actor)


def _submit(db, pilot, actor, decisions):
    return service.create_round_submission(db, schemas.RoundSubmissionCreate(
        project_id=pilot.project.id, annotation_round_id=pilot.round.id,
        decision_ids=[decision.id for decision in decisions],
    ), actor)


def _close(db, pilot):
    service.transition_annotation_round(db, pilot.project.id, pilot.round.id, "closed")


def _complete(client, pilot, evaluation_id=None):
    response = client.post(
        f"/api/workflow-guidelines/impact-evaluations/{evaluation_id or pilot.impact.id}/complete",
        params={"project_id": pilot.project.id}, json={"minimum_agreement": 1.0},
        headers=_headers(pilot.manager),
    )
    assert response.status_code == 200, response.text
    return response.json()


def _activate(client, pilot):
    return client.post(
        f"/api/workflow-guidelines/revisions/{pilot.revision.id}/transition",
        params={"project_id": pilot.project.id}, json={"status": "active"},
        headers=_headers(pilot.manager),
    )


def _incremental_submissions(db, pilot, *, opposite_order=False):
    for index, actor in enumerate(pilot.actors):
        items = list(reversed(pilot.items)) if index and opposite_order else pilot.items
        for item in items:
            _submit(db, pilot, actor, [_decision(db, pilot, actor, item)])


@pytest.mark.parametrize(
    "opposite_order", [False, True], ids=["same-last-item", "different-last-item"],
)
def test_complementary_one_paper_submissions_complete_and_activate_pilot(
    client, db, pilot, opposite_order,
):
    _incremental_submissions(db, pilot, opposite_order=opposite_order)
    _close(db, pilot)
    impact = _complete(client, pilot)
    activated = _activate(client, pilot)
    assert activated.status_code == 200, activated.text
    assert activated.json()["status"] == "active"
    assert impact["passed"] is True
    assert impact["metrics"]["coverage"] == 1.0
    assert impact["metrics"]["compared_item_count"] == 2
    assert impact["metrics"]["annotator_count"] == 2
    assert impact["metrics"]["exact_agreement"] == 1.0


def test_latest_finalized_revision_replaces_only_that_authors_item(client, db, pilot):
    submitted = {}
    for actor in pilot.actors:
        decisions = [_decision(db, pilot, actor, item) for item in pilot.items]
        _submit(db, pilot, actor, decisions)
        submitted[actor.id] = decisions
    for actor in pilot.actors:
        revised = _decision(
            db, pilot, actor, pilot.items[0], "irrelevant", submitted[actor.id][0],
        )
        _submit(db, pilot, actor, [revised])
    _close(db, pilot)
    impact = _complete(client, pilot)
    assert impact["metrics"]["compared_item_count"] == 2
    assert impact["metrics"]["exact_agreement"] == 1.0
    assert impact["passed"] is True


def test_unsubmitted_superseding_draft_keeps_finalized_pilot_evidence(client, db, pilot):
    for actor in pilot.actors:
        decisions = [_decision(db, pilot, actor, item) for item in pilot.items]
        _submit(db, pilot, actor, decisions)
        if actor.id == pilot.actors[0].id:
            _decision(db, pilot, actor, pilot.items[0], "irrelevant", decisions[0])
    _close(db, pilot)
    impact = _complete(client, pilot)
    assert impact["metrics"]["compared_item_count"] == 2
    assert impact["metrics"]["exact_agreement"] == 1.0
    assert impact["passed"] is True


def test_repeated_submissions_do_not_count_one_author_as_two_annotators(client, db, pilot):
    pilot.round.annotator_user_ids = []
    pilot.round.open_to_all_annotators = True
    db.commit()
    actor = pilot.actors[0]
    decisions = [_decision(db, pilot, actor, item) for item in pilot.items]
    _submit(db, pilot, actor, decisions)
    _submit(db, pilot, actor, decisions)
    _close(db, pilot)
    impact = _complete(client, pilot)
    assert impact["metrics"]["coverage"] == 1.0
    assert impact["metrics"]["annotator_count"] == 1
    assert impact["metrics"]["compared_item_count"] == 0
    assert impact["metrics"]["exact_agreement"] == 0.0
    assert impact["passed"] is False
    assert _activate(client, pilot).status_code == 409


def test_disjoint_one_author_submissions_do_not_invent_round_agreement(client, db, pilot):
    pilot.round.annotator_user_ids = []
    pilot.round.open_to_all_annotators = True
    db.commit()
    for actor, item in zip(pilot.actors, pilot.items, strict=True):
        _submit(db, pilot, actor, [_decision(db, pilot, actor, item)])
    _close(db, pilot)
    impact = _complete(client, pilot)
    assert impact["metrics"]["coverage"] == 1.0
    assert impact["metrics"]["annotator_count"] == 2
    assert impact["metrics"]["compared_item_count"] == 0
    assert impact["passed"] is False
    assert _activate(client, pilot).status_code == 409


def test_latest_finalized_disagreement_keeps_pilot_failed(client, db, pilot):
    for actor in pilot.actors:
        for index, item in enumerate(pilot.items):
            output = "irrelevant" if actor.id == pilot.actors[1].id and index == 0 else "relevant"
            _submit(db, pilot, actor, [_decision(db, pilot, actor, item, output)])
    _close(db, pilot)
    impact = _complete(client, pilot)
    assert impact["metrics"]["coverage"] == 1.0
    assert impact["metrics"]["compared_item_count"] == 2
    assert impact["metrics"]["exact_agreement"] == 0.5
    assert impact["passed"] is False
    assert _activate(client, pilot).status_code == 409


def test_new_evaluation_recovers_historical_failed_attempt_without_rewriting_it(client, db, pilot):
    _incremental_submissions(db, pilot)
    _close(db, pilot)
    historical_metrics = {"coverage": 0.5, "compared_item_count": 1, "exact_agreement": 1.0}
    pilot.impact.status = "completed"
    pilot.impact.passed = False
    pilot.impact.metrics = historical_metrics
    pilot.impact.completed_at = datetime.now(UTC)
    db.commit()
    repeated = client.post(
        f"/api/workflow-guidelines/impact-evaluations/{pilot.impact.id}/complete",
        params={"project_id": pilot.project.id}, json={"minimum_agreement": 1.0},
        headers=_headers(pilot.manager),
    )
    assert repeated.status_code == 409, repeated.text
    fresh = client.post("/api/workflow-guidelines/impact-evaluations", json={
        "project_id": pilot.project.id, "guideline_revision_id": pilot.revision.id,
        "pilot_round_id": pilot.round.id,
    }, headers=_headers(pilot.manager))
    assert fresh.status_code == 201, fresh.text
    assert fresh.json()["id"] != pilot.impact.id
    completed = _complete(client, pilot, fresh.json()["id"])
    assert completed["passed"] is True
    assert _activate(client, pilot).status_code == 200
    db.refresh(pilot.impact)
    assert pilot.impact.passed is False
    assert pilot.impact.metrics == historical_metrics
    assert db.query(models.GuidelineImpactEvaluation).count() == 2
