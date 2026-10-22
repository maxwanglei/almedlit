from types import SimpleNamespace

import pytest
from sqlalchemy import event
from test_workflow import _headers
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.core.exceptions import ValidationError
from al_medlit.workflow import models, schemas, service


@pytest.fixture
def assisted_item(db, workflow_scope, workflow_data):
    scope = workflow_scope
    project_id = scope.project.id
    feedback_run = service.create_feedback_run(db, schemas.FeedbackRunCreate(
        project_id=project_id,
        dataset_version_id=workflow_data.dataset_version.id,
        task_version_id=workflow_data.task_version.id,
        producer_type="external_llm", provider="approved-local",
        external_model_id="reviewer", exact_revision="r1",
        prompt_template_hash="a" * 64, configuration={"temperature": 0},
        data_egress_policy={"mode": "local_only"},
    ), scope.manager)
    feedback_set = service.create_feedback_set(db, schemas.FeedbackSetCreate(
        project_id=project_id, feedback_run_id=feedback_run.id,
        candidates=[schemas.FeedbackCandidateCreate(
            dataset_item_id=workflow_data.items["pool-1"].id,
            output={"label": "positive"}, score=0.9,
        )],
    ), scope.trainer)
    candidate = service.list_feedback_candidates(db, project_id, feedback_set.id)[0]
    annotation_round = models.AnnotationRound(
        project_id=project_id, name="Atomic assisted save", sequence=1,
        dataset_version_id=workflow_data.dataset_version.id,
        task_version_id=workflow_data.task_version.id,
        feedback_set_version_id=feedback_set.id, assistance_policy="immediate_suggestions",
        reannotation_mode="targeted_subset", annotator_user_ids=[scope.annotator.id],
        status="open", created_by_user_id=scope.manager.id,
    )
    db.add(annotation_round)
    db.flush()
    item = models.RoundItem(project_id=project_id, annotation_round_id=annotation_round.id,
                            dataset_item_id=workflow_data.items["pool-1"].id)
    db.add(item)
    db.commit()
    exposure = service.create_feedback_exposure(db, schemas.FeedbackExposureCreate(
        project_id=project_id, round_item_id=item.id, feedback_candidate_id=candidate.id,
        exposure_mode="immediate_suggestions",
    ), scope.annotator)
    return SimpleNamespace(item=item, exposure=exposure, scope=scope)


def _payload(context):
    return {"project_id": context.scope.project.id, "round_item_id": context.item.id,
            "output": {"label": "positive"},
            "feedback_decision": {"exposure_id": context.exposure.id, "decision": "accepted"}}


def test_assisted_save_returns_only_after_both_records_are_persisted(client, db, assisted_item):
    context = assisted_item
    saved = client.post("/api/rounds/decisions", json=_payload(context),
                        headers=_headers(context.scope.annotator))
    assert saved.status_code == 201, saved.text
    feedback = db.query(models.FeedbackDecision).one()
    assert feedback.round_annotation_decision_id == saved.json()["id"]
    assert feedback.feedback_exposure_id == context.exposure.id
    assert feedback.decision == "accepted"
    assert db.query(models.RoundAnnotationDecision).count() == 1


def test_feedback_write_failure_rolls_back_annotation_before_retry(client, db, assisted_item):
    context = assisted_item

    def fail_feedback(_mapper, _connection, _target):
        raise ValidationError("Feedback storage unavailable")

    event.listen(models.FeedbackDecision, "before_insert", fail_feedback)
    try:
        failed = client.post("/api/rounds/decisions", json=_payload(context),
                             headers=_headers(context.scope.annotator))
    finally:
        event.remove(models.FeedbackDecision, "before_insert", fail_feedback)
    assert failed.status_code == 422, failed.text
    assert "Feedback storage unavailable" in failed.text
    assert db.query(models.RoundAnnotationDecision).count() == 0
    assert db.query(models.FeedbackDecision).count() == 0
    retried = client.post("/api/rounds/decisions", json=_payload(context),
                          headers=_headers(context.scope.annotator))
    assert retried.status_code == 201, retried.text
    assert db.query(models.RoundAnnotationDecision).count() == 1
    assert db.query(models.FeedbackDecision).count() == 1


def test_invalid_feedback_exposure_cannot_leave_an_annotation(client, db, assisted_item):
    payload = _payload(assisted_item)
    payload["feedback_decision"]["exposure_id"] = 999999
    failed = client.post("/api/rounds/decisions", json=payload,
                         headers=_headers(assisted_item.scope.annotator))
    assert failed.status_code == 404, failed.text
    assert db.query(models.RoundAnnotationDecision).count() == 0
    assert db.query(models.FeedbackDecision).count() == 0
