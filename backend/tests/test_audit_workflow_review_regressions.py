import pytest
from test_iaa import _add_entity_annotation, _make_project_and_document
from test_standalone_predictions import complete
from test_standalone_predictions import scope as scope
from test_workflow import workflow_scope as workflow_scope

from al_medlit.co_learning.error_guideline_learning import service as learning_service
from al_medlit.co_learning.error_guideline_learning.models import ErrorPattern
from al_medlit.core.exceptions import ValidationError
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.iaa import service as iaa_service
from al_medlit.submission.models import AnnotationSubmission
from al_medlit.workflow import models, prediction_schemas, schemas, service
from al_medlit.workflow.services import predictions, selection


def test_prediction_review_respects_cross_version_reservation(db, scope):
    run = complete(db, scope)
    row = next(row for row in predictions.result_rows(db, run) if not row["protected"])
    item = db.get(models.DatasetItem, row["dataset_item_id"])
    dataset = db.get(models.DatasetVersion, run.dataset_version_id)
    duplicate = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=run.project_id,
            dataset_id=dataset.dataset_id,
            source_revision="duplicate",
            source_format="jsonl",
            items=[schemas.DatasetItemCreate(stable_key=item.stable_key, payload=item.payload)],
        ),
        scope.actor,
    )
    reserved_round = service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=run.project_id,
            name="Already reserved",
            dataset_version_id=duplicate.id,
            task_version_id=run.task_version_id,
            annotator_user_ids=[scope.actor.id],
            assistance_policy="blind",
            reannotation_mode="full_dataset",
        ),
        scope.actor,
    )
    service.transition_annotation_round(db, run.project_id, reserved_round.id, "open")
    selection_run = models.SelectionRun(
        project_id=run.project_id,
        dataset_version_id=run.dataset_version_id,
        task_version_id=run.task_version_id,
        strategy="all",
    )
    assert item.id in selection._unavailable_selection_item_ids(db, selection_run)
    with pytest.raises(ValidationError, match="No eligible"):
        predictions.create_review_round(
            db,
            run.project_id,
            run.id,
            prediction_schemas.PredictionReviewCreate(
                name="Duplicate review", request_key="duplicate-review", dataset_item_ids=[item.id]
            ),
            scope.actor,
        )


def test_unassigned_historical_submission_does_not_enter_new_iaa_scope(db):
    project, document = _make_project_and_document(db, "independent-iaa")
    guideline = GuidelineVersion(project_id=project.id, version_label="new")
    db.add(guideline)
    db.flush()
    _add_entity_annotation(
        db, project.id, document.id, "alice", 0, 5, "Drug", guideline_version_id=guideline.id
    )
    db.add(
        AnnotationSubmission(
            project_id=project.id,
            document_id=document.id,
            annotator_id="bob",
            kind="submission",
            assignment_id=None,
            storage_key="historical",
            file_name="historical.json",
            size_bytes=1,
            checksum_sha256="a" * 64,
            annotation_count=0,
        )
    )
    db.commit()
    report = iaa_service.compute_iaa(
        db,
        project_id=project.id,
        annotation_type="entity",
        document_id=document.id,
        guideline_version_id=guideline.id,
    )
    assert report.annotator_ids == ["alice"], report.model_dump()


def test_accepted_atom_keeps_original_approver(db, workflow_scope):
    pattern = ErrorPattern(
        project_id=workflow_scope.project.id,
        task_type="entity",
        error_type="boundary_error",
        description="Boundary",
        example_ids=[],
        example_count=1,
    )
    db.add(pattern)
    db.commit()
    atom = learning_service.draft_guideline_atom(db, pattern.id)
    learning_service.approve_guideline_atom(db, atom.id, "first-manager")
    learning_service.approve_guideline_atom(db, atom.id, "second-manager")
    assert atom.approved_by == "first-manager"
