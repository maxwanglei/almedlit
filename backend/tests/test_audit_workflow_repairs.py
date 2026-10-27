import threading
from types import SimpleNamespace

import pytest
from sqlalchemy.exc import IntegrityError
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_document_assigned
from al_medlit.co_learning.error_guideline_learning import service as learning_service
from al_medlit.co_learning.error_guideline_learning.models import (
    ErrorPattern,
    MicroQuestionTemplate,
)
from al_medlit.core.exceptions import ForbiddenError, ValidationError
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import Document
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.iaa.alignment import NONE_LABEL, build_rating_matrix
from al_medlit.importers.pubmed import _NCBIRequestGate
from al_medlit.project import service as project_service
from al_medlit.project.models import Project, ProjectTask, TaskAssignment
from al_medlit.project.schemas import TaskAssignmentUpdate
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.services import selection
from al_medlit.workspace import service as workspace_service


def _round(db, scope, data, user_ids=None):
    return service.create_annotation_round(
        db,
        schemas.AnnotationRoundCreate(
            project_id=scope.project.id,
            name="Audit round",
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=user_ids or [scope.annotator.id],
        ),
        scope.manager,
    )


def _task_assignment(db, scope, *, status="assigned"):
    document = corpus_service.create_document(
        db, DocumentCreate(project_id=scope.project.id, text="Alpha. Beta.")
    )
    task = ProjectTask(project_id=scope.project.id, annotation_type="entity", display_name="NER")
    db.add(task)
    db.flush()
    assignment = TaskAssignment(
        project_id=scope.project.id, document_id=document.id, task_id=task.id,
        assignee_user_id=scope.annotator.id, annotator_id=scope.annotator.username,
        status=status, assignment_scope_key="audit",
    )
    db.add(assignment)
    db.commit()
    return assignment


def test_withdrawn_assignment_does_not_grant_document_access(db, workflow_scope):
    assignment = _task_assignment(db, workflow_scope, status="withdrawn")
    with pytest.raises(ForbiddenError):
        assert_document_assigned(
            db, workflow_scope.annotator,
            workspace_service.get_member(
                db, workflow_scope.workspace.id, workflow_scope.annotator.id
            ),
            project_id=workflow_scope.project.id, document_id=assignment.document_id,
        )


def test_manager_cannot_finalize_empty_assignment(db, workflow_scope):
    assignment = _task_assignment(db, workflow_scope)
    with pytest.raises(ValidationError, match="submission workflow"):
        project_service.update_task_assignment(
            db, workflow_scope.project.id, assignment.id, TaskAssignmentUpdate(status="submitted")
        )
    assert assignment.status == "assigned"


def test_removed_round_annotator_can_be_replaced(db, workflow_scope, workflow_data):
    scope = workflow_scope
    annotation_round = _round(db, scope, workflow_data)
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    workspace_service.remove_member(
        db, scope.workspace.id, scope.annotator.id, actor_user_id=scope.manager.id
    )
    db.commit()
    assert annotation_round.annotator_user_ids == []
    updated = service.update_annotation_round_assignees(
        db, scope.project.id, annotation_round.id,
        schemas.AnnotationRoundAssigneesUpdate(annotator_user_ids=[scope.trainer.id]),
    )
    assert updated.annotator_user_ids == [scope.trainer.id]
    assert updated.status == "open"


def test_round_reassignment_rejects_nonmembers(db, workflow_scope, workflow_data):
    annotation_round = _round(db, workflow_scope, workflow_data)
    with pytest.raises(ValidationError, match="workspace members"):
        service.update_annotation_round_assignees(
            db, workflow_scope.project.id, annotation_round.id,
            schemas.AnnotationRoundAssigneesUpdate(
                annotator_user_ids=[workflow_scope.outsider.id]
            ),
        )


def test_iaa_round_requires_each_assignee_to_submit(db, workflow_scope, workflow_data):
    scope = workflow_scope
    annotation_round = _round(db, scope, workflow_data, [scope.annotator.id, scope.trainer.id])
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    decisions = [service.create_annotation_decision(
        db,
        schemas.AnnotationDecisionCreate(
            project_id=scope.project.id, round_item_id=item.id, output={"label": "positive"}
        ),
        scope.annotator,
    ) for item in service.list_round_items(db, scope.project.id, annotation_round.id)]
    service.create_round_submission(
        db,
        schemas.RoundSubmissionCreate(
            project_id=scope.project.id, annotation_round_id=annotation_round.id,
            decision_ids=[decision.id for decision in decisions],
        ),
        scope.annotator,
    )
    with pytest.raises(ValidationError, match="Every assigned annotator"):
        service.transition_annotation_round(db, scope.project.id, annotation_round.id, "closed")


def test_selection_respects_stable_keys_across_dataset_versions(db, workflow_scope, workflow_data):
    scope, data = workflow_scope, workflow_data
    annotation_round = _round(db, scope, data)
    service.transition_annotation_round(db, scope.project.id, annotation_round.id, "open")
    version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=scope.project.id, dataset_id=data.dataset.id,
            source_revision="next", source_format="jsonl",
            items=[schemas.DatasetItemCreate(stable_key="pool-1", payload={"text": "Unclear"})],
        ),
        scope.manager,
    )
    run = models.SelectionRun(
        project_id=scope.project.id, dataset_version_id=version.id,
        task_version_id=data.task_version.id, strategy="all",
    )
    current_item = db.query(models.DatasetItem).filter_by(dataset_version_id=version.id).one()
    assert selection._unavailable_selection_item_ids(db, run) == {current_item.id}


def test_document_external_ids_are_project_unique(db, workflow_scope):
    data = DocumentCreate(project_id=workflow_scope.project.id, external_id="PMID:1", text="First")
    first = corpus_service.create_document(db, data)
    from al_medlit.core.exceptions import ConflictError

    with pytest.raises(ConflictError, match="external ID"):
        corpus_service.create_document(db, data.model_copy(update={"text": "Duplicate"}))
    assert db.get(Document, first.id).text == "First"


def test_only_one_default_storage_policy_per_project(db, workflow_scope):
    for name in ("first", "second"):
        db.add(models.StoragePolicy(
            project_id=workflow_scope.project.id, name=name,
            artifact_prefix="workspaces/1/blobs", is_default=True,
        ))
    with pytest.raises(IntegrityError):
        db.flush()


def test_zero_span_submitted_annotator_is_counted():
    annotation = SimpleNamespace(
        annotator_id="alice", document_id=1, start_offset=0, end_offset=4, label="Drug"
    )
    annotators, items, spans = build_rating_matrix(
        [annotation], submitted_annotators={"alice", "bob"}
    )
    assert annotators == ["alice", "bob"]
    assert items == [{"alice": "Drug", "bob": NONE_LABEL}]
    assert spans["bob"] == set()


def test_guideline_draft_does_not_claim_approval(db, workflow_scope):
    pattern = ErrorPattern(
        project_id=workflow_scope.project.id, task_type="entity", error_type="boundary_error",
        description="Boundary", example_ids=[], example_count=1,
    )
    db.add(pattern)
    db.commit()
    atom = learning_service.draft_guideline_atom(db, pattern.id, manager_id="manager")
    assert atom.approved_by is None
    learning_service.approve_guideline_atom(db, atom.id, approved_by="manager")
    question = db.query(MicroQuestionTemplate).filter_by(guideline_atom_id=atom.id).one()
    assert question.status == "draft"
    assert question.answer_options == []


def test_ncbi_gate_releases_mutex_during_download(monkeypatch):
    gate = _NCBIRequestGate()
    first_started = threading.Event()
    second_started = threading.Event()
    allow_first_to_finish = threading.Event()
    monkeypatch.setattr("al_medlit.importers.pubmed.NCBI_REQUEST_INTERVAL", 0)

    def first():
        with gate.request():
            first_started.set()
            assert allow_first_to_finish.wait(3)

    def second():
        assert first_started.wait(3)
        with gate.request():
            second_started.set()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    try:
        assert second_started.wait(2)
    finally:
        allow_first_to_finish.set()
        for thread in threads:
            thread.join(3)


def test_personal_self_assignments_belong_to_owner_for_superuser_mutations(db, workflow_scope):
    owner = workflow_scope.manager
    workspace = workspace_service.create_personal_workspace(db, owner)
    project = Project(name="Personal audit", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    task = ProjectTask(project_id=project.id, annotation_type="entity", display_name="Entities")
    db.add(task)
    db.flush()
    corpus_service.create_document(db, DocumentCreate(project_id=project.id, text="Paper."))
    superuser = User(username="audit-root", password_hash="test-only", is_superuser=True)
    db.add(superuser)
    db.commit()
    assignments = project_service.ensure_personal_project_self_assignments(
        db, project.id, superuser, assigned_by_user=superuser
    )
    assert len(assignments) == 1
    assert assignments[0].assignee_user_id == owner.id


def test_accepting_existing_member_upgrade_invite_updates_role(db, workflow_scope):
    scope = workflow_scope
    invite = workspace_service.create_invite(
        db, scope.workspace.id, created_by=scope.manager.id, role="manager", expires_minutes=60
    )
    member = workspace_service.accept_invite(db, invite, scope.annotator)
    db.commit()
    assert member.role == "manager"
    assert invite.accepted_by == scope.annotator.id


def test_prepared_generated_snapshot_can_complete_its_source_cycle(client, db):
    from test_training_preparation import _context, _prepare, _records, _request, _source

    project, task = _context(client)
    source = _source(client, project, "Cycle source", _records("cycle", 20))
    prepared = _prepare(client, project, _request(task, source))
    version_id = prepared["training_dataset_version"]["id"]
    cycle = models.LearningCycle(
        project_id=project["id"], name="Complete prepared snapshot", sequence=1,
        task_version_id=task["id"], source_dataset_version_id=source["id"],
        status="active", current_stage="evaluate", goal="expand_pool",
    )
    db.add(cycle)
    db.commit()
    completed = service.transition_learning_cycle(
        db, project["id"], cycle.id,
        schemas.CycleTransition(
            status="completed", current_stage="complete",
            output_training_dataset_version_id=version_id,
        ),
    )
    assert completed.status == "completed"


def test_deactivation_cancels_training_and_releases_reservation(db, monkeypatch):
    from test_training_run_execution import MemoryObjectStorage, _training_run_fixture

    from al_medlit.administration import service as administration_service
    from al_medlit.model_artifacts.models import ArtifactStorageReservation

    monkeypatch.setenv("AL_MEDLIT_WORKER_IMAGE_DIGEST", "a" * 64)
    scope = _training_run_fixture(db, storage=MemoryObjectStorage(), suffix="deactivate-owner")
    administrator = User(
        username="deactivation-root", password_hash="test-only", is_superuser=True
    )
    db.add(administrator)
    db.commit()
    administration_service.set_user_status(
        db, actor_user_id=administrator.id, user_id=scope.actor.id, is_active=False
    )
    db.commit()
    db.refresh(scope.run)
    assert scope.run.status == "cancelled"
    assert scope.run.failure_code == "owner_deactivated"
    reservation = db.get(ArtifactStorageReservation, scope.run.artifact_reservation_id)
    assert reservation.status == "released"


def test_progress_counts_finalized_adjudication_decisions(db, workflow_scope, workflow_data):
    from al_medlit.workflow.services.source_progress import annotation_progress

    scope = workflow_scope
    annotation_round = _round(db, scope, workflow_data)
    item = service.list_round_items(db, scope.project.id, annotation_round.id)[0]
    decision = models.RoundAnnotationDecision(
        project_id=scope.project.id, round_item_id=item.id,
        annotator_user_id=scope.annotator.id, decision_kind="adjudication",
        output={"label": "positive"}, content_hash="a" * 64,
    )
    db.add(decision)
    db.flush()
    db.add(models.RoundSubmission(
        project_id=scope.project.id, annotation_round_id=annotation_round.id,
        annotator_user_id=scope.annotator.id, sequence=1,
        decision_ids=[decision.id], content_hash="b" * 64,
    ))
    annotation_round.status = "open"
    db.commit()
    rows = annotation_progress(db, scope.project.id, workflow_data.task_version.id)
    assert next(row for row in rows if (
        row["dataset_version_id"] == workflow_data.dataset_version.id
    ))["submitted"] == 1
