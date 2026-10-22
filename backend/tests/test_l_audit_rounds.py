"""Regression coverage for round access, immutable publication, and assignment totals."""

import importlib.util
import json
from pathlib import Path

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import IntegrityError
from test_evidence_blocks import _setup_evidence_project
from test_inference_candidate_queries import _inference_scope
from test_round_adjudication_permissions import _decision, _round
from test_round_adjudication_permissions import paper_tasks as paper_tasks
from test_round_adjudication_permissions import scope as scope
from test_round_adjudication_permissions import workflow_scope as workflow_scope
from test_submission_flow import _setup_project_with_annotation
from test_workflow import _headers

from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_task_assigned
from al_medlit.evidence.models import EvidenceReviewCoverage
from al_medlit.iaa.service import compute_iaa
from al_medlit.project import service as project_service
from al_medlit.project.models import Project, TaskAssignment
from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workspace.models import Workspace, WorkspaceMember


def test_l2_empty_modules_remain_manageable(client, scope):
    url = f"/api/projects/{scope.project.id}/modules"
    disabled = client.patch(url, json={"selected": []}, headers=_headers(scope.manager))
    assert disabled.status_code == 200, disabled.text
    read = client.get(url, headers=_headers(scope.manager))
    assert read.status_code == 200, read.text
    restored = client.patch(url, json={"selected": ["annotate"]}, headers=_headers(scope.manager))
    assert restored.status_code == 200, restored.text
    denied = client.patch(url, json={"selected": []}, headers=_headers(scope.annotators[0]))
    assert denied.status_code == 403, denied.text


def test_l3_annotator_cannot_forge_governance_event(client, scope):
    annotation_round, _ = _round(client, scope)
    payload = {
        "project_id": scope.project.id,
        "annotation_round_id": annotation_round["id"],
        "event_type": "adjudication",
        "severity": "critical",
        "payload": {},
    }
    rejected = client.post(
        "/api/feedback-runs/events", json=payload, headers=_headers(scope.annotators[0])
    )
    assert rejected.status_code == 403, rejected.text
    created = client.post(
        "/api/feedback-runs/events", json=payload, headers=_headers(scope.manager)
    )
    assert created.status_code == 201, created.text


def test_l5_work_items_expose_only_task_inputs(client, db, scope):
    annotation_round, items = _round(client, scope)
    item = db.get(models.DatasetItem, items[0]["dataset_item_id"])
    db.query(models.DatasetItem).filter_by(id=item.id).update(
        {
            "payload": {**item.payload, "label": "relevant", "secret_answer": "gold"},
        },
        synchronize_session=False,
    )
    db.commit()
    response = client.get(
        f"/api/rounds/{annotation_round['id']}/work-items",
        params={"project_id": scope.project.id},
        headers=_headers(scope.annotators[0]),
    )
    assert response.status_code == 200, response.text
    payload = response.json()[0]["dataset_item"]["payload"]
    assert payload["text"]
    assert "label" not in payload
    assert "secret_answer" not in payload
    assert response.json()[0]["dataset_item"]["content_hash"] == _canonical_hash(payload)


def test_l6_historical_multi_decision_submission_remains_publishable(client, db, scope):
    annotation_round, items = _round(client, scope)
    initial = _decision(client, scope, scope.annotators[0], items[0], output="irrelevant").json()
    revised = _decision(client, scope, scope.annotators[0], items[0], output="relevant").json()
    legacy = models.RoundSubmission(
        project_id=scope.project.id,
        annotation_round_id=annotation_round["id"],
        annotator_user_id=scope.annotators[0].id,
        sequence=1,
        decision_ids=[revised["id"], initial["id"]],
        content_hash=_canonical_hash([]),
    )
    db.add(legacy)
    db.commit()
    resolved = service.resolve_round_labels(
        db,
        annotation_round["id"],
        schemas.RoundLabelSetCreate(
            project_id=scope.project.id,
            name="Historical snapshot",
            source_kind="human",
            submission_ids=[legacy.id],
            publication_mode="submitted_snapshot",
        ),
    )
    assert list(resolved["labels"].values()) == ["relevant"]
    assert resolved["source_decision_ids"] == [revised["id"]]


@pytest.mark.parametrize("suffix", ["", "/work-context"])
def test_l7_foreign_and_absent_rounds_are_indistinguishable(client, scope, workflow_scope, suffix):
    annotation_round, _ = _round(client, scope)
    responses = [
        client.get(f"/api/rounds/{round_id}{suffix}", headers=_headers(workflow_scope.outsider))
        for round_id in [annotation_round["id"], 999999]
    ]
    assert [response.status_code for response in responses] == [404, 404]
    assert responses[0].json() == responses[1].json()


def test_l13_manager_reexports_requested_annotator(client, object_storage):
    ctx = _setup_project_with_annotation(client)
    response = client.post(
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}/submissions",
        json={"kind": "re_export", "assignment_id": ctx["assignment"]["id"]},
    )
    assert response.status_code == 200, response.text
    snapshot = json.loads(object_storage.get_bytes(response.json()["storage_key"]))
    assert snapshot["annotator_id"] == "alice"
    assert [annotation["annotator_id"] for annotation in snapshot["annotations"]] == ["alice"]


def test_l16_mutation_prefers_reopened_older_assignment(client, db):
    ctx = _setup_project_with_annotation(client)
    original = db.get(TaskAssignment, ctx["assignment"]["id"])
    newer = TaskAssignment(
        project_id=original.project_id,
        task_id=original.task_id,
        document_id=original.document_id,
        assignee_user_id=original.assignee_user_id,
        annotator_id=original.annotator_id,
        assignment_scope_key="newer-guideline",
        status="submitted",
        metadata_={},
    )
    db.add(newer)
    db.commit()
    user = db.get(User, original.assignee_user_id)
    member = (
        db.query(WorkspaceMember)
        .filter_by(workspace_id=original.project.workspace_id, user_id=user.id)
        .one()
    )
    assignment = assert_task_assigned(
        db,
        user,
        member,
        project_id=original.project_id,
        document_id=original.document_id,
        annotation_type="entity",
        require_mutable_assignment=True,
    )
    assert assignment.id == original.id


def test_l20_progress_excludes_withdrawn_assignments(client, db):
    ctx = _setup_project_with_annotation(client)
    assignment = db.get(TaskAssignment, ctx["assignment"]["id"])
    assignment.status = "withdrawn"
    db.commit()
    progress = project_service.get_project_progress(db, assignment.project_id)
    assert progress.total == 0
    assert progress.by_status == {}
    assert progress.by_document == []


def test_l20_evidence_item_count_counts_unique_sentences(client, db):
    ctx = _setup_evidence_project(client, "three-reviewers")
    for username in ("alice", "bob", "carol"):
        client._token_for(username)
        user = db.query(User).filter_by(username=username).one()
        db.add(
            EvidenceReviewCoverage(
                project_id=ctx["project"]["id"],
                document_id=ctx["document"]["id"],
                reviewer_user_id=user.id,
                target_version_id=ctx["version"]["id"],
                structure_version_id=ctx["structure"]["structure_version"]["id"],
                guideline_version_id=ctx["guideline"]["id"],
                start_sentence_id=ctx["structure"]["sentences"][0]["id"],
                end_sentence_id=ctx["structure"]["sentences"][2]["id"],
                start_sentence_ordinal=0,
                end_sentence_ordinal=2,
            )
        )
    db.commit()
    report = compute_iaa(
        db,
        project_id=ctx["project"]["id"],
        annotation_type="evidence_block",
        document_id=ctx["document"]["id"],
        target_version_id=ctx["version"]["id"],
    )
    assert len(report.evidence_metrics.pairs) == 3
    assert report.item_count == 3


def test_l4_null_cycle_sequence_is_unique(client, db, scope):
    annotation_round, _ = _round(client, scope)
    db.add(
        models.AnnotationRound(
            project_id=scope.project.id,
            sequence=annotation_round["sequence"],
            name="Concurrent duplicate",
            dataset_version_id=scope.source.id,
            task_version_id=scope.task.id,
            assistance_policy="blind",
            reannotation_mode="full_dataset",
            annotator_user_ids=[scope.annotators[0].id],
            status="draft",
        )
    )
    with pytest.raises(IntegrityError):
        db.flush()


def test_l1_round_pages_do_not_omit_or_duplicate_items(client, scope):
    annotation_round, items = _round(client, scope)
    url = f"/api/rounds/{annotation_round['id']}/work-items"
    pages = [
        client.get(
            url,
            params={"project_id": scope.project.id, "limit": 1, "offset": offset},
            headers=_headers(scope.annotators[0]),
        )
        for offset in range(3)
    ]
    assert [page.status_code for page in pages] == [200, 200, 200]
    assert [page.json()[0]["round_item"]["id"] for page in pages[:2]] == [
        item["id"] for item in items
    ]
    assert pages[2].json() == []
    rejected = client.get(
        url,
        params={"project_id": scope.project.id, "limit": 501},
        headers=_headers(scope.annotators[0]),
    )
    assert rejected.status_code == 422


def test_l1_submission_and_publication_queries_are_batched(client, db, scope):
    annotation_round, _ = _round(client, scope)
    decision_ids = []
    for index in range(80):
        item = models.DatasetItem(
            project_id=scope.project.id,
            dataset_version_id=scope.source.id,
            stable_key=f"batch-{index}",
            payload={"text": str(index)},
            content_hash=_canonical_hash(index),
        )
        db.add(item)
        db.flush()
        round_item = models.RoundItem(
            project_id=scope.project.id,
            annotation_round_id=annotation_round["id"],
            dataset_item_id=item.id,
        )
        db.add(round_item)
        db.flush()
        decision = models.RoundAnnotationDecision(
            project_id=scope.project.id,
            round_item_id=round_item.id,
            annotator_user_id=scope.annotators[0].id,
            output="relevant",
            decision_kind="annotation",
            content_hash=_canonical_hash(index),
        )
        db.add(decision)
        db.flush()
        decision_ids.append(decision.id)
    db.commit()
    project_id, actor_id = scope.project.id, scope.annotators[0].id
    actor = db.get(User, actor_id)
    statements = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.bind, "before_cursor_execute", record_statement)
    try:
        submission = service.create_round_submission(
            db,
            schemas.RoundSubmissionCreate(
                project_id=project_id,
                annotation_round_id=annotation_round["id"],
                decision_ids=decision_ids,
            ),
            actor,
        )
        submission_queries = len(statements)
        statements.clear()
        resolved = service.resolve_round_labels(
            db,
            annotation_round["id"],
            schemas.RoundLabelSetCreate(
                project_id=project_id,
                name="Batched labels",
                source_kind="human",
                submission_ids=[submission.id],
                publication_mode="submitted_snapshot",
            ),
        )
        publication_queries = len(statements)
    finally:
        event.remove(db.bind, "before_cursor_execute", record_statement)
    assert len(resolved["labels"]) == 80
    assert submission_queries <= 14, submission_queries
    assert publication_queries <= 8, publication_queries


def test_l13_annotators_cannot_choose_another_export_owner(client, db):
    ctx = _setup_project_with_annotation(client)
    user = db.query(User).filter_by(username="alice").one()
    member = db.query(WorkspaceMember).filter_by(user_id=user.id).one()
    member.role = "annotator"
    db.commit()
    rejected = client.post(
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}/submissions",
        json={"kind": "re_export", "annotator_id": "manager"},
        headers=_headers(user),
    )
    assert rejected.status_code == 403, rejected.text


def test_l13_manager_reexports_offboarded_annotator_history(client, db, object_storage):
    ctx = _setup_project_with_annotation(client)
    owner = db.query(User).filter_by(username="alice").one()
    member = db.query(WorkspaceMember).filter_by(user_id=owner.id).one()
    db.delete(member)
    db.commit()
    response = client.post(
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}/submissions",
        json={"kind": "re_export", "assignment_id": ctx["assignment"]["id"]},
    )
    assert response.status_code == 200, response.text
    snapshot = json.loads(object_storage.get_bytes(response.json()["storage_key"]))
    assert [annotation["annotator_id"] for annotation in snapshot["annotations"]] == ["alice"]


def test_l4_migration_preserves_round_ids_and_repairs_only_duplicate_sequences():
    path = (
        Path(__file__).resolve().parents[1]
        / "migrations/versions/c9e4a7b12d68_unique_round_sequences_without_cycle.py"
    )
    spec = importlib.util.spec_from_file_location("round_sequence_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE annotation_rounds (id INTEGER PRIMARY KEY, "
                "project_id INTEGER, cycle_id INTEGER, sequence INTEGER)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO annotation_rounds VALUES "
                "(1,1,NULL,1),(2,1,NULL,1),(3,1,NULL,2),(4,1,NULL,2),(5,1,99,1),(6,2,NULL,1)"
            )
        )
        with Operations.context(MigrationContext.configure(connection)):
            migration.upgrade()
        assert connection.execute(
            text("SELECT id, sequence FROM annotation_rounds ORDER BY id")
        ).all() == [
            (1, 1),
            (2, 3),
            (3, 2),
            (4, 4),
            (5, 1),
            (6, 1),
        ]
        with pytest.raises(IntegrityError):
            connection.execute(text("INSERT INTO annotation_rounds VALUES (7,1,NULL,1)"))
        with Operations.context(MigrationContext.configure(connection)):
            migration.downgrade()
        connection.execute(text("INSERT INTO annotation_rounds VALUES (8,1,NULL,1)"))
    engine.dispose()


def test_l7_checkpoint_launch_hides_other_project_ids(client, db):
    inference = _inference_scope(db, with_window=False)
    db.get(Workspace, inference.project.workspace_id).capability_preset = "full"
    db.query(WorkspaceMember).filter_by(user_id=inference.user.id).update({"role": "trainer"})
    other = Project(name="Another inference project", workspace_id=inference.project.workspace_id)
    db.add(other)
    db.commit()
    payload = {
        "name": "Probe",
        "corpus_snapshot_id": inference.run.corpus_snapshot_id,
        "compute_profile_id": inference.run.compute_profile_id,
        "target_version_ids": [inference.target_version.id],
        "idempotency_key": "scope-probe",
    }
    responses = [
        client.post(
            f"/api/projects/{other.id}/inference/runs",
            json={
                **payload,
                "checkpoint_id": checkpoint_id,
            },
            headers=_headers(inference.user),
        )
        for checkpoint_id in [inference.run.checkpoint_id, 999999]
    ]
    assert [response.status_code for response in responses] == [404, 404]
    assert responses[0].json() == responses[1].json()


def test_l1_workspace_context_pages_batch_shared_references(client, db, scope, workflow_scope):
    first, _ = _round(client, scope)
    db.add_all(
        [
            models.AnnotationRound(
                project_id=scope.project.id,
                sequence=index + 2,
                name=f"Round {index + 2}",
                dataset_version_id=scope.source.id,
                task_version_id=scope.task.id,
                annotator_user_ids=[scope.annotators[0].id],
                assistance_policy="blind",
                status="open",
            )
            for index in range(30)
        ]
    )
    db.commit()
    workspace_id = workflow_scope.workspace.id
    actor = db.get(User, scope.annotators[0].id)
    statements = []

    def record_statement(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.bind, "before_cursor_execute", record_statement)
    try:
        contexts = service.list_workspace_round_work_contexts(
            db, workspace_id=workspace_id, actor=actor, limit=100
        )
    finally:
        event.remove(db.bind, "before_cursor_execute", record_statement)
    assert len(contexts) == 31
    assert len(statements) <= 15, len(statements)
    url = f"/api/workspaces/{workspace_id}/my-work/rounds"
    pages = [
        client.get(url, params={"limit": 20, "offset": offset}, headers=_headers(actor))
        for offset in [0, 20, 40]
    ]
    assert [page.status_code for page in pages] == [200, 200, 200]
    assert [len(page.json()) for page in pages] == [20, 11, 0]
    assert len({item["round"]["id"] for page in pages for item in page.json()}) == 31
