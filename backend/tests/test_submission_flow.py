"""Submission lifecycle: snapshot file creation, assignment finalization, re-export."""

import hashlib
import json

import pytest


def _setup_project_with_annotation(client) -> dict:
    """Create project + entity task + document + assignment + one annotation."""
    project = client.post(
        "/api/projects",
        json={
            "name": "submission-flow",
            "description": "submission tests",
            "tasks": [
                {
                    "annotation_type": "entity",
                    "labels": [{"name": "Drug", "color": "#3366ff"}],
                }
            ],
            "settings": {},
        },
    ).json()
    document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "external_id": "PMID:sub-1",
            "title": "Submission test document",
            "text": "Aspirin reduces fever.",
            "source": "manual",
            "metadata_": {},
        },
    ).json()
    task_id = project["tasks"][0]["id"]
    assignment = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": task_id,
            "document_id": document["id"],
            "annotator_id": "alice",
            "status": "in_progress",
        },
    ).json()
    annotation = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
            "annotator_id": "alice",
        },
    ).json()
    return {
        "project": project,
        "document": document,
        "assignment": assignment,
        "annotation": annotation,
    }


def test_submit_creates_file_record_and_finalizes_assignments(client, object_storage):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]

    response = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"annotator_id": "alice"},
    )
    assert response.status_code == 200
    submission = response.json()
    assert submission["project_id"] == project_id
    assert submission["document_id"] == document_id
    assert submission["annotator_id"] == "alice"
    assert submission["kind"] == "submission"
    assert submission["annotation_count"] == 1
    assert submission["content_type"] == "application/json"

    payload = object_storage.get_bytes(submission["storage_key"])
    assert len(payload) == submission["size_bytes"]
    assert hashlib.sha256(payload).hexdigest() == submission["checksum_sha256"]

    snapshot = json.loads(payload)
    assert snapshot["format_version"] == 1
    assert snapshot["document"]["id"] == document_id
    assert snapshot["document"]["text"] == "Aspirin reduces fever."
    assert snapshot["annotator_id"] == "alice"
    assert len(snapshot["annotations"]) == 1
    assert snapshot["annotations"][0]["label"] == "Drug"

    assignments = client.get(
        f"/api/projects/{project_id}/assignments",
        params={"document_id": document_id},
    ).json()
    assert [assignment["status"] for assignment in assignments] == ["submitted"]


def test_submitted_ordinary_assignment_is_immutable_and_not_resubmittable(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    assignment_id = ctx["assignment"]["id"]
    alice_headers = {"Authorization": f"Bearer {client._token_for('alice')}"}
    submission_url = (
        f"/api/projects/{project_id}/documents/{document_id}/submissions"
    )

    submitted = client.post(
        submission_url,
        json={"assignment_id": assignment_id},
        headers=alice_headers,
    )
    assert submitted.status_code == 200, submitted.text

    create_after_submit = client.post(
        "/api/annotations",
        json={
            "project_id": project_id,
            "document_id": document_id,
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 8,
            "end_offset": 15,
            "text_span": "reduces",
        },
        headers=alice_headers,
    )
    update_after_submit = client.patch(
        f"/api/annotations/{ctx['annotation']['id']}",
        json={"label": "Changed"},
        headers=alice_headers,
    )
    delete_after_submit = client.delete(
        f"/api/annotations/{ctx['annotation']['id']}",
        headers=alice_headers,
    )
    assert {
        create_after_submit.status_code,
        update_after_submit.status_code,
        delete_after_submit.status_code,
    } == {403}

    repeated = client.post(
        submission_url,
        json={"assignment_id": assignment_id},
        headers=alice_headers,
    )
    assert repeated.status_code == 409
    assert "final status" in repeated.json()["detail"]


def test_new_versions_allow_a_fresh_assignment_round_and_scoped_snapshot(
    client,
    object_storage,
):
    project = client.post(
        "/api/projects",
        json={
            "name": "versioned-assignment-rounds",
            "tasks": [{"annotation_type": "entity", "labels": []}],
        },
    ).json()
    document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Versioned rounds",
            "text": "Aspirin improves outcomes.",
        },
    ).json()
    first_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v1",
            "markdown": "First-round policy.",
        },
    ).json()
    assignment_url = f"/api/projects/{project['id']}/assignments"
    first_assignment = client.post(
        assignment_url,
        json={
            "task_id": project["tasks"][0]["id"],
            "document_id": document["id"],
            "annotator_id": "alice",
        },
    ).json()
    alice_headers = {"Authorization": f"Bearer {client._token_for('alice')}"}
    first_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug-v1",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
        },
        headers=alice_headers,
    ).json()
    assert first_annotation["guideline_version_id"] == first_guideline["id"]
    finalized = client.patch(
        f"{assignment_url}/{first_assignment['id']}",
        json={"status": "submitted"},
    )
    assert finalized.status_code == 200, finalized.text

    second_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "v2",
            "markdown": "Second-round policy.",
        },
    ).json()
    rebuilt = client.post(
        f"/api/documents/{document['id']}/structure/rebuild",
        json={"activate": True},
    )
    assert rebuilt.status_code == 200, rebuilt.text
    second_assignment_response = client.post(
        assignment_url,
        json={
            "task_id": project["tasks"][0]["id"],
            "document_id": document["id"],
            "annotator_id": "alice",
        },
    )
    assert second_assignment_response.status_code == 200, second_assignment_response.text
    second_assignment = second_assignment_response.json()
    assert second_assignment["assignment_scope_key"] != first_assignment[
        "assignment_scope_key"
    ]
    assert second_assignment["guideline_version_id"] == second_guideline["id"]
    assert second_assignment["structure_version_id"] == rebuilt.json()[
        "structure_version"
    ]["id"]

    second_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": project["id"],
            "document_id": document["id"],
            "annotation_type": "entity",
            "label": "Drug-v2",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
        },
        headers=alice_headers,
    )
    assert second_annotation.status_code == 200, second_annotation.text
    assert second_annotation.json()["guideline_version_id"] == second_guideline["id"]

    submitted = client.post(
        f"/api/projects/{project['id']}/documents/{document['id']}/submissions",
        json={"assignment_id": second_assignment["id"]},
        headers=alice_headers,
    )
    assert submitted.status_code == 200, submitted.text
    snapshot = json.loads(object_storage.get_bytes(submitted.json()["storage_key"]))
    assert [item["id"] for item in snapshot["annotations"]] == [
        second_annotation.json()["id"]
    ]


def test_open_assignment_keeps_its_version_pin_until_the_round_is_finalized(
    client,
    object_storage,
):
    ctx = _setup_project_with_annotation(client)
    alice_headers = {"Authorization": f"Bearer {client._token_for('alice')}"}
    old_structure_id = ctx["assignment"]["structure_version_id"]
    rebuilt = client.post(
        f"/api/documents/{ctx['document']['id']}/structure/rebuild",
        json={"activate": True},
    )
    assert rebuilt.status_code == 200, rebuilt.text
    assert rebuilt.json()["structure_version"]["id"] != old_structure_id

    continued = client.post(
        "/api/annotations",
        json={
            "project_id": ctx["project"]["id"],
            "document_id": ctx["document"]["id"],
            "annotation_type": "entity",
            "label": "Finding",
            "start_offset": 8,
            "end_offset": 15,
            "text_span": "reduces",
            "structure_version_id": rebuilt.json()["structure_version"]["id"],
        },
        headers=alice_headers,
    )
    assert continued.status_code == 200, continued.text
    assert continued.json()["structure_version_id"] == old_structure_id

    assignment_url = f"/api/projects/{ctx['project']['id']}/assignments"
    premature_round = client.post(
        assignment_url,
        json={
            "task_id": ctx["project"]["tasks"][0]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
        },
    )
    assert premature_round.status_code == 409
    assert "current assignment round" in premature_round.json()["detail"]

    finalized = client.patch(
        f"{assignment_url}/{ctx['assignment']['id']}",
        json={"status": "submitted"},
    )
    assert finalized.status_code == 200, finalized.text
    next_round = client.post(
        assignment_url,
        json={
            "task_id": ctx["project"]["tasks"][0]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
        },
    )
    assert next_round.status_code == 200, next_round.text
    assert next_round.json()["structure_version_id"] == rebuilt.json()[
        "structure_version"
    ]["id"]
    next_annotation = client.post(
        "/api/annotations",
        json={
            "project_id": ctx["project"]["id"],
            "document_id": ctx["document"]["id"],
            "annotation_type": "entity",
            "label": "Finding-v2",
            "start_offset": 0,
            "end_offset": 7,
            "text_span": "Aspirin",
        },
        headers=alice_headers,
    )
    assert next_annotation.status_code == 200, next_annotation.text
    submitted = client.post(
        (
            f"/api/projects/{ctx['project']['id']}"
            f"/documents/{ctx['document']['id']}/submissions"
        ),
        json={"assignment_id": next_round.json()["id"]},
        headers=alice_headers,
    )
    assert submitted.status_code == 200, submitted.text
    snapshot = json.loads(object_storage.get_bytes(submitted.json()["storage_key"]))
    assert [item["id"] for item in snapshot["annotations"]] == [
        next_annotation.json()["id"]
    ]


def test_submit_uses_token_identity_and_leaves_other_assignments_open(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    task_id = ctx["project"]["tasks"][0]["id"]
    client.post(
        f"/api/projects/{project_id}/assignments",
        json={
            "task_id": task_id,
            "document_id": document_id,
            "annotator_id": "bob",
            "status": "in_progress",
        },
    )
    client.post(
        f"/api/projects/{project_id}/assignments",
        json={
            "task_id": task_id,
            "document_id": document_id,
            "annotator_id": "tester",
            "status": "in_progress",
        },
    )

    response = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={},
    )
    assert response.status_code == 200
    assert response.json()["annotator_id"] == "tester"

    assignments = client.get(
        f"/api/projects/{project_id}/assignments",
        params={"document_id": document_id},
    ).json()
    statuses = {assignment["annotator_id"]: assignment["status"] for assignment in assignments}
    assert statuses == {
        "alice": "in_progress",
        "bob": "in_progress",
        "tester": "submitted",
    }
    assert len(client.get(f"/api/projects/{project_id}/submissions").json()) == 1


def test_assignment_managed_team_rejects_assignmentless_manager_submission(client):
    ctx = _setup_project_with_annotation(client)
    response = client.post(
        (
            f"/api/projects/{ctx['project']['id']}"
            f"/documents/{ctx['document']['id']}/submissions"
        ),
        json={},
    )
    assert response.status_code == 422
    assert "assignment_id is required" in response.json()["detail"]


def test_submit_only_finalizes_matching_annotator(client, object_storage):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    task_id = ctx["project"]["tasks"][0]["id"]
    client.post(
        f"/api/projects/{project_id}/assignments",
        json={
            "task_id": task_id,
            "document_id": document_id,
            "annotator_id": "bob",
            "status": "in_progress",
        },
    )
    client.post(
        "/api/annotations",
        json={
            "project_id": project_id,
            "document_id": document_id,
            "annotation_type": "entity",
            "label": "Drug",
            "start_offset": 8,
            "end_offset": 15,
            "text_span": "reduces",
            "annotator_id": "bob",
        },
    )

    response = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"annotator_id": "alice"},
    )
    assert response.status_code == 200
    snapshot = json.loads(object_storage.get_bytes(response.json()["storage_key"]))
    assert [annotation["annotator_id"] for annotation in snapshot["annotations"]] == ["alice"]

    assignments = client.get(
        f"/api/projects/{project_id}/assignments",
        params={"document_id": document_id},
    ).json()
    statuses = {assignment["annotator_id"]: assignment["status"] for assignment in assignments}
    assert statuses == {"alice": "submitted", "bob": "in_progress"}


def test_reexport_leaves_assignments_untouched(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]

    response = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"kind": "re_export"},
    )
    assert response.status_code == 200
    assert response.json()["kind"] == "re_export"

    assignments = client.get(
        f"/api/projects/{project_id}/assignments",
        params={"document_id": document_id},
    ).json()
    assert [assignment["status"] for assignment in assignments] == ["in_progress"]


def test_submit_unknown_document_returns_404(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    response = client.post(
        f"/api/projects/{project_id}/documents/999999/submissions",
        json={},
    )
    assert response.status_code == 404


def test_submission_commit_failure_removes_written_object_and_rolls_back(
    client,
    db,
    object_storage,
    monkeypatch,
):
    from sqlalchemy.orm import Session

    from al_medlit.project.models import TaskAssignment
    from al_medlit.submission.models import AnnotationSubmission

    ctx = _setup_project_with_annotation(client)
    original_commit = Session.commit

    def fail_commit(_session):
        raise RuntimeError("forced submission commit failure")

    monkeypatch.setattr(Session, "commit", fail_commit)
    with pytest.raises(RuntimeError, match="forced submission commit failure"):
        client.post(
            (
                f"/api/projects/{ctx['project']['id']}"
                f"/documents/{ctx['document']['id']}/submissions"
            ),
            json={"annotator_id": "alice"},
        )
    monkeypatch.setattr(Session, "commit", original_commit)

    assert not [path for path in object_storage.root.rglob("*") if path.is_file()]
    assert db.query(AnnotationSubmission).count() == 0
    assignment = db.get(TaskAssignment, ctx["assignment"]["id"])
    db.refresh(assignment)
    assert assignment.status == "in_progress"


def test_submission_storage_failure_after_publish_cleans_object_and_rolls_back(
    client,
    db,
    object_storage,
    monkeypatch,
):
    from al_medlit.project.models import TaskAssignment
    from al_medlit.submission.models import AnnotationSubmission

    ctx = _setup_project_with_annotation(client)
    original_put = object_storage.put_bytes

    def store_then_fail(key, data, *, content_type=None):
        original_put(key, data, content_type=content_type)
        raise RuntimeError("forced post-publish storage failure")

    monkeypatch.setattr(object_storage, "put_bytes", store_then_fail)
    with pytest.raises(RuntimeError, match="post-publish storage failure"):
        client.post(
            (
                f"/api/projects/{ctx['project']['id']}"
                f"/documents/{ctx['document']['id']}/submissions"
            ),
            json={"annotator_id": "alice"},
        )

    assert not [path for path in object_storage.root.rglob("*") if path.is_file()]
    assert db.query(AnnotationSubmission).count() == 0
    assignment = db.get(TaskAssignment, ctx["assignment"]["id"])
    db.refresh(assignment)
    assert assignment.status == "in_progress"


def test_submission_create_queues_object_when_rollback_cleanup_fails(
    client,
    db,
    object_storage,
    monkeypatch,
):
    from al_medlit.storage_reclaim.models import OrphanedStorageObject
    from al_medlit.submission.models import AnnotationSubmission

    ctx = _setup_project_with_annotation(client)
    original_put = object_storage.put_bytes

    def store_then_fail(key, data, *, content_type=None):
        original_put(key, data, content_type=content_type)
        raise RuntimeError("forced post-publish storage failure")

    def fail_delete(_key):
        raise RuntimeError("forced storage cleanup failure")

    monkeypatch.setattr(object_storage, "put_bytes", store_then_fail)
    monkeypatch.setattr(object_storage, "delete", fail_delete)
    with pytest.raises(RuntimeError, match="post-publish storage failure"):
        client.post(
            (
                f"/api/projects/{ctx['project']['id']}"
                f"/documents/{ctx['document']['id']}/submissions"
            ),
            json={"annotator_id": "alice"},
        )

    assert db.query(AnnotationSubmission).count() == 0
    queued = db.query(OrphanedStorageObject).one()
    assert queued.origin == "submission.create_rollback"
    assert object_storage.get_bytes(queued.storage_key)


def test_submission_lock_refresh_rejects_cached_finalized_assignment(
    client,
    object_storage,
    testing_session_factory,
):
    from al_medlit.core.exceptions import ConflictError
    from al_medlit.project.models import TaskAssignment
    from al_medlit.submission.schemas import SubmissionCreate
    from al_medlit.submission.service import create_submission

    ctx = _setup_project_with_annotation(client)
    waiting_session = testing_session_factory()
    winner_session = testing_session_factory()
    try:
        cached = waiting_session.get(TaskAssignment, ctx["assignment"]["id"])
        assert cached.status == "in_progress"

        finalized = winner_session.get(TaskAssignment, cached.id)
        finalized.status = "submitted"
        winner_session.commit()

        with pytest.raises(ConflictError, match="final status"):
            create_submission(
                waiting_session,
                object_storage,
                project_id=ctx["project"]["id"],
                document_id=ctx["document"]["id"],
                data=SubmissionCreate(
                    assignment_id=cached.id,
                    annotator_user_id=cached.assignee_user_id,
                ),
            )
    finally:
        waiting_session.rollback()
        waiting_session.close()
        winner_session.close()


def test_list_submissions_with_filters(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"annotator_id": "alice"},
    )
    client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"kind": "re_export"},
    )

    all_rows = client.get(f"/api/projects/{project_id}/submissions").json()
    assert len(all_rows) == 2

    alice_rows = client.get(
        f"/api/projects/{project_id}/submissions",
        params={"annotator_id": "alice"},
    ).json()
    assert len(alice_rows) == 1
    assert alice_rows[0]["annotator_id"] == "alice"

    reexports = client.get(
        f"/api/projects/{project_id}/submissions",
        params={"kind": "re_export"},
    ).json()
    assert len(reexports) == 1

    missing_document = client.get(
        f"/api/projects/{project_id}/submissions",
        params={"document_id": 999999},
    )
    assert missing_document.status_code == 404


def test_download_submission(client):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    submission = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"annotator_id": "alice"},
    ).json()

    response = client.get(f"/api/submissions/{submission['id']}/download")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert submission["file_name"] in response.headers["content-disposition"]
    snapshot = response.json()
    assert snapshot["document"]["id"] == document_id


def test_download_submission_rejects_tampered_snapshot(client, object_storage):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    submission = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"annotator_id": "alice"},
    ).json()
    object_storage.put_bytes(
        submission["storage_key"],
        b'{"annotations": [], "tampered": true}',
        content_type="application/json",
    )

    response = client.get(f"/api/submissions/{submission['id']}/download")
    assert response.status_code == 409


def test_delete_submission_removes_record_and_file(client, object_storage):
    ctx = _setup_project_with_annotation(client)
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    submission = client.post(
        f"/api/projects/{project_id}/documents/{document_id}/submissions",
        json={"kind": "re_export"},
    ).json()

    response = client.delete(f"/api/submissions/{submission['id']}")
    assert response.status_code == 204

    assert client.get(f"/api/projects/{project_id}/submissions").json() == []
    assert client.get(f"/api/submissions/{submission['id']}/download").status_code == 404

    from al_medlit.core.storage import ObjectNotFoundError

    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(submission["storage_key"])


def test_submission_delete_remains_committed_when_storage_cleanup_fails(
    client,
    db,
    object_storage,
    monkeypatch,
):
    from al_medlit.storage_reclaim.models import OrphanedStorageObject

    ctx = _setup_project_with_annotation(client)
    submission = client.post(
        (
            f"/api/projects/{ctx['project']['id']}"
            f"/documents/{ctx['document']['id']}/submissions"
        ),
        json={"kind": "re_export"},
    ).json()

    def fail_delete(_key):
        raise RuntimeError("forced storage cleanup failure")

    monkeypatch.setattr(object_storage, "delete", fail_delete)
    response = client.delete(f"/api/submissions/{submission['id']}")
    assert response.status_code == 204
    assert client.get(f"/api/submissions/{submission['id']}").status_code == 404
    # The unreferenced object survives the failed delete, so it must be queued
    # for the reclaim sweep rather than left to leak.
    assert object_storage.get_bytes(submission["storage_key"])
    queued = (
        db.query(OrphanedStorageObject)
        .filter(OrphanedStorageObject.storage_key == submission["storage_key"])
        .one()
    )
    assert queued.origin == "submission.delete"
    assert queued.attempts == 1
    assert "forced storage cleanup failure" in queued.last_error


def test_download_unknown_submission_returns_404(client):
    assert client.get("/api/submissions/424242/download").status_code == 404


def test_submission_reference_blocks_assignment_delete_and_reassignment(client):
    project = client.post(
        "/api/projects",
        json={
            "name": "submission-assignment-integrity",
            "tasks": [{"annotation_type": "entity"}],
        },
    ).json()
    document = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "No annotations required",
            "text": "Document text.",
        },
    ).json()
    assignment = client.post(
        f"/api/projects/{project['id']}/assignments",
        json={
            "task_id": project["tasks"][0]["id"],
            "document_id": document["id"],
            "annotator_id": "alice",
        },
    ).json()
    submitted = client.post(
        f"/api/projects/{project['id']}/documents/{document['id']}/submissions",
        json={"assignment_id": assignment["id"], "annotator_id": "alice"},
    )
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["annotation_count"] == 0

    client._token_for("bob")
    reassigned = client.patch(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}",
        json={"annotator_id": "bob"},
    )
    assert reassigned.status_code == 409
    deleted = client.delete(
        f"/api/projects/{project['id']}/assignments/{assignment['id']}"
    )
    assert deleted.status_code == 409
