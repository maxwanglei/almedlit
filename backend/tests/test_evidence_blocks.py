import pytest

from al_medlit.annotation.models import Annotation
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_task_assigned
from al_medlit.core.exceptions import ForbiddenError
from al_medlit.evidence.models import (
    EvidenceBlockAnnotation,
    EvidenceReviewCoverage,
    EvidenceTargetVersion,
    ImmutableEvidenceTargetVersionError,
)
from al_medlit.iaa import evidence_metrics
from al_medlit.iaa import service as iaa_service
from al_medlit.project.models import Project, TaskAssignment
from al_medlit.submission.snapshot import build_snapshot
from al_medlit.workspace.models import WorkspaceMember


def _setup_evidence_project(client, name: str = "evidence-api") -> dict:
    project_response = client.post(
        "/api/projects",
        json={
            "name": name,
            "tasks": [{"annotation_type": "evidence_block"}],
            "settings": {},
        },
    )
    assert project_response.status_code == 200, project_response.text
    project = project_response.json()
    guideline_response = client.post(
        "/api/guidelines",
        json={
            "project_id": project["id"],
            "version_label": "evidence-v1",
            "markdown": "Select sentence-aligned evidence.",
        },
    )
    assert guideline_response.status_code == 200, guideline_response.text
    guideline = guideline_response.json()
    task = project["tasks"][0]
    assert task["settings"]["active_target_ids"] == []
    assert task["settings"]["model_context_tokens"] == 4096
    assert task["settings"]["keyboard_shortcuts"]["expand_start"] == "shift+arrowup"

    document_response = client.post(
        "/api/documents",
        json={
            "project_id": project["id"],
            "title": "Evidence document",
            "text": "Alpha improves outcomes. Beta is contextual. Gamma confirms benefit.",
            "source": "manual",
            "metadata_": {},
        },
    )
    assert document_response.status_code == 200, document_response.text
    document = document_response.json()
    structure_response = client.get(f"/api/documents/{document['id']}/structure")
    assert structure_response.status_code == 200, structure_response.text
    structure = structure_response.json()
    assert len(structure["sentences"]) == 3

    target_response = client.post(
        f"/api/projects/{project['id']}/evidence-targets",
        json={
            "task_id": task["id"],
            "key": "benefit",
            "name": "Treatment benefit",
            "description": "Evidence of benefit",
            "initial_version": {
                "text": "Does the treatment improve outcomes?",
                "guidance": "Select complete supporting sentences.",
            },
        },
    )
    assert target_response.status_code == 200, target_response.text
    target = target_response.json()
    version = target["versions"][0]
    activate_response = client.post(
        f"/api/projects/{project['id']}/evidence-targets/{target['id']}/activate",
        json={"version_id": version["id"]},
    )
    assert activate_response.status_code == 200, activate_response.text
    assert activate_response.json()["active_version_id"] == version["id"]
    task_response = client.get(f"/api/projects/{project['id']}/tasks")
    assert task_response.json()[0]["settings"]["active_target_ids"] == [target["id"]]
    return {
        "project": project,
        "task": task,
        "document": document,
        "structure": structure,
        "target": target,
        "version": version,
        "guideline": guideline,
    }


def _block_payload(ctx: dict, start_index: int, end_index: int, **extra) -> dict:
    sentences = ctx["structure"]["sentences"]
    payload = {
        "project_id": ctx["project"]["id"],
        "document_id": ctx["document"]["id"],
        "annotation_type": "evidence_block",
        "label": "forged-label",
        "start_offset": 999,
        "end_offset": 1000,
        "text_span": "forged",
        "source": "model",
        "status": "gold",
        "guideline_version_id": ctx["guideline"]["id"],
        "evidence_block": {
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "target_version_id": ctx["version"]["id"],
            "start_sentence_id": sentences[start_index]["id"],
            "end_sentence_id": sentences[end_index]["id"],
            "labels": ["supporting"],
            "note": "review note",
            "boundary_policy": "sentence",
        },
    }
    payload.update(extra)
    return payload


def test_evidence_lock_refreshes_cached_block_revision(
    client,
    testing_session_factory,
):
    from al_medlit.evidence.service import _required_block_for_update

    ctx = _setup_evidence_project(client, "evidence-stale-revision-refresh")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0))
    assert created.status_code == 200, created.text
    annotation_id = created.json()["id"]

    waiting_session = testing_session_factory()
    winner_session = testing_session_factory()
    try:
        cached = waiting_session.get(Annotation, annotation_id)
        cached_revision = cached.evidence_block.revision

        updated = winner_session.get(EvidenceBlockAnnotation, annotation_id)
        updated.revision += 1
        winner_session.commit()

        _annotation, refreshed_block = _required_block_for_update(
            waiting_session,
            annotation_id,
        )
        assert refreshed_block.revision == cached_revision + 1
    finally:
        waiting_session.rollback()
        waiting_session.close()
        winner_session.close()


def test_required_evidence_assignment_cannot_disappear_after_authorization(client, db):
    from al_medlit.evidence.service import (
        _lock_actor_assignment_if_present,
        _lock_evidence_scope,
    )

    ctx = _setup_evidence_project(client, "evidence-assignment-disappeared")
    assignment = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "vanished-assignee",
            "target_version_id": ctx["version"]["id"],
        },
    ).json()
    assignment_row = db.get(TaskAssignment, assignment["id"])
    db.delete(assignment_row)
    db.commit()

    _lock_evidence_scope(db, ctx["version"]["id"])
    with pytest.raises(ForbiddenError, match="no longer assigned"):
        _lock_actor_assignment_if_present(
            db,
            project_id=ctx["project"]["id"],
            document_id=ctx["document"]["id"],
            target_version_id=ctx["version"]["id"],
            structure_version_id=ctx["structure"]["structure_version"]["id"],
            guideline_version_id=ctx["guideline"]["id"],
            actor_user_id=assignment["assignee_user_id"],
            required_assignment_id=assignment["id"],
        )
    db.rollback()


def test_targets_assignments_and_server_derived_block_payload(client):
    ctx = _setup_evidence_project(client, "evidence-target-assignment")
    assignment_response = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
            "target_version_id": ctx["version"]["id"],
        },
    )
    assert assignment_response.status_code == 200, assignment_response.text
    assignment = assignment_response.json()
    assert assignment["assignment_scope_key"] == (
        f"target:{ctx['version']['id']}:"
        f"structure:{ctx['structure']['structure_version']['id']}:"
        f"guideline:{ctx['guideline']['id']}"
    )
    assert assignment["structure_version_id"] == ctx["structure"]["structure_version"][
        "id"
    ]

    create_response = client.post("/api/annotations", json=_block_payload(ctx, 0, 1))
    assert create_response.status_code == 200, create_response.text
    block = create_response.json()
    first, second = ctx["structure"]["sentences"][:2]
    assert block["label"] == "evidence_block"
    assert block["source"] == "human"
    assert block["status"] == "draft"
    assert block["start_offset"] == first["start_offset"]
    assert block["end_offset"] == second["end_offset"]
    assert block["text_span"] == ctx["document"]["text"][
        first["start_offset"] : second["end_offset"]
    ]
    assert block["evidence_block"]["revision"] == 1
    assert block["evidence_block"]["last_command_group_key"]

    overlap = client.post("/api/annotations", json=_block_payload(ctx, 1, 2))
    assert overlap.status_code == 422
    assert "may not overlap" in overlap.json()["detail"]


def test_adjacency_policy_still_applies_when_overlap_is_allowed(client):
    ctx = _setup_evidence_project(client, "evidence-adjacency-policy")
    task = client.get(f"/api/projects/{ctx['project']['id']}/tasks").json()[0]
    settings = {
        **task["settings"],
        "same_target_overlap_allowed": True,
        "adjacency_allowed": False,
    }
    updated = client.patch(
        f"/api/projects/{ctx['project']['id']}/tasks/{ctx['task']['id']}",
        json={"settings": settings},
    )
    assert updated.status_code == 200, updated.text

    assert client.post(
        "/api/annotations", json=_block_payload(ctx, 0, 0)
    ).status_code == 200
    adjacent = client.post("/api/annotations", json=_block_payload(ctx, 1, 1))
    assert adjacent.status_code == 422
    assert "Adjacent evidence blocks are disabled" in adjacent.json()["detail"]


def test_evidence_keyboard_shortcuts_are_normalized_and_must_be_unique(client):
    ctx = _setup_evidence_project(client, "evidence-keyboard-shortcuts")
    task = client.get(f"/api/projects/{ctx['project']['id']}/tasks").json()[0]
    shortcuts = {**task["settings"]["keyboard_shortcuts"], "create": " Shift + E "}
    settings = {**task["settings"], "keyboard_shortcuts": shortcuts}
    updated = client.patch(
        f"/api/projects/{ctx['project']['id']}/tasks/{ctx['task']['id']}",
        json={"settings": settings},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["settings"]["keyboard_shortcuts"]["create"] == "shift+e"

    duplicate_shortcuts = {
        **updated.json()["settings"]["keyboard_shortcuts"],
        "merge": "shift+e",
    }
    duplicate_settings = {
        **updated.json()["settings"],
        "keyboard_shortcuts": duplicate_shortcuts,
    }
    rejected = client.patch(
        f"/api/projects/{ctx['project']['id']}/tasks/{ctx['task']['id']}",
        json={"settings": duplicate_settings},
    )
    assert rejected.status_code == 422


def test_restricted_annotator_cannot_cross_target_assignment(client, db):
    ctx = _setup_evidence_project(client, "evidence-target-auth")
    second_target_response = client.post(
        f"/api/projects/{ctx['project']['id']}/evidence-targets",
        json={
            "task_id": ctx["task"]["id"],
            "key": "harm",
            "name": "Treatment harm",
            "initial_version": {"text": "Does the treatment cause harm?"},
        },
    )
    assert second_target_response.status_code == 200
    second_target = second_target_response.json()
    second_version = second_target["versions"][0]
    assert client.post(
        f"/api/projects/{ctx['project']['id']}/evidence-targets/"
        f"{second_target['id']}/activate",
        json={"version_id": second_version["id"]},
    ).status_code == 200
    for username, target_version_id in (
        ("alice", ctx["version"]["id"]),
        ("bob", second_version["id"]),
    ):
        response = client.post(
            f"/api/projects/{ctx['project']['id']}/assignments",
            json={
                "task_id": ctx["task"]["id"],
                "document_id": ctx["document"]["id"],
                "annotator_id": username,
                "target_version_id": target_version_id,
            },
        )
        assert response.status_code == 200, response.text

    alice = db.query(User).filter(User.username == "alice").one()
    restricted_member = WorkspaceMember(
        workspace_id=ctx["project"]["workspace_id"],
        user_id=alice.id,
        role="annotator",
    )
    assert_task_assigned(
        db,
        alice,
        restricted_member,
        project_id=ctx["project"]["id"],
        document_id=ctx["document"]["id"],
        annotation_type="evidence_block",
        target_version_id=ctx["version"]["id"],
        structure_version_id=ctx["structure"]["structure_version"]["id"],
    )
    with pytest.raises(ForbiddenError):
        assert_task_assigned(
            db,
            alice,
            restricted_member,
            project_id=ctx["project"]["id"],
            document_id=ctx["document"]["id"],
            annotation_type="evidence_block",
            target_version_id=second_version["id"],
            structure_version_id=ctx["structure"]["structure_version"]["id"],
        )


def test_revision_coverage_and_persisted_undo_redo(client):
    ctx = _setup_evidence_project(client, "evidence-revision-coverage")
    sentences = ctx["structure"]["sentences"]
    coverage_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/evidence-review-coverage"
    )
    review_body = {
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "start_sentence_id": sentences[0]["id"],
        "end_sentence_id": sentences[2]["id"],
    }
    reviewed = client.post(f"{coverage_url}/mark-reviewed", json=review_body)
    assert reviewed.status_code == 200, reviewed.text
    assert reviewed.json()["fully_reviewed"] is True

    created_response = client.post("/api/annotations", json=_block_payload(ctx, 1, 1))
    assert created_response.status_code == 200, created_response.text
    created = created_response.json()
    coverage = client.get(
        coverage_url,
        params={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
        },
    ).json()
    assert coverage["fully_reviewed"] is False
    assert [
        (item["start_sentence_ordinal"], item["end_sentence_ordinal"])
        for item in coverage["intervals"]
    ] == [(0, 0), (2, 2)]

    evidence_payload = _block_payload(ctx, 0, 1)["evidence_block"]
    updated_response = client.patch(
        f"/api/annotations/{created['id']}",
        json={"expected_revision": 1, "evidence_block": evidence_payload},
    )
    assert updated_response.status_code == 200, updated_response.text
    updated = updated_response.json()
    assert updated["evidence_block"]["revision"] == 2
    stale = client.patch(
        f"/api/annotations/{created['id']}",
        json={"expected_revision": 1, "evidence_block": evidence_payload},
    )
    assert stale.status_code == 409

    command_key = updated["evidence_block"]["last_command_group_key"]
    undo = client.post(f"/api/annotations/evidence-blocks/commands/{command_key}/undo")
    assert undo.status_code == 200, undo.text
    assert undo.json()["status"] == "undone"
    assert undo.json()["annotations"][0]["evidence_block"]["revision"] == 1
    redo = client.post(f"/api/annotations/evidence-blocks/commands/{command_key}/redo")
    assert redo.status_code == 200, redo.text
    assert redo.json()["status"] == "applied"
    assert redo.json()["annotations"][0]["evidence_block"]["revision"] == 2


def test_delete_command_can_be_discovered_undone_and_redone(client):
    ctx = _setup_evidence_project(client, "evidence-delete-undo")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    deleted = client.delete(
        f"/api/annotations/{created['id']}", params={"expected_revision": 1}
    )
    assert deleted.status_code == 204
    commands = client.get(
        "/api/annotations/evidence-blocks/commands",
        params={
            "project_id": ctx["project"]["id"],
            "document_id": ctx["document"]["id"],
        },
    )
    assert commands.status_code == 200, commands.text
    delete_command = next(
        command for command in commands.json() if command["operation"] == "delete"
    )
    command_key = delete_command["command_group_key"]
    undo = client.post(f"/api/annotations/evidence-blocks/commands/{command_key}/undo")
    assert undo.status_code == 200, undo.text
    assert undo.json()["annotations"][0]["id"] == created["id"]
    redo = client.post(f"/api/annotations/evidence-blocks/commands/{command_key}/redo")
    assert redo.status_code == 200, redo.text
    assert redo.json()["annotations"] == []
    assert client.get(f"/api/annotations/{created['id']}").status_code == 404


@pytest.mark.parametrize("operation", ["delete", "merge", "split"])
def test_evidence_rewrites_reject_correction_referenced_sources(client, operation):
    ctx = _setup_evidence_project(client, f"evidence-correction-ref-{operation}")
    if operation == "merge":
        first = client.post(
            "/api/annotations",
            json=_block_payload(ctx, 0, 0),
        ).json()
        second = client.post(
            "/api/annotations",
            json=_block_payload(ctx, 1, 1),
        ).json()
    else:
        first = client.post(
            "/api/annotations",
            json=_block_payload(ctx, 0, 1),
        ).json()
        second = None

    correction = client.post(
        "/api/annotations/corrections",
        json={
            "project_id": ctx["project"]["id"],
            "document_id": ctx["document"]["id"],
            "original_annotation_id": first["id"],
            "correction_note": "Keep this source for the audit trail.",
        },
    )
    assert correction.status_code == 200, correction.text

    if operation == "delete":
        response = client.delete(
            f"/api/annotations/{first['id']}",
            params={"expected_revision": 1},
        )
    elif operation == "merge":
        response = client.post(
            "/api/annotations/evidence-blocks/merge",
            json={
                "annotation_ids": [first["id"], second["id"]],
                "expected_revisions": {
                    str(first["id"]): 1,
                    str(second["id"]): 1,
                },
            },
        )
    else:
        response = client.post(
            f"/api/annotations/{first['id']}/split",
            json={
                "expected_revision": 1,
                "split_before_sentence_id": ctx["structure"]["sentences"][1]["id"],
            },
        )
    assert response.status_code == 409, response.text
    assert "correction" in response.json()["detail"].lower()


def test_command_history_is_version_scoped_and_filters_guideline(client):
    ctx = _setup_evidence_project(client, "evidence-command-scope")
    first = client.post("/api/annotations", json=_block_payload(ctx, 0, 0))
    assert first.status_code == 200, first.text
    old_structure_id = ctx["structure"]["structure_version"]["id"]

    rebuilt_response = client.post(
        f"/api/documents/{ctx['document']['id']}/structure/rebuild",
        json={"activate": True},
    )
    assert rebuilt_response.status_code == 200, rebuilt_response.text
    rebuilt = rebuilt_response.json()
    rebuilt_ctx = {**ctx, "structure": rebuilt}
    second = client.post("/api/annotations", json=_block_payload(rebuilt_ctx, 1, 1))
    assert second.status_code == 200, second.text
    new_structure_id = rebuilt["structure_version"]["id"]

    command_url = "/api/annotations/evidence-blocks/commands"
    common = {
        "project_id": ctx["project"]["id"],
        "document_id": ctx["document"]["id"],
        "target_version_id": ctx["version"]["id"],
        "guideline_version_id": ctx["guideline"]["id"],
    }
    old_commands = client.get(
        command_url,
        params={**common, "structure_version_id": old_structure_id},
    )
    new_commands = client.get(
        command_url,
        params={**common, "structure_version_id": new_structure_id},
    )
    assert old_commands.status_code == 200, old_commands.text
    assert new_commands.status_code == 200, new_commands.text
    assert {
        command["command_group_key"] for command in old_commands.json()
    } == {first.json()["evidence_block"]["last_command_group_key"]}
    assert {
        command["command_group_key"] for command in new_commands.json()
    } == {second.json()["evidence_block"]["last_command_group_key"]}
    for command, structure_id in (
        (old_commands.json()[0], old_structure_id),
        (new_commands.json()[0], new_structure_id),
    ):
        assert command["structure_version_id"] == structure_id
        assert command["guideline_version_id"] == ctx["guideline"]["id"]

    other_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": ctx["project"]["id"],
            "version_label": "unrelated-command-scope",
            "markdown": "Different guidance.",
        },
    ).json()
    assert client.get(
        command_url,
        params={**common, "guideline_version_id": other_guideline["id"]},
    ).json() == []


def test_command_stack_order_and_redo_branch_invalidation(client):
    ctx = _setup_evidence_project(client, "evidence-command-stack")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    create_key = created["evidence_block"]["last_command_group_key"]

    payload = _block_payload(ctx, 0, 0)["evidence_block"]
    payload["note"] = "second"
    second = client.patch(
        f"/api/annotations/{created['id']}",
        json={"expected_revision": 1, "evidence_block": payload},
    ).json()
    second_key = second["evidence_block"]["last_command_group_key"]
    payload["note"] = "third"
    third = client.patch(
        f"/api/annotations/{created['id']}",
        json={"expected_revision": 2, "evidence_block": payload},
    ).json()
    third_key = third["evidence_block"]["last_command_group_key"]
    command_url = "/api/annotations/evidence-blocks/commands"

    assert client.post(f"{command_url}/{create_key}/undo").status_code == 409
    assert client.post(f"{command_url}/{third_key}/undo").status_code == 200
    assert client.post(f"{command_url}/{second_key}/undo").status_code == 200
    assert client.post(f"{command_url}/{third_key}/redo").status_code == 409
    assert client.post(f"{command_url}/{second_key}/redo").status_code == 200

    payload["note"] = "new branch"
    branched = client.patch(
        f"/api/annotations/{created['id']}",
        json={"expected_revision": 2, "evidence_block": payload},
    )
    assert branched.status_code == 200, branched.text
    assert client.post(f"{command_url}/{third_key}/redo").status_code == 409
    visible_keys = {
        command["command_group_key"]
        for command in client.get(
            command_url,
            params={"project_id": ctx["project"]["id"]},
        ).json()
    }
    assert third_key not in visible_keys


def test_undo_delete_rejects_reused_annotation_id(client, db):
    ctx = _setup_evidence_project(client, "evidence-command-id-reuse")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    assert client.delete(
        f"/api/annotations/{created['id']}", params={"expected_revision": 1}
    ).status_code == 204
    delete_command = next(
        command
        for command in client.get(
            "/api/annotations/evidence-blocks/commands",
            params={"project_id": ctx["project"]["id"]},
        ).json()
        if command["operation"] == "delete"
    )

    reused = Annotation(
        id=created["id"],
        project_id=ctx["project"]["id"],
        document_id=ctx["document"]["id"],
        annotation_type="entity",
        label="must-survive",
        start_offset=0,
        end_offset=5,
        text_span="Alpha",
        source="human",
        status="draft",
        annotator_user_id=created["annotator_user_id"],
        annotator_id=created["annotator_id"],
    )
    db.add(reused)
    db.commit()

    replay = client.post(
        "/api/annotations/evidence-blocks/commands/"
        f"{delete_command['command_group_key']}/undo"
    )
    assert replay.status_code == 409, replay.text
    db.expire_all()
    preserved = db.get(Annotation, created["id"])
    assert preserved.annotation_type == "entity"
    assert preserved.label == "must-survive"


def test_delete_and_undo_automatically_reopen_reviewed_coverage(client):
    ctx = _setup_evidence_project(client, "evidence-delete-reopens-coverage")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    coverage_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/evidence-review-coverage"
    )
    review_body = {
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "start_sentence_id": ctx["structure"]["sentences"][0]["id"],
        "end_sentence_id": ctx["structure"]["sentences"][-1]["id"],
    }
    assert client.post(f"{coverage_url}/mark-reviewed", json=review_body).status_code == 200
    assert client.delete(
        f"/api/annotations/{created['id']}", params={"expected_revision": 1}
    ).status_code == 204
    coverage = client.get(
        coverage_url,
        params={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
        },
    ).json()
    assert coverage["fully_reviewed"] is False
    assert coverage["intervals"][0]["start_sentence_ordinal"] == 1

    assert client.post(f"{coverage_url}/mark-reviewed", json=review_body).status_code == 200
    delete_command = next(
        command
        for command in client.get(
            "/api/annotations/evidence-blocks/commands",
            params={"project_id": ctx["project"]["id"]},
        ).json()
        if command["operation"] == "delete"
    )
    assert client.post(
        "/api/annotations/evidence-blocks/commands/"
        f"{delete_command['command_group_key']}/undo"
    ).status_code == 200
    coverage = client.get(
        coverage_url,
        params={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
        },
    ).json()
    assert coverage["fully_reviewed"] is False
    assert coverage["intervals"][0]["start_sentence_ordinal"] == 1


def test_merge_split_and_adjudication_gold_is_read_only(client, db):
    ctx = _setup_evidence_project(client, "evidence-merge-adjudication")
    project = db.get(Project, ctx["project"]["id"])
    project.workspace.kind = "individual"
    db.commit()
    left = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    right = client.post("/api/annotations", json=_block_payload(ctx, 1, 1)).json()
    merge_response = client.post(
        "/api/annotations/evidence-blocks/merge",
        json={
            "annotation_ids": [left["id"], right["id"]],
            "expected_revisions": {str(left["id"]): 1, str(right["id"]): 1},
        },
    )
    assert merge_response.status_code == 200, merge_response.text
    merged = merge_response.json()
    assert merged["evidence_block"]["start_sentence_ordinal"] == 0
    assert merged["evidence_block"]["end_sentence_ordinal"] == 1

    split_response = client.post(
        f"/api/annotations/{merged['id']}/split",
        json={
            "expected_revision": 1,
            "split_before_sentence_id": ctx["structure"]["sentences"][1]["id"],
        },
    )
    assert split_response.status_code == 200, split_response.text
    split_blocks = split_response.json()
    assert len(split_blocks) == 2

    sentences = ctx["structure"]["sentences"]
    reviewed = client.post(
        (
            f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
            "/evidence-review-coverage/mark-reviewed"
        ),
        json={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "start_sentence_id": sentences[0]["id"],
            "end_sentence_id": sentences[-1]["id"],
        },
    )
    assert reviewed.status_code == 200, reviewed.text

    adjudication_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/evidence-adjudication"
    )
    adjudication_payload = {
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "guideline_version_id": ctx["guideline"]["id"],
        "strategy": "union",
        "source_annotation_ids": [item["id"] for item in split_blocks],
        "labels": ["gold-support"],
    }
    implicit_solo = client.post(adjudication_url, json=adjudication_payload)
    assert implicit_solo.status_code == 422
    assert "solo_gold=true" in implicit_solo.json()["detail"]
    adjudication_response = client.post(
        adjudication_url,
        json={**adjudication_payload, "solo_gold": True},
    )
    assert adjudication_response.status_code == 201, adjudication_response.text
    gold = adjudication_response.json()
    assert gold["status"] == "gold"
    assert gold["evidence_block"]["locked"] is True
    assert (
        client.patch(
            f"/api/annotations/{gold['id']}", json={"status": "draft"}
        ).status_code
        == 409
    )
    assert client.delete(
        f"/api/annotations/{gold['id']}", params={"expected_revision": 1}
    ).status_code == 409
    adjudication_command = gold["evidence_block"]["last_command_group_key"]
    assert (
        client.post(
            f"/api/annotations/evidence-blocks/commands/{adjudication_command}/undo"
        ).status_code
        == 409
    )


def test_team_adjudication_requires_two_matching_reviewed_annotators(client):
    ctx = _setup_evidence_project(client, "evidence-team-adjudication")
    project_id = ctx["project"]["id"]
    document_id = ctx["document"]["id"]
    structure_id = ctx["structure"]["structure_version"]["id"]
    target_id = ctx["version"]["id"]
    guideline_id = ctx["guideline"]["id"]
    sentences = ctx["structure"]["sentences"]

    for username in ("alice", "bob"):
        assignment = client.post(
            f"/api/projects/{project_id}/assignments",
            json={
                "task_id": ctx["task"]["id"],
                "document_id": document_id,
                "annotator_id": username,
                "target_version_id": target_id,
                "guideline_version_id": guideline_id,
            },
        )
        assert assignment.status_code == 200, assignment.text

    alice = client.post(
        "/api/annotations",
        json=_block_payload(ctx, 0, 0, annotator_id="alice"),
    ).json()
    bob = client.post(
        "/api/annotations",
        json=_block_payload(ctx, 1, 1, annotator_id="bob"),
    ).json()
    manager_draft = client.post(
        "/api/annotations",
        json=_block_payload(ctx, 2, 2),
    ).json()
    coverage_url = (
        f"/api/projects/{project_id}/documents/{document_id}"
        "/evidence-review-coverage/mark-reviewed"
    )
    for username, sentence_index in (("alice", 0), ("bob", 1)):
        token = client._token_for(username)
        reviewed = client.post(
            coverage_url,
            json={
                "target_version_id": target_id,
                "structure_version_id": structure_id,
                "start_sentence_id": sentences[sentence_index]["id"],
                "end_sentence_id": sentences[sentence_index]["id"],
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert reviewed.status_code == 200, reviewed.text
    manager_reviewed = client.post(
        coverage_url,
        json={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": guideline_id,
            "start_sentence_id": sentences[2]["id"],
            "end_sentence_id": sentences[2]["id"],
        },
    )
    assert manager_reviewed.status_code == 200, manager_reviewed.text

    assignments = client.get(
        f"/api/projects/{project_id}/assignments"
    ).json()
    for assignment in assignments:
        finalized = client.patch(
            f"/api/projects/{project_id}/assignments/{assignment['id']}",
            json={"status": "submitted"},
        )
        assert finalized.status_code == 200, finalized.text

    adjudication_url = (
        f"/api/projects/{project_id}/documents/{document_id}/evidence-adjudication"
    )
    comparison = client.get(
        adjudication_url,
        params={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": guideline_id,
        },
    )
    assert comparison.status_code == 200, comparison.text
    assert {block["annotation_id"] for block in comparison.json()["blocks"]} == {
        alice["id"],
        bob["id"],
    }
    assert manager_draft["id"] not in {
        block["annotation_id"] for block in comparison.json()["blocks"]
    }

    one_annotator = client.post(
        adjudication_url,
        json={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": guideline_id,
            "strategy": "a",
            "source_annotation_ids": [alice["id"]],
        },
    )
    assert one_annotator.status_code == 422
    assert "at least two annotators" in one_annotator.json()["detail"]

    uncovered_custom = client.post(
        adjudication_url,
        json={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": guideline_id,
            "strategy": "custom",
            "source_annotation_ids": [alice["id"], bob["id"]],
            "start_sentence_id": sentences[0]["id"],
            "end_sentence_id": sentences[1]["id"],
        },
    )
    assert uncovered_custom.status_code == 422
    assert "reviewed coverage for every source annotator" in uncovered_custom.json()[
        "detail"
    ]

    other_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": project_id,
            "version_label": "evidence-v2",
            "markdown": "Changed rules.",
        },
    ).json()
    mismatched_guideline = client.post(
        adjudication_url,
        json={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": other_guideline["id"],
            "strategy": "union",
            "source_annotation_ids": [alice["id"], bob["id"]],
        },
    )
    assert mismatched_guideline.status_code == 422
    assert "matching, human, unlocked, reviewed" in mismatched_guideline.json()[
        "detail"
    ]

    union = client.post(
        adjudication_url,
        json={
            "target_version_id": target_id,
            "structure_version_id": structure_id,
            "guideline_version_id": guideline_id,
            "strategy": "union",
            "source_annotation_ids": [alice["id"], bob["id"]],
        },
    )
    assert union.status_code == 201, union.text
    assert union.json()["guideline_version_id"] == guideline_id
    assert union.json()["attributes"]["adjudication"]["source_annotation_ids"] == [
        alice["id"],
        bob["id"],
    ]


def test_adjudication_comparison_requires_complete_version_scope(client):
    ctx = _setup_evidence_project(client, "evidence-comparison-scope")
    response = client.get(
        (
            f"/api/projects/{ctx['project']['id']}/documents/"
            f"{ctx['document']['id']}/evidence-adjudication"
        ),
        params={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
        },
    )
    assert response.status_code == 422


def test_individual_solo_custom_gold_requires_explicit_reviewed_coverage(client, db):
    ctx = _setup_evidence_project(client, "evidence-solo-custom")
    project = db.get(Project, ctx["project"]["id"])
    project.workspace.kind = "individual"
    db.commit()
    sentences = ctx["structure"]["sentences"]
    coverage_url = (
        f"/api/projects/{project.id}/documents/{ctx['document']['id']}"
        "/evidence-review-coverage/mark-reviewed"
    )
    reviewed = client.post(
        coverage_url,
        json={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "guideline_version_id": ctx["guideline"]["id"],
            "start_sentence_id": sentences[0]["id"],
            "end_sentence_id": sentences[0]["id"],
        },
    )
    assert reviewed.status_code == 200, reviewed.text
    adjudication_url = (
        f"/api/projects/{project.id}/documents/{ctx['document']['id']}"
        "/evidence-adjudication"
    )
    outside_review = client.post(
        adjudication_url,
        json={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "guideline_version_id": ctx["guideline"]["id"],
            "strategy": "custom",
            "start_sentence_id": sentences[1]["id"],
            "end_sentence_id": sentences[1]["id"],
            "solo_gold": True,
        },
    )
    assert outside_review.status_code == 422
    inside_review = client.post(
        adjudication_url,
        json={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "guideline_version_id": ctx["guideline"]["id"],
            "strategy": "custom",
            "start_sentence_id": sentences[0]["id"],
            "end_sentence_id": sentences[0]["id"],
            "solo_gold": True,
        },
    )
    assert inside_review.status_code == 201, inside_review.text
    assert inside_review.json()["attributes"]["adjudication"] == {
        "strategy": "custom",
        "source_annotation_ids": [],
        "solo_gold": True,
    }


def test_public_patch_cannot_promote_human_annotation_to_gold(client):
    ctx = _setup_evidence_project(client, "evidence-public-provenance")
    created = client.post("/api/annotations", json=_block_payload(ctx, 0, 0)).json()
    response = client.patch(
        f"/api/annotations/{created['id']}",
        json={"status": "gold", "expected_revision": 1},
    )
    assert response.status_code == 422


def test_public_patch_and_delete_keep_legacy_model_and_gold_rows_read_only(
    client, db
):
    ctx = _setup_evidence_project(client, "evidence-read-only-provenance")
    user = db.query(User).filter(User.username == "tester").one()
    rows = [
        Annotation(
            project_id=ctx["project"]["id"],
            document_id=ctx["document"]["id"],
            annotation_type="entity",
            label="legacy",
            start_offset=0,
            end_offset=5,
            text_span="Alpha",
            source="model",
            status="draft",
            annotator_user_id=user.id,
            annotator_id=user.username,
        ),
        Annotation(
            project_id=ctx["project"]["id"],
            document_id=ctx["document"]["id"],
            annotation_type="entity",
            label="legacy",
            start_offset=0,
            end_offset=5,
            text_span="Alpha",
            source="human",
            status="gold",
            annotator_user_id=user.id,
            annotator_id=user.username,
        ),
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        assert client.patch(
            f"/api/annotations/{row.id}", json={"label": "changed"}
        ).status_code == 409
        assert client.delete(f"/api/annotations/{row.id}").status_code == 409


def test_target_version_rows_have_no_mutation_route(client):
    ctx = _setup_evidence_project(client, "evidence-immutable-target")
    response = client.patch(
        f"/api/projects/{ctx['project']['id']}/evidence-targets/"
        f"{ctx['target']['id']}/versions/{ctx['version']['id']}",
        json={"text": "changed"},
    )
    assert response.status_code in (404, 405)


def test_target_versions_reject_direct_orm_mutation(client, db):
    ctx = _setup_evidence_project(client, "evidence-immutable-orm")
    version = db.get(EvidenceTargetVersion, ctx["version"]["id"])
    version.text = "Mutated text"
    with pytest.raises(ImmutableEvidenceTargetVersionError):
        db.commit()
    db.rollback()


def test_all_evidence_mutations_share_the_stable_target_version_lock():
    from al_medlit.evidence import service
    from al_medlit.submission import service as submission_service

    assert "with_for_update" in service._lock_evidence_scope.__code__.co_names
    for function in (
        service._validate_block_policy,
        service.mark_reviewed,
        service._reopen_coverage_ordinals,
        service.assignment_has_full_review_coverage,
    ):
        assert "_lock_evidence_scope" in function.__code__.co_names
    assert "_lock_evidence_scope" in submission_service.create_submission.__code__.co_names


def test_maximum_weight_iou_matching_is_global(monkeypatch):
    left = [(0, 0), (1, 1)]
    right = [(2, 2), (3, 3)]
    weights = {
        (left[0], right[0]): 0.90,
        (left[0], right[1]): 0.80,
        (left[1], right[0]): 0.85,
        (left[1], right[1]): 0.10,
    }
    monkeypatch.setattr(
        evidence_metrics,
        "span_iou",
        lambda left_span, right_span: weights[(left_span, right_span)],
    )
    matching = evidence_metrics.maximum_weight_matching(left, right)
    assert {(left_index, right_index) for left_index, right_index, _weight in matching} == {
        (0, 1),
        (1, 0),
    }


def test_evidence_iaa_scores_only_shared_reviewed_coverage(client, db):
    ctx = _setup_evidence_project(client, "evidence-iaa")
    for username in ("alice", "bob"):
        assignment = client.post(
            f"/api/projects/{ctx['project']['id']}/assignments",
            json={
                "task_id": ctx["task"]["id"],
                "document_id": ctx["document"]["id"],
                "annotator_id": username,
                "target_version_id": ctx["version"]["id"],
            },
        )
        assert assignment.status_code == 200, assignment.text

    alice_payload = _block_payload(ctx, 0, 1, annotator_id="alice")
    bob_payload = _block_payload(ctx, 1, 2, annotator_id="bob")
    assert client.post("/api/annotations", json=alice_payload).status_code == 200
    assert client.post("/api/annotations", json=bob_payload).status_code == 200

    sentences = ctx["structure"]["sentences"]
    coverage_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/evidence-review-coverage/mark-reviewed"
    )
    review_body = {
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "start_sentence_id": sentences[0]["id"],
        "end_sentence_id": sentences[2]["id"],
    }
    for username in ("alice", "bob"):
        token = client._token_for(username)
        response = client.post(
            coverage_url,
            json=review_body,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text

    coverage_rows = db.query(EvidenceReviewCoverage).all()
    assert {row.guideline_version_id for row in coverage_rows} == {
        ctx["guideline"]["id"]
    }
    replacement_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": ctx["project"]["id"],
            "version_label": "evidence-v2",
            "markdown": "Replacement guidance.",
        },
    ).json()
    for assignment in db.query(TaskAssignment).filter(
        TaskAssignment.project_id == ctx["project"]["id"],
        TaskAssignment.target_version_id == ctx["version"]["id"],
    ):
        assignment.guideline_version_id = replacement_guideline["id"]
    db.commit()

    report = iaa_service.compute_iaa(
        db,
        project_id=ctx["project"]["id"],
        annotation_type="evidence_block",
        document_id=ctx["document"]["id"],
        target_version_id=ctx["version"]["id"],
        structure_version_id=ctx["structure"]["structure_version"]["id"],
        guideline_version_id=ctx["guideline"]["id"],
    )
    assert report.status == "ok"
    assert report.annotator_ids == ["alice", "bob"]
    pair = report.evidence_metrics.pairs[0]
    assert pair.reviewed_sentence_count == 3
    assert pair.sentence_f1 == 0.5
    assert pair.iou_f1 == {"0.25": 1.0, "0.50": 0.0, "0.75": 0.0}


def test_evidence_submission_is_assignment_scoped_and_requires_full_coverage(client):
    ctx = _setup_evidence_project(client, "evidence-scoped-submission")
    assignment_response = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
            "target_version_id": ctx["version"]["id"],
            "status": "in_progress",
        },
    )
    assert assignment_response.status_code == 200, assignment_response.text
    assignment = assignment_response.json()
    assert client.post(
        "/api/annotations",
        json=_block_payload(ctx, 0, 0, annotator_id="alice"),
    ).status_code == 200
    submission_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/submissions"
    )
    submit_body = {"assignment_id": assignment["id"], "annotator_id": "alice"}
    incomplete = client.post(submission_url, json=submit_body)
    assert incomplete.status_code == 422
    assert "full reviewed coverage" in incomplete.json()["detail"]

    sentences = ctx["structure"]["sentences"]
    alice_token = client._token_for("alice")
    reviewed = client.post(
        (
            f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
            "/evidence-review-coverage/mark-reviewed"
        ),
        json={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
            "start_sentence_id": sentences[0]["id"],
            "end_sentence_id": sentences[-1]["id"],
        },
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert reviewed.status_code == 200, reviewed.text
    submitted = client.post(submission_url, json=submit_body)
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["assignment_id"] == assignment["id"]
    assignments = client.get(
        f"/api/projects/{ctx['project']['id']}/assignments",
        params={"target_version_id": ctx["version"]["id"]},
    ).json()
    alice_assignment = next(item for item in assignments if item["id"] == assignment["id"])
    assert alice_assignment["status"] == "submitted"
    repeated = client.post(submission_url, json=submit_body)
    assert repeated.status_code == 409
    assert "final status" in repeated.json()["detail"]


def test_submitted_evidence_assignment_and_foreign_evidence_are_read_only(
    client,
    db,
):
    ctx = _setup_evidence_project(client, "evidence-submitted-read-only")
    assignment = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
            "target_version_id": ctx["version"]["id"],
        },
    ).json()
    alice_token = client._token_for("alice")
    alice = db.query(User).filter(User.username == "alice").one()
    membership = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == ctx["project"]["workspace_id"],
            WorkspaceMember.user_id == alice.id,
        )
        .one()
    )
    membership.role = "annotator"
    db.commit()
    alice_headers = {"Authorization": f"Bearer {alice_token}"}

    spanning = client.post(
        "/api/annotations",
        json=_block_payload(ctx, 0, 1),
        headers=alice_headers,
    ).json()
    trailing = client.post(
        "/api/annotations",
        json=_block_payload(ctx, 2, 2),
        headers=alice_headers,
    ).json()
    command_key = spanning["evidence_block"]["last_command_group_key"]
    submitted = client.patch(
        f"/api/projects/{ctx['project']['id']}/assignments/{assignment['id']}",
        json={"status": "submitted"},
    )
    assert submitted.status_code == 200, submitted.text

    evidence_payload = _block_payload(ctx, 0, 1)["evidence_block"]
    mutation_responses = [
        client.post(
            "/api/annotations",
            json=_block_payload(ctx, 0, 0),
            headers=alice_headers,
        ),
        client.patch(
            f"/api/annotations/{spanning['id']}",
            json={"expected_revision": 1, "evidence_block": evidence_payload},
            headers=alice_headers,
        ),
        client.delete(
            f"/api/annotations/{spanning['id']}",
            params={"expected_revision": 1},
            headers=alice_headers,
        ),
        client.post(
            "/api/annotations/evidence-blocks/merge",
            json={
                "annotation_ids": [spanning["id"], trailing["id"]],
                "expected_revisions": {
                    str(spanning["id"]): 1,
                    str(trailing["id"]): 1,
                },
            },
            headers=alice_headers,
        ),
        client.post(
            f"/api/annotations/{spanning['id']}/split",
            json={
                "expected_revision": 1,
                "split_before_sentence_id": ctx["structure"]["sentences"][1]["id"],
            },
            headers=alice_headers,
        ),
        client.post(
            f"/api/annotations/evidence-blocks/commands/{command_key}/undo",
            headers=alice_headers,
        ),
        client.post(
            f"/api/annotations/evidence-blocks/commands/{command_key}/redo",
            headers=alice_headers,
        ),
    ]
    review_body = {
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "start_sentence_id": ctx["structure"]["sentences"][0]["id"],
        "end_sentence_id": ctx["structure"]["sentences"][-1]["id"],
    }
    coverage_url = (
        f"/api/projects/{ctx['project']['id']}/documents/{ctx['document']['id']}"
        "/evidence-review-coverage"
    )
    mutation_responses.extend(
        [
            client.post(
                f"{coverage_url}/mark-reviewed",
                json=review_body,
                headers=alice_headers,
            ),
            client.post(
                f"{coverage_url}/reopen",
                json=review_body,
                headers=alice_headers,
            ),
        ]
    )
    assert {response.status_code for response in mutation_responses} == {403}

    # Existing read access remains available after submission.
    coverage = client.get(
        coverage_url,
        params={
            "target_version_id": ctx["version"]["id"],
            "structure_version_id": ctx["structure"]["structure_version"]["id"],
        },
        headers=alice_headers,
    )
    assert coverage.status_code == 200, coverage.text

    # A manager must use comparison/adjudication rather than edit another
    # annotator's evidence in place.
    evidence_payload["note"] = "manager correction"
    manager_mutations = [
        client.patch(
            f"/api/annotations/{spanning['id']}",
            json={"expected_revision": 1, "evidence_block": evidence_payload},
        ),
        client.delete(
            f"/api/annotations/{spanning['id']}",
            params={"expected_revision": 1},
        ),
        client.post(
            "/api/annotations/evidence-blocks/merge",
            json={
                "annotation_ids": [spanning["id"], trailing["id"]],
                "expected_revisions": {
                    str(spanning["id"]): 1,
                    str(trailing["id"]): 1,
                },
            },
        ),
        client.post(
            f"/api/annotations/{spanning['id']}/split",
            json={
                "expected_revision": 1,
                "split_before_sentence_id": ctx["structure"]["sentences"][1]["id"],
            },
        ),
        client.post(
            f"/api/annotations/evidence-blocks/commands/{command_key}/undo"
        ),
        client.post(
            f"/api/annotations/evidence-blocks/commands/{command_key}/redo"
        ),
    ]
    assert {response.status_code for response in manager_mutations} == {403}

    # Managers must use the review/correction workflow for every annotation type;
    # directly rewriting another annotator's source record is not auditable.
    legacy = Annotation(
        project_id=ctx["project"]["id"],
        document_id=ctx["document"]["id"],
        annotation_type="entity",
        label="legacy",
        start_offset=0,
        end_offset=5,
        text_span="Alpha",
        source="human",
        status="draft",
        annotator_user_id=alice.id,
        annotator_id=alice.username,
    )
    db.add(legacy)
    db.commit()
    assert client.delete(f"/api/annotations/{legacy.id}").status_code == 403

    client._token_for("bob")
    reassignment = client.patch(
        f"/api/projects/{ctx['project']['id']}/assignments/{assignment['id']}",
        json={"annotator_id": "bob"},
    )
    assert reassignment.status_code == 409
    assert client.delete(
        f"/api/projects/{ctx['project']['id']}/assignments/{assignment['id']}"
    ).status_code == 409
    preserved = next(
        item
        for item in client.get(
            f"/api/projects/{ctx['project']['id']}/assignments"
        ).json()
        if item["id"] == assignment["id"]
    )
    assert preserved["assignee_user_id"] == alice.id
    assert preserved["target_version_id"] == ctx["version"]["id"]
    assert preserved["structure_version_id"] == ctx["structure"]["structure_version"][
        "id"
    ]
    assert preserved["guideline_version_id"] == ctx["guideline"]["id"]


def test_only_open_evidence_assignment_statuses_authorize_mutations(client, db):
    ctx = _setup_evidence_project(client, "evidence-mutable-statuses")
    created = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
            "target_version_id": ctx["version"]["id"],
        },
    ).json()
    assignment = db.get(TaskAssignment, created["id"])
    alice = db.get(User, assignment.assignee_user_id)
    member = WorkspaceMember(
        workspace_id=ctx["project"]["workspace_id"],
        user_id=alice.id,
        role="annotator",
    )
    scope = {
        "project_id": ctx["project"]["id"],
        "document_id": ctx["document"]["id"],
        "annotation_type": "evidence_block",
        "target_version_id": ctx["version"]["id"],
        "structure_version_id": ctx["structure"]["structure_version"]["id"],
        "guideline_version_id": ctx["guideline"]["id"],
        "require_mutable_assignment": True,
    }
    for open_status in ("assigned", "in_progress", "blocked"):
        assignment.status = open_status
        db.flush()
        assert assert_task_assigned(db, alice, member, **scope).id == assignment.id
    for final_status in (
        "submitted",
        "adjudication_ready",
        "adjudicated",
        "completed",
    ):
        assignment.status = final_status
        db.flush()
        with pytest.raises(ForbiddenError):
            assert_task_assigned(db, alice, member, **scope)
    db.rollback()


def test_assignment_snapshot_and_workbench_keep_all_version_pins(client, db):
    ctx = _setup_evidence_project(client, "evidence-snapshot-version-pins")
    assignment = client.post(
        f"/api/projects/{ctx['project']['id']}/assignments",
        json={
            "task_id": ctx["task"]["id"],
            "document_id": ctx["document"]["id"],
            "annotator_id": "alice",
            "target_version_id": ctx["version"]["id"],
        },
    ).json()
    replacement_guideline = client.post(
        "/api/guidelines",
        json={
            "project_id": ctx["project"]["id"],
            "version_label": "evidence-v2",
            "markdown": "New active rules that must not replace the assignment pin.",
        },
    ).json()
    rebuilt = client.post(
        f"/api/documents/{ctx['document']['id']}/structure/rebuild",
        json={"activate": False},
    ).json()
    second_target = client.post(
        f"/api/projects/{ctx['project']['id']}/evidence-targets",
        json={
            "task_id": ctx["task"]["id"],
            "key": "harm-snapshot",
            "name": "Treatment harm",
            "initial_version": {"text": "Does treatment cause harm?"},
        },
    ).json()
    second_target_version_id = second_target["versions"][0]["id"]

    alice = db.query(User).filter(User.username == "alice").one()
    project = db.get(Project, ctx["project"]["id"])
    document = project.documents[0]
    assignment_row = db.get(TaskAssignment, assignment["id"])

    def add_block(*, target_id, structure, guideline_id, sentence_index):
        sentence = structure["sentences"][sentence_index]
        annotation = Annotation(
            project_id=project.id,
            document_id=document.id,
            annotation_type="evidence_block",
            label="evidence_block",
            start_offset=sentence["start_offset"],
            end_offset=sentence["end_offset"],
            text_span=document.text[sentence["start_offset"] : sentence["end_offset"]],
            source="human",
            status="draft",
            annotator_user_id=alice.id,
            annotator_id=alice.username,
            guideline_version_id=guideline_id,
            structure_version_id=structure["structure_version"]["id"],
            evidence={},
            attributes={},
        )
        annotation.evidence_block = EvidenceBlockAnnotation(
            structure_version_id=structure["structure_version"]["id"],
            target_version_id=target_id,
            start_sentence_id=sentence["id"],
            end_sentence_id=sentence["id"],
            start_sentence_ordinal=sentence["ordinal"],
            end_sentence_ordinal=sentence["ordinal"],
            labels=[],
            boundary_policy="sentence",
            revision=1,
            locked=False,
        )
        db.add(annotation)
        db.flush()
        return annotation.id

    matching_id = add_block(
        target_id=ctx["version"]["id"],
        structure=ctx["structure"],
        guideline_id=ctx["guideline"]["id"],
        sentence_index=0,
    )
    add_block(
        target_id=second_target_version_id,
        structure=ctx["structure"],
        guideline_id=ctx["guideline"]["id"],
        sentence_index=1,
    )
    add_block(
        target_id=ctx["version"]["id"],
        structure=rebuilt,
        guideline_id=ctx["guideline"]["id"],
        sentence_index=1,
    )
    add_block(
        target_id=ctx["version"]["id"],
        structure=ctx["structure"],
        guideline_id=replacement_guideline["id"],
        sentence_index=2,
    )
    db.commit()

    snapshot = build_snapshot(
        db,
        project=project,
        document=document,
        annotator_user_id=alice.id,
        annotator_id=alice.username,
        assignment=assignment_row,
    )
    assert [annotation["id"] for annotation in snapshot["annotations"]] == [matching_id]

    workbench = client.get(
        f"/api/annotation-workbench/documents/{ctx['document']['id']}",
        headers={"Authorization": f"Bearer {client._token_for('alice')}"},
    ).json()
    assert workbench["active_guideline"]["id"] == replacement_guideline["id"]
    assert workbench["assignments"][0]["guideline_version_id"] == ctx["guideline"][
        "id"
    ]
    assert workbench["guideline_versions_by_id"][str(ctx["guideline"]["id"])][
        "markdown"
    ] == ctx["guideline"]["markdown"]
