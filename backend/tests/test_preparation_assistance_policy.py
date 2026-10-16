"""Assisted review can supply training labels but cannot create evaluation data."""

import pytest
from test_training_preparation import _context, _prepare, _preview, _records, _request, _source

from al_medlit.auth.models import User
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels
from al_medlit.workflow.services.common import _canonical_hash


def _submitted_labels(client, db, project, task, source, policy, kind):
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    annotation_round = models.AnnotationRound(
        project_id=project["id"], name="Submitted review", sequence=1,
        dataset_version_id=source["id"], task_version_id=task["id"],
        assistance_policy=policy, reannotation_mode="full_dataset", status="open",
        annotator_user_ids=[actor.id], created_by_user_id=actor.id,
    )
    db.add(annotation_round)
    db.flush()
    decisions = []
    for item in db.query(models.DatasetItem).filter_by(dataset_version_id=source["id"]):
        round_item = models.RoundItem(
            project_id=project["id"], annotation_round_id=annotation_round.id,
            dataset_item_id=item.id,
        )
        db.add(round_item)
        db.flush()
        decision = models.RoundAnnotationDecision(
            project_id=project["id"], round_item_id=round_item.id,
            annotator_user_id=actor.id, output="yes",
            decision_kind="adjudication" if kind == "adjudicated" else "annotation",
            content_hash=_canonical_hash({"item_id": item.id, "output": "yes"}),
        )
        db.add(decision)
        db.flush()
        decisions.append(decision.id)
    submission = models.RoundSubmission(
        project_id=project["id"], annotation_round_id=annotation_round.id,
        annotator_user_id=actor.id, sequence=1, decision_ids=decisions,
        content_hash=_canonical_hash(decisions),
    )
    db.add(submission)
    db.commit()
    if kind == "round":
        return {"annotation_round_id": annotation_round.id, "submission_ids": [submission.id]}
    label_set = labels.create_round_label_set(
        db, annotation_round.id,
        schemas.RoundLabelSetCreate(
            project_id=project["id"], name="Submitted labels",
            source_kind="human" if kind in {"descendant", "decision-provenance"} else kind,
            submission_ids=[submission.id], publication_mode="submitted_snapshot",
        ),
        actor,
    )
    if kind == "descendant":
        label_set = labels.create_label_set_version(
            db, schemas.LabelSetVersionCreate(
                project_id=project["id"], dataset_version_id=source["id"],
                task_version_id=task["id"], name="Copied labels", source_kind="imported",
                parent_version_id=label_set.id, labels=label_set.labels,
            ), actor,
        )
    elif kind == "decision-provenance":
        label_set = models.LabelSetVersion(
            project_id=project["id"], dataset_version_id=source["id"],
            task_version_id=task["id"], name="Historical composed labels", version_number=1,
            source_kind="composed", composition_policy="replace", labels=label_set.labels,
            label_count=label_set.label_count, content_hash=_canonical_hash(label_set.labels),
            source_decision_ids=decisions,
        )
        db.add(label_set)
        db.commit()
    return {"label_set_version_id": label_set.id}


@pytest.mark.parametrize("policy", ["immediate_suggestions", "reveal_after_first_pass"])
@pytest.mark.parametrize(
    "kind", ["round", "human", "adjudicated", "descendant", "decision-provenance"],
)
def test_assisted_labels_alone_cannot_create_protected_test_split(client, db, policy, kind):
    project, task = _context(client)
    source = _source(client, project, "Reviewed", _records("reviewed", 5))
    reference = _submitted_labels(client, db, project, task, source, policy, kind)
    payload = _request(task)
    payload["sources"] = [{"dataset_version_id": source["id"], **reference}]
    preview = _preview(client, project, payload)
    assert not preview["ready"]
    assert preview["split_counts"]["test"] == 0
    assert any(issue["code"] == "empty_split" for issue in preview["issues"])
    response = client.post(f"/api/projects/{project['id']}/training-datasets/prepare", json=payload)
    assert response.status_code == 422, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 0


@pytest.mark.parametrize("kind", ["round", "human", "adjudicated"])
def test_blind_labels_can_supply_new_protected_test_items(client, db, kind):
    project, task = _context(client)
    source = _source(client, project, "Blind", _records("blind", 5))
    reference = _submitted_labels(client, db, project, task, source, "blind", kind)
    payload = _request(task)
    payload["sources"] = [{"dataset_version_id": source["id"], **reference}]
    preview = _preview(client, project, payload)
    assert preview["ready"], preview["issues"]
    assert preview["split_counts"]["test"] > 0


@pytest.mark.parametrize("kind", ["round", "human"])
def test_assisted_labels_can_be_combined_with_independent_test_data(client, db, kind):
    project, task = _context(client)
    assisted = _source(client, project, "Reviewed", _records("reviewed", 5))
    independent = _source(client, project, "Independent", _records("independent", 3))
    reference = _submitted_labels(
        client, db, project, task, assisted, "immediate_suggestions", kind,
    )
    payload = _request(task, independent)
    payload["sources"].append({"dataset_version_id": assisted["id"], **reference})
    result = _prepare(client, project, payload)
    version = result["training_dataset_version"]
    split = db.get(models.SplitMap, version["split_map_id"])
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    assisted_count = 0
    for key, provenance in generated.provenance["items"].items():
        reviewed = any(
            origin["dataset_version_id"] == assisted["id"] for origin in provenance["origins"]
        )
        assert provenance["model_assisted"] is reviewed
        if reviewed:
            assisted_count += 1
            assert split.assignments[key] in {"train", "validation"}
    assert assisted_count == 5
    assert result["preview"]["split_counts"]["test"] > 0

    # Reusing a prepared input snapshot for another compatible task must keep
    # its assistance provenance even though the new labels say "imported".
    definition = client.post("/api/tasks", json={
        "project_id": project["id"], "key": "classify_again", "name": "Another classification",
    })
    assert definition.status_code == 201, definition.text
    new_task_response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": definition.json()["id"],
        "task_kind": "classification", "input_schema": task["input_schema"],
        "output_schema": task["output_schema"],
    })
    assert new_task_response.status_code == 201, new_task_response.text
    new_task = new_task_response.json()
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    copied_labels = labels.create_label_set_version(
        db, schemas.LabelSetVersionCreate(
            project_id=project["id"], dataset_version_id=generated.id,
            task_version_id=new_task["id"], name="New task labels", source_kind="imported",
            labels=db.get(models.LabelSetVersion, version["label_set_version_ids"][0]).labels,
        ), actor,
    )
    legacy = client.post("/api/datasets/training-versions/compose", json={
        "project_id": project["id"], "name": "Legacy reused snapshot",
        "dataset_version_id": generated.id, "task_version_id": new_task["id"],
        "input_field": "text", "label_set_version_id": copied_labels.id,
        "train_percent": 80, "validation_percent": 10,
    })
    assert legacy.status_code == 201, legacy.text
    legacy_split = db.get(models.SplitMap, legacy.json()["split_map_id"])
    for key, provenance in generated.provenance["items"].items():
        if provenance["model_assisted"]:
            assert legacy_split.assignments[key] != "test"
    reused = _request(new_task, name="Reused snapshot", idempotency_key="reuse")
    reused["sources"] = [{
        "dataset_version_id": generated.id, "label_set_version_id": copied_labels.id,
    }]
    second = _prepare(client, project, reused)
    second_version = second["training_dataset_version"]
    second_split = db.get(models.SplitMap, second_version["split_map_id"])
    second_generated = db.get(models.DatasetVersion, second_version["dataset_version_id"])
    for key, provenance in second_generated.provenance["items"].items():
        if provenance["model_assisted"]:
            assert second_split.assignments[key] != "test"


def test_assisted_article_siblings_keep_assistance_provenance(client, db):
    project, task = _context(client)
    assisted_records = _records("reviewed", 3)
    assisted_records[0]["metadata"] = {"pmid": "12345"}
    assisted = _source(client, project, "Reviewed", assisted_records)
    independent_records = _records("independent", 3)
    independent_records[0]["metadata"] = {"pmid": "12345"}
    independent = _source(client, project, "Independent", independent_records)
    reference = _submitted_labels(
        client, db, project, task, assisted, "immediate_suggestions", "round",
    )
    payload = _request(task, independent)
    payload["sources"].append({"dataset_version_id": assisted["id"], **reference})
    result = _prepare(client, project, payload)
    version = result["training_dataset_version"]
    split = db.get(models.SplitMap, version["split_map_id"])
    generated = db.get(models.DatasetVersion, version["dataset_version_id"])
    sibling = next(
        (key, provenance) for key, provenance in generated.provenance["items"].items()
        if any(
            origin["dataset_version_id"] == independent["id"]
            and origin["stable_key"] == "independent-0"
            for origin in provenance["origins"]
        )
    )
    assert sibling[1]["model_assisted"] is True
    assert split.assignments[sibling[0]] != "test"


@pytest.mark.parametrize("policy", ["blind", "immediate_suggestions", "reveal_after_first_pass"])
@pytest.mark.parametrize("kind", ["human", "descendant"])
def test_legacy_composition_respects_assisted_label_ancestry(client, db, policy, kind):
    project, task = _context(client)
    source = _source(client, project, "Reviewed", [
        {"key": f"reviewed-{index}", "text": f"Reviewed paper {index}"} for index in range(5)
    ])
    reference = _submitted_labels(client, db, project, task, source, policy, kind)
    response = client.post("/api/datasets/training-versions/compose", json={
        "project_id": project["id"], "name": "Legacy reviewed training",
        "dataset_version_id": source["id"], "task_version_id": task["id"],
        "input_field": "text", "train_percent": 80, "validation_percent": 10, **reference,
    })
    if policy == "blind":
        assert response.status_code == 201, response.text
    else:
        assert response.status_code == 422, response.text
        assert "protected test" in response.json()["detail"]
        assert db.query(models.TrainingDatasetVersion).count() == 0
