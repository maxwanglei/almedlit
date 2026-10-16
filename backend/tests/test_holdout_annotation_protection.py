"""Protected source identities cannot enter active learning or assisted annotation."""

import pytest
from test_legacy_training_holdouts import _prepared_source
from test_workflow_ingestion_selection import _create_feedback_set, _selection_context

from al_medlit.workflow import models
from al_medlit.workflow.services.common import _canonical_hash


def _run(client, db, context, split_mode, *, strategy="all", feedback_id=None):
    split_id = context["split_map"]["id"]
    if split_mode == "alternate":
        alternate = client.post("/api/datasets/split-maps", json={
            "project_id": context["project"]["id"],
            "dataset_version_id": context["dataset_version"]["id"],
            "name": "Alternate map without protected flags", "strategy": "manual",
            "assignments": context["split_map"]["assignments"], "protected_splits": [],
        })
        assert alternate.status_code == 201, alternate.text
        split_id = alternate.json()["id"]
    payload = {
        "project_id": context["project"]["id"],
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "split_map_id": split_id, "strategy": strategy,
        "feedback_set_version_id": feedback_id, "seed": 19,
    }
    if split_mode == "legacy-no-map":
        # New runs require a map. Historical rows can still omit one and must
        # receive the same global source-identity protection when materialized.
        run = models.SelectionRun(**{**payload, "split_map_id": None})
        db.add(run)
        db.commit()
        return {"id": run.id}
    response = client.post("/api/selection-runs", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _feedback(client, context):
    return _create_feedback_set(client, context, [
        {"dataset_item_id": item["id"], "output": "yes", "score": 1.0}
        for item in context["item_by_key"].values()
    ])


@pytest.mark.parametrize("strategy", ["all", "random", "uncertainty"])
@pytest.mark.parametrize("split_mode", ["own", "alternate", "legacy-no-map"])
def test_automatic_selection_excludes_holdouts_independently_of_selected_map(
    client, db, strategy, split_mode,
):
    context = _selection_context(client, "global-selection-protection")
    feedback_id = _feedback(client, context)["id"] if strategy == "uncertainty" else None
    run = _run(client, db, context, split_mode, strategy=strategy, feedback_id=feedback_id)
    response = client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize",
    )
    assert response.status_code == 201, response.text
    assert {item["dataset_item_id"] for item in response.json()["items"]} == {
        item["id"] for key, item in context["item_by_key"].items() if key != "d"
    }


@pytest.mark.parametrize("split_mode", ["own", "alternate", "legacy-no-map"])
def test_manual_selection_rejects_holdouts_independently_of_selected_map(client, db, split_mode):
    context = _selection_context(client, "manual-selection-protection")
    run = _run(client, db, context, split_mode)
    response = client.post("/api/selection-runs/sets", json={
        "project_id": context["project"]["id"], "selection_run_id": run["id"],
        "items": [{"dataset_item_id": context["item_by_key"]["d"]["id"], "rank": 1}],
    })
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.SelectionSetVersion).count() == 0


def _historical_selection(client, db, context):
    run = _run(client, db, context, "own")
    items = [{"dataset_item_id": context["item_by_key"]["d"]["id"], "rank": 1}]
    selection = models.SelectionSetVersion(
        project_id=context["project"]["id"], selection_run_id=run["id"],
        version_number=1, items=items, item_count=1, content_hash=_canonical_hash(items),
    )
    db.add(selection)
    db.commit()
    return selection.id


@pytest.mark.parametrize("mode", ["full_dataset", "historical-selection"])
@pytest.mark.parametrize("policy", ["immediate_suggestions", "reveal_after_first_pass"])
def test_assisted_round_rejects_protected_items_before_persisting_round(
    client, db, mode, policy,
):
    context = _selection_context(client, "assisted-round-protection")
    feedback = _feedback(client, context)
    payload = {
        "project_id": context["project"]["id"], "name": "Protected assisted annotation",
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "feedback_set_version_id": feedback["id"], "assistance_policy": policy,
        "reannotation_mode": "full_dataset" if mode == "full_dataset" else "targeted_subset",
        "open_to_all_annotators": True,
    }
    if mode == "historical-selection":
        payload["selection_set_version_id"] = _historical_selection(client, db, context)
    response = client.post("/api/rounds", json=payload)
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.AnnotationRound).count() == 0
    assert db.query(models.RoundItem).count() == 0


def test_blind_full_dataset_annotation_keeps_protected_evaluation_items_available(client):
    context = _selection_context(client, "blind-evaluation-annotation")
    response = client.post("/api/rounds", json={
        "project_id": context["project"]["id"], "name": "Blind holdout labeling",
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "assistance_policy": "blind", "reannotation_mode": "full_dataset",
        "open_to_all_annotators": True,
    })
    assert response.status_code == 201, response.text
    items = client.get(
        f"/api/rounds/{response.json()['id']}/items",
        params={"project_id": context["project"]["id"]},
    )
    assert items.status_code == 200, items.text
    assert context["item_by_key"]["d"]["id"] in {
        item["dataset_item_id"] for item in items.json()
    }


@pytest.mark.parametrize(("policy", "operation"), [
    ("immediate_suggestions", operation)
    for operation in (
        "open", "work-items", "decision", "submission", "feedback-exposure", "feedback-reveal",
        "feedback-decision",
    )
] + [
    ("reveal_after_first_pass", operation)
    for operation in ("decision", "feedback-exposure", "feedback-reveal")
])
def test_historical_assisted_round_cannot_process_protected_items(
    client, db, policy, operation,
):
    context = _selection_context(client, "historical-assisted-round")
    feedback = _feedback(client, context)
    project_id = context["project"]["id"]
    actor_id = client.get("/api/auth/me").json()["user"]["id"]
    annotation_round = models.AnnotationRound(
        project_id=project_id, name="Legacy assisted round", sequence=1,
        dataset_version_id=context["dataset_version"]["id"],
        task_version_id=context["task_version"]["id"], feedback_set_version_id=feedback["id"],
        assistance_policy=policy, reannotation_mode="full_dataset", open_to_all_annotators=True,
        status="draft" if operation == "open" else "open", created_by_user_id=actor_id,
    )
    db.add(annotation_round)
    db.flush()
    item = models.RoundItem(
        project_id=project_id, annotation_round_id=annotation_round.id,
        dataset_item_id=context["item_by_key"]["d"]["id"],
        selection_reason={"strategy": "all"}, metadata_={},
    )
    db.add(item)
    db.flush()
    seeded_decision = None
    if operation == "submission" or (
        policy == "reveal_after_first_pass"
        and operation in {"feedback-exposure", "feedback-reveal"}
    ):
        seeded_decision = models.RoundAnnotationDecision(
            project_id=project_id, round_item_id=item.id, annotator_user_id=actor_id,
            output="yes", decision_kind="annotation", is_initial_checkpoint=True,
            content_hash=_canonical_hash({"output": "yes", "round_item_id": item.id}),
        )
        db.add(seeded_decision)
    seeded_exposure = None
    if operation == "feedback-decision":
        candidate = db.query(models.FeedbackCandidate).filter_by(
            feedback_set_version_id=feedback["id"], dataset_item_id=item.dataset_item_id,
        ).one()
        seeded_exposure = models.FeedbackExposure(
            project_id=project_id, round_item_id=item.id, feedback_candidate_id=candidate.id,
            user_id=actor_id, exposure_mode=policy, context={},
        )
        db.add(seeded_exposure)
    db.commit()
    if operation == "open":
        response = client.post(
            f"/api/rounds/{annotation_round.id}/transition", params={"project_id": project_id},
            json={"status": "open"},
        )
    elif operation == "work-items":
        response = client.get(
            f"/api/rounds/{annotation_round.id}/work-items", params={"project_id": project_id},
        )
    elif operation == "decision":
        response = client.post("/api/rounds/decisions", json={
            "project_id": project_id, "round_item_id": item.id, "output": "yes",
            "is_initial_checkpoint": True,
        })
    elif operation == "submission":
        response = client.post("/api/rounds/submissions", json={
            "project_id": project_id, "annotation_round_id": annotation_round.id,
            "decision_ids": [seeded_decision.id],
        })
    elif operation == "feedback-exposure":
        candidate = db.query(models.FeedbackCandidate).filter_by(
            feedback_set_version_id=feedback["id"], dataset_item_id=item.dataset_item_id,
        ).one()
        response = client.post("/api/feedback-runs/exposures", json={
            "project_id": project_id, "round_item_id": item.id,
            "feedback_candidate_id": candidate.id, "exposure_mode": policy,
        })
    elif operation == "feedback-decision":
        response = client.post("/api/feedback-runs/decisions", json={
            "project_id": project_id, "feedback_exposure_id": seeded_exposure.id,
            "decision": "accepted",
        })
    else:
        response = client.post(
            f"/api/rounds/{annotation_round.id}/items/{item.id}/feedback-reveal",
            json={"project_id": project_id},
        )
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.RoundAnnotationDecision).count() == int(seeded_decision is not None)
    assert db.query(models.RoundSubmission).count() == 0
    assert db.query(models.FeedbackExposure).count() == int(seeded_exposure is not None)
    assert db.query(models.FeedbackDecision).count() == 0
    if operation == "open":
        db.refresh(annotation_round)
        assert annotation_round.status == "draft"


def _prepared_context(client, db, source_scope="raw"):
    project, task, source, _labels, prepared, _payload, source_assignments = _prepared_source(
        client, db,
    )
    split = db.get(models.SplitMap, prepared["training_dataset_version"]["split_map_id"])
    if source_scope == "prepared":
        source = {"id": prepared["training_dataset_version"]["dataset_version_id"]}
        assignments = split.assignments
    else:
        assignments = source_assignments
    items = db.query(models.DatasetItem).filter_by(dataset_version_id=source["id"]).all()
    context = {
        "project": project, "task_version": task, "dataset_version": source,
        "item_by_key": {item.stable_key: {"id": item.id} for item in items},
        "split_map": {"id": split.id, "assignments": assignments},
    }
    held_out_ids = {item.id for item in items if assignments[item.stable_key] == "test"}
    assert held_out_ids
    return context, held_out_ids


@pytest.mark.parametrize("source_scope", ["raw", "prepared"])
def test_prepared_holdout_ancestry_protects_selection_without_protected_map_flags(
    client, db, source_scope,
):
    context, held_out_ids = _prepared_context(client, db, source_scope)
    run = _run(client, db, context, "alternate")
    response = client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize",
    )
    assert response.status_code == 201, response.text
    assert {item["dataset_item_id"] for item in response.json()["items"]} == {
        item["id"] for item in context["item_by_key"].values()
    } - held_out_ids


def test_prepared_holdouts_do_not_exclude_another_tasks_selection_items(client, db):
    context, held_out_ids = _prepared_context(client, db)
    task = context["task_version"]
    other = client.post("/api/tasks/versions", json={
        "project_id": context["project"]["id"], "task_definition_id": task["task_definition_id"],
        "task_kind": task["task_kind"], "input_schema": task["input_schema"],
        "output_schema": {"type": "string", "enum": ["yes", "no", "unsure"]},
    })
    assert other.status_code == 201, other.text
    context["task_version"] = other.json()
    run = _run(client, db, context, "alternate")
    response = client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize",
    )
    assert response.status_code == 201, response.text
    selected_ids = {item["dataset_item_id"] for item in response.json()["items"]}
    assert selected_ids == {item["id"] for item in context["item_by_key"].values()}
    assert held_out_ids <= selected_ids


def test_assisted_round_rejects_raw_items_protected_through_prepared_ancestry(client, db):
    context, _held_out_ids = _prepared_context(client, db)
    feedback = _feedback(client, context)
    response = client.post("/api/rounds", json={
        "project_id": context["project"]["id"], "name": "Assisted prepared source",
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "feedback_set_version_id": feedback["id"], "assistance_policy": "immediate_suggestions",
        "reannotation_mode": "full_dataset", "open_to_all_annotators": True,
    })
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.AnnotationRound).count() == 0


def _regrouped_holdout_context(client, db):
    context = _selection_context(client, "regrouped-holdout-protection")
    items = [
        {
            "stable_key": key,
            "group_key": "revised-shared-group" if key == "d" else item["group_key"],
            "payload": item["payload"],
        }
        for key, item in context["item_by_key"].items()
    ]
    items.append({
        "stable_key": "new-sibling", "group_key": "revised-shared-group",
        "payload": {"id": "new-sibling", "patient": "new", "text": "New heldout article row"},
    })
    source = client.post("/api/datasets/versions", json={
        "project_id": context["project"]["id"],
        "dataset_id": context["dataset_version"]["dataset_id"],
        "source_revision": "regrouped-revision", "source_format": "jsonl", "items": items,
    })
    assert source.status_code == 201, source.text
    context["dataset_version"] = source.json()
    context["item_by_key"] = {
        item.stable_key: {"id": item.id}
        for item in db.query(models.DatasetItem).filter_by(dataset_version_id=source.json()["id"])
    }
    assignments = {key: "pool" for key in context["item_by_key"]}
    alternate = client.post("/api/datasets/split-maps", json={
        "project_id": context["project"]["id"], "dataset_version_id": source.json()["id"],
        "name": "Regrouped all-pool map", "strategy": "manual", "assignments": assignments,
        "protected_splits": [],
    })
    assert alternate.status_code == 201, alternate.text
    context["split_map"] = alternate.json()
    return context


def test_automatic_selection_protects_new_siblings_of_regrouped_holdout(client, db):
    context = _regrouped_holdout_context(client, db)
    run = _run(client, db, context, "own")
    response = client.post(
        f"/api/projects/{context['project']['id']}/selection-runs/{run['id']}/materialize",
    )
    assert response.status_code == 201, response.text
    assert {item["dataset_item_id"] for item in response.json()["items"]} == {
        item["id"] for key, item in context["item_by_key"].items()
        if key not in {"d", "new-sibling"}
    }


def test_manual_selection_of_only_new_sibling_keeps_regrouped_holdout_protection(client, db):
    context = _regrouped_holdout_context(client, db)
    run = _run(client, db, context, "own")
    response = client.post("/api/selection-runs/sets", json={
        "project_id": context["project"]["id"], "selection_run_id": run["id"],
        "items": [{"dataset_item_id": context["item_by_key"]["new-sibling"]["id"], "rank": 1}],
    })
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.SelectionSetVersion).count() == 0


def test_assisted_subset_of_only_new_sibling_keeps_regrouped_holdout_protection(client, db):
    context = _regrouped_holdout_context(client, db)
    feedback = _feedback(client, context)
    subset_context = {**context, "item_by_key": {"d": context["item_by_key"]["new-sibling"]}}
    selection_id = _historical_selection(client, db, subset_context)
    response = client.post("/api/rounds", json={
        "project_id": context["project"]["id"], "name": "Assisted holdout sibling",
        "dataset_version_id": context["dataset_version"]["id"],
        "task_version_id": context["task_version"]["id"],
        "selection_set_version_id": selection_id, "feedback_set_version_id": feedback["id"],
        "assistance_policy": "immediate_suggestions", "reannotation_mode": "targeted_subset",
        "open_to_all_annotators": True,
    })
    assert response.status_code == 422, response.text
    assert "protect" in response.json()["detail"].lower()
    assert db.query(models.AnnotationRound).count() == 0
