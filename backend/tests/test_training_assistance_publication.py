import pytest
from test_preparation_assistance_policy import _submitted_labels
from test_training_preparation import _context, _prepare, _records, _request, _source

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ValidationError
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workflow.services.preparation import PREPARATION_KIND

ASSISTED_POLICIES = ["immediate_suggestions", "reveal_after_first_pass"]
LABEL_KINDS = ["human", "adjudicated", "descendant", "decision-provenance"]


def _actor(client, db):
    return db.get(User, client.get("/api/auth/me").json()["user"]["id"])


def _imported_labels(client, db, project, task, source, name, keys, *, parent=None):
    return labels.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=project["id"], dataset_version_id=source["id"],
            task_version_id=task["id"], name=name, source_kind="imported",
            parent_version_id=parent, labels={key: "yes" for key in keys},
        ),
        _actor(client, db),
    )


def _reviewed_source(client, db, policy, kind="human", *, records=None):
    project, task = _context(client)
    records = records or [
        {"key": f"reviewed-{index}", "text": f"Reviewed paper {index}"}
        for index in range(5)
    ]
    source = _source(client, project, "Reviewed", records)
    reference = _submitted_labels(client, db, project, task, source, policy, kind)
    label_set = db.get(models.LabelSetVersion, reference["label_set_version_id"])
    return project, task, source, label_set


def _publish(client, project, task, source, label_sets, *, assignments=None,
             protected_splits=None, composition=None, split_map_id=None):
    assignments = assignments or {
        f"reviewed-{index}": "test" if index == 0 else "validation" if index == 1 else "train"
        for index in range(5)
    }
    if split_map_id is None:
        split = client.post("/api/datasets/split-maps", json={
            "project_id": project["id"], "dataset_version_id": source["id"],
            "name": "Manual split", "strategy": "manual", "assignments": assignments,
            "protected_splits": protected_splits or ["test"],
        })
        assert split.status_code == 201, split.text
        split_map_id = split.json()["id"]
    return client.post("/api/datasets/training-versions", json={
        "project_id": project["id"], "dataset_version_id": source["id"],
        "task_version_id": task["id"], "name": "Direct reviewed training",
        "split_map_id": split_map_id,
        "label_set_version_ids": [label_set.id for label_set in label_sets],
        "composition": composition or [], "preprocessing": {"input_field": "text"},
    })


def _assert_rejected_without_publication(response, db):
    assert response.status_code == 422, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 0
    assert db.query(models.TrainingDataset).count() == 0


@pytest.mark.parametrize("policy", ["blind", *ASSISTED_POLICIES])
@pytest.mark.parametrize("kind", LABEL_KINDS)
def test_direct_publication_requires_blind_protected_labels(client, db, policy, kind):
    project, task, source, reviewed = _reviewed_source(client, db, policy, kind)
    response = _publish(client, project, task, source, [reviewed])
    if policy == "blind":
        assert response.status_code == 201, response.text
        assert db.query(models.TrainingDatasetVersion).count() == 1
        assert db.query(models.TrainingDataset).count() == 1
    else:
        _assert_rejected_without_publication(response, db)


@pytest.mark.parametrize("policy", ASSISTED_POLICIES)
@pytest.mark.parametrize("kind", ["human", "adjudicated"])
def test_direct_publication_accepts_assisted_learning_with_independent_test_labels(
    client, db, policy, kind,
):
    project, task, source, reviewed = _reviewed_source(client, db, policy, kind)
    learning = _imported_labels(
        client, db, project, task, source, "Assisted learning subset",
        [f"reviewed-{index}" for index in range(1, 5)], parent=reviewed.id,
    )
    independent = _imported_labels(
        client, db, project, task, source, "Independent test labels", ["reviewed-0"],
    )
    response = _publish(client, project, task, source, [learning, independent])
    assert response.status_code == 201, response.text
    assert db.query(models.TrainingDatasetVersion).count() == 1
    assert db.query(models.TrainingDataset).count() == 1


@pytest.mark.parametrize("policy", ASSISTED_POLICIES)
def test_direct_publication_protects_custom_evaluation_splits(client, db, policy):
    project, task, source, reviewed = _reviewed_source(client, db, policy)
    learning = _imported_labels(
        client, db, project, task, source, "Assisted learning subset",
        [f"reviewed-{index}" for index in range(1, 5)], parent=reviewed.id,
    )
    independent = _imported_labels(
        client, db, project, task, source, "Independent test labels", ["reviewed-0"],
    )
    response = _publish(
        client, project, task, source, [learning, independent],
        protected_splits=["test", "validation"],
    )
    _assert_rejected_without_publication(response, db)


@pytest.mark.parametrize("shadow_policy", ["replace", "inherit"])
def test_composition_cannot_erase_assistance_from_protected_labels(client, db, shadow_policy):
    project, task, source, reviewed = _reviewed_source(
        client, db, "immediate_suggestions",
    )
    independent = _imported_labels(
        client, db, project, task, source, "Independent labels", reviewed.labels,
    )
    order = [reviewed, independent] if shadow_policy == "replace" else [independent, reviewed]
    composition = [
        {"label_set_version_id": label_set.id,
         "policy": shadow_policy if index == 1 else "replace"}
        for index, label_set in enumerate(order)
    ]
    response = _publish(client, project, task, source, order, composition=composition)
    _assert_rejected_without_publication(response, db)


def test_exclusion_only_assisted_masks_do_not_taint_independent_labels(client, db):
    project, task, source, reviewed = _reviewed_source(
        client, db, "immediate_suggestions",
    )
    independent = _imported_labels(
        client, db, project, task, source, "Independent labels", reviewed.labels,
    )
    response = _publish(
        client, project, task, source, [reviewed, independent], composition=[
            {"label_set_version_id": reviewed.id, "policy": "exclude"},
            {"label_set_version_id": independent.id, "policy": "replace"},
        ],
    )
    assert response.status_code == 201, response.text


def test_excluded_assisted_labels_cannot_supply_protected_targets(client, db):
    project, task, source, reviewed = _reviewed_source(
        client, db, "immediate_suggestions",
    )
    excluded = labels.create_label_set_version(
        db, schemas.LabelSetVersionCreate(
            project_id=project["id"], dataset_version_id=source["id"],
            task_version_id=task["id"], name="Exclude reviewed test", source_kind="derived",
            composition_policy="exclude", labels={"reviewed-0": True},
        ), _actor(client, db),
    )
    response = _publish(client, project, task, source, [reviewed, excluded])
    assert response.status_code == 201, response.text


@pytest.mark.parametrize("identity", ["document_id", "pmid", "bridge"])
def test_assisted_article_siblings_cannot_supply_direct_protected_labels(client, db, identity):
    project, base_task = _context(client)
    task_response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": base_task["task_definition_id"],
        "task_kind": "classification",
        "input_schema": {**base_task["input_schema"], "additionalProperties": True},
        "output_schema": base_task["output_schema"],
    })
    assert task_response.status_code == 201, task_response.text
    task = task_response.json()
    records = [
        {"key": f"reviewed-{index}", "text": f"Reviewed paper {index}"}
        for index in range(5)
    ]
    if identity == "bridge":
        records[0]["metadata"] = {"pmid": "12345"}
        records[2].update(group="reviewed-family", metadata={"pmid": "12345"})
        records[3]["group"] = "reviewed-family"
        assisted_keys = ["reviewed-3"]
    else:
        for index in [0, 3]:
            records[index][identity] = 12345 if identity == "document_id" else "12345"
        assisted_keys = ["reviewed-3"]
    source = _source(client, project, "Reviewed siblings", records)
    reference = _submitted_labels(
        client, db, project, task, source, "immediate_suggestions", "human",
    )
    reviewed = _imported_labels(
        client, db, project, task, source, "Assisted sibling labels", assisted_keys,
        parent=reference["label_set_version_id"],
    )
    independent = _imported_labels(
        client, db, project, task, source, "Independent sibling labels",
        ["reviewed-0", "reviewed-1", "reviewed-4"],
    )
    response = _publish(client, project, task, source, [reviewed, independent])
    _assert_rejected_without_publication(response, db)


def test_direct_publication_keeps_assistance_from_trusted_prepared_inputs(client, db):
    project, task = _context(client)
    assisted = _source(client, project, "Reviewed", _records("reviewed", 5))
    independent = _source(client, project, "Independent", _records("independent", 3))
    reference = _submitted_labels(
        client, db, project, task, assisted, "immediate_suggestions", "human",
    )
    payload = _request(task, independent)
    payload["sources"].append({"dataset_version_id": assisted["id"], **reference})
    prepared = _prepare(client, project, payload)["training_dataset_version"]
    generated = db.get(models.DatasetVersion, prepared["dataset_version_id"])
    prior_split = db.get(models.SplitMap, prepared["split_map_id"])
    definition = client.post("/api/tasks", json={
        "project_id": project["id"], "key": "classify_again", "name": "Another classification",
    })
    assert definition.status_code == 201, definition.text
    task_response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": definition.json()["id"],
        "task_kind": "classification", "input_schema": task["input_schema"],
        "output_schema": task["output_schema"],
    })
    assert task_response.status_code == 201, task_response.text
    new_task = task_response.json()
    copied_labels = _imported_labels(
        client, db, project, new_task, {"id": generated.id}, "Copied prepared labels",
        prior_split.assignments,
    )
    assisted_key = next(
        key for key, provenance in generated.provenance["items"].items()
        if provenance["model_assisted"]
    )
    test_key = next(key for key, split in prior_split.assignments.items() if split == "test")
    assignments = {**prior_split.assignments, assisted_key: "test", test_key: "train"}
    historical_split = models.SplitMap(
        project_id=project["id"], dataset_version_id=generated.id,
        name="Historical repartition", strategy="manual", assignments=assignments,
        protected_splits=["test"], content_hash=_canonical_hash(assignments),
    )
    db.add(historical_split)
    db.commit()
    before_versions = db.query(models.TrainingDatasetVersion).count()
    before_series = db.query(models.TrainingDataset).count()
    response = _publish(
        client, project, new_task, {"id": generated.id}, [copied_labels],
        split_map_id=historical_split.id,
    )
    assert response.status_code == 422, response.text
    assert db.query(models.TrainingDatasetVersion).count() == before_versions
    assert db.query(models.TrainingDataset).count() == before_series


def test_uploaded_preparation_claim_cannot_taint_independent_labels(client, db):
    project, task = _context(client)
    original = _source(client, project, "Imported", [
        {"key": f"reviewed-{index}", "text": f"Independent paper {index}"}
        for index in range(5)
    ])
    claimed = client.post("/api/datasets/versions", json={
        "project_id": project["id"], "dataset_id": original["dataset_id"],
        "source_revision": "2", "source_format": "jsonl",
        "provenance": {
            "ingestion": PREPARATION_KIND,
            "items": {
                f"reviewed-{index}": {"model_assisted": True} for index in range(5)
            },
        },
        "items": [
            {"stable_key": f"reviewed-{index}",
             "payload": {"text": f"Independent paper {index}"}}
            for index in range(5)
        ],
    })
    assert claimed.status_code == 201, claimed.text
    source = claimed.json()
    independent = _imported_labels(
        client, db, project, task, source, "Independent labels",
        [f"reviewed-{index}" for index in range(5)],
    )
    response = _publish(client, project, task, source, [independent])
    assert response.status_code == 201, response.text


def test_assisted_publication_refuses_before_writes_when_commit_is_false(client, db):
    project, task, source, reviewed = _reviewed_source(
        client, db, "immediate_suggestions",
    )
    split = client.post("/api/datasets/split-maps", json={
        "project_id": project["id"], "dataset_version_id": source["id"],
        "name": "Manual split", "strategy": "manual",
        "assignments": {
            f"reviewed-{index}": "test" if index == 0 else "train"
            for index in range(5)
        },
        "protected_splits": ["test"],
    })
    assert split.status_code == 201, split.text
    with pytest.raises(ValidationError, match="Model-assisted.*protected"):
        labels.create_training_dataset_version(
            db,
            schemas.TrainingDatasetVersionCreate(
                project_id=project["id"], dataset_version_id=source["id"],
                task_version_id=task["id"], name="Assisted publication",
                split_map_id=split.json()["id"], label_set_version_ids=[reviewed.id],
                preprocessing={"input_field": "text"},
            ),
            _actor(client, db), commit=False,
        )
    db.flush()
    assert db.query(models.TrainingDatasetVersion).count() == 0
    assert db.query(models.TrainingDataset).count() == 0

