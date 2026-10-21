"""Holdout checks must avoid loading unrelated immutable project history."""

from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
from sqlalchemy import event
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.workflow import models, schemas, service
from al_medlit.workflow.services.common import _canonical_hash
from al_medlit.workflow.services.holdouts import (
    protected_dataset_item_ids,
    source_split_constraints,
)
from al_medlit.workflow.services.preparation import _protected_source_context


@dataclass
class _Loads:
    item_ids: set[int] = field(default_factory=set)
    version_ids: set[int] = field(default_factory=set)
    statements: list[str] = field(default_factory=list)


@contextmanager
def _capture_loads(engine):
    """Observe real ORM loads in a fresh session, without relying on timing."""
    loads = _Loads()

    def item_loaded(item, _context):
        loads.item_ids.add(item.id)

    def version_loaded(version, _context):
        loads.version_ids.add(version.id)

    def executed(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith("SELECT"):
            loads.statements.append(statement)

    listeners = [
        (models.DatasetItem, "load", item_loaded),
        (models.DatasetVersion, "load", version_loaded),
        (engine, "before_cursor_execute", executed),
    ]
    for target, name, callback in listeners:
        event.listen(target, name, callback)
    try:
        yield loads
    finally:
        for target, name, callback in listeners:
            event.remove(target, name, callback)


def _history(
    db,
    project_id,
    *,
    count,
    prefix,
    task_version_id=None,
    with_splits=True,
    learning_fields=None,
):
    dataset = models.Dataset(
        project_id=project_id,
        name=f"{prefix} history",
        source_type="direct_upload",
    )
    db.add(dataset)
    db.flush()
    protected_ids, learning_ids, version_ids = set(), set(), set()
    for index in range(count):
        version = models.DatasetVersion(
            project_id=project_id,
            dataset_id=dataset.id,
            version_number=index + 1,
            source_revision=f"{prefix}-{index}",
            source_format="jsonl",
            provenance={"unrelated_metadata": {str(index): "x" * 1000}},
            content_hash=_canonical_hash([prefix, index]),
            item_count=2,
        )
        db.add(version)
        db.flush()
        version_ids.add(version.id)
        assignments = {"heldout": "test", "learning": "train"}
        for key, destination in [("heldout", protected_ids), ("learning", learning_ids)]:
            payload = {"text": f"{prefix} historical {index} {key}"}
            if key == "learning":
                payload.update(learning_fields or {})
            item = models.DatasetItem(
                project_id=project_id,
                dataset_version_id=version.id,
                stable_key=key,
                group_key=f"{prefix}-{index}-{key}",
                payload=payload,
                content_hash=_canonical_hash(payload),
            )
            db.add(item)
            db.flush()
            destination.add(item.id)
        if with_splits:
            split = models.SplitMap(
                project_id=project_id,
                dataset_version_id=version.id,
                name=f"{prefix}-{index}",
                strategy="manual",
                assignments=assignments,
                protected_splits=["test"],
                content_hash=_canonical_hash(assignments),
            )
            db.add(split)
            db.flush()
            if task_version_id is not None:
                db.add(
                    models.TrainingDatasetVersion(
                        project_id=project_id,
                        name=f"{prefix} training {index}",
                        dataset_version_id=version.id,
                        task_version_id=task_version_id,
                        split_map_id=split.id,
                        content_hash=_canonical_hash(["training", prefix, index]),
                    )
                )
    db.commit()
    return protected_ids, learning_ids, version_ids


def test_protected_context_loads_only_protected_historical_items(
    db,
    workflow_scope,
    testing_session_factory,
    test_engine,
):
    project_id = workflow_scope.project.id
    protected_ids, learning_ids, _versions = _history(
        db,
        project_id,
        count=6,
        prefix="reserved",
    )
    _unused, _other_items, unrelated_versions = _history(
        db,
        project_id,
        count=3,
        prefix="unused",
        with_splits=False,
    )
    with testing_session_factory() as fresh:
        with _capture_loads(test_engine) as loads:
            actual, tokens = _protected_source_context(fresh, project_id)
    assert actual == protected_ids
    assert tokens
    assert not (loads.item_ids & learning_ids), "Unprotected historical payloads were loaded"
    assert not (loads.version_ids & unrelated_versions)


def test_protected_context_queries_do_not_grow_per_historical_split_version(
    db,
    workflow_scope,
    testing_session_factory,
    test_engine,
):
    project_id = workflow_scope.project.id
    _history(db, project_id, count=1, prefix="first")

    def query_count():
        with testing_session_factory() as fresh:
            with _capture_loads(test_engine) as loads:
                _protected_source_context(fresh, project_id)
        return len(loads.statements)

    before = query_count()
    _history(db, project_id, count=12, prefix="more")
    after = query_count()
    assert after <= before + 3, (before, after)


@pytest.mark.parametrize("with_splits", [False, True])
def test_prediction_protection_does_not_materialize_unrelated_history(
    db,
    workflow_scope,
    workflow_data,
    testing_session_factory,
    test_engine,
    with_splits,
):
    project_id = workflow_scope.project.id
    task_id = workflow_data.task_version.id
    heldout_id = workflow_data.items["test-1"].id
    selected_ids = [workflow_data.items[key].id for key in ("train-1", "test-1")]
    historical_protected, historical_learning, unrelated_versions = _history(
        db,
        project_id,
        count=8,
        prefix="unrelated-history",
        with_splits=with_splits,
    )
    with testing_session_factory() as fresh:
        items = (
            fresh.query(models.DatasetItem).filter(models.DatasetItem.id.in_(selected_ids)).all()
        )
        with _capture_loads(test_engine) as loads:
            actual = protected_dataset_item_ids(fresh, items, task_id)
    assert actual == {heldout_id}
    assert not (loads.version_ids & unrelated_versions), "Unrelated version provenance was loaded"
    assert not (loads.item_ids & (historical_protected | historical_learning)), (
        "Protection of one prediction page materialized unrelated historical payloads"
    )


@pytest.mark.parametrize("same_task", [True, False])
def test_source_constraints_do_not_materialize_unrelated_training_history(
    db,
    workflow_scope,
    workflow_data,
    testing_session_factory,
    test_engine,
    same_task,
):
    project_id = workflow_scope.project.id
    task_id = workflow_data.task_version.id
    source_id = workflow_data.dataset_version.id
    other_task = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=project_id,
            task_definition_id=workflow_data.task.id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"label": ["negative", "positive"]},
        ),
        workflow_scope.manager,
    )
    historical_protected, historical_learning, unrelated_versions = _history(
        db,
        project_id,
        count=6,
        prefix="unrelated-training",
        task_version_id=task_id if same_task else other_task.id,
    )
    with testing_session_factory() as fresh:
        version = fresh.get(models.DatasetVersion, source_id)
        items = fresh.query(models.DatasetItem).filter_by(dataset_version_id=source_id).all()
        with _capture_loads(test_engine) as loads:
            constraints = source_split_constraints(
                fresh,
                version,
                items,
                task_id,
                input_mapping={"text": "text"},
            )
    assert constraints == {"test-1": {"test"}}
    assert not (loads.version_ids & unrelated_versions), "Unrelated history provenance was loaded"
    assert not (loads.item_ids & historical_learning), "Unrelated learning payloads were loaded"
    if not same_task:
        assert not (loads.item_ids & historical_protected)


@pytest.mark.parametrize(
    "historical_identity",
    [
        {"pmid": 123},
        {"pmid": " 123 "},
        {"pmid": "\t123\n"},
        {"metadata": {"pmid": "123"}},
        {"source": "pubmed_abstract", "external_id": "123"},
    ],
)
def test_history_filter_preserves_normalized_article_identity(
    db,
    workflow_scope,
    workflow_data,
    testing_session_factory,
    historical_identity,
):
    project_id = workflow_scope.project.id
    task_id = workflow_data.task_version.id
    _history(
        db,
        project_id,
        count=1,
        prefix="prior-learning",
        task_version_id=task_id,
        learning_fields=historical_identity,
    )
    current = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=workflow_data.dataset.id,
            source_revision="copied-article",
            source_format="jsonl",
            items=[
                schemas.DatasetItemCreate(
                    stable_key="copied-article",
                    payload={"text": "A changed passage from the same article", "pmid": "123"},
                )
            ],
        ),
        workflow_scope.manager,
    )
    current_id = current.id
    with testing_session_factory() as fresh:
        version = fresh.get(models.DatasetVersion, current_id)
        items = fresh.query(models.DatasetItem).filter_by(dataset_version_id=current_id).all()
        constraints = source_split_constraints(
            fresh,
            version,
            items,
            task_id,
            input_mapping={"text": "text"},
        )
    assert constraints == {"copied-article": {"train", "validation", "pool"}}


@pytest.mark.parametrize("ingestion", ["direct_upload", "public_registry_snapshot"])
def test_reordered_upload_protection_is_independent_of_prediction_page(
    db, workflow_scope, workflow_data, testing_session_factory, ingestion,
):
    project_id = workflow_scope.project.id
    dataset = models.Dataset(
        project_id=project_id, name="Positional upload", source_type="direct_upload",
    )
    db.add(dataset)
    db.flush()
    versions = []
    for number, texts in enumerate(
        [("Reserved evaluation text", "Earlier learning text"),
         ("New unrelated text", "Reserved evaluation text")],
        start=1,
    ):
        version = models.DatasetVersion(
            project_id=project_id, dataset_id=dataset.id, version_number=number,
            source_revision=f"upload-{number}", source_format="jsonl",
            provenance={"ingestion": ingestion}, content_hash=_canonical_hash(texts),
            item_count=len(texts),
        )
        db.add(version)
        db.flush()
        versions.append(version)
        for index, text in enumerate(texts):
            payload = {"text": text}
            db.add(models.DatasetItem(
                project_id=project_id, dataset_version_id=version.id,
                stable_key=f"row-{index + 1:08d}", payload=payload,
                content_hash=_canonical_hash(payload),
            ))
    assignments = {"row-00000001": "test", "row-00000002": "train"}
    db.add(models.SplitMap(
        project_id=project_id, dataset_version_id=versions[0].id,
        name="Original upload holdout", strategy="manual", assignments=assignments,
        protected_splits=["test"], content_hash=_canonical_hash(assignments),
    ))
    db.commit()
    current_id = versions[1].id
    with testing_session_factory() as fresh:
        current = (
            fresh.query(models.DatasetItem)
            .filter_by(dataset_version_id=current_id)
            .order_by(models.DatasetItem.stable_key)
            .all()
        )
        task_id = workflow_data.task_version.id
        assert protected_dataset_item_ids(fresh, [current[0]], task_id) == set()
        assert protected_dataset_item_ids(fresh, [current[1]], task_id) == {current[1].id}
        assert protected_dataset_item_ids(fresh, current, task_id) == {current[1].id}


@pytest.mark.parametrize(
    "value, unrelated",
    [
        (42, 999),
        (1.5, 2.5),
        (True, False),
        (None, "unrelated"),
        ({"a": "shared", "b": [1, {"quoted.key": True}]},
         {"a": "shared", "b": [2, {"quoted.key": True}]}),
        ([{"features": [1, None]}], [{"features": [2, None]}]),
        ({}, {"other": 1}),
        ([], [1]),
    ],
)
def test_mapped_input_history_filters_json_values_without_losing_learning_identity(
    db, workflow_scope, workflow_data, testing_session_factory, test_engine, value, unrelated,
):
    project_id = workflow_scope.project.id
    task_id = workflow_data.task_version.id
    unrelated_protected, unrelated_learning, _ = _history(
        db, project_id, count=3, prefix="unrelated-inputs", task_version_id=task_id,
        learning_fields={"features": unrelated},
    )
    _heldout, matching_learning, _ = _history(
        db, project_id, count=1, prefix="matching-input", task_version_id=task_id,
        learning_fields={"features": value},
    )
    reordered = dict(reversed(list(value.items()))) if isinstance(value, dict) else value
    current = service.create_dataset_version(
        db, schemas.DatasetVersionCreate(
            project_id=project_id, dataset_id=workflow_data.dataset.id,
            source_revision="structured-copy", source_format="jsonl",
            items=[schemas.DatasetItemCreate(
                stable_key="mapped-copy", payload={"renamed": reordered},
            )],
        ), workflow_scope.manager,
    )
    current_id = current.id
    with testing_session_factory() as fresh:
        version = fresh.get(models.DatasetVersion, current_id)
        items = fresh.query(models.DatasetItem).filter_by(dataset_version_id=current_id).all()
        with _capture_loads(test_engine) as loads:
            constraints = source_split_constraints(
                fresh, version, items, task_id, input_mapping={"features": "renamed"},
            )
    assert constraints == {"mapped-copy": {"train", "validation", "pool"}}
    assert loads.item_ids & matching_learning
    assert not (loads.item_ids & (unrelated_protected | unrelated_learning))


@pytest.mark.parametrize("action, status", [("preview", 200), ("prepare", 201)])
def test_structured_preparation_does_not_materialize_unrelated_history(
    client, db, test_engine, action, status,
):
    from test_training_preparation import _context, _request, _source

    project, original = _context(client)
    response = client.post("/api/tasks/versions", json={
        "project_id": project["id"], "task_definition_id": original["task_definition_id"],
        "task_kind": "classification",
        "input_schema": {
            "type": "object", "required": ["features"],
            "properties": {"features": {"type": "object"}},
        },
        "output_schema": original["output_schema"],
    })
    assert response.status_code == 201, response.text
    task = response.json()
    source = _source(client, project, "Structured", [
        {"key": str(index), "features": {"measurement": index}, "label": "yes"}
        for index in range(5)
    ])
    protected, learning, _ = _history(
        db, project["id"], count=4, prefix="unrelated-structured",
        task_version_id=task["id"], learning_fields={"features": {"measurement": 999}},
    )
    payload = _request(task, source)
    payload["sources"][0]["input_mapping"] = {"features": "features"}
    with _capture_loads(test_engine) as loads:
        response = client.post(
            f"/api/projects/{project['id']}/training-datasets/{action}", json=payload,
        )
    assert response.status_code == status, response.text
    assert not (loads.item_ids & (protected | learning))
