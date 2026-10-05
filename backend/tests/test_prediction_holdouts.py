"""A new source version must not reset established article holdout protection."""

import pytest
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.workflow import schemas, service
from al_medlit.workflow.services.predictions import _protected_items


@pytest.mark.parametrize("changed_group", [False, True])
def test_manual_review_retains_taskless_source_holdout_across_versions(
    db,
    workflow_scope,
    workflow_data,
    changed_group,
):
    scope, data = workflow_scope, workflow_data
    version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=scope.project.id,
            dataset_id=data.dataset.id,
            source_uri="hf://example/reviews",
            source_revision="second-source-version",
            source_format="jsonl",
            items=[
                schemas.DatasetItemCreate(
                    stable_key=item.stable_key,
                    group_key=(f"updated-{item.group_key}" if changed_group else item.group_key),
                    payload=item.payload,
                )
                for item in data.items.values()
            ],
        ),
        scope.manager,
    )
    items = service.list_dataset_items(db, scope.project.id, version.id)
    protected = _protected_items(db, items, data.task_version.id)
    assert protected == {item.id for item in items if item.stable_key == "test-1"}


def test_manual_review_ignores_source_holdout_owned_only_by_another_task(
    db, workflow_scope, workflow_data
):
    scope, data = workflow_scope, workflow_data
    labels = service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=scope.project.id,
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            name="Training labels",
            source_kind="imported",
            labels={"train-1": {"label": "positive"}},
        ),
        scope.manager,
    )
    service.create_training_dataset_version(
        db,
        schemas.TrainingDatasetVersionCreate(
            project_id=scope.project.id,
            name="Task-owned source split",
            dataset_version_id=data.dataset_version.id,
            task_version_id=data.task_version.id,
            label_set_version_ids=[labels.id],
            split_map_id=data.split_map.id,
        ),
        scope.manager,
    )
    other_task = service.create_task_version(
        db,
        schemas.TaskVersionCreate(
            project_id=scope.project.id,
            task_definition_id=data.task.id,
            task_kind="classification",
            input_schema={"text": "string"},
            output_schema={"label": ["negative", "positive"]},
            metrics=["f1"],
            trainer_compatibility=["tfidf_logistic_regression"],
        ),
        scope.manager,
    )
    items = list(data.items.values())
    assert _protected_items(db, items, data.task_version.id) == {data.items["test-1"].id}
    assert _protected_items(db, items, other_task.id) == set()
