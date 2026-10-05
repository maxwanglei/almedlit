"""Training composition governs only examples with finalized labels."""

import pytest
from test_training_run_execution import MemoryObjectStorage
from test_workflow import workflow_data as workflow_data
from test_workflow import workflow_scope as workflow_scope

from al_medlit.core.exceptions import ValidationError
from al_medlit.workflow import models, schemas, service


def _source(db, scope, workflow_data, *, grouped=True):
    version = service.create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=scope.project.id,
            dataset_id=workflow_data.dataset.id,
            source_revision="partial-labels" if grouped else "independent-partial-labels",
            source_format="jsonl",
            items=[
                schemas.DatasetItemCreate(
                    stable_key=f"paper-{index}",
                    group_key=f"group-{index // 2}" if grouped else None,
                    payload={"text": f"Paper {index}"},
                )
                for index in range(12)
            ],
        ),
        scope.manager,
    )
    return version


@pytest.fixture
def partial_composition_source(db, workflow_scope, workflow_data):
    return _source(db, workflow_scope, workflow_data)


def _compose(db, scope, task_version, source, label_set):
    return service.compose_training_dataset_version(
        db,
        schemas.TrainingDatasetComposeCreate(
            project_id=scope.project.id,
            name="Partially annotated papers",
            dataset_version_id=source.id,
            task_version_id=task_version.id,
            input_field="text",
            label_set_version_id=label_set.id,
            train_percent=70,
            validation_percent=15,
            seed=42,
        ),
        scope.trainer,
        storage=MemoryObjectStorage(),
    )


def _labels(db, scope, task_version, source, keys):
    return service.create_label_set_version(
        db,
        schemas.LabelSetVersionCreate(
            project_id=scope.project.id,
            dataset_version_id=source.id,
            task_version_id=task_version.id,
            name="Submitted annotations",
            source_kind="imported",
            labels={key: {"label": "positive"} for key in keys},
        ),
        scope.manager,
    )


def test_partial_label_composition_splits_only_labeled_items(
    db, workflow_scope, workflow_data, partial_composition_source,
):
    source = partial_composition_source
    label_set = _labels(
        db, workflow_scope, workflow_data.task_version, source,
        [f"paper-{index}" for index in range(6)],
    )
    composed = _compose(db, workflow_scope, workflow_data.task_version, source, label_set)
    split_map = db.get(models.SplitMap, composed["split_map_id"])
    supervised = {
        key: split for key, split in split_map.assignments.items()
        if split in {"train", "validation", "test"}
    }
    assert set(supervised) == set(label_set.labels)
    assert set(supervised.values()) == {"train", "validation", "test"}
    assert supervised["paper-0"] == supervised["paper-1"]
    assert composed["group_count"] == 3
    assert sum(composed["split_counts"].values()) == label_set.label_count
    assert {
        key for key, split in split_map.assignments.items() if split == "pool"
    } == {f"paper-{index}" for index in range(6, 12)}
    assert source.item_count == 12
    effective = service.compose_training_dataset_labels(
        db, workflow_scope.project.id, composed["training_dataset_version"].id,
    )
    assert set(effective.labels) == set(supervised)


def test_partial_label_composition_rejects_insufficient_labeled_groups_without_writes(
    db, workflow_scope, workflow_data, partial_composition_source,
):
    source = partial_composition_source
    label_set = _labels(
        db, workflow_scope, workflow_data.task_version, source,
        [f"paper-{index}" for index in range(4)],
    )
    tracked = (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    before = [db.query(model).count() for model in tracked]
    with pytest.raises(ValidationError, match="three independent.*labeled groups"):
        _compose(db, workflow_scope, workflow_data.task_version, source, label_set)
    assert [db.query(model).count() for model in tracked] == before


def test_partial_group_coverage_is_rejected_without_writes(
    db, workflow_scope, workflow_data, partial_composition_source,
):
    source = partial_composition_source
    label_set = _labels(
        db, workflow_scope, workflow_data.task_version, source,
        [f"paper-{index}" for index in range(7)],
    )
    tracked = (models.SplitMap, models.TrainingDataset, models.TrainingDatasetVersion)
    before = [db.query(model).count() for model in tracked]
    with pytest.raises(ValidationError, match="fully labeled groups.*group-3"):
        _compose(db, workflow_scope, workflow_data.task_version, source, label_set)
    assert [db.query(model).count() for model in tracked] == before


def test_partial_labels_without_explicit_groups_leave_unlabeled_documents_in_pool(
    db, workflow_scope, workflow_data,
):
    source = _source(db, workflow_scope, workflow_data, grouped=False)
    label_set = _labels(
        db, workflow_scope, workflow_data.task_version, source,
        ["paper-0", "paper-3", "paper-6"],
    )
    composed = _compose(db, workflow_scope, workflow_data.task_version, source, label_set)
    split_map = db.get(models.SplitMap, composed["split_map_id"])
    assert composed["group_count"] == 3
    assert composed["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert {
        key for key, split in split_map.assignments.items() if split != "pool"
    } == set(label_set.labels)
    assert sum(split == "pool" for split in split_map.assignments.values()) == 9
