"""Exclusion masks remain masks through composition overrides and saved plans."""

from types import SimpleNamespace

import pytest
from test_training_preparation import _context, _records, _source

from al_medlit.auth.models import User
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactStorageReservation,
)
from al_medlit.workflow import models, schemas
from al_medlit.workflow.services import labels
from al_medlit.workflow.services.common import _canonical_hash


@pytest.fixture
def scope(client, db):
    project, task = _context(client)
    source = _source(client, project, "Mask source", [
        {"key": row["key"], "text": row["text"]} for row in _records("paper", 4)
    ])
    actor = db.get(User, client.get("/api/auth/me").json()["user"]["id"])
    base = labels.create_label_set_version(db, schemas.LabelSetVersionCreate(
        project_id=project["id"], dataset_version_id=source["id"], task_version_id=task["id"],
        name="Actual imported labels", source_kind="imported",
        labels={f"paper-{index}": "yes" for index in range(4)},
    ), actor)
    split = client.post("/api/datasets/split-maps", json={
        "project_id": project["id"], "dataset_version_id": source["id"],
        "name": "Training split", "strategy": "manual",
        "assignments": {
            "paper-0": "train", "paper-1": "train", "paper-2": "validation", "paper-3": "test",
        },
    })
    assert split.status_code == 201, split.text
    return SimpleNamespace(
        project=project, task=task, source=source, actor=actor, base=base, split=split.json(),
    )


def _mask(db, scope, source_kind, *, keys=None, value=True):
    return labels.create_label_set_version(db, schemas.LabelSetVersionCreate(
        project_id=scope.project["id"], dataset_version_id=scope.source["id"],
        task_version_id=scope.task["id"], name="Exclusion-only mask",
        source_kind=source_kind, composition_policy="exclude",
        labels={key: value for key in (keys if keys is not None else scope.base.labels)},
    ), scope.actor)


def _training_payload(scope, label_sets, composition):
    return {
        "project_id": scope.project["id"], "dataset_version_id": scope.source["id"],
        "task_version_id": scope.task["id"], "name": "Composed training examples",
        "split_map_id": scope.split["id"], "label_set_version_ids": [row.id for row in label_sets],
        "composition": composition, "preprocessing": {"input_field": "text"},
    }


def _counts(db):
    return {
        model: db.query(model).count()
        for model in (
            models.TrainingDataset, models.TrainingDatasetVersion, models.SplitMap,
            models.LabelSetVersion, ArtifactPackage, ArtifactBlob, ArtifactStorageReservation,
        )
    }


@pytest.mark.parametrize("source_kind", ["derived", "imported"])
@pytest.mark.parametrize("override", ["replace", "inherit"])
def test_explicit_composition_cannot_promote_exclusion_masks_to_labels(
    client, db, scope, source_kind, override,
):
    mask = _mask(db, scope, source_kind)
    before = _counts(db)
    response = client.post("/api/datasets/training-versions", json=_training_payload(
        scope, [mask], [{"label_set_version_id": mask.id, "policy": override}],
    ))
    assert response.status_code == 422, response.text
    assert "supply training labels" in response.json()["detail"].lower()
    assert _counts(db) == before


@pytest.mark.parametrize("source_kind", ["derived", "imported"])
def test_legacy_composer_cannot_force_an_exclusion_mask_to_replace_labels(
    client, db, scope, source_kind,
):
    mask = _mask(db, scope, source_kind)
    before = _counts(db)
    response = client.post("/api/datasets/training-versions/compose", json={
        "project_id": scope.project["id"], "dataset_version_id": scope.source["id"],
        "task_version_id": scope.task["id"], "name": "Mask promoted by legacy composition",
        "input_field": "text", "label_set_version_id": mask.id,
        "train_percent": 80, "validation_percent": 10, "seed": 42,
    })
    assert response.status_code == 422, response.text
    assert "supply training labels" in response.json()["detail"].lower()
    assert _counts(db) == before


def test_schema_valid_exclusion_values_still_cannot_be_promoted_to_training_labels(
    client, db, scope,
):
    mask = _mask(db, scope, "imported", value="yes")
    before = _counts(db)
    response = client.post("/api/datasets/training-versions", json=_training_payload(
        scope, [mask], [{"label_set_version_id": mask.id, "policy": "replace"}],
    ))
    assert response.status_code == 422, response.text
    assert "supply training labels" in response.json()["detail"].lower()
    assert _counts(db) == before


@pytest.mark.parametrize("source_kind", ["derived", "imported"])
@pytest.mark.parametrize("override", ["replace", "inherit"])
def test_saved_historical_composition_cannot_promote_exclusion_masks_when_read(
    client, db, scope, source_kind, override,
):
    mask = _mask(db, scope, source_kind)
    historical = models.TrainingDatasetVersion(
        **_training_payload(scope, [mask], [{"label_set_version_id": mask.id, "policy": override}]),
        content_hash=_canonical_hash({"historical": mask.id, "override": override}),
    )
    db.add(historical)
    db.commit()
    response = client.get(
        f"/api/datasets/training-versions/{historical.id}/labels",
        params={"project_id": scope.project["id"]},
    )
    assert response.status_code == 422, response.text
    assert "supply training labels" in response.json()["detail"].lower()


def test_historical_derived_contributing_policy_cannot_supply_implicit_composition(
    client, db, scope,
):
    historical_labels = models.LabelSetVersion(
        project_id=scope.project["id"], dataset_version_id=scope.source["id"],
        task_version_id=scope.task["id"], name="Historical contributing derived set",
        version_number=1,
        source_kind="derived", composition_policy="replace", labels=scope.base.labels,
        label_count=len(scope.base.labels), content_hash=_canonical_hash({"historical": "derived"}),
    )
    db.add(historical_labels)
    db.flush()
    historical_training = models.TrainingDatasetVersion(
        **_training_payload(scope, [historical_labels], []),
        content_hash=_canonical_hash({"historical": "implicit-derived-composition"}),
    )
    db.add(historical_training)
    db.commit()
    response = client.get(
        f"/api/datasets/training-versions/{historical_training.id}/labels",
        params={"project_id": scope.project["id"]},
    )
    assert response.status_code == 422, response.text
    assert "supply training labels" in response.json()["detail"].lower()


@pytest.mark.parametrize("source_kind", ["derived", "imported"])
@pytest.mark.parametrize("explicit", [False, True])
def test_exclusion_masks_keep_removing_keys_with_default_or_explicit_policy(
    client, db, scope, source_kind, explicit,
):
    mask = _mask(db, scope, source_kind, keys=["paper-0"])
    composition = [
        {"label_set_version_id": scope.base.id, "policy": "replace"},
        {"label_set_version_id": mask.id, "policy": "exclude"},
    ] if explicit else []
    response = client.post("/api/datasets/training-versions", json=_training_payload(
        scope, [scope.base, mask], composition,
    ))
    assert response.status_code == 201, response.text
    resolved = client.get(
        f"/api/datasets/training-versions/{response.json()['id']}/labels",
        params={"project_id": scope.project["id"]},
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["labels"] == {f"paper-{index}": "yes" for index in (1, 2, 3)}


@pytest.mark.parametrize(("declared", "override", "expected"), [
    ("inherit", "replace", "no"), ("replace", "inherit", "yes"),
])
def test_contributing_label_sets_still_allow_valid_composition_overrides(
    client, db, scope, declared, override, expected,
):
    corrections = labels.create_label_set_version(db, schemas.LabelSetVersionCreate(
        project_id=scope.project["id"], dataset_version_id=scope.source["id"],
        task_version_id=scope.task["id"], name="Actual imported correction",
        source_kind="imported", composition_policy=declared, labels={"paper-0": "no"},
    ), scope.actor)
    response = client.post("/api/datasets/training-versions", json=_training_payload(
        scope, [scope.base, corrections], [
            {"label_set_version_id": scope.base.id, "policy": "inherit"},
            {"label_set_version_id": corrections.id, "policy": override},
        ],
    ))
    assert response.status_code == 201, response.text
    resolved = client.get(
        f"/api/datasets/training-versions/{response.json()['id']}/labels",
        params={"project_id": scope.project["id"]},
    )
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["labels"] == {**scope.base.labels, "paper-0": expected}


def test_preparation_rejects_historical_derived_set_with_contributing_declared_policy(
    client, db, scope,
):
    historical = models.LabelSetVersion(
        project_id=scope.project["id"], dataset_version_id=scope.source["id"],
        task_version_id=scope.task["id"], name="Historical derived labels", version_number=1,
        source_kind="derived", composition_policy="replace", labels=scope.base.labels,
        label_count=len(scope.base.labels), content_hash=_canonical_hash({"historical": "derived"}),
    )
    db.add(historical)
    db.commit()
    response = client.post(
        f"/api/projects/{scope.project['id']}/training-datasets/preview", json={
            "name": "Unreviewed derived labels", "task_version_id": scope.task["id"],
            "sources": [{
                "dataset_version_id": scope.source["id"], "label_set_version_id": historical.id,
            }],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["ready"] is False, response.text
    assert any(
        issue["code"] == "invalid_labels" and "supply training labels" in issue["message"].lower()
        for issue in response.json()["issues"]
    ), response.text
