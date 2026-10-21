"""Model provenance preserves independent visibility and download restrictions."""

import pytest
from test_artifact_quotas import _headers, _package_files, _package_spec, _workspace_projects

from al_medlit.model_artifacts import service
from al_medlit.model_artifacts.models import (
    ArtifactPackageReference,
    BaseModelAsset,
    BaseModelAssetState,
)
from al_medlit.model_artifacts.schemas import ArtifactPackageReferenceCreate


@pytest.fixture
def access_project(db):
    manager, trainer, _annotator, workspace, project, _second = _workspace_projects(db)
    workspace.capability_preset = "full"
    db.commit()
    return manager, trainer, project


def _publish(db, storage, actors, name, license_info=None, references=()):
    manager, _trainer, project = actors
    return service.publish_artifact_package(
        db,
        storage,
        project_id=project.id,
        data=_package_spec(name).model_copy(
            update={
                "license_info": license_info or {},
                "references": [
                    ArtifactPackageReferenceCreate(
                        target_package_id=package.id,
                        relationship_type=kind,
                    )
                    for package, kind in references
                ],
            }
        ),
        files=_package_files(),
        actor_user_id=manager.id,
    )


def _download(client, package, user):
    return client.get(
        f"/api/artifact-packages/{package.id}/files/{package.files[0].id}/download",
        headers=_headers(user),
    )


def test_downloadable_package_reaches_payload_with_training_enabled(
    db,
    client,
    object_storage,
    access_project,
):
    package = _publish(db, object_storage, access_project, "public")
    db.commit()
    for user in access_project[:2]:
        response = _download(client, package, user)
        assert response.status_code == 200
        assert response.content


@pytest.mark.parametrize(
    "relationship",
    [
        "uses_base_model",
        "derived_from_parent_model",
        "promoted_from_candidate",
        "imported_from",
    ],
)
@pytest.mark.parametrize(
    ("mode", "trainer_status", "manager_status"),
    [
        ("execution_only", 403, 403),
        ("manager_only", 404, 200),
    ],
)
def test_model_reference_inherits_download_and_visibility_policy(
    db,
    client,
    object_storage,
    access_project,
    relationship,
    mode,
    trainer_status,
    manager_status,
):
    manager, trainer, project = access_project
    base = _publish(db, object_storage, access_project, "base", {"access_mode": mode})
    derived = _publish(
        db,
        object_storage,
        access_project,
        "derived",
        {"access_mode": "downloadable"},
        [(base, relationship)],
    )
    db.commit()
    assert _download(client, derived, trainer).status_code == trainer_status
    assert _download(client, derived, manager).status_code == manager_status
    detail = client.get(f"/api/artifact-packages/{derived.id}", headers=_headers(trainer))
    assert detail.status_code == (404 if mode == "manager_only" else 200)
    listed = client.get(f"/api/projects/{project.id}/artifact-packages", headers=_headers(trainer))
    assert listed.status_code == 200
    assert (derived.id in {row["id"] for row in listed.json()}) == (mode != "manager_only")


def test_transitive_reference_cycle_combines_both_dimensions(
    db,
    client,
    object_storage,
    access_project,
):
    manager, trainer, _project = access_project
    execution = _publish(
        db, object_storage, access_project, "execution", {"access_mode": "execution_only"}
    )
    hidden = _publish(
        db,
        object_storage,
        access_project,
        "hidden",
        {"access_mode": "manager_only"},
        [(execution, "uses_base_model")],
    )
    derived = _publish(
        db,
        object_storage,
        access_project,
        "derived",
        references=[(hidden, "promoted_from_candidate")],
    )
    db.add(
        ArtifactPackageReference(
            source_package_id=execution.id,
            target_package_id=derived.id,
            relationship_type="derived_from_parent_model",
        )
    )
    db.commit()
    assert _download(client, derived, trainer).status_code == 404
    assert _download(client, derived, manager).status_code == 403


@pytest.mark.parametrize("mode", ["invalid", None, ["execution_only"]])
def test_invalid_explicit_license_mode_fails_closed_for_manager_download(
    db,
    client,
    object_storage,
    access_project,
    mode,
):
    manager, trainer, _project = access_project
    package = _publish(db, object_storage, access_project, "invalid", {"access_mode": mode})
    db.commit()
    assert _download(client, package, trainer).status_code == 404
    assert _download(client, package, manager).status_code == 403


def test_reference_uses_raw_catalog_policy_when_json_omits_it(
    db,
    client,
    object_storage,
    access_project,
):
    manager, trainer, project = access_project
    package = _publish(
        db, object_storage, access_project, "legacy base", {"access_mode": "downloadable"}
    )
    asset = BaseModelAsset(
        project_id=project.id,
        workspace_id=project.workspace_id,
        package_id=package.id,
        provider="internal",
        source_model_id="test/raw-catalog",
        exact_revision="revision-1",
        display_name="Legacy catalog base",
        model_family="deep_learning",
        model_type="transformer",
        license_name="restricted",
        access_mode="execution_only",
        created_by_user_id=manager.id,
        state=BaseModelAssetState(readiness="ready"),
    )
    db.add(asset)
    db.flush()
    derived = _publish(
        db,
        object_storage,
        access_project,
        "derived",
        references=[(asset.package, "uses_base_model")],
    )
    db.commit()
    assert _download(client, derived, trainer).status_code == 403
    assert _download(client, derived, manager).status_code == 403


def test_catalog_routes_hide_inherited_manager_policy(db, client, object_storage, access_project):
    manager, trainer, project = access_project
    base = _publish(
        db, object_storage, access_project, "restricted", {"access_mode": "manager_only"}
    )
    package = _publish(
        db, object_storage, access_project, "catalog package", references=[(base, "imported_from")]
    )
    asset = BaseModelAsset(
        project_id=project.id,
        workspace_id=project.workspace_id,
        package_id=package.id,
        provider="internal",
        source_model_id="test/legacy-import",
        exact_revision="revision-1",
        display_name="Legacy imported model",
        model_family="deep_learning",
        model_type="transformer",
        license_name="restricted",
        access_mode="downloadable",
        created_by_user_id=manager.id,
        state=BaseModelAssetState(readiness="ready"),
    )
    db.add(asset)
    db.commit()
    assert client.get(f"/api/base-models/{asset.id}", headers=_headers(trainer)).status_code == 404
    listed = client.get(f"/api/projects/{project.id}/base-models", headers=_headers(trainer))
    assert listed.status_code == 200
    assert listed.json() == []
    assert client.get(f"/api/base-models/{asset.id}", headers=_headers(manager)).status_code == 200


@pytest.mark.parametrize(
    ("source_license", "requested"),
    [
        ({"access_mode": "execution_only"}, "manager_only"),
        ({"base_model": {"access_mode": "execution_only"}}, "downloadable"),
        ({"base_model": {"access_mode": "manager_only"}}, "execution_only"),
        (
            {"access_mode": "manager_only", "base_model": {"access_mode": "execution_only"}},
            "manager_only",
        ),
    ],
)
def test_import_cannot_drop_either_inherited_restriction(
    db,
    client,
    object_storage,
    access_project,
    source_license,
    requested,
):
    manager, _trainer, project = access_project
    source = _publish(db, object_storage, access_project, "source", source_license)
    db.commit()
    response = client.post(
        f"/api/projects/{project.id}/base-models/import",
        headers=_headers(manager),
        json={
            "source_package_id": source.id,
            "provider": "internal",
            "source_model_id": "test/reimport",
            "exact_revision": "revision-1",
            "display_name": "Reimported model",
            "model_family": "deep_learning",
            "model_type": "transformer",
            "license_name": "restricted",
            "access_mode": requested,
        },
    )
    assert response.status_code == 409, response.text
    assert db.query(BaseModelAsset).count() == 0


def test_non_model_reference_does_not_inherit_model_license(
    db,
    client,
    object_storage,
    access_project,
):
    _manager, trainer, _project = access_project
    evaluation = _publish(
        db, object_storage, access_project, "evaluation", {"access_mode": "manager_only"}
    )
    package = _publish(
        db,
        object_storage,
        access_project,
        "trained",
        references=[(evaluation, "validated_by_evaluation")],
    )
    db.commit()
    assert _download(client, package, trainer).status_code == 200


def test_import_checks_reference_ancestry_even_without_license_json(
    db,
    client,
    object_storage,
    access_project,
):
    manager, _trainer, project = access_project
    base = _publish(db, object_storage, access_project, "base", {"access_mode": "execution_only"})
    source = _publish(
        db,
        object_storage,
        access_project,
        "derived",
        references=[(base, "derived_from_parent_model")],
    )
    db.commit()
    response = client.post(
        f"/api/projects/{project.id}/base-models/import",
        headers=_headers(manager),
        json={
            "source_package_id": source.id,
            "provider": "internal",
            "source_model_id": "test/ancestry-import",
            "exact_revision": "revision-1",
            "display_name": "Imported model",
            "model_family": "deep_learning",
            "model_type": "transformer",
            "license_name": "restricted",
            "access_mode": "manager_only",
        },
    )
    assert response.status_code == 409, response.text
    assert db.query(BaseModelAsset).count() == 0
