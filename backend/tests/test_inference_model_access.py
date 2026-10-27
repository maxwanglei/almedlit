import pytest
from fastapi import HTTPException
from test_artifact_quotas import _headers, _package_files, _package_spec
from test_inference_candidate_queries import _inference_scope

from al_medlit.inference import router as inference_router
from al_medlit.model_artifacts import service
from al_medlit.model_artifacts.models import BaseModelAsset, BaseModelAssetState
from al_medlit.model_artifacts.schemas import ArtifactPackageReferenceCreate
from al_medlit.training.model_types import catalog
from al_medlit.training.models import ModelCheckpoint
from al_medlit.workspace.models import Workspace, WorkspaceMember


@pytest.mark.parametrize("kind", ["full", "adapter"])
@pytest.mark.parametrize("operation", ["launch", "retry"])
@pytest.mark.parametrize(
    "mode,role,superuser,expected",
    [
        ("manager_only", "trainer", False, 403),
        ("manager_only", "manager", False, 418),
        ("manager_only", "admin", False, 418),
        ("manager_only", "trainer", True, 418),
        ("execution_only", "trainer", False, 418),
    ],
)
def test_inference_enforces_selected_checkpoint_package_policy(
    db,
    client,
    object_storage,
    monkeypatch,
    kind,
    operation,
    mode,
    role,
    superuser,
    expected,
):
    model_type = "evidence_lora" if kind == "adapter" else "evidence_block_sentence_tagger"
    scope = _inference_scope(db, checkpoint_manifest={"model_type": model_type}, with_window=False)
    db.get(Workspace, scope.project.workspace_id).capability_preset = "full"
    db.query(WorkspaceMember).filter(WorkspaceMember.user_id == scope.user.id).update(
        {"role": role}
    )
    scope.user.is_superuser = superuser
    references = []
    if kind == "adapter":
        base = service.publish_artifact_package(
            db,
            object_storage,
            project_id=scope.project.id,
            data=_package_spec("downloadable-base").model_copy(
                update={"package_kind": "base_model"}
            ),
            files=_package_files(),
            actor_user_id=scope.user.id,
        )
        asset = BaseModelAsset(
            project_id=scope.project.id,
            workspace_id=scope.project.workspace_id,
            package_id=base.id,
            provider="internal",
            source_model_id="test/base",
            exact_revision="revision-1",
            display_name="Downloadable base",
            model_family="deep_learning",
            model_type="transformer",
            license_name="test",
            access_mode="downloadable",
            created_by_user_id=scope.user.id,
            state=BaseModelAssetState(readiness="ready"),
        )
        db.add(asset)
        references = [
            ArtifactPackageReferenceCreate(
                target_package_id=base.id, relationship_type="uses_base_model"
            )
        ]
    package = service.publish_artifact_package(
        db,
        object_storage,
        project_id=scope.project.id,
        data=_package_spec("manager-only-checkpoint").model_copy(
            update={
                "license_info": {"access_mode": mode},
                "references": references,
                "model_type": model_type,
                "package_kind": "peft_adapter" if kind == "adapter" else "trained_model",
            }
        ),
        files=_package_files()
        + (
            [
                service.PackageFileUpload(
                    "adapter_config.json",
                    b'{"peft_type":"LORA","task_type":"CAUSAL_LM"}',
                    role="adapter_config",
                ),
                service.PackageFileUpload(
                    "adapter_model.safetensors", b"adapter-weights", role="model_weights"
                ),
            ]
            if kind == "adapter"
            else []
        ),
        actor_user_id=scope.user.id,
    )
    original = scope.run.checkpoint
    checkpoint = ModelCheckpoint(
        project_id=scope.project.id,
        training_job_id=original.training_job_id,
        artifact_id=package.lineage_artifact_id,
        model_type=model_type,
        training_mode=original.training_mode,
        trained_target_version_ids=original.trained_target_version_ids,
        max_context_tokens=512,
        manifest={"model_type": model_type},
        readiness="ready",
        package_id=package.id,
    )
    db.add(checkpoint)
    db.flush()
    scope.run.checkpoint_id = checkpoint.id
    scope.run.status = "queued"
    db.commit()
    assert service.artifact_package_access_restrictions(db, package).requires_manager == (
        mode == "manager_only"
    )

    def execute(*args, **kwargs):
        raise HTTPException(status_code=418, detail="Restricted checkpoint reached compute")

    get_descriptor = catalog.get_builtin_model_descriptor

    def ready_descriptor(key):
        descriptor = get_descriptor(key)
        return descriptor.model_copy(
            update={
                "availability": descriptor.availability.model_copy(
                    update={"missing_dependencies": ()}
                ),
            }
        )

    monkeypatch.setattr(catalog, "get_builtin_model_descriptor", ready_descriptor)
    monkeypatch.setattr(inference_router, "execute_backend_inference", execute)
    path = (
        f"/api/projects/{scope.project.id}/inference/runs"
        if operation == "launch"
        else f"/api/inference/runs/{scope.run.id}/execute"
    )
    response = client.post(
        path,
        headers=_headers(scope.user),
        json={
            "name": "restricted-launch",
            "corpus_snapshot_id": scope.run.corpus_snapshot_id,
            "checkpoint_id": checkpoint.id,
            "compute_profile_id": scope.run.compute_profile_id,
            "target_version_ids": [scope.target_version.id],
            "window_config": {"max_tokens": 128, "overlap_tokens": 0},
            "idempotency_key": "review-restricted-launch",
        },
    )
    assert response.status_code == expected, response.text


def test_idempotent_launch_rejects_a_different_requested_checkpoint(
    db,
    client,
    object_storage,
    monkeypatch,
):
    scope = _inference_scope(db, with_window=False)
    original = scope.run.checkpoint
    db.get(Workspace, scope.project.workspace_id).capability_preset = "full"
    db.query(WorkspaceMember).filter(WorkspaceMember.user_id == scope.user.id).update(
        {"role": "trainer"}
    )
    package = service.publish_artifact_package(
        db,
        object_storage,
        project_id=scope.project.id,
        data=_package_spec("restricted-idempotent-model").model_copy(
            update={"license_info": {"access_mode": "manager_only"}}
        ),
        files=_package_files(),
        actor_user_id=scope.user.id,
    )
    checkpoint = ModelCheckpoint(
        project_id=scope.project.id,
        training_job_id=original.training_job_id,
        artifact_id=package.lineage_artifact_id,
        model_type=original.model_type,
        training_mode=original.training_mode,
        trained_target_version_ids=original.trained_target_version_ids,
        max_context_tokens=512,
        manifest=original.manifest,
        readiness="ready",
        package_id=package.id,
    )
    db.add(checkpoint)
    db.flush()
    scope.run.checkpoint_id = checkpoint.id
    scope.run.status = "queued"
    db.commit()

    def execute(*args, **kwargs):
        raise HTTPException(status_code=418, detail="Restricted idempotent run reached compute")

    monkeypatch.setattr(inference_router, "execute_backend_inference", execute)
    response = client.post(
        f"/api/projects/{scope.project.id}/inference/runs",
        headers=_headers(scope.user),
        json={
            "name": "idempotent-access",
            "corpus_snapshot_id": scope.run.corpus_snapshot_id,
            "checkpoint_id": original.id,
            "compute_profile_id": scope.run.compute_profile_id,
            "target_version_ids": [scope.target_version.id],
            "window_config": {"max_tokens": 128, "overlap_tokens": 0},
            "idempotency_key": scope.run.idempotency_key,
        },
    )
    assert response.status_code == 409, response.text
