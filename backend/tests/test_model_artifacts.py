import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from al_medlit.auth.models import User
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import ObjectNotFoundError
from al_medlit.lineage.models import LineageEdge

# Import before the shared create_all fixture builds metadata. Production registration
# and Alembic integration are intentionally left to the parent implementation task.
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    ArtifactPackageFile,
)
from al_medlit.model_artifacts.schemas import (
    ArtifactPackageCreate,
    ArtifactPackageReferenceCreate,
)
from al_medlit.model_artifacts.service import (
    PackageFileUpload,
    approve_artifact_package_purge,
    artifact_storage_usage,
    garbage_collect_artifact_blobs,
    iter_package_file,
    public_package_descriptor,
    publish_artifact_package,
    request_artifact_package_archive,
    update_artifact_package_retention,
    verify_artifact_package,
)
from al_medlit.project.models import Project
from al_medlit.workspace.models import Workspace


def _workspace_project(db, *, suffix: str = "one", workspace=None):
    user = User(
        username=f"artifact-user-{suffix}",
        password_hash="not-used",
        display_name="Artifact User",
    )
    db.add(user)
    db.flush()
    if workspace is None:
        workspace = Workspace(
            name=f"Artifact workspace {suffix}",
            kind="individual",
            created_by=user.id,
        )
        db.add(workspace)
        db.flush()
    project = Project(
        name=f"Artifact project {suffix}",
        workspace_id=workspace.id,
        annotation_schema={},
        settings={},
    )
    db.add(project)
    db.commit()
    return user, workspace, project


def _neural_spec(**overrides):
    values = {
        "package_kind": "trained_model",
        "package_format": "safetensors",
        "display_name": "Evidence transformer",
        "model_family": "deep_learning",
        "model_type": "evidence_transformer",
        "readiness": "ready",
        "deployable": True,
        "task_contract": {
            "task": "evidence_block",
            "prediction_schema": "evidence-block-v1",
        },
        "runtime": {"transformers": "4.42.0"},
    }
    values.update(overrides)
    return ArtifactPackageCreate(**values)


def _neural_files():
    return [
        PackageFileUpload(
            "config.json",
            json.dumps({"architectures": ["EvidenceModel"]}).encode(),
            role="model_config",
            content_type="application/json",
        ),
        PackageFileUpload(
            "model.safetensors",
            b"small-safe-tensor-fixture",
            role="model_weights",
        ),
    ]


def test_publish_package_is_immutable_sanitized_and_verifiable(db, object_storage):
    user, _workspace, project = _workspace_project(db)
    spec = _neural_spec()
    files = _neural_files()
    package = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=spec,
        files=files,
        actor_user_id=user.id,
    )
    db.commit()

    blob_count = db.query(ArtifactBlob).count()
    repeated = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=spec,
        files=files,
        actor_user_id=user.id,
    )
    assert repeated.id == package.id
    assert db.query(ArtifactBlob).count() == blob_count

    assert package.file_count == 2
    assert package.logical_size_bytes == sum(item.size_bytes for item in package.files)
    assert package.lineage_artifact.content_hash == package.manifest_digest
    assert package.retention.retention_class == "indefinite"
    verify_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        package_id=package.id,
    )

    descriptor = public_package_descriptor(package)
    serialized = descriptor.model_dump_json()
    assert "storage_key" not in serialized
    assert "artifact-blobs" not in serialized
    assert descriptor.files[1].relative_path == "model.safetensors"

    package_file, chunks = iter_package_file(
        db,
        object_storage,
        project_id=project.id,
        package_id=package.id,
        relative_path="model.safetensors",
    )
    assert package_file.checksum_sha256 == hashlib.sha256(
        b"small-safe-tensor-fixture"
    ).hexdigest()
    assert b"".join(chunks) == b"small-safe-tensor-fixture"


def test_blobs_deduplicate_across_projects_only_inside_workspace(db, object_storage):
    user, workspace, first_project = _workspace_project(db, suffix="first")
    second_user, _workspace, second_project = _workspace_project(
        db,
        suffix="second",
        workspace=workspace,
    )
    first = publish_artifact_package(
        db,
        object_storage,
        project_id=first_project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    second = publish_artifact_package(
        db,
        object_storage,
        project_id=second_project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=second_user.id,
    )
    db.commit()

    first_weights = next(item for item in first.files if item.role == "model_weights")
    second_weights = next(item for item in second.files if item.role == "model_weights")
    assert first_weights.blob_id == second_weights.blob_id
    assert first.manifest_digest != second.manifest_digest

    usage = artifact_storage_usage(
        db,
        project_id=first_project.id,
        include_workspace_physical=True,
    )
    assert usage.package_count == 1
    assert usage.workspace_deduplicated_bytes == sum(item.size_bytes for item in first.files)
    assert usage.workspace_reclaimable_bytes == 0

    archive_time = datetime.now(UTC) - timedelta(days=8)
    request_artifact_package_archive(
        db,
        package=first,
        actor_user_id=user.id,
        reason="Retire first project copy",
        now=archive_time,
    )
    approve_artifact_package_purge(
        db,
        package=first,
        actor_user_id=user.id,
        reason="Approved first project purge",
    )
    db.commit()
    gc_result = garbage_collect_artifact_blobs(
        db,
        object_storage,
        workspace_id=workspace.id,
    )
    db.commit()
    assert gc_result.deleted_blob_count == 1  # only the first package manifest
    db.refresh(first_weights)
    assert first_weights.blob.status == "ready"
    assert object_storage.get_bytes(first_weights.blob.storage_key) == (
        b"small-safe-tensor-fixture"
    )
    assert second.retention.purged_at is None

    # Identical bytes in a different workspace receive a different physical record/key.
    third_user, _third_workspace, third_project = _workspace_project(db, suffix="third")
    third = publish_artifact_package(
        db,
        object_storage,
        project_id=third_project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=third_user.id,
    )
    db.commit()
    third_weights = next(item for item in third.files if item.role == "model_weights")
    assert third_weights.blob_id != first_weights.blob_id


@pytest.mark.parametrize(
    ("upload", "message"),
    [
        (PackageFileUpload("../escape.safetensors", b"x"), "path"),
        (PackageFileUpload("model.pkl", b"x"), "unsafe"),
        (
            PackageFileUpload("model.safetensors", b"x", expected_checksum_sha256="0" * 64),
            "Checksum mismatch",
        ),
    ],
)
def test_manifest_path_format_and_hash_validation(db, object_storage, upload, message):
    suffix = hashlib.md5(message.encode()).hexdigest()
    user, _workspace, project = _workspace_project(db, suffix=suffix)
    with pytest.raises((ValidationError, ConflictError), match=message):
        publish_artifact_package(
            db,
            object_storage,
            project_id=project.id,
            data=_neural_spec(),
            files=[upload],
            actor_user_id=user.id,
        )
    assert db.query(ArtifactPackage).count() == 0


def test_peft_adapter_requires_and_records_immutable_base_reference(db, object_storage):
    user, _workspace, project = _workspace_project(db, suffix="peft")
    base = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(package_kind="base_model", model_family="llm_finetune"),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    adapter = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=ArtifactPackageCreate(
            package_kind="peft_adapter",
            package_format="peft-adapter",
            model_family="llm_finetune",
            model_type="lora",
            deployable=True,
            task_contract={"task": "evidence_block"},
            references=[
                ArtifactPackageReferenceCreate(
                    target_package_id=base.id,
                    relationship_type="uses_base_model",
                )
            ],
            retention_class="candidate",
        ),
        files=[
            PackageFileUpload(
                "adapter_config.json",
                b'{"peft_type":"LORA"}',
                role="adapter_config",
                content_type="application/json",
            ),
            PackageFileUpload(
                "adapter_model.safetensors",
                b"adapter",
                role="adapter_weights",
            ),
        ],
        actor_user_id=user.id,
    )
    db.commit()

    descriptor = public_package_descriptor(adapter)
    assert descriptor.references[0].target_manifest_digest == base.manifest_digest
    assert descriptor.retention.pinned is True
    edge = (
        db.query(LineageEdge)
        .filter(
            LineageEdge.upstream_artifact_id == base.lineage_artifact_id,
            LineageEdge.downstream_artifact_id == adapter.lineage_artifact_id,
            LineageEdge.relationship_type == "uses_base_model",
        )
        .one()
    )
    assert edge.metadata_ == {}


def test_failed_publication_rolls_back_database_references(db, object_storage, monkeypatch):
    user, _workspace, project = _workspace_project(db, suffix="failure")
    original = object_storage.put_stream
    calls = 0

    def fail_second_upload(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated object store outage")
        return original(*args, **kwargs)

    monkeypatch.setattr(object_storage, "put_stream", fail_second_upload)
    with pytest.raises(RuntimeError, match="outage"):
        publish_artifact_package(
            db,
            object_storage,
            project_id=project.id,
            data=_neural_spec(),
            files=_neural_files(),
            actor_user_id=user.id,
        )
    assert db.query(ArtifactPackage).count() == 0
    assert db.query(ArtifactBlob).count() == 0


def test_archive_purge_gc_preserves_tombstone_and_rehydrates_by_hash(
    db,
    object_storage,
):
    user, workspace, project = _workspace_project(db, suffix="purge")
    package = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    db.commit()
    blob_ids = {package.manifest_blob_id, *(item.blob_id for item in package.files)}
    blob_keys = {
        blob.storage_key
        for blob in db.query(ArtifactBlob).filter(ArtifactBlob.id.in_(blob_ids)).all()
    }
    archive_time = datetime.now(UTC) - timedelta(days=8)
    retention = request_artifact_package_archive(
        db,
        package=package,
        actor_user_id=user.id,
        reason="No longer an approved candidate",
        now=archive_time,
    )
    assert retention.purge_after == archive_time + timedelta(days=7)
    approve_artifact_package_purge(
        db,
        package=package,
        actor_user_id=user.id,
        reason="Manager approved storage purge",
    )
    db.commit()

    gc_result = garbage_collect_artifact_blobs(
        db,
        object_storage,
        workspace_id=workspace.id,
    )
    db.commit()
    assert gc_result.deleted_blob_count == len(blob_ids)
    assert {
        blob.status
        for blob in db.query(ArtifactBlob).filter(ArtifactBlob.id.in_(blob_ids)).all()
    } == {"purged"}
    for storage_key in blob_keys:
        with pytest.raises(ObjectNotFoundError):
            object_storage.get_bytes(storage_key)

    # Immutable package, file, and lineage rows remain as the audit tombstone.
    assert db.get(ArtifactPackage, package.id) is not None
    assert (
        db.query(ArtifactPackageFile)
        .filter(ArtifactPackageFile.package_id == package.id)
        .count()
        == 2
    )
    assert package.lineage_artifact is not None
    descriptor = public_package_descriptor(package)
    assert descriptor.retention.purged_at is not None
    assert descriptor.retention.purge_reason == "Manager approved storage purge"
    usage = artifact_storage_usage(
        db,
        project_id=project.id,
        include_workspace_physical=True,
    )
    assert usage.package_count == 0
    assert usage.logical_bytes == 0
    assert usage.workspace_physical_bytes == 0
    with pytest.raises(ConflictError, match="purged"):
        iter_package_file(
            db,
            object_storage,
            project_id=project.id,
            package_id=package.id,
            relative_path="model.safetensors",
        )
    with pytest.raises(ConflictError, match="cannot be republished"):
        publish_artifact_package(
            db,
            object_storage,
            project_id=project.id,
            data=_neural_spec(),
            files=_neural_files(),
            actor_user_id=user.id,
        )

    # A different project in the same trusted workspace may reuse the checksum;
    # publication rehydrates the tombstoned physical objects without reviving the
    # purged project's package.
    second_user, _workspace, second_project = _workspace_project(
        db,
        suffix="purge-rehydrate",
        workspace=workspace,
    )
    replacement = publish_artifact_package(
        db,
        object_storage,
        project_id=second_project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=second_user.id,
    )
    db.commit()
    assert replacement.id != package.id
    assert {item.blob.status for item in replacement.files} == {"ready"}
    assert package.retention.purged_at is not None


def test_purge_requires_grace_and_archive_refuses_live_protections(db, object_storage):
    user, workspace, project = _workspace_project(db, suffix="retention-blockers")
    package = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    db.commit()
    request_artifact_package_archive(
        db,
        package=package,
        actor_user_id=user.id,
    )
    with pytest.raises(ConflictError, match="grace period"):
        approve_artifact_package_purge(
            db,
            package=package,
            actor_user_id=user.id,
            reason="Too early purge attempt",
        )
    db.rollback()

    pinned = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(display_name="Pinned", pinned=True),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    db.commit()
    with pytest.raises(ConflictError, match="package pin"):
        request_artifact_package_archive(
            db,
            package=pinned,
            actor_user_id=user.id,
        )
    db.rollback()

    held = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(display_name="Legally held"),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    update_artifact_package_retention(
        db,
        package=held,
        actor_user_id=user.id,
        legal_hold=True,
    )
    db.commit()
    with pytest.raises(ConflictError, match="legal hold"):
        request_artifact_package_archive(
            db,
            package=held,
            actor_user_id=user.id,
        )
    db.rollback()

    base = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=_neural_spec(display_name="Referenced base"),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    _second_user, _workspace, second_project = _workspace_project(
        db,
        suffix="retention-dependent",
        workspace=workspace,
    )
    publish_artifact_package(
        db,
        object_storage,
        project_id=second_project.id,
        data=_neural_spec(
            display_name="Dependent package",
            references=[
                ArtifactPackageReferenceCreate(
                    target_package_id=base.id,
                    relationship_type="uses_base_model",
                )
            ],
        ),
        files=_neural_files(),
        actor_user_id=user.id,
    )
    db.commit()
    with pytest.raises(ConflictError, match="package dependency"):
        request_artifact_package_archive(
            db,
            package=base,
            actor_user_id=user.id,
        )
