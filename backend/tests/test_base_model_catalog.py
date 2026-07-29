import hashlib
import json
import struct
from datetime import UTC, datetime, timedelta
from io import BytesIO

import pytest

from al_medlit.auth.models import User
from al_medlit.auth.security import create_access_token
from al_medlit.core.exceptions import ConflictError, ValidationError
from al_medlit.core.storage import ObjectNotFoundError
from al_medlit.lineage.models import AnnotationSet, CorpusSnapshot, LineageArtifact
from al_medlit.model_artifacts.models import (
    ArtifactBlob,
    ArtifactPackage,
    BaseModelAsset,
    BaseModelAssetEvent,
)
from al_medlit.model_artifacts.schemas import (
    ArtifactPackageCreate,
    BaseModelImportCreate,
    BaseModelReadinessUpdate,
    BaseModelUploadCreate,
    BaseModelUploadFile,
)
from al_medlit.model_artifacts.service import (
    PackageFileUpload,
    approve_artifact_package_purge,
    archive_base_model_asset,
    garbage_collect_purged_package_payloads,
    import_base_model_from_package,
    iter_package_file,
    list_base_model_assets,
    migrate_legacy_checkpoints,
    public_base_model_descriptor,
    publish_artifact_package,
    publish_uploaded_base_model,
    request_artifact_package_archive,
    update_base_model_readiness,
)
from al_medlit.project.models import Project
from al_medlit.training.models import (
    ComputeProfile,
    ModelCheckpoint,
    TrainingExperiment,
    TrainingJob,
)
from al_medlit.workspace.models import Workspace, WorkspaceMember


def _project(db, suffix: str):
    user = User(
        username=f"base-model-{suffix}",
        password_hash="unused",
        display_name="Base Model Manager",
    )
    workspace = Workspace(
        name=f"Base model workspace {suffix}",
        kind="team",
        capability_preset="full",
    )
    db.add_all([user, workspace])
    db.flush()
    project = Project(
        name=f"Base model project {suffix}",
        workspace_id=workspace.id,
        annotation_schema={},
        settings={},
    )
    db.add(project)
    db.commit()
    return user, project


def _upload_data(*, access_mode: str = "execution_only", revision: str = "a" * 40):
    return BaseModelUploadCreate(
        provider="huggingface",
        source_model_id="org/tiny-evidence-model",
        exact_revision=revision,
        display_name="Tiny evidence base model",
        model_family="deep_learning",
        model_type="transformer",
        license_name="apache-2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        license_terms_sha256="b" * 64,
        access_mode=access_mode,
        package_format="safetensors",
        files=[
            BaseModelUploadFile(
                relative_path="config.json",
                role="model_config",
                content_type="application/json",
            ),
            BaseModelUploadFile(
                relative_path="tokenizer_config.json",
                role="tokenizer_config",
                content_type="application/json",
            ),
            BaseModelUploadFile(
                relative_path="tokenizer.json",
                role="tokenizer",
                content_type="application/json",
            ),
            BaseModelUploadFile(
                relative_path="model.safetensors",
                role="model_weights",
            ),
        ],
    )


def _upload_sources():
    return [
        BytesIO(b'{"architectures":["TinyModel"],"model_type":"tiny"}'),
        BytesIO(b'{"model_max_length":512,"tokenizer_class":"TinyTokenizerFast"}'),
        BytesIO(
            b'{"model":{"type":"WordLevel","vocab":{"[UNK]":0}},"version":"1.0"}'
        ),
        BytesIO(_safetensors_blob({"encoder.weight": ("F32", [1], b"\x00" * 4)})),
    ]


def _safetensors_blob(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    offset = 0
    header = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        data.extend(payload)
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + bytes(data)


def _raw_safetensors_blob(header: dict, data: bytes) -> bytes:
    header_bytes = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    return struct.pack("<Q", len(header_bytes)) + header_bytes + data


def test_base_model_upload_is_exact_sanitized_and_audited(db, object_storage):
    user, project = _project(db, "upload")
    data = _upload_data()
    asset = publish_uploaded_base_model(
        db,
        object_storage,
        project_id=project.id,
        data=data,
        sources=_upload_sources(),
        actor_user_id=user.id,
    )
    db.commit()

    descriptor = public_base_model_descriptor(asset)
    assert descriptor.exact_revision == "a" * 40
    assert descriptor.readiness == "ready"
    assert descriptor.package.package_kind == "base_model"
    assert descriptor.package.deployable is True
    assert descriptor.package.license_info["access_mode"] == "execution_only"
    assert "storage_key" not in descriptor.model_dump_json()
    assert [event.action for event in asset.events] == ["uploaded"]

    repeated = publish_uploaded_base_model(
        db,
        object_storage,
        project_id=project.id,
        data=data,
        sources=_upload_sources(),
        actor_user_id=user.id,
    )
    assert repeated.id == asset.id

    update_base_model_readiness(
        db,
        asset=asset,
        data=BaseModelReadinessUpdate(
            readiness="quarantined",
            reason="License review requested",
        ),
        actor_user_id=user.id,
    )
    archive_base_model_asset(db, asset=asset, actor_user_id=user.id)
    db.commit()
    assert asset.state.readiness == "archived"
    assert [event.action for event in asset.events] == [
        "uploaded",
        "readiness_changed",
        "archived",
    ]
    assert db.query(BaseModelAssetEvent).count() == 3


@pytest.mark.parametrize(
    "invalid_weights",
    [
        b"this-is-not-a-safetensors-file",
        _raw_safetensors_blob(
            {
                "encoder.weight": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8],
                }
            },
            b"\x00" * 4,
        ),
    ],
    ids=["mislabeled-bytes", "offset-past-end"],
)
def test_uploaded_base_model_rejects_invalid_safetensors_before_publication(
    db,
    object_storage,
    invalid_weights,
):
    user, project = _project(db, f"invalid-safe-{len(invalid_weights)}")
    sources = _upload_sources()
    sources[-1] = BytesIO(invalid_weights)

    with pytest.raises(ValidationError, match="Safetensors"):
        publish_uploaded_base_model(
            db,
            object_storage,
            project_id=project.id,
            data=_upload_data(revision=hashlib.sha256(invalid_weights).hexdigest()),
            sources=sources,
            actor_user_id=user.id,
        )

    assert db.query(ArtifactPackage).count() == 0
    assert db.query(BaseModelAsset).count() == 0


def test_uploaded_safetensors_base_model_requires_hf_tokenizer_metadata(
    db,
    object_storage,
):
    user, project = _project(db, "missing-tokenizer")
    data = _upload_data(revision="d" * 40)
    data.files = [
        descriptor
        for descriptor in data.files
        if descriptor.relative_path != "tokenizer_config.json"
    ]
    sources = [
        source
        for descriptor, source in zip(_upload_data().files, _upload_sources(), strict=True)
        if descriptor.relative_path != "tokenizer_config.json"
    ]

    with pytest.raises(ValidationError, match="tokenizer_config.json"):
        publish_uploaded_base_model(
            db,
            object_storage,
            project_id=project.id,
            data=data,
            sources=sources,
            actor_user_id=user.id,
        )


def test_valid_sharded_multifile_base_model_upload_becomes_ready(db, object_storage):
    user, project = _project(db, "valid-shards")
    shard_one = _safetensors_blob({"encoder.weight": ("F32", [1], b"\x01" * 4)})
    shard_two = _safetensors_blob({"classifier.bias": ("F32", [1], b"\x02" * 4)})
    index = json.dumps(
        {
            "metadata": {"total_size": 8},
            "weight_map": {
                "classifier.bias": "model-00002-of-00002.safetensors",
                "encoder.weight": "model-00001-of-00002.safetensors",
            },
        }
    ).encode()
    files = [
        BaseModelUploadFile(
            relative_path="config.json",
            role="model_config",
            content_type="application/json",
        ),
        BaseModelUploadFile(
            relative_path="tokenizer_config.json",
            role="tokenizer_config",
            content_type="application/json",
        ),
        BaseModelUploadFile(
            relative_path="tokenizer.json",
            role="tokenizer",
            content_type="application/json",
        ),
        BaseModelUploadFile(
            relative_path="model.safetensors.index.json",
            role="model_index",
            content_type="application/json",
        ),
        BaseModelUploadFile(
            relative_path="model-00001-of-00002.safetensors",
            role="model_weights",
        ),
        BaseModelUploadFile(
            relative_path="model-00002-of-00002.safetensors",
            role="model_weights",
        ),
    ]
    data = BaseModelUploadCreate(
        provider="huggingface",
        source_model_id="org/tiny-sharded-model",
        exact_revision="e" * 40,
        display_name="Tiny sharded model",
        model_family="llm_finetune",
        model_type="transformer",
        license_name="apache-2.0",
        access_mode="execution_only",
        package_format="safetensors",
        files=files,
    )
    asset = publish_uploaded_base_model(
        db,
        object_storage,
        project_id=project.id,
        data=data,
        sources=[
            BytesIO(b'{"architectures":["TinyModel"],"model_type":"tiny"}'),
            BytesIO(b'{"model_max_length":512,"tokenizer_class":"TinyTokenizerFast"}'),
            BytesIO(
                b'{"model":{"type":"WordLevel","vocab":{"[UNK]":0}},'
                b'"version":"1.0"}'
            ),
            BytesIO(index),
            BytesIO(shard_one),
            BytesIO(shard_two),
        ],
        actor_user_id=user.id,
    )
    db.commit()

    assert asset.state.readiness == "ready"
    assert asset.package.readiness == "ready"
    assert asset.package.deployable is True
    assert asset.package.file_count == 6


def test_active_base_model_catalog_entry_protects_package_retention(db, object_storage):
    user, project = _project(db, "retention")
    asset = publish_uploaded_base_model(
        db,
        object_storage,
        project_id=project.id,
        data=_upload_data(revision="c" * 40),
        sources=_upload_sources(),
        actor_user_id=user.id,
    )
    db.commit()
    with pytest.raises(ConflictError, match="active base-model catalog entry"):
        request_artifact_package_archive(
            db,
            package=asset.package,
            actor_user_id=user.id,
        )
    db.rollback()

    archive_base_model_asset(db, asset=asset, actor_user_id=user.id)
    retention = request_artifact_package_archive(
        db,
        package=asset.package,
        actor_user_id=user.id,
    )
    assert retention.archived_at is not None


def test_base_model_import_reuses_blobs_and_cannot_broaden_license(db, object_storage):
    user, project = _project(db, "import")
    source = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=ArtifactPackageCreate(
            package_kind="trained_model",
            package_format="safetensors",
            display_name="Source",
            model_family="deep_learning",
            model_type="transformer",
            readiness="ready",
            deployable=True,
            license_info={"access_mode": "execution_only"},
        ),
        files=[
            PackageFileUpload(
                "model.safetensors",
                b"source-model",
                role="model_weights",
            )
        ],
        actor_user_id=user.id,
    )
    before_blob_count = db.query(ArtifactBlob).count()
    common = dict(
        source_package_id=source.id,
        provider="internal",
        source_model_id="lab/evidence-base",
        exact_revision="revision-2026-07-22",
        display_name="Lab evidence base",
        model_family="deep_learning",
        model_type="transformer",
        license_name="lab-research-only",
    )
    with pytest.raises(ConflictError, match="broaden"):
        import_base_model_from_package(
            db,
            object_storage,
            project_id=project.id,
            data=BaseModelImportCreate(**common, access_mode="downloadable"),
            actor_user_id=user.id,
        )

    asset = import_base_model_from_package(
        db,
        object_storage,
        project_id=project.id,
        data=BaseModelImportCreate(**common, access_mode="manager_only"),
        actor_user_id=user.id,
    )
    db.commit()
    source_file = source.files[0]
    imported_file = asset.package.files[0]
    assert imported_file.blob_id == source_file.blob_id
    assert db.query(ArtifactBlob).count() == before_blob_count + 1  # new manifest only
    assert asset.package.outgoing_references[0].target_package_id == source.id
    assert list_base_model_assets(
        db,
        project_id=project.id,
        include_manager_only=False,
    ) == []


def _legacy_checkpoint(db, object_storage, user, project, payload: bytes):
    corpus_artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="corpus_snapshot",
        content_hash="c" * 64,
        storage_key=f"legacy/{project.id}/corpus.json",
        content_type="application/json",
        size_bytes=1,
        manifest={},
    )
    annotation_artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="annotation_set",
        content_hash="d" * 64,
        storage_key=f"legacy/{project.id}/annotations.json",
        content_type="application/json",
        size_bytes=1,
        manifest={},
    )
    db.add_all([corpus_artifact, annotation_artifact])
    db.flush()
    snapshot = CorpusSnapshot(
        project_id=project.id,
        artifact_id=corpus_artifact.id,
        name="legacy",
        document_count=0,
    )
    db.add(snapshot)
    db.flush()
    annotation_set = AnnotationSet(
        project_id=project.id,
        artifact_id=annotation_artifact.id,
        corpus_snapshot_id=snapshot.id,
        name="legacy",
        target_version_ids=[],
        block_count=0,
        reviewed_region_count=0,
    )
    profile = ComputeProfile(
        project_id=project.id,
        name="legacy-local",
        backend="local",
        config={},
        created_by_user_id=user.id,
    )
    db.add_all([annotation_set, profile])
    db.flush()
    experiment = TrainingExperiment(
        project_id=project.id,
        annotation_set_id=annotation_set.id,
        compute_profile_id=profile.id,
        name="legacy experiment",
        model_type="evidence_block_sentence_tagger",
        mode="conditioned",
        target_version_ids=[],
        config={},
        idempotency_key=f"legacy-experiment-{project.id}",
        created_by_user_id=user.id,
    )
    db.add(experiment)
    db.flush()
    job = TrainingJob(
        experiment_id=experiment.id,
        compute_profile_id=profile.id,
        idempotency_key=f"legacy-job-{project.id}",
        status="succeeded",
    )
    db.add(job)
    db.flush()

    key = f"legacy/{project.id}/checkpoint.zip"
    object_storage.put_bytes(key, payload, content_type="application/zip")
    artifact = LineageArtifact(
        project_id=project.id,
        artifact_type="model_checkpoint",
        content_hash=hashlib.sha256(payload).hexdigest(),
        storage_key=key,
        content_type="application/zip",
        size_bytes=len(payload),
        manifest={},
        created_by_user_id=user.id,
    )
    db.add(artifact)
    db.flush()
    checkpoint = ModelCheckpoint(
        project_id=project.id,
        training_job_id=job.id,
        artifact_id=artifact.id,
        model_type="evidence_block_sentence_tagger",
        training_mode="conditioned",
        trained_target_version_ids=[],
        max_context_tokens=512,
        manifest={},
        readiness="legacy_unverified",
    )
    db.add(checkpoint)
    db.commit()
    return checkpoint


def test_legacy_zip_migration_is_byte_preserving_and_idempotent(db, object_storage):
    user, project = _project(db, "legacy")
    payload = b"PK\x03\x04old checkpoint bytes stay exactly unchanged\x00\xff"
    checkpoint = _legacy_checkpoint(db, object_storage, user, project, payload)

    first = migrate_legacy_checkpoints(
        db,
        object_storage,
        project_id=project.id,
        checkpoint_id=checkpoint.id,
        actor_user_id=user.id,
    )
    db.commit()
    assert first.migrated_count == 1
    assert first.items[0].size_bytes == len(payload)

    db.refresh(checkpoint)
    assert checkpoint.package_id == first.items[0].package_id
    assert checkpoint.package.package_format == "legacy_zip"
    assert checkpoint.package.readiness == "legacy_unverified"
    assert checkpoint.package.deployable is False
    _, chunks = iter_package_file(
        db,
        object_storage,
        project_id=project.id,
        package_id=checkpoint.package_id,
        relative_path="checkpoint.zip",
    )
    assert b"".join(chunks) == payload

    repeated = migrate_legacy_checkpoints(
        db,
        object_storage,
        project_id=project.id,
        checkpoint_id=checkpoint.id,
        actor_user_id=user.id,
    )
    assert repeated.migrated_count == 0
    assert repeated.existing_count == 1
    assert repeated.items[0].package_id == checkpoint.package_id

    archive_time = datetime.now(UTC) - timedelta(days=8)
    request_artifact_package_archive(
        db,
        package=checkpoint.package,
        actor_user_id=user.id,
        reason="Retire legacy checkpoint",
        now=archive_time,
    )
    approve_artifact_package_purge(
        db,
        package=checkpoint.package,
        actor_user_id=user.id,
        reason="Approved legacy checkpoint purge",
    )
    db.commit()
    result = garbage_collect_purged_package_payloads(
        db,
        object_storage,
        workspace_id=project.workspace_id,
    )
    db.commit()
    assert result.processed_package_count == 1
    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(checkpoint.artifact.storage_key)
    assert db.get(LineageArtifact, checkpoint.artifact_id) is not None
    assert checkpoint.package.retention.payload_gc_processed_at is not None


def test_base_model_api_enforces_manager_mutation_and_license_visibility(
    db,
    client,
    object_storage,
):
    manager, project = _project(db, "api")
    trainer = User(
        username="base-model-api-trainer",
        password_hash="unused",
        display_name="Trainer",
    )
    db.add(trainer)
    db.flush()
    db.add_all(
        [
            WorkspaceMember(
                workspace_id=project.workspace_id,
                user_id=manager.id,
                role="manager",
            ),
            WorkspaceMember(
                workspace_id=project.workspace_id,
                user_id=trainer.id,
                role="trainer",
            ),
        ]
    )
    source = publish_artifact_package(
        db,
        object_storage,
        project_id=project.id,
        data=ArtifactPackageCreate(
            package_kind="trained_model",
            package_format="safetensors",
            model_family="deep_learning",
            model_type="transformer",
            readiness="ready",
            deployable=True,
        ),
        files=[
            PackageFileUpload(
                "model.safetensors",
                b"api-source-model",
                role="model_weights",
            )
        ],
        actor_user_id=manager.id,
    )
    db.commit()
    payload = {
        "source_package_id": source.id,
        "provider": "internal",
        "source_model_id": "lab/api-model",
        "exact_revision": hashlib.sha256(b"api-source-model").hexdigest(),
        "display_name": "API model",
        "model_family": "deep_learning",
        "model_type": "transformer",
        "license_name": "research-only",
        "access_mode": "manager_only",
    }
    trainer_headers = {"Authorization": f"Bearer {create_access_token(trainer.id)}"}
    manager_headers = {"Authorization": f"Bearer {create_access_token(manager.id)}"}

    denied = client.post(
        f"/api/projects/{project.id}/base-models/import",
        json=payload,
        headers=trainer_headers,
    )
    assert denied.status_code == 403
    created = client.post(
        f"/api/projects/{project.id}/base-models/import",
        json=payload,
        headers=manager_headers,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["access_mode"] == "manager_only"
    assert "storage_key" not in json.dumps(body)

    trainer_list = client.get(
        f"/api/projects/{project.id}/base-models",
        headers=trainer_headers,
    )
    assert trainer_list.status_code == 200
    assert trainer_list.json() == []
    manager_list = client.get(
        f"/api/projects/{project.id}/base-models",
        headers=manager_headers,
    )
    assert [item["id"] for item in manager_list.json()] == [body["id"]]

    upload_metadata = {
        "provider": "internal",
        "source_model_id": "lab/uploaded-model",
        "exact_revision": "upload-revision-1",
        "display_name": "Uploaded API model",
        "model_family": "deep_learning",
        "model_type": "transformer",
        "license_name": "apache-2.0",
        "access_mode": "downloadable",
        "package_format": "safetensors",
        "files": [
            {
                "relative_path": "config.json",
                "role": "model_config",
                "content_type": "application/json",
            },
            {
                "relative_path": "tokenizer_config.json",
                "role": "tokenizer_config",
                "content_type": "application/json",
            },
            {
                "relative_path": "tokenizer.json",
                "role": "tokenizer",
                "content_type": "application/json",
            },
            {
                "relative_path": "model.safetensors",
                "role": "model_weights",
            },
        ],
    }
    uploaded = client.post(
        f"/api/projects/{project.id}/base-models/upload",
        data={"metadata": json.dumps(upload_metadata)},
        files=[
            (
                "files",
                (
                    "config.json",
                    b'{"architectures":["TinyModel"],"model_type":"tiny"}',
                    "application/json",
                ),
            ),
            (
                "files",
                (
                    "tokenizer_config.json",
                    b'{"model_max_length":512,"tokenizer_class":"TinyTokenizerFast"}',
                    "application/json",
                ),
            ),
            (
                "files",
                (
                    "tokenizer.json",
                    b'{"model":{"type":"WordLevel","vocab":{"[UNK]":0}},'
                    b'"version":"1.0"}',
                    "application/json",
                ),
            ),
            (
                "files",
                (
                    "model.safetensors",
                    _safetensors_blob(
                        {"encoder.weight": ("F32", [1], b"\x00" * 4)}
                    ),
                    "application/octet-stream",
                ),
            ),
        ],
        headers=manager_headers,
    )
    assert uploaded.status_code == 201, uploaded.text
    assert uploaded.json()["exact_revision"] == "upload-revision-1"
