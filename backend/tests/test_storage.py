"""Unit tests for the object-storage abstraction (local backend only)."""

import hashlib
import io

import pytest

from al_medlit.core.storage import (
    LocalObjectStorage,
    MinioObjectStorage,
    ObjectNotFoundError,
    ObjectStorageError,
)


def test_put_get_roundtrip(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    storage.put_bytes("projects/1/documents/2/submission-abc.json", b'{"a": 1}')
    assert storage.get_bytes("projects/1/documents/2/submission-abc.json") == b'{"a": 1}'


def test_get_missing_key_raises(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    with pytest.raises(ObjectNotFoundError):
        storage.get_bytes("projects/1/missing.json")


def test_delete_is_idempotent(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    storage.put_bytes("a/b.json", b"x")
    storage.delete("a/b.json")
    storage.delete("a/b.json")
    with pytest.raises(ObjectNotFoundError):
        storage.get_bytes("a/b.json")


def test_path_traversal_key_rejected(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    with pytest.raises(ObjectStorageError):
        storage.put_bytes("../escape.json", b"x")


def test_build_object_storage_local(monkeypatch, tmp_path):
    from al_medlit.core import storage as storage_module

    monkeypatch.setattr(storage_module.settings, "storage_backend", "local")
    monkeypatch.setattr(storage_module.settings, "storage_local_dir", str(tmp_path))
    built = storage_module.build_object_storage()
    assert isinstance(built, LocalObjectStorage)


def test_build_object_storage_unknown_backend(monkeypatch):
    from al_medlit.core import storage as storage_module

    monkeypatch.setattr(storage_module.settings, "storage_backend", "unknown-backend")
    with pytest.raises(ObjectStorageError):
        storage_module.build_object_storage()


def test_streaming_roundtrip_calculates_checksum_and_downloads_atomically(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    payload = b"abcdefghij" * 1000
    stored = storage.put_stream(
        "datasets/windows.jsonl",
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/x-ndjson",
        chunk_size=37,
    )

    assert stored.size_bytes == len(payload)
    assert stored.checksum_sha256 == hashlib.sha256(payload).hexdigest()
    assert b"".join(storage.iter_bytes(stored.key, chunk_size=41)) == payload

    destination = tmp_path / "download" / "windows.jsonl"
    downloaded = storage.download_file(stored.key, destination)
    assert destination.read_bytes() == payload
    assert downloaded.checksum_sha256 == stored.checksum_sha256


def test_stream_length_mismatch_does_not_publish_partial_object(tmp_path):
    storage = LocalObjectStorage(tmp_path / "objects")
    with pytest.raises(ObjectStorageError, match="length mismatch"):
        storage.put_stream("bad.bin", io.BytesIO(b"short"), length=100)
    with pytest.raises(ObjectNotFoundError):
        storage.get_bytes("bad.bin")


def test_minio_sse_s3_is_applied_to_each_upload(monkeypatch):
    class FakeMinio:
        def __init__(self, *_args, **_kwargs):
            self.put_kwargs = None

        def bucket_exists(self, _bucket):
            return True

        def put_object(self, _bucket, _key, stream, **kwargs):
            stream.read()
            self.put_kwargs = kwargs

    monkeypatch.setattr("minio.Minio", FakeMinio)
    storage = MinioObjectStorage(
        endpoint="storage.test",
        access_key="access",
        secret_key="secret",
        bucket="models",
        secure=True,
        encryption_mode="sse-s3",
    )

    storage.put_stream("model.bin", io.BytesIO(b"model"), length=5)

    assert storage.encryption_mode == "sse-s3"
    assert type(storage.client.put_kwargs["sse"]).__name__ == "SseS3"


def test_minio_sse_kms_requires_and_applies_the_configured_key(monkeypatch):
    class FakeMinio:
        def __init__(self, *_args, **_kwargs):
            self.put_kwargs = None

        def bucket_exists(self, _bucket):
            return True

        def put_object(self, _bucket, _key, stream, **kwargs):
            stream.read()
            self.put_kwargs = kwargs

    monkeypatch.setattr("minio.Minio", FakeMinio)
    with pytest.raises(ObjectStorageError, match="requires a KMS key"):
        MinioObjectStorage(
            endpoint="storage.test",
            access_key="access",
            secret_key="secret",
            bucket="models",
            secure=True,
            encryption_mode="sse-kms",
        )

    storage = MinioObjectStorage(
        endpoint="storage.test",
        access_key="access",
        secret_key="secret",
        bucket="models",
        secure=True,
        encryption_mode="sse-kms",
        kms_key_id="models-key",
    )
    storage.put_stream("model.bin", io.BytesIO(b"model"), length=5)

    assert storage.encryption_key_id == "models-key"
    assert type(storage.client.put_kwargs["sse"]).__name__ == "SseKMS"
