"""Object storage abstraction.

MinIO is the canonical backend (see infra/docker-compose.yml). The local
filesystem backend exists for unit tests and docker-less laptop development.
"""

import hashlib
import io
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from al_medlit.core.config import settings


class ObjectStorageError(RuntimeError):
    pass


class ObjectNotFoundError(ObjectStorageError):
    pass


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Metadata calculated while an object is streamed into storage."""

    key: str
    size_bytes: int
    checksum_sha256: str
    content_type: str


class ObjectStorage(Protocol):
    backend_name: str
    encryption_mode: str
    encryption_key_id: str | None

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None: ...

    def get_bytes(self, key: str) -> bytes: ...

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = 1024 * 1024,
    ) -> StoredObject: ...

    def iter_bytes(self, key: str, *, chunk_size: int = 1024 * 1024) -> Iterator[bytes]: ...

    def put_file(
        self,
        key: str,
        source: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> StoredObject: ...

    def download_file(self, key: str, destination: str | Path) -> StoredObject: ...

    def delete(self, key: str) -> None: ...


class LocalObjectStorage:
    backend_name = "local"
    encryption_mode = "none"
    encryption_key_id = None

    def __init__(self, root_dir: str | Path) -> None:
        self.root = Path(root_dir)

    def _path(self, key: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ObjectStorageError(f"Invalid object key: {key!r}")
        return path

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.put_stream(
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def get_bytes(self, key: str) -> bytes:
        return b"".join(self.iter_bytes(key))

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = 1024 * 1024,
    ) -> StoredObject:
        if chunk_size <= 0:
            raise ObjectStorageError("chunk_size must be positive")
        if length is not None and length < 0:
            raise ObjectStorageError("length must be non-negative")

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as target:
                temporary_path = Path(target.name)
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise ObjectStorageError("Object streams must yield bytes")
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if length is not None and size != length:
                raise ObjectStorageError(
                    f"Stream length mismatch for {key!r}: expected {length}, received {size}"
                )
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return StoredObject(
            key=key,
            size_bytes=size,
            checksum_sha256=digest.hexdigest(),
            content_type=content_type,
        )

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        if chunk_size <= 0:
            raise ObjectStorageError("chunk_size must be positive")
        path = self._path(key)
        if not path.is_file():
            raise ObjectNotFoundError(f"Object not found: {key!r}")
        with path.open("rb") as source:
            while chunk := source.read(chunk_size):
                yield chunk

    def put_file(
        self,
        key: str,
        source: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> StoredObject:
        source_path = Path(source)
        with source_path.open("rb") as stream:
            return self.put_stream(
                key,
                stream,
                length=source_path.stat().st_size,
                content_type=content_type,
            )

    def download_file(self, key: str, destination: str | Path) -> StoredObject:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination_path.parent,
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                for chunk in self.iter_bytes(key):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            os.replace(temporary_path, destination_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return StoredObject(
            key=key,
            size_bytes=size,
            checksum_sha256=digest.hexdigest(),
            content_type="application/octet-stream",
        )

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class MinioObjectStorage:
    backend_name = "minio"

    def __init__(
        self,
        *,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool,
        ca_cert_path: str | Path | None = None,
        encryption_mode: str = "none",
        kms_key_id: str | None = None,
    ) -> None:
        from minio import Minio

        if not secure:
            raise ObjectStorageError(
                "MinIO connections must use TLS; insecure HTTP transport is not supported"
            )
        normalized_ca_path = str(ca_cert_path or "").strip()
        http_client = None
        if normalized_ca_path:
            ca_path = Path(normalized_ca_path)
            if not ca_path.is_file():
                raise ObjectStorageError(f"MinIO CA certificate not found: {ca_path}")
            import urllib3

            http_client = urllib3.PoolManager(
                cert_reqs="CERT_REQUIRED",
                ca_certs=str(ca_path),
            )

        normalized_mode = encryption_mode.strip().lower()
        if normalized_mode not in {"none", "sse-s3", "sse-kms"}:
            raise ObjectStorageError(
                f"Unsupported MinIO encryption mode: {encryption_mode!r}"
            )
        normalized_key_id = (kms_key_id or "").strip() or None
        if normalized_mode == "sse-kms" and normalized_key_id is None:
            raise ObjectStorageError("MinIO SSE-KMS requires a KMS key identifier")
        if normalized_mode != "sse-kms" and normalized_key_id is not None:
            raise ObjectStorageError(
                "A MinIO KMS key identifier is valid only with SSE-KMS"
            )
        self.bucket = bucket
        self.encryption_mode = normalized_mode
        self.encryption_key_id = normalized_key_id
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
            http_client=http_client,
        )
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def _server_side_encryption(self):
        if self.encryption_mode == "none":
            return None
        if self.encryption_mode == "sse-s3":
            from minio.sse import SseS3

            return SseS3()
        from minio.sse import SseKMS

        return SseKMS(self.encryption_key_id, {})

    def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.put_stream(
            key,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )

    def get_bytes(self, key: str) -> bytes:
        return b"".join(self.iter_bytes(key))

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        length: int | None = None,
        content_type: str = "application/octet-stream",
        chunk_size: int = 1024 * 1024,
    ) -> StoredObject:
        if chunk_size <= 0:
            raise ObjectStorageError("chunk_size must be positive")
        if length is not None and length < 0:
            raise ObjectStorageError("length must be non-negative")

        hashing_stream = _HashingReader(stream)
        kwargs: dict = {}
        object_length = -1 if length is None else length
        if length is None:
            kwargs["part_size"] = max(10 * 1024 * 1024, chunk_size)
        server_side_encryption = self._server_side_encryption()
        if server_side_encryption is not None:
            kwargs["sse"] = server_side_encryption
        self.client.put_object(
            self.bucket,
            key,
            hashing_stream,
            length=object_length,
            content_type=content_type,
            **kwargs,
        )
        if length is not None and hashing_stream.size_bytes != length:
            self.delete(key)
            raise ObjectStorageError(
                f"Stream length mismatch for {key!r}: expected {length}, "
                f"received {hashing_stream.size_bytes}"
            )
        return StoredObject(
            key=key,
            size_bytes=hashing_stream.size_bytes,
            checksum_sha256=hashing_stream.checksum_sha256,
            content_type=content_type,
        )

    def iter_bytes(
        self,
        key: str,
        *,
        chunk_size: int = 1024 * 1024,
    ) -> Iterator[bytes]:
        from minio.error import S3Error

        if chunk_size <= 0:
            raise ObjectStorageError("chunk_size must be positive")
        try:
            response = self.client.get_object(self.bucket, key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object not found: {key!r}") from exc
            raise ObjectStorageError(str(exc)) from exc
        try:
            while chunk := response.read(chunk_size):
                yield chunk
        finally:
            response.close()
            response.release_conn()

    def put_file(
        self,
        key: str,
        source: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> StoredObject:
        source_path = Path(source)
        with source_path.open("rb") as stream:
            return self.put_stream(
                key,
                stream,
                length=source_path.stat().st_size,
                content_type=content_type,
            )

    def download_file(self, key: str, destination: str | Path) -> StoredObject:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination_path.parent,
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                for chunk in self.iter_bytes(key):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            os.replace(temporary_path, destination_path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return StoredObject(
            key=key,
            size_bytes=size,
            checksum_sha256=digest.hexdigest(),
            content_type="application/octet-stream",
        )

    def delete(self, key: str) -> None:
        self.client.remove_object(self.bucket, key)


class _HashingReader:
    """File-like adapter used to checksum a MinIO upload without buffering it."""

    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self._digest = hashlib.sha256()
        self.size_bytes = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.source.read(size)
        if chunk and not isinstance(chunk, bytes):
            raise ObjectStorageError("Object streams must yield bytes")
        if chunk:
            self._digest.update(chunk)
            self.size_bytes += len(chunk)
        return chunk

    @property
    def checksum_sha256(self) -> str:
        return self._digest.hexdigest()


def build_object_storage() -> ObjectStorage:
    if settings.storage_backend == "local":
        return LocalObjectStorage(settings.storage_local_dir)
    if settings.storage_backend == "minio":
        return MinioObjectStorage(
            endpoint=settings.storage_endpoint,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
            bucket=settings.storage_bucket,
            secure=settings.storage_secure,
            ca_cert_path=settings.storage_ca_cert_path,
            encryption_mode=settings.storage_encryption_mode,
            kms_key_id=settings.storage_kms_key_id,
        )
    raise ObjectStorageError(f"Unknown storage backend: {settings.storage_backend!r}")


_storage: ObjectStorage | None = None


def get_object_storage() -> ObjectStorage:
    """FastAPI dependency. Lazily built so importing the app never dials MinIO."""
    global _storage
    if _storage is None:
        _storage = build_object_storage()
    return _storage
