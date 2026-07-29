"""Domain operations for the canonical learning workflow."""

import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from al_medlit.auth.models import User
from al_medlit.core.exceptions import (
    ValidationError,
)
from al_medlit.core.storage import ObjectStorage
from al_medlit.corpus.models import Document
from al_medlit.model_artifacts import service as artifact_service
from al_medlit.model_artifacts.schemas import ArtifactPackageCreate
from al_medlit.workflow import models, schemas

from .common import (
    _canonical_hash,
    _commit,
    _next_version,
    _optional_package,
    _project,
    _scoped,
)

MAX_DATASET_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DATASET_UPLOAD_ITEMS = 100_000
MAX_JSONL_LINE_BYTES = 1024 * 1024
_PINNED_HUGGING_FACE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_AUTOMATIC_SELECTION_STRATEGIES = {
    "all",
    "random",
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_FEEDBACK_SELECTION_STRATEGIES = {
    "uncertainty",
    "diversity",
    "disagreement",
    "error_based",
    "hybrid_uncertainty_diversity",
}
_SENSITIVE_TRAINING_CONFIG_FRAGMENTS = {
    "access_token",
    "api_key",
    "credential",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "storage_key",
}




def create_dataset(db: Session, data: schemas.DatasetCreate, actor: User) -> models.Dataset:
    _project(db, data.project_id)
    dataset = models.Dataset(**data.model_dump(), created_by_user_id=actor.id)
    db.add(dataset)
    _commit(db, "A dataset with this name already exists in the project")
    db.refresh(dataset)
    return dataset


def list_datasets(db: Session, project_id: int) -> list[models.Dataset]:
    _project(db, project_id)
    return (
        db.query(models.Dataset)
        .filter(models.Dataset.project_id == project_id)
        .order_by(models.Dataset.name, models.Dataset.id)
        .all()
    )


def create_dataset_version(
    db: Session, data: schemas.DatasetVersionCreate, actor: User
) -> models.DatasetVersion:
    dataset = _scoped(db, models.Dataset, data.dataset_id, data.project_id, "Dataset")
    db.query(models.Dataset).filter(models.Dataset.id == dataset.id).with_for_update().one()
    _optional_package(db, data.project_id, data.artifact_package_id)

    stable_keys = [item.stable_key for item in data.items]
    if len(stable_keys) != len(set(stable_keys)):
        raise ValidationError("Dataset item stable_key values must be unique within a version")
    item_payloads = [
        {
            **item.model_dump(),
            "content_hash": _canonical_hash(item.payload),
        }
        for item in data.items
    ]
    version_content = {
        **data.model_dump(exclude={"project_id", "dataset_id", "items"}),
        "items": item_payloads,
    }
    version = models.DatasetVersion(
        project_id=data.project_id,
        dataset_id=dataset.id,
        version_number=_next_version(
            db,
            models.DatasetVersion,
            models.DatasetVersion.dataset_id,
            dataset.id,
        ),
        item_count=len(item_payloads),
        content_hash=_canonical_hash(version_content),
        created_by_user_id=actor.id,
        **data.model_dump(exclude={"project_id", "dataset_id", "items", "artifact_package_id"}),
        artifact_package_id=data.artifact_package_id,
    )
    db.add(version)
    db.flush()
    for item in item_payloads:
        db.add(
            models.DatasetItem(
                project_id=data.project_id,
                dataset_version_id=version.id,
                **item,
            )
        )
    _commit(db, "This exact dataset version already exists")
    db.refresh(version)
    return version


def _project_document_group_key(document: Document) -> str:
    metadata = document.metadata_ or {}
    raw_group_key = metadata.get("group_id")
    if raw_group_key is None:
        raw_group_key = document.external_id or document.id
    if isinstance(raw_group_key, (str, int, float, bool)):
        group_key = str(raw_group_key)
        if len(group_key) <= 255:
            return group_key
    return f"sha256:{_canonical_hash(raw_group_key)}"


def create_project_corpus_dataset_version(
    db: Session,
    *,
    project_id: int,
    dataset_id: int,
    actor: User,
) -> models.DatasetVersion:
    dataset = _scoped(db, models.Dataset, dataset_id, project_id, "Dataset")
    if dataset.source_type != "project_corpus":
        raise ValidationError("Project corpus snapshots require a project_corpus dataset")

    # Serialize snapshots per dataset so an unchanged corpus is materialized once.
    db.query(models.Dataset).filter(models.Dataset.id == dataset.id).with_for_update().one()
    documents = (
        db.query(Document).filter(Document.project_id == project_id).order_by(Document.id).all()
    )
    if not documents:
        raise ValidationError("The project corpus has no documents to snapshot")

    items = [
        schemas.DatasetItemCreate(
            stable_key=f"project-document:{document.id}",
            group_key=_project_document_group_key(document),
            payload={
                "document_id": document.id,
                "external_id": document.external_id,
                "title": document.title,
                "text": document.text,
                "source": document.source,
                "metadata": document.metadata_ or {},
                "active_structure_version_id": document.active_structure_version_id,
            },
        )
        for document in documents
    ]
    source_revision = _canonical_hash(
        {
            "project_id": project_id,
            "identity_scheme": "project-document-id",
            "items": [item.model_dump() for item in items],
        }
    )
    version_data = schemas.DatasetVersionCreate(
        project_id=project_id,
        dataset_id=dataset.id,
        source_uri=f"project://projects/{project_id}/documents",
        source_revision=source_revision,
        source_format="project_corpus",
        data_schema={
            "type": "object",
            "required": ["document_id", "text", "metadata"],
            "properties": {
                "document_id": {"type": "integer"},
                "external_id": {"type": ["string", "null"]},
                "title": {"type": ["string", "null"]},
                "text": {"type": "string"},
                "source": {"type": ["string", "null"]},
                "metadata": {"type": "object"},
                "active_structure_version_id": {"type": ["integer", "null"]},
            },
            "additionalProperties": False,
        },
        provenance={
            "ingestion": "project_corpus_snapshot",
            "source_project_id": project_id,
            "source_document_ids": [document.id for document in documents],
            "source_document_count": len(documents),
            "source_revision": source_revision,
            "stable_identity": "project-document:{document_id}",
        },
        license_info={
            "status": "inherited_from_project",
            "source_project_id": project_id,
        },
        items=items,
    )
    item_payloads = [
        {
            **item.model_dump(),
            "content_hash": _canonical_hash(item.payload),
        }
        for item in version_data.items
    ]
    content_hash = _canonical_hash(
        {
            **version_data.model_dump(exclude={"project_id", "dataset_id", "items"}),
            "items": item_payloads,
        }
    )
    existing = (
        db.query(models.DatasetVersion)
        .filter(
            models.DatasetVersion.dataset_id == dataset.id,
            models.DatasetVersion.content_hash == content_hash,
        )
        .one_or_none()
    )
    if existing is not None:
        return existing
    return create_dataset_version(db, version_data, actor)


def list_dataset_versions(
    db: Session, project_id: int, dataset_id: int
) -> list[models.DatasetVersion]:
    _scoped(db, models.Dataset, dataset_id, project_id, "Dataset")
    return (
        db.query(models.DatasetVersion)
        .filter(
            models.DatasetVersion.project_id == project_id,
            models.DatasetVersion.dataset_id == dataset_id,
        )
        .order_by(models.DatasetVersion.version_number)
        .all()
    )


def list_dataset_items(
    db: Session, project_id: int, dataset_version_id: int
) -> list[models.DatasetItem]:
    _scoped(
        db,
        models.DatasetVersion,
        dataset_version_id,
        project_id,
        "Dataset version",
    )
    return (
        db.query(models.DatasetItem)
        .filter(models.DatasetItem.dataset_version_id == dataset_version_id)
        .order_by(models.DatasetItem.id)
        .all()
    )


def _read_dataset_upload(stream) -> bytes:
    content = stream.read(MAX_DATASET_UPLOAD_BYTES + 1)
    if not isinstance(content, bytes):
        raise ValidationError("Dataset upload must be a binary file")
    if len(content) > MAX_DATASET_UPLOAD_BYTES:
        raise ValidationError(
            f"Dataset upload exceeds the {MAX_DATASET_UPLOAD_BYTES // (1024 * 1024)} MB limit"
        )
    if not content:
        raise ValidationError("Dataset upload is empty")
    return content


def _decode_dataset_text(content: bytes) -> str:
    if b"\x00" in content:
        raise ValidationError("Text dataset uploads cannot contain NUL bytes")
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValidationError("Dataset upload must use UTF-8 encoding") from exc


def _json_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def _normalize_json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"Dataset value at {path} must be finite")
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [
            _normalize_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"Dataset object keys at {path} must be strings")
            normalized[key] = _normalize_json_value(item, path=f"{path}.{key}")
        return normalized
    scalar = getattr(value, "item", None)
    if callable(scalar):
        return _normalize_json_value(scalar(), path=path)
    raise ValidationError(f"Dataset value at {path} has unsupported type {type(value).__name__}")


def _parse_csv_records(content: bytes) -> list[dict]:
    text = _decode_dataset_text(content)
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        raw_header = next(reader, None)
        if raw_header is None:
            raise ValidationError("CSV upload is missing a header row")
        header = [column.strip() for column in raw_header]
        if not header or any(not column for column in header):
            raise ValidationError("CSV header names cannot be empty")
        if len(header) != len(set(header)):
            raise ValidationError("CSV header names must be unique")

        records: list[dict] = []
        for row_number, row in enumerate(reader, start=2):
            if not row or all(not value for value in row):
                continue
            if len(row) != len(header):
                raise ValidationError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            records.append(dict(zip(header, row, strict=True)))
            if len(records) > MAX_DATASET_UPLOAD_ITEMS:
                raise ValidationError(
                    f"Dataset upload exceeds the {MAX_DATASET_UPLOAD_ITEMS} item limit"
                )
    except csv.Error as exc:
        raise ValidationError(f"Invalid CSV upload: {exc}") from exc
    if not records:
        raise ValidationError("Dataset upload contains no records")
    return records


def _parse_jsonl_records(content: bytes) -> list[dict]:
    text = _decode_dataset_text(content)
    records: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_JSONL_LINE_BYTES:
            raise ValidationError(
                f"JSONL line {line_number} exceeds the {MAX_JSONL_LINE_BYTES // 1024} KB line limit"
            )
        try:
            parsed = json.loads(
                line,
                object_pairs_hook=_json_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValidationError(f"Invalid JSONL object on line {line_number}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValidationError(f"JSONL line {line_number} must contain a JSON object")
        records.append(_normalize_json_value(parsed, path=f"$[{line_number}]"))
        if len(records) > MAX_DATASET_UPLOAD_ITEMS:
            raise ValidationError(
                f"Dataset upload exceeds the {MAX_DATASET_UPLOAD_ITEMS} item limit"
            )
    if not records:
        raise ValidationError("Dataset upload contains no records")
    return records


def _parse_parquet_records(content: bytes) -> list[dict]:
    try:
        import pyarrow.parquet as parquet
    except (ImportError, ModuleNotFoundError) as exc:
        raise ValidationError(
            "Parquet ingestion is unavailable; install the optional 'pyarrow' package "
            "in the API environment"
        ) from exc
    try:
        table = parquet.read_table(io.BytesIO(content))
    except Exception as exc:
        raise ValidationError(f"Invalid Parquet upload: {exc}") from exc
    if table.num_rows > MAX_DATASET_UPLOAD_ITEMS:
        raise ValidationError(f"Dataset upload exceeds the {MAX_DATASET_UPLOAD_ITEMS} item limit")
    records = [
        _normalize_json_value(record, path=f"$[{index}]")
        for index, record in enumerate(table.to_pylist(), start=1)
    ]
    if not records:
        raise ValidationError("Dataset upload contains no records")
    return records


def _resolve_upload_format(source_format: str, file_name: str | None) -> str:
    if source_format != "auto":
        return source_format
    suffix = Path(file_name or "").suffix.lower()
    inferred = {
        ".csv": "csv",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".parquet": "parquet",
    }.get(suffix)
    if inferred is None:
        raise ValidationError("Could not infer dataset format; specify csv, jsonl, or parquet")
    return inferred


def _record_identity(
    record: dict,
    field: str,
    *,
    row_number: int,
    label: str,
) -> str:
    if field not in record:
        raise ValidationError(f"{label} field {field!r} is missing from row {row_number}")
    value = record[field]
    if value is None or isinstance(value, (dict, list)):
        raise ValidationError(f"{label} field {field!r} must be a scalar on row {row_number}")
    normalized = str(value).strip()
    if not normalized:
        raise ValidationError(f"{label} field {field!r} cannot be empty on row {row_number}")
    if len(normalized) > 255:
        raise ValidationError(f"{label} field {field!r} exceeds 255 characters on row {row_number}")
    return normalized


def _dataset_items_from_records(
    records: list[dict],
    *,
    stable_key_field: str | None,
    group_key_field: str | None,
) -> list[schemas.DatasetItemCreate]:
    items: list[schemas.DatasetItemCreate] = []
    seen_keys: set[str] = set()
    for row_number, record in enumerate(records, start=1):
        stable_key = (
            _record_identity(
                record,
                stable_key_field,
                row_number=row_number,
                label="Stable key",
            )
            if stable_key_field
            else f"row-{row_number:08d}"
        )
        if stable_key in seen_keys:
            raise ValidationError(f"Stable key {stable_key!r} is duplicated on row {row_number}")
        seen_keys.add(stable_key)
        group_key = (
            _record_identity(
                record,
                group_key_field,
                row_number=row_number,
                label="Group key",
            )
            if group_key_field
            else None
        )
        items.append(
            schemas.DatasetItemCreate(
                stable_key=stable_key,
                group_key=group_key,
                payload=record,
            )
        )
    return items


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _infer_record_schema(records: list[dict]) -> dict:
    field_names = sorted({key for record in records for key in record})
    properties: dict[str, dict] = {}
    required = set(field_names)
    for field in field_names:
        values = [record[field] for record in records if field in record]
        types = sorted({_json_type(value) for value in values})
        properties[field] = {"type": types[0] if len(types) == 1 else types}
        if len(values) != len(records) or any(value is None for value in values):
            required.discard(field)
    return {
        "type": "object",
        "properties": properties,
        "required": sorted(required),
    }


def create_dataset_version_from_upload(
    db: Session,
    *,
    project_id: int,
    dataset_id: int,
    source_format: str,
    file_name: str | None,
    content_type: str | None,
    stream,
    stable_key_field: str | None,
    group_key_field: str | None,
    actor: User,
    storage: ObjectStorage | None = None,
) -> models.DatasetVersion:
    dataset = _scoped(db, models.Dataset, dataset_id, project_id, "Dataset")
    if dataset.source_type != "upload":
        raise ValidationError("File uploads require a dataset with source_type='upload'")
    resolved_format = _resolve_upload_format(source_format, file_name)
    content = _read_dataset_upload(stream)
    parsers = {
        "csv": _parse_csv_records,
        "jsonl": _parse_jsonl_records,
        "parquet": _parse_parquet_records,
    }
    records = parsers[resolved_format](content)
    items = _dataset_items_from_records(
        records,
        stable_key_field=stable_key_field,
        group_key_field=group_key_field,
    )
    digest = hashlib.sha256(content).hexdigest()
    safe_name = Path((file_name or "dataset").replace("\\", "/")).name or "dataset"
    record_schema = _infer_record_schema(records)
    artifact_package_id = None
    if storage is not None:
        content_type_by_format = {
            "csv": "text/csv",
            "jsonl": "application/x-ndjson",
            "parquet": "application/vnd.apache.parquet",
        }
        package = artifact_service.publish_artifact_package(
            db,
            storage,
            project_id=project_id,
            data=ArtifactPackageCreate(
                package_kind="dataset_source",
                package_format=resolved_format,
                schema_version="dataset-source-v1",
                display_name=safe_name,
                loader_policy="safe",
                task_contract={
                    "record_schema": record_schema,
                    "stable_key_field": stable_key_field,
                    "group_key_field": group_key_field,
                },
                sensitivity="project",
                metadata={
                    "source_revision": digest,
                    "source_format": resolved_format,
                    "item_count": len(records),
                },
            ),
            files=[
                artifact_service.PackageFileUpload(
                    relative_path=f"source/{safe_name}",
                    source=content,
                    role="source_dataset",
                    content_type=content_type_by_format[resolved_format],
                    expected_checksum_sha256=digest,
                    expected_size_bytes=len(content),
                )
            ],
            actor_user_id=actor.id,
        )
        artifact_package_id = package.id
    return create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=dataset.id,
            source_uri=f"upload://sha256/{digest}",
            source_revision=digest,
            source_format=resolved_format,
            data_schema=record_schema,
            provenance={
                "ingestion": "direct_upload",
                "original_file_name": safe_name,
                "content_type": content_type,
                "sha256": digest,
                "stable_key_field": stable_key_field,
                "group_key_field": group_key_field,
            },
            license_info={},
            artifact_package_id=artifact_package_id,
            items=items,
        ),
        actor,
    )


def create_public_registry_dataset_version(
    db: Session,
    *,
    project_id: int,
    dataset_id: int,
    data: schemas.PublicRegistryDatasetVersionCreate,
    actor: User,
) -> models.DatasetVersion:
    dataset = _scoped(db, models.Dataset, dataset_id, project_id, "Dataset")
    if dataset.source_type != "public_registry":
        raise ValidationError(
            "Public registry references require a dataset with source_type='public_registry'"
        )
    if (
        data.provider != "hugging_face"
        or _PINNED_HUGGING_FACE_REVISION.fullmatch(data.exact_revision) is None
    ):
        raise ValidationError("Hugging Face datasets require an exact 40-character commit revision")
    provenance = {
        **data.provenance,
        "ingestion": "public_registry_reference",
        "provider": data.provider,
        "registry_dataset_id": data.registry_dataset_id,
        "config_name": data.config_name,
        "exact_revision": data.exact_revision.lower(),
        "expected_content_sha256": (
            data.expected_content_sha256.lower()
            if data.expected_content_sha256 is not None
            else None
        ),
        "fetch_performed": False,
    }
    return create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=dataset.id,
            source_uri=f"hf://datasets/{data.registry_dataset_id}",
            source_revision=data.exact_revision.lower(),
            source_format=data.source_format,
            data_schema=data.data_schema,
            provenance=provenance,
            license_info=data.license_info,
            items=[],
        ),
        actor,
    )


def create_public_registry_dataset_version_from_snapshot(
    db: Session,
    *,
    project_id: int,
    dataset_id: int,
    registry: schemas.PublicRegistryDatasetVersionCreate,
    file_name: str | None,
    content_type: str | None,
    stream,
    stable_key_field: str | None,
    group_key_field: str | None,
    actor: User,
    storage: ObjectStorage,
) -> models.DatasetVersion:
    """Materialize an exported, revision-pinned public dataset snapshot.

    Registry SDKs and network clients deliberately stay out of the API image.
    Administrators or ingestion workers can export the pinned revision, and this
    boundary verifies and stores the exact bytes before records become usable.
    """

    dataset = _scoped(db, models.Dataset, dataset_id, project_id, "Dataset")
    if dataset.source_type != "public_registry":
        raise ValidationError(
            "Registry snapshots require a dataset with source_type='public_registry'"
        )
    if (
        registry.provider != "hugging_face"
        or _PINNED_HUGGING_FACE_REVISION.fullmatch(registry.exact_revision) is None
    ):
        raise ValidationError("Hugging Face datasets require an exact 40-character commit revision")
    resolved_format = _resolve_upload_format(registry.source_format, file_name)
    if resolved_format not in {"csv", "jsonl", "parquet"}:
        raise ValidationError("Registry snapshots must be CSV, JSONL, or Parquet")
    content = _read_dataset_upload(stream)
    digest = hashlib.sha256(content).hexdigest()
    if (
        registry.expected_content_sha256 is not None
        and digest != registry.expected_content_sha256.lower()
    ):
        raise ValidationError("Registry snapshot checksum does not match expected_content_sha256")
    parsers = {
        "csv": _parse_csv_records,
        "jsonl": _parse_jsonl_records,
        "parquet": _parse_parquet_records,
    }
    records = parsers[resolved_format](content)
    items = _dataset_items_from_records(
        records,
        stable_key_field=stable_key_field,
        group_key_field=group_key_field,
    )
    record_schema = _infer_record_schema(records)
    safe_name = Path((file_name or "registry-snapshot").replace("\\", "/")).name
    content_type_by_format = {
        "csv": "text/csv",
        "jsonl": "application/x-ndjson",
        "parquet": "application/vnd.apache.parquet",
    }
    package = artifact_service.publish_artifact_package(
        db,
        storage,
        project_id=project_id,
        data=ArtifactPackageCreate(
            package_kind="dataset_source",
            package_format=resolved_format,
            schema_version="dataset-source-v1",
            display_name=safe_name,
            loader_policy="safe",
            task_contract={
                "record_schema": record_schema,
                "stable_key_field": stable_key_field,
                "group_key_field": group_key_field,
            },
            sensitivity="project",
            license_info=registry.license_info,
            metadata={
                "provider": registry.provider,
                "registry_dataset_id": registry.registry_dataset_id,
                "config_name": registry.config_name,
                "exact_revision": registry.exact_revision.lower(),
                "snapshot_sha256": digest,
                "source_format": resolved_format,
                "item_count": len(records),
            },
        ),
        files=[
            artifact_service.PackageFileUpload(
                relative_path=f"source/{safe_name}",
                source=content,
                role="source_dataset",
                content_type=content_type or content_type_by_format[resolved_format],
                expected_checksum_sha256=digest,
                expected_size_bytes=len(content),
            )
        ],
        actor_user_id=actor.id,
    )
    provenance = {
        **registry.provenance,
        "ingestion": "public_registry_snapshot",
        "provider": registry.provider,
        "registry_dataset_id": registry.registry_dataset_id,
        "config_name": registry.config_name,
        "exact_revision": registry.exact_revision.lower(),
        "snapshot_sha256": digest,
        "expected_content_sha256": (
            registry.expected_content_sha256.lower()
            if registry.expected_content_sha256 is not None
            else None
        ),
        "fetch_performed": False,
        "materialized": True,
        "stable_key_field": stable_key_field,
        "group_key_field": group_key_field,
    }
    return create_dataset_version(
        db,
        schemas.DatasetVersionCreate(
            project_id=project_id,
            dataset_id=dataset.id,
            source_uri=f"hf://datasets/{registry.registry_dataset_id}",
            source_revision=registry.exact_revision.lower(),
            source_format=resolved_format,
            data_schema=record_schema,
            provenance=provenance,
            license_info=registry.license_info,
            artifact_package_id=package.id,
            items=items,
        ),
        actor,
    )
