import hashlib
import json
from uuid import uuid4

from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ConflictError, NotFoundError, ValidationError
from al_medlit.core.storage import ObjectStorage
from al_medlit.export.formats import EvidenceBlocksJSONLFormat, TrainingWindowsJSONLFormat
from al_medlit.export.registry import ExportContext, export_formats
from al_medlit.export.schemas import ExportCreate
from al_medlit.lineage.models import AnnotationSet, CorpusSnapshot, ExportArtifact
from al_medlit.lineage.service import IterableBytesReader, register_export_artifact


def register_builtin_export_formats() -> None:
    for plugin in (EvidenceBlocksJSONLFormat(), TrainingWindowsJSONLFormat()):
        try:
            export_formats.register(plugin)
        except ValidationError as exc:
            if "already registered" not in exc.message:
                raise


def create_export(
    db: Session,
    storage: ObjectStorage,
    *,
    project_id: int,
    data: ExportCreate,
    actor_user_id: int | None,
) -> ExportArtifact:
    register_builtin_export_formats()
    plugin = export_formats.get(data.format_key)
    annotation_set = (
        db.get(AnnotationSet, data.annotation_set_id) if data.annotation_set_id else None
    )
    snapshot = (
        db.get(CorpusSnapshot, data.corpus_snapshot_id)
        if data.corpus_snapshot_id
        else None
    )
    if annotation_set is not None:
        if annotation_set.project_id != project_id:
            raise ValidationError("Annotation set is not in project")
        snapshot = annotation_set.corpus_snapshot
    if snapshot is None or snapshot.project_id != project_id:
        raise NotFoundError("Corpus snapshot not found in project")

    context = ExportContext(
        db=db,
        project_id=project_id,
        corpus_snapshot=snapshot,
        annotation_set=annotation_set,
        options=data.options,
    )
    counter = [0]

    def chunks():
        for row in plugin.iter_rows(context):
            counter[0] += 1
            yield (
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")

    token = uuid4().hex
    file_name = data.file_name or f"{plugin.key}-{token[:8]}.{plugin.extension}"
    storage_key = f"projects/{project_id}/exports/{token}/{file_name}"
    stored = storage.put_stream(
        storage_key,
        IterableBytesReader(chunks()),
        content_type=plugin.content_type,
    )
    export = register_export_artifact(
        db,
        project_id=project_id,
        stored=stored,
        format_key=plugin.key,
        file_name=file_name,
        row_count=counter[0],
        actor_user_id=actor_user_id,
        corpus_snapshot_id=snapshot.id,
        annotation_set_id=annotation_set.id if annotation_set else None,
        metadata={**data.metadata_, "options": data.options},
    )
    return export


def verify_export(storage: ObjectStorage, export: ExportArtifact) -> None:
    digest = hashlib.sha256()
    size = 0
    for chunk in storage.iter_bytes(export.artifact.storage_key):
        digest.update(chunk)
        size += len(chunk)
    if size != export.artifact.size_bytes or digest.hexdigest() != export.artifact.content_hash:
        raise ConflictError("Export object checksum does not match immutable metadata")


def list_exports(db: Session, project_id: int) -> list[ExportArtifact]:
    return (
        db.query(ExportArtifact)
        .filter(ExportArtifact.project_id == project_id)
        .order_by(ExportArtifact.created_at.desc())
        .all()
    )


def get_export(db: Session, export_id: int) -> ExportArtifact:
    export = db.get(ExportArtifact, export_id)
    if export is None:
        raise NotFoundError("Export not found")
    return export
