from sqlalchemy import or_
from sqlalchemy.orm import Session

from al_medlit.annotation import types as annotation_types
from al_medlit.annotation.models import Annotation, AnnotationCorrection
from al_medlit.annotation.schemas import (
    AnnotationCorrectionCreate,
    AnnotationCreate,
    AnnotationUpdate,
)
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import (
    annotation_reference_access_clause,
    resource_access_clause,
)
from al_medlit.core.events import event_bus
from al_medlit.core.exceptions import ConflictError, ForbiddenError
from al_medlit.corpus.models import Document, DocumentStructureVersion
from al_medlit.evidence.models import EvidenceBlockAnnotation
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.project.models import Project, ProjectTask


class AnnotationInUseError(ConflictError):
    pass


def _validate_annotation_shape(values: dict) -> None:
    annotation_types.validate_create(values)


def _validate_relation_targets(
    db: Session, values: dict, *, self_id: int | None = None
) -> None:
    target_fields = ("head_annotation_id", "tail_annotation_id")
    target_ids = {
        field: values.get(field)
        for field in target_fields
        if values.get(field) is not None
    }
    if not target_ids:
        return

    if self_id is not None:
        self_referencing = sorted(
            field for field, target_id in target_ids.items() if target_id == self_id
        )
        if self_referencing:
            raise annotation_types.AnnotationValidationError(
                "Annotation cannot reference itself: "
                f"{', '.join(self_referencing)}"
            )

    locked_targets = {
        annotation.id: annotation
        for annotation in (
            db.query(Annotation)
            .filter(Annotation.id.in_(sorted(set(target_ids.values()))))
            .order_by(Annotation.id)
            .populate_existing()
            .with_for_update()
            .all()
        )
    }
    targets = {
        field: locked_targets.get(annotation_id)
        for field, annotation_id in target_ids.items()
    }
    missing = [field for field, target in targets.items() if target is None]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise annotation_types.AnnotationValidationError(
            f"Referenced annotations not found: {missing_list}"
        )

    non_span = sorted(
        field
        for field, target in targets.items()
        if target is not None
        and not getattr(
            annotation_types.REGISTRY.get(target.annotation_type),
            "relation_endpoint_allowed",
            False,
        )
    )
    if non_span:
        raise annotation_types.AnnotationValidationError(
            "Relation head/tail must reference span annotations "
            f"(types with offsets): {', '.join(non_span)}"
        )

    project_id = values.get("project_id")
    document_id = values.get("document_id")
    mismatched = [
        field
        for field, target in targets.items()
        if target is not None
        and (target.project_id != project_id or target.document_id != document_id)
    ]
    if mismatched:
        mismatched_list = ", ".join(sorted(mismatched))
        raise annotation_types.AnnotationValidationError(
            "Referenced annotations must belong to the same project and document: "
            f"{mismatched_list}"
        )
    version_mismatched = [
        field
        for field, target in targets.items()
        if target is not None
        and (
            target.structure_version_id != values.get("structure_version_id")
            or target.guideline_version_id != values.get("guideline_version_id")
        )
    ]
    if version_mismatched:
        raise annotation_types.AnnotationValidationError(
            "Referenced annotations must belong to the same structure and guideline "
            f"round: {', '.join(sorted(version_mismatched))}"
        )


def _validate_relation_uniqueness(
    db: Session, values: dict, *, self_id: int | None = None
) -> None:
    head_id = values.get("head_annotation_id")
    tail_id = values.get("tail_annotation_id")
    if head_id is None or tail_id is None:
        return

    query = db.query(Annotation.id).filter(
        Annotation.project_id == values["project_id"],
        Annotation.document_id == values["document_id"],
        Annotation.annotation_type == values["annotation_type"],
        Annotation.label == values["label"],
        Annotation.head_annotation_id == head_id,
        Annotation.tail_annotation_id == tail_id,
    )
    if self_id is not None:
        query = query.filter(Annotation.id != self_id)
    if query.first() is not None:
        raise annotation_types.AnnotationValidationError(
            f"An identical '{values['label']}' relation between these "
            "annotations already exists"
        )


def _validate_annotation_values(
    db: Session, values: dict, *, self_id: int | None = None
) -> None:
    _validate_annotation_shape(values)
    _validate_relation_targets(db, values, self_id=self_id)
    _validate_relation_uniqueness(db, values, self_id=self_id)


def _validate_project_annotation_config(db: Session, values: dict) -> None:
    project = db.get(Project, values["project_id"])
    if project is None or project.annotation_validation_mode != "strict":
        return

    annotation_type = values["annotation_type"]
    task = (
        db.query(ProjectTask)
        .filter(
            ProjectTask.project_id == project.id,
            ProjectTask.annotation_type == annotation_type,
            ProjectTask.enabled.is_(True),
        )
        .first()
    )
    if task is None:
        raise annotation_types.AnnotationValidationError(
            f"Annotation type '{annotation_type}' is not enabled for project {project.id}"
        )

    label = values["label"]
    allowed_labels = {label_def.get("name") for label_def in task.labels}
    if annotation_type == "evidence_block" and label == "evidence_block":
        pass
    elif label not in allowed_labels:
        raise annotation_types.AnnotationValidationError(
            f"Label '{label}' is not enabled for {annotation_type} task in project {project.id}"
        )

    if annotation_type != "relation":
        return

    constraints = (task.settings or {}).get("relation_constraints") or {}
    constraint = constraints.get(label)
    if not isinstance(constraint, dict):
        return

    sides = (
        ("head", values.get("head_annotation_id")),
        ("tail", values.get("tail_annotation_id")),
    )
    for side, annotation_id in sides:
        allowed_side_labels = constraint.get(side) or []
        if not allowed_side_labels:
            continue
        target = db.get(Annotation, annotation_id)
        target_label = target.label if target is not None else "missing"
        if target_label not in allowed_side_labels:
            raise annotation_types.AnnotationValidationError(
                f"Relation '{label}' requires a {side} annotation labeled one of "
                f"{sorted(allowed_side_labels)}; got '{target_label}'"
            )


def _validate_scope(
    db: Session,
    *,
    project_id: int,
    document_id: int,
    guideline_version_id: int | None = None,
    structure_version_id: int | None = None,
) -> Document:
    document = db.get(Document, document_id)
    if document is None:
        raise annotation_types.AnnotationValidationError(
            f"Document {document_id} not found"
        )
    if document.project_id != project_id:
        raise annotation_types.AnnotationValidationError(
            f"Document {document_id} does not belong to project {project_id}"
        )

    if guideline_version_id is not None:
        guideline_version = db.get(GuidelineVersion, guideline_version_id)
        if guideline_version is None:
            raise annotation_types.AnnotationValidationError(
                f"GuidelineVersion {guideline_version_id} not found"
            )
        if guideline_version.project_id != project_id:
            raise annotation_types.AnnotationValidationError(
                "GuidelineVersion "
                f"{guideline_version_id} does not belong to project {project_id}"
            )

    if structure_version_id is not None:
        structure_version = db.get(DocumentStructureVersion, structure_version_id)
        if structure_version is None:
            raise annotation_types.AnnotationValidationError(
                f"DocumentStructureVersion {structure_version_id} not found"
            )
        if structure_version.document_id != document_id:
            raise annotation_types.AnnotationValidationError(
                "DocumentStructureVersion "
                f"{structure_version_id} does not belong to document {document_id}"
            )

    return document


def _validate_span(values: dict, document: Document) -> None:
    start = values.get("start_offset")
    end = values.get("end_offset")
    text_span = values.get("text_span")

    if start is None and end is None:
        if text_span is not None:
            raise annotation_types.AnnotationValidationError(
                "text_span requires start_offset and end_offset"
            )
        return
    if start is None or end is None:
        raise annotation_types.AnnotationValidationError(
            "start_offset and end_offset must both be set when one is provided"
        )

    if start < 0 or end < 0:
        raise annotation_types.AnnotationValidationError(
            f"Offsets must be non-negative: start_offset={start}, end_offset={end}"
        )
    if start >= end:
        raise annotation_types.AnnotationValidationError(
            "start_offset must be less than end_offset: "
            f"start_offset={start}, end_offset={end}"
        )

    text = document.text or ""
    if end > len(text):
        raise annotation_types.AnnotationValidationError(
            f"end_offset {end} exceeds document length {len(text)}"
        )

    if text_span is not None and text_span != text[start:end]:
        raise annotation_types.AnnotationValidationError(
            "text_span does not match document text at the given offsets"
        )


def _validate_correction_references(db: Session, values: dict) -> None:
    project_id = values["project_id"]
    document_id = values["document_id"]
    ref_fields = ("original_annotation_id", "corrected_annotation_id")
    ref_ids = {
        field: values.get(field)
        for field in ref_fields
        if values.get(field) is not None
    }
    if not ref_ids:
        return

    locked_targets = {
        annotation.id: annotation
        for annotation in (
            db.query(Annotation)
            .filter(Annotation.id.in_(sorted(set(ref_ids.values()))))
            .order_by(Annotation.id)
            .populate_existing()
            .with_for_update()
            .all()
        )
    }
    targets = {
        field: locked_targets.get(annotation_id)
        for field, annotation_id in ref_ids.items()
    }
    missing = sorted(field for field, target in targets.items() if target is None)
    if missing:
        raise annotation_types.AnnotationValidationError(
            f"Referenced annotations not found: {', '.join(missing)}"
        )

    mismatched = sorted(
        field
        for field, target in targets.items()
        if target is not None
        and (target.project_id != project_id or target.document_id != document_id)
    )
    if mismatched:
        raise annotation_types.AnnotationValidationError(
            "Referenced annotations must belong to the same project and document: "
            f"{', '.join(mismatched)}"
        )


def create_annotation(
    db: Session,
    data: AnnotationCreate,
    *,
    required_assignment_id: int | None = None,
) -> Annotation:
    payload = data.model_dump()
    evidence_payload = payload.pop("evidence_block", None)
    document = _validate_scope(
        db,
        project_id=payload["project_id"],
        document_id=payload["document_id"],
        guideline_version_id=payload.get("guideline_version_id"),
        structure_version_id=payload.get("structure_version_id"),
    )
    if payload.get("structure_version_id") is None:
        payload["structure_version_id"] = document.active_structure_version_id
    if payload["annotation_type"] == "evidence_block":
        if evidence_payload is None:
            raise annotation_types.AnnotationValidationError(
                "evidence_block payload is required for evidence annotations"
            )
        payload["structure_version_id"] = evidence_payload["structure_version_id"]
        _validate_project_annotation_config(
            db, {**payload, "label": "evidence_block"}
        )
        from al_medlit.evidence.schemas import EvidenceBlockPayloadV1
        from al_medlit.evidence.service import create_evidence_block

        return create_evidence_block(
            db,
            payload,
            EvidenceBlockPayloadV1.model_validate(evidence_payload),
            actor_user_id=payload.get("annotator_user_id"),
            required_assignment_id=required_assignment_id,
        )
    if evidence_payload is not None:
        raise annotation_types.AnnotationValidationError(
            "evidence_block payload is only valid for evidence_block annotations"
        )
    _validate_annotation_values(db, payload)
    _validate_project_annotation_config(db, payload)
    _validate_span(payload, document)

    annotation = Annotation(**payload)
    db.add(annotation)
    db.commit()
    db.refresh(annotation)
    return annotation


def get_annotation(db: Session, annotation_id: int) -> Annotation | None:
    return db.get(Annotation, annotation_id)


def list_annotations(
    db: Session,
    project_id: int | None = None,
    document_id: int | None = None,
    annotator_user_id: int | None = None,
    annotator_id: str | None = None,
    user: User | None = None,
    target_version_id: int | None = None,
    structure_version_id: int | None = None,
    guideline_version_id: int | None = None,
    filter_guideline_version: bool = False,
    annotation_type: str | None = None,
) -> list[Annotation]:
    query = db.query(Annotation)
    if project_id is not None:
        query = query.filter(Annotation.project_id == project_id)
    if document_id is not None:
        query = query.filter(Annotation.document_id == document_id)
    if annotation_type is not None:
        query = query.filter(Annotation.annotation_type == annotation_type)
    if annotator_user_id is not None:
        query = query.filter(Annotation.annotator_user_id == annotator_user_id)
    elif annotator_id is not None:
        query = query.filter(Annotation.annotator_id == annotator_id)
    evidence_joined = target_version_id is not None
    if evidence_joined:
        query = query.join(
            EvidenceBlockAnnotation,
            EvidenceBlockAnnotation.annotation_id == Annotation.id,
        )
    if target_version_id is not None:
        query = query.filter(
            EvidenceBlockAnnotation.target_version_id == target_version_id
        )
    if structure_version_id is not None:
        query = query.filter(Annotation.structure_version_id == structure_version_id)
    if filter_guideline_version or guideline_version_id is not None:
        query = query.filter(
            Annotation.guideline_version_id == guideline_version_id
        )
    if user is not None:
        if not evidence_joined:
            query = query.outerjoin(
                EvidenceBlockAnnotation,
                EvidenceBlockAnnotation.annotation_id == Annotation.id,
            )
        query = query.join(
            Project,
            Project.id == Annotation.project_id,
        ).join(
            Document,
            Document.id == Annotation.document_id,
        ).filter(
            Document.project_id == Annotation.project_id,
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=Annotation.project_id,
                document_id=Annotation.document_id,
                annotation_type=Annotation.annotation_type,
                owner_user_id=Annotation.annotator_user_id,
                target_version_id=EvidenceBlockAnnotation.target_version_id,
                structure_version_id=EvidenceBlockAnnotation.structure_version_id,
            )
        )
    return query.order_by(Annotation.created_at.desc()).all()


def update_annotation(
    db: Session,
    annotation_id: int,
    data: AnnotationUpdate,
    *,
    actor_user_id: int | None = None,
    required_assignment_id: int | None = None,
) -> Annotation | None:
    candidate = db.get(Annotation, annotation_id)
    if candidate is None:
        return None

    updates = data.model_dump(exclude_unset=True)
    expected_revision = updates.pop("expected_revision", None)
    evidence_payload = updates.pop("evidence_block", None)
    if candidate.annotation_type == "evidence_block":
        from al_medlit.evidence.schemas import EvidenceBlockPayloadV1
        from al_medlit.evidence.service import update_evidence_block

        disallowed = set(updates) - {"label"}
        if disallowed:
            raise annotation_types.AnnotationValidationError(
                "Evidence annotations can only be changed through evidence_block payload"
            )
        if "label" in updates and updates["label"] != "evidence_block":
            raise annotation_types.AnnotationValidationError(
                "Evidence annotation label is fixed as 'evidence_block'"
            )
        return update_evidence_block(
            db,
            annotation_id,
            (
                EvidenceBlockPayloadV1.model_validate(evidence_payload)
                if evidence_payload is not None
                else None
            ),
            expected_revision=expected_revision,
            actor_user_id=actor_user_id,
            required_assignment_id=required_assignment_id,
        )
    annotation = (
        db.query(Annotation)
        .filter(Annotation.id == annotation_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if annotation is None:
        return None
    if actor_user_id is not None:
        if annotation.annotator_user_id != actor_user_id:
            raise ForbiddenError("Annotation belongs to another annotator")
        if annotation.source != "human" or annotation.status == "gold":
            raise ConflictError("Model and gold annotations are read-only")
    if evidence_payload is not None or expected_revision is not None:
        raise annotation_types.AnnotationValidationError(
            "Evidence mutation fields are only valid for evidence_block annotations"
        )
    merged_values = {
        "project_id": annotation.project_id,
        "document_id": annotation.document_id,
        "annotation_type": annotation.annotation_type,
        "label": annotation.label,
        "structure_version_id": annotation.structure_version_id,
        "guideline_version_id": annotation.guideline_version_id,
        "start_offset": annotation.start_offset,
        "end_offset": annotation.end_offset,
        "head_annotation_id": annotation.head_annotation_id,
        "tail_annotation_id": annotation.tail_annotation_id,
        **updates,
    }
    _validate_annotation_values(db, merged_values, self_id=annotation.id)
    _validate_project_annotation_config(db, merged_values)
    document = db.get(Document, annotation.document_id)
    if document is not None:
        _validate_span(merged_values, document)

    for field, value in updates.items():
        setattr(annotation, field, value)

    db.commit()
    db.refresh(annotation)
    return annotation


def delete_annotation(
    db: Session,
    annotation_id: int,
    *,
    expected_revision: int | None = None,
    actor_user_id: int | None = None,
    required_assignment_id: int | None = None,
) -> bool:
    candidate = db.get(Annotation, annotation_id)
    if candidate is None:
        return False
    if candidate.annotation_type == "evidence_block":
        from al_medlit.evidence.service import delete_evidence_block

        return delete_evidence_block(
            db,
            annotation_id,
            expected_revision=expected_revision,
            actor_user_id=actor_user_id,
            required_assignment_id=required_assignment_id,
        )
    annotation = (
        db.query(Annotation)
        .filter(Annotation.id == annotation_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if annotation is None:
        return False
    if actor_user_id is not None:
        if annotation.annotator_user_id != actor_user_id:
            raise ForbiddenError("Annotation belongs to another annotator")
        if annotation.source != "human" or annotation.status == "gold":
            raise ConflictError("Model and gold annotations are read-only")
    assert_annotations_deletable(db, (annotation_id,))

    db.delete(annotation)
    db.commit()
    return True


def assert_annotations_deletable(
    db: Session,
    annotation_ids: tuple[int, ...] | list[int],
) -> None:
    """Reject deletion when records outside the delete set still reference annotations."""

    ids = tuple(sorted(set(annotation_ids)))
    if not ids:
        return
    dependent_relation = (
        db.query(Annotation.id)
        .filter(
            ~Annotation.id.in_(ids),
            or_(
                Annotation.head_annotation_id.in_(ids),
                Annotation.tail_annotation_id.in_(ids),
            ),
        )
        .first()
    )
    if dependent_relation is not None:
        raise AnnotationInUseError(
            "Annotation is referenced by relation annotations; delete those relations first."
        )

    dependent_correction = (
        db.query(AnnotationCorrection.id)
        .filter(
            or_(
                AnnotationCorrection.original_annotation_id.in_(ids),
                AnnotationCorrection.corrected_annotation_id.in_(ids),
            ),
        )
        .first()
    )
    if dependent_correction is not None:
        raise AnnotationInUseError(
            "Annotation is referenced by annotation corrections; delete those corrections first."
        )

    from al_medlit.inference.models import EvidencePredictionReview
    from al_medlit.lineage.models import AnnotationSetItem

    dependent_prediction_review = (
        db.query(EvidencePredictionReview.id)
        .filter(EvidencePredictionReview.resulting_annotation_id.in_(ids))
        .first()
    )
    if dependent_prediction_review is not None:
        raise AnnotationInUseError(
            "Annotation is referenced by an inference review and must remain auditable."
        )
    dependent_annotation_set = (
        db.query(AnnotationSetItem.id)
        .filter(AnnotationSetItem.source_annotation_id.in_(ids))
        .first()
    )
    if dependent_annotation_set is not None:
        raise AnnotationInUseError(
            "Annotation is referenced by a lineage annotation set and must remain auditable."
        )


def get_correction_locked_annotation_ids(
    db: Session,
    document_id: int,
    *,
    user: User | None = None,
) -> list[int]:
    """Annotation ids in a document that a correction references and so cannot be deleted."""
    document_project_id = db.query(Document.project_id).filter(
        Document.id == document_id
    ).scalar()
    if document_project_id is None:
        return []
    rows = db.query(
        AnnotationCorrection.original_annotation_id,
        AnnotationCorrection.corrected_annotation_id,
    ).filter(
        AnnotationCorrection.document_id == document_id,
        AnnotationCorrection.project_id == document_project_id,
    ).all()
    locked = {
        annotation_id
        for original_id, corrected_id in rows
        for annotation_id in (original_id, corrected_id)
        if annotation_id is not None
    }
    if user is not None:
        visible_annotation_ids = {
            annotation_id
            for (annotation_id,) in (
                db.query(Annotation.id)
                .join(Project, Project.id == Annotation.project_id)
                .filter(
                    Annotation.document_id == document_id,
                    resource_access_clause(
                        user,
                        workspace_id=Project.workspace_id,
                        project_id=Annotation.project_id,
                        document_id=Annotation.document_id,
                        annotation_type=Annotation.annotation_type,
                        owner_user_id=Annotation.annotator_user_id,
                    ),
                )
                .all()
            )
        }
        locked.intersection_update(visible_annotation_ids)
    return sorted(locked)


def create_correction(
    db: Session,
    data: AnnotationCorrectionCreate,
    *,
    created_by_user_id: int | None,
) -> AnnotationCorrection:
    payload = data.model_dump()
    payload["created_by_user_id"] = created_by_user_id
    _validate_scope(
        db,
        project_id=payload["project_id"],
        document_id=payload["document_id"],
    )
    _validate_correction_references(db, payload)

    correction = AnnotationCorrection(**payload)
    db.add(correction)
    db.commit()
    db.refresh(correction)

    event_bus.publish(
        "annotation.correction.created",
        {
            "correction_id": correction.id,
            "project_id": correction.project_id,
            "document_id": correction.document_id,
            "created_by_user_id": correction.created_by_user_id,
            "error_type": correction.error_type,
        },
    )
    return correction


def list_corrections(
    db: Session,
    project_id: int | None = None,
    user: User | None = None,
) -> list[AnnotationCorrection]:
    query = db.query(AnnotationCorrection)
    if project_id is not None:
        query = query.filter(AnnotationCorrection.project_id == project_id)
    if user is not None:
        query = query.join(
            Project,
            Project.id == AnnotationCorrection.project_id,
        ).join(
            Document,
            Document.id == AnnotationCorrection.document_id,
        ).filter(
            Document.project_id == AnnotationCorrection.project_id,
            resource_access_clause(
                user,
                workspace_id=Project.workspace_id,
                project_id=AnnotationCorrection.project_id,
                document_id=AnnotationCorrection.document_id,
                owner_user_id=AnnotationCorrection.created_by_user_id,
            ),
            annotation_reference_access_clause(
                user,
                AnnotationCorrection.original_annotation_id,
                project_id=AnnotationCorrection.project_id,
                document_id=AnnotationCorrection.document_id,
            ),
            annotation_reference_access_clause(
                user,
                AnnotationCorrection.corrected_annotation_id,
                project_id=AnnotationCorrection.project_id,
                document_id=AnnotationCorrection.document_id,
            ),
        )
    return query.order_by(AnnotationCorrection.created_at.desc()).all()
