"""Build the Project JSON snapshot written to object storage on submission.

format_version 1: a single self-describing JSON document containing the
project labeling context, the full document text, and every annotation on
the document at export time.
"""

from sqlalchemy.orm import Session

from al_medlit.annotation import service as annotation_service
from al_medlit.annotation.schemas import AnnotationRead
from al_medlit.auth.models import User
from al_medlit.core.models import utc_now
from al_medlit.corpus.models import Document
from al_medlit.project.models import Project, TaskAssignment

FORMAT_VERSION = 1


def build_snapshot(
    db: Session,
    *,
    project: Project,
    document: Document,
    annotator_user_id: int | None,
    annotator_id: str | None,
    user: User | None = None,
    assignment: TaskAssignment | None = None,
) -> dict:
    annotations = annotation_service.list_annotations(
        db,
        project.id,
        document.id,
        annotator_user_id=annotator_user_id,
        annotator_id=annotator_id,
        user=user,
        target_version_id=(
            assignment.target_version_id if assignment is not None else None
        ),
        structure_version_id=(
            assignment.structure_version_id
            if assignment is not None
            else None
        ),
        guideline_version_id=(
            assignment.guideline_version_id
            if assignment is not None
            else None
        ),
        filter_guideline_version=assignment is not None,
        annotation_type=(
            assignment.task.annotation_type if assignment is not None else None
        ),
    )
    return {
        "format_version": FORMAT_VERSION,
        "exported_at": utc_now().isoformat(),
        "project": {
            "id": project.id,
            "name": project.name,
            "annotation_schema": project.annotation_schema,
        },
        "document": {
            "id": document.id,
            "external_id": document.external_id,
            "title": document.title,
            "text": document.text,
        },
        "annotator_id": annotator_id,
        "assignment": (
            {
                "id": assignment.id,
                "task_id": assignment.task_id,
                "annotation_type": assignment.task.annotation_type,
                "assignment_scope_key": assignment.assignment_scope_key,
                "target_version_id": assignment.target_version_id,
                "structure_version_id": assignment.structure_version_id,
                "guideline_version_id": assignment.guideline_version_id,
            }
            if assignment is not None
            else None
        ),
        "annotations": [
            AnnotationRead.model_validate(annotation).model_dump(mode="json")
            for annotation in annotations
        ],
    }
