from pydantic import BaseModel

from al_medlit.annotation.schemas import AnnotationRead
from al_medlit.annotation.types import AnnotationType
from al_medlit.corpus.schemas import DocumentRead
from al_medlit.guideline.schemas import GuidelineVersionRead
from al_medlit.project.schemas import ProjectRead, ProjectTaskRead, TaskAssignmentRead


class AnnotationTypeSpecRead(BaseModel):
    name: AnnotationType
    requires_span: bool
    requires_head_tail: bool
    description: str
    selection_mode: str
    renderer_key: str
    relation_endpoint_allowed: bool
    handler_key: str


class AnnotationWorkbenchTaskRead(ProjectTaskRead):
    annotation_type_spec: AnnotationTypeSpecRead


class AnnotationWorkbenchRead(BaseModel):
    project: ProjectRead
    document: DocumentRead
    active_guideline: GuidelineVersionRead | None
    guideline_versions_by_id: dict[int, GuidelineVersionRead]
    tasks: list[AnnotationWorkbenchTaskRead]
    annotation_type_specs: list[AnnotationTypeSpecRead]
    annotations: list[AnnotationRead]
    assignments: list[TaskAssignmentRead]
    correction_locked_annotation_ids: list[int]
