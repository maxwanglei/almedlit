from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from al_medlit.annotation.types import (
    REGISTRY as ANNOTATION_TYPE_REGISTRY,
)
from al_medlit.annotation.types import (
    AnnotationType,
)

KNOWN_ANNOTATION_TYPES = frozenset(ANNOTATION_TYPE_REGISTRY)
AnnotationValidationMode = Literal["strict", "relaxed"]
TaskAssignmentStatus = Literal[
    "assigned",
    "in_progress",
    "submitted",
    "adjudication_ready",
    "adjudicated",
    "completed",
    "blocked",
    "withdrawn",
]


class LabelDef(BaseModel):
    name: str
    color: str = "#888888"
    description: str | None = None


class LabelSet(BaseModel):
    labels: dict[str, list[LabelDef]] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        normalized = dict(value)
        normalized.pop("annotation_types", None)

        labels = normalized.get("labels")
        if isinstance(labels, list):
            normalized["labels"] = {"entity": labels}

        return normalized

    @model_validator(mode="after")
    def validate_annotation_type_keys(self) -> "LabelSet":
        unknown = set(self.labels) - KNOWN_ANNOTATION_TYPES
        if unknown:
            unknown_list = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown annotation_type keys in labels: {unknown_list}"
            )
        return self


class RelationConstraint(BaseModel):
    head: list[str] = Field(default_factory=list)
    tail: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class EvidenceBlockKeyboardShortcutsV1(BaseModel):
    create: str = "e"
    expand_start: str = "shift+arrowup"
    expand_end: str = "shift+arrowdown"
    contract_start: str = "alt+arrowup"
    contract_end: str = "alt+arrowdown"
    merge: str = "m"
    split: str = "s"
    delete: str = "delete"
    mark_reviewed: str = "r"
    cancel: str = "escape"

    model_config = {"extra": "forbid"}

    @field_validator("*", mode="before")
    @classmethod
    def normalize_shortcut(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("evidence-block keyboard shortcuts cannot be empty")
        return value.strip().lower().replace(" ", "")


class EvidenceBlockTaskSettingsV1(BaseModel):
    schema_version: Literal["1"] = "1"
    active_target_ids: list[int] = Field(default_factory=list)
    sentence_boundaries: bool = True
    multi_paragraph_allowed: bool = True
    cross_section_allowed: bool = False
    same_target_overlap_allowed: bool = False
    adjacency_allowed: bool = True
    soft_token_warning: int = Field(default=3072, ge=1)
    model_context_tokens: int = Field(default=4096, ge=1)
    window_overlap_tokens: int = Field(default=512, ge=0)
    review_scope: Literal["document"] = "document"
    keyboard_shortcuts: EvidenceBlockKeyboardShortcutsV1 = Field(
        default_factory=EvidenceBlockKeyboardShortcutsV1
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_token_limits(self) -> "EvidenceBlockTaskSettingsV1":
        if self.soft_token_warning > self.model_context_tokens:
            raise ValueError("soft_token_warning cannot exceed model_context_tokens")
        if self.window_overlap_tokens >= self.model_context_tokens:
            raise ValueError("window_overlap_tokens must be less than model_context_tokens")
        if len(self.active_target_ids) != len(set(self.active_target_ids)):
            raise ValueError("active_target_ids must not contain duplicates")
        configured_shortcuts = list(
            self.keyboard_shortcuts.model_dump(mode="json").values()
        )
        if len(configured_shortcuts) != len(set(configured_shortcuts)):
            raise ValueError("evidence-block keyboard shortcuts must be unique")
        return self


def validate_relation_constraint_settings(settings: dict) -> dict:
    constraints = settings.get("relation_constraints")
    if constraints is None:
        return settings
    if not isinstance(constraints, dict):
        raise ValueError("relation_constraints must be an object keyed by relation label")
    for label, constraint in constraints.items():
        try:
            RelationConstraint.model_validate(constraint)
        except ValidationError as exc:
            first_error = exc.errors()[0]
            raise ValueError(
                f"Invalid relation constraint for label '{label}': "
                f"{first_error['msg']}"
            ) from exc
    return settings


class ProjectTaskBase(BaseModel):
    annotation_type: AnnotationType
    display_name: str | None = None
    description: str | None = None
    enabled: bool = True
    sort_order: int = 0
    labels: list[LabelDef] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)

    @field_validator("settings")
    @classmethod
    def validate_task_settings(cls, value: dict) -> dict:
        return validate_relation_constraint_settings(value)

    @model_validator(mode="after")
    def validate_typed_settings(self) -> "ProjectTaskBase":
        if self.annotation_type == "evidence_block":
            self.settings = EvidenceBlockTaskSettingsV1.model_validate(
                self.settings
            ).model_dump()
        return self


class ProjectTaskCreate(ProjectTaskBase):
    pass


class ProjectTaskUpdate(BaseModel):
    display_name: str | None = None
    description: str | None = None
    enabled: bool | None = None
    sort_order: int | None = None
    labels: list[LabelDef] | None = None
    settings: dict | None = None

    @field_validator("settings")
    @classmethod
    def validate_task_settings(cls, value: dict | None) -> dict | None:
        if value is None:
            return value
        return validate_relation_constraint_settings(value)


class ProjectTaskRead(ProjectTaskBase):
    id: int
    project_id: int
    display_name: str

    model_config = {"from_attributes": True}


class TaskAssignmentCreate(BaseModel):
    task_id: int
    document_id: int
    assignee_user_id: int | None = None
    annotator_id: str | None = None
    target_version_id: int | None = None
    structure_version_id: int | None = None
    guideline_version_id: int | None = None
    status: TaskAssignmentStatus = "assigned"
    assigned_by_user_id: int | None = None
    assigned_by: str | None = None
    notes: str | None = None
    metadata_: dict = Field(default_factory=dict)


class TaskAssignmentUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assignee_user_id: int | None = None
    annotator_id: str | None = None
    status: TaskAssignmentStatus | None = None
    notes: str | None = None
    metadata_: dict | None = None

    @model_validator(mode="after")
    def reject_null_for_non_nullable_fields(self) -> "TaskAssignmentUpdate":
        for field in ("status", "metadata_"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class TaskAssignmentRead(TaskAssignmentCreate):
    id: int
    project_id: int
    assignee_user_id: int
    annotator_id: str
    assignment_scope_key: str

    model_config = {"from_attributes": True}


class StatusCount(BaseModel):
    total: int
    by_status: dict[str, int]


class TaskProgress(StatusCount):
    task_id: int
    annotation_type: AnnotationType
    display_name: str


class TargetProgress(StatusCount):
    target_version_id: int
    target_id: int
    target_key: str
    target_name: str


class DocumentProgress(StatusCount):
    document_id: int


class AnnotatorProgress(StatusCount):
    assignee_user_id: int | None = None
    annotator_id: str


class ProjectProgressRead(StatusCount):
    project_id: int
    by_task: list[TaskProgress]
    by_document: list[DocumentProgress]
    by_annotator: list[AnnotatorProgress]
    by_target: list[TargetProgress] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str
    description: str | None = None
    annotation_schema: LabelSet = Field(default_factory=LabelSet)
    annotation_validation_mode: AnnotationValidationMode = "relaxed"
    tasks: list[ProjectTaskCreate] = Field(default_factory=list)
    settings: dict = Field(default_factory=dict)
    workspace_id: int | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    annotation_validation_mode: AnnotationValidationMode | None = None
    settings: dict | None = None


class ProjectRead(ProjectCreate):
    id: int
    tasks: list[ProjectTaskRead] = Field(default_factory=list)

    model_config = {"from_attributes": True}
