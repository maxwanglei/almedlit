from pydantic import BaseModel, Field


class ErrorPatternCreate(BaseModel):
    project_id: int
    task_type: str
    error_type: str
    label_type: str | None = None
    description: str
    example_count: int = 1
    severity: str = "medium"
    detected_from: str = "manual"
    status: str = "active"
    example_ids: list = Field(default_factory=list)
    metadata_: dict = Field(default_factory=dict)


class ErrorPatternRead(ErrorPatternCreate):
    id: int

    model_config = {"from_attributes": True}


class GuidelineAtomCreate(BaseModel):
    project_id: int
    guideline_version_id: int | None = None
    error_pattern_id: int | None = None
    task_type: str
    error_type: str | None = None
    rule_text: str
    rationale: str | None = None
    positive_examples: list = Field(default_factory=list)
    negative_examples: list = Field(default_factory=list)
    applies_to: list = Field(default_factory=list)
    status: str = "pending"
    approved_by: str | None = None


class GuidelineAtomRead(GuidelineAtomCreate):
    id: int

    model_config = {"from_attributes": True}


class TrainingActionCreate(BaseModel):
    project_id: int
    error_pattern_id: int | None = None
    guideline_atom_id: int | None = None
    action_type: str
    target_model: str
    example_ids: list = Field(default_factory=list)
    priority: str = "medium"
    status: str = "planned"
    notes: str | None = None


class TrainingActionRead(TrainingActionCreate):
    id: int

    model_config = {"from_attributes": True}


class MicroQuestionTemplateCreate(BaseModel):
    project_id: int | None = None
    guideline_atom_id: int | None = None
    error_type: str
    target_annotation_type: str = "entity"
    question_text: str
    answer_options: list = Field(default_factory=list)
    status: str = "active"


class MicroQuestionTemplateRead(MicroQuestionTemplateCreate):
    id: int

    model_config = {"from_attributes": True}


class PatternToGuidelineRequest(BaseModel):
    manager_id: str | None = None
    use_llm: bool = False
    rule_text_override: str | None = None


class ApproveGuidelineAtomRequest(BaseModel):
    approved_by: str
