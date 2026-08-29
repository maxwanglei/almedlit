from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation, AnnotationCorrection
from al_medlit.co_learning.error_guideline_learning.guideline_writer import (
    draft_guideline_atom_from_pattern,
)
from al_medlit.co_learning.error_guideline_learning.models import (
    ErrorPattern,
    GuidelineAtom,
    MicroQuestionTemplate,
    TrainingAction,
)
from al_medlit.co_learning.error_guideline_learning.phenotyper import describe_error_type
from al_medlit.co_learning.error_guideline_learning.schemas import (
    ErrorPatternCreate,
    MicroQuestionTemplateCreate,
    TrainingActionCreate,
)
from al_medlit.core.exceptions import ConflictError


def create_error_pattern(db: Session, data: ErrorPatternCreate) -> ErrorPattern:
    existing_pattern = _find_matching_error_pattern(db, data)
    if existing_pattern is not None:
        return existing_pattern

    if data.status == "active" and _find_active_error_pattern(
        db,
        project_id=data.project_id,
        task_type=data.task_type,
        error_type=data.error_type,
        label_type=data.label_type,
    ) is not None:
        raise ConflictError("An active error pattern already exists for this scope")

    pattern = ErrorPattern(**data.model_dump())
    try:
        with db.begin_nested():
            db.add(pattern)
            db.flush()
    except IntegrityError:
        existing_pattern = _find_matching_error_pattern(db, data)
        if existing_pattern is not None:
            return existing_pattern
        if data.status == "active" and _find_active_error_pattern(
            db,
            project_id=data.project_id,
            task_type=data.task_type,
            error_type=data.error_type,
            label_type=data.label_type,
        ) is not None:
            raise ConflictError("An active error pattern already exists for this scope") from None
        raise
    db.commit()
    db.refresh(pattern)
    return pattern


def list_error_patterns(
    db: Session,
    project_id: int,
    status: str | None = None,
) -> list[ErrorPattern]:
    query = db.query(ErrorPattern).filter(ErrorPattern.project_id == project_id)
    if status:
        query = query.filter(ErrorPattern.status == status)
    return query.order_by(ErrorPattern.updated_at.desc()).all()


def upsert_pattern_from_correction(db: Session, correction: AnnotationCorrection) -> ErrorPattern:
    """Create or increment an ErrorPattern from an AnnotationCorrection.

    This is the first production hook: every adjudication/correction can feed the
    error-guideline learning loop.
    """

    error_type = correction.error_type or "guideline_ambiguity"

    corrected = (
        db.get(Annotation, correction.corrected_annotation_id)
        if correction.corrected_annotation_id
        else None
    )

    task_type = corrected.annotation_type if corrected else "entity"
    label_type = corrected.label if corrected else None

    pattern = _find_active_error_pattern(
        db,
        project_id=correction.project_id,
        task_type=task_type,
        error_type=error_type,
        label_type=label_type,
        for_update=True,
    )

    if pattern:
        _append_correction_example(pattern, correction)
    else:
        pattern = ErrorPattern(
            project_id=correction.project_id,
            task_type=task_type,
            error_type=error_type,
            label_type=label_type,
            description=describe_error_type(error_type),
            example_count=1,
            severity=correction.severity,
            detected_from=correction.correction_source,
            example_ids=[{"correction_id": correction.id, "document_id": correction.document_id}],
        )
        try:
            with db.begin_nested():
                db.add(pattern)
                db.flush()
        except IntegrityError:
            pattern = _find_active_error_pattern(
                db,
                project_id=correction.project_id,
                task_type=task_type,
                error_type=error_type,
                label_type=label_type,
                for_update=True,
            )
            if pattern is None:
                raise
            _append_correction_example(pattern, correction)

    db.commit()
    db.refresh(pattern)
    return pattern


def _find_active_error_pattern(
    db: Session,
    *,
    project_id: int,
    task_type: str,
    error_type: str,
    label_type: str | None,
    for_update: bool = False,
) -> ErrorPattern | None:
    query = db.query(ErrorPattern).filter(
        ErrorPattern.project_id == project_id,
        ErrorPattern.task_type == task_type,
        ErrorPattern.error_type == error_type,
        ErrorPattern.label_type == label_type,
        ErrorPattern.status == "active",
    )
    if for_update:
        query = query.populate_existing().with_for_update()
    return query.one_or_none()


def _append_correction_example(
    pattern: ErrorPattern,
    correction: AnnotationCorrection,
) -> bool:
    example_ids = list(pattern.example_ids or [])
    if any(
        isinstance(example, dict) and example.get("correction_id") == correction.id
        for example in example_ids
    ):
        return False
    example_ids.append(
        {"correction_id": correction.id, "document_id": correction.document_id}
    )
    pattern.example_count += 1
    pattern.example_ids = example_ids
    return True


def draft_guideline_atom(
    db: Session,
    pattern_id: int,
    manager_id: str | None = None,
    rule_text_override: str | None = None,
) -> GuidelineAtom:
    pattern = db.get(ErrorPattern, pattern_id)
    if pattern is None:
        raise ValueError("ErrorPattern not found")

    rule_text, rationale = draft_guideline_atom_from_pattern(pattern)
    if rule_text_override:
        rule_text = rule_text_override

    existing_atom = (
        db.query(GuidelineAtom)
        .filter(
            GuidelineAtom.project_id == pattern.project_id,
            GuidelineAtom.error_pattern_id == pattern.id,
            GuidelineAtom.task_type == pattern.task_type,
            GuidelineAtom.error_type == pattern.error_type,
            GuidelineAtom.rule_text == rule_text,
            GuidelineAtom.status.in_(("pending", "accepted")),
        )
        .order_by(GuidelineAtom.created_at.desc())
        .first()
    )
    if existing_atom is not None:
        if manager_id is not None and existing_atom.approved_by is None:
            existing_atom.approved_by = manager_id
            db.commit()
            db.refresh(existing_atom)
        return existing_atom

    atom = GuidelineAtom(
        project_id=pattern.project_id,
        error_pattern_id=pattern.id,
        task_type=pattern.task_type,
        error_type=pattern.error_type,
        rule_text=rule_text,
        rationale=rationale,
        positive_examples=[],
        negative_examples=pattern.example_ids or [],
        applies_to=[
            "guideline",
            "annotation_ui_hint",
            "llm_prompt",
            "active_learning",
            "training",
            "calibration",
        ],
        status="pending",
        approved_by=manager_id,
    )
    db.add(atom)
    db.commit()
    db.refresh(atom)
    return atom


def approve_guideline_atom(db: Session, atom_id: int, approved_by: str) -> GuidelineAtom:
    atom = db.get(GuidelineAtom, atom_id)
    if atom is None:
        raise ValueError("GuidelineAtom not found")

    atom.status = "accepted"
    atom.approved_by = approved_by

    # When a guideline atom is accepted, create a default micro-question and training action.
    _ensure_default_micro_question_template(db, atom)
    _ensure_default_training_action(db, atom)

    db.commit()
    db.refresh(atom)
    return atom


def list_guideline_atoms(
    db: Session,
    project_id: int,
    status: str | None = None,
) -> list[GuidelineAtom]:
    query = db.query(GuidelineAtom).filter(GuidelineAtom.project_id == project_id)
    if status:
        query = query.filter(GuidelineAtom.status == status)
    return query.order_by(GuidelineAtom.updated_at.desc()).all()


def create_training_action(db: Session, data: TrainingActionCreate) -> TrainingAction:
    existing_action = _find_matching_training_action(db, data)
    if existing_action is not None:
        return existing_action

    action = TrainingAction(**data.model_dump())
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def list_training_actions(db: Session, project_id: int) -> list[TrainingAction]:
    return (
        db.query(TrainingAction)
        .filter(TrainingAction.project_id == project_id)
        .order_by(TrainingAction.created_at.desc())
        .all()
    )


def create_micro_question_template(
    db: Session,
    data: MicroQuestionTemplateCreate,
) -> MicroQuestionTemplate:
    existing_template = _find_matching_micro_question_template(db, data)
    if existing_template is not None:
        return existing_template

    template = MicroQuestionTemplate(**data.model_dump())
    db.add(template)
    db.commit()
    db.refresh(template)
    return template


def list_micro_question_templates(
    db: Session,
    project_id: int | None = None,
) -> list[MicroQuestionTemplate]:
    query = db.query(MicroQuestionTemplate)
    if project_id is not None:
        query = query.filter(
            (MicroQuestionTemplate.project_id == project_id)
            | (MicroQuestionTemplate.project_id.is_(None))
        )
    else:
        query = query.filter(MicroQuestionTemplate.project_id.is_(None))
    return query.order_by(MicroQuestionTemplate.created_at.desc()).all()


def _default_question_for_error(error_type: str | None) -> str:
    if error_type == "boundary_error":
        return "What is the correct span boundary?"
    if error_type == "ontology_error":
        return "Which normalized concept is correct?"
    if error_type == "role_confusion":
        return "What role does this mention play in the study context?"
    if error_type == "evidence_error":
        return "Does the evidence span directly support this annotation?"
    return "How should this recurring annotation case be handled?"


def _training_action_for_error(error_type: str | None) -> str:
    if error_type == "boundary_error":
        return "add_boundary_hard_examples"
    if error_type == "ontology_error":
        return "improve_normalization_examples"
    if error_type == "role_confusion":
        return "train_role_classifier"
    if error_type == "evidence_error":
        return "train_evidence_verifier"
    return "add_hard_examples"


def _target_model_for_task(task_type: str, error_type: str | None) -> str:
    if error_type == "evidence_error":
        return "evidence_verifier"
    if error_type == "ontology_error":
        return "ontology_linker"
    if error_type == "role_confusion":
        return "role_classifier"
    if task_type == "entity":
        return "bert_ner"
    return f"{task_type}_model"


def _ensure_default_micro_question_template(
    db: Session,
    atom: GuidelineAtom,
) -> MicroQuestionTemplate:
    error_type = atom.error_type or "guideline_ambiguity"
    question_text = _default_question_for_error(atom.error_type)

    existing_template = (
        db.query(MicroQuestionTemplate)
        .filter(
            MicroQuestionTemplate.guideline_atom_id == atom.id,
            MicroQuestionTemplate.error_type == error_type,
            MicroQuestionTemplate.target_annotation_type == atom.task_type,
            MicroQuestionTemplate.question_text == question_text,
        )
        .first()
    )
    if existing_template is not None:
        return existing_template

    template = MicroQuestionTemplate(
        project_id=atom.project_id,
        guideline_atom_id=atom.id,
        error_type=error_type,
        target_annotation_type=atom.task_type,
        question_text=question_text,
        answer_options=[],
        status="active",
    )
    db.add(template)
    return template


def _ensure_default_training_action(db: Session, atom: GuidelineAtom) -> TrainingAction:
    action_type = _training_action_for_error(atom.error_type)
    target_model = _target_model_for_task(atom.task_type, atom.error_type)

    existing_action = (
        db.query(TrainingAction)
        .filter(
            TrainingAction.guideline_atom_id == atom.id,
            TrainingAction.error_pattern_id == atom.error_pattern_id,
            TrainingAction.action_type == action_type,
            TrainingAction.target_model == target_model,
        )
        .first()
    )
    if existing_action is not None:
        return existing_action

    action = TrainingAction(
        project_id=atom.project_id,
        error_pattern_id=atom.error_pattern_id,
        guideline_atom_id=atom.id,
        action_type=action_type,
        target_model=target_model,
        example_ids=atom.negative_examples or [],
        priority="medium",
        status="planned",
        notes="Automatically planned after guideline atom approval.",
    )
    db.add(action)
    return action


def _find_matching_error_pattern(
    db: Session,
    data: ErrorPatternCreate,
) -> ErrorPattern | None:
    query = (
        db.query(ErrorPattern)
        .filter(
            ErrorPattern.project_id == data.project_id,
            ErrorPattern.task_type == data.task_type,
            ErrorPattern.error_type == data.error_type,
            ErrorPattern.status == data.status,
        )
        .order_by(ErrorPattern.created_at.desc())
    )
    query = _filter_optional_value(query, ErrorPattern.label_type, data.label_type)

    for pattern in query.all():
        if (
            pattern.description == data.description
            and pattern.example_count == data.example_count
            and pattern.severity == data.severity
            and pattern.detected_from == data.detected_from
            and pattern.example_ids == data.example_ids
            and pattern.metadata_ == data.metadata_
        ):
            return pattern

    return None


def _find_matching_training_action(
    db: Session,
    data: TrainingActionCreate,
) -> TrainingAction | None:
    query = (
        db.query(TrainingAction)
        .filter(
            TrainingAction.project_id == data.project_id,
            TrainingAction.action_type == data.action_type,
            TrainingAction.target_model == data.target_model,
            TrainingAction.priority == data.priority,
            TrainingAction.status == data.status,
        )
        .order_by(TrainingAction.created_at.desc())
    )
    query = _filter_optional_value(
        query, TrainingAction.error_pattern_id, data.error_pattern_id
    )
    query = _filter_optional_value(
        query, TrainingAction.guideline_atom_id, data.guideline_atom_id
    )

    for action in query.all():
        if action.example_ids == data.example_ids and action.notes == data.notes:
            return action

    return None


def _find_matching_micro_question_template(
    db: Session,
    data: MicroQuestionTemplateCreate,
) -> MicroQuestionTemplate | None:
    query = (
        db.query(MicroQuestionTemplate)
        .filter(
            MicroQuestionTemplate.error_type == data.error_type,
            MicroQuestionTemplate.target_annotation_type == data.target_annotation_type,
            MicroQuestionTemplate.question_text == data.question_text,
            MicroQuestionTemplate.status == data.status,
        )
        .order_by(MicroQuestionTemplate.created_at.desc())
    )
    query = _filter_optional_value(
        query, MicroQuestionTemplate.project_id, data.project_id
    )
    query = _filter_optional_value(
        query, MicroQuestionTemplate.guideline_atom_id, data.guideline_atom_id
    )

    for template in query.all():
        if template.answer_options == data.answer_options:
            return template

    return None


def _filter_optional_value(query, column, value):
    if value is None:
        return query.filter(column.is_(None))
    return query.filter(column == value)
