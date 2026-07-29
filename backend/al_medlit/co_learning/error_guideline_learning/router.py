from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from al_medlit.annotation.models import AnnotationCorrection
from al_medlit.auth.dependencies import get_current_user
from al_medlit.auth.models import User
from al_medlit.auth.tenancy import assert_document_resource_access, assert_project_member
from al_medlit.co_learning.error_guideline_learning import service
from al_medlit.co_learning.error_guideline_learning.models import (
    ErrorPattern,
    GuidelineAtom,
)
from al_medlit.co_learning.error_guideline_learning.schemas import (
    ApproveGuidelineAtomRequest,
    ErrorPatternCreate,
    ErrorPatternRead,
    GuidelineAtomRead,
    MicroQuestionTemplateCreate,
    MicroQuestionTemplateRead,
    PatternToGuidelineRequest,
    TrainingActionCreate,
    TrainingActionRead,
)
from al_medlit.core.database import get_db
from al_medlit.core.exceptions import ForbiddenError, ValidationError
from al_medlit.workspace.capability_dependencies import (
    enforce_capability,
)

router = APIRouter(
    prefix="/co-learning/error-guideline",
    tags=["co-learning: error-guideline learning"],
)


def _enforce_project_access_and_capability(
    db: Session,
    current_user: User,
    project_id: int,
) -> None:
    # Co-learning surfaces aggregate correction-derived evidence across
    # annotators, so they are part of the manager/adjudication boundary.
    assert_project_member(db, current_user, project_id, min_role="manager")
    enforce_capability(db, project_id=project_id, key="co_learning")


def _require_error_pattern(db: Session, pattern_id: int) -> ErrorPattern:
    pattern = db.get(ErrorPattern, pattern_id)
    if pattern is None:
        raise HTTPException(status_code=404, detail="ErrorPattern not found")
    return pattern


def _require_guideline_atom(db: Session, atom_id: int) -> GuidelineAtom:
    atom = db.get(GuidelineAtom, atom_id)
    if atom is None:
        raise HTTPException(status_code=404, detail="GuidelineAtom not found")
    return atom


def _assert_same_project(
    *,
    project_id: int,
    referenced_project_id: int,
    field_name: str,
) -> None:
    if referenced_project_id != project_id:
        raise ValidationError(f"{field_name} does not belong to project {project_id}")


@router.post("/patterns", response_model=ErrorPatternRead)
def create_error_pattern(
    payload: ErrorPatternCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_project_access_and_capability(db, current_user, payload.project_id)
    return service.create_error_pattern(db, payload)


@router.get("/patterns", response_model=list[ErrorPatternRead])
def list_error_patterns(
    project_id: int = Query(...),
    status: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_project_access_and_capability(db, current_user, project_id)
    return service.list_error_patterns(db, project_id, status)


@router.post("/patterns/from-correction/{correction_id}", response_model=ErrorPatternRead)
def create_pattern_from_correction(
    correction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    correction = db.get(AnnotationCorrection, correction_id)
    if not correction:
        raise HTTPException(status_code=404, detail="Correction not found")
    _enforce_project_access_and_capability(db, current_user, correction.project_id)
    assert_document_resource_access(
        db,
        current_user,
        project_id=correction.project_id,
        document_id=correction.document_id,
        min_role="manager",
    )
    return service.upsert_pattern_from_correction(db, correction)


@router.post("/patterns/{pattern_id}/draft-guideline-atom", response_model=GuidelineAtomRead)
def draft_guideline_atom(
    pattern_id: int,
    payload: PatternToGuidelineRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    pattern = _require_error_pattern(db, pattern_id)
    _enforce_project_access_and_capability(db, current_user, pattern.project_id)
    try:
        return service.draft_guideline_atom(
            db,
            pattern_id=pattern_id,
            manager_id=payload.manager_id,
            rule_text_override=payload.rule_text_override,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/guideline-atoms", response_model=list[GuidelineAtomRead])
def list_guideline_atoms(
    project_id: int = Query(...),
    status: str | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_project_access_and_capability(db, current_user, project_id)
    return service.list_guideline_atoms(db, project_id, status)


@router.post("/guideline-atoms/{atom_id}/approve", response_model=GuidelineAtomRead)
def approve_guideline_atom(
    atom_id: int,
    payload: ApproveGuidelineAtomRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    atom = _require_guideline_atom(db, atom_id)
    _enforce_project_access_and_capability(db, current_user, atom.project_id)
    try:
        return service.approve_guideline_atom(db, atom_id, payload.approved_by)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/training-actions", response_model=TrainingActionRead)
def create_training_action(
    payload: TrainingActionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_project_access_and_capability(db, current_user, payload.project_id)
    if payload.error_pattern_id is not None:
        pattern = _require_error_pattern(db, payload.error_pattern_id)
        _assert_same_project(
            project_id=payload.project_id,
            referenced_project_id=pattern.project_id,
            field_name="error_pattern_id",
        )
    if payload.guideline_atom_id is not None:
        atom = _require_guideline_atom(db, payload.guideline_atom_id)
        _assert_same_project(
            project_id=payload.project_id,
            referenced_project_id=atom.project_id,
            field_name="guideline_atom_id",
        )
    return service.create_training_action(db, payload)


@router.get("/training-actions", response_model=list[TrainingActionRead])
def list_training_actions(
    project_id: int = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _enforce_project_access_and_capability(db, current_user, project_id)
    return service.list_training_actions(db, project_id)


@router.post("/micro-question-templates", response_model=MicroQuestionTemplateRead)
def create_micro_question_template(
    payload: MicroQuestionTemplateCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if payload.guideline_atom_id is not None:
        atom = _require_guideline_atom(db, payload.guideline_atom_id)
        if payload.project_id is None:
            payload = payload.model_copy(update={"project_id": atom.project_id})
        else:
            _assert_same_project(
                project_id=payload.project_id,
                referenced_project_id=atom.project_id,
                field_name="guideline_atom_id",
            )
    if payload.project_id is None:
        if not current_user.is_superuser:
            raise ForbiddenError("Only superusers can create global micro-question templates")
    else:
        _enforce_project_access_and_capability(db, current_user, payload.project_id)
    return service.create_micro_question_template(db, payload)


@router.get("/micro-question-templates", response_model=list[MicroQuestionTemplateRead])
def list_micro_question_templates(
    project_id: int | None = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if project_id is not None:
        _enforce_project_access_and_capability(db, current_user, project_id)
    return service.list_micro_question_templates(db, project_id)
