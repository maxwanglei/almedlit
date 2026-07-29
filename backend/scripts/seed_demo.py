from sqlalchemy.orm import Session

from al_medlit.annotation.models import Annotation, AnnotationCorrection
from al_medlit.annotation.schemas import AnnotationCorrectionCreate, AnnotationCreate
from al_medlit.annotation.service import create_annotation, create_correction
from al_medlit.co_learning.error_guideline_learning.events import register_event_handlers
from al_medlit.co_learning.error_guideline_learning.models import (
    ErrorPattern,
    GuidelineAtom,
    MicroQuestionTemplate,
    TrainingAction,
)
from al_medlit.co_learning.error_guideline_learning.service import (
    approve_guideline_atom,
    draft_guideline_atom,
    list_error_patterns,
)
from al_medlit.core.database import SessionLocal, ensure_schema_ready
from al_medlit.corpus.models import Document
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.corpus.service import create_document
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.guideline.schemas import GuidelineVersionCreate
from al_medlit.guideline.service import create_guideline_version
from al_medlit.project.models import Project
from al_medlit.project.schemas import ProjectCreate
from al_medlit.project.service import create_project

DEMO_PROJECT_NAME = "Drug NER Demo"


def _find_existing_demo_project(db: Session) -> Project | None:
    return db.query(Project).filter(Project.name == DEMO_PROJECT_NAME).first()


def _missing_demo_seed_components(db: Session, project_id: int) -> list[str]:
    component_queries = [
        ("documents", db.query(Document).filter(Document.project_id == project_id)),
        (
            "guideline versions",
            db.query(GuidelineVersion).filter(GuidelineVersion.project_id == project_id),
        ),
        ("annotations", db.query(Annotation).filter(Annotation.project_id == project_id)),
        (
            "annotation corrections",
            db.query(AnnotationCorrection).filter(AnnotationCorrection.project_id == project_id),
        ),
        (
            "error patterns",
            db.query(ErrorPattern).filter(ErrorPattern.project_id == project_id),
        ),
        (
            "guideline atoms",
            db.query(GuidelineAtom).filter(GuidelineAtom.project_id == project_id),
        ),
        (
            "training actions",
            db.query(TrainingAction).filter(TrainingAction.project_id == project_id),
        ),
        (
            "micro-question templates",
            db.query(MicroQuestionTemplate).filter(
                MicroQuestionTemplate.project_id == project_id
            ),
        ),
    ]

    return [name for name, query in component_queries if query.first() is None]


def main() -> None:
    ensure_schema_ready()
    register_event_handlers()
    db = SessionLocal()
    try:
        existing_project = _find_existing_demo_project(db)
        if existing_project is not None:
            missing_components = _missing_demo_seed_components(db, existing_project.id)
            if missing_components:
                missing = ", ".join(missing_components)
                raise SystemExit(
                    "Demo project already exists but seed is incomplete. "
                    f"Missing demo artifacts: {missing}."
                )

            print("Demo project already seeded; nothing to do.")
            print(f"Project ID: {existing_project.id}")
            return

        project = create_project(
            db,
            ProjectCreate(
                name=DEMO_PROJECT_NAME,
                description="Demo project for AL-MedLit Error-Guideline Learning.",
                annotation_schema={
                    "labels": {
                        "entity": [
                            {"name": "Drug", "color": "#7aa2f7"},
                            {"name": "Disease", "color": "#e74c3c"},
                        ],
                        "relation": [
                            {"name": "treats", "color": "#2ecc71"},
                        ],
                        "sentence_label": [
                            {"name": "background", "color": "#95a5a6"},
                            {"name": "methods", "color": "#3498db"},
                            {"name": "results", "color": "#9b59b6"},
                            {"name": "conclusion", "color": "#f1c40f"},
                        ],
                    },
                },
            ),
        )

        create_guideline_version(
            db,
            GuidelineVersionCreate(
                project_id=project.id,
                version_label="v1",
                markdown="# Drug NER Guideline\n\nAnnotate drug mentions.",
                author_id="lei",
            ),
        )

        doc = create_document(
            db,
            DocumentCreate(
                project_id=project.id,
                external_id="PMID:demo001",
                title="Demo abstract",
                text="Patients receiving low-dose aspirin therapy were excluded.",
                source="manual",
            ),
        )

        model_ann = create_annotation(
            db,
            AnnotationCreate(
                project_id=project.id,
                document_id=doc.id,
                annotation_type="entity",
                label="Drug",
                start_offset=19,
                end_offset=43,
                text_span="low-dose aspirin therapy",
                source="model",
                status="rejected",
                confidence=0.91,
                model_checkpoint_id="pubmedbert-drug-ner-demo",
            ),
        )

        human_ann = create_annotation(
            db,
            AnnotationCreate(
                project_id=project.id,
                document_id=doc.id,
                annotation_type="entity",
                label="Drug",
                start_offset=28,
                end_offset=35,
                text_span="aspirin",
                source="human",
                status="gold",
                annotator_id="curator1",
            ),
        )

        disease_ann = create_annotation(
            db,
            AnnotationCreate(
                project_id=project.id,
                document_id=doc.id,
                annotation_type="entity",
                label="Disease",
                start_offset=0,
                end_offset=8,
                text_span="Patients",
                source="human",
                status="gold",
                annotator_id="curator1",
            ),
        )

        create_annotation(
            db,
            AnnotationCreate(
                project_id=project.id,
                document_id=doc.id,
                annotation_type="relation",
                label="treats",
                head_annotation_id=human_ann.id,
                tail_annotation_id=disease_ann.id,
                source="human",
                status="gold",
                annotator_id="curator1",
            ),
        )

        correction = create_correction(
            db,
            AnnotationCorrectionCreate(
                project_id=project.id,
                document_id=doc.id,
                original_annotation_id=model_ann.id,
                corrected_annotation_id=human_ann.id,
                correction_source="adjudication",
                correction_note="Model included dose/context phrase.",
                error_type="boundary_error",
            ),
            created_by_user_id=None,
        )

        patterns = list_error_patterns(db, project.id)
        pattern = patterns[0]
        atom = draft_guideline_atom(db, pattern.id, manager_id="lei")
        approve_guideline_atom(db, atom.id, approved_by="lei")

        print("Seeded demo project")
        print(f"Project ID: {project.id}")
        print(f"Document ID: {doc.id}")
        print(f"Sentence spans: {len(doc.sentences)}")
        print(f"Correction ID: {correction.id}")
        print(f"ErrorPattern ID: {pattern.id}")
        print(f"GuidelineAtom ID: {atom.id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
