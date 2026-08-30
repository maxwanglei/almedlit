from sqlalchemy.orm import Session

from al_medlit.core.exceptions import ValidationError
from al_medlit.guideline.models import GuidelineVersion
from al_medlit.guideline.schemas import GuidelineVersionCreate
from al_medlit.project.models import Project


def create_guideline_version(db: Session, data: GuidelineVersionCreate) -> GuidelineVersion:
    payload = data.model_dump()
    project = (
        db.query(Project)
        .filter(Project.id == payload["project_id"])
        .populate_existing()
        .with_for_update()
        .first()
    )
    if project is None:
        raise ValidationError(f"Project {payload['project_id']} not found")

    if payload.get("status") == "active":
        (
            db.query(GuidelineVersion)
            .filter(
                GuidelineVersion.project_id == payload["project_id"],
                GuidelineVersion.status == "active",
            )
            .update({"status": "superseded"}, synchronize_session="fetch")
        )

    version = GuidelineVersion(**payload)
    db.add(version)
    db.commit()
    db.refresh(version)
    return version


def list_guideline_versions(db: Session, project_id: int) -> list[GuidelineVersion]:
    return (
        db.query(GuidelineVersion)
        .filter(GuidelineVersion.project_id == project_id)
        .order_by(GuidelineVersion.created_at.desc())
        .all()
    )


def get_active_guideline(db: Session, project_id: int) -> GuidelineVersion | None:
    return (
        db.query(GuidelineVersion)
        .filter(GuidelineVersion.project_id == project_id, GuidelineVersion.status == "active")
        .order_by(GuidelineVersion.created_at.desc())
        .first()
    )
