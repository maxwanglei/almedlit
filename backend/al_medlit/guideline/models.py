from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin


class GuidelineVersion(Base, IntPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "guideline_versions"

    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"), index=True)
    version_label: Mapped[str] = mapped_column(String(80), default="v1")
    markdown: Mapped[str] = mapped_column(Text, default="")
    author_id: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="active")

    project = relationship("Project", back_populates="guideline_versions")
