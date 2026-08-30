from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from al_medlit.core.database import Base
from al_medlit.core.models import IntPrimaryKeyMixin, TimestampMixin, utc_now


class OrphanedStorageObject(Base, IntPrimaryKeyMixin, TimestampMixin):
    """A storage key that outlived the database row referencing it.

    Removing a stored object and the row that points at it cannot be made
    atomic, so callers commit the authoritative database state first and then
    delete the object. When that delete fails the object would leak with no
    remaining reference, so its key is queued here for the reclaim sweep.
    """

    __tablename__ = "orphaned_storage_objects"
    __table_args__ = (
        CheckConstraint(
            "attempts >= 0",
            name="ck_orphaned_storage_objects_nonnegative_attempts",
        ),
    )

    storage_key: Mapped[str] = mapped_column(String(512), unique=True)

    # Which code path abandoned the key, e.g. "submission.delete".
    origin: Mapped[str] = mapped_column(String(100), index=True)

    attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    next_attempt_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
        index=True,
    )
