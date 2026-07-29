import pytest
from sqlalchemy.exc import IntegrityError

from al_medlit.corpus.models import Document


def test_sqlite_foreign_keys_are_enforced(db):
    orphan = Document(
        project_id=9999,
        external_id="orphan",
        title="orphan",
        text="orphan body",
        source="manual",
        metadata_={},
    )
    db.add(orphan)
    with pytest.raises(IntegrityError):
        db.commit()
