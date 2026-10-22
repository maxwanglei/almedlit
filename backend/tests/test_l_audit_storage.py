from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.orm import Session
from test_submission_flow import _setup_project_with_annotation

from al_medlit.storage_reclaim.service import reclaim_orphaned_objects
from al_medlit.submission.models import AnnotationSubmission


def test_ambiguous_commit_preserves_live_submission(client, db, object_storage, monkeypatch):
    context = _setup_project_with_annotation(client)
    original_commit = Session.commit
    failed = False

    def lose_commit_acknowledgement(session):
        nonlocal failed
        original_commit(session)
        if not failed:
            failed = True
            raise RuntimeError("commit acknowledgement lost")

    with monkeypatch.context() as patch:
        patch.setattr(Session, "commit", lose_commit_acknowledgement)
        with pytest.raises(RuntimeError, match="commit acknowledgement lost"):
            client.post(
                f"/api/projects/{context['project']['id']}"
                f"/documents/{context['document']['id']}/submissions",
                json={"annotator_id": "alice"},
            )
    submission = db.query(AnnotationSubmission).one()
    snapshot = object_storage.get_bytes(submission.storage_key)
    assert snapshot
    assert (
        reclaim_orphaned_objects(
            db,
            object_storage,
            now=datetime.now(UTC) + timedelta(hours=1),
        ).reclaimed_count
        == 0
    )
    assert object_storage.get_bytes(submission.storage_key) == snapshot


def test_sweep_reclaims_rolled_back_rehydrated_purged_blob(db, object_storage):
    import os

    from al_medlit.core.storage import ObjectNotFoundError
    from al_medlit.model_artifacts.models import ArtifactBlob
    from al_medlit.model_artifacts.service import (
        blob_storage_key,
        sweep_unregistered_artifact_objects,
    )
    from al_medlit.workspace.models import Workspace

    workspace = Workspace(name="Failed rehydration", kind="team")
    db.add(workspace)
    db.flush()
    key = blob_storage_key(workspace.id, "a" * 64)
    blob = ArtifactBlob(
        workspace_id=workspace.id,
        checksum_sha256="a" * 64,
        storage_key=key,
        status="purged",
        size_bytes=7,
        content_type="application/octet-stream",
    )
    db.add(blob)
    db.commit()
    object_storage.put_bytes(key, b"orphan!")
    old = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(object_storage.root / key, (old, old))
    assert sweep_unregistered_artifact_objects(db, object_storage) == 1
    with pytest.raises(ObjectNotFoundError):
        object_storage.get_bytes(key)
    assert db.get(ArtifactBlob, blob.id).status == "purged"
