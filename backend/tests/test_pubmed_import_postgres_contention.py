"""Opt-in PostgreSQL concurrency coverage for PubMed import locks."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import httpx
import pytest
from test_system_administration_postgres_contention import (
    _new_user,
    _run_concurrently,
    _set_transaction_timeouts,
)
from test_system_administration_postgres_contention import (
    postgres_session_factory as _shared_postgres_session_factory,
)

from al_medlit.annotation.models import Annotation
from al_medlit.annotation.router import create_annotation
from al_medlit.annotation.schemas import AnnotationCreate
from al_medlit.auth.models import User
from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import Document, DocumentStructureVersion
from al_medlit.corpus.schemas import DocumentCreate
from al_medlit.importers import service as importer_service
from al_medlit.importers.router import run_pubmed_import
from al_medlit.importers.schemas import ImportRequest
from al_medlit.project.models import Project
from al_medlit.workspace.models import Workspace, WorkspaceMember

postgres_session_factory = _shared_postgres_session_factory

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(
        os.getenv("AL_MEDLIT_RUN_POSTGRES_TESTS") != "1",
        reason="Set AL_MEDLIT_RUN_POSTGRES_TESTS=1 to run Docker-backed Postgres tests",
    ),
]


def _scope(db):
    manager = _new_user(db, "import-manager")
    other_manager = _new_user(db, "other-import-manager")
    annotator = _new_user(db, "import-annotator")
    workspace = Workspace(name=f"import-{uuid4().hex}", kind="team")
    db.add(workspace)
    db.flush()
    db.add_all([
        WorkspaceMember(workspace_id=workspace.id, user_id=manager.id, role="manager"),
        WorkspaceMember(workspace_id=workspace.id, user_id=other_manager.id, role="manager"),
        WorkspaceMember(workspace_id=workspace.id, user_id=annotator.id, role="annotator"),
    ])
    project = Project(name=f"import-{uuid4().hex}", workspace_id=workspace.id)
    db.add(project)
    db.flush()
    document = corpus_service.create_document(
        db, DocumentCreate(project_id=project.id, text="Existing annotation document."),
        commit=False,
    )
    db.commit()
    return project.id, manager.id, other_manager.id, annotator.id, document.id


def _import(db, project_id, manager_id):
    def handler(request):
        if "idconv" in request.url.path:
            return httpx.Response(200, json={"records": []})
        return httpx.Response(200, text=(
            "<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID>"
            "<Article><ArticleTitle>Imported article</ArticleTitle>"
            "<Abstract><AbstractText>Imported abstract.</AbstractText></Abstract>"
            "</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
        ))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return run_pubmed_import(
            project_id=project_id,
            payload=ImportRequest(pmids=["12345"], include_abstract_only=True),
            _member=None,
            current_user=db.get(User, manager_id),
            db=db,
            client=client,
        )


def test_other_annotator_can_save_while_import_segments(postgres_session_factory, monkeypatch):
    with postgres_session_factory() as db:
        project_id, manager_id, _, annotator_id, document_id = _scope(db)
    segmenting = threading.Event()
    release = threading.Event()
    original_segment = corpus_service.segment_document

    def paused_segment(*args, **kwargs):
        segmenting.set()
        assert release.wait(timeout=10)
        return original_segment(*args, **kwargs)

    def import_document():
        with postgres_session_factory() as db:
            return _import(db, project_id, manager_id)

    monkeypatch.setattr(corpus_service, "segment_document", paused_segment)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(import_document)
        try:
            assert segmenting.wait(timeout=5)
            with postgres_session_factory() as db:
                _set_transaction_timeouts(db)
                annotation = create_annotation(
                    AnnotationCreate(
                        project_id=project_id, document_id=document_id,
                        annotation_type="doc_label", label="Include",
                    ),
                    current_user=db.get(User, annotator_id), db=db,
                )
                annotation_id = annotation.id
                assert not future.done()
            release.set()
            response = future.result(timeout=15)
        finally:
            release.set()
    assert len(response.created) == 1
    with postgres_session_factory() as db:
        assert db.get(Annotation, annotation_id).annotator_user_id == annotator_id
        imported = db.get(Document, response.created[0].document_id)
        assert imported.active_structure_version_id is not None


def test_concurrent_imports_reuse_one_complete_document(postgres_session_factory, monkeypatch):
    with postgres_session_factory() as db:
        project_id, manager_id, other_manager_id, _, _ = _scope(db)
    both_fetching = threading.Barrier(2)
    original_gather = importer_service._gather

    def gather_after_both_imports_check_duplicates(*args, **kwargs):
        both_fetching.wait(timeout=5)
        return original_gather(*args, **kwargs)

    monkeypatch.setattr(importer_service, "_gather", gather_after_both_imports_check_duplicates)

    def import_for(manager_id):
        return lambda db: _import(db, project_id, manager_id)

    left, right = _run_concurrently(
        postgres_session_factory, import_for(manager_id), import_for(other_manager_id)
    )
    assert left[0] == "ok", left
    assert right[0] == "ok", right
    responses = [left[1], right[1]]
    assert sorted(len(response.created) for response in responses) == [0, 1]
    assert sorted(len(response.skipped) for response in responses) == [0, 1]
    ids = [
        item.document_id for response in responses for item in response.created + response.skipped
    ]
    assert ids[0] == ids[1]
    with postgres_session_factory() as db:
        documents = db.query(Document).filter_by(project_id=project_id, external_id="12345").all()
        assert len(documents) == 1
        structure_id = documents[0].active_structure_version_id
        structure = db.get(DocumentStructureVersion, structure_id)
        assert structure.document_id == documents[0].id
        assert len(structure.sentences) == 1


def test_failed_import_releases_lock_and_rolls_back_structure(
    postgres_session_factory, monkeypatch
):
    with postgres_session_factory() as db:
        project_id, manager_id, other_manager_id, _, _ = _scope(db)
    original_segment = corpus_service.segment_document

    def fail_segment(*_args, **_kwargs):
        raise ValueError("Unusable structure")

    monkeypatch.setattr(corpus_service, "segment_document", fail_segment)
    with postgres_session_factory() as db:
        with pytest.raises(ValueError, match="Unusable structure"):
            _import(db, project_id, manager_id)
    with postgres_session_factory() as db:
        assert db.query(Document).filter_by(project_id=project_id, external_id="12345").count() == 0
    monkeypatch.setattr(corpus_service, "segment_document", original_segment)
    with postgres_session_factory() as db:
        response = _import(db, project_id, other_manager_id)
    assert len(response.created) == 1
