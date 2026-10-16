"""Request-level transaction and projection regressions for PubMed imports."""

from __future__ import annotations

import weakref

import httpx
import pytest
from sqlalchemy import event
from sqlalchemy.orm import Session
from test_pubmed_importer import (
    PUBMED_XML,
    _create_project_via_api,
    fixture_api_client,
)

from al_medlit.corpus import service as corpus_service
from al_medlit.corpus.models import Document
from al_medlit.importers.router import get_ncbi_client
from al_medlit.project.models import Project

api_client = fixture_api_client


@pytest.mark.parametrize("endpoint", ["pubmed/preview", "pubmed"])
def test_remote_fetch_has_no_open_database_transaction(api_client, endpoint):
    project_id = _create_project_via_api(api_client)
    sessions: weakref.WeakSet[Session] = weakref.WeakSet()
    observed = []

    def track_session(session, _transaction, _connection):
        sessions.add(session)

    def handler(request: httpx.Request) -> httpx.Response:
        active = [session for session in sessions if session.in_transaction()]
        observed.append(len(active))
        assert active == [], "Import held a database transaction across an NCBI request"
        if "idconv" in request.url.path:
            return httpx.Response(200, json={"records": []})
        return httpx.Response(200, text=PUBMED_XML)

    def override_ncbi_client():
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            yield client

    api_client.app.dependency_overrides[get_ncbi_client] = override_ncbi_client
    event.listen(Session, "after_begin", track_session)
    try:
        response = api_client.post(
            f"/api/projects/{project_id}/import/{endpoint}",
            json={"pmids": ["33333"], "include_abstract_only": True},
        )
    finally:
        event.remove(Session, "after_begin", track_session)
    assert response.status_code == 200, response.text
    assert observed and all(count == 0 for count in observed)


@pytest.mark.parametrize("endpoint", ["pubmed/preview", "pubmed"])
def test_existing_papers_use_requested_scalar_projection(
    api_client, testing_session_factory, endpoint
):
    project_id = _create_project_via_api(api_client)
    with testing_session_factory() as db:
        db.add_all([
            Document(project_id=project_id, external_id="11111", title="Existing", text="body",
                     source="pmc", metadata_={"journal": "J", "year": "2026", "pmcid": "PMC1"}),
            Document(project_id=project_id, external_id="77777", title="Unrequested",
                     text="large unrelated body" * 1000, metadata_={"huge": "value" * 1000}),
        ])
        db.commit()
    loaded_documents = []
    document_selects = []

    def track_load(_session, instance):
        if isinstance(instance, Document):
            loaded_documents.append(instance.id)

    def track_query(execute_state):
        statement = execute_state.statement
        if execute_state.is_select and "FROM documents" in str(statement):
            document_selects.append(str(statement))

    event.listen(Session, "loaded_as_persistent", track_load)
    event.listen(Session, "do_orm_execute", track_query)
    try:
        response = api_client.post(
            f"/api/projects/{project_id}/import/{endpoint}", json={"pmids": ["11111"]}
        )
    finally:
        event.remove(Session, "loaded_as_persistent", track_load)
        event.remove(Session, "do_orm_execute", track_query)
    assert response.status_code == 200, response.text
    assert loaded_documents == [], "Importer hydrated complete Document rows"
    assert document_selects
    assert all("documents.text" not in sql for sql in document_selects)
    assert all("documents.external_id IN" in sql for sql in document_selects)


def test_segmenting_import_does_not_hold_project_for_update(api_client, monkeypatch):
    project_id = _create_project_via_api(api_client)
    project_locks = []
    observed_segments = []
    original_segment = corpus_service.segment_document

    def track_query(execute_state):
        statement = execute_state.statement
        if (execute_state.is_select and getattr(statement, "_for_update_arg", None) is not None
                and any(item.get("entity") is Project for item in statement.column_descriptions)):
            project_locks.append(statement)

    def segment_without_project_lock(*args, **kwargs):
        observed_segments.append(True)
        assert project_locks == [], "Import locked Project while segmenting documents"
        return original_segment(*args, **kwargs)

    monkeypatch.setattr(corpus_service, "segment_document", segment_without_project_lock)
    event.listen(Session, "do_orm_execute", track_query)
    try:
        response = api_client.post(
            f"/api/projects/{project_id}/import/pubmed",
            json={"pmids": ["11111"], "include_abstract_only": True},
        )
    finally:
        event.remove(Session, "do_orm_execute", track_query)
    assert response.status_code == 200, response.text
    assert observed_segments


@pytest.mark.parametrize("endpoint", ["pubmed/preview", "pubmed"])
@pytest.mark.parametrize("change", ["deactivate", "revoke_session", "demote", "remove"])
def test_access_revoked_during_fetch_prevents_import_and_preview(
    api_client, testing_session_factory, endpoint, change
):
    from al_medlit.auth.models import User
    from al_medlit.workspace.models import WorkspaceMember

    project_id = _create_project_via_api(api_client)
    changed = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal changed
        if not changed:
            with testing_session_factory() as db:
                user = db.query(User).filter(User.username == "pubmed-api-tester").one()
                if change == "deactivate":
                    user.is_active = False
                elif change == "revoke_session":
                    user.session_version += 1
                else:
                    member = db.query(WorkspaceMember).filter_by(user_id=user.id).one()
                    if change == "demote":
                        member.role = "annotator"
                    else:
                        db.delete(member)
                db.commit()
            changed = True
        if "idconv" in request.url.path:
            return httpx.Response(200, json={"records": []})
        return httpx.Response(200, text=PUBMED_XML)

    def override_ncbi_client():
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            yield client

    api_client.app.dependency_overrides[get_ncbi_client] = override_ncbi_client
    response = api_client.post(
        f"/api/projects/{project_id}/import/{endpoint}",
        json={"pmids": ["33333"], "include_abstract_only": True},
    )
    expected = 401 if change in {"deactivate", "revoke_session"} else 403
    assert changed
    assert response.status_code == expected, response.text
    with testing_session_factory() as db:
        assert db.query(Document).filter_by(project_id=project_id).count() == 0


def test_assignment_failure_rolls_back_imported_documents_and_structure(
    api_client, testing_session_factory, monkeypatch
):
    from al_medlit.core.exceptions import ValidationError
    from al_medlit.corpus.models import DocumentStructureVersion
    from al_medlit.project import service as project_service

    project_id = _create_project_via_api(api_client)

    def fail_assignments(db, *_args, **_kwargs):
        assert db.query(Document).filter_by(project_id=project_id).count() == 1
        raise ValidationError("Assignment creation failed")

    monkeypatch.setattr(
        project_service, "ensure_personal_project_self_assignments", fail_assignments
    )
    response = api_client.post(
        f"/api/projects/{project_id}/import/pubmed", json={"pmids": ["11111"]}
    )
    assert response.status_code == 422
    with testing_session_factory() as db:
        assert db.query(Document).filter_by(project_id=project_id).count() == 0
        assert db.query(DocumentStructureVersion).count() == 0
