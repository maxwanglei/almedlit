"""Tests for the PubMed/PMC PMID importer (al_medlit.importers)."""

import httpx
import pytest

from al_medlit.importers import pubmed

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111</PMID>
      <Article>
        <Journal>
          <Title>Journal of Testing</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>A study of things</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Things are studied.</AbstractText>
          <AbstractText Label="RESULTS">Things were found.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222</PMID>
      <Article>
        <Journal>
          <Title>Journal of No Abstract</Title>
          <JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Title without abstract</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>33333</PMID>
      <Article>
        <Journal>
          <Title>Journal of Abstract Only</Title>
          <JournalIssue><PubDate><Year>2020</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Abstract only article</ArticleTitle>
        <Abstract>
          <AbstractText>Only an abstract is available here.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_pubmed_xml_extracts_metadata():
    result = pubmed.parse_pubmed_xml(PUBMED_XML)

    assert set(result) == {"11111", "22222", "33333"}
    meta = result["11111"]
    assert meta.title == "A study of things"
    assert meta.journal == "Journal of Testing"
    assert meta.year == "2021"
    assert "Things are studied." in meta.abstract
    assert "Things were found." in meta.abstract


def test_parse_pubmed_xml_handles_missing_abstract():
    result = pubmed.parse_pubmed_xml(PUBMED_XML)

    assert result["22222"].abstract == ""
    assert result["22222"].title == "Title without abstract"


PMC_XML = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmc">9999</article-id>
    </article-meta></front>
    <body>
      <sec>
        <title>Introduction</title>
        <p>The first paragraph of the body.</p>
        <p>The second paragraph of the body.</p>
      </sec>
      <sec>
        <title>Methods</title>
        <p>We did the methods.</p>
      </sec>
    </body>
  </article>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmc">8888</article-id>
    </article-meta></front>
  </article>
</pmc-articleset>
"""

PMC_XML_PMCID_TYPE = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front><article-meta>
      <article-id pub-id-type="pmcid-ver">PMC9999.1</article-id>
      <article-id pub-id-type="pmcid">PMC9999</article-id>
    </article-meta></front>
    <body>
      <sec>
        <title>Results</title>
        <p>The body is keyed by the canonical PMCID.</p>
      </sec>
    </body>
  </article>
</pmc-articleset>
"""


def test_parse_pmc_xml_returns_body_text_keyed_by_pmcid():
    result = pubmed.parse_pmc_xml(PMC_XML)

    assert set(result) == {"PMC9999"}
    body = result["PMC9999"]
    assert "The first paragraph of the body." in body
    assert "The second paragraph of the body." in body
    assert "We did the methods." in body


def test_parse_pmc_xml_accepts_pmcid_type_and_ignores_versioned_id():
    result = pubmed.parse_pmc_xml(PMC_XML_PMCID_TYPE)

    assert set(result) == {"PMC9999"}
    assert "canonical PMCID" in result["PMC9999"]


def test_parse_pmc_xml_omits_articles_without_body():
    result = pubmed.parse_pmc_xml(PMC_XML)

    assert "PMC8888" not in result


IDCONV_JSON = {
    "status": "ok",
    "records": [
        {"pmid": "11111", "pmcid": "PMC9999"},
        {"pmid": "22222", "errmsg": "invalid article id"},
        {"pmid": "33333"},
    ],
}


def test_parse_idconv_json_maps_pmid_to_pmcid():
    result = pubmed.parse_idconv_json(IDCONV_JSON)

    assert result == {"11111": "PMC9999"}


def _fake_ncbi_client(pmc_xml: str = PMC_XML) -> httpx.Client:
    """Route fake NCBI requests by path and ``db`` to the sample fixtures."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("efetch.fcgi"):
            if request.url.params.get("db") == "pmc":
                return httpx.Response(200, text=pmc_xml)
            return httpx.Response(200, text=PUBMED_XML)
        if "idconv" in path:
            return httpx.Response(200, json=IDCONV_JSON)
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_pubmed_metadata_calls_efetch_and_parses():
    with _fake_ncbi_client() as client:
        result = pubmed.fetch_pubmed_metadata(client, ["11111", "22222"])

    assert result["11111"].title == "A study of things"


def test_resolve_pmcids_calls_idconv_and_parses():
    with _fake_ncbi_client() as client:
        result = pubmed.resolve_pmcids(client, ["11111", "22222", "33333"])

    assert result == {"11111": "PMC9999"}


def test_resolve_pmcids_marks_converter_input_as_pmids():
    seen_request: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_request
        seen_request = request
        return httpx.Response(200, json=IDCONV_JSON)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pubmed.resolve_pmcids(client, ["11111", "33333"])

    assert seen_request is not None
    assert "idconv/api/v1/articles" in seen_request.url.path
    assert seen_request.url.params["format"] == "json"
    assert seen_request.url.params["idtype"] == "pmid"
    assert seen_request.url.params["ids"] == "11111,33333"


def test_fetch_pmc_fulltext_calls_efetch_pmc_and_parses():
    with _fake_ncbi_client() as client:
        result = pubmed.fetch_pmc_fulltext(client, ["PMC9999", "PMC8888"])

    assert "PMC9999" in result
    assert "PMC8888" not in result


def test_fetcher_raises_importer_error_on_bad_status():
    def boom(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    with httpx.Client(transport=httpx.MockTransport(boom)) as client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.fetch_pubmed_metadata(client, ["11111"])


# --- Service: preview + import ----------------------------------------------

from al_medlit.importers import service as importer_service  # noqa: E402
from al_medlit.project.models import Project  # noqa: E402
from al_medlit.workspace import service as workspace_service  # noqa: E402


def _make_project(db) -> Project:
    workspace = workspace_service.ensure_default_workspace(db)
    project = Project(name="import-test", workspace_id=workspace.id)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def test_preview_import_classifies_each_pmid():
    with _fake_ncbi_client() as client:
        items = importer_service.preview_import(client, ["11111", "33333", "44444"])

    by_pmid = {item.pmid: item for item in items}
    assert by_pmid["11111"].status == "full_text"
    assert by_pmid["11111"].has_full_text is True
    assert by_pmid["11111"].pmcid == "PMC9999"
    assert by_pmid["33333"].status == "abstract_only"
    assert by_pmid["33333"].has_full_text is False
    assert by_pmid["33333"].has_abstract is True
    assert by_pmid["44444"].status == "not_found"


def test_preview_import_treats_pmcid_type_article_as_full_text():
    with _fake_ncbi_client(PMC_XML_PMCID_TYPE) as client:
        items = importer_service.preview_import(client, ["11111"])

    item = items[0]
    assert item.status == "full_text"
    assert item.has_full_text is True
    assert item.pmcid == "PMC9999"


def test_run_import_creates_full_text_document(db):
    project = _make_project(db)
    with _fake_ncbi_client() as client:
        result = importer_service.run_import(
            db, client, project.id, ["11111"], include_abstract_only=False
        )

    assert len(result.created) == 1
    outcome = result.created[0]
    assert outcome.pmid == "11111"
    from al_medlit.corpus.models import Document

    doc = db.get(Document, outcome.document_id)
    assert doc.source == "pmc"
    assert doc.external_id == "11111"
    assert doc.metadata_["has_full_text"] is True
    assert "first paragraph of the body" in doc.text


def test_run_import_skips_abstract_only_when_not_included(db):
    project = _make_project(db)
    with _fake_ncbi_client() as client:
        result = importer_service.run_import(
            db, client, project.id, ["33333"], include_abstract_only=False
        )

    assert result.created == []
    assert len(result.skipped) == 1
    assert result.skipped[0].pmid == "33333"


def test_run_import_creates_abstract_document_when_included(db):
    project = _make_project(db)
    with _fake_ncbi_client() as client:
        result = importer_service.run_import(
            db, client, project.id, ["33333"], include_abstract_only=True
        )

    assert len(result.created) == 1
    from al_medlit.corpus.models import Document

    doc = db.get(Document, result.created[0].document_id)
    assert doc.source == "pubmed_abstract"
    assert doc.metadata_["has_full_text"] is False
    assert "Only an abstract is available" in doc.text


def test_run_import_skips_duplicate_external_id(db):
    project = _make_project(db)
    with _fake_ncbi_client() as client:
        importer_service.run_import(
            db, client, project.id, ["11111"], include_abstract_only=False
        )
        result = importer_service.run_import(
            db, client, project.id, ["11111"], include_abstract_only=False
        )

    assert result.created == []
    assert len(result.skipped) == 1
    assert "duplicate" in result.skipped[0].reason.lower()


# --- Router -----------------------------------------------------------------

from fastapi.testclient import TestClient  # noqa: E402

from al_medlit.core.database import get_db  # noqa: E402
from al_medlit.importers.router import get_ncbi_client  # noqa: E402
from al_medlit.main import create_app  # noqa: E402


@pytest.fixture(name="api_client")
def fixture_api_client(request):
    from al_medlit.auth.models import User
    from al_medlit.auth.security import create_access_token, hash_password
    from al_medlit.workspace import service as workspace_service

    testing_session_factory = request.getfixturevalue("testing_session_factory")
    session = testing_session_factory()
    try:
        user = User(
            username="pubmed-api-tester",
            password_hash=hash_password("pw"),
            display_name="PubMed API Tester",
            is_active=True,
        )
        session.add(user)
        session.flush()
        default_workspace = workspace_service.ensure_default_workspace(session)
        workspace_service.add_member(
            session,
            default_workspace.id,
            user.id,
            role="admin",
        )
        session.commit()
        token = create_access_token(str(user.id))
    finally:
        session.close()

    app = create_app()

    def override_get_db():
        session = testing_session_factory()
        try:
            yield session
        finally:
            session.close()

    def override_get_ncbi_client():
        with _fake_ncbi_client() as client:
            yield client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ncbi_client] = override_get_ncbi_client
    with TestClient(app) as test_client:
        test_client.headers.update({"Authorization": f"Bearer {token}"})
        yield test_client
    app.dependency_overrides.clear()


def _create_project_via_api(api_client) -> int:
    response = api_client.post(
        "/api/projects",
        json={"name": "import-api", "annotation_schema": {"labels": {}}, "settings": {}},
    )
    assert response.status_code == 200
    return response.json()["id"]


def test_preview_endpoint_returns_classified_items(api_client):
    project_id = _create_project_via_api(api_client)
    response = api_client.post(
        f"/api/projects/{project_id}/import/pubmed/preview",
        json={"pmids": ["11111", "33333", "44444"]},
    )

    assert response.status_code == 200
    by_pmid = {item["pmid"]: item for item in response.json()["items"]}
    assert by_pmid["11111"]["status"] == "full_text"
    assert by_pmid["33333"]["status"] == "abstract_only"
    assert by_pmid["44444"]["status"] == "not_found"


def test_import_endpoint_creates_documents(api_client):
    project_id = _create_project_via_api(api_client)
    response = api_client.post(
        f"/api/projects/{project_id}/import/pubmed",
        json={"pmids": ["11111", "33333"], "include_abstract_only": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["created"]) == 2

    docs = api_client.get(f"/api/documents?project_id={project_id}").json()
    assert len(docs) == 2


@pytest.fixture(name="personal_api_client")
def fixture_personal_api_client(request):
    testing_session_factory = request.getfixturevalue("testing_session_factory")
    app = create_app()

    def override_get_db():
        session = testing_session_factory()
        try:
            yield session
        finally:
            session.close()

    def override_get_ncbi_client():
        with _fake_ncbi_client(PMC_XML_PMCID_TYPE) as client:
            yield client

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ncbi_client] = override_get_ncbi_client
    with TestClient(app) as test_client:
        response = test_client.post(
            "/api/auth/register",
            json={
                "username": "personal-pubmed-importer",
                "password": "strong-password",
                "workspace_kind": "individual",
            },
        )
        assert response.status_code == 200
        token = response.json()["access_token"]
        test_client.headers.update({"Authorization": f"Bearer {token}"})
        yield test_client
    app.dependency_overrides.clear()


def test_import_endpoint_creates_documents_in_personal_workspace(personal_api_client):
    me = personal_api_client.get("/api/auth/me").json()
    membership = me["memberships"][0]
    assert membership["workspace_kind"] == "individual"
    assert membership["role"] == "admin"

    project_response = personal_api_client.post(
        "/api/projects",
        json={
            "name": "personal-import-api",
            "workspace_id": membership["workspace_id"],
            "tasks": [
                {
                    "annotation_type": "entity",
                    "display_name": "Entity Annotation",
                    "labels": [{"name": "Finding", "color": "#2563eb"}],
                }
            ],
            "settings": {},
        },
    )
    assert project_response.status_code == 200
    project_id = project_response.json()["id"]

    import_response = personal_api_client.post(
        f"/api/projects/{project_id}/import/pubmed",
        json={"pmids": ["11111"], "include_abstract_only": False},
    )
    assert import_response.status_code == 200
    assert import_response.json()["created"][0]["status"] == "full_text"

    docs = personal_api_client.get(f"/api/documents?project_id={project_id}&scope=all")
    assert docs.status_code == 200
    body = docs.json()
    assert len(body) == 1
    assert body[0]["source"] == "pmc"

    mine_docs = personal_api_client.get(f"/api/documents?project_id={project_id}&scope=mine")
    assert mine_docs.status_code == 200
    assert [item["id"] for item in mine_docs.json()] == [body[0]["id"]]

    assignments = personal_api_client.get(
        f"/api/projects/{project_id}/assignments",
        params={"scope": "mine"},
    )
    assert assignments.status_code == 200
    assignment_body = assignments.json()
    assert len(assignment_body) == 1
    assert assignment_body[0]["document_id"] == body[0]["id"]
    assert assignment_body[0]["annotator_id"] == me["user"]["username"]
    assert assignment_body[0]["status"] == "assigned"

    progress = personal_api_client.get(
        f"/api/projects/{project_id}/progress",
        params={"scope": "mine"},
    )
    assert progress.status_code == 200
    assert progress.json()["total"] == 1
    assert progress.json()["by_status"] == {"assigned": 1}


def test_preview_endpoint_unknown_project_returns_404(api_client):
    response = api_client.post(
        "/api/projects/9999/import/pubmed/preview",
        json={"pmids": ["11111"]},
    )
    assert response.status_code == 404
