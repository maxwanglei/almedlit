"""Regression coverage for imports larger than NCBI's per-request ID limit."""

import httpx
import pytest

from al_medlit.core.config import settings
from al_medlit.corpus.models import Document
from al_medlit.importers import pubmed
from al_medlit.importers import service as importer_service
from al_medlit.project.models import Project
from al_medlit.workspace import service as workspace_service


def _pmids(count: int) -> list[str]:
    return [str(10_000_000 + index) for index in range(count)]


def _pubmed_xml(ids: list[str], *, without_abstract: set[str] | None = None) -> str:
    without_abstract = without_abstract or set()
    articles = []
    for pmid in ids:
        abstract = (
            "" if pmid in without_abstract else
            f"<Abstract><AbstractText>Abstract for {pmid}.</AbstractText></Abstract>"
        )
        articles.append(
            f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article>"
            f"<ArticleTitle>Paper {pmid}</ArticleTitle>"
            "<Journal><Title>Batch Journal</Title>"
            "<JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>"
            f"{abstract}</Article></MedlineCitation></PubmedArticle>"
        )
    return "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"


def _pmc_xml(ids: list[str]) -> str:
    articles = [
        "<article><front><article-meta>"
        f'<article-id pub-id-type="pmcid">PMC{pmcid}</article-id>'
        "</article-meta></front><body><sec><title>Results</title>"
        f"<p>Full text for {pmcid}.</p></sec></body></article>"
        for pmcid in ids
    ]
    return "<pmc-articleset>" + "".join(articles) + "</pmc-articleset>"


def _request_ids(request: httpx.Request) -> tuple[str, list[str]]:
    if "idconv" in request.url.path:
        assert request.url.params["idtype"] == "pmid"
        assert request.url.params["format"] == "json"
        return "converter", request.url.params["ids"].split(",")
    assert request.url.path.endswith("efetch.fcgi")
    assert request.url.params["retmode"] == "xml"
    return request.url.params["db"], request.url.params["id"].split(",")


def _successful_response(operation: str, ids: list[str]) -> httpx.Response:
    if operation == "converter":
        return httpx.Response(200, json={
            "status": "ok",
            "records": [{"pmid": pmid, "pmcid": f"PMC{pmid}"} for pmid in ids],
        })
    if operation == "pubmed":
        return httpx.Response(200, text=_pubmed_xml(ids))
    assert operation == "pmc"
    return httpx.Response(200, text=_pmc_xml(ids))


@pytest.mark.parametrize("count", [0, 1, 200, 201, 401])
@pytest.mark.parametrize("operation", ["converter", "pubmed", "documents", "fulltext"])
def test_fetch_batches_preserve_every_requested_record(count, operation):
    ids = _pmids(count)
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_operation, batch = _request_ids(request)
        batches.append(batch)
        if len(batch) > 200:
            return httpx.Response(400, text="Too many IDs")
        expected_operation = "pmc" if operation in {"documents", "fulltext"} else operation
        assert requested_operation == expected_operation
        # Responses contain only this request's IDs, so missed or overwritten
        # batches cannot be hidden by a fixture returning the entire input set.
        return _successful_response(requested_operation, batch)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        if operation == "converter":
            result = pubmed.resolve_pmcids(client, ids)
            assert result == {pmid: f"PMC{pmid}" for pmid in ids}
        elif operation == "pubmed":
            result = pubmed.fetch_pubmed_metadata(client, ids)
            assert set(result) == set(ids)
            for pmid in ids:
                assert result[pmid].pmid == pmid
                assert result[pmid].title == f"Paper {pmid}"
                assert result[pmid].abstract == f"Abstract for {pmid}."
                assert result[pmid].journal == "Batch Journal"
                assert result[pmid].year == "2026"
        elif operation == "documents":
            result = pubmed.fetch_pmc_documents(client, [f"PMC{pmid}" for pmid in ids])
            assert set(result) == {f"PMC{pmid}" for pmid in ids}
            for pmid in ids:
                document = result[f"PMC{pmid}"]
                assert document.text == f"Results\n\nFull text for {pmid}."
                assert document.structure_source["format"] == "jats-v1"
                blocks = document.structure_source["blocks"]
                assert len(blocks) == 2
                assert blocks[-1]["locator"]["jats_path"] == "/body/sec[1]/p[1]"
                assert document.text[blocks[-1]["start_offset"]:blocks[-1]["end_offset"]] == (
                    f"Full text for {pmid}."
                )
        else:
            result = pubmed.fetch_pmc_fulltext(client, [f"PMC{pmid}" for pmid in ids])
            assert result == {
                f"PMC{pmid}": f"Results\n\nFull text for {pmid}." for pmid in ids
            }

    assert batches == [ids[offset:offset + 200] for offset in range(0, count, 200)]


def test_preview_300_pmids_preserves_mixed_availability_and_input_order():
    ids = _pmids(300)
    fulltext = set(ids[::4])
    missing = set(ids[2::4])
    no_content = set(ids[3::4])
    batches: dict[str, list[list[str]]] = {"pubmed": [], "converter": [], "pmc": []}

    def handler(request: httpx.Request) -> httpx.Response:
        operation, batch = _request_ids(request)
        batches[operation].append(batch)
        if len(batch) > 200:
            return httpx.Response(400, text="Too many IDs")
        if operation == "pubmed":
            found = [pmid for pmid in batch if pmid not in missing]
            return httpx.Response(200, text=_pubmed_xml(found, without_abstract=no_content))
        if operation == "converter":
            return _successful_response(operation, [pmid for pmid in batch if pmid in fulltext])
        return _successful_response(operation, batch)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        items = importer_service.preview_import(client, ids)

    assert [item.pmid for item in items] == ids
    assert [item.status for item in items] == [
        ("full_text", "abstract_only", "not_found", "error")[index % 4]
        for index in range(300)
    ]
    for item in items:
        assert item.has_full_text == (item.pmid in fulltext)
        assert item.has_abstract == (item.pmid not in missing | no_content)
        if item.pmid not in missing:
            assert item.title == f"Paper {item.pmid}"
    assert batches["pubmed"] == [ids[:200], ids[200:]]
    assert batches["converter"] == [ids[:200], ids[200:]]
    assert batches["pmc"] == [ids[::4]]


@pytest.mark.parametrize("failing_operation", ["pubmed", "converter", "pmc"])
def test_later_batch_failure_does_not_import_partial_documents(db, failing_operation):
    workspace = workspace_service.ensure_default_workspace(db)
    project = Project(name="failed-batch-import", workspace_id=workspace.id)
    db.add(project)
    db.commit()
    calls = {"pubmed": 0, "converter": 0, "pmc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        operation, batch = _request_ids(request)
        assert len(batch) <= 200
        calls[operation] += 1
        if operation == failing_operation and calls[operation] == 2:
            return httpx.Response(400, text="Unusable later batch")
        return _successful_response(operation, batch)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError):
            importer_service.run_import(
                db, client, project.id, _pmids(300), include_abstract_only=True,
            )

    assert calls[failing_operation] == 2
    db.flush()
    assert db.query(Document).filter(Document.project_id == project.id).count() == 0


@pytest.mark.parametrize("failure", [400, 429, 500, "timeout", "connection"])
def test_upstream_errors_do_not_expose_request_parameters(monkeypatch, failure):
    api_key = "private-ncbi-api-key-for-regression"
    email = "private-maintainer@example.invalid"
    pmid = "98765432"
    monkeypatch.setattr(settings, "ncbi_api_key", api_key)
    monkeypatch.setattr(settings, "ncbi_email", email)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["api_key"] == api_key
        assert request.url.params["email"] == email
        # Transport exceptions can contain the complete URL as well as HTTP
        # status exceptions; both need sanitizing before reaching the UI.
        if failure == "timeout":
            raise httpx.ReadTimeout(f"Timed out fetching {request.url}", request=request)
        if failure == "connection":
            raise httpx.ConnectError(f"Could not connect to {request.url}", request=request)
        return httpx.Response(failure, text=f"Upstream body echoes {api_key} and {email}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError) as caught:
            pubmed.resolve_pmcids(client, [pmid])

    message = str(caught.value)
    assert message
    assert len(message) < 300
    for private_value in (api_key, email, pmid, "https://", "api_key=", "email="):
        assert private_value not in message
