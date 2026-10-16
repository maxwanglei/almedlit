from concurrent.futures import ThreadPoolExecutor
from email.utils import formatdate
from threading import Barrier

import httpx
import pytest
from pydantic import ValidationError

from al_medlit.core.config import settings
from al_medlit.importers import pubmed
from al_medlit.importers.schemas import ImportPreviewRequest, ImportRequest


@pytest.mark.parametrize("with_api_key", [False, True])
def test_ncbi_endpoints_share_request_rate_limit(ncbi_clock, monkeypatch, with_api_key):
    monkeypatch.setattr(settings, "ncbi_api_key", "secret" if with_api_key else "")
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if "idconv" in request.url.path:
            return httpx.Response(200, json={"records": []})
        return httpx.Response(200, text="<root />")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        pubmed.fetch_pubmed_metadata(client, ["123"])
        pubmed.resolve_pmcids(client, ["123"])
        pubmed.fetch_pmc_documents(client, ["PMC123"])

    assert starts == pytest.approx([0.0, 1 / 3, 2 / 3])


def test_concurrent_clients_share_request_rate_limit(ncbi_clock):
    starts = []
    ready = Barrier(8)

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        return httpx.Response(200, json={"records": []})

    def fetch(_index):
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            ready.wait(timeout=5)
            return pubmed.resolve_pmcids(client, ["123"])

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(fetch, range(8))) == [{}] * 8

    ordered = sorted(starts)
    assert ordered == pytest.approx([index / 3 for index in range(8)])


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_error_recovers(ncbi_clock, status):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if len(starts) == 1:
            return httpx.Response(status)
        return httpx.Response(200, json={"records": [{"pmid": "123", "pmcid": "PMC123"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert pubmed.resolve_pmcids(client, ["123"]) == {"123": "PMC123"}

    assert starts == [0.0, 1.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 501])
def test_permanent_http_errors_are_not_retried(ncbi_clock, status):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(client, ["123"])

    assert starts == [0.0]


@pytest.mark.parametrize("error_type", [
    httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError,
])
def test_transient_transport_errors_recover(ncbi_clock, error_type):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if len(starts) == 1:
            raise error_type("secret request URL", request=request)
        return httpx.Response(200, json={"records": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert pubmed.resolve_pmcids(client, ["123"]) == {}

    assert starts == [0.0, 1.0]


@pytest.mark.parametrize("failure", [429, 503, "timeout"])
def test_retries_stop_after_three_attempts_with_safe_error(ncbi_clock, monkeypatch, failure):
    monkeypatch.setattr(settings, "ncbi_api_key", "private-api-key")
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if failure == "timeout":
            raise httpx.ReadTimeout(str(request.url), request=request)
        return httpx.Response(failure, text=str(request.url))

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError) as caught:
            pubmed.resolve_pmcids(client, ["123"])

    assert starts == [0.0, 1.0, 3.0]
    assert "private-api-key" not in str(caught.value)
    assert "123" not in str(caught.value)
    assert "https://" not in str(caught.value)


@pytest.mark.parametrize("http_date", [False, True])
def test_retry_after_is_respected(ncbi_clock, http_date):
    starts = []
    value = formatdate(ncbi_clock.time() + 5, usegmt=True) if http_date else "5"

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if len(starts) == 1:
            return httpx.Response(429, headers={"Retry-After": value})
        return httpx.Response(200, json={"records": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert pubmed.resolve_pmcids(client, ["123"]) == {}

    assert starts == [0.0, 5.0]


@pytest.mark.parametrize("http_date", [False, True])
def test_retry_after_beyond_wait_budget_fails_without_retrying_early(ncbi_clock, http_date):
    starts = []
    value = formatdate(ncbi_clock.time() + 60, usegmt=True) if http_date else "60"

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        return httpx.Response(429, headers={"Retry-After": value})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(client, ["123"])

    assert starts == [0.0]
    assert ncbi_clock.sleeps == []


@pytest.mark.parametrize("value", ["malformed", "-1", "nan", "inf"])
def test_invalid_retry_after_uses_bounded_backoff(ncbi_clock, value):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        return httpx.Response(503, headers={"Retry-After": value})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(client, ["123"])

    assert starts == [0.0, 1.0, 3.0]


@pytest.mark.parametrize("schema", [ImportPreviewRequest, ImportRequest])
def test_pmids_are_capped_before_deduplicating(schema):
    assert len(schema(pmids=["123"] * 1000).pmids) == 1000
    with pytest.raises(ValidationError) as caught:
        schema(pmids=["123"] * 1001)
    assert caught.value.errors()[0]["type"] == "too_long"


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_remains_shared_after_attempts_are_exhausted(ncbi_clock, status):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if len(starts) <= 3:
            return httpx.Response(status, headers={"Retry-After": "5"})
        return httpx.Response(200, json={"records": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as first_client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(first_client, ["123"])
    with httpx.Client(transport=httpx.MockTransport(handler)) as next_client:
        assert pubmed.resolve_pmcids(next_client, ["456"]) == {}

    assert starts == [0.0, 5.0, 10.0, 15.0]


def test_long_retry_after_blocks_other_clients_without_sleeping_or_sending(ncbi_clock):
    starts = []

    def handler(request):
        starts.append(ncbi_clock.monotonic())
        if len(starts) == 1:
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, json={"records": []})

    with httpx.Client(transport=httpx.MockTransport(handler)) as first_client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(first_client, ["123"])
    with httpx.Client(transport=httpx.MockTransport(handler)) as next_client:
        with pytest.raises(pubmed.ImporterFetchError):
            pubmed.resolve_pmcids(next_client, ["456"])
        assert starts == [0.0]
        assert ncbi_clock.sleeps == []
        ncbi_clock.sleep(60.0)
        assert pubmed.resolve_pmcids(next_client, ["456"]) == {}

    assert starts == [0.0, 60.0]
